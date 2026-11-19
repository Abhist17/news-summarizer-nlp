"""Evaluate Lead-3, Extractive and Abstractive on a CNN/DM test sample.

Outputs (results/):
  scores.csv / scores.json      ROUGE-1/2/L F1 (x100, mean, std, 95% bootstrap CI), novel n-gram %,
                                repeated n-gram %, avg summary length, avg latency per article
  classifier_metrics.csv        extractive sentence classifier vs oracle labels (val + test):
                                accuracy / precision / recall / F1 / ROC-AUC
  long_doc.csv                  ROUGE for articles over vs under 512 / 1024 tokens
  efficiency.json, factcheck_stats.json
  per_article.jsonl             every summary + per-article scores (for inspection / figures)
  rouge_comparison.png          our ROUGE-1 next to the published numbers
Figures for the report are written to report/figures/ (see src/make_figures.py).

Usage:  python -m src.evaluate --n 200 --abs-n 100 [--fast] [--xsum]
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (ABS_MODEL, ABS_MODEL_FAST, BART_MAX_INPUT, ENCODER_MAX_TOKENS, ENCODER_NAME,
                    EXT_CHECKPOINT, PUBLISHED_ROUGE1, RESULTS, get_device)
from src.abstractive import AbstractiveSummarizer
from src.data import clean_article, highlights_to_sents, load_cnndm, load_xsum, split_sentences
from src.extractive import ExtractiveSummarizer
from src.factcheck import backend_name, check_summary
from src.lead3 import lead3
from src.metrics import rouge_scores, summary_stats
from src.oracle import greedy_oracle, oracle_labels

log = logging.getLogger("evaluate")
ROUGE_KEYS = ("rouge1", "rouge2", "rougeL")


# =========================================================================== scoring helpers
def per_article(items: list[dict]) -> pd.DataFrame:
    """items: {summary, summary_sents?, article, reference?, latency}. One row of scores per item."""
    rows = []
    for it in items:
        st = summary_stats(it["summary"], it["article"])
        if it.get("reference") is not None:
            st.update(rouge_scores(it.get("summary_sents") or it["summary"],
                                   highlights_to_sents(it["reference"])))
        st["latency"] = it.get("latency", np.nan)
        st["id"] = it.get("id")
        rows.append(st)
    return pd.DataFrame(rows)


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate(method: str, dataset: str, df: pd.DataFrame) -> dict:
    out = {"method": method, "dataset": dataset, "n_articles": len(df)}
    for col in ROUGE_KEYS:
        if col in df:
            x = df[col].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(x)
            out.update({col: round(float(x.mean()), 2), f"{col}_std": round(float(x.std(ddof=1)), 2),
                        f"{col}_ci_lo": round(lo, 2), f"{col}_ci_hi": round(hi, 2)})
        else:
            out.update({col: None, f"{col}_std": None, f"{col}_ci_lo": None, f"{col}_ci_hi": None})
    for col in ("novel_1", "novel_2", "novel_3", "repeated_2", "repeated_3"):
        out[col] = round(float(df[col].mean()), 2)
    out["avg_words"] = round(float(df["words"].mean()), 1)
    lat = df["latency"].astype(float)
    out["avg_latency_s"] = round(float(lat.mean()), 4) if lat.notna().any() else None
    return out


def classifier_metrics(probs: list[float], labels: list[int], preds: list[int]) -> dict:
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)

    y, p, yhat = np.array(labels), np.array(probs), np.array(preds)
    return {"n_sentences": int(len(y)), "positive_rate": round(float(y.mean()), 4),
            "accuracy": round(float(accuracy_score(y, yhat)), 4),
            "precision": round(float(precision_score(y, yhat, zero_division=0)), 4),
            "recall": round(float(recall_score(y, yhat, zero_division=0)), 4),
            "f1": round(float(f1_score(y, yhat, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y, p)), 4) if len(set(y)) > 1 and len(p) else None,
            "confusion": confusion_matrix(y, yhat, labels=[0, 1]).tolist()}


def collect_classifier(ext: ExtractiveSummarizer, rows: list[dict]) -> dict:
    """Per-sentence probabilities vs in-window oracle labels, three decision rules."""
    probs, labels, thr, top3, lead = [], [], [], [], []
    for r in tqdm(rows, desc="Classifier"):
        sents = split_sentences(r["article"])
        scores = ext.score(sents)
        n = sum(1 for s in scores if s is not None)
        if n == 0:
            continue
        lab = oracle_labels(sents[:n], r["highlights"])
        sc = scores[:n]
        order = set(np.argsort(-np.array(sc))[:3].tolist())
        probs += sc
        labels += lab
        thr += [int(s >= 0.5) for s in sc]
        top3 += [int(i in order) for i in range(n)]
        lead += [int(i < 3) for i in range(n)]
    return {"BertSumExt-style, threshold 0.5": classifier_metrics(probs, labels, thr),
            "BertSumExt-style, top-3 per article": classifier_metrics(probs, labels, top3),
            "Lead-3 rule (first 3 = positive)": classifier_metrics(probs, labels, lead)}


def run_abstractive(model: AbstractiveSummarizer, rows: list[dict], batch_size: int) -> list[dict]:
    items = []
    for i in tqdm(range(0, len(rows), batch_size), desc=f"Abstractive ({model.model_name})"):
        chunk = rows[i:i + batch_size]
        t0 = time.perf_counter()
        outs = model.summarize_batch([r["article"] for r in chunk], batch_size=batch_size)
        per = (time.perf_counter() - t0) / len(chunk)
        for r, s in zip(chunk, outs):
            items.append({"id": r["id"], "summary": s, "article": clean_article(r["article"]),
                          "reference": r["highlights"], "latency": per})
    return items


def n_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def hardware() -> dict:
    import torch

    info = {"platform": platform.platform(), "python": platform.python_version(),
            "cpu": platform.processor() or platform.machine(), "torch": torch.__version__}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    import os

    info["cpu_threads"] = os.cpu_count()
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


# =========================================================================== chart
def plot_comparison(our: dict[str, float], n_label: str, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    entries = [(k, v, "paper") for k, v in PUBLISHED_ROUGE1.items()]
    entries += [(f"{k} (ours)", v, "ours") for k, v in our.items() if v is not None]
    entries.sort(key=lambda e: e[1])
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    paper_c, ours_c = "#b9b8b1", "#2a78d6"

    fig, ax = plt.subplots(figsize=(10, 0.48 * len(entries) + 1.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    y = np.arange(len(entries))
    colors = [ours_c if e[2] == "ours" else paper_c for e in entries]
    bars = ax.barh(y, [e[1] for e in entries], color=colors, height=0.72, edgecolor="#fcfcfb", linewidth=2)
    for b, e in zip(bars, entries):
        if e[2] == "ours":
            b.set_hatch("///")
            b.set_edgecolor("#ffffff")
        ax.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2, f"{e[1]:.2f}",
                va="center", ha="left", fontsize=9, color=ink,
                fontweight="bold" if e[2] == "ours" else "normal")
    ax.set_yticks(y, [e[0] for e in entries], fontsize=9, color=ink)
    lo = min(e[1] for e in entries)
    ax.set_xlim(max(0, lo - 6), max(e[1] for e in entries) + 3)
    ax.set_xlabel("ROUGE-1 F1", color=muted)
    ax.tick_params(colors=muted, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.xaxis.grid(True, color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("ROUGE-1 on CNN/DailyMail test", loc="left", fontsize=13, color=ink, pad=22)
    ax.text(0, 1.02, "Published numbers are on the full test set (11,490 articles); ours are on a "
            f"{n_label} sample, so they are indicative only.", transform=ax.transAxes,
            fontsize=8.5, color=muted)
    ax.legend(handles=[Patch(facecolor=ours_c, hatch="///", edgecolor="white", label=f"Ours ({n_label} sample)"),
                       Patch(facecolor=paper_c, label="Paper (full test set)")],
              loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


# =========================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200, help="test articles for Lead-3 / extractive / oracle")
    ap.add_argument("--abs-n", type=int, default=100, help="test articles for the abstractive model")
    ap.add_argument("--val-n", type=int, default=300, help="validation articles for classifier metrics (0=skip)")
    ap.add_argument("--fast", action="store_true", help="use DistilBART instead of BART-large")
    ap.add_argument("--abs-batch-size", type=int, default=None)
    ap.add_argument("--skip-abs", action="store_true")
    ap.add_argument("--xsum", action="store_true", help="also stress-test on XSum")
    ap.add_argument("--xsum-n", type=int, default=50)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = get_device(args.device)
    t_start = time.time()
    bs = args.abs_batch_size or (8 if device == "cuda" else 4)

    rows = load_cnndm("test", args.n)
    ext = ExtractiveSummarizer(device=device)
    log.info("Device %s | extractive method: %s", device, ext.method)

    # ---------------------------------------------------------------- Lead-3 / extractive / oracle
    lead_items, ext_items, orc_items, ref_items = [], [], [], []
    for r in tqdm(rows, desc="Lead-3 / extractive / oracle"):
        art = clean_article(r["article"])
        sents = split_sentences(art)
        l3 = lead3(art, sentences=sents)
        ex = ext.summarize(art, sentences=sents)
        orc = [sents[i] for i in greedy_oracle(sents, r["highlights"])]
        base = {"id": r["id"], "article": art, "reference": r["highlights"]}
        lead_items.append({**base, "summary": l3["summary"], "summary_sents": l3["summary_sents"],
                           "latency": l3["latency"]})
        ext_items.append({**base, "summary": ex["summary"], "summary_sents": ex["summary_sents"],
                          "latency": ex["latency"]})
        orc_items.append({**base, "summary": " ".join(orc), "summary_sents": orc, "latency": None})
        ref_items.append({"id": r["id"], "article": art, "summary": r["highlights"].replace("\n", " ")})

    ext_label = "Extractive: BertSumExt-style (ours)" if ext.is_trained else "Extractive: TextRank fallback"
    dfs = {"Lead-3": per_article(lead_items), ext_label: per_article(ext_items),
           "Oracle (extractive upper bound)": per_article(orc_items)}
    abs_label, abs_items, xsum_rows, abs_model = None, [], [], None

    if not args.skip_abs and args.abs_n > 0:
        name = ABS_MODEL_FAST if args.fast else ABS_MODEL
        abs_model = AbstractiveSummarizer(name, device=device)
        abs_items = run_abstractive(abs_model, rows[:args.abs_n], bs)
        abs_label = "Abstractive: DistilBART-CNN" if args.fast else "Abstractive: BART-large-CNN"
        dfs[abs_label] = per_article(abs_items)
        if args.xsum:
            try:
                xs = load_xsum("test", args.xsum_n)
                x_items = run_abstractive(abs_model, xs, bs)
                x_lead = []
                for r in xs:
                    art = clean_article(r["article"])
                    l3 = lead3(art)
                    x_lead.append({"article": art, "summary": l3["summary"], "summary_sents": l3["summary_sents"],
                                   "reference": r["highlights"], "latency": l3["latency"]})
                xsum_rows = [aggregate("Lead-3", "XSum", per_article(x_lead)),
                             aggregate(abs_label, "XSum", per_article(x_items))]
            except Exception as e:
                log.warning("XSum stress test skipped: %s", e)

    table = [aggregate(m, "CNN/DM", d) for m, d in dfs.items()]
    ref_row = aggregate("Reference highlights", "CNN/DM", per_article(ref_items))
    ref_row["avg_latency_s"] = None
    table.append(ref_row)
    table += xsum_rows

    # paired comparison: every method on exactly the articles the abstractive model saw
    paired = []
    if abs_items:
        k = len(abs_items)
        paired = [aggregate(m, "CNN/DM", d.iloc[:k]) for m, d in dfs.items()]

    # ---------------------------------------------------------------- token lengths / long documents
    from transformers import AutoTokenizer

    from src.hub import from_pretrained

    bert_tok = ext.encoder.tok if ext.is_trained else from_pretrained(AutoTokenizer, ENCODER_NAME)
    bart_tok = abs_model.tok if abs_model else from_pretrained(AutoTokenizer, ABS_MODEL)
    arts = [it["article"] for it in lead_items]
    bert_len = [len(bert_tok(a)["input_ids"]) for a in arts]
    bart_len = [len(bart_tok(a)["input_ids"]) for a in arts]
    for d in dfs.values():
        d["bert_tokens"] = bert_len[:len(d)]
        d["bart_tokens"] = bart_len[:len(d)]
    long_rows = []
    for m, d in dfs.items():
        for limit, col in ((ENCODER_MAX_TOKENS, "bert_tokens"), (BART_MAX_INPUT, "bart_tokens")):
            for grp, mask in (("over", d[col] > limit), ("within", d[col] <= limit)):
                sub = d[mask]
                long_rows.append({"method": m, "limit": f"{limit} ({col.split('_')[0].upper()})",
                                  "group": grp, "n_articles": int(len(sub)),
                                  **{k: round(float(sub[k].mean()), 2) if len(sub) else None
                                     for k in ROUGE_KEYS}})
    long_stats = {"n_test": len(arts),
                  "over_512_bert": int(sum(x > ENCODER_MAX_TOKENS for x in bert_len)),
                  "over_1024_bart": int(sum(x > BART_MAX_INPUT for x in bart_len)),
                  "median_bert_tokens": float(np.median(bert_len)),
                  "median_bart_tokens": float(np.median(bart_len))}
    pd.DataFrame(long_rows).to_csv(RESULTS / "long_doc.csv", index=False)

    # ---------------------------------------------------------------- factual check stats
    fact = {}
    if abs_items:
        n_flags = [len(check_summary(it["summary"], it["article"])["flags"]) for it in abs_items]
        dfs[abs_label]["n_flags"] = n_flags
        fact = {"model": abs_label, "backend": backend_name(), "n_summaries": len(n_flags),
                "pct_with_flag": round(100 * float(np.mean([f > 0 for f in n_flags])), 2),
                "avg_flags_per_summary": round(float(np.mean(n_flags)), 3),
                "max_flags": int(max(n_flags))}
        (RESULTS / "factcheck_stats.json").write_text(json.dumps(fact, indent=2))

    # ---------------------------------------------------------------- classifier metrics
    clf = {}
    if ext.is_trained:
        clf["test"] = collect_classifier(ext, rows)
        if args.val_n > 0:
            clf["validation"] = collect_classifier(ext, load_cnndm("validation", args.val_n))
        flat = [{"split": s, "rule": rule, **{k: v for k, v in m.items() if k != "confusion"}}
                for s, d in clf.items() for rule, m in d.items()]
        pd.DataFrame(flat).to_csv(RESULTS / "classifier_metrics.csv", index=False)
        (RESULTS / "classifier_metrics.json").write_text(json.dumps(clf, indent=2))

    # ---------------------------------------------------------------- efficiency
    train_log = json.loads((RESULTS / "train_log.json").read_text()) if (RESULTS / "train_log.json").exists() else {}
    eff = {"device": device, "hardware": hardware(),
           "avg_latency_s": {r["method"]: r["avg_latency_s"] for r in table if r["dataset"] == "CNN/DM"},
           "abstractive_batch_size": bs if abs_items else None,
           "model_params": {}, "training_minutes": round(train_log.get("seconds", {}).get("total", 0) / 60, 2)}
    if ext.is_trained:
        eff["model_params"]["extractive encoder (frozen) " + ext.encoder_name] = n_params(ext.encoder.model)
        eff["model_params"]["extractive head (trained)"] = n_params(ext.head)
        eff["extractive_checkpoint_mb"] = round(EXT_CHECKPOINT.stat().st_size / 1e6, 2)
    if abs_model:
        eff["model_params"][abs_model.model_name] = n_params(abs_model.model)
    (RESULTS / "efficiency.json").write_text(json.dumps(eff, indent=2))

    # ---------------------------------------------------------------- save
    df = pd.DataFrame(table)
    df.to_csv(RESULTS / "scores.csv", index=False)
    meta = {"device": device, "n_test": args.n, "abs_n": len(abs_items), "val_n_classifier": args.val_n,
            "extractive_method": ext.method, "abstractive_model": abs_model.model_name if abs_model else None,
            "rouge": "rouge-score 0.1.2, F1 x100, Porter stemmer; ROUGE-L = rougeLsum (summary-level); "
                     "95% CI = percentile bootstrap over articles (2000 resamples)",
            "xsum_n": args.xsum_n if xsum_rows else 0,
            "runtime_minutes": round((time.time() - t_start) / 60, 2)}
    (RESULTS / "scores.json").write_text(json.dumps(
        {"meta": meta, "scores": table, "paired_on_abstractive_subset": paired, "long_documents": long_stats,
         "factcheck": fact, "classifier": clf, "efficiency": eff, "published_rouge1": PUBLISHED_ROUGE1},
        indent=2))
    with (RESULTS / "per_article.jsonl").open("w") as f:
        abs_by_id = {it["id"]: it["summary"] for it in abs_items}
        score_cols = {m: d.set_index("id") for m, d in dfs.items()}
        for l, e, o in zip(lead_items, ext_items, orc_items):
            rec = {"id": l["id"], "lead3": l["summary"], "extractive": e["summary"], "oracle": o["summary"],
                   "abstractive": abs_by_id.get(l["id"]), "reference": l["reference"], "scores": {}}
            for m, d in score_cols.items():
                if l["id"] in d.index:
                    rec["scores"][m] = {k: round(float(v), 3) for k, v in d.loc[l["id"]].items()
                                        if isinstance(v, (int, float, np.floating, np.integer))
                                        and not pd.isna(v)}
            f.write(json.dumps(rec) + "\n")

    ours = {"Lead-3": table[0]["rouge1"], "BertSumExt-style, frozen BERT": table[1]["rouge1"]}
    if abs_items:
        ours["BART-large-CNN" if not args.fast else "DistilBART-CNN"] = table[3]["rouge1"]
    n_label = f"{args.n}-article" if len(abs_items) in (0, args.n) else f"{args.n}/{len(abs_items)}-article"
    plot_comparison(ours, n_label, RESULTS / "rouge_comparison.png")

    try:
        from src.make_figures import make_all

        make_all()
    except Exception as e:  # figures are a nice-to-have; never fail the evaluation
        log.warning("Figure generation failed: %s", e)

    with pd.option_context("display.max_columns", 30, "display.width", 220):
        print(df[["method", "dataset", "n_articles", "rouge1", "rouge1_ci_lo", "rouge1_ci_hi", "rouge2",
                  "rougeL", "novel_2", "repeated_3", "avg_words", "avg_latency_s"]].to_string(index=False))
    log.info("Done in %.1f min -> %s", (time.time() - t_start) / 60, RESULTS)


if __name__ == "__main__":
    main()
