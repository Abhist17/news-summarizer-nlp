"""Generate all report figures (PNG) from results/ into report/figures/.

Usage:  python -m src.make_figures
"""
from __future__ import annotations

import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import BART_MAX_INPUT, ENCODER_MAX_TOKENS, RESULTS, ROOT  # noqa: E402

FIG = ROOT / "report" / "figures"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# colour follows the entity, consistently across every figure
COLORS = {"Lead-3": "#8f8e87", "Extractive": "#2a78d6", "Abstractive": "#1baf7a",
          "Reference": "#eb6834", "Oracle": "#6250d6"}


def _kind(method: str) -> str:
    for k in ("Lead-3", "Extractive", "Abstractive", "Reference", "Oracle"):
        if method.startswith(k):
            return k
    return "Lead-3"


def _short(method: str) -> str:
    return {"Lead-3": "Lead-3", "Extractive": "Extractive (ours)", "Abstractive": "BART-large-CNN",
            "Reference": "Reference highlights", "Oracle": "Oracle"}[_kind(method)]


def _style(ax, title: str, subtitle: str | None = None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=20 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=8.5, color=MUTED)
    ax.set_axisbelow(True)


def _fig(w=8.5, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, facecolor=fig.get_facecolor())
    plt.close(fig)


def novel_ngrams(scores: list[dict]):
    rows = [r for r in scores if r["dataset"] == "CNN/DM" and not r["method"].startswith("Oracle")]
    fig, ax = _fig()
    ns = [1, 2, 3]
    w = 0.8 / len(rows)
    for j, r in enumerate(rows):
        vals = [r[f"novel_{n}"] for n in ns]
        x = np.arange(3) + (j - (len(rows) - 1) / 2) * w
        bars = ax.bar(x, vals, w * 0.92, color=COLORS[_kind(r["method"])], label=_short(r["method"]))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(np.arange(3), ["unigrams", "bigrams", "trigrams"])
    ax.set_ylabel("% of summary n-grams not in article", color=MUTED, fontsize=9)
    ax.yaxis.grid(True, color=GRID)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _style(ax, "Novel n-grams: how abstractive is each summary?",
           "Extractive methods copy whole sentences (≈0%); human highlights are far more abstractive than BART.")
    _save(fig, "novel_ngrams.png")


def latency(eff: dict, scores: list[dict]):
    lat = {m: v for m, v in eff["avg_latency_s"].items() if v is not None}
    items = sorted(lat.items(), key=lambda kv: kv[1])
    floor = 2e-5  # Lead-3 takes microseconds (older results round it to 0): keep its bar visible on the log axis
    fig, ax = _fig(8.5, 0.6 * len(items) + 1.6)
    y = np.arange(len(items))
    bars = ax.barh(y, [max(v, floor) for _, v in items], color=[COLORS[_kind(m)] for m, _ in items], height=0.6)
    for b, (_, v) in zip(bars, items):
        lab = "< 0.1 ms" if v < 1e-4 else f"{v * 1000:.1f} ms" if v < 1 else f"{v:.2f} s"
        ax.text(b.get_width() * 1.15, b.get_y() + b.get_height() / 2, lab, va="center", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_yticks(y, [_short(m) for m, _ in items])
    ax.set_xlabel("seconds per article (log scale)", color=MUTED, fontsize=9)
    ax.xaxis.grid(True, color=GRID)
    ax.set_xlim(floor / 4, max(v for _, v in items) * 12)
    hw = eff["hardware"].get("gpu") if eff["device"] == "cuda" else eff["hardware"].get("cpu")
    _style(ax, "Average latency per article",
           f"{hw}; BART in batches of {eff.get('abstractive_batch_size')} (latency = batch time / batch size).")
    fig.tight_layout()
    fig.subplots_adjust(left=0.2)
    fig.savefig(FIG / "latency.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def rouge_vs_length(per: list[dict]):
    recs = []
    for p in per:
        for m, s in p["scores"].items():
            if "bart_tokens" in s and "rouge1" in s and not m.startswith("Oracle"):
                recs.append({"method": m, "tokens": s["bart_tokens"], "rouge1": s["rouge1"]})
    df = pd.DataFrame(recs)
    if df.empty:
        return
    edges = np.unique(np.quantile(df["tokens"], np.linspace(0, 1, 6)))
    df["bin"] = pd.cut(df["tokens"], edges, include_lowest=True)
    fig, ax = _fig()
    for m, d in df.groupby("method"):
        g = d.groupby("bin", observed=True).agg(x=("tokens", "median"), y=("rouge1", "mean"), n=("rouge1", "size"))
        ax.plot(g["x"], g["y"], marker="o", markersize=6, linewidth=2, color=COLORS[_kind(m)],
                label=f"{_short(m)} (n={len(d)})")
    ax.axvline(BART_MAX_INPUT, color=MUTED, linestyle="--", linewidth=1)
    ax.text(BART_MAX_INPUT, ax.get_ylim()[1], " BART input limit (1024)", fontsize=8, color=MUTED, va="top")
    ax.set_xlabel("article length (BART tokens; points = median of each length quintile)", color=MUTED, fontsize=9)
    ax.set_ylabel("mean ROUGE-1", color=MUTED, fontsize=9)
    ax.yaxis.grid(True, color=GRID)
    ax.legend(frameon=False, fontsize=8.5)
    _style(ax, "ROUGE-1 vs article length",
           f"Extractive encoder sees only the first {ENCODER_MAX_TOKENS} BERT tokens; BART (eval) truncates at "
           f"{BART_MAX_INPUT}.")
    _save(fig, "rouge_vs_length.png")


def confusion(clf: dict):
    if "test" not in clf:
        return
    rules = [r for r in clf["test"] if r.startswith("BertSumExt")]
    fig, axes = plt.subplots(1, len(rules), figsize=(3.6 * len(rules), 3.9), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, rule in zip(axes, rules):
        m = clf["test"][rule]
        cm = np.array(m["confusion"])
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({100 * norm[i, j]:.1f}% of row)", ha="center", va="center",
                        fontsize=9, color="white" if norm[i, j] > 0.55 else INK)
        ax.set_xticks([0, 1], ["pred: no", "pred: yes"], fontsize=8.5, color=MUTED)
        ax.set_yticks([0, 1], ["oracle: no", "oracle: yes"], fontsize=8.5, color=MUTED)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        name = rule.split(", ", 1)[1].replace(" (F1-tuned on val)", "\n(F1-tuned on validation)")
        ax.set_title(f"{name}\nF1 {m['f1']:.3f} · P {m['precision']:.2f} · R {m['recall']:.2f}",
                     fontsize=9.5, color=INK)
    auc = clf["test"][rules[0]]["roc_auc"]
    fig.suptitle(f"Extractive sentence classifier vs greedy-oracle labels (test sample, ROC-AUC {auc:.3f})",
                 fontsize=11.5,
                 color=INK, x=0.02, ha="left")
    _save(fig, "confusion_matrix.png")


def training_curve(tl: dict):
    h = pd.DataFrame(tl["history"])
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    axes[0].plot(h["epoch"], h["loss"], marker="o", color=COLORS["Extractive"], linewidth=2)
    axes[0].set_xlabel("epoch", color=MUTED, fontsize=9)
    _style(axes[0], "Training loss (BCE)")
    axes[0].yaxis.grid(True, color=GRID)
    axes[1].plot(h["epoch"], h["rouge1"], marker="o", color=COLORS["Extractive"], linewidth=2, label="model")
    axes[1].axhline(tl["val_lead3"]["rouge1"], color=COLORS["Lead-3"], linestyle="--", linewidth=1.5,
                    label="Lead-3")
    axes[1].axvline(tl["best_epoch"], color=MUTED, linestyle=":", linewidth=1)
    axes[1].set_xlabel("epoch", color=MUTED, fontsize=9)
    axes[1].legend(frameon=False, fontsize=8.5)
    axes[1].yaxis.grid(True, color=GRID)
    _style(axes[1], f"Validation ROUGE-1 (n={tl['val_n']})")
    _save(fig, "training_curve.png")


def pipeline_diagram():
    """Static PNG of the system pipeline (mirrors the Mermaid diagram in README / REPORT)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 116)
    ax.set_ylim(-5, 55)
    ax.axis("off")

    def box(x, y, w, h, text, color, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    facecolor=color, edgecolor="#9a998f", linewidth=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.2, color=INK,
                fontweight="bold" if bold else "normal", linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
                                     color="#6d6c66", linewidth=1))

    grey, blue, green, orange, purple = "#eeede9", "#dbe9fb", "#d8f1e6", "#fde8d7", "#e6e2fa"
    box(1, 24, 13, 8, "News article\n(CNN/DM or pasted)", grey, True)
    box(18, 24, 15, 8, "Clean + split\n(punkt sentences,\nDM header removal)", grey)
    arrow(14.4, 28, 17.6, 28)
    # branches
    box(38, 44, 17, 7, "Lead-3\nfirst 3 sentences", grey, True)
    box(38, 24, 17, 9, "BERT-base (frozen)\n[CLS] + mean-pool per\nsentence, alt. segments,\n512 tokens", blue, True)
    box(59, 24, 17, 9, "2-layer Transformer\nhead + sigmoid\n(trained on oracle)", blue)
    box(80, 24, 15, 9, "Top-3 + trigram\nblocking, article\norder", blue)
    box(38, 3, 17, 9, "Abstractive input\n> 1024 BART tokens?", green, True)
    box(59, 6.5, 17, 7, "no: BART-large-CNN\nbeam 4, no-repeat 3", green)
    box(59, -1.5, 17, 7, "yes: chunk ~900 tok,\nsummarize chunks,\nthen summaries", green)
    box(80, 3, 15, 9, "Fact check\n(spaCy NER / regex:\nnew names, numbers)", orange)
    for y in (47.5, 28.5, 7.5):
        arrow(33.4, 28, 37.6, y)
    arrow(55.4, 28.5, 58.6, 28.5)
    arrow(76.4, 28.5, 79.6, 28.5)
    arrow(55.4, 8.5, 58.6, 10)
    arrow(55.4, 6.5, 58.6, 2)
    arrow(76.4, 10, 79.6, 8)
    arrow(76.4, 2, 79.6, 6)
    box(99.5, 16, 15, 24, "Metrics\nROUGE-1/2/L\nnovel n-gram %\nrepeated n-gram %\ncompression\n\n→ Streamlit app\n→ results/",
        purple, True)
    arrow(55.4, 47.5, 99.1, 38)
    arrow(95.4, 28.5, 99.1, 28.5)
    arrow(95.4, 7.5, 99.1, 18)
    _save(fig, "pipeline.png")


def make_all():
    FIG.mkdir(parents=True, exist_ok=True)
    pipeline_diagram()
    js = json.loads((RESULTS / "scores.json").read_text(encoding="utf-8"))
    if (RESULTS / "rouge_comparison.png").exists():
        shutil.copy(RESULTS / "rouge_comparison.png", FIG / "rouge_comparison.png")
    novel_ngrams(js["scores"])
    latency(js["efficiency"], js["scores"])
    per = [json.loads(l) for l in (RESULTS / "per_article.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rouge_vs_length(per)
    confusion(js.get("classifier", {}))
    if (RESULTS / "train_log.json").exists():
        training_curve(json.loads((RESULTS / "train_log.json").read_text(encoding="utf-8")))
    print(f"Figures written to {FIG}")


if __name__ == "__main__":
    make_all()
