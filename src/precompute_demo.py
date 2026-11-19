"""Precompute demo outputs for ~10 diverse CNN/DM test articles -> demo_cache/samples.json.

Picks articles across the length distribution, including at least two that
exceed BART's 1024-token limit (to show chunking), then runs Lead-3,
Extractive, both abstractive models (BART-large & DistilBART), all metrics and
the factual check. The app then shows these instantly.

Usage:  python -m src.precompute_demo [--n-samples 10] [--pool 400] [--skip-fast]
"""
from __future__ import annotations

import argparse
import json
import logging
import time

from config import ABS_MODELS, BART_MAX_INPUT, DEMO_CACHE, get_device
from src.abstractive import AbstractiveSummarizer
from src.data import clean_article, load_cnndm, split_sentences
from src.extractive import ExtractiveSummarizer
from src.factcheck import backend_name
from src.pipeline import run_article

log = logging.getLogger("precompute")


def choose_samples(rows: list[dict], tok, n: int, n_long: int = 2) -> list[tuple[dict, int]]:
    scored = []
    for r in rows:
        art = clean_article(r["article"])
        if len(split_sentences(art)) < 5:
            continue
        scored.append((r, len(tok(art)["input_ids"])))
    scored.sort(key=lambda x: x[1])
    long_ = [s for s in scored if s[1] > BART_MAX_INPUT]
    short = [s for s in scored if s[1] <= BART_MAX_INPUT]
    picks = []
    if long_:  # one moderately long + one very long
        picks.append(long_[len(long_) // 3])
        if n_long > 1 and len(long_) > 1:
            picks.append(long_[min(len(long_) - 1, (len(long_) * 9) // 10)])
    k = n - len(picks)
    for j in range(k):
        q = (j + 0.5) / k
        cand = short[int(q * (len(short) - 1))]
        if cand not in picks:
            picks.append(cand)
    picks.sort(key=lambda x: x[1])
    return picks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--pool", type=int, default=400, help="test articles to choose from")
    ap.add_argument("--skip-fast", action="store_true", help="don't precompute DistilBART outputs")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = get_device(args.device)
    t0 = time.time()

    rows = load_cnndm("test", args.pool)
    ext = ExtractiveSummarizer(device=device)
    models = {}
    for name, mid in ABS_MODELS.items():
        if args.skip_fast and "fast" in name.lower():
            continue
        log.info("Loading %s", mid)
        models[name] = AbstractiveSummarizer(mid, device=device)
    first = next(iter(models.values()))
    picks = choose_samples(rows, first.tok, args.n_samples)

    samples = []
    for i, (r, ntok) in enumerate(picks, 1):
        log.info("[%d/%d] %s (%d BART tokens)", i, len(picks), r["id"], ntok)
        res = run_article(r["article"], r["highlights"], ext, models, long_mode="chunk")
        first_sent = res["sentences"][0] if res["sentences"] else r["id"]
        title = first_sent if len(first_sent) <= 80 else first_sent[:77].rsplit(" ", 1)[0] + "..."
        res.update(id=r["id"], title=title, n_bart_tokens=ntok)
        res["n_flags"] = {k: len(v["factcheck"]["flags"]) for k, v in res["abstractive"].items()}
        samples.append(res)

    out = {"meta": {"created": time.strftime("%Y-%m-%d %H:%M"), "device": device,
                    "extractive_method": ext.method, "factcheck_backend": backend_name(),
                    "models": {k: v.model_name for k, v in models.items()},
                    "seconds": round(time.time() - t0, 1)},
           "samples": samples}
    path = DEMO_CACHE / "samples.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for s in samples:
        a = next(iter(s["abstractive"].values()))
        log.info("%-14s %5d tok chunked=%s flags=%s R1 lead=%.1f ext=%.1f abs=%.1f | %s", s["id"][:14],
                 s["n_bart_tokens"], a["chunked"], s["n_flags"], s["lead3"]["rouge"]["rouge1"],
                 s["extractive"]["rouge"]["rouge1"], a["rouge"]["rouge1"], s["title"][:50])
    log.info("Saved %d samples -> %s in %.1f min", len(samples), path, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
