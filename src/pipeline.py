"""Run all three summarizers + metrics + factual check on one article.

Produces the exact same dict structure for precomputed demo samples
(demo_cache/samples.json) and for live pasted articles in the app.
"""
from __future__ import annotations

from src.data import clean_article, highlights_to_sents, split_sentences
from src.factcheck import check_summary
from src.lead3 import lead3
from src.metrics import rouge_scores, summary_stats, tokenize


def _score_block(res: dict, article: str, reference: str | None) -> dict:
    res["metrics"] = summary_stats(res["summary"], article)
    res["metrics"]["latency"] = res.get("latency", 0.0)
    if reference:
        summ = res.get("summary_sents") or res["summary"]
        res["rouge"] = rouge_scores(summ, highlights_to_sents(reference))
    return res


def run_article(article: str, reference: str | None, extractive, abstractive: dict,
                long_mode: str = "chunk") -> dict:
    """`abstractive` maps a display name -> AbstractiveSummarizer (may be empty)."""
    article_clean = clean_article(article)
    sentences = split_sentences(article)
    out = {
        "article": article_clean,
        "reference": reference,
        "n_article_words": len(tokenize(article_clean)),
        "sentences": sentences,
        "lead3": _score_block(lead3(article_clean, sentences=sentences), article_clean, reference),
        "extractive": _score_block(extractive.summarize(article_clean, sentences=sentences),
                                   article_clean, reference),
        "abstractive": {},
    }
    for name, model in abstractive.items():
        res = model.summarize(article_clean, long_mode=long_mode)
        res = _score_block(res, article_clean, reference)
        res["factcheck"] = check_summary(res["summary"], article_clean)
        out["abstractive"][name] = res
    return out
