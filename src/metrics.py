"""Evaluation metrics: ROUGE-1/2/L, % novel n-grams, % repeated n-grams, compression ratio.

ROUGE uses Google's `rouge-score` package with Porter stemming. ROUGE-L is the
summary-level variant (`rougeLsum`, sentences separated by newlines), which is
what pyrouge / the papers report as ROUGE-L.
"""
from __future__ import annotations

import re
from collections import Counter

_TOK = re.compile(r"[a-z0-9]+")
_scorer = None


def tokenize(text: str) -> list[str]:
    return _TOK.findall((text or "").lower())


def ngrams(tokens: list[str], n: int) -> list[tuple]:
    return list(zip(*(tokens[i:] for i in range(n))))


def _get_scorer():
    global _scorer
    if _scorer is None:
        from rouge_score import rouge_scorer

        _scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    return _scorer


def _as_lines(text_or_sents) -> str:
    if isinstance(text_or_sents, (list, tuple)):
        return "\n".join(s.strip() for s in text_or_sents if s.strip())
    text = (text_or_sents or "").strip()
    if "\n" in text:
        return text
    from src.data import split_sentences

    return "\n".join(split_sentences(text, min_words=0))


def rouge_scores(summary, reference) -> dict:
    """ROUGE F1 x100. `summary`/`reference` may be strings or lists of sentences."""
    s = _get_scorer().score(_as_lines(reference), _as_lines(summary))
    return {"rouge1": 100 * s["rouge1"].fmeasure,
            "rouge2": 100 * s["rouge2"].fmeasure,
            "rougeL": 100 * s["rougeLsum"].fmeasure}


def sentence_ngrams(text, n: int) -> list[tuple]:
    """n-grams that do not cross sentence boundaries (so concatenating two copied,
    non-adjacent sentences does not create artificial 'novel' n-grams)."""
    if isinstance(text, (list, tuple)):
        sents = text
    else:
        from src.data import split_sentences

        sents = split_sentences(text, min_words=0) if text and text.strip() else []
    out: list[tuple] = []
    for s in sents:
        out += ngrams(tokenize(s), n)
    return out


def novel_ngram_pct(summary: str, source: str, n: int) -> float:
    """% of the summary's n-grams that never appear in the source (See et al. 2017)."""
    summ = sentence_ngrams(summary, n)
    if not summ:
        return 0.0
    src = set(sentence_ngrams(source, n)) | set(ngrams(tokenize(source), n))
    return 100.0 * sum(1 for g in summ if g not in src) / len(summ)


def repeated_ngram_pct(summary: str, n: int) -> float:
    """% of the summary's n-gram occurrences that repeat an earlier n-gram in the same summary
    (n-grams are taken within sentences; repeats are counted across the whole summary)."""
    summ = sentence_ngrams(summary, n)
    if not summ:
        return 0.0
    counts = Counter(summ)
    return 100.0 * sum(c - 1 for c in counts.values()) / len(summ)


def compression_ratio(source: str, summary: str) -> float:
    s = len(tokenize(summary))
    return len(tokenize(source)) / s if s else 0.0


def summary_stats(summary: str, source: str) -> dict:
    return {
        "words": len(tokenize(summary)),
        "compression": compression_ratio(source, summary),
        "novel_1": novel_ngram_pct(summary, source, 1),
        "novel_2": novel_ngram_pct(summary, source, 2),
        "novel_3": novel_ngram_pct(summary, source, 3),
        "repeated_2": repeated_ngram_pct(summary, 2),
        "repeated_3": repeated_ngram_pct(summary, 3),
    }
