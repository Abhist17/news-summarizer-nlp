"""Lead-3 baseline: the first three sentences of the article."""
from __future__ import annotations

import time

from src.data import split_sentences


def lead3(article: str, k: int = 3, sentences: list[str] | None = None) -> dict:
    t0 = time.perf_counter()
    sents = sentences if sentences is not None else split_sentences(article)
    idx = list(range(min(k, len(sents))))
    chosen = [sents[i] for i in idx]
    return {
        "summary": " ".join(chosen),
        "summary_sents": chosen,
        "indices": idx,
        "latency": time.perf_counter() - t0,
    }
