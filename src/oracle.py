"""Greedy ROUGE oracle used to create extractive training labels (as in BertSum).

Greedily adds the sentence that most increases ROUGE-1 F + ROUGE-2 F of the
selected set against the reference, stopping when nothing improves or when
`max_sents` sentences are chosen. Uses fast set-based n-gram overlap, the same
approximation as BertSum's `greedy_selection`.
"""
from __future__ import annotations

import re

_TOK = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOK.findall(text.lower())


def _ngram_set(tokens: list[str], n: int) -> set[tuple]:
    return set(zip(*(tokens[i:] for i in range(n))))


def _f1(cand: set, ref: set) -> float:
    if not cand or not ref:
        return 0.0
    overlap = len(cand & ref)
    if overlap == 0:
        return 0.0
    p, r = overlap / len(cand), overlap / len(ref)
    return 2 * p * r / (p + r)


def greedy_oracle(sentences: list[str], reference: str, max_sents: int = 3) -> list[int]:
    """Return sorted indices of the oracle-selected sentences."""
    ref_toks = _tokens(reference)
    ref1, ref2 = _ngram_set(ref_toks, 1), _ngram_set(ref_toks, 2)
    sent_toks = [_tokens(s) for s in sentences]
    sent1 = [_ngram_set(t, 1) for t in sent_toks]
    sent2 = [_ngram_set(t, 2) for t in sent_toks]

    selected: list[int] = []
    best = 0.0
    for _ in range(max_sents):
        cur_best, cur_id = best, -1
        sel1 = set().union(*(sent1[j] for j in selected)) if selected else set()
        sel2 = set().union(*(sent2[j] for j in selected)) if selected else set()
        for i in range(len(sentences)):
            if i in selected:
                continue
            score = _f1(sel1 | sent1[i], ref1) + _f1(sel2 | sent2[i], ref2)
            if score > cur_best:
                cur_best, cur_id = score, i
        if cur_id == -1:
            break
        selected.append(cur_id)
        best = cur_best
    return sorted(selected)


def oracle_labels(sentences: list[str], reference: str, max_sents: int = 3) -> list[int]:
    chosen = set(greedy_oracle(sentences, reference, max_sents))
    return [1 if i in chosen else 0 for i in range(len(sentences))]
