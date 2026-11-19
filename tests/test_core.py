"""Unit tests for Lead-3, trigram blocking, n-gram metrics, the oracle and the fact check."""
import math

import pytest

from src.extractive import get_trigrams, has_trigram_overlap, select_sentences, textrank_scores
from src.factcheck import check_summary
from src.lead3 import lead3
from src.metrics import compression_ratio, novel_ngram_pct, repeated_ngram_pct, rouge_scores
from src.oracle import greedy_oracle, oracle_labels


# ------------------------------------------------------------------ Lead-3
def test_lead3_empty():
    r = lead3("")
    assert r["summary"] == "" and r["indices"] == []
    assert lead3("   \n  ")["summary"] == ""


def test_lead3_fewer_than_three_sentences():
    r = lead3("The cat sat on the mat today. The dog barked at it loudly.")
    assert r["indices"] == [0, 1]
    assert r["summary"] == "The cat sat on the mat today. The dog barked at it loudly."


def test_lead3_single_short_sentence_kept():
    # a lone short sentence is below the min-word filter but must not vanish
    assert lead3("Hello there.")["summary"] == "Hello there."


def test_lead3_takes_first_three():
    text = " ".join(f"This is sentence number {w} here." for w in ["one", "two", "three", "four", "five"])
    r = lead3(text)
    assert r["indices"] == [0, 1, 2]
    assert "four" not in r["summary"] and "three" in r["summary"]


# ------------------------------------------------------------------ trigram blocking
def test_get_trigrams():
    assert get_trigrams("The cat sat down") == {("the", "cat", "sat"), ("cat", "sat", "down")}
    assert get_trigrams("Two words") == set()


def test_trigram_overlap():
    assert has_trigram_overlap("police said the man fled", ["The man fled on foot"])
    assert not has_trigram_overlap("police said the man fled", ["A woman stayed home"])


def test_trigram_blocking_skips_redundant_sentence():
    sents = ["The president visited Paris on Monday.",          # 0
             "Weather was mild across the region.",              # 1
             "On Monday the president visited Paris for talks.",  # 2 shares "president visited paris"
             "Markets rose sharply after the news."]              # 3
    scores = [0.9, 0.2, 0.8, 0.5]
    sel, blocked = select_sentences(sents, scores, k=3, trigram_blocking=True)
    assert blocked == [2]
    assert sel == [0, 1, 3]              # returned in original article order
    sel_nb, blocked_nb = select_sentences(sents, scores, k=3, trigram_blocking=False)
    assert sel_nb == [0, 2, 3] and blocked_nb == []


def test_selection_ignores_unscored_sentences():
    sents = ["a b c d", "e f g h", "i j k l"]
    sel, _ = select_sentences(sents, [0.1, None, float("nan")], k=3)
    assert sel == [0]


def test_textrank_fallback_shapes():
    assert textrank_scores([]) == []
    s = textrank_scores(["the cat sat", "the cat ran", "dogs bark loudly"])
    assert len(s) == 3 and all(math.isfinite(x) for x in s)


# ------------------------------------------------------------------ n-gram metrics
def test_novel_ngrams():
    src = "the cat sat on the mat"
    assert novel_ngram_pct("the cat sat", src, 1) == 0.0
    assert novel_ngram_pct("the dog sat", src, 1) == pytest.approx(100 / 3)
    assert novel_ngram_pct("the dog sat", src, 2) == 100.0
    assert novel_ngram_pct("", src, 2) == 0.0


def test_repeated_ngrams():
    # bigrams: (a b) (b a) (a b) -> 1 of 3 repeats
    assert repeated_ngram_pct("a b a b", 2) == pytest.approx(100 / 3)
    assert repeated_ngram_pct("a b c d", 2) == 0.0
    # trigrams of "x y z x y z": xyz yzx zxy xyz -> 1 of 4
    assert repeated_ngram_pct("x y z x y z", 3) == pytest.approx(25.0)


def test_compression_ratio():
    assert compression_ratio("one two three four", "one two") == 2.0
    assert compression_ratio("one two", "") == 0.0


def test_rouge_identity():
    r = rouge_scores(["The cat sat on the mat."], ["The cat sat on the mat."])
    assert r["rouge1"] == pytest.approx(100.0) and r["rougeL"] == pytest.approx(100.0)


# ------------------------------------------------------------------ oracle
def test_oracle_picks_matching_sentences():
    sents = ["Stocks fell sharply in Tokyo trading.",
             "The mayor opened a new bridge over the river.",
             "Rain is expected later this week.",
             "The bridge cost 20 million dollars to build."]
    ref = "The mayor opened a new bridge.\nThe bridge cost 20 million dollars."
    assert greedy_oracle(sents, ref) == [1, 3]
    assert oracle_labels(sents, ref) == [0, 1, 0, 1]


def test_oracle_max_three_and_empty():
    sents = [f"word{i} shared token" for i in range(6)]
    ref = " ".join(f"word{i}" for i in range(6)) + " shared token"
    assert len(greedy_oracle(sents, ref)) <= 3
    assert greedy_oracle([], "anything") == []


# ------------------------------------------------------------------ factual check
def test_factcheck_flags_unsupported_number_and_name():
    src = "The company reported profits of 5 million dollars on Tuesday, chief executive Jane Smith said."
    ok = check_summary("Jane Smith said profits were 5 million dollars.", src)
    assert ok["flags"] == []
    bad = check_summary("John Brown said profits were 7 million dollars.", src)
    texts = " ".join(f["text"] for f in bad["flags"])
    assert "7" in texts and "Brown" in texts
