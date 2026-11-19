"""CNN/DailyMail (and XSum) subset loading + sentence splitting.

Subsets are streamed from the Hugging Face Hub (so we never download the full
dataset) and cached as JSONL under data_cache/ so later runs work offline.
"""
from __future__ import annotations

import json
import logging
import re
from itertools import islice
from pathlib import Path

from config import DATA_CACHE, DATASET_CONFIG, DATASET_NAME, MIN_SENT_WORDS, XSUM_NAME

log = logging.getLogger(__name__)

WORD_RE = re.compile(r"[A-Za-z0-9]+")

# --------------------------------------------------------------------------- sentence splitting
_PUNKT_OK: bool | None = None
_FALLBACK_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def _punkt_available() -> bool:
    global _PUNKT_OK
    if _PUNKT_OK is None:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt_tab/english/")
            _PUNKT_OK = True
        except LookupError:
            try:
                nltk.download("punkt_tab", quiet=True)
                nltk.data.find("tokenizers/punkt_tab/english/")
                _PUNKT_OK = True
            except Exception:  # offline and not cached: regex fallback
                log.warning("nltk punkt_tab unavailable; using regex sentence splitter")
                _PUNKT_OK = False
    return _PUNKT_OK


# Daily Mail articles start with a byline/timestamp header like
# "By . Jane Doe . PUBLISHED: . 10:01 EST, 3 May 2013 . | . UPDATED: . 12:00 EST, 3 May 2013 . "
_DM_HEADER = [
    re.compile(r"^.{0,300}?UPDATED:\s*\.\s*\d{1,2}:\d{2}\s*(?:EST|GMT|BST)?,?\s*\d{1,2}\s+\w+\s+\d{4}\s*\.\s*", re.S),
    re.compile(r"^.{0,300}?PUBLISHED:\s*\.\s*\d{1,2}:\d{2}\s*(?:EST|GMT|BST)?,?\s*\d{1,2}\s+\w+\s+\d{4}\s*\.\s*", re.S),
    re.compile(r"^.{0,200}?Last updated at \d{1,2}:\d{2}\s*[AP]M on \d{1,2}\w*\s+\w+\s+\d{4}\s*\.\s*", re.S),
    re.compile(r"^By \.\s*[^.]{1,80}?\s*\.\s*(?:and [^.]{1,80}?\s*\.\s*)?"),
]


def clean_article(text: str) -> str:
    """Normalise whitespace and strip the Daily Mail byline/timestamp header."""
    text = re.sub(r"\s+", " ", (text or "")).strip()
    for pat in _DM_HEADER:
        text = pat.sub("", text, count=1)
    return text.strip()


def split_sentences(text: str, min_words: int = MIN_SENT_WORDS) -> list[str]:
    """Split an article into sentences (nltk punkt, regex fallback).

    Sentences with fewer than `min_words` word tokens (e.g. "By ." debris) are
    dropped; if that would drop everything, the unfiltered list is returned.
    """
    text = clean_article(text)
    if not text:
        return []
    if _punkt_available():
        from nltk.tokenize import sent_tokenize

        sents = sent_tokenize(text)
    else:
        sents = _FALLBACK_SPLIT.split(text)
    sents = [s.strip() for s in sents if s and s.strip()]
    kept = [s for s in sents if len(WORD_RE.findall(s)) >= min_words]
    return kept if kept else sents


def highlights_to_sents(highlights: str) -> list[str]:
    return [h.strip() for h in (highlights or "").split("\n") if h.strip()]


# --------------------------------------------------------------------------- dataset subsets
SPLIT_SIZES = {"train": 287113, "validation": 13368, "test": 11490}


def _cache_path(prefix: str, split: str, n: int, tag: str = "spread") -> Path:
    return DATA_CACHE / f"{prefix}_{split}_{tag}_{n}.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _row(ex) -> dict:
    return {"id": ex["id"], "article": ex["article"], "highlights": ex["highlights"]}


def load_cnndm(split: str = "train", n: int = 100) -> list[dict]:
    """Return a representative subset of `n` CNN/DM examples as {id, article, highlights}.

    CNN/DM files list all CNN stories before the Daily Mail ones, so taking the first
    n rows would give a CNN-only sample. Instead:
      * train: n/3 rows from the start of each of the 3 parquet shards (shard 0 = CNN,
        shards 1-2 = Daily Mail), matching the ~1:2 CNN:DM ratio of the full split,
        while streaming only a few MB;
      * validation / test: every k-th article over the whole split (k = size // n),
        i.e. a deterministic, evenly spread sample (streams the ~50 MB split once).
    Results are cached in data_cache/ so later runs are offline and identical.
    """
    path = _cache_path("cnndm", split, n)
    if path.exists():
        return _read_jsonl(path)

    from datasets import load_dataset

    log.info("Streaming %d %s examples from %s/%s ...", n, split, DATASET_NAME, DATASET_CONFIG)
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split, streaming=True)
    if split == "train" and ds.n_shards > 1:
        per = -(-n // ds.n_shards)
        rows = []
        for i in range(ds.n_shards):
            rows += [_row(ex) for ex in islice(ds.shard(ds.n_shards, i), per)]
        rows = rows[:n]
    else:
        stride = max(1, SPLIT_SIZES.get(split, n) // n)
        rows = [_row(ex) for ex in islice(ds, 0, stride * n, stride)]
    _write_jsonl(path, rows)
    return rows


def load_xsum(split: str = "test", n: int = 50) -> list[dict]:
    """Return `n` XSum examples as {id, article, highlights}. Tries the parquet branch first."""
    path = _cache_path("xsum", split, n, "head")
    if path.exists():
        return _read_jsonl(path)
    from datasets import load_dataset

    errors = []
    for kwargs in ({"revision": "refs/convert/parquet"}, {}, {"trust_remote_code": True}):
        try:
            ds = load_dataset(XSUM_NAME, split=split, streaming=True, **kwargs)
            rows = [{"id": str(ex["id"]), "article": ex["document"], "highlights": ex["summary"]}
                    for ex in islice(ds, n)]
            if rows:
                _write_jsonl(path, rows)
                return rows
        except Exception as e:  # pragma: no cover - network dependent
            errors.append(f"{kwargs}: {e}")
    raise RuntimeError("Could not load XSum:\n" + "\n".join(errors))
