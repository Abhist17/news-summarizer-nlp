"""Lightweight factual-consistency flags for abstractive summaries (a heuristic).

Flags numbers, dates and named-entity-like spans that appear in the summary
but cannot be found in the source article. Uses spaCy `en_core_web_sm` NER when
installed, otherwise a regex heuristic (numbers + capitalised word runs).

This is NOT a faithfulness model: it catches only surface-level "new facts"
(hallucinated names/numbers) and will miss paraphrased errors (wrong relations
between correct entities). It can also flag legitimate rewrites, e.g. "3" for "three".
"""
from __future__ import annotations

import html
import re

_WORD = re.compile(r"[a-z0-9]+")
_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
_CAPRUN = re.compile(r"\b[A-Z][a-zA-Z'’\-]+(?:\s+(?:of\s+|de\s+|the\s+)?[A-Z][a-zA-Z'’\-]+)*")

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
    "hundred": "100", "thousand": "1000", "million": "1000000", "billion": "1000000000",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "dozen": "12",
}
STOP = {"the", "a", "an", "of", "and", "in", "on", "at", "to", "for", "s", "mr", "mrs", "ms",
        "dr", "he", "she", "it", "they", "his", "her", "their", "this", "that", "new", "de"}
SENT_INITIAL = {"The", "A", "An", "He", "She", "It", "They", "His", "Her", "Their", "This", "That",
                "These", "Those", "But", "And", "In", "On", "At", "After", "Before", "When", "While",
                "There", "We", "I", "If", "As", "Its", "One", "Some", "Many", "Police", "Officials"}
ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "EVENT", "PRODUCT", "WORK_OF_ART", "LAW"}
NUMERIC_LABELS = {"DATE", "TIME", "CARDINAL", "MONEY", "PERCENT", "QUANTITY", "ORDINAL"}

_NLP = None
_NLP_TRIED = False


def get_nlp():
    """spaCy pipeline or None (regex fallback)."""
    global _NLP, _NLP_TRIED
    if not _NLP_TRIED:
        _NLP_TRIED = True
        try:
            import spacy

            _NLP = spacy.load("en_core_web_sm", disable=["lemmatizer"])
        except Exception:
            _NLP = None
    return _NLP


def backend_name() -> str:
    return "spaCy en_core_web_sm NER" if get_nlp() is not None else "regex heuristic"


def _norm_num(s: str) -> str:
    s = s.replace(",", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


class _Source:
    def __init__(self, text: str):
        self.lower = re.sub(r"\s+", " ", text.lower())
        self.words = set(_WORD.findall(self.lower))
        self.numbers = {_norm_num(m) for m in _NUM.findall(text)}
        self.numbers |= {v for w, v in NUMBER_WORDS.items() if w in self.words}

    def has_number(self, tok: str) -> bool:
        t = _norm_num(tok.lower())
        return t in self.numbers or (t in NUMBER_WORDS and NUMBER_WORDS[t] in self.numbers) \
            or (t in NUMBER_WORDS and t in self.words)

    def has_entity(self, ent: str) -> bool:
        e = re.sub(r"\s+", " ", ent.lower()).strip(" .,'’")
        e = re.sub(r"['’]s$", "", e)
        if not e or e in self.lower:
            return True
        content = [w for w in _WORD.findall(e) if w not in STOP]
        return all(w in self.words for w in content)


def _numeric_span_ok(span: str, src: _Source) -> bool:
    nums = _NUM.findall(span)
    words = _WORD.findall(span.lower())
    for n in nums:
        if not src.has_number(n):
            return False
    for w in words:
        if w in NUMBER_WORDS and not src.has_number(w):
            return False
        if not w.isdigit() and w not in NUMBER_WORDS and w not in STOP and len(w) > 2 \
                and w not in src.words and w[:3] in {"jan", "feb", "mar", "apr", "may", "jun", "jul",
                                                   "aug", "sep", "oct", "nov", "dec", "mon", "tue",
                                                   "wed", "thu", "fri", "sat", "sun"}:
            return False
    return True


def check_summary(summary: str, source: str) -> dict:
    """Return {'flags': [{text,start,end,type}], 'backend': str}."""
    src = _Source(source or "")
    flags: list[dict] = []
    nlp = get_nlp()
    if summary:
        if nlp is not None:
            doc = nlp(summary)
            for ent in doc.ents:
                if ent.label_ in ENTITY_LABELS and not src.has_entity(ent.text):
                    flags.append(dict(text=ent.text, start=ent.start_char, end=ent.end_char,
                                      type=ent.label_))
                elif ent.label_ in NUMERIC_LABELS and not _numeric_span_ok(ent.text, src):
                    flags.append(dict(text=ent.text, start=ent.start_char, end=ent.end_char,
                                      type=ent.label_))
        else:
            for m in _CAPRUN.finditer(summary):
                text = m.group(0)
                words = text.split()
                if words and words[0] in SENT_INITIAL:
                    words = words[1:]
                    if not words:
                        continue
                    text = " ".join(words)
                    start = m.end() - len(text)
                else:
                    start = m.start()
                if not src.has_entity(text):
                    flags.append(dict(text=text, start=start, end=start + len(text), type="NAME?"))
        # numbers spaCy may have missed (and the regex path)
        for m in _NUM.finditer(summary):
            if not src.has_number(m.group(0)) and not any(f["start"] <= m.start() < f["end"] for f in flags):
                flags.append(dict(text=m.group(0), start=m.start(), end=m.end(), type="NUMBER"))
    flags.sort(key=lambda f: f["start"])
    dedup: list[dict] = []
    for f in flags:
        if dedup and f["start"] < dedup[-1]["end"]:
            continue
        dedup.append(f)
    return {"flags": dedup, "backend": "spaCy" if nlp is not None else "regex"}


def highlight_html(summary: str, flags: list[dict]) -> str:
    """Summary as HTML with flagged spans wrapped in <mark class="flag">."""
    out, pos = [], 0
    for f in flags:
        out.append(html.escape(summary[pos:f["start"]]))
        out.append(f'<mark class="flag" title="Possibly unsupported ({html.escape(f["type"])}): '
                   f'not found in the source article">&#9888; {html.escape(summary[f["start"]:f["end"]])}</mark>')
        pos = f["end"]
    out.append(html.escape(summary[pos:]))
    return "".join(out)
