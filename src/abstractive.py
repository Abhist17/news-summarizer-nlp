"""Abstractive summarizer: pretrained BART fine-tuned on CNN/DM (facebook/bart-large-cnn).

Long-document handling: BART reads at most 1024 tokens. If an article is
longer, it is split on sentence boundaries into ~900-token chunks, each chunk
is summarized, and the concatenated chunk summaries are summarized again
("summarize-then-summarize"). The result records whether this happened.
"""
from __future__ import annotations

import time

import torch

from config import (ABS_MODEL, BART_MAX_INPUT, CHUNK_GEN_KWARGS, CHUNK_TOKENS, GEN_KWARGS,
                    get_device)
from src.data import clean_article, split_sentences


class AbstractiveSummarizer:
    def __init__(self, model_name: str = ABS_MODEL, device: str | None = None):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        from src.hub import from_pretrained

        self.model_name = model_name
        self.device = device or get_device()
        self.tok = from_pretrained(AutoTokenizer, model_name)
        self.model = from_pretrained(AutoModelForSeq2SeqLM, model_name).to(self.device).eval()

    # ------------------------------------------------------------------ helpers
    def n_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=True, truncation=False)["input_ids"])

    @torch.no_grad()
    def generate(self, texts: list[str], **gen) -> list[str]:
        kwargs = {**GEN_KWARGS, **gen}
        enc = self.tok(texts, max_length=BART_MAX_INPUT, truncation=True, padding=True,
                       return_tensors="pt").to(self.device)
        out = self.model.generate(**enc, **kwargs)
        return [s.strip() for s in self.tok.batch_decode(out, skip_special_tokens=True)]

    def chunk(self, text: str, max_tokens: int = CHUNK_TOKENS) -> list[str]:
        """Group sentences into chunks of at most `max_tokens` BART tokens."""
        chunks, cur, cur_len = [], [], 0
        for s in split_sentences(text, min_words=0):
            n = len(self.tok(s, add_special_tokens=False)["input_ids"]) + 1
            if n > max_tokens:  # pathological single sentence: hard-split by tokens
                ids = self.tok(s, add_special_tokens=False)["input_ids"]
                pieces = [self.tok.decode(ids[i:i + max_tokens]) for i in range(0, len(ids), max_tokens)]
            else:
                pieces = [s]
            for p in pieces:
                pn = len(self.tok(p, add_special_tokens=False)["input_ids"]) + 1
                if cur and cur_len + pn > max_tokens:
                    chunks.append(" ".join(cur))
                    cur, cur_len = [], 0
                cur.append(p)
                cur_len += pn
        if cur:
            chunks.append(" ".join(cur))
        return chunks

    # ------------------------------------------------------------------ public API
    def summarize(self, article: str, long_mode: str = "chunk") -> dict:
        """Summarize one article. long_mode: 'chunk' (default) or 'truncate'."""
        t0 = time.perf_counter()
        text = clean_article(article)
        n_tok = self.n_tokens(text) if text else 0
        result = {"chunked": False, "n_chunks": 1, "chunk_summaries": [], "n_input_tokens": n_tok,
                  "model": self.model_name}
        if not text:
            result.update(summary="", latency=0.0)
            return result
        if n_tok <= BART_MAX_INPUT or long_mode == "truncate":
            summary = self.generate([text])[0]
        else:
            chunks = self.chunk(text)
            chunk_sums = []
            for i in range(0, len(chunks), 4):
                chunk_sums += self.generate(chunks[i:i + 4], **CHUNK_GEN_KWARGS)
            joined = " ".join(chunk_sums)
            # summaries of summaries; recurse if still too long (very long docs)
            while self.n_tokens(joined) > BART_MAX_INPUT:
                sub = self.chunk(joined)
                joined = " ".join(self.generate(sub, **CHUNK_GEN_KWARGS))
            summary = self.generate([joined])[0]
            result.update(chunked=True, n_chunks=len(chunks), chunk_summaries=chunk_sums)
        result.update(summary=summary, latency=time.perf_counter() - t0)
        return result

    def summarize_batch(self, articles: list[str], batch_size: int = 4) -> list[str]:
        """Batched, truncate-at-1024 summarization (standard evaluation protocol)."""
        texts = [clean_article(a) for a in articles]
        outs: list[str] = []
        for i in range(0, len(texts), batch_size):
            outs += self.generate(texts[i:i + batch_size])
        return outs
