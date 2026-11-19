"""BertSumExt-style extractive summarizer (Liu & Lapata, 2019), lightweight variant.

Document encoding (as in BertSum):
    [CLS] sent_1 [SEP] [CLS] sent_2 [SEP] ...   truncated to 512 word pieces
with alternating segment ids (0,1,0,1,...) per sentence. Each sentence is
represented by the vector of its own [CLS] token.

Deviation for CPU speed: the BERT encoder is *frozen* (embeddings are cached
once), and only the inter-sentence head is trained:
    Linear(768->d) + sinusoidal sentence positions -> 2 Transformer encoder
    layers -> Linear -> sigmoid
Inference: rank sentences by score, take the top 3 with trigram blocking,
and output them in original article order.

If no trained checkpoint is available, a TF-IDF TextRank scorer is used instead
so the app never crashes (the UI labels it clearly).
"""
from __future__ import annotations

import math
import re
import time

import numpy as np
import torch
import torch.nn as nn

from config import (ENCODER_MAX_TOKENS, ENCODER_NAME, EXT_CHECKPOINT, EXT_POOLING, EXT_TOP_K,
                    MAX_SENT_WORDPIECES, get_device)
from src.data import split_sentences

_TOK = re.compile(r"[a-z0-9]+")

METHOD_BERTSUM = "BertSumExt-style (frozen BERT + 2-layer Transformer head)"
METHOD_TEXTRANK = "Fallback: unsupervised TF-IDF TextRank (no trained checkpoint found)"


# =========================================================================== trigram blocking
def get_trigrams(sentence: str) -> set[tuple]:
    toks = _TOK.findall(sentence.lower())
    return set(zip(toks, toks[1:], toks[2:]))


def has_trigram_overlap(candidate: str, selected: list[str]) -> bool:
    tri_c = get_trigrams(candidate)
    return any(tri_c & get_trigrams(s) for s in selected)


def select_sentences(sentences: list[str], scores, k: int = EXT_TOP_K,
                     trigram_blocking: bool = True) -> tuple[list[int], list[int]]:
    """Pick up to k sentences by descending score, skipping trigram-overlapping ones.

    Sentences whose score is None / NaN / -inf (e.g. outside the 512-token window)
    are never selected. Returns (selected indices in article order, blocked indices).
    """
    valid = [(i, float(s)) for i, s in enumerate(scores)
             if s is not None and math.isfinite(float(s))]
    order = [i for i, _ in sorted(valid, key=lambda x: -x[1])]
    selected, blocked = [], []
    for i in order:
        if len(selected) >= k:
            break
        if trigram_blocking and has_trigram_overlap(sentences[i], [sentences[j] for j in selected]):
            blocked.append(i)
            continue
        selected.append(i)
    return sorted(selected), blocked


# =========================================================================== model pieces
class SentenceScorerHead(nn.Module):
    """Inter-sentence Transformer (BertSumExt head) over frozen [CLS] vectors."""

    def __init__(self, in_dim: int = 768, d_model: int = 256, n_layers: int = 2,
                 n_heads: int = 4, ff_dim: int = 1024, dropout: float = 0.1, max_sents: int = 128):
        super().__init__()
        self.config = dict(in_dim=in_dim, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                           ff_dim=ff_dim, dropout=dropout, max_sents=max_sents)
        self.proj = nn.Linear(in_dim, d_model)
        pe = torch.zeros(max_sents, d_model)
        pos = torch.arange(max_sents).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: [B, S, in_dim], mask: [B, S] (True = real sentence). Returns logits [B, S]."""
        h = self.proj(x) + self.pe[: x.size(1)].unsqueeze(0)
        h = self.encoder(h, src_key_padding_mask=~mask)
        return self.out(self.norm(h)).squeeze(-1)


class BertSumEncoder:
    """Frozen BERT that returns one [CLS] vector per sentence."""

    def __init__(self, name: str = ENCODER_NAME, device: str | None = None, pooling: str = EXT_POOLING):
        from transformers import AutoModel, AutoTokenizer

        from src.hub import from_pretrained

        self.name = name
        self.pooling = pooling  # "cls" (BertSum) or "cls+mean" ([CLS] ++ mean of the sentence's tokens)
        self.device = device or get_device()
        self.tok = from_pretrained(AutoTokenizer, name)
        self.model = from_pretrained(AutoModel, name).to(self.device).eval()
        self.use_segments = ("token_type_ids" in self.tok.model_input_names
                             and getattr(self.model.config, "type_vocab_size", 1) >= 2)
        self.hidden = self.model.config.hidden_size
        self.out_dim = self.hidden * (2 if pooling == "cls+mean" else 1)

    def build_input(self, sentences: list[str]) -> tuple[list[int], list[int], list[int]]:
        """Return (input_ids, segment_ids, cls_positions) for one document."""
        cls_id, sep_id = self.tok.cls_token_id, self.tok.sep_token_id
        ids, segs, cls_pos = [], [], []
        pieces = self.tok(sentences, add_special_tokens=False)["input_ids"] if sentences else []
        for i, wp in enumerate(pieces):
            if len(ids) + 3 > ENCODER_MAX_TOKENS:   # need room for [CLS] tok [SEP]
                break
            piece = [cls_id] + wp[:MAX_SENT_WORDPIECES] + [sep_id]
            cls_pos.append(len(ids))
            ids.extend(piece)
            segs.extend([i % 2] * len(piece))
        if len(ids) > ENCODER_MAX_TOKENS:
            ids, segs = ids[:ENCODER_MAX_TOKENS], segs[:ENCODER_MAX_TOKENS]
            ids[-1] = sep_id
        return ids, segs, cls_pos

    @torch.no_grad()
    def encode(self, docs: list[list[str]], batch_size: int = 8) -> list[torch.Tensor]:
        """For each document (list of sentences) return a [n_in_window, hidden] float tensor."""
        built = [self.build_input(d) for d in docs]
        out: list[torch.Tensor | None] = [None] * len(docs)
        order = sorted(range(len(docs)), key=lambda i: len(built[i][0]))
        pad = self.tok.pad_token_id
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            maxlen = max(1, max(len(built[i][0]) for i in idx))
            ids = torch.full((len(idx), maxlen), pad, dtype=torch.long)
            seg = torch.zeros((len(idx), maxlen), dtype=torch.long)
            att = torch.zeros((len(idx), maxlen), dtype=torch.long)
            for r, i in enumerate(idx):
                L = len(built[i][0])
                ids[r, :L] = torch.tensor(built[i][0])
                seg[r, :L] = torch.tensor(built[i][1])
                att[r, :L] = 1
            kwargs = dict(input_ids=ids.to(self.device), attention_mask=att.to(self.device))
            if self.use_segments:
                kwargs["token_type_ids"] = seg.to(self.device)
            hidden = self.model(**kwargs).last_hidden_state.float().cpu()
            for r, i in enumerate(idx):
                pos = built[i][2]
                if not pos:
                    out[i] = torch.zeros((0, self.out_dim))
                    continue
                cls = hidden[r, pos]
                if self.pooling == "cls+mean":
                    L = len(built[i][0])
                    ends = pos[1:] + [L]
                    # mean over the sentence's word pieces (between its [CLS] and [SEP])
                    means = torch.stack([hidden[r, a + 1:max(a + 2, b - 1)].mean(0) for a, b in zip(pos, ends)])
                    cls = torch.cat([cls, means], dim=-1)
                out[i] = cls
        return out  # type: ignore[return-value]


# =========================================================================== TextRank fallback
def textrank_scores(sentences: list[str], damping: float = 0.85, iters: int = 50) -> list[float]:
    n = len(sentences)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    toks = [_TOK.findall(s.lower()) for s in sentences]
    vocab = {w: j for j, w in enumerate(sorted({w for t in toks for w in t}))}
    tf = np.zeros((n, max(1, len(vocab))))
    for i, t in enumerate(toks):
        for w in t:
            tf[i, vocab[w]] += 1
    df = (tf > 0).sum(0)
    tfidf = tf * np.log((1 + n) / (1 + df) + 1)
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1
    v = tfidf / norms
    sim = v @ v.T
    np.fill_diagonal(sim, 0)
    row = sim.sum(1, keepdims=True)
    row[row == 0] = 1
    P = sim / row
    r = np.ones(n) / n
    for _ in range(iters):
        r = (1 - damping) / n + damping * P.T @ r
    return r.tolist()


# =========================================================================== summarizer
class ExtractiveSummarizer:
    def __init__(self, checkpoint=EXT_CHECKPOINT, device: str | None = None):
        self.device = device or get_device()
        self.method = METHOD_TEXTRANK
        self.encoder = None
        self.head = None
        self.load_error = None
        try:
            if checkpoint is None or not checkpoint.exists():
                raise FileNotFoundError(f"no checkpoint at {checkpoint}")
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.head = SentenceScorerHead(**ckpt["head_config"])
            self.head.load_state_dict({k: v.float() for k, v in ckpt["state_dict"].items()})
            self.head.to(self.device).eval()
            self.encoder = BertSumEncoder(ckpt["encoder_name"], device=self.device,
                                          pooling=ckpt.get("pooling", "cls"))
            self.method = METHOD_BERTSUM
            self.encoder_name = ckpt["encoder_name"]
        except Exception as e:  # missing checkpoint, offline without cached encoder, ...
            self.load_error = str(e)
            self.head = self.encoder = None
            self.method = METHOD_TEXTRANK

    @property
    def is_trained(self) -> bool:
        return self.head is not None

    @torch.no_grad()
    def score(self, sentences: list[str]) -> list[float | None]:
        """Per-sentence probability; None for sentences outside the 512-token window."""
        if not sentences:
            return []
        if not self.is_trained:
            return textrank_scores(sentences)
        vecs = self.encoder.encode([sentences])[0]
        if vecs.shape[0] == 0:
            return [None] * len(sentences)
        x = vecs.unsqueeze(0).to(self.device)
        mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=self.device)
        probs = torch.sigmoid(self.head(x, mask))[0].cpu().tolist()
        return probs + [None] * (len(sentences) - len(probs))

    def summarize(self, article: str, k: int = EXT_TOP_K, sentences: list[str] | None = None,
                  trigram_blocking: bool = True) -> dict:
        t0 = time.perf_counter()
        sents = sentences if sentences is not None else split_sentences(article)
        scores = self.score(sents)
        selected, blocked = select_sentences(sents, scores, k, trigram_blocking)
        chosen = [sents[i] for i in selected]
        return {
            "summary": " ".join(chosen),
            "summary_sents": chosen,
            "indices": selected,
            "blocked": blocked,
            "scores": scores,
            "method": self.method,
            "trained": self.is_trained,
            "latency": time.perf_counter() - t0,
        }
