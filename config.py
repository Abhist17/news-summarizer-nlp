"""Central configuration: paths, model names, generation settings, published numbers."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_CACHE = ROOT / "data_cache"        # streamed dataset subsets + cached [CLS] embeddings (gitignored)
CHECKPOINTS = ROOT / "checkpoints"      # trained extractive head
RESULTS = ROOT / "results"              # evaluation outputs (committed)
DEMO_CACHE = ROOT / "demo_cache"        # precomputed demo samples (committed)

for _d in (DATA_CACHE, CHECKPOINTS, RESULTS, DEMO_CACHE):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data
DATASET_NAME = "abisee/cnn_dailymail"
DATASET_CONFIG = "3.0.0"
XSUM_NAME = "EdinburghNLP/xsum"
MIN_SENT_WORDS = 3          # sentences with fewer word tokens are dropped (BertSum drops < 5 tokens)

# ---------------------------------------------------------------- extractive
ENCODER_NAME = os.environ.get("SUMM_ENCODER", "bert-base-uncased")
ENCODER_MAX_TOKENS = 512    # BertSum truncates the document to 512 word pieces
MAX_SENT_WORDPIECES = 200   # cap any single sentence so it can't eat the whole window
EXT_CHECKPOINT = CHECKPOINTS / "ext_head.pt"
EXT_POOLING = "cls+mean"    # sentence vector = [CLS] ++ mean of its word pieces ("cls" = pure BertSum)
EXT_TOP_K = 3

# ---------------------------------------------------------------- abstractive
ABS_MODEL = "facebook/bart-large-cnn"
ABS_MODEL_FAST = "sshleifer/distilbart-cnn-12-6"
ABS_MODELS = {"BART-large-CNN": ABS_MODEL, "DistilBART-CNN (fast)": ABS_MODEL_FAST}
BART_MAX_INPUT = 1024
CHUNK_TOKENS = 900          # target chunk size when an article exceeds BART_MAX_INPUT

# Same settings as the bart-large-cnn model card / CNN-DM fine-tuning config.
GEN_KWARGS = dict(num_beams=4, no_repeat_ngram_size=3, length_penalty=2.0,
                  min_length=56, max_length=142, early_stopping=True)
# Shorter outputs for per-chunk summaries (stage 1 of long-document handling).
CHUNK_GEN_KWARGS = dict(num_beams=4, no_repeat_ngram_size=3, length_penalty=2.0,
                        min_length=30, max_length=90, early_stopping=True)

# ---------------------------------------------------------------- published ROUGE-1 (CNN/DM full test set)
PUBLISHED_ROUGE1 = {
    "Seq2seq + attn (Nallapati 2016)": 31.33,
    "Pointer-gen (See 2017)": 36.44,
    "Pointer-gen + coverage (See 2017)": 39.53,
    "Lead-3": 40.42,
    "TransformerExt (Liu & Lapata 2019)": 40.90,
    "BertSumAbs (Liu & Lapata 2019)": 41.72,
    "BertSumExtAbs (Liu & Lapata 2019)": 42.13,
    "BertSumExt (Liu & Lapata 2019)": 43.25,
}

AUTHORS = "Vedansh Shukla & Abhist Kamle"
TITLE = "Neural Text Summarization: News-Article Summarizer"


def get_device(prefer: str | None = None) -> str:
    """cuda -> mps -> cpu. Override with SUMM_DEVICE=cpu (or the `prefer` argument)."""
    import torch

    choice = (prefer or os.environ.get("SUMM_DEVICE") or "").lower()
    if choice in {"cpu", "cuda", "mps"}:
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
