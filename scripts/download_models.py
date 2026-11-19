"""Pre-download every model / resource the demo needs so nothing downloads live.

Usage:  python scripts/download_models.py [--skip-fast]
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ABS_MODEL, ABS_MODEL_FAST, ENCODER_NAME  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fast", action="store_true")
    args = ap.parse_args()

    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
        print("nltk punkt_tab: ok")
    except LookupError:
        for _ in range(3):  # retry: the nltk data host occasionally resets connections
            if nltk.download("punkt_tab", quiet=True):
                break
        print("nltk punkt_tab downloaded")

    try:
        import spacy

        spacy.load("en_core_web_sm")
        print("spaCy en_core_web_sm: ok")
    except Exception:
        print("spaCy en_core_web_sm missing -> downloading (the fact check falls back to regex if this fails)")
        subprocess.call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

    from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

    models = [(ENCODER_NAME, AutoModel), (ABS_MODEL, AutoModelForSeq2SeqLM)]
    if not args.skip_fast:
        models.append((ABS_MODEL_FAST, AutoModelForSeq2SeqLM))
    for name, cls in models:
        print(f"Downloading {name} ...", flush=True)
        AutoTokenizer.from_pretrained(name)
        cls.from_pretrained(name)
    print("All models cached.")


if __name__ == "__main__":
    main()
