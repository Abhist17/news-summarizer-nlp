"""Load Hugging Face models from the local cache first, so the demo works offline."""
from __future__ import annotations


def from_pretrained(cls, name: str, **kwargs):
    try:
        return cls.from_pretrained(name, local_files_only=True, **kwargs)
    except Exception:
        return cls.from_pretrained(name, **kwargs)
