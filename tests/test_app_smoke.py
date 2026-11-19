"""Smoke tests: the app module imports, helpers work, and the Streamlit script runs headlessly."""
import json

import pytest

import app
from config import DEMO_CACHE


def test_app_imports_and_helpers():
    assert callable(app.main)
    ext = {"indices": [1], "blocked": [2], "scores": [0.1, 0.9, 0.8, None]}
    html = app.article_html(["s one", "s two", "s three", "s four"], ext, [0, 1, 2])
    assert "s-sel" in html and "s-blk" in html and "s-out" in html
    df = app.sentence_table(["a", "b", "c", "d"], ext, [0, 1, 2])
    assert list(df["status"]) == ["", "selected", "trigram-blocked", "outside 512 tokens"]
    card = app.metrics_card_html({"words": 10, "compression": 5.0, "novel_2": 1.0, "novel_3": 2.0,
                                  "repeated_3": 0.0, "latency": 0.5}, {"rouge1": 40, "rouge2": 18, "rougeL": 36}, True)
    assert "ROUGE-1" in card


@pytest.mark.skipif(not (DEMO_CACHE / "samples.json").exists(), reason="run `make demo-cache` first")
def test_demo_cache_schema():
    data = json.loads((DEMO_CACHE / "samples.json").read_text(encoding="utf-8"))
    assert len(data["samples"]) >= 8
    assert any(next(iter(s["abstractive"].values()))["chunked"] for s in data["samples"])
    for s in data["samples"]:
        for k in ("lead3", "extractive", "abstractive", "sentences", "article", "reference"):
            assert k in s


@pytest.mark.skipif(not (DEMO_CACHE / "samples.json").exists(), reason="run `make demo-cache` first")
def test_streamlit_app_runs_headless():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 3
    # switch sample and model -> still no exception
    at.sidebar.selectbox[0].set_value(len(at.sidebar.selectbox[0].options) - 1).run()
    at.sidebar.radio[1].set_value(at.sidebar.radio[1].options[1]).run()
    assert not at.exception, at.exception
