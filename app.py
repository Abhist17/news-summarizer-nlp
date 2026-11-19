"""Streamlit demo: Lead-3 vs Extractive (BertSumExt-style) vs Abstractive (BART).

Run:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import html
import json

import pandas as pd
import streamlit as st

from config import ABS_MODELS, AUTHORS, BART_MAX_INPUT, DEMO_CACHE, RESULTS, TITLE

# =========================================================================== helpers (importable)
SAMPLES_PATH = DEMO_CACHE / "samples.json"

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1500px;}
.app-title {font-size: 2.05rem; font-weight: 700; margin: 0; line-height: 1.2;}
.app-sub {color: #52514e; font-size: 1.02rem; margin: .25rem 0 .2rem 0;}
.app-line {font-size: 1.02rem; margin: .35rem 0 .6rem 0;}
.pill {display:inline-block; padding: 1px 9px; border-radius: 999px; font-size: .78rem;
       font-weight: 600; margin-right: 6px; vertical-align: middle;}
.pill-base {background:#eceae4; color:#3d3c39;}
.pill-ext {background:#dbe9fb; color:#174f95;}
.pill-abs {background:#d8f1e6; color:#0e5c40;}
.col-head {font-size: 1.22rem; font-weight: 700; margin-bottom: 0;}
.col-cap {color:#52514e; font-size: .88rem; margin-bottom: .5rem;}
.summary-box {border: 1px solid #e4e3df; border-radius: 10px; padding: 14px 16px; background: #ffffff;
              font-size: 1.04rem; line-height: 1.55; min-height: 190px;}
.metric-grid {display:grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-top: 10px;}
.metric {background:#f6f5f2; border-radius: 8px; padding: 6px 8px;}
.metric .k {font-size: .72rem; color:#52514e; text-transform: uppercase; letter-spacing: .03em;}
.metric .v {font-size: 1.05rem; font-weight: 650; color:#0b0b0b;}
.rouge-row {display:flex; gap:6px; margin-top:6px;}
.rouge-row .metric {flex:1; background:#eef4fd;}
mark.flag {background:#fff1c2; border-bottom: 2px solid #c98500; padding: 0 2px; border-radius: 3px;}
.article-box {border: 1px solid #e4e3df; border-radius: 10px; padding: 14px 18px; background:#ffffff;
              line-height: 1.75; font-size: .98rem; max-height: 560px; overflow-y: auto;}
.s-sel {background:#dbe9fb; border-bottom: 2px solid #2a78d6; padding: 1px 2px; border-radius: 3px;}
.s-blk {background:#fde8d7; border-bottom: 2px dashed #d95926; padding: 1px 2px; border-radius: 3px;}
.s-out {color:#8a8984;}
.score {font-size: .7rem; font-weight: 700; color:#52514e; background:#eceae4; border-radius: 6px;
        padding: 0 5px; margin: 0 3px 0 2px; vertical-align: 1px; font-family: monospace;}
.s-sel .score {background:#2a78d6; color:#ffffff;}
.legend span {margin-right: 14px; font-size: .86rem;}
</style>
"""

COL_INFO = {
    "lead3": ("Lead-3 baseline", "pill-base", "baseline", "The first three sentences. Strong on news (lead bias)."),
    "extractive": ("Extractive", "pill-ext", "highlight", "BertSumExt-style scorer, top-3 with trigram blocking."),
    "abstractive": ("Abstractive", "pill-abs", "rewrite", "Pretrained BART encoder-decoder, beam search."),
}


def load_samples() -> dict | None:
    if not SAMPLES_PATH.exists():
        return None
    return json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))


def sample_label(i: int, s: dict) -> str:
    abs_any = next(iter(s["abstractive"].values()), {})
    tag = "  [long → chunked]" if abs_any.get("chunked") else ""
    return f"{i + 1}. {s['title'][:60]} ({s['n_article_words']} words){tag}"


def metrics_card_html(metrics: dict, rouge: dict | None, precomputed: bool) -> str:
    lat = metrics.get("latency", 0.0) or 0.0
    lat_s = f"{lat * 1000:.0f} ms" if lat < 1 else f"{lat:.1f} s"
    cells = [("Words", f"{metrics['words']}"),
             ("Compression", f"{metrics['compression']:.1f}×"),
             ("Novel bigrams", f"{metrics['novel_2']:.1f}%"),
             ("Novel trigrams", f"{metrics['novel_3']:.1f}%"),
             ("Repeated 3-grams", f"{metrics['repeated_3']:.1f}%"),
             ("Latency" + (" (cached)" if precomputed else ""), lat_s)]
    out = '<div class="metric-grid">' + "".join(
        f'<div class="metric"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in cells) + "</div>"
    if rouge:
        out += '<div class="rouge-row">' + "".join(
            f'<div class="metric"><div class="k">{k}</div><div class="v">{rouge[c]:.1f}</div></div>'
            for k, c in (("ROUGE-1", "rouge1"), ("ROUGE-2", "rouge2"), ("ROUGE-L", "rougeL"))) + "</div>"
    return out


def article_html(sentences: list[str], ext: dict, lead_idx: list[int]) -> str:
    sel, blk = set(ext["indices"]), set(ext.get("blocked", []))
    parts = []
    for i, s in enumerate(sentences):
        sc = ext["scores"][i] if i < len(ext["scores"]) else None
        score = f'<span class="score">{sc:.2f}</span>' if sc is not None else ""
        lead = "  ·L3" if i in lead_idx else ""
        title = (f"Sentence {i + 1}{lead} | model score: {sc:.3f}" if sc is not None
                 else f"Sentence {i + 1}{lead} | beyond the 512-token encoder window (not scored)")
        if i in sel:
            title += " | SELECTED"
            cls = "s-sel"
        elif i in blk:
            title += " | skipped by trigram blocking"
            cls = "s-blk"
        elif sc is None:
            cls = "s-out"
        else:
            cls = ""
        parts.append(f'<span class="{cls}" title="{html.escape(title)}">{score}{html.escape(s)}</span>')
    return '<div class="article-box">' + " ".join(parts) + "</div>"


def sentence_table(sentences: list[str], ext: dict, lead_idx: list[int]) -> pd.DataFrame:
    sel, blk = set(ext["indices"]), set(ext.get("blocked", []))
    rows = []
    for i, s in enumerate(sentences):
        sc = ext["scores"][i] if i < len(ext["scores"]) else None
        status = ("selected" if i in sel else "trigram-blocked" if i in blk
                  else "outside 512 tokens" if sc is None else "")
        rows.append({"#": i + 1, "score": None if sc is None else round(sc, 3), "status": status,
                     "lead-3": "✓" if i in lead_idx else "", "sentence": s})
    return pd.DataFrame(rows)


def live_result(text: str, model_name: str, reference: str | None, extractive, abstractive) -> dict:
    """Run the whole pipeline on pasted text (used by the app and by smoke tests)."""
    from src.pipeline import run_article

    return run_article(text, reference or None, extractive, {model_name: abstractive})


# =========================================================================== cached resources
@st.cache_resource(show_spinner="Loading extractive model ...")
def get_extractive():
    from src.extractive import ExtractiveSummarizer

    return ExtractiveSummarizer()


@st.cache_resource(show_spinner="Loading abstractive model (first time takes ~10-20 s) ...")
def get_abstractive(model_id: str):
    from src.abstractive import AbstractiveSummarizer

    return AbstractiveSummarizer(model_id)


@st.cache_data
def cached_samples():
    return load_samples()


# =========================================================================== UI
def render_header():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f'<p class="app-title">📰 {TITLE}</p>'
                f'<p class="app-sub">{AUTHORS} · NLP Lab</p>'
                '<p class="app-line"><span class="pill pill-ext">Extractive</span> <b>highlights</b>: '
                'copies the most important sentences verbatim. '
                '<span class="pill pill-abs">Abstractive</span> <b>rewrites</b>: generates new sentences '
                'that may use words not in the article.</p>', unsafe_allow_html=True)


def render_sidebar(samples: dict | None):
    st.sidebar.header("Input")
    options = ["Precomputed CNN/DM sample (instant)", "Paste an article (live)"]
    mode = st.sidebar.radio("Mode", options, index=0 if samples else 1, label_visibility="collapsed")
    cfg = {"mode": "sample" if mode == options[0] else "paste"}
    if cfg["mode"] == "sample":
        if not samples:
            st.sidebar.error("demo_cache/samples.json not found. Run `make demo-cache`.")
            cfg["mode"] = "paste"
        else:
            labels = [sample_label(i, s) for i, s in enumerate(samples["samples"])]
            cfg["sample_idx"] = st.sidebar.selectbox("CNN/DM test article", range(len(labels)),
                                                     format_func=lambda i: labels[i])
    if cfg["mode"] == "paste":
        cfg["text"] = st.sidebar.text_area("Article text", height=260,
                                           placeholder="Paste a news article here ...")
        cfg["reference"] = st.sidebar.text_area("Reference summary (optional, one sentence per line)",
                                                height=90)
        cfg["go"] = st.sidebar.button("Summarize", type="primary", use_container_width=True)

    st.sidebar.header("Settings")
    cfg["model_name"] = st.sidebar.radio("Abstractive model", list(ABS_MODELS),
                                         help="DistilBART is ~2× faster with slightly lower quality.")
    cfg["factcheck"] = st.sidebar.checkbox("Factual-consistency check (heuristic)", value=True,
                                           help="Flags numbers / names in the abstractive summary that "
                                                "do not appear in the article.")
    if samples:
        st.sidebar.caption(f"Cache built {samples['meta']['created']} on {samples['meta']['device']}. "
                           f"Fact-check backend: {samples['meta']['factcheck_backend']}.")
    return cfg


def render_summaries(res: dict, model_name: str, factcheck: bool, precomputed: bool):
    from src.factcheck import highlight_html

    abs_res = res["abstractive"].get(model_name)
    if abs_res is None and res["abstractive"]:
        alt = next(iter(res["abstractive"]))
        st.warning(f"No cached output for {model_name}; showing {alt}.")
        model_name, abs_res = alt, res["abstractive"][alt]

    n_sent = len(res["sentences"])
    ntok = abs_res["n_input_tokens"] if abs_res else None
    stats = f"**Article:** {res['n_article_words']} words · {n_sent} sentences"
    if ntok:
        stats += f" · {ntok} BART tokens"
    st.markdown(stats)

    cols = st.columns(3, gap="medium")
    for col, key in zip(cols, ("lead3", "extractive", "abstractive")):
        name, pill, verb, cap = COL_INFO[key]
        block = abs_res if key == "abstractive" else res[key]
        with col:
            st.markdown(f'<p class="col-head"><span class="pill {pill}">{verb}</span>{name}</p>',
                        unsafe_allow_html=True)
            if key == "abstractive":
                cap = f"{model_name}: {cap}"
            if key == "extractive" and not res["extractive"].get("trained", True):
                st.warning(res["extractive"]["method"])
            st.markdown(f'<p class="col-cap">{cap}</p>', unsafe_allow_html=True)
            if block is None:
                st.info("Abstractive output unavailable.")
                continue
            if key == "abstractive" and factcheck:
                flags = block.get("factcheck", {}).get("flags", [])
                body = highlight_html(block["summary"], flags)
            else:
                flags, body = [], html.escape(block["summary"])
            if key in ("lead3", "extractive") and block.get("summary_sents"):
                body = "<br><br>".join(html.escape(s) for s in block["summary_sents"])
            st.markdown(f'<div class="summary-box">{body or "<i>(empty)</i>"}</div>', unsafe_allow_html=True)
            st.markdown(metrics_card_html(block["metrics"], block.get("rouge"), precomputed),
                        unsafe_allow_html=True)
            if key == "abstractive":
                if block.get("chunked"):
                    st.info(f"**Long article:** {block['n_input_tokens']} tokens > {BART_MAX_INPUT} limit. "
                            f"Split into {block['n_chunks']} chunks, summarized each, then summarized "
                            "the summaries.")
                    with st.expander("Chunk summaries (stage 1)"):
                        for j, cs in enumerate(block["chunk_summaries"], 1):
                            st.markdown(f"**Chunk {j}.** {cs}")
                if factcheck:
                    if flags:
                        st.warning("⚠ **Possibly unsupported:** " + ", ".join(f"`{f['text']}` ({f['type']})"
                                                                            for f in flags)
                                   + "  \nNot found in the article. Heuristic check, so verify manually.")
                    else:
                        st.success("✓ No unsupported numbers or names found (heuristic check).")

    if res.get("reference"):
        with st.expander("Reference highlights (human-written, CNN/DM)", expanded=False):
            for h in res["reference"].split("\n"):
                if h.strip():
                    st.markdown(f"- {h.strip()}")

    st.subheader("Article: what the extractive model highlighted")
    lead_idx = res["lead3"]["indices"]
    st.markdown('<div class="legend"><span><span class="s-sel">&nbsp;selected&nbsp;</span></span>'
                '<span><span class="s-blk">&nbsp;skipped by trigram blocking&nbsp;</span></span>'
                '<span><span class="s-out">grey = beyond 512-token encoder window</span></span>'
                '<span><span class="score">0.42</span> = sentence score (hover for details)</span></div>',
                unsafe_allow_html=True)
    a_col, t_col = st.columns([3, 2], gap="medium")
    with a_col:
        st.markdown(article_html(res["sentences"], res["extractive"], lead_idx), unsafe_allow_html=True)
    with t_col:
        df = sentence_table(res["sentences"], res["extractive"], lead_idx)
        st.dataframe(df, hide_index=True, use_container_width=True, height=560,
                     column_config={"score": st.column_config.ProgressColumn(
                         "score", min_value=0.0, max_value=1.0, format="%.3f"),
                         "sentence": st.column_config.TextColumn("sentence", width="large")})


def render_evaluation():
    csv = RESULTS / "scores.csv"
    if not csv.exists():
        st.info("No evaluation results yet. Run `make eval`.")
        return
    df = pd.read_csv(csv)
    meta = {}
    js = RESULTS / "scores.json"
    if js.exists():
        meta = json.loads(js.read_text()).get("meta", {})
    st.markdown(f"**Our numbers** on a CNN/DM test sample ({meta.get('n_test', '?')} articles; "
                f"abstractive on the first {meta.get('abs_n', '?')}). Device: `{meta.get('device', '?')}`. "
                f"{meta.get('rouge', '')}.")
    show = df.rename(columns={"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L",
                              "novel_1": "novel 1-g %", "novel_2": "novel 2-g %", "novel_3": "novel 3-g %",
                              "repeated_2": "repeat 2-g %", "repeated_3": "repeat 3-g %",
                              "avg_words": "avg words", "avg_latency_s": "latency s/article",
                              "n_articles": "n"})
    st.dataframe(show, hide_index=True, use_container_width=True)
    png = RESULTS / "rouge_comparison.png"
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        if png.exists():
            st.image(str(png), use_container_width=True)
    with c2:
        from config import PUBLISHED_ROUGE1

        st.markdown("**Published ROUGE-1 (full CNN/DM test set)**")
        st.dataframe(pd.DataFrame({"system": list(PUBLISHED_ROUGE1), "ROUGE-1": list(PUBLISHED_ROUGE1.values())}),
                     hide_index=True, use_container_width=True)
        tl = RESULTS / "train_log.json"
        if tl.exists():
            t = json.loads(tl.read_text())
            st.markdown(f"**Extractive training:** {t['train_n']} articles, encoder `{t['encoder']}` (frozen), "
                        f"best epoch {t['best_epoch']}/{t['epochs']}, val ROUGE-1 "
                        f"{t['best_val']['rouge1']:.2f} (val Lead-3 {t['val_lead3']['rouge1']:.2f}, "
                        f"oracle {t['val_oracle']['rouge1']:.2f}); {t['seconds']['total'] / 60:.1f} min on "
                        f"`{t['device']}`.")
    st.caption("Novel n-grams: % of summary n-grams not in the article (how abstractive). Repeated n-grams: "
               "% of summary n-grams that repeat earlier ones (redundancy). Sample sizes are small, so treat "
               "differences under ~1 ROUGE point as noise.")


def render_about():
    st.markdown("""
**Papers.** *See et al. (2017)*: pointer-generator network that can copy words from the source or generate new
ones, plus a coverage loss that reduces repetition (ROUGE-1 39.53 on CNN/DM). *Liu & Lapata (2019)*, BertSum:
pretrained BERT as a document encoder with a [CLS] token per sentence; BertSumExt adds inter-sentence Transformer
layers that score sentences (43.25 ROUGE-1), BertSumAbs/ExtAbs fine-tune an encoder-decoder for abstraction.

**What we built.** Lead-3 baseline · a lightweight BertSumExt (frozen BERT, trained 2-layer head on greedy-oracle
labels, trigram blocking) · pretrained `facebook/bart-large-cnn` as the abstractive model instead of training a
pointer-generator for days.

**Dataset.** CNN/DailyMail 3.0.0 (`abisee/cnn_dailymail`): ~287k / 13k / 11.5k news articles paired with
human-written bullet-point highlights. Optional XSum stress test (one-sentence, highly abstractive summaries).

**Metrics.** ROUGE-1/2/L (Lin, 2004) against the highlights; % novel n-grams (abstractiveness) and % repeated
n-grams (redundancy), the same copy/repetition analysis used by See et al.

**Gaps we address (lightly).** *Factual accuracy*: neither paper checks it, so we flag numbers/names in the
abstractive summary that are absent from the article (heuristic). *Long documents*: both papers cut the input at
400–512 tokens; we chunk articles longer than BART's 1024 tokens and summarize the chunk summaries.
""")


def main():
    st.set_page_config(page_title="News-Article Summarizer", page_icon="📰", layout="wide")
    render_header()
    samples = cached_samples()
    cfg = render_sidebar(samples)
    tab_sum, tab_eval, tab_about = st.tabs(["Summaries", "Evaluation", "About"])

    with tab_sum:
        if cfg["mode"] == "sample":
            res = samples["samples"][cfg["sample_idx"]]
            render_summaries(res, cfg["model_name"], cfg["factcheck"], precomputed=True)
        else:
            text = (cfg.get("text") or "").strip()
            key = hashlib.md5(f"{text}|{cfg.get('reference')}|{cfg['model_name']}".encode()).hexdigest()
            if cfg.get("go"):
                if len(text.split()) < 20:
                    st.warning("Please paste a longer article (at least a few sentences).")
                else:
                    with st.spinner(f"Summarizing with Lead-3, extractive and {cfg['model_name']} ..."):
                        ext = get_extractive()
                        abs_model = get_abstractive(ABS_MODELS[cfg["model_name"]])
                        st.session_state["live"] = (key, live_result(text, cfg["model_name"],
                                                                     cfg.get("reference"), ext, abs_model))
            live = st.session_state.get("live")
            if live and live[0] == key:
                render_summaries(live[1], cfg["model_name"], cfg["factcheck"], precomputed=False)
            elif live:
                st.info("Input or model changed. Click **Summarize** to update.")
            else:
                st.info("Paste an article in the sidebar and click **Summarize**.")

    with tab_eval:
        render_evaluation()
    with tab_about:
        render_about()


if __name__ == "__main__":
    main()
