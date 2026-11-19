# Neural Text Summarization: News-Article Summarizer

NLP Lab project by **Vedansh Shukla & Abhist Kamle**.

Three summarizers compared side by side on CNN/DailyMail news articles:

- **Lead-3**: the first three sentences (the classic news baseline).
- **Extractive (BertSumExt-style)**: frozen `bert-base-uncased` sentence vectors, a 2-layer Transformer scoring head trained on greedy-oracle labels, top-3 selection with trigram blocking.
- **Abstractive**: pretrained `facebook/bart-large-cnn` (DistilBART available as a faster option), with chunking for articles longer than 1,024 tokens.

Plus a heuristic factual-consistency check (names and numbers in the summary that are not in the article), a full evaluation (ROUGE with bootstrap confidence intervals, novel / repeated n-grams, classifier metrics, long-document and latency analysis) and a Streamlit demo.

**Report:** [`report/Text_Summarization_Prototype_Report.pdf`](report/Text_Summarization_Prototype_Report.pdf) (source: [`report/REPORT.md`](report/REPORT.md)). Results are in [`results/`](results/).

## Quick start (Windows, PowerShell)

```powershell
Set-ExecutionPolicy -Scope Process Bypass   # if scripts are blocked
.\run.ps1 setup       # venv + pinned packages + CUDA PyTorch (if an NVIDIA GPU is found) + all models
.\run.ps1 app         # demo at http://localhost:8501
```

Other steps: `train` (extractive head, ~2 min on GPU; a trained head is already included), `eval` (200 test articles), `eval-full` (1,000 articles + XSum, as in the report), `demo-cache`, `report` (figures + PDF), `test`. On Linux / macOS use `make <target>` with the same names.

In a OneDrive folder the virtual environment is created in `%USERPROFILE%\.venvs\news-summarizer-nlp` (OneDrive sync locks files during install); set `SUMM_VENV` to override.

## Layout

| Path | Contents |
|---|---|
| `src/` | data loading, Lead-3, oracle, extractive model + training, BART summarizer, fact check, metrics, evaluation, figures |
| `app.py` | Streamlit demo |
| `checkpoints/ext_head.pt` | trained extractive head (4 MB) |
| `results/` | evaluation outputs (scores, classifier metrics, per-article summaries) |
| `demo_cache/` | precomputed demo articles |
| `report/` | report source, figures and PDF |
| `tests/` | unit and app smoke tests |
