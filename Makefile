# News-Article Summarizer: one command per step.  Windows: use run.ps1 (same targets).
PY_SYS ?= $(shell for p in python3.12 python3.11 python3.10 python3; do command -v $$p >/dev/null 2>&1 && echo $$p && break; done)
VENV   := .venv
PY     := $(VENV)/bin/python

# subset sizes (override: make train TRAIN_N=1000)
TRAIN_N ?= 3000
VAL_N   ?= 300
TEST_N  ?= 200
ABS_N   ?= 100
PORT    ?= 8501

.PHONY: setup setup-cpu models train eval eval-full demo-cache app test report all clean

setup:            ## venv + pinned deps + pre-download all models
	$(PY_SYS) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) scripts/download_models.py

setup-cpu:        ## same, but installs the small CPU-only PyTorch wheel (no CUDA download)
	$(PY_SYS) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements.txt
	$(PY) scripts/download_models.py

models:
	$(PY) scripts/download_models.py

train:            ## train the extractive head (default 3000 train / 300 val articles)
	$(PY) -m src.train_extractive --train-n $(TRAIN_N) --val-n $(VAL_N)

eval:             ## evaluate on CNN/DM test (default 200 articles, BART on 100)
	$(PY) -m src.evaluate --n $(TEST_N) --abs-n $(ABS_N)

eval-full:        ## bigger evaluation (GPU recommended) incl. the XSum stress test
	$(PY) -m src.evaluate --n 1000 --abs-n 1000 --val-n 300 --xsum --xsum-n 100

demo-cache:       ## precompute the ~10 demo samples
	$(PY) -m src.precompute_demo

app:              ## launch the Streamlit demo
	$(PY) -m streamlit run app.py --server.port $(PORT)

test:
	$(PY) -m pytest -q

report:           ## regenerate figures + the PDF report
	$(PY) -m src.make_figures
	$(PY) report/build_pdf.py

all: train eval demo-cache test

clean:            ## remove caches (keeps committed results / demo cache / checkpoint)
	rm -rf data_cache .pytest_cache **/__pycache__
