# Common tasks. Everything here is also spelled out in the README for people without make.
PY ?= .venv/bin/python
TASK ?= How many transactions are recorded in data/payments.csv?

.PHONY: help setup data model run web notebook notebook-build notebook-run test lint fmt docker clean

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-15s %s\n", $$1, $$2}'

setup:           ## create .venv and install the package with dev + notebook extras
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt -e ".[dev,notebook]"

data:            ## download the DABstep dataset into workspace/data/
	$(PY) scripts/download_data.py

model:           ## pull the default local model
	ollama pull qwen3.5:4b

run:             ## ask the agent one question (TASK="...")
	$(PY) -m agent "$(TASK)"

web:             ## start the web UI on http://localhost:8765
	$(PY) webapp/server.py

notebook:        ## open the step-by-step build in Jupyter Lab
	$(PY) -m jupyter lab notebook/agent_from_scratch.ipynb

notebook-build:  ## regenerate the notebook from notebook/build_notebook.py
	$(PY) notebook/build_notebook.py

notebook-run:    ## execute the notebook top to bottom (needs Ollama or a free API key)
	$(PY) notebook/run_notebook.py

test:            ## run the test suite (no LLM or network needed)
	$(PY) -m pytest

lint:            ## ruff
	$(PY) -m ruff check .

fmt:             ## ruff autofix
	$(PY) -m ruff check --fix .

docker:          ## web UI + Ollama in Docker
	docker compose up --build

clean:           ## remove caches and saved runs
	rm -rf .pytest_cache .ruff_cache runs **/__pycache__
