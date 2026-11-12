# Entry points for the ml/ component. Run from this directory.
.PHONY: setup lint test check fetch prepare data train evaluate compare all

setup:  ## Create the virtual environment from uv.lock
	uv sync --locked

lint:  ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

test:  ## Run unit tests
	uv run pytest

check: lint test  ## Everything CI would run

fetch:  ## Download the labeled IEEE-CIS tables (needs accepted rules and a Kaggle API token)
	uv sync --locked --extra fetch
	uv run python -m fraud_ml.fetch

prepare:  ## Validate, split by time, fit preprocessing on train, write artifacts + manifest
	uv run python -m fraud_ml.prepare

data: fetch prepare  ## Raw download through processed artifacts

train:  ## Train baseline and LightGBM from prepared artifacts (verifies the manifest first)
	uv run python -m fraud_ml.train

RUN ?= latest
evaluate:  ## Score validation and test ranges into reports/runs/$(RUN); e.g. make evaluate RUN=m3-features
	uv run python -m fraud_ml.evaluate --run $(RUN)

compare:  ## Tabulate all runs into reports/comparison.md
	uv run python -m fraud_ml.compare

all: data train evaluate compare  ## Reproduce everything from download to report
