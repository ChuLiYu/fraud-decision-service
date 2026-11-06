# Entry points for the ml/ component. Run from this directory.
.PHONY: setup lint test check

setup:  ## Create the virtual environment from uv.lock
	uv sync --locked

lint:  ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

test:  ## Run unit tests
	uv run pytest

check: lint test  ## Everything CI would run
