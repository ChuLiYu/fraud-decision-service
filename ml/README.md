# ml — training and evaluation

Course project component (scope phase 1). Requires [uv](https://docs.astral.sh/uv/); uv installs the pinned Python version.

```bash
make setup   # create .venv from uv.lock
make check   # lint, format check, tests
```

Versions: ranges in `pyproject.toml`, exact pins in `uv.lock`. Pipeline stages are described in `src/fraud_ml/__init__.py` and added per milestone ([scope](../docs/product/scope.md)).
