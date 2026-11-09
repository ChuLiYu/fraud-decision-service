# ml — training and evaluation

Course project component (scope phase 1). Requires [uv](https://docs.astral.sh/uv/); uv installs the pinned Python version.

```bash
make setup   # create .venv from uv.lock
make check   # lint, format check, tests
make all     # fetch -> prepare -> train -> evaluate
```

Stages hand off through files and each verifies the previous stage's manifest:

| Stage | Reads | Writes |
|---|---|---|
| `fetch` | Kaggle | `data/raw/` |
| `prepare` | `data/raw/` | `data/processed/`: train/valid/test Parquet, `preprocessor.json`, manifest |
| `train` | `data/processed/` (train, valid) | `data/models/`: baseline, LightGBM, manifest |
| `evaluate` | `data/processed/` (valid, test), `data/models/` | `reports/metrics.json`, `reports/evaluation.md` (committed; aggregates only) |

Tunable values are in `params.yaml`, explained in the [tuning guide](../docs/ml/tuning-guide.md).

Versions: ranges in `pyproject.toml`, exact pins in `uv.lock`. Pipeline stages are described in `src/fraud_ml/__init__.py` and added per milestone ([scope](../docs/product/scope.md)).
