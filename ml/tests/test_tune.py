import json

import pandas as pd
import pytest

from fraud_ml.config import TuningParams, load_params
from fraud_ml.prepare import prepare
from fraud_ml.tune import tune


def test_tune_never_reads_the_test_range(workspace, monkeypatch):
    params = load_params(workspace / "params.yaml")
    prepare(params)
    read = []
    original = pd.read_parquet

    def spy(path, *args, **kwargs):
        read.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    report = tune(params)

    assert read and not any(path.endswith("test.parquet") for path in read)
    result = json.loads((report.parent / "tuning.json").read_text())
    assert len(result["grid"]) == 2  # num_leaves: [3, 7]
    assert set(result["seed_study"]) == {"current", "best_grid"}
    assert len(result["seed_study"]["current"]["scores"]) == 2


def test_unknown_grid_setting_rejected():
    with pytest.raises(ValueError, match="unknown"):
        TuningParams(grid={"num_trees": [1]}, seeds=[1])
