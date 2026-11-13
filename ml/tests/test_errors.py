import json

import numpy as np
import pandas as pd
import pytest

from fraud_ml.config import load_params
from fraud_ml.data import LABEL
from fraud_ml.errors import errors, fold_small, review_threshold, run_errors, segment_table
from fraud_ml.evaluate import load_models, score
from fraud_ml.prepare import prepare
from fraud_ml.train import load_prepared, train


def test_threshold_sends_the_review_rate_to_review():
    scores = np.linspace(0, 1, 1_001)
    assert (scores >= review_threshold(scores, 0.02)).mean() == pytest.approx(0.02, abs=0.002)


def test_small_segments_are_folded():
    groups = pd.Series(["a"] * 300 + ["b"] * 10 + ["c"] * 5)
    folded = fold_small(groups, min_rows=200)
    assert set(folded) == {"a", "other"}


def test_segment_table_counts_catches_and_misses():
    groups = pd.Series(["x", "x", "y", "y"])
    y = np.array([1, 0, 1, 1])
    p = np.array([0.9, 0.8, 0.1, 0.95])
    rows = {r["segment"]: r for r in segment_table(groups, y, p, threshold=0.5)}
    assert rows["x"]["recall"] == 1.0 and rows["x"]["flagged_legit"] == 1
    assert rows["y"]["recall"] == 0.5 and rows["y"]["missed_share"] == 1.0


def test_report_threshold_ignores_test_labels(workspace):
    params = load_params(workspace / "params.yaml")
    prepare(params)
    train(params)
    errors(params)
    first = json.loads((params.data.reports_dir / "errors" / "errors.json").read_text())

    path = params.data.processed_dir / "test.parquet"
    frame = pd.read_parquet(path)
    flipped = frame.assign(**{LABEL: 1 - frame[LABEL]})
    pre, ranges = load_prepared(params, ("valid",))
    models, _ = load_models(params)
    lgb = {"lightgbm": models["lightgbm"]}
    result = run_errors(
        score(lgb, ranges["valid"], pre)["lightgbm"],
        flipped,
        score(lgb, flipped, pre)["lightgbm"],
        pre,
    )
    assert result["threshold"] == first["threshold"]
    assert "amount band" in first["segments"]
