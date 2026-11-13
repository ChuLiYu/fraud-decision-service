import json

import numpy as np
import pandas as pd
import pytest

from fraud_ml.config import load_params
from fraud_ml.data import LABEL, TIME
from fraud_ml.evaluate import load_models, score
from fraud_ml.fusion import REFERENCE, ecdf, fuse, run_ablation, split_halves, verdict
from fraud_ml.prepare import prepare
from fraud_ml.train import load_prepared, train


@pytest.fixture
def scored(workspace):
    params = load_params(workspace / "params.yaml")
    prepare(params)
    train(params)
    pre, ranges = load_prepared(params, ("valid", "test"))
    models, _ = load_models(params)
    valid, test = ranges["valid"], ranges["test"]
    return params, valid, score(models, valid, pre), test, score(models, test, pre)


def test_fuse_writes_report_with_every_method_and_a_verdict(scored):
    params = scored[0]
    report = fuse(params)
    result = json.loads((params.data.reports_dir / "fusion" / "fusion.json").read_text())
    assert set(result["test"]) == {"baseline", "lightgbm", "mlp", "rank_mean", "blend", "stack"}
    assert result["fitted"]["chosen"] in {"rank_mean", "blend", "stack"}
    assert isinstance(result["verdict"]["adopt"], bool)
    assert "## Verdict" in report.read_text()


def test_fitting_and_choice_ignore_test_labels(scored):
    # Leakage rule: fusion is fitted and chosen on validation data only, so
    # inverting every test label must not change anything that was learned.
    params, valid, valid_scores, test, test_scores = scored
    flipped = test.assign(**{LABEL: 1 - test[LABEL]})
    original = run_ablation(valid, valid_scores, test, test_scores, params)
    changed = run_ablation(valid, valid_scores, flipped, test_scores, params)
    assert original["fitted"] == changed["fitted"]
    assert original["test"] != changed["test"]


def test_split_halves_are_consecutive_in_time():
    frame = pd.DataFrame({TIME: [5, 1, 3, 3, 4, 2], "TransactionID": range(6)})
    first, second = split_halves(frame)
    assert frame.loc[first, TIME].max() < frame.loc[second, TIME].min()
    assert (first ^ second).all()


def test_ecdf_is_a_monotone_percentile():
    reference = np.array([0.1, 0.2, 0.3, 0.4])
    values = np.array([0.0, 0.25, 0.4, 9.0])
    assert ecdf(reference, values).tolist() == [0.0, 0.5, 1.0, 1.0]


def _slices(*scores):
    return [{"pr_auc": s} for s in scores]


def test_verdict_requires_both_gain_and_every_slice():
    test = {
        REFERENCE: {"pr_auc": 0.50, "time_slices": _slices(0.5, 0.5)},
        "wins_all": {"pr_auc": 0.51, "time_slices": _slices(0.6, 0.6)},
        "loses_one": {"pr_auc": 0.52, "time_slices": _slices(0.7, 0.4)},
        "too_small": {"pr_auc": 0.502, "time_slices": _slices(0.6, 0.6)},
    }
    assert verdict(test, "wins_all", 0.005)["adopt"]
    assert not verdict(test, "loses_one", 0.005)["adopt"]
    assert not verdict(test, "too_small", 0.005)["adopt"]
