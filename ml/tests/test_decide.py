import json

import numpy as np
import pytest

from fraud_ml.config import load_params
from fraud_ml.data import LABEL
from fraud_ml.decide import (
    APPROVE,
    DECLINE,
    REVIEW,
    apply_policy,
    decide,
    fit_thresholds,
    outcome_summary,
    run_decisions,
)
from fraud_ml.evaluate import load_models, score
from fraud_ml.prepare import prepare
from fraud_ml.train import load_prepared, train


@pytest.fixture
def settings(workspace):
    return load_params(workspace / "params.yaml").decision


def test_costs_are_integer_cents_per_outcome(settings):
    y = np.array([1, 0, 1, 0])
    cents = np.array([10_000, 10_000, 2_000, 3_000])
    outcome = np.array([APPROVE, DECLINE, REVIEW, APPROVE])
    summary = outcome_summary(y, cents, outcome, settings)
    # approved fraud 10,000 + false decline 10% of 10,000 + one review 500
    assert summary["cost_cents"] == 10_000 + 1_000 + 500
    assert summary["false_declines"] == 1
    assert summary["fraud_amount_stopped_share"] == pytest.approx(2_000 / 12_000)


def test_thresholds_respect_review_capacity_and_order(settings):
    rng = np.random.default_rng(0)
    y = (rng.random(2_000) < 0.05).astype(int)
    p = np.clip(0.6 * y + rng.random(2_000) * 0.5, 0, 1)
    cents = rng.integers(1_000, 50_000, 2_000)
    fitted = fit_thresholds(p, y, cents, 0.05, settings)
    outcome = apply_policy(p, fitted["t_review"], fitted["t_decline"])
    assert (outcome == REVIEW).mean() <= 0.05 + 1e-9
    assert fitted["t_decline"] >= fitted["t_review"]
    # Never worse than approving everything, on the data it was fitted on.
    approve_all = outcome_summary(y, cents, np.full(len(y), APPROVE), settings)["cost_cents"]
    assert fitted["fit_cost_cents"] <= approve_all


def test_thresholds_ignore_test_labels(workspace):
    params = load_params(workspace / "params.yaml")
    prepare(params)
    train(params)
    pre, ranges = load_prepared(params, ("valid", "test"))
    models, _ = load_models(params)
    valid, test = ranges["valid"], ranges["test"]
    vs, ts = score(models, valid, pre), score(models, test, pre)
    flipped = test.assign(**{LABEL: 1 - test[LABEL]})
    keys = ("model", "capacity", "t_review", "t_decline")
    original = run_decisions(valid, vs, test, ts, params)["policies"]
    changed = run_decisions(valid, vs, flipped, ts, params)["policies"]
    assert [{k: p[k] for k in keys} for p in original] == [{k: p[k] for k in keys} for p in changed]

    report = decide(params)
    result = json.loads((params.data.reports_dir / "decision" / "decision.json").read_text())
    assert len(result["policies"]) == 3 * len(params.decision.review_capacities)
    assert "## Drift" in report.read_text()
