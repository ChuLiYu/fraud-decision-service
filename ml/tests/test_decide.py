import json

import numpy as np
import pytest

from fraud_ml.config import load_params
from fraud_ml.data import LABEL
from fraud_ml.decide import (
    APPROVE,
    DECLINE,
    REVIEW,
    apply_expected_policy,
    apply_thresholds,
    decide,
    fit_expected_policy,
    fit_thresholds,
    outcome_summary,
    run_decisions,
)
from fraud_ml.evaluate import load_models, score
from fraud_ml.prepare import prepare
from fraud_ml.train import load_prepared, train


@pytest.fixture
def params(workspace):
    return load_params(workspace / "params.yaml")


def _synthetic(n=2_000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    p = np.clip(0.6 * y + rng.random(n) * 0.5, 0, 1)
    cents = rng.integers(1_000, 50_000, n)
    return p, y, cents


def test_costs_are_integer_cents_per_outcome(params):
    costs = params.decision.scenarios["costly_declines"]
    y = np.array([1, 0, 1, 0])
    cents = np.array([10_000, 10_000, 2_000, 3_000])
    outcome = np.array([APPROVE, DECLINE, REVIEW, APPROVE])
    summary = outcome_summary(y, cents, outcome, costs)
    # approved fraud 10,000 + false decline (20% of 10,000 + 5,000 fixed) + one review 500
    assert summary["cost_cents"] == 10_000 + 2_000 + 5_000 + 500
    assert summary["false_declines"] == 1
    assert summary["fraud_amount_stopped_share"] == pytest.approx(2_000 / 12_000)


def test_thresholds_respect_review_capacity_and_order(params):
    p, y, cents = _synthetic()
    costs = params.decision.scenarios["base"]
    fitted = fit_thresholds(p, y, cents, 0.05, costs, 0.1, 50)
    outcome = apply_thresholds(p, fitted["t_review"], fitted["t_decline"])
    assert (outcome == REVIEW).mean() <= 0.05 + 1e-9
    assert fitted["t_decline"] >= fitted["t_review"]
    # Never worse than approving everything, on the data it was fitted on.
    approve_all = outcome_summary(y, cents, np.full(len(y), APPROVE), costs)["cost_cents"]
    assert fitted["fit_cost_cents"] <= approve_all


def test_expected_policy_respects_capacity(params):
    p, _, cents = _synthetic()
    costs = params.decision.scenarios["base"]
    min_gain = fit_expected_policy(p, cents, 0.02, costs)
    outcome = apply_expected_policy(p, cents, min_gain, costs)
    assert (outcome == REVIEW).mean() <= 0.02 + 1e-9


def test_expected_policy_uses_the_amount(params):
    # Same fraud probability: a small purchase is approved, a large one is not.
    costs = params.decision.scenarios["base"]
    q = np.array([0.05, 0.05])
    cents = np.array([1_000, 500_000])
    outcome = apply_expected_policy(q, cents, 0.0, costs)
    assert outcome[0] == APPROVE
    assert outcome[1] in (REVIEW, DECLINE)


def test_policies_ignore_test_labels(params):
    prepare(params)
    train(params)
    pre, ranges = load_prepared(params, ("valid", "test"))
    models, _ = load_models(params)
    valid, test = ranges["valid"], ranges["test"]
    vs, ts = score(models, valid, pre), score(models, test, pre)
    flipped = test.assign(**{LABEL: 1 - test[LABEL]})

    def learned(result):
        keys = ("scenario", "capacity", "model", "t_review", "t_decline", "min_review_gain_cents")
        return [{k: p.get(k) for k in keys} for p in result["policies"]]

    original = run_decisions(valid, vs, test, ts, params)
    changed = run_decisions(valid, vs, flipped, ts, params)
    assert learned(original) == learned(changed)

    report = decide(params)
    result = json.loads((params.data.reports_dir / "decision" / "decision.json").read_text())
    settings = params.decision
    assert len(result["policies"]) == len(settings.scenarios) * len(settings.review_capacities) * 4
    text = report.read_text()
    assert "## Sensitivity" in text and "## Drift" in text
