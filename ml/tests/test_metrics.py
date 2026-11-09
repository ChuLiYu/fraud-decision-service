import math

import numpy as np
import pytest

from fraud_ml import metrics

Y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
P = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05])


def test_review_rate_counts_top_scores():
    # Top 30% = 3 rows (0.9, 0.8, 0.7): 2 of them fraud, 2 of 3 frauds caught.
    result = metrics.at_review_rate(Y, P, 0.3)
    assert result["reviewed"] == 3
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["threshold"] == pytest.approx(0.7)


def test_review_rate_rounds_up_to_at_least_one_row():
    assert metrics.at_review_rate(Y, P, 0.01)["reviewed"] == 1


def test_ranking_is_nan_with_one_class():
    result = metrics.ranking(np.zeros(5), np.linspace(0, 1, 5))
    assert math.isnan(result["roc_auc"]) and math.isnan(result["pr_auc"])


def test_perfectly_calibrated_scores_have_zero_ece():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    result = metrics.calibration(y, p, bins=2)
    assert result["ece"] == pytest.approx(0.0)
    assert [b["rows"] for b in result["bins"]] == [4, 4]


def test_time_slices_follow_time_order():
    time = np.array([5, 1, 4, 2, 3, 6])
    y = np.array([0, 1, 0, 1, 0, 1])
    slices = metrics.time_slices(time, y, np.full(6, 0.5), slices=2)
    assert [s["start_time"] for s in slices] == [1, 4]
    assert [s["fraud_rate"] for s in slices] == pytest.approx([2 / 3, 1 / 3])
