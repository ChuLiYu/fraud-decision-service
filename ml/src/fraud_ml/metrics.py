"""Evaluation metrics for rare-event scoring.

Fraud is about 3.5% of transactions, so accuracy is meaningless (always
predicting "legitimate" scores 96.5%). These metrics answer the questions a
fraud team asks instead:

* **ROC-AUC:** how well scores rank fraud above legitimate overall.
* **PR-AUC (average precision):** ranking quality focused on the rare class;
  a random model scores about the fraud rate.
* **At a review rate:** if the top ``rate`` share of transactions by score
  go to manual review, what share of them are fraud (precision) and what
  share of all fraud is caught (recall)?
* **Calibration:** whether a score of 0.2 means about 20% of such
  transactions are fraud, which decision thresholds and costs rely on.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def ranking(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """ROC-AUC and PR-AUC; ``nan`` when only one class is present."""
    if len(np.unique(y)) < 2:
        return {"roc_auc": math.nan, "pr_auc": math.nan}
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }


def at_review_rate(y: np.ndarray, p: np.ndarray, rate: float) -> dict[str, float]:
    """Precision and recall when the top ``rate`` share of scores is reviewed.

    The reviewed count is ``ceil(rate * n)``. Ties at the cut are broken by
    original order, so the result is deterministic.
    """
    n = len(y)
    k = max(1, math.ceil(rate * n))
    top = np.argsort(-p, kind="stable")[:k]
    caught = int(y[top].sum())
    total = int(y.sum())
    return {
        "review_rate": rate,
        "reviewed": k,
        "threshold": float(p[top[-1]]),
        "precision": caught / k,
        "recall": caught / total if total else math.nan,
    }


def calibration(y: np.ndarray, p: np.ndarray, bins: int) -> dict:
    """Reliability table over equal-count score bins, plus ECE and Brier score.

    Equal-count bins are used because most scores sit near zero; equal-width
    bins would put nearly every row in the first bin.

    ECE (expected calibration error) is the row-weighted mean gap between
    average predicted score and observed fraud rate across bins.
    """
    order = np.argsort(p, kind="stable")
    table, gap = [], 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk) == 0:
            continue
        predicted, observed = float(p[chunk].mean()), float(y[chunk].mean())
        table.append({"rows": len(chunk), "mean_score": predicted, "fraud_rate": observed})
        gap += len(chunk) * abs(predicted - observed)
    return {"ece": gap / len(y), "brier": float(brier_score_loss(y, p)), "bins": table}


def time_slices(time: np.ndarray, y: np.ndarray, p: np.ndarray, slices: int) -> list[dict]:
    """Ranking metrics on consecutive equal-row slices in time order.

    A model that degrades in later slices is drifting: the fraud patterns it
    learned are going stale.
    """
    order = np.argsort(time, kind="stable")
    result = []
    for index, chunk in enumerate(np.array_split(order, slices)):
        result.append(
            {
                "slice": index + 1,
                "start_time": int(time[chunk].min()),
                "rows": len(chunk),
                "fraud_rate": float(y[chunk].mean()),
                **ranking(y[chunk], p[chunk]),
            }
        )
    return result
