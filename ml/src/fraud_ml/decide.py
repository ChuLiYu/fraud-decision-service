"""Stage 7: decision-level evaluation (scope S-17).

A fraud score is only useful once it becomes an action. This stage turns each
model's score into an approve / review / decline policy and measures what the
policy costs, instead of ranking quality alone.

Policy: ``decline`` if score >= ``t_decline``; ``review`` if score >=
``t_review``; otherwise ``approve``. Reviewed fraud is assumed stopped and
reviewed legitimate transactions approved after review.

Cost per transaction, in integer cents (scope: no binary floating point for
money), from scenario parameters in ``params.yaml`` (not measured costs;
dataset card):

* approved fraud: ``fraud_loss_rate`` x amount (chargeback);
* review: ``review_cost_cents``;
* declined legitimate transaction: ``false_decline_rate`` x amount.

Protocol: for each review capacity (the most a review team can handle, as a
share of transactions), thresholds that minimize cost are found on the
**validation** range, then applied unchanged to the **test** range. Test
labels never influence a threshold (tested). Fixed thresholds also show
drift: if score distributions move, the realized review share on later test
slices departs from the planned capacity.

Outputs ``reports/decision/decision.json`` and ``decision.md``: aggregates only.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_ml import manifest
from fraud_ml.config import DecisionParams, Params, load_params
from fraud_ml.data import LABEL, TIME
from fraud_ml.evaluate import load_models, score
from fraud_ml.train import load_prepared

AMOUNT = "TransactionAmt"
APPROVE, REVIEW, DECLINE = 0, 1, 2
OUTCOMES = {"approve": APPROVE, "review": REVIEW, "decline": DECLINE}


def amount_cents(frame: pd.DataFrame) -> np.ndarray:
    """Transaction amounts as integer cents."""
    return np.rint(frame[AMOUNT].to_numpy() * 100).astype(np.int64)


def row_costs(y: np.ndarray, cents: np.ndarray, settings: DecisionParams) -> dict[int, np.ndarray]:
    """Cost of each row under each outcome, in integer cents."""
    fraud = y.astype(bool)
    return {
        APPROVE: np.where(fraud, np.rint(settings.fraud_loss_rate * cents), 0).astype(np.int64),
        REVIEW: np.full(len(y), settings.review_cost_cents, dtype=np.int64),
        DECLINE: np.where(fraud, 0, np.rint(settings.false_decline_rate * cents)).astype(np.int64),
    }


def fit_thresholds(
    p: np.ndarray, y: np.ndarray, cents: np.ndarray, capacity: float, settings: DecisionParams
) -> dict:
    """Cost-minimizing thresholds on labelled data, with review share <= ``capacity``.

    Rows are sorted by score; declining the top ``d`` rows and reviewing the
    next ``r`` is evaluated on a grid of ``(d, r)`` with prefix sums, so the
    search is fast even on 90,000 rows.

    Returns:
        ``t_review`` and ``t_decline`` (``inf`` means the outcome is never
        used) and the fitted cost.
    """
    n = len(p)
    order = np.argsort(-p, kind="stable")
    ranked = p[order]
    costs = {k: v[order] for k, v in row_costs(y, cents, settings).items()}
    # prefix[k] = cost of the top k rows under that outcome.
    prefix = {k: np.concatenate([[0], np.cumsum(v)]) for k, v in costs.items()}
    total_approve = prefix[APPROVE][-1]

    grid = settings.threshold_grid
    declines = np.unique(np.linspace(0, int(n * settings.max_decline_rate), grid).astype(int))
    reviews = np.unique(np.linspace(0, int(n * capacity), grid).astype(int))
    d, r = np.meshgrid(declines, reviews, indexing="ij")
    end = np.minimum(d + r, n)
    cost = (
        prefix[DECLINE][d]
        + (prefix[REVIEW][end] - prefix[REVIEW][d])
        + (total_approve - prefix[APPROVE][end])
    )
    i, j = np.unravel_index(np.argmin(cost), cost.shape)
    best_d, best_end = int(d[i, j]), int(end[i, j])
    t_decline = float(ranked[best_d - 1]) if best_d > 0 else np.inf
    t_review = float(ranked[best_end - 1]) if best_end > best_d else t_decline
    return {"t_review": t_review, "t_decline": t_decline, "fit_cost_cents": int(cost[i, j])}


def apply_policy(p: np.ndarray, t_review: float, t_decline: float) -> np.ndarray:
    """Outcome code per row for the given thresholds."""
    return np.where(p >= t_decline, DECLINE, np.where(p >= t_review, REVIEW, APPROVE))


def outcome_summary(
    y: np.ndarray, cents: np.ndarray, outcome: np.ndarray, settings: DecisionParams
) -> dict:
    """Volumes, fraud stopped, false declines, and cost of one policy on one range."""
    per_row = per_row_cost(y, cents, outcome, settings)
    fraud = y.astype(bool)
    stopped = fraud & (outcome != APPROVE)
    fraud_cents = int(cents[fraud].sum())
    n = len(y)
    return {
        "rows": n,
        **{f"{name}_share": float((outcome == code).mean()) for name, code in OUTCOMES.items()},
        "fraud_stopped_share": float(stopped.sum() / fraud.sum()) if fraud.any() else float("nan"),
        "fraud_amount_stopped_share": float(cents[stopped].sum() / fraud_cents)
        if fraud_cents
        else float("nan"),
        "false_declines": int((~fraud & (outcome == DECLINE)).sum()),
        "cost_cents": int(per_row.sum()),
        "cost_per_1000_cents": int(round(per_row.sum() * 1000 / n)),
    }


def per_row_cost(
    y: np.ndarray, cents: np.ndarray, outcome: np.ndarray, settings: DecisionParams
) -> np.ndarray:
    """Cost of each row under its assigned outcome, in cents."""
    costs = row_costs(y, cents, settings)
    return np.choose(outcome, [costs[APPROVE], costs[REVIEW], costs[DECLINE]])


def bootstrap_difference(a: np.ndarray, b: np.ndarray, resamples: int, seed: int) -> dict:
    """95% interval of ``sum(b - a)`` per 1,000 rows, resampling rows with replacement.

    Positive values mean policy ``a`` costs less than policy ``b``.
    """
    rng = np.random.default_rng(seed)
    diff = (b - a).astype(np.float64)
    n = len(diff)
    draws = np.array([diff[rng.integers(0, n, n)].sum() for _ in range(resamples)])
    scale = 1000 / n
    return {
        "mean_cents_per_1000": float(diff.sum() * scale),
        "low_cents_per_1000": float(np.percentile(draws, 2.5) * scale),
        "high_cents_per_1000": float(np.percentile(draws, 97.5) * scale),
    }


def slice_drift(
    frame: pd.DataFrame, outcome: np.ndarray, y: np.ndarray, cents: np.ndarray, params: Params
) -> list[dict]:
    """Realized review share and cost on consecutive equal-row time slices."""
    order = np.argsort(frame[TIME].to_numpy(), kind="stable")
    result = []
    for index, chunk in enumerate(np.array_split(order, params.evaluation.time_slices)):
        summary = outcome_summary(y[chunk], cents[chunk], outcome[chunk], params.decision)
        result.append(
            {
                "slice": index + 1,
                "review_share": summary["review_share"],
                "decline_share": summary["decline_share"],
                "cost_per_1000_cents": summary["cost_per_1000_cents"],
            }
        )
    return result


def run_decisions(
    valid: pd.DataFrame, valid_scores: dict, test: pd.DataFrame, test_scores: dict, params: Params
) -> dict:
    """Fit thresholds on validation per model and capacity; report them on test."""
    settings = params.decision
    yv, cv = valid[LABEL].to_numpy(), amount_cents(valid)
    yt, ct = test[LABEL].to_numpy(), amount_cents(test)
    approve_all = np.full(len(yt), APPROVE)
    result = {
        "approve_all": outcome_summary(yt, ct, approve_all, settings),
        "policies": [],
        "comparisons": [],
    }
    for capacity in settings.review_capacities:
        test_costs = {}
        for name in valid_scores:
            thresholds = fit_thresholds(valid_scores[name], yv, cv, capacity, settings)
            outcome = apply_policy(
                test_scores[name], thresholds["t_review"], thresholds["t_decline"]
            )
            test_costs[name] = per_row_cost(yt, ct, outcome, settings)
            result["policies"].append(
                {
                    "model": name,
                    "capacity": capacity,
                    # JSON has no infinity: None means the outcome is never used.
                    **{k: None if np.isinf(v) else v for k, v in thresholds.items()},
                    "test": outcome_summary(yt, ct, outcome, settings),
                    "test_slices": slice_drift(test, outcome, yt, ct, params),
                }
            )
        for other in sorted(set(test_costs) - {"lightgbm"}):
            result["comparisons"].append(
                {
                    "capacity": capacity,
                    "better": "lightgbm",
                    "than": other,
                    **bootstrap_difference(
                        test_costs["lightgbm"], test_costs[other], settings.bootstrap, params.seed
                    ),
                }
            )
    return result


def _dollars(cents: float) -> str:
    return f"${cents / 100:,.0f}"


def _threshold(value: float | None) -> str:
    return "never" if value is None else f"{value:.3f}"


def render(result: dict, settings: DecisionParams) -> str:
    """Markdown report generated from ``decision.json``; do not edit by hand."""
    base = result["approve_all"]
    lines = [
        "# Decision-level evaluation",
        "",
        "Generated by `make decide`; do not edit by hand. Thresholds are fitted on the",
        "validation range and applied unchanged to the test range.",
        "",
        f"- Code: `{result['code']['commit']}`"
        + (" (dirty tree)" if result["code"]["dirty"] else ""),
        f"- Models params hash: `{result['models_params_hash'][:12]}`",
        "- Scenario (assumptions, not measured costs): review "
        f"{_dollars(settings.review_cost_cents)} each; approved fraud loses "
        f"{settings.fraud_loss_rate:.0%} of its amount; a declined legitimate transaction "
        f"loses {settings.false_decline_rate:.0%} of its amount; at most "
        f"{settings.max_decline_rate:.0%} of transactions declined",
        f"- Reference, approve everything: {_dollars(base['cost_per_1000_cents'])} "
        "per 1,000 test transactions",
        "",
        "## Test range by review capacity",
        "",
        "| Capacity | Model | t_review | t_decline | Review | Decline | Fraud $ stopped "
        "| False declines | Cost / 1,000 | Saving vs approve-all |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for policy in result["policies"]:
        t = policy["test"]
        saving = 1 - t["cost_cents"] / base["cost_cents"]
        lines.append(
            f"| {policy['capacity']:.1%} | {policy['model']} | {_threshold(policy['t_review'])} | "
            f"{_threshold(policy['t_decline'])} | {t['review_share']:.2%} | "
            f"{t['decline_share']:.2%} | {t['fraud_amount_stopped_share']:.1%} | "
            f"{t['false_declines']:,} | {_dollars(t['cost_per_1000_cents'])} | {saving:.1%} |"
        )
    lines += [
        "",
        "## Is LightGBM's saving real? (bootstrap over test rows)",
        "",
        "Cost saved per 1,000 test transactions by LightGBM's policy compared with each "
        f"other model's, with a 95% interval from {settings.bootstrap} resamples.",
        "",
        "| Capacity | Compared with | Saving / 1,000 | 95% interval |",
        "|---|---|---|---|",
    ]
    for c in result["comparisons"]:
        lines.append(
            f"| {c['capacity']:.1%} | {c['than']} | {_dollars(c['mean_cents_per_1000'])} | "
            f"{_dollars(c['low_cents_per_1000'])} to {_dollars(c['high_cents_per_1000'])} |"
        )
    lines += [
        "",
        "## Drift: fixed thresholds on later test slices (LightGBM)",
        "",
        "If the review share moves away from the planned capacity, the review team is",
        "over- or under-loaded even though the thresholds did not change.",
        "",
        "| Capacity | Slice | Review share | Decline share | Cost / 1,000 |",
        "|---|---|---|---|---|",
    ]
    for policy in result["policies"]:
        if policy["model"] != "lightgbm":
            continue
        for s in policy["test_slices"]:
            lines.append(
                f"| {policy['capacity']:.1%} | {s['slice']} | {s['review_share']:.2%} | "
                f"{s['decline_share']:.2%} | {_dollars(s['cost_per_1000_cents'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def decide(params: Params) -> Path:
    """Run the decision-level evaluation and return the report path."""
    pre, ranges = load_prepared(params, ("valid", "test"))
    models, trained = load_models(params)
    valid, test = ranges["valid"], ranges["test"]
    result = {
        "code": manifest.code_version(),
        "models_params_hash": trained["params_hash"],
        "settings": dataclasses.asdict(params.decision),
        **run_decisions(valid, score(models, valid, pre), test, score(models, test, pre), params),
    }
    out = params.data.reports_dir / "decision"
    out.mkdir(parents=True, exist_ok=True)
    (out / "decision.json").write_text(json.dumps(result, indent=2, default=_json_default))
    report = out / "decision.md"
    report.write_text(render(result, params.decision))
    return report


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)}")


def main() -> None:
    """Command-line entry point: ``python -m fraud_ml.decide``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args()
    print(f"wrote {decide(load_params(args.params))}")


if __name__ == "__main__":
    main()
