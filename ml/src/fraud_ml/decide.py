"""Stage 7: decision-level evaluation (scope S-17).

A fraud score is only useful once it becomes an action. This stage turns
model scores into approve / review / decline policies and measures what each
policy costs, instead of ranking quality alone.

Costs per transaction, in integer cents (scope: no binary floating point for
money), come from named **scenarios** in ``params.yaml``. They are
assumptions, not measured costs (dataset card), so every conclusion is
checked under each scenario:

* approved fraud: ``fraud_loss_rate`` x amount (chargeback);
* review: ``review_cost_cents``; reviewed fraud is assumed stopped;
* declined legitimate transaction: ``false_decline_rate`` x amount (lost
  margin) + ``false_decline_fixed_cents`` (customer friction).

Two kinds of policy, both limited by a review capacity (the most a review
team can handle, as a share of transactions):

* **Score thresholds** (every model): ``decline`` if score >= ``t_decline``,
  ``review`` if score >= ``t_review``, else ``approve``. The pair minimizing
  cost is searched on the validation range.
* **Expected cost** (LightGBM): scores are calibrated into probabilities
  (isotonic regression on the validation range), then each transaction gets
  the outcome with the lowest expected cost given its probability *and its
  amount*. A $5 and a $5,000 transaction with the same score are treated
  differently, which a single score threshold cannot do. Review goes to the
  transactions where it saves the most, up to capacity.

Protocol: everything a policy learns (thresholds, calibration, review cut-off)
comes from the **validation** range; the **test** range only reports. Test
labels never influence a policy (tested). Fixed policies also show drift: if
score distributions move, the realized review share on later test slices
departs from the planned capacity.

Outputs ``reports/decision/decision.json`` and ``decision.md``: aggregates only.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from fraud_ml import manifest
from fraud_ml.config import CostScenario, Params, load_params
from fraud_ml.data import LABEL, TIME
from fraud_ml.evaluate import load_models, score
from fraud_ml.train import load_prepared

AMOUNT = "TransactionAmt"
APPROVE, REVIEW, DECLINE = 0, 1, 2
OUTCOMES = {"approve": APPROVE, "review": REVIEW, "decline": DECLINE}
EXPECTED = "lightgbm_expected_cost"
BASE = "base"


def amount_cents(frame: pd.DataFrame) -> np.ndarray:
    """Transaction amounts as integer cents."""
    return np.rint(frame[AMOUNT].to_numpy() * 100).astype(np.int64)


def row_costs(y: np.ndarray, cents: np.ndarray, costs: CostScenario) -> dict[int, np.ndarray]:
    """Realized cost of each row under each outcome, in integer cents."""
    fraud = y.astype(bool)
    false_decline = np.rint(costs.false_decline_rate * cents) + costs.false_decline_fixed_cents
    return {
        APPROVE: np.where(fraud, np.rint(costs.fraud_loss_rate * cents), 0).astype(np.int64),
        REVIEW: np.full(len(y), costs.review_cost_cents, dtype=np.int64),
        DECLINE: np.where(fraud, 0, false_decline).astype(np.int64),
    }


def per_row_cost(
    y: np.ndarray, cents: np.ndarray, outcome: np.ndarray, costs: CostScenario
) -> np.ndarray:
    """Realized cost of each row under its assigned outcome, in cents."""
    table = row_costs(y, cents, costs)
    return np.choose(outcome, [table[APPROVE], table[REVIEW], table[DECLINE]])


# --- Score-threshold policy -------------------------------------------------


def fit_thresholds(
    p: np.ndarray,
    y: np.ndarray,
    cents: np.ndarray,
    capacity: float,
    costs: CostScenario,
    max_decline_rate: float,
    grid: int,
) -> dict:
    """Cost-minimizing thresholds on labelled data, with review share <= ``capacity``.

    Rows are sorted by score; declining the top ``d`` rows and reviewing the
    next ``r`` is evaluated on a grid of ``(d, r)`` with prefix sums, so the
    search is fast even on 90,000 rows.

    Returns:
        ``t_review`` and ``t_decline`` (``inf`` means the outcome is never
        used), the fitted cost, and whether the decline search hit its bound.
    """
    n = len(p)
    order = np.argsort(-p, kind="stable")
    ranked = p[order]
    table = {k: v[order] for k, v in row_costs(y, cents, costs).items()}
    # prefix[k][i] = cost of the top i rows under outcome k.
    prefix = {k: np.concatenate([[0], np.cumsum(v)]) for k, v in table.items()}
    total_approve = prefix[APPROVE][-1]

    max_declines = int(n * max_decline_rate)
    declines = np.unique(np.linspace(0, max_declines, grid).astype(int))
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
    return {
        "t_review": t_review,
        "t_decline": t_decline,
        "fit_cost_cents": int(cost[i, j]),
        "decline_at_bound": bool(max_declines > 0 and best_d == max_declines),
    }


def apply_thresholds(p: np.ndarray, t_review: float, t_decline: float) -> np.ndarray:
    """Outcome code per row for the given thresholds."""
    return np.where(p >= t_decline, DECLINE, np.where(p >= t_review, REVIEW, APPROVE))


# --- Expected-cost policy ---------------------------------------------------


def fit_calibration(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    """Map scores to probabilities with a monotone fit on labelled data.

    The expected-cost policy multiplies probabilities by amounts, so a score
    that is systematically too high or too low would bias every decision.
    Isotonic regression keeps the ranking and only corrects the scale.
    """
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)


def expected_costs(q: np.ndarray, cents: np.ndarray, costs: CostScenario) -> dict[int, np.ndarray]:
    """Expected cost of each outcome for fraud probabilities ``q``, in cents."""
    false_decline = costs.false_decline_rate * cents + costs.false_decline_fixed_cents
    return {
        APPROVE: q * costs.fraud_loss_rate * cents,
        REVIEW: np.full(len(q), float(costs.review_cost_cents)),
        DECLINE: (1 - q) * false_decline,
    }


def review_gain(q: np.ndarray, cents: np.ndarray, costs: CostScenario) -> np.ndarray:
    """Expected saving from reviewing instead of the cheaper of approve and decline."""
    e = expected_costs(q, cents, costs)
    return np.minimum(e[APPROVE], e[DECLINE]) - e[REVIEW]


def fit_expected_policy(
    q: np.ndarray, cents: np.ndarray, capacity: float, costs: CostScenario
) -> float:
    """Smallest review gain that earns a review, so review share <= ``capacity``.

    Review only pays when its gain is positive; when capacity is short, the
    transactions with the largest gains are reviewed first.
    """
    gain = review_gain(q, cents, costs)
    k = int(len(q) * capacity)
    positive = np.sort(gain[gain > 0])[::-1]
    if k == 0 or positive.size == 0:
        return np.inf
    if positive.size > k:
        # Strictly above the first excluded gain, so ties cannot push the
        # review share over capacity on the data the policy was fitted on.
        return float(np.nextafter(positive[k], np.inf))
    return float(positive[-1])


def apply_expected_policy(
    q: np.ndarray, cents: np.ndarray, min_gain: float, costs: CostScenario
) -> np.ndarray:
    """Outcome per row: review if its gain clears ``min_gain``, else the cheaper of the others."""
    e = expected_costs(q, cents, costs)
    gain = np.minimum(e[APPROVE], e[DECLINE]) - e[REVIEW]
    fallback = np.where(e[DECLINE] < e[APPROVE], DECLINE, APPROVE)
    return np.where((gain > 0) & (gain >= min_gain), REVIEW, fallback)


# --- Reporting --------------------------------------------------------------


def outcome_summary(
    y: np.ndarray, cents: np.ndarray, outcome: np.ndarray, costs: CostScenario
) -> dict:
    """Volumes, fraud stopped, false declines, and cost of one policy on one range."""
    per_row = per_row_cost(y, cents, outcome, costs)
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
    frame: pd.DataFrame,
    outcome: np.ndarray,
    y: np.ndarray,
    cents: np.ndarray,
    costs: CostScenario,
    slices: int,
) -> list[dict]:
    """Realized review share and cost on consecutive equal-row time slices."""
    order = np.argsort(frame[TIME].to_numpy(), kind="stable")
    result = []
    for index, chunk in enumerate(np.array_split(order, slices)):
        summary = outcome_summary(y[chunk], cents[chunk], outcome[chunk], costs)
        result.append(
            {
                "slice": index + 1,
                "review_share": summary["review_share"],
                "decline_share": summary["decline_share"],
                "cost_per_1000_cents": summary["cost_per_1000_cents"],
            }
        )
    return result


def _no_inf(values: dict) -> dict:
    """JSON has no infinity: ``None`` means the outcome is never used."""
    return {k: None if isinstance(v, float) and np.isinf(v) else v for k, v in values.items()}


def run_decisions(
    valid: pd.DataFrame, valid_scores: dict, test: pd.DataFrame, test_scores: dict, params: Params
) -> dict:
    """Fit every policy on validation, per scenario and capacity; report on test."""
    settings = params.decision
    yv, cv = valid[LABEL].to_numpy(), amount_cents(valid)
    yt, ct = test[LABEL].to_numpy(), amount_cents(test)
    calibration = fit_calibration(valid_scores["lightgbm"], yv)
    qv = calibration.predict(valid_scores["lightgbm"])
    qt = calibration.predict(test_scores["lightgbm"])

    result = {"approve_all": {}, "policies": [], "comparisons": []}
    for scenario, costs in settings.scenarios.items():
        result["approve_all"][scenario] = outcome_summary(yt, ct, np.full(len(yt), APPROVE), costs)
        for capacity in settings.review_capacities:
            outcomes = {}
            for name in valid_scores:
                fitted = fit_thresholds(
                    valid_scores[name],
                    yv,
                    cv,
                    capacity,
                    costs,
                    settings.max_decline_rate,
                    settings.threshold_grid,
                )
                outcomes[name] = (
                    apply_thresholds(test_scores[name], fitted["t_review"], fitted["t_decline"]),
                    _no_inf(fitted),
                )
            min_gain = fit_expected_policy(qv, cv, capacity, costs)
            outcomes[EXPECTED] = (
                apply_expected_policy(qt, ct, min_gain, costs),
                _no_inf({"min_review_gain_cents": min_gain}),
            )
            for name, (outcome, fitted) in outcomes.items():
                policy = {
                    "scenario": scenario,
                    "capacity": capacity,
                    "model": name,
                    **fitted,
                    "test": outcome_summary(yt, ct, outcome, costs),
                }
                if scenario == BASE and name in ("lightgbm", EXPECTED):
                    policy["test_slices"] = slice_drift(
                        test, outcome, yt, ct, costs, params.evaluation.time_slices
                    )
                result["policies"].append(policy)
            if scenario == BASE:
                costs_by = {n: per_row_cost(yt, ct, o, costs) for n, (o, _) in outcomes.items()}
                pairs = [(EXPECTED, "lightgbm")] + [
                    ("lightgbm", other) for other in sorted(valid_scores) if other != "lightgbm"
                ]
                for better, than in pairs:
                    result["comparisons"].append(
                        {
                            "capacity": capacity,
                            "better": better,
                            "than": than,
                            **bootstrap_difference(
                                costs_by[better], costs_by[than], settings.bootstrap, params.seed
                            ),
                        }
                    )
    return result


def _dollars(cents: float) -> str:
    return f"${cents / 100:,.0f}"


def _threshold(value: float | None) -> str:
    return "never" if value is None else f"{value:.3f}"


def render(result: dict, params: Params) -> str:
    """Markdown report generated from ``decision.json``; do not edit by hand."""
    settings = params.decision
    base_costs = settings.scenarios[BASE]
    base_ref = result["approve_all"][BASE]
    lines = [
        "# Decision-level evaluation",
        "",
        "Generated by `make decide`; do not edit by hand. Every policy is fitted on the",
        "validation range and applied unchanged to the test range.",
        "",
        f"- Code: `{result['code']['commit']}`"
        + (" (dirty tree)" if result["code"]["dirty"] else ""),
        f"- Models params hash: `{result['models_params_hash'][:12]}`",
        "",
        "## Scenarios (assumptions, not measured costs)",
        "",
        "| Scenario | Review | Approved fraud loses | False decline loses "
        "| Approve-all cost / 1,000 |",
        "|---|---|---|---|---|",
    ]
    for name, c in settings.scenarios.items():
        lines.append(
            f"| {name} | {_dollars(c.review_cost_cents)} | {c.fraud_loss_rate:.0%} of amount | "
            f"{c.false_decline_rate:.0%} of amount + {_dollars(c.false_decline_fixed_cents)} | "
            f"{_dollars(result['approve_all'][name]['cost_per_1000_cents'])} |"
        )
    lines += [
        "",
        f"## Base scenario on test (review {_dollars(base_costs.review_cost_cents)})",
        "",
        f"`{EXPECTED}`: calibrated LightGBM probability x amount, cheapest expected outcome.",
        "",
        "| Capacity | Policy | t_review | t_decline | Review | Decline | Fraud $ stopped "
        "| False declines | Cost / 1,000 | Saving vs approve-all |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    bound_hits = []
    for p in result["policies"]:
        if p.get("decline_at_bound"):
            bound_hits.append(f"{p['scenario']} {p['capacity']:.1%} {p['model']}")
        if p["scenario"] != BASE:
            continue
        t = p["test"]
        saving = 1 - t["cost_cents"] / base_ref["cost_cents"]
        t_review = _threshold(p["t_review"]) if "t_review" in p else "by gain"
        t_decline = _threshold(p["t_decline"]) if "t_decline" in p else "by gain"
        lines.append(
            f"| {p['capacity']:.1%} | {p['model']} | {t_review} | {t_decline} | "
            f"{t['review_share']:.2%} | {t['decline_share']:.2%} | "
            f"{t['fraud_amount_stopped_share']:.1%} | {t['false_declines']:,} | "
            f"{_dollars(t['cost_per_1000_cents'])} | {saving:.1%} |"
        )
    lines += [
        "",
        "## Sensitivity: test cost per 1,000 transactions under each scenario",
        "",
        "| Scenario | Capacity | Approve all | baseline | mlp | lightgbm "
        f"| {EXPECTED} | Cheapest |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for scenario in settings.scenarios:
        for capacity in settings.review_capacities:
            row = {
                p["model"]: p["test"]["cost_per_1000_cents"]
                for p in result["policies"]
                if p["scenario"] == scenario and p["capacity"] == capacity
            }
            cheapest = min(row, key=row.get)
            lines.append(
                f"| {scenario} | {capacity:.1%} | "
                f"{_dollars(result['approve_all'][scenario]['cost_per_1000_cents'])} | "
                + " | ".join(_dollars(row[m]) for m in ("baseline", "mlp", "lightgbm", EXPECTED))
                + f" | {cheapest} |"
            )
    lines += [
        "",
        "## Are the savings real? (base scenario, bootstrap over test rows)",
        "",
        f"Cost saved per 1,000 test transactions, with a 95% interval from {settings.bootstrap} "
        "resamples. An interval above zero means the saving is unlikely to be noise.",
        "",
        "| Capacity | Policy | Compared with | Saving / 1,000 | 95% interval |",
        "|---|---|---|---|---|",
    ]
    for c in result["comparisons"]:
        lines.append(
            f"| {c['capacity']:.1%} | {c['better']} | {c['than']} | "
            f"{_dollars(c['mean_cents_per_1000'])} | {_dollars(c['low_cents_per_1000'])} to "
            f"{_dollars(c['high_cents_per_1000'])} |"
        )
    lines += [
        "",
        "## Drift: fixed policies on later test slices (base scenario)",
        "",
        "If the review share moves away from the planned capacity, the review team is",
        "over- or under-loaded even though the policy did not change.",
        "",
        "| Capacity | Policy | Slice | Review share | Decline share | Cost / 1,000 |",
        "|---|---|---|---|---|---|",
    ]
    for p in result["policies"]:
        for s in p.get("test_slices", []):
            lines.append(
                f"| {p['capacity']:.1%} | {p['model']} | {s['slice']} | {s['review_share']:.2%} | "
                f"{s['decline_share']:.2%} | {_dollars(s['cost_per_1000_cents'])} |"
            )
    if bound_hits:
        bound = "Policies that stopped at that bound (raise `max_decline_rate`): " + "; ".join(
            bound_hits
        )
    else:
        bound = "No policy reached the bound."
    lines += [
        "",
        "## Search bound",
        "",
        f"Threshold policies search declines up to {settings.max_decline_rate:.0%} of "
        f"transactions. {bound}",
        "",
    ]
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
    report.write_text(render(result, params))
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
