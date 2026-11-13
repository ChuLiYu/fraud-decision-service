"""Stage 6: offline fusion ablation (scope S-18).

Question: does combining LightGBM with a model that makes different errors
(the MLP) rank fraud better than LightGBM alone?

Protocol, fixed before the run (fusion proposal; dataset card leakage rules):

1. The base models are trained on the train range only (``make train``).
2. The validation range is split by time into two halves. Everything a
   fusion method learns (blend weight, stacking coefficients, score
   reference distributions) is fitted on the **first half** only.
3. The fusion method is **chosen on the second half**, which no fitting
   step has seen.
4. The **test range** reports every method once. Fusion is adopted only if
   the chosen method beats LightGBM alone on test PR-AUC by at least
   ``fusion.min_gain`` *and* on every test time slice.

Methods compared:

* ``rank_mean``: average of each model's score percentile, measured
  against the first validation half. Needs no labels; robust to models
  whose probabilities are on different scales.
* ``blend``: ``sigmoid(w * logit(lightgbm) + (1 - w) * logit(mlp))`` with
  ``w`` chosen from ``fusion.blend_weights`` on the first half.
* ``stack``: a logistic regression meta-model on the logits of all three
  base models, fitted on the first half.

Limit: LightGBM and the MLP both early-stop on the whole validation range,
so their validation scores are slightly optimistic in both halves. The test
range is unaffected.

Outputs ``reports/fusion/fusion.json`` and ``fusion.md``: aggregates only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fraud_ml import manifest, metrics
from fraud_ml.config import Params, load_params
from fraud_ml.data import ID, LABEL, TIME
from fraud_ml.evaluate import load_models, score
from fraud_ml.split import time_boundary
from fraud_ml.train import load_prepared

FUSED = ("rank_mean", "blend", "stack")
REFERENCE = "lightgbm"
PARTNER = "mlp"
REVIEW_RATE = 0.01
_EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    """Log-odds, clipped so scores of exactly 0 or 1 stay finite."""
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def ecdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Share of ``reference`` scores at or below each value (a percentile in [0, 1]).

    Using a fixed reference (the first validation half) instead of ranking
    within the scored range keeps the transform usable for one request at a
    time, as a serving system would need.
    """
    ordered = np.sort(reference)
    return np.searchsorted(ordered, values, side="right") / len(ordered)


def split_halves(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks for the earlier and later half of a range, cut between timestamps."""
    ordered = frame.sort_values([TIME, ID], kind="stable")
    cut = time_boundary(ordered[TIME].reset_index(drop=True), len(ordered) // 2)
    first = (frame[TIME] < cut).to_numpy()
    if first.all() or not first.any():
        raise ValueError("validation range cannot be split into two time halves")
    return first, ~first


def fit(scores: dict[str, np.ndarray], y: np.ndarray, params: Params) -> dict:
    """Learn every fusion method from one set of labelled scores (the first validation half).

    Returns:
        Fitted state: score references, the blend weight, and the stacking model.
    """
    weights = params.fusion.blend_weights
    blend_scores = [metrics.ranking(y, _blend(scores, w))["pr_auc"] for w in weights]
    stack = LogisticRegression(random_state=params.seed).fit(_stack_inputs(scores), y)
    return {
        "references": {name: scores[name].copy() for name in (REFERENCE, PARTNER)},
        "blend_weight": float(weights[int(np.nanargmax(blend_scores))]),
        "stack": stack,
    }


def apply(fitted: dict, scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Fused scores for a range, using state learned by :func:`fit` only."""
    refs = fitted["references"]
    return {
        "rank_mean": (
            ecdf(refs[REFERENCE], scores[REFERENCE]) + ecdf(refs[PARTNER], scores[PARTNER])
        )
        / 2,
        "blend": _blend(scores, fitted["blend_weight"]),
        "stack": fitted["stack"].predict_proba(_stack_inputs(scores))[:, 1],
    }


def _blend(scores: dict[str, np.ndarray], weight: float) -> np.ndarray:
    z = weight * logit(scores[REFERENCE]) + (1 - weight) * logit(scores[PARTNER])
    return 1 / (1 + np.exp(-z))


def _stack_inputs(scores: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([logit(scores[name]) for name in sorted(scores)])


def verdict(test: dict, chosen: str, min_gain: float) -> dict:
    """Apply the adoption rule fixed in ``params.yaml`` before the run."""
    gain = test[chosen]["pr_auc"] - test[REFERENCE]["pr_auc"]
    slice_wins = [
        c["pr_auc"] > r["pr_auc"]
        for c, r in zip(test[chosen]["time_slices"], test[REFERENCE]["time_slices"], strict=True)
    ]
    return {
        "chosen": chosen,
        "test_gain": gain,
        "min_gain": min_gain,
        "slices_won": sum(slice_wins),
        "slices": len(slice_wins),
        "adopt": bool(gain >= min_gain and all(slice_wins)),
    }


def _summary(frame: pd.DataFrame, p: np.ndarray, params: Params) -> dict:
    y = frame[LABEL].to_numpy()
    return {
        **metrics.ranking(y, p),
        "recall_at_review_rate": metrics.at_review_rate(y, p, REVIEW_RATE)["recall"],
        "time_slices": metrics.time_slices(
            frame[TIME].to_numpy(), y, p, params.evaluation.time_slices
        ),
    }


def diversity(frame: pd.DataFrame, scores: dict[str, np.ndarray]) -> dict:
    """How differently LightGBM and the MLP rank the same rows.

    Fusion can only help if the models disagree on some frauds: low rank
    correlation and frauds caught by only one model at the review rate.
    """
    y = frame[LABEL].to_numpy().astype(bool)
    k = max(1, int(np.ceil(REVIEW_RATE * len(y))))
    flagged = {
        name: set(np.argsort(-scores[name], kind="stable")[:k]) for name in (REFERENCE, PARTNER)
    }
    frauds = set(np.flatnonzero(y))
    caught = {name: flagged[name] & frauds for name in flagged}
    return {
        "spearman": float(
            pd.Series(scores[REFERENCE]).corr(pd.Series(scores[PARTNER]), method="spearman")
        ),
        "frauds": len(frauds),
        "caught_by_both": len(caught[REFERENCE] & caught[PARTNER]),
        "caught_only_by_lightgbm": len(caught[REFERENCE] - caught[PARTNER]),
        "caught_only_by_mlp": len(caught[PARTNER] - caught[REFERENCE]),
    }


def run_ablation(
    valid: pd.DataFrame, valid_scores: dict, test: pd.DataFrame, test_scores: dict, params: Params
) -> dict:
    """Fit on the first validation half, choose on the second, report on test.

    Test labels are read only after every fitted value and the choice are
    fixed; the leakage test checks that changing them changes neither.
    """
    first, second = split_halves(valid)
    fitted = fit(
        {n: s[first] for n, s in valid_scores.items()}, valid[LABEL].to_numpy()[first], params
    )

    later = valid[second]
    later_scores = {n: s[second] for n, s in valid_scores.items()}
    later_all = {**later_scores, **apply(fitted, later_scores)}
    selection = {name: _summary(later, p, params) for name, p in later_all.items()}
    chosen = max(FUSED, key=lambda name: selection[name]["pr_auc"])

    test_all = {**test_scores, **apply(fitted, test_scores)}
    test_summary = {name: _summary(test, p, params) for name, p in test_all.items()}
    return {
        "fitted": {
            "fit_rows": int(first.sum()),
            "select_rows": int(second.sum()),
            "blend_weight": fitted["blend_weight"],
            "stack_coefficients": dict(
                zip(sorted(valid_scores), fitted["stack"].coef_[0].round(6).tolist(), strict=True)
            ),
            "stack_intercept": round(float(fitted["stack"].intercept_[0]), 6),
            "chosen": chosen,
        },
        "selection": selection,
        "test": test_summary,
        "diversity": {"valid_second_half": diversity(later, later_scores)},
        "test_diversity": diversity(test, test_scores),
        "verdict": verdict(test_summary, chosen, params.fusion.min_gain),
    }


def render(result: dict) -> str:
    """Markdown report generated from ``fusion.json``; do not edit by hand."""
    fitted, v = result["fitted"], result["verdict"]
    lines = [
        "# Fusion ablation",
        "",
        "Generated by `make fuse`; do not edit by hand. Fitted on the first half of the",
        "validation range, chosen on the second half, reported once on test.",
        "",
        f"- Code: `{result['code']['commit']}`"
        + (" (dirty tree)" if result["code"]["dirty"] else ""),
        f"- Models params hash: `{result['models_params_hash'][:12]}`",
        f"- Rows: fit {fitted['fit_rows']:,}, select {fitted['select_rows']:,}",
        f"- Blend weight on LightGBM: {fitted['blend_weight']}",
        "- Stacking coefficients (logits): "
        + ", ".join(f"{k} {c}" for k, c in fitted["stack_coefficients"].items())
        + f"; intercept {fitted['stack_intercept']}",
        "",
        "## Choice on the second validation half",
        "",
        "| Method | PR-AUC | ROC-AUC | Recall @1% |",
        "|---|---|---|---|",
    ]
    for name, m in result["selection"].items():
        mark = " (chosen)" if name == fitted["chosen"] else ""
        lines.append(
            f"| {name}{mark} | {m['pr_auc']:.4f} | {m['roc_auc']:.4f} | "
            f"{m['recall_at_review_rate']:.3f} |"
        )
    lines += [
        "",
        "## Test range",
        "",
        "| Method | PR-AUC | ROC-AUC | Recall @1% | PR-AUC by time slice |",
        "|---|---|---|---|---|",
    ]
    for name, m in result["test"].items():
        slices = ", ".join(f"{s['pr_auc']:.3f}" for s in m["time_slices"])
        lines.append(
            f"| {name} | {m['pr_auc']:.4f} | {m['roc_auc']:.4f} | "
            f"{m['recall_at_review_rate']:.3f} | {slices} |"
        )
    lines += [
        "",
        "## Do the models make different errors?",
        "",
        "| Range | Spearman (LightGBM, MLP) | Frauds | Caught by both @1% "
        "| Only LightGBM | Only MLP |",
        "|---|---|---|---|---|---|",
    ]
    for label, d in (
        ("valid, second half", result["diversity"]["valid_second_half"]),
        ("test", result["test_diversity"]),
    ):
        lines.append(
            f"| {label} | {d['spearman']:.3f} | {d['frauds']:,} | {d['caught_by_both']:,} | "
            f"{d['caught_only_by_lightgbm']:,} | {d['caught_only_by_mlp']:,} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        f"Rule fixed before the run: adopt fusion only if the chosen method gains at least "
        f"{v['min_gain']} test PR-AUC over LightGBM alone and wins every test time slice.",
        "",
        f"- Chosen: `{v['chosen']}`; test PR-AUC gain {v['test_gain']:+.4f}; "
        f"time slices won {v['slices_won']} of {v['slices']}",
        f"- **{'Adopt fusion' if v['adopt'] else 'Keep LightGBM alone'}**",
        "",
    ]
    return "\n".join(lines)


def fuse(params: Params) -> Path:
    """Run the fusion ablation and return the report path."""
    pre, ranges = load_prepared(params, ("valid", "test"))
    models, trained = load_models(params)
    valid, test = ranges["valid"], ranges["test"]
    result = {
        "code": manifest.code_version(),
        "models_params_hash": trained["params_hash"],
        **run_ablation(valid, score(models, valid, pre), test, score(models, test, pre), params),
    }
    out = params.data.reports_dir / "fusion"
    out.mkdir(parents=True, exist_ok=True)
    (out / "fusion.json").write_text(json.dumps(result, indent=2, default=_json_default))
    report = out / "fusion.md"
    report.write_text(render(result))
    return report


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)}")


def main() -> None:
    """Command-line entry point: ``python -m fraud_ml.fusion``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args()
    print(f"wrote {fuse(load_params(args.params))}")


if __name__ == "__main__":
    main()
