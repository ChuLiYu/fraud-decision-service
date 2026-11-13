"""Stage: error analysis of LightGBM on the test range (scope phase 1 acceptance).

Overall metrics hide where a model fails. This stage fixes one operating
point, then breaks the test range into segments and reports, per segment,
how much fraud is caught and missed and how many legitimate transactions are
flagged.

Operating point: the score threshold that sends ``REVIEW_RATE`` of the
**validation** range to review (a team that reviews 2% of traffic). It is
applied unchanged to the test range, so test labels never set it (tested).

Segments use fields whose values are categories, not identities: product
code, amount band, device type (missing when the transaction has no identity
record), and card network. Segments with fewer than ``MIN_ROWS`` rows are
folded into ``other`` so no reported group is small enough to describe
individual transactions (dataset card, rules 7B).

Outputs ``reports/errors/errors.json`` and ``errors.md``: aggregates only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_ml import manifest, metrics
from fraud_ml.config import Params, load_params
from fraud_ml.data import LABEL
from fraud_ml.evaluate import load_models, score
from fraud_ml.preprocess import UNKNOWN, Preprocessor
from fraud_ml.train import load_prepared

MODEL = "lightgbm"
REVIEW_RATE = 0.02
MIN_ROWS = 200
AMOUNT_BANDS = (0, 25, 100, 500, np.inf)
CATEGORY_FIELDS = {"ProductCD": "product", "DeviceType": "device type", "card4": "card network"}


def review_threshold(valid_scores: np.ndarray, rate: float = REVIEW_RATE) -> float:
    """Score above which the top ``rate`` share of the validation range lies."""
    return float(np.quantile(valid_scores, 1 - rate))


def decode(frame: pd.DataFrame, column: str, pre: Preprocessor) -> pd.Series:
    """Category names for an encoded column; unknown, rare, or missing become ``missing/rare``."""
    vocabulary = pre.categorical[column]
    codes = frame[column].to_numpy()
    names = np.array([*vocabulary, "missing/rare"], dtype=object)
    return pd.Series(names[np.where(codes == UNKNOWN, len(vocabulary), codes)], index=frame.index)


def amount_band(frame: pd.DataFrame) -> pd.Series:
    """Amount bands in dollars, labelled like ``$25-100``."""
    edges = AMOUNT_BANDS
    labels = [
        f"${edges[i]:g}-{edges[i + 1]:g}" if np.isfinite(edges[i + 1]) else f"${edges[i]:g}+"
        for i in range(len(edges) - 1)
    ]
    return pd.Series(
        pd.cut(frame["TransactionAmt"], bins=list(edges), labels=labels, right=False).astype(str),
        index=frame.index,
    )


def fold_small(groups: pd.Series, min_rows: int = MIN_ROWS) -> pd.Series:
    """Replace segments with fewer than ``min_rows`` rows by ``other``."""
    counts = groups.value_counts()
    small = counts[counts < min_rows].index
    return groups.where(~groups.isin(small), "other")


def segment_table(groups: pd.Series, y: np.ndarray, p: np.ndarray, threshold: float) -> list[dict]:
    """Catch and miss statistics per segment at one threshold, largest segments first."""
    flagged = p >= threshold
    fraud = y.astype(bool)
    total_fraud = int(fraud.sum())
    total_missed = int((fraud & ~flagged).sum())
    rows = []
    for name in groups.value_counts().index:
        mask = (groups == name).to_numpy()
        seg_fraud = fraud[mask]
        caught = int((seg_fraud & flagged[mask]).sum())
        missed = int((seg_fraud & ~flagged[mask]).sum())
        rows.append(
            {
                "segment": str(name),
                "rows": int(mask.sum()),
                "fraud_rate": float(seg_fraud.mean()),
                "fraud_share": float(seg_fraud.sum() / total_fraud) if total_fraud else 0.0,
                "recall": float(caught / seg_fraud.sum()) if seg_fraud.any() else float("nan"),
                "missed_share": float(missed / total_missed) if total_missed else 0.0,
                "flagged_legit": int((~seg_fraud & flagged[mask]).sum()),
                "pr_auc": metrics.ranking(y[mask], p[mask])["pr_auc"],
            }
        )
    return rows


def missed_by_amount(test: pd.DataFrame, y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """Share of missed fraud by count and by money: misses may be few but large."""
    fraud = y.astype(bool)
    missed = fraud & (p < threshold)
    cents = np.rint(test["TransactionAmt"].to_numpy() * 100).astype(np.int64)
    return {
        "missed_count_share": float(missed.sum() / fraud.sum()),
        "missed_amount_share": float(cents[missed].sum() / cents[fraud].sum()),
        "median_missed_amount": float(np.median(test["TransactionAmt"].to_numpy()[missed])),
        "median_caught_amount": float(
            np.median(test["TransactionAmt"].to_numpy()[fraud & ~missed])
        ),
    }


def run_errors(
    valid_scores: np.ndarray, test: pd.DataFrame, test_scores: np.ndarray, pre: Preprocessor
) -> dict:
    """Fix the threshold on validation, then analyse test errors by segment."""
    threshold = review_threshold(valid_scores)
    y = test[LABEL].to_numpy()
    segments = {"amount band": amount_band(test)}
    for column, label in CATEGORY_FIELDS.items():
        if column in pre.categorical:
            segments[label] = decode(test, column, pre)
    flagged = test_scores >= threshold
    return {
        "review_rate": REVIEW_RATE,
        "threshold": threshold,
        "test_review_share": float(flagged.mean()),
        "test_recall": float((flagged & y.astype(bool)).sum() / y.sum()),
        "missed": missed_by_amount(test, y, test_scores, threshold),
        "segments": {
            label: segment_table(fold_small(groups), y, test_scores, threshold)
            for label, groups in segments.items()
        },
    }


def render(result: dict) -> str:
    """Markdown report generated from ``errors.json``; do not edit by hand."""
    m = result["missed"]
    lines = [
        "# Error analysis: LightGBM on the test range",
        "",
        "Generated by `make errors`; do not edit by hand.",
        "",
        f"- Code: `{result['code']['commit']}`"
        + (" (dirty tree)" if result["code"]["dirty"] else ""),
        f"- Models params hash: `{result['models_params_hash'][:12]}`",
        f"- Operating point: threshold {result['threshold']:.4f}, set so validation sends "
        f"{result['review_rate']:.0%} to review; on test it sends "
        f"{result['test_review_share']:.2%} and catches {result['test_recall']:.1%} of fraud",
        f"- Segments under {MIN_ROWS} rows are folded into `other`",
        "",
        "## Missed fraud by count and by money",
        "",
        f"Missed: {m['missed_count_share']:.1%} of fraud transactions, "
        f"{m['missed_amount_share']:.1%} of fraud dollars. Median amount: missed "
        f"${m['median_missed_amount']:,.2f}, caught ${m['median_caught_amount']:,.2f}.",
        "",
    ]
    for label, rows in result["segments"].items():
        lines += [
            f"## By {label}",
            "",
            "| Segment | Rows | Fraud rate | Share of fraud | Recall | Share of misses "
            "| Legit flagged | PR-AUC |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            recall = "n/a" if np.isnan(r["recall"]) else f"{r['recall']:.1%}"
            pr_auc = "n/a" if np.isnan(r["pr_auc"]) else f"{r['pr_auc']:.3f}"
            lines.append(
                f"| {r['segment']} | {r['rows']:,} | {r['fraud_rate']:.2%} | "
                f"{r['fraud_share']:.1%} | {recall} | {r['missed_share']:.1%} | "
                f"{r['flagged_legit']:,} | {pr_auc} |"
            )
        lines.append("")
    return "\n".join(lines)


def errors(params: Params) -> Path:
    """Run the error analysis and return the report path."""
    pre, ranges = load_prepared(params, ("valid", "test"))
    models, trained = load_models(params)
    model = {MODEL: models[MODEL]}
    valid, test = ranges["valid"], ranges["test"]
    result = {
        "code": manifest.code_version(),
        "models_params_hash": trained["params_hash"],
        **run_errors(score(model, valid, pre)[MODEL], test, score(model, test, pre)[MODEL], pre),
    }
    out = params.data.reports_dir / "errors"
    out.mkdir(parents=True, exist_ok=True)
    (out / "errors.json").write_text(json.dumps(result, indent=2, default=_json_default))
    report = out / "errors.md"
    report.write_text(render(result))
    return report


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)}")


def main() -> None:
    """Command-line entry point: ``python -m fraud_ml.errors``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args()
    print(f"wrote {errors(load_params(args.params))}")


if __name__ == "__main__":
    main()
