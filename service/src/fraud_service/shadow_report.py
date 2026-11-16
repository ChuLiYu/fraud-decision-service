"""Summarise stored active-versus-shadow score comparisons as Markdown.

Run: ``python -m fraud_service.shadow_report`` with ``DATABASE_URL`` set.
"""

from __future__ import annotations

import os

from psycopg import Connection

_STATUSES = ("scored", "shadow_error", "invalid_score", "no_request")


def summary(conn: Connection) -> dict:
    """Return JSON-compatible aggregates grouped by active and shadow model versions."""
    pairs = conn.execute(
        "SELECT DISTINCT active_model_version, shadow_model_version FROM shadow_scores "
        "ORDER BY active_model_version, shadow_model_version"
    ).fetchall()
    report_pairs = []
    for active_model, shadow_model in pairs:
        status_counts = dict.fromkeys(_STATUSES, 0)
        for status, count in conn.execute(
            "SELECT status, COUNT(*) FROM shadow_scores "
            "WHERE active_model_version = %s AND shadow_model_version = %s GROUP BY status",
            (active_model, shadow_model),
        ):
            status_counts[status] = count
        matrix: dict[str, dict[str, int]] = {}
        matching = 0
        outcomes = conn.execute(
            "SELECT active_outcome, shadow_outcome, COUNT(*) FROM shadow_scores "
            "WHERE active_model_version = %s AND shadow_model_version = %s "
            "AND active_outcome IS NOT NULL AND shadow_outcome IS NOT NULL "
            "GROUP BY active_outcome, shadow_outcome ORDER BY active_outcome, shadow_outcome",
            (active_model, shadow_model),
        ).fetchall()
        outcome_rows = 0
        for active_outcome, shadow_outcome, count in outcomes:
            matrix.setdefault(active_outcome, {})[shadow_outcome] = count
            outcome_rows += count
            if active_outcome == shadow_outcome:
                matching += count
        score_rows, mean_abs, max_abs = conn.execute(
            "SELECT COUNT(*), AVG(ABS(shadow_score - active_score)), "
            "MAX(ABS(shadow_score - active_score)) FROM shadow_scores "
            "WHERE active_model_version = %s AND shadow_model_version = %s "
            "AND active_score IS NOT NULL AND shadow_score IS NOT NULL",
            (active_model, shadow_model),
        ).fetchone()
        report_pairs.append(
            {
                "active_model_version": active_model,
                "shadow_model_version": shadow_model,
                "status_counts": status_counts,
                "outcome_agreement": {
                    "matching": matching,
                    "rows": outcome_rows,
                    "rate": matching / outcome_rows if outcome_rows else None,
                },
                "confusion_matrix": matrix,
                "score_difference": {
                    "rows": score_rows,
                    "mean_abs": float(mean_abs) if mean_abs is not None else None,
                    "max_abs": float(max_abs) if max_abs is not None else None,
                },
            }
        )
    return {"pairs": report_pairs}


def markdown(report: dict) -> str:
    """Render a summary returned by :func:`summary` as Markdown."""
    lines = ["# Shadow scoring summary"]
    for pair in report["pairs"]:
        lines.extend(
            [
                "",
                f"## `{pair['active_model_version']}` → `{pair['shadow_model_version']}`",
                "",
                "| Status | Rows |",
                "| --- | ---: |",
            ]
        )
        lines.extend(f"| {status} | {count} |" for status, count in pair["status_counts"].items())
        agreement = pair["outcome_agreement"]
        rate = "n/a" if agreement["rate"] is None else f"{agreement['rate']:.2%}"
        lines.extend(
            [
                "",
                f"Outcome agreement: {agreement['matching']} / {agreement['rows']} ({rate})",
                "",
            ]
        )
        matrix = pair["confusion_matrix"]
        shadow_outcomes = sorted({outcome for counts in matrix.values() for outcome in counts})
        if shadow_outcomes:
            lines.append("| Active \\ shadow | " + " | ".join(shadow_outcomes) + " |")
            lines.append("| --- | " + " | ".join("---:" for _ in shadow_outcomes) + " |")
            for active_outcome, shadow_counts in matrix.items():
                values = " | ".join(
                    str(shadow_counts.get(outcome, 0)) for outcome in shadow_outcomes
                )
                lines.append(f"| {active_outcome} | {values} |")
        else:
            lines.append("No rows have both active and shadow outcomes.")
        difference = pair["score_difference"]
        mean_abs = "n/a" if difference["mean_abs"] is None else f"{difference['mean_abs']:.6f}"
        max_abs = "n/a" if difference["max_abs"] is None else f"{difference['max_abs']:.6f}"
        lines.append(
            f"Mean / max |shadow score − active score|: {mean_abs} / {max_abs} "
            f"over {difference['rows']} rows"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Entry point for ``python -m fraud_service.shadow_report``."""
    with Connection.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        print(markdown(summary(conn)), end="")


if __name__ == "__main__":
    main()
