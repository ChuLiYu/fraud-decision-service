"""Stage 8: presentation figures (scope S-19), drawn from committed reports only.

Every number comes from a report under ``reports/`` (``metrics.json``,
``tuning.json``, ``fusion.json``, ``decision.json``), never from the data, so
the figures contain aggregates only (dataset card) and can be redrawn on any
machine without the dataset or the models.

Style follows one rule set for every figure: each model keeps one color in
every chart (LightGBM blue, MLP orange, baseline aqua, LightGBM expected-cost
policy yellow), one y-axis per chart, thin marks, recessive grid, and the
values that carry the message labelled directly. Yellow and aqua sit below
3:1 contrast on the light surface, so bars carry value labels as well.

Run with ``make figures``; outputs go to ``reports/figures/*.png``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only; no display needed

import matplotlib.pyplot as plt  # noqa: E402

from fraud_ml.config import load_params  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#8a8984"  # neutral gray for reference lines, never a series color
COLORS = {
    "lightgbm": "#2a78d6",
    "mlp": "#eb6834",
    "baseline": "#1baf7a",
    "lightgbm_expected_cost": "#eda100",
}
LIGHT_BLUE = "#86b6ef"  # same hue as LightGBM: its validation-range variant
LABELS = {
    "lightgbm": "LightGBM",
    "mlp": "MLP",
    "baseline": "Logistic regression",
    "lightgbm_expected_cost": "LightGBM, expected cost",
}
RUN_LABELS = {
    "m2": "M2\nfirst model",
    "m3a-features": "M3a\n+features",
    "m3b-tuned": "M3b\n+tuning",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "font.size": 10,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "lines.markersize": 7,
        }
    )


def _save(fig: plt.Figure, out: Path, name: str, note: str) -> Path:
    fig.text(0.01, 0.01, note, color=INK_SECONDARY, fontsize=8, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = out / f"{name}.png"
    # No software metadata, so a rerun with the same inputs gives the same file.
    fig.savefig(path, dpi=200, metadata={"Software": None})
    plt.close(fig)
    return path


def _bar_labels(ax: plt.Axes, bars, fmt: str) -> None:
    for bar in bars:
        ax.annotate(
            fmt.format(bar.get_height()),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK,
        )


def progress(reports: Path, out: Path) -> Path:
    """PR-AUC per milestone: validation vs test for LightGBM, test for the baseline."""
    runs = [r for r in RUN_LABELS if (reports / "runs" / r / "metrics.json").exists()]
    data = [json.loads((reports / "runs" / r / "metrics.json").read_text()) for r in runs]
    x = range(len(runs))

    def series(range_name, model):
        return [d["ranges"][range_name]["models"][model]["pr_auc"] for d in data]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    lines = [
        (series("valid", "lightgbm"), LIGHT_BLUE, "LightGBM, validation", "--"),
        (series("test", "lightgbm"), COLORS["lightgbm"], "LightGBM, test", "-"),
        (series("test", "baseline"), COLORS["baseline"], "Logistic regression, test", "-"),
    ]
    for values, color, label, style in lines:
        ax.plot(x, values, style, color=color, marker="o", label=label)
        ax.annotate(
            f"{values[-1]:.3f}",
            (len(runs) - 1, values[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.set_xticks(list(x), [RUN_LABELS[r] for r in runs])
    ax.set_xlim(-0.3, len(runs) - 0.4)
    ax.set_ylim(0.3, 0.7)
    ax.set_ylabel("PR-AUC")
    ax.set_title("Each step helps a little; validation stays optimistic")
    ax.legend(loc="center right", bbox_to_anchor=(1, 0.45))
    return _save(fig, out, "progress", "Source: reports/runs/*/metrics.json")


def seed_noise(reports: Path, out: Path) -> Path:
    """Validation PR-AUC per seed for the old and the tuned setting."""
    study = json.loads((reports / "tuning" / "tuning.json").read_text())["seed_study"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    rows = [
        ("current", "Before tuning\n63 leaves, 100 per leaf"),
        ("best_grid", "Tuned\n127 leaves, 50 per leaf"),
    ]
    for y, (key, _) in enumerate(rows):
        entry = study[key]
        ax.scatter(
            entry["scores"], [y] * len(entry["scores"]), color=COLORS["lightgbm"], s=50, zorder=3
        )
        ax.plot([entry["mean"]] * 2, [y - 0.22, y + 0.22], color=INK, linewidth=2)
        ax.annotate(
            f"mean {entry['mean']:.3f}, sd {entry['stdev']:.3f}",
            (max(entry["scores"]), y),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.set_yticks([0, 1], [label for _, label in rows])
    ax.set_ylim(-0.6, 1.6)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Validation PR-AUC (one dot per random seed; bar = mean)")
    ax.set_title("Tuned setting scores higher and varies less across seeds")
    lo = min(min(study[k]["scores"]) for k, _ in rows)
    hi = max(max(study[k]["scores"]) for k, _ in rows)
    ax.set_xlim(lo - 0.01, hi + 0.03)
    return _save(fig, out, "seed_noise", "Source: reports/tuning/tuning.json (seeds 1, 2, 3)")


def time_drift(reports: Path, out: Path) -> Path:
    """LightGBM PR-AUC over consecutive time slices of validation and test."""
    data = json.loads((reports / "runs" / "m3b-tuned" / "metrics.json").read_text())
    values, labels = [], []
    for range_name in ("valid", "test"):
        for s in data["ranges"][range_name]["models"]["lightgbm"]["time_slices"]:
            values.append(s["pr_auc"])
            labels.append(f"{'V' if range_name == 'valid' else 'T'}{s['slice']}")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(range(len(values)), values, color=COLORS["lightgbm"], marker="o")
    split = len(values) / 2 - 0.5
    ax.axvline(split, color=REFERENCE, linewidth=1, linestyle=":")
    ax.text(split - 0.1, 0.84, "validation", ha="right", color=INK_SECONDARY, fontsize=9)
    ax.text(split + 0.1, 0.84, "test", ha="left", color=INK_SECONDARY, fontsize=9)
    for i in (0, len(values) - 1):
        ax.annotate(
            f"{values[i]:.2f}",
            (i, values[i]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.set_xticks(range(len(values)), labels)
    ax.set_ylim(0.4, 0.9)
    ax.set_ylabel("PR-AUC")
    ax.set_xlabel("Consecutive time slices with equal row counts")
    ax.set_title("Quality drops as data moves away from the training period")
    return _save(fig, out, "time_drift", "Source: reports/runs/m3b-tuned/metrics.json, LightGBM")


def review_tradeoff(reports: Path, out: Path) -> Path:
    """Share of fraud caught at each review rate, per model, on test."""
    data = json.loads((reports / "runs" / "m3c-mlp" / "metrics.json").read_text())
    models = data["ranges"]["test"]["models"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name in ("lightgbm", "mlp", "baseline"):
        rows = models[name]["at_review_rate"]
        rates = [r["review_rate"] * 100 for r in rows]
        recall = [r["recall"] for r in rows]
        ax.plot(rates, recall, color=COLORS[name], marker="o", label=LABELS[name])
        ax.annotate(
            f"{recall[-1]:.0%}",
            (rates[-1], recall[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    ax.set_xticks(rates, [f"{r:g}%" for r in rates])
    ax.set_ylim(0, 0.8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of transactions sent to review")
    ax.set_ylabel("Share of fraud caught (recall)")
    best = models["lightgbm"]["at_review_rate"][-1]
    ax.set_title(
        f"LightGBM catches {best['recall']:.0%} of fraud "
        f"when {best['review_rate']:.0%} of transactions are reviewed"
    )
    ax.legend(loc="upper left")
    return _save(
        fig, out, "review_tradeoff", "Source: reports/runs/m3c-mlp/metrics.json, test range"
    )


def fusion_diversity(reports: Path, out: Path) -> Path:
    """Frauds caught at a 1% review rate: by both models or only one."""
    d = json.loads((reports / "fusion" / "fusion.json").read_text())["test_diversity"]
    parts = [
        ("Both", d["caught_by_both"], REFERENCE),
        ("Only LightGBM", d["caught_only_by_lightgbm"], COLORS["lightgbm"]),
        ("Only MLP", d["caught_only_by_mlp"], COLORS["mlp"]),
    ]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(
        [p[0] for p in parts],
        [p[1] for p in parts],
        color=[p[2] for p in parts],
        width=0.55,
        edgecolor=SURFACE,
        linewidth=2,
    )
    _bar_labels(ax, bars, "{:,.0f}")
    ax.set_ylabel(f"Frauds caught (of {d['frauds']:,} in test)")
    ax.set_title("Few frauds are caught only by the MLP, so fusion did not help")
    return _save(
        fig,
        out,
        "fusion_diversity",
        f"Source: reports/fusion/fusion.json, test range, top 1% of scores; "
        f"rank correlation {d['spearman']:.2f}",
    )


def _policies(decision: dict, scenario: str) -> dict:
    by = {}
    for p in decision["policies"]:
        if p["scenario"] == scenario:
            by.setdefault(p["model"], {})[p["capacity"]] = p["test"]["cost_per_1000_cents"] / 100
    return by


def decision_cost(reports: Path, out: Path) -> Path:
    """Test cost per 1,000 transactions by review capacity, base scenario."""
    decision = json.loads((reports / "decision" / "decision.json").read_text())
    by = _policies(decision, "base")
    capacities = sorted(next(iter(by.values())))
    names = ["baseline", "mlp", "lightgbm", "lightgbm_expected_cost"]
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for i, name in enumerate(names):
        xs = [c + (i - 1.5) * width for c in range(len(capacities))]
        bars = ax.bar(
            xs,
            [by[name][c] for c in capacities],
            width=width,
            color=COLORS[name],
            label=LABELS[name],
            edgecolor=SURFACE,
            linewidth=2,
        )
        _bar_labels(ax, bars, "${:,.0f}")
    approve_all = decision["approve_all"]["base"]["cost_per_1000_cents"] / 100
    ax.axhline(approve_all, color=REFERENCE, linestyle="--", linewidth=1.5)
    ax.text(
        len(capacities) - 0.5,
        approve_all,
        f"approve everything ${approve_all:,.0f}",
        ha="right",
        va="bottom",
        color=INK_SECONDARY,
        fontsize=9,
    )
    ax.set_xticks(range(len(capacities)), [f"{c:.1%}" for c in capacities])
    ax.set_xlabel("Review capacity (share of transactions the team can review)")
    ax.set_ylabel("Cost per 1,000 test transactions ($)")
    ax.set_ylim(0, approve_all * 1.32)
    ax.set_title("Deciding with probability x amount cuts cost the most")
    ax.legend(loc="upper left", ncols=4, fontsize=8.5)
    return _save(
        fig,
        out,
        "decision_cost",
        "Source: reports/decision/decision.json, base scenario (assumed costs)",
    )


def _capacity(decision: dict, preferred: float = 0.02) -> float:
    """The review capacity to show: 2% if it was evaluated, else the middle one."""
    capacities = sorted({p["capacity"] for p in decision["policies"]})
    return preferred if preferred in capacities else capacities[len(capacities) // 2]


def sensitivity(reports: Path, out: Path) -> Path:
    """Saving versus approve-all under each cost scenario, at one capacity."""
    decision = json.loads((reports / "decision" / "decision.json").read_text())
    capacity = _capacity(decision)
    scenarios = list(decision["approve_all"])
    names = ["baseline", "mlp", "lightgbm", "lightgbm_expected_cost"]
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, name in enumerate(names):
        savings = []
        for scenario in scenarios:
            reference = decision["approve_all"][scenario]["cost_per_1000_cents"]
            savings.append(1 - _policies(decision, scenario)[name][capacity] * 100 / reference)
        xs = [s + (i - 1.5) * width for s in range(len(scenarios))]
        bars = ax.bar(
            xs,
            savings,
            width=width,
            color=COLORS[name],
            label=LABELS[name],
            edgecolor=SURFACE,
            linewidth=2,
        )
        _bar_labels(ax, bars, "{:.0%}")
    ax.set_xticks(range(len(scenarios)), [s.replace("_", " ") for s in scenarios])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Cost saved vs approving everything")
    ax.set_title(f"Same ranking under every cost scenario ({capacity:.0%} review capacity)")
    ax.legend(loc="upper left", ncols=2)
    return _save(
        fig,
        out,
        "sensitivity",
        "Source: reports/decision/decision.json; scenario costs are assumptions",
    )


def review_drift(reports: Path, out: Path) -> Path:
    """Realized review share on later test slices with a fixed policy."""
    decision = json.loads((reports / "decision" / "decision.json").read_text())
    capacity = _capacity(decision)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for p in decision["policies"]:
        if p["scenario"] != "base" or p["capacity"] != capacity or "test_slices" not in p:
            continue
        shares = [s["review_share"] for s in p["test_slices"]]
        xs = [s["slice"] for s in p["test_slices"]]
        ax.plot(xs, shares, color=COLORS[p["model"]], marker="o", label=LABELS[p["model"]])
    ax.axhline(capacity, color=REFERENCE, linestyle="--", linewidth=1.5)
    ax.text(
        1,
        capacity,
        f"planned capacity {capacity:.0%}",
        va="bottom",
        color=INK_SECONDARY,
        fontsize=9,
    )
    ax.set_xticks(xs, [f"T{x}" for x in xs])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax.set_ylim(capacity * 0.8, capacity * 1.35)
    ax.set_xlabel("Consecutive test time slices")
    ax.set_ylabel("Transactions sent to review")
    ax.set_title("Fixed policies drift above the planned review capacity")
    ax.legend(loc="lower right")
    return _save(fig, out, "review_drift", "Source: reports/decision/decision.json, base scenario")


FIGURES = (
    progress,
    seed_noise,
    time_drift,
    review_tradeoff,
    fusion_diversity,
    decision_cost,
    sensitivity,
    review_drift,
)


def figures(reports: Path) -> list[Path]:
    """Draw every figure from the reports under ``reports`` and return their paths."""
    _style()
    out = reports / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return [draw(reports, out) for draw in FIGURES]


def main() -> None:
    """Command-line entry point: ``python -m fraud_ml.figures``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args()
    for path in figures(load_params(args.params).data.reports_dir):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
