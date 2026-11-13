"""Load and validate ``params.yaml``.

All tunable values live in ``params.yaml``; code reads them only through
:func:`load_params` so a typo or out-of-range value fails fast instead of
silently producing a different experiment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataParams:
    """Locations of raw and processed data, relative to the params file."""

    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class SplitParams:
    """Shares of rows for the train and validation time ranges."""

    train_fraction: float
    valid_fraction: float

    def __post_init__(self) -> None:
        """Reject fractions that would leave an empty or negative range."""
        if not (0 < self.train_fraction < 1 and 0 < self.valid_fraction < 1):
            raise ValueError("split fractions must be between 0 and 1")
        if self.train_fraction + self.valid_fraction >= 1:
            raise ValueError("train_fraction + valid_fraction must leave rows for test")


@dataclass(frozen=True)
class PreprocessParams:
    """Settings for fitting the shared preprocessing on the training range."""

    min_category_count: int

    def __post_init__(self) -> None:
        """Reject a non-positive minimum count."""
        if self.min_category_count < 1:
            raise ValueError("min_category_count must be at least 1")


@dataclass(frozen=True)
class FeatureParams:
    """Time-safe feature engineering settings (see fraud_ml.features)."""

    enabled: bool
    group_keys: list[list[str]]


@dataclass(frozen=True)
class BaselineParams:
    """Logistic regression settings."""

    C: float
    max_iter: int


@dataclass(frozen=True)
class LightGBMParams:
    """LightGBM settings; see docs/ml/tuning-guide.md for each one."""

    learning_rate: float
    num_leaves: int
    min_child_samples: int
    feature_fraction: float
    bagging_fraction: float
    bagging_freq: int
    lambda_l2: float
    max_rounds: int
    early_stopping_rounds: int


@dataclass(frozen=True)
class MLPParams:
    """Comparison neural network settings (see fraud_ml.train.fit_mlp)."""

    hidden_layer_sizes: list[int]
    alpha: float
    learning_rate_init: float
    batch_size: int
    max_epochs: int
    patience: int

    def __post_init__(self) -> None:
        """Reject empty layers and non-positive counts."""
        if not self.hidden_layer_sizes or min(self.hidden_layer_sizes) < 1:
            raise ValueError("hidden_layer_sizes must list positive layer widths")
        if self.batch_size < 1 or self.max_epochs < 1 or self.patience < 1:
            raise ValueError("batch_size, max_epochs and patience must be at least 1")


@dataclass(frozen=True)
class EvaluationParams:
    """How results are summarized."""

    review_rates: list[float]
    calibration_bins: int
    time_slices: int

    def __post_init__(self) -> None:
        """Reject review rates outside (0, 1) and non-positive counts."""
        if not all(0 < rate < 1 for rate in self.review_rates):
            raise ValueError("review_rates must be between 0 and 1")
        if self.calibration_bins < 2 or self.time_slices < 1:
            raise ValueError("calibration_bins must be >= 2 and time_slices >= 1")


@dataclass(frozen=True)
class TuningParams:
    """Validation-only tuning: a LightGBM grid and seeds for noise estimates."""

    grid: dict[str, list]
    seeds: list[int]

    def __post_init__(self) -> None:
        """Reject unknown LightGBM settings and empty lists early."""
        unknown = set(self.grid) - set(LightGBMParams.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown lightgbm settings in tuning.grid: {sorted(unknown)}")
        if not self.seeds or not all(self.grid.values()):
            raise ValueError("tuning.grid values and tuning.seeds must be non-empty")


@dataclass(frozen=True)
class FusionParams:
    """Offline fusion ablation (see fraud_ml.fusion)."""

    blend_weights: list[float]
    min_gain: float

    def __post_init__(self) -> None:
        """Reject weights outside [0, 1] and a negative required gain."""
        if not self.blend_weights or not all(0 <= w <= 1 for w in self.blend_weights):
            raise ValueError("blend_weights must be a non-empty list of values in [0, 1]")
        if self.min_gain < 0:
            raise ValueError("min_gain must not be negative")


@dataclass(frozen=True)
class DecisionParams:
    """Decision-level evaluation scenario (see fraud_ml.decide); assumptions, not measurements."""

    review_cost_cents: int
    fraud_loss_rate: float
    false_decline_rate: float
    review_capacities: list[float]
    max_decline_rate: float
    threshold_grid: int
    bootstrap: int

    def __post_init__(self) -> None:
        """Reject negative costs, shares outside (0, 1), and tiny grids."""
        if self.review_cost_cents < 0 or self.fraud_loss_rate < 0 or self.false_decline_rate < 0:
            raise ValueError("decision costs must not be negative")
        if not self.review_capacities or not all(0 < c < 1 for c in self.review_capacities):
            raise ValueError("review_capacities must be a non-empty list of shares in (0, 1)")
        if not 0 <= self.max_decline_rate < 1:
            raise ValueError("max_decline_rate must be in [0, 1)")
        if self.threshold_grid < 2 or self.bootstrap < 1:
            raise ValueError("threshold_grid must be >= 2 and bootstrap >= 1")


@dataclass(frozen=True)
class Params:
    """All pipeline parameters."""

    data: DataParams
    split: SplitParams
    preprocess: PreprocessParams
    features: FeatureParams
    seed: int
    baseline: BaselineParams
    lightgbm: LightGBMParams
    mlp: MLPParams
    evaluation: EvaluationParams
    tuning: TuningParams
    fusion: FusionParams
    decision: DecisionParams

    def stage_hash(self, *sections: str) -> str:
        """Return a stable SHA-256 of the named sections.

        Paths are excluded on purpose: moving the data directory must not
        invalidate artifacts, but changing a split fraction must.
        """
        payload = {}
        for name in sections:
            value = getattr(self, name)
            payload[name] = asdict(value) if hasattr(value, "__dataclass_fields__") else value
        text = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(text.encode()).hexdigest()


def load_params(path: Path) -> Params:
    """Read ``params.yaml`` and resolve data paths relative to its directory.

    Args:
        path: Location of the params file.

    Returns:
        Validated parameters.

    Raises:
        ValueError: If a value is out of range.
        KeyError: If a required key is missing.
    """
    raw = yaml.safe_load(path.read_text())
    base = path.resolve().parent
    return Params(
        data=DataParams(**{key: (base / value).resolve() for key, value in raw["data"].items()}),
        split=SplitParams(**raw["split"]),
        preprocess=PreprocessParams(**raw["preprocess"]),
        features=FeatureParams(**raw["features"]),
        seed=int(raw["seed"]),
        baseline=BaselineParams(**raw["baseline"]),
        lightgbm=LightGBMParams(**raw["lightgbm"]),
        mlp=MLPParams(**raw["mlp"]),
        evaluation=EvaluationParams(**raw["evaluation"]),
        tuning=TuningParams(**raw["tuning"]),
        fusion=FusionParams(**raw["fusion"]),
        decision=DecisionParams(**raw["decision"]),
    )
