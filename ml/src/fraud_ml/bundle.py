"""Stage: build a serving bundle for the decision service (ADR-005).

A bundle is one directory the service loads at start-up:

* ``lightgbm.txt``: the serving model.
* ``preprocessor.json``: fitted preprocessing (same code as training).
* ``policy.json``: approve / review / decline score thresholds.
* ``bundle.json``: what the bundle is (``real`` or ``demo``), the raw fields a
  request may carry, feature settings, file hashes, and test metrics.

The serving model is trained **without history features**: they need past
transactions per card, and the MVP has no online feature store (scope S-9).
Request-time features (amount, hour, day offsets) stay. The gap to the full
model is visible by comparing ``bundle.json`` test metrics with the
evaluation reports.

Thresholds come from the decision evaluation method: cost-minimizing score
thresholds on the **validation** range, base cost scenario, review capacity
``REVIEW_CAPACITY``. Test is scored once, for the record.

Two sources:

* ``real``: the downloaded IEEE-CIS tables (local only; the bundle must not be
  committed or published, dataset card rules 7B).
* ``demo``: synthetic rows generated here, so a fresh clone can run the
  service without the dataset. Its scores mean nothing about real fraud.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_ml import metrics
from fraud_ml.config import Params, load_params
from fraud_ml.data import ID, LABEL, TIME, load_raw, validate
from fraud_ml.decide import BASE, amount_cents, fit_thresholds
from fraud_ml.features import add_features
from fraud_ml.preprocess import Preprocessor
from fraud_ml.split import time_split
from fraud_ml.train import fit_lightgbm

REVIEW_CAPACITY = 0.02
FORMAT_VERSION = 1
MODEL_FILE = "lightgbm.txt"
PREPROCESSOR_FILE = "preprocessor.json"
POLICY_FILE = "policy.json"
BUNDLE_FILE = "bundle.json"


def serving_params(params: Params) -> Params:
    """``params`` with history features switched off (they need past rows)."""
    features = dataclasses.replace(
        params.features, group_keys=[], client_key=[], client_means=[], client_nunique=[]
    )
    return dataclasses.replace(params, features=features)


def synthetic_transactions(n: int = 20_000, seed: int = 0) -> pd.DataFrame:
    """Generated rows with IEEE-CIS column names; no Competition Data.

    Fraud is made likelier for large amounts, product ``C``, mobile devices,
    and new cards, so the demo model learns something visible.
    """
    rng = np.random.default_rng(seed)
    times = np.sort(rng.integers(86_400, 86_400 * 120, size=n))
    amount = rng.gamma(2.0, 60.0, size=n).round(2)
    product = rng.choice(["W", "C", "R", "H", "S"], size=n, p=[0.7, 0.12, 0.07, 0.06, 0.05])
    device = rng.choice(["mobile", "desktop", None], size=n, p=[0.15, 0.15, 0.7])
    card_age = rng.integers(0, 400, size=n).astype(float)
    risk = (
        -5.0
        + 2.5 * (amount > 300)
        + 2.5 * (product == "C")
        + 1.5 * (device == "mobile")
        + 2.5 * (card_age < 10)
    )
    fraud = rng.random(n) < 1 / (1 + np.exp(-risk))
    return pd.DataFrame(
        {
            ID: np.arange(1, n + 1),
            TIME: times,
            LABEL: fraud.astype(int),
            "TransactionAmt": amount,
            "ProductCD": product,
            "card4": rng.choice(["visa", "mastercard", "discover", None], size=n),
            "card6": rng.choice(["debit", "credit"], size=n),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com", None], size=n),
            "D1": card_age,
            "C1": rng.integers(1, 6, size=n).astype(float),
            "DeviceType": device,
        }
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(raw: pd.DataFrame, params: Params, out: Path, kind: str) -> Path:
    """Train the serving model on ``raw`` and write a bundle to ``out``."""
    params = serving_params(params)
    validate(raw)
    raw_fields = [c for c in raw.columns if c not in (ID, LABEL)]
    df = add_features(raw, params.features)
    split = time_split(df, params.split)
    pre = Preprocessor(params.preprocess.min_category_count).fit(split.train)
    train, valid, test = (pre.transform(f) for f in (split.train, split.valid, split.test))
    booster = fit_lightgbm(train, valid, pre, params)

    def scores(frame: pd.DataFrame) -> np.ndarray:
        return booster.predict(frame[pre.feature_names], num_iteration=booster.best_iteration)

    p_valid, p_test = scores(valid), scores(test)
    costs = params.decision.scenarios[BASE]
    fitted = fit_thresholds(
        p_valid,
        valid[LABEL].to_numpy(),
        amount_cents(split.valid),
        REVIEW_CAPACITY,
        costs,
        params.decision.max_decline_rate,
        params.decision.threshold_grid,
    )

    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(out / MODEL_FILE, num_iteration=booster.best_iteration)
    pre.save(out / PREPROCESSOR_FILE)
    policy = {
        "name": f"thresholds-{BASE}-review{REVIEW_CAPACITY:g}",
        "t_review": fitted["t_review"],
        "t_decline": fitted["t_decline"],
        "review_capacity": REVIEW_CAPACITY,
        "scenario": BASE,
    }
    (out / POLICY_FILE).write_text(_dumps(policy))
    model_hash = _sha256(out / MODEL_FILE)
    info = {
        "format_version": FORMAT_VERSION,
        "kind": kind,
        "model_version": f"{kind}-{model_hash[:12]}",
        "policy_version": f"{policy['name']}-{_sha256(out / POLICY_FILE)[:8]}",
        "raw_fields": raw_fields,
        "features": dataclasses.asdict(params.features),
        "best_iteration": booster.best_iteration,
        "test": metrics.ranking(test[LABEL].to_numpy(), p_test),
        "rows": {"train": len(train), "valid": len(valid), "test": len(test)},
    }
    (out / BUNDLE_FILE).write_text(_dumps(info))
    return out / BUNDLE_FILE


def _plain(value: object) -> object:
    """JSON-safe copy: numpy scalars to Python, non-finite floats to ``None``.

    ``None`` for a threshold means "never" (for example no declines).
    """
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _dumps(value: dict) -> str:
    return json.dumps(_plain(value), indent=1, allow_nan=False)


def main() -> None:
    """Command-line entry point: ``python -m fraud_ml.bundle``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--demo", action="store_true", help="train on synthetic rows")
    args = parser.parse_args()
    params = load_params(args.params)
    if args.demo:
        raw, kind = synthetic_transactions(), "demo"
        # Synthetic rows carry only D1 among the day-offset columns.
        features = dataclasses.replace(params.features, day_offsets=["D1"])
        params = dataclasses.replace(params, features=features)
    else:
        raw, kind = load_raw(params.data.raw_dir), "real"
    print(f"wrote {build(raw, params, args.out, kind)}")


if __name__ == "__main__":
    main()
