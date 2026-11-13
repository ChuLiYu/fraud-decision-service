"""Synthetic stand-ins for the raw tables.

Tests never use Competition Data: rows here are generated, so they can be
committed and shared (dataset card, terms of use).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


def make_transactions(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = np.sort(rng.integers(86_400, 86_400 * 30, size=n))
    times[10:13] = times[10]  # a run of tied timestamps
    amount = rng.gamma(2.0, 50.0, size=n).round(2)
    # Large amounts are riskier, so models have a signal to learn.
    fraud = rng.random(n) < np.where(amount > 150, 0.3, 0.05)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(3_000_000, 3_000_000 + n),
            "TransactionDT": times,
            "isFraud": fraud.astype(int),
            "TransactionAmt": amount,
            "ProductCD": rng.choice(["W", "C", "R"], size=n),
            "card4": rng.choice(["visa", "mastercard", None], size=n),
        }
    )


def make_identity(transactions: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = transactions["TransactionID"].sample(frac=0.4, random_state=seed)
    return pd.DataFrame(
        {"TransactionID": ids, "DeviceType": rng.choice(["mobile", "desktop"], size=len(ids))}
    )


@pytest.fixture
def raw_tables():
    transactions = make_transactions()
    return transactions, make_identity(transactions)


@pytest.fixture
def workspace(tmp_path: Path, raw_tables) -> Path:
    """A params.yaml plus raw CSVs laid out like the real project."""
    transactions, identity = raw_tables
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    transactions.to_csv(raw / "train_transaction.csv", index=False)
    identity.to_csv(raw / "train_identity.csv", index=False)
    params = {
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "models_dir": "data/models",
            "reports_dir": "reports",
        },
        "split": {"train_fraction": 0.7, "valid_fraction": 0.15},
        "preprocess": {"min_category_count": 2},
        "features": {"enabled": True, "group_keys": [["card4"], ["card4", "ProductCD"]]},
        "seed": 0,
        "baseline": {"C": 1.0, "max_iter": 200},
        "lightgbm": {
            "learning_rate": 0.1,
            "num_leaves": 7,
            "min_child_samples": 5,
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
            "lambda_l2": 0.0,
            "max_rounds": 30,
            "early_stopping_rounds": 10,
        },
        "mlp": {
            "hidden_layer_sizes": [8],
            "alpha": 0.0001,
            "learning_rate_init": 0.01,
            "batch_size": 64,
            "max_epochs": 5,
            "patience": 2,
        },
        "evaluation": {"review_rates": [0.05, 0.1], "calibration_bins": 5, "time_slices": 2},
        "tuning": {"grid": {"num_leaves": [3, 7]}, "seeds": [1, 2]},
        "fusion": {"blend_weights": [0.0, 0.5, 1.0], "min_gain": 0.005},
        "decision": {
            "review_cost_cents": 500,
            "fraud_loss_rate": 1.0,
            "false_decline_rate": 0.1,
            "review_capacities": [0.05, 0.2],
            "max_decline_rate": 0.1,
            "threshold_grid": 20,
            "bootstrap": 20,
        },
    }
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(params))
    return tmp_path
