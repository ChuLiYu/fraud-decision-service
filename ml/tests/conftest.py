"""Synthetic stand-ins for the raw tables.

Tests never use Competition Data: rows here are generated, so they can be
committed and shared (dataset card, terms of use).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


def make_transactions(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = np.sort(rng.integers(86_400, 86_400 * 30, size=n))
    times[10:13] = times[10]  # a run of tied timestamps
    return pd.DataFrame(
        {
            "TransactionID": np.arange(3_000_000, 3_000_000 + n),
            "TransactionDT": times,
            "isFraud": (rng.random(n) < 0.1).astype(int),
            "TransactionAmt": rng.gamma(2.0, 50.0, size=n).round(2),
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
        "data": {"raw_dir": "data/raw", "processed_dir": "data/processed"},
        "split": {"train_fraction": 0.7, "valid_fraction": 0.15},
        "preprocess": {"min_category_count": 2},
    }
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(params))
    return tmp_path
