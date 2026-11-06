"""Load the raw tables, join them, and validate the result.

Validation runs before anything is split or fitted, so a corrupted or
unexpected download stops the pipeline instead of producing a model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ID = "TransactionID"
TIME = "TransactionDT"
LABEL = "isFraud"


class DataValidationError(ValueError):
    """The raw data does not meet the expectations in the dataset card."""


def load_raw(raw_dir: Path) -> pd.DataFrame:
    """Read both training tables and left-join identity onto transactions.

    Not every transaction has an identity row; those rows keep missing
    identity columns rather than being dropped.

    Args:
        raw_dir: Directory containing ``train_transaction.csv`` and
            ``train_identity.csv``.

    Returns:
        One row per transaction.

    Raises:
        FileNotFoundError: If the raw files have not been downloaded.
    """
    for name in ("train_transaction.csv", "train_identity.csv"):
        if not (raw_dir / name).exists():
            raise FileNotFoundError(f"{raw_dir / name} not found; run `make fetch` first")
    transactions = pd.read_csv(raw_dir / "train_transaction.csv", engine="pyarrow")
    identity = pd.read_csv(raw_dir / "train_identity.csv", engine="pyarrow")
    joined = transactions.merge(identity, on=ID, how="left", validate="one_to_one")
    return joined


def validate(df: pd.DataFrame) -> None:
    """Check the invariants later stages rely on.

    Args:
        df: Joined table from :func:`load_raw`.

    Raises:
        DataValidationError: If a required column is missing, IDs repeat,
            time is missing or negative, or labels are not 0/1.
    """
    missing = {ID, TIME, LABEL} - set(df.columns)
    if missing:
        raise DataValidationError(f"missing required columns: {sorted(missing)}")
    if df.empty:
        raise DataValidationError("no rows")
    if df[ID].duplicated().any():
        raise DataValidationError(f"{ID} is not unique")
    if df[TIME].isna().any() or (df[TIME] < 0).any():
        raise DataValidationError(f"{TIME} has missing or negative values")
    if not df[LABEL].isin([0, 1]).all():
        raise DataValidationError(f"{LABEL} must be 0 or 1")
