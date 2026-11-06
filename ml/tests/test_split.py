import pandas as pd
import pytest

from fraud_ml.config import SplitParams
from fraud_ml.split import time_split

PARAMS = SplitParams(train_fraction=0.7, valid_fraction=0.15)


def test_ranges_are_consecutive_and_disjoint(raw_tables):
    split = time_split(raw_tables[0], PARAMS)
    assert split.train["TransactionDT"].max() < split.valid_start
    assert split.valid["TransactionDT"].min() >= split.valid_start
    assert split.valid["TransactionDT"].max() < split.test_start
    assert split.test["TransactionDT"].min() >= split.test_start
    assert len(split.train) + len(split.valid) + len(split.test) == len(raw_tables[0])


def test_tied_timestamps_never_span_two_ranges():
    # Rows 6-8 share timestamp 7, and a cut at row int(0.6 * 12) = 7 would split them.
    times = [1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 9, 10]
    df = pd.DataFrame({"TransactionID": range(12), "TransactionDT": times, "isFraud": 0})
    split = time_split(df, SplitParams(train_fraction=0.6, valid_fraction=0.25))
    assert split.valid_start == 8
    assert set(split.train["TransactionDT"]) == {1, 2, 3, 4, 5, 6, 7}


def test_ties_ordered_by_transaction_id():
    df = pd.DataFrame(
        {"TransactionID": [5, 3, 4, 1, 2], "TransactionDT": [1, 1, 1, 2, 3], "isFraud": 0}
    )
    split = time_split(df, SplitParams(train_fraction=0.6, valid_fraction=0.2))
    assert list(split.train["TransactionID"]) == [3, 4, 5]


def test_empty_range_rejected():
    df = pd.DataFrame({"TransactionID": range(4), "TransactionDT": [1, 1, 1, 1], "isFraud": 0})
    with pytest.raises(ValueError):
        time_split(df, PARAMS)
