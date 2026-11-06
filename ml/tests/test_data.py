import pytest

from fraud_ml.data import DataValidationError, load_raw, validate


def test_left_join_keeps_transactions_without_identity(workspace, raw_tables):
    transactions, identity = raw_tables
    df = load_raw(workspace / "data" / "raw")
    assert len(df) == len(transactions)
    assert df["DeviceType"].isna().sum() == len(transactions) - len(identity)
    validate(df)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda df: df.drop(columns="isFraud"),
        lambda df: df.assign(TransactionID=1),
        lambda df: df.assign(TransactionDT=-1),
        lambda df: df.assign(isFraud=2),
        lambda df: df.iloc[0:0],
    ],
    ids=["missing-label", "duplicate-id", "negative-time", "bad-label", "empty"],
)
def test_validate_rejects_broken_data(raw_tables, corrupt):
    with pytest.raises(DataValidationError):
        validate(corrupt(raw_tables[0]))
