import pandas as pd
import pytest

from fraud_ml.preprocess import UNKNOWN, Preprocessor

TRAIN = pd.DataFrame(
    {
        "TransactionID": [1, 2, 3, 4, 5],
        "TransactionDT": [10, 20, 30, 40, 50],
        "isFraud": [0, 1, 0, 0, 1],
        "amount": [1.0, 2.0, None, 4.0, 5.0],
        "color": ["red", "red", "blue", "blue", "green"],
    }
)


def fitted(min_count: int = 2) -> Preprocessor:
    return Preprocessor(min_category_count=min_count).fit(TRAIN)


def test_identifiers_time_and_label_are_not_features():
    assert fitted().feature_names == ["amount", "color"]


def test_vocabulary_comes_from_train_only():
    # "purple" appears only outside the training range and must be unknown.
    later = TRAIN.assign(color=["purple", "red", None, "blue", "green"])
    codes = fitted().transform(later)["color"].tolist()
    assert codes[0] == UNKNOWN  # unseen in train
    assert codes[2] == UNKNOWN  # missing
    assert codes[4] == UNKNOWN  # rarer than min_category_count in train


def test_codes_are_deterministic():
    # Ties in count (red=2, blue=2) are broken by value, so blue gets code 0.
    assert fitted().categorical["color"] == ["blue", "red"]


def test_numeric_missing_values_are_kept():
    out = fitted().transform(TRAIN)
    assert out["amount"].dtype == "float32"
    assert out["amount"].isna().sum() == 1


def test_fit_does_not_look_at_transform_input():
    pre = fitted()
    before = dict(pre.categorical)
    pre.transform(TRAIN.assign(color="new"))
    assert pre.categorical == before


def test_save_load_round_trip(tmp_path):
    pre = fitted()
    pre.save(tmp_path / "p.json")
    loaded = Preprocessor.load(tmp_path / "p.json")
    pd.testing.assert_frame_equal(loaded.transform(TRAIN), pre.transform(TRAIN))


def test_missing_column_at_transform_fails():
    with pytest.raises(KeyError):
        fitted().transform(TRAIN.drop(columns="color"))
