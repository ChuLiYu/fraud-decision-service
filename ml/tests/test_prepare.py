import json

import pandas as pd
import pytest

from fraud_ml import manifest
from fraud_ml.config import load_params
from fraud_ml.prepare import params_hash, prepare


@pytest.fixture
def prepared(workspace):
    params = load_params(workspace / "params.yaml")
    prepare(params)
    return params


def test_writes_all_artifacts(prepared):
    out = prepared.data.processed_dir
    for name in ("train.parquet", "valid.parquet", "test.parquet", "preprocessor.json"):
        assert (out / name).exists()
    recorded = manifest.verify(out, expected_params_hash=params_hash(prepared))
    assert set(recorded["split"]["ranges"]) == {"train", "valid", "test"}
    assert recorded["inputs"].keys() == {"train_transaction.csv", "train_identity.csv"}


def test_outputs_respect_split_boundaries(prepared):
    out = prepared.data.processed_dir
    recorded = json.loads((out / "manifest.json").read_text())
    train = pd.read_parquet(out / "train.parquet")
    test = pd.read_parquet(out / "test.parquet")
    assert train["TransactionDT"].max() < recorded["split"]["valid_start"]
    assert test["TransactionDT"].min() >= recorded["split"]["test_start"]


def test_verify_rejects_changed_output(prepared):
    out = prepared.data.processed_dir
    frame = pd.read_parquet(out / "valid.parquet")
    frame.iloc[:5].to_parquet(out / "valid.parquet", index=False)
    with pytest.raises(manifest.ManifestMismatchError, match="valid.parquet"):
        manifest.verify(out, expected_params_hash=params_hash(prepared))


def test_verify_rejects_different_params(prepared):
    with pytest.raises(manifest.ManifestMismatchError, match="parameters"):
        manifest.verify(prepared.data.processed_dir, expected_params_hash="0" * 64)


def test_verify_requires_manifest(tmp_path):
    with pytest.raises(manifest.ManifestMismatchError, match="run prepare"):
        manifest.verify(tmp_path, expected_params_hash="0" * 64)
