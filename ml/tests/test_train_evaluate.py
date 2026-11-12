import json

import pandas as pd
import pytest

from fraud_ml import manifest
from fraud_ml.compare import compare
from fraud_ml.config import load_params
from fraud_ml.evaluate import evaluate
from fraud_ml.manifest import ManifestMismatchError
from fraud_ml.prepare import prepare
from fraud_ml.preprocess import Preprocessor
from fraud_ml.train import fit_baseline, train


@pytest.fixture
def trained(workspace):
    params = load_params(workspace / "params.yaml")
    prepare(params)
    train(params)
    return workspace, params


def test_evaluate_writes_report_for_both_models(trained):
    _, params = trained
    report = evaluate(params, "unit")
    result = json.loads((params.data.reports_dir / "runs" / "unit" / "metrics.json").read_text())
    assert set(result["ranges"]) == {"valid", "test"}
    for data in result["ranges"].values():
        assert set(data["models"]) == {"baseline", "lightgbm"}
        assert len(data["models"]["lightgbm"]["at_review_rate"]) == 2
    assert "| lightgbm |" in report.read_text()
    assert "| unit | lightgbm |" in compare(params.data.reports_dir).read_text()


def test_train_refuses_changed_prepared_data(workspace):
    params = load_params(workspace / "params.yaml")
    prepare(params)
    path = params.data.processed_dir / "train.parquet"
    pd.read_parquet(path).iloc[:50].to_parquet(path, index=False)
    with pytest.raises(ManifestMismatchError):
        train(params)


def test_evaluate_refuses_models_from_other_params(trained):
    workspace, _ = trained
    text = (
        (workspace / "params.yaml").read_text().replace("learning_rate: 0.1", "learning_rate: 0.2")
    )
    (workspace / "params.yaml").write_text(text)
    with pytest.raises(ManifestMismatchError, match="parameters"):
        evaluate(load_params(workspace / "params.yaml"), "unit")


def test_evaluate_refuses_models_trained_on_older_data(trained):
    workspace, params = trained
    # Change train.parquet and re-sign it in the prepare manifest, as a rerun of
    # prepare would; the models manifest still points at the old file.
    path = params.data.processed_dir / "train.parquet"
    pd.read_parquet(path).iloc[:-1].to_parquet(path, index=False)
    prepare_manifest = params.data.processed_dir / "manifest.json"
    data = json.loads(prepare_manifest.read_text())
    data["outputs"]["train.parquet"] = manifest.sha256_file(path)
    prepare_manifest.write_text(json.dumps(data))
    with pytest.raises(ManifestMismatchError, match="retrain"):
        evaluate(params, "unit")


def test_baseline_does_not_saturate_on_values_beyond_training_range(workspace):
    # Regression: on the real test range some features exceed the training
    # maximum by hundreds of standard deviations, which pushed the linear
    # baseline to p = 1.0 for legitimate transactions.
    params = load_params(workspace / "params.yaml")
    prepare(params)
    frame = pd.read_parquet(params.data.processed_dir / "train.parquet")
    pre = Preprocessor.load(params.data.processed_dir / "preprocessor.json")
    model = fit_baseline(frame, pre, params)

    at_max = frame.iloc[[0]].assign(TransactionAmt=frame["TransactionAmt"].max())
    far_beyond = at_max.assign(TransactionAmt=1e9)
    p_max = model.predict_proba(at_max[pre.feature_names])[0, 1]
    p_far = model.predict_proba(far_beyond[pre.feature_names])[0, 1]
    assert p_far == pytest.approx(p_max, abs=1e-6)
