from fraud_ml.config import load_params
from fraud_ml.decide import decide
from fraud_ml.evaluate import evaluate
from fraud_ml.figures import FIGURES, figures
from fraud_ml.fusion import fuse
from fraud_ml.prepare import prepare
from fraud_ml.train import train
from fraud_ml.tune import tune


def test_figures_draw_from_reports_only(workspace):
    params = load_params(workspace / "params.yaml")
    prepare(params)
    train(params)
    for run in ("m2", "m3b-tuned", "m3c-mlp"):
        evaluate(params, run)
    tune(params)
    fuse(params)
    decide(params)

    paths = figures(params.data.reports_dir)
    assert len(paths) == len(FIGURES)
    assert all(path.suffix == ".png" and path.stat().st_size > 0 for path in paths)
