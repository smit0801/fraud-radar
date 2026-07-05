"""Training pipeline smoke test: artifacts exist and metrics are sane."""
import json


def test_training_produces_artifacts_and_sane_metrics(tmp_path):
    from data.generate_synthetic import make_dataset
    from training.train import main as train_main

    csv = tmp_path / "d.csv"
    make_dataset(3000, fraud_rate=0.01).to_csv(csv, index=False)
    out = tmp_path / "models"
    train_main(str(csv), str(out), contamination=0.01)

    for name in [
        "isolation_forest.joblib",
        "amount_scaler.joblib",
        "calibration_ref.npy",
        "metrics.json",
    ]:
        assert (out / name).exists(), f"missing artifact: {name}"

    metrics = json.loads((out / "metrics.json").read_text())
    assert {"auroc", "average_precision", "suggested_risk_threshold"} <= set(metrics)
    assert 0.5 < metrics["auroc"] <= 1.0  # must beat random on separable synthetic data
