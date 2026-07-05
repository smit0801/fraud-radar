"""
Shared test fixtures.

The session-scoped `model_dir` fixture trains a small Isolation Forest on
synthetic data into a temp directory, so tests never depend on committed
model binaries or the real dataset. `client` points the API at it via env var.
"""
import os

import pytest


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    from data.generate_synthetic import make_dataset
    from training.train import main as train_main

    root = tmp_path_factory.mktemp("artifacts")
    csv = root / "tiny.csv"
    make_dataset(4000, fraud_rate=0.01).to_csv(csv, index=False)
    out = root / "models"
    train_main(str(csv), str(out), contamination=0.01)
    return out


@pytest.fixture(scope="session")
def client(model_dir):
    os.environ["FRAUD_RADAR_MODEL_DIR"] = str(model_dir)
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)
