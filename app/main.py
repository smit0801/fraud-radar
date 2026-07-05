"""
Fraud Radar — real-time transaction risk scoring API.

Endpoints:
    POST /score        -> score a single transaction
    POST /score/batch  -> score up to 1000 transactions in one call
    GET  /health       -> liveness + model metadata

Run:
    uvicorn app.main:app --reload --port 8000
"""
import os
import time
from pathlib import Path
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Override with FRAUD_RADAR_MODEL_DIR (used by tests and Docker)
MODEL_DIR = Path(os.environ.get("FRAUD_RADAR_MODEL_DIR", Path(__file__).resolve().parents[1] / "models"))
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

app = FastAPI(title="Fraud Radar", version="0.1.0")

# ---- load artifacts once at startup (cold-start cost paid once)
model = joblib.load(MODEL_DIR / "isolation_forest.joblib")
scaler = joblib.load(MODEL_DIR / "amount_scaler.joblib")
calibration_ref = np.load(MODEL_DIR / "calibration_ref.npy")

RISK_THRESHOLD_REVIEW = 90.0   # flag for manual review
RISK_THRESHOLD_BLOCK = 99.0    # auto-decline


class Transaction(BaseModel):
    transaction_id: str = Field(..., examples=["txn_00042"])
    features: List[float] = Field(
        ..., min_length=28, max_length=28, description="V1..V28 PCA components"
    )
    amount: float = Field(..., ge=0)


class ScoreResponse(BaseModel):
    transaction_id: str
    risk_score: float
    decision: str          # "approve" | "review" | "block"
    latency_ms: float


def _vectorize(txns: List[Transaction]) -> np.ndarray:
    X = np.array([t.features + [t.amount] for t in txns], dtype=float)
    # scale the Amount column (last col) with the training scaler
    X[:, -1] = scaler.transform(X[:, -1].reshape(-1, 1)).ravel()
    return X


def _score(X: np.ndarray) -> np.ndarray:
    raw = -model.decision_function(X)
    return np.searchsorted(calibration_ref, raw) / len(calibration_ref) * 100


def _decision(risk: float) -> str:
    if risk >= RISK_THRESHOLD_BLOCK:
        return "block"
    if risk >= RISK_THRESHOLD_REVIEW:
        return "review"
    return "approve"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "IsolationForest",
        "n_estimators": model.n_estimators,
        "thresholds": {"review": RISK_THRESHOLD_REVIEW, "block": RISK_THRESHOLD_BLOCK},
    }


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction):
    t0 = time.perf_counter()
    risk = float(_score(_vectorize([txn]))[0])
    return ScoreResponse(
        transaction_id=txn.transaction_id,
        risk_score=round(risk, 2),
        decision=_decision(risk),
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@app.post("/score/batch", response_model=List[ScoreResponse])
def score_batch(txns: List[Transaction]):
    if len(txns) > 1000:
        raise HTTPException(413, "Batch limit is 1000 transactions")
    t0 = time.perf_counter()
    risks = _score(_vectorize(txns))
    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    return [
        ScoreResponse(
            transaction_id=t.transaction_id,
            risk_score=round(float(r), 2),
            decision=_decision(float(r)),
            latency_ms=elapsed,
        )
        for t, r in zip(txns, risks)
    ]
