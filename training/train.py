"""
Train an Isolation Forest anomaly detector on the credit card fraud dataset.

Key design decisions (worth knowing for interviews):
- Isolation Forest is UNSUPERVISED: we train mostly on normal data and never
  show it labels. Labels are used only for evaluation. This mirrors real
  fraud systems where labeled fraud is scarce and delayed (chargebacks take weeks).
- We standardize Time/Amount but not V1-V28 (already PCA-scaled in the real data).
- We convert the raw anomaly score into a calibrated 0-100 "risk score"
  via percentile ranking on a held-out normal sample, which is much easier
  to reason about and threshold than sklearn's raw decision_function.

Usage:
    python training/train.py --data data/transactions.csv --out models/
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def main(data_path: str, out_dir: str, contamination: float):
    df = pd.read_csv(data_path)
    X = df[FEATURES].copy()
    y = df["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # Scale Amount only — V1..V28 are PCA outputs, already centered/scaled.
    scaler = StandardScaler()
    X_train, X_test = X_train.values.copy(), X_test.values.copy()
    X_train[:, -1] = scaler.fit_transform(X_train[:, -1].reshape(-1, 1)).ravel()
    X_test[:, -1] = scaler.transform(X_test[:, -1].reshape(-1, 1)).ravel()

    model = IsolationForest(
        n_estimators=200,
        max_samples=256,          # classic IF setting; small subsamples isolate anomalies faster
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)

    # ---- calibration: map raw scores to 0-100 risk via normal-traffic percentiles
    normal_scores = -model.decision_function(X_train[y_train == 0])
    calibration_ref = np.sort(normal_scores)

    def risk_score(raw: np.ndarray) -> np.ndarray:
        """Percentile of raw anomaly score among normal training traffic (0-100)."""
        return np.searchsorted(calibration_ref, raw) / len(calibration_ref) * 100

    # ---- evaluation
    raw_test = -model.decision_function(X_test)
    risk_test = risk_score(raw_test)

    auroc = roc_auc_score(y_test, raw_test)
    ap = average_precision_score(y_test, raw_test)

    # pick an operating threshold: highest recall subject to precision >= 0.5
    prec, rec, thr = precision_recall_curve(y_test, risk_test)
    viable = np.where(prec[:-1] >= 0.5)[0]
    threshold = float(thr[viable[0]]) if len(viable) else 95.0

    metrics = {
        "auroc": round(float(auroc), 4),
        "average_precision": round(float(ap), 4),
        "suggested_risk_threshold": round(threshold, 2),
        "test_fraud_count": int(y_test.sum()),
        "test_size": int(len(y_test)),
    }
    print(json.dumps(metrics, indent=2))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "isolation_forest.joblib")
    joblib.dump(scaler, out / "amount_scaler.joblib")
    np.save(out / "calibration_ref.npy", calibration_ref)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model artifacts -> {out}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/transactions.csv")
    p.add_argument("--out", default="models/")
    p.add_argument("--contamination", type=float, default=0.002)
    args = p.parse_args()
    main(args.data, args.out, args.contamination)
