"""API contract tests for the scoring service."""


def _txn(txn_id, feats, amount):
    return {"transaction_id": txn_id, "features": feats, "amount": amount}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "IsolationForest"


def test_anomalous_scores_higher_than_normal(client):
    normal = _txn("n1", [0.0] * 28, 20.0)
    weird = [0.0] * 28
    for i in (2, 3, 9, 11, 13, 16):  # components the generator shifts for fraud
        weird[i] = -8.0
    fraud_like = _txn("f1", weird, 2500.0)

    rn = client.post("/score", json=normal).json()
    rf = client.post("/score", json=fraud_like).json()

    assert 0.0 <= rn["risk_score"] <= 100.0
    assert 0.0 <= rf["risk_score"] <= 100.0
    assert rf["risk_score"] > rn["risk_score"]
    assert rn["decision"] == "approve"
    assert rf["decision"] in {"review", "block"}


def test_rejects_wrong_feature_count(client):
    r = client.post("/score", json=_txn("x", [0.0] * 27, 5.0))
    assert r.status_code == 422


def test_rejects_negative_amount(client):
    r = client.post("/score", json=_txn("x", [0.0] * 28, -5.0))
    assert r.status_code == 422


def test_batch_roundtrip(client):
    batch = [_txn(f"t{i}", [0.0] * 28, 10.0) for i in range(5)]
    r = client.post("/score/batch", json=batch)
    assert r.status_code == 200
    results = r.json()
    assert len(results) == 5
    assert {x["transaction_id"] for x in results} == {f"t{i}" for i in range(5)}


def test_batch_size_limit(client):
    big = [_txn(f"b{i}", [0.0] * 28, 1.0) for i in range(1001)]
    assert client.post("/score/batch", json=big).status_code == 413
