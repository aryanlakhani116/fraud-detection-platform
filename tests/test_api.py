# tests/test_api.py
from fastapi.testclient import TestClient
from app.main import app


class FakePredictor:
    def predict_xgboost(self, transaction):
        return {"fraud_probability": 0.15, "risk_bucket": "low", "model_version": "xgboost_v1"}

    def predict_ann(self, transaction):
        return {"fraud_probability": 0.65, "risk_bucket": "medium", "model_version": "ann_v1"}


def test_predict_endpoint_returns_valid_response():
    app.state.predictor = FakePredictor()  # skip the real lifespan loading entirely

    client = TestClient(app)
    response = client.post(
        "/predict/transaction/",
        json={"TransactionAmt": 100.0, "ProductCD": "W"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["fraud_probability"] == 0.15
    assert data["risk_bucket"] == "low"


def test_predict_endpoint_rejects_invalid_amount():
    app.state.predictor = FakePredictor()

    client = TestClient(app)
    response = client.post(
        "/predict/transaction/",
        json={"TransactionAmt": -50.0, "ProductCD": "W"},  # violates gt=0
    )

    assert response.status_code == 422