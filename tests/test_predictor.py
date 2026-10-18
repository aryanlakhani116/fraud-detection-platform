# tests/test_predictor.py
import pandas as pd
import pytest

from src.data.preprocess import TabularPreprocessor
from src.inference.tabular_predictor import TabularFraudPredictor


class FakeXGBModel:
    """A tiny stand-in that mimics XGBoost's interface without needing a real trained model."""
    def predict_proba(self, X):
        # always returns a fixed, predictable probability — perfect for testing
        import numpy as np
        return np.array([[0.8, 0.2]] * len(X))


@pytest.fixture
def fitted_preprocessor():
    df = pd.DataFrame({
        "TransactionAmt": [50.0, 75.0, 200.0, 30.0],
        "ProductCD": ["W", "C", "W", "R"],
    })
    pre = TabularPreprocessor()
    pre.fit(df)
    return pre


def test_predict_xgboost_returns_expected_shape(fitted_preprocessor):
    predictor = TabularFraudPredictor.__new__(TabularFraudPredictor)  # skip __init__
    predictor.preprocessor = fitted_preprocessor
    predictor.xgb_model = FakeXGBModel()

    result = predictor.predict_xgboost({"TransactionAmt": 100.0, "ProductCD": "W"})

    assert "fraud_probability" in result
    assert "risk_bucket" in result
    assert "model_version" in result
    assert result["fraud_probability"] == 0.2
    assert result["risk_bucket"] == "low"