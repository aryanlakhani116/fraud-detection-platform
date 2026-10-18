import joblib
import numpy as np
import pandas as pd
import torch
from src.config import XGB_MODEL_PATH
from src.data.preprocess import TabularPreprocessor
from src.config import ANN_MODEL_PATH
from src.training.ann_model import FraudANN

import torch
from src.config import ANN_MODEL_PATH
from src.training.ann_model import FraudANN


class TabularFraudPredictor:
    def __init__(self):
        self.preprocessor = TabularPreprocessor.load()
        self.xgb_model = joblib.load(XGB_MODEL_PATH)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ann_model = self._load_ann()

    def _load_ann(self):
        checkpoint = torch.load(ANN_MODEL_PATH, map_location=self.device)

        model = FraudANN(
            input_dim=checkpoint["input_dim"],
            hidden_sizes=checkpoint["hidden_sizes"],
            dropout=checkpoint["dropout"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        return model
    
    def _prepare(self, transaction: dict) -> pd.DataFrame:
        df = pd.DataFrame([transaction])
        return self.preprocessor.transform(df)

    def predict_xgboost(self, transaction: dict) -> dict:
        X = self._prepare(transaction)
        proba = float(self.xgb_model.predict_proba(X)[0][1])
        return self._format_result(proba, "xgboost_v1")

    def predict_ann(self, transaction: dict) -> dict:
        X = self._prepare(transaction)
        X_tensor = torch.tensor(X.values.astype(np.float32)).to(self.device)

        with torch.no_grad():
            logit = self.ann_model(X_tensor)
            proba = float(torch.sigmoid(logit).cpu().numpy()[0])

        return self._format_result(proba, "ann_v1")

    @staticmethod
    def _format_result(proba: float, model_version: str) -> dict:
        bucket = "high" if proba > 0.7 else "medium" if proba > 0.3 else "low"
        return {
            "fraud_probability": round(proba, 4),
            "risk_bucket": bucket,
            "model_version": model_version,
        }

if __name__ == "__main__":
    predictor = TabularFraudPredictor()

    sample_transaction = {
        "TransactionAmt": 125.50,
        "ProductCD": "W",
        "card4": "visa",
        "card6": "credit",
        "P_emaildomain": "gmail.com",
    }

    print("XGBoost prediction:", predictor.predict_xgboost(sample_transaction))
    print("ANN prediction:", predictor.predict_ann(sample_transaction))