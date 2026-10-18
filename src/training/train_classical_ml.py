import joblib
import xgboost as xgb
from sklearn.metrics import average_precision_score, classification_report

from src.config import TIME_COLUMN, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, DATA_PROCESSED_DIR
from src.data.load_data import load_transaction_data
from src.data.preprocess import TabularPreprocessor, time_based_split

XGB_MODEL_PATH = DATA_PROCESSED_DIR.parent.parent / "models" / "tabular" / "xgb_model.joblib"

def train_xgboost():
    df = load_transaction_data()
    train_df, test_df = time_based_split(df, TIME_COLUMN, TEST_SIZE)

    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]
    X_train_raw = train_df.drop(columns=[TARGET_COLUMN])
    X_test_raw = test_df.drop(columns=[TARGET_COLUMN])

    preprocessor = TabularPreprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, proba)
    print(f"XGBoost Test PR-AUC: {pr_auc:.4f}")
    print(classification_report(y_test, (proba >= 0.5).astype(int)))

    joblib.dump(model, XGB_MODEL_PATH)
    preprocessor.save()
    print(f"Model saved to {XGB_MODEL_PATH}")

    return model, pr_auc


if __name__ == "__main__":
    train_xgboost()