from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent 
DATA_RAW_DIR = ROOT_DIR / 'data' / 'raw'
DATA_PROCESSED_DIR = ROOT_DIR / 'data' / 'processed'
TRANSACTION_CSV = DATA_RAW_DIR / 'train_transaction.csv'
TARGET_COLUMN = 'isFraud'
TIME_COLUMN = 'TransactionDT'
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODELS_TABULAR_DIR = ROOT_DIR / "models" / "tabular"
MODELS_TABULAR_DIR.mkdir(parents=True, exist_ok=True)
XGB_MODEL_PATH = MODELS_TABULAR_DIR / "xgb_model.joblib"

DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_TABULAR_DIR.mkdir(parents=True, exist_ok=True)


ANN_HIDDEN_SIZES = [128, 64, 32]
ANN_DROPOUT = 0.3
ANN_LEARNING_RATE = 1e-3
ANN_BATCH_SIZE = 512
ANN_EPOCHS = 15
ANN_MODEL_PATH = MODELS_TABULAR_DIR / "ann_model.pt"
# if __name__ == '__main__':
#     print('Root : ', ROOT_DIR)
#     print('Data File exists:', TRANSACTION_CSV.exists())

# print(ROOT_DIR)