import pandas as pd
from src.config import TRANSACTION_CSV

def load_transaction_data(path = None) -> pd.DataFrame:
    path = path or TRANSACTION_CSV

    if not path.exists():
        raise FileNotFoundError(
            f'Expected data at {path}, but it doesn"t exists.\n'
            f'Download train_transcation.csv from the IEEE-CIS Fraud Detection'
            f'Kaggle competition and place it there'
        )

    df = pd.read_csv(path)
    return df

if __name__ == '__main__':
    df = load_transaction_data()
    print(f'Loaded {df.shape[0]:,} rows and {df.shape[1]} columns')
    print(df.head(3))