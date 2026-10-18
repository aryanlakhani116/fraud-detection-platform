import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from src.config import RANDOM_STATE, DATA_PROCESSED_DIR

MAX_MISSING_PCT = 90
CONTINUOUS_COLS = ['TransactionAmt', 'C1', 'C2', 'C13', 'C14']
PREPROCESSOR_PATH = DATA_PROCESSED_DIR / 'preprocessor.joblib'

class TabularPreprocessor:
    def __init__(self):
        self.fitted_ = False
        self.cols_to_drop_ = None 
        self.numeric_medians_ = None 
        self.expected_columns_ = None
        self.scaler_ = StandardScaler()
        self.continuous_cols_present_ = None

    def fit(self, df:pd.DataFrame):
        missing_pct = df.isnull().mean() * 100
        self.cols_to_drop_ = missing_pct[missing_pct > MAX_MISSING_PCT].index.tolist()

        df_after_drop = df.drop(columns = self.cols_to_drop_, errors = 'ignore')
        numeric_cols = df_after_drop.select_dtypes(include = 'number').columns.tolist()
        self.numeric_medians_ = df_after_drop[numeric_cols].median().to_dict()

        # run the full cleaned sequence once, to see the final shape
        transformed_preview = self._clean_and_encode(df_after_drop)

        self.continuous_cols_present_ = [c for c in CONTINUOUS_COLS if c in transformed_preview.columns]
        self.scaler_.fit(transformed_preview[self.continuous_cols_present_])

        self.expected_columns_ = transformed_preview.columns.tolist()
        self.fitted_ = True
        return self

    def transform(self, df:pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError('Call .fit() before .transform().')

        df = df.drop(columns = self.cols_to_drop_, errors = 'ignore')
        df = self._clean_and_encode(df)
        df = df.reindex(columns = self.expected_columns_, fill_value = 0)

        df[self.continuous_cols_present_] = self.scaler_.transform(df[self.continuous_cols_present_])
        # df = df.iloc[:5000]
        return df

    def _clean_and_encode(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        shared logic used by BOTH fit() and transform() - fill gaps, then one-hot encode.
        """
        for col, median_val in self.numeric_medians_.items():
            if col in df.columns:
                df[col] = df[col].fillna(median_val)

        categorical_cols = df.select_dtypes(include = ['object','string']).columns.tolist()
        for col in categorical_cols:
            df[col] = df[col].fillna('missing')
        
        if categorical_cols:
            df = pd.get_dummies(df, columns = categorical_cols, drop_first = True)

        return df

    def save(self, path = None):
        path = path or PREPROCESSOR_PATH
        joblib.dump(self, path)
        return path


    def fit_transform(self, df:pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    @staticmethod
    def load(path = None):
        path = path or PREPROCESSOR_PATH
        return joblib.load(path)

def time_based_split(df: pd.DataFrame, time_col: str, test_size:float):
    df_sorted = df.sort_values(time_col)
    split_index = int(len(df_sorted) * (1-test_size))
    train_df = df_sorted.iloc[: split_index]
    test_df = df_sorted.iloc[split_index: ]
    return train_df, test_df

if __name__ == "__main__":
    from src.data.load_data import load_transaction_data

    df = load_transaction_data()
    train_slice = df.iloc[:400000]
    test_slice = df.iloc[400000:]

    pre = TabularPreprocessor()
    pre.fit(train_slice)
    original_result = pre.transform(test_slice)

    saved_path = pre.save()
    print("Saved to:", saved_path)

    reloaded_pre = TabularPreprocessor.load()
    reloaded_result = reloaded_pre.transform(test_slice)

    print("Results identical after reload:", original_result.equals(reloaded_result))