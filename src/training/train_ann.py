"""
Only job of this file: train the PyTorch ANN on tabular data and save it.
Run directly: python -m src.training.train_ann
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import average_precision_score, classification_report

from src.config import (
    TIME_COLUMN,
    TARGET_COLUMN,
    TEST_SIZE,
    ANN_HIDDEN_SIZES,
    ANN_DROPOUT,
    ANN_LEARNING_RATE,
    ANN_BATCH_SIZE,
    ANN_EPOCHS,
    ANN_MODEL_PATH,
)
from src.data.load_data import load_transaction_data
from src.data.preprocess import TabularPreprocessor, time_based_split
from src.training.ann_model import FraudANN


def to_dataloader(X, y, batch_size, shuffle):
    """Converts a prepared DataFrame + labels into a PyTorch DataLoader
    that hands out small batches, one at a time, during training/evaluation."""
    X_tensor = torch.tensor(X.values.astype(np.float32))
    y_tensor = torch.tensor(y.values.astype(np.float32))
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_ann():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- data ----
    df = load_transaction_data()
    train_df, test_df = time_based_split(df, TIME_COLUMN, TEST_SIZE)

    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]
    X_train_raw = train_df.drop(columns=[TARGET_COLUMN])
    X_test_raw = test_df.drop(columns=[TARGET_COLUMN])

    preprocessor = TabularPreprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    train_loader = to_dataloader(X_train, y_train, ANN_BATCH_SIZE, shuffle=True)

    # ---- model ----
    model = FraudANN(
        input_dim=X_train.shape[1],
        hidden_sizes=ANN_HIDDEN_SIZES,
        dropout=ANN_DROPOUT,
    ).to(device)

    # class imbalance handling: weight the rare class more heavily inside
    # the loss function itself (ANN equivalent of class_weight='balanced')
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=ANN_LEARNING_RATE)

    # ---- training loop ----
    model.train()
    for epoch in range(ANN_EPOCHS):
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)

        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{ANN_EPOCHS}  loss={epoch_loss:.4f}")

    # ---- evaluation ----
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test.values.astype(np.float32)).to(device)
        test_logits = model(X_test_tensor)
        test_proba = torch.sigmoid(test_logits).cpu().numpy()

    pr_auc = average_precision_score(y_test, test_proba)
    print(f"\nANN Test PR-AUC: {pr_auc:.4f}")
    print(classification_report(y_test, (test_proba >= 0.5).astype(int)))

    # ---- save ----
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": X_train.shape[1],
            "hidden_sizes": ANN_HIDDEN_SIZES,
            "dropout": ANN_DROPOUT,
        },
        ANN_MODEL_PATH,
    )
    preprocessor.save()
    print(f"Model saved to {ANN_MODEL_PATH}")

    return model, pr_auc


if __name__ == "__main__":
    train_ann()