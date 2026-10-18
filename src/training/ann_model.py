"""
The ANN architecture itself. Kept separate from the training LOOP (train_ann.py)
so the model definition can be imported by both training code and inference code
without dragging in training-only dependencies (optimizers, data loaders, etc.).
"""
import torch
import torch.nn as nn


class FraudANN(nn.Module):
    """
    A plain feedforward network for tabular fraud classification.
    input -> [Linear -> BatchNorm -> ReLU -> Dropout] x N -> single logit output
    """

    def __init__(self, input_dim: int, hidden_sizes=(128, 64, 32), dropout=0.3):
        super().__init__()

        layers = []
        prev_size = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # final layer: one raw logit (no sigmoid here — BCEWithLogitsLoss applies it
        # internally during training, which is numerically more stable)
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)  # shape (batch,) not (batch, 1)


if __name__ == "__main__":
    # quick plumbing check — not a real test, just confirms shapes flow correctly
    model = FraudANN(input_dim=20)
    dummy_input = torch.randn(5, 20)  # 5 fake rows, 20 fake columns
    output = model(dummy_input)
    print("Output shape:", output.shape)  # should be torch.Size([5])