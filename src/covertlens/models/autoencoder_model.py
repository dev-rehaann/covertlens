"""PyTorch autoencoder for flow reconstruction anomaly scoring."""

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


class FlowAutoencoder(nn.Module):
    """Symmetric feedforward autoencoder for standardized flow features."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reconstruct one batch of standardized feature vectors."""
        return self.network(inputs)


def train_autoencoder(
    X: pd.DataFrame,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    random_state: int = 42,
) -> FlowAutoencoder:
    """Train on flows believed to be mostly normal, without passing labels.

    We train assuming most training data is normal, consistent with real
    deployment where covert traffic is rare. Selecting majority/baseline data
    for training is therefore an explicit design choice and study assumption,
    even when ground-truth labels happen to be available for later evaluation.
    """
    if X.empty:
        raise ValueError("X must contain at least one flow")

    torch.manual_seed(random_state)
    inputs = torch.tensor(X.to_numpy(), dtype=torch.float32)
    model = FlowAutoencoder(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()
    loader = DataLoader(inputs, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch), batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - loss: {total_loss / len(inputs):.6f}")

    model.eval()
    return model


def reconstruction_error(model: FlowAutoencoder, X: pd.DataFrame) -> pd.Series:
    """Return per-flow reconstruction MSE; higher means more anomalous."""
    inputs = torch.tensor(X.to_numpy(), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        errors = torch.mean((inputs - model(inputs)) ** 2, dim=1).numpy()
    return pd.Series(errors, index=X.index, name="reconstruction_error")
