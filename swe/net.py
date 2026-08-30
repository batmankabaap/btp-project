"""The subgrid-flux network: architecture, stencil, and save/load."""

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

MODEL_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "models"


class Net(nn.Module):
    """8 inputs -> 3 x hidden (LeakyReLU) -> 2 outputs [gh, gq]."""

    def __init__(self, hidden=128):
        super().__init__()

        self.hidden = hidden
        self.stack = nn.Sequential(
            nn.Linear(8, hidden), nn.LeakyReLU(0.01),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.01),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.01),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.stack(x)


def stencil(h, q):
    """Four-point stencil [h_{i-1},q_{i-1},h_i,q_i,h_{i+1},q_{i+1},h_{i+2},q_{i+2}] per interface."""

    hs = [np.roll(h, 1), h, np.roll(h, -1), np.roll(h, -2)]
    qs = [np.roll(q, 1), q, np.roll(q, -1), np.roll(q, -2)]

    cols = [c for pair in zip(hs, qs) for c in pair]

    return np.column_stack(cols)


def load_net(tag, hidden=128, model_dir=None):
    """Load a trained Net + its input/output scalers from artifacts/models/<tag>/."""

    d = Path(model_dir) if model_dir is not None else MODEL_DIR
    d = d / tag

    with open(d / "in.pkl", "rb") as f:
        x_scaler = pickle.load(f)
    with open(d / "out.pkl", "rb") as f:
        y_scaler = pickle.load(f)

    state = torch.load(d / "net.pt", map_location="cpu", weights_only=False)
    # The "main" checkpoint predates this package's short attribute name.
    state = {k.replace("linear_relu_stack.", "stack."): v for k, v in state.items()}

    net = Net(hidden=hidden)
    net.load_state_dict(state)
    net.eval()

    return net, x_scaler, y_scaler


def predict(net, x_scaler, y_scaler, h, q):
    """Predict the raw subgrid flux correction [gh, gq] at every interface."""

    x = torch.tensor(x_scaler.transform(stencil(h, q)), dtype=torch.float32)

    with torch.no_grad():
        y = net(x).numpy()

    y = y_scaler.inverse_transform(y)

    return y[:, 0], y[:, 1]
