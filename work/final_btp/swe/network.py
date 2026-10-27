"""Subgrid-flux networks (Section 3.1, Fig. 1) and a mirror-symmetric variant."""

import numpy as np
import torch
from torch import nn

from .data import stencil

# x -> -x maps the stencil [U_{i-1}, U_i, U_{i+1}, U_{i+2}] to the reversed
# stencil with q negated, and the subgrid flux [G_h, G_q] to [-G_h, G_q].
MIRROR_INPUT = [6, 7, 4, 5, 2, 3, 0, 1]
MIRROR_INPUT_SIGN = [1.0, -1.0] * 4
MIRROR_OUTPUT_SIGN = [-1.0, 1.0]


class SubgridNet(nn.Module):
    """8 -> hidden x 3 (leaky ReLU, alpha = 0.01) -> 2."""

    def __init__(self, hidden=128):
        super().__init__()
        self.hidden = hidden
        layers, width = [], 8
        for _ in range(3):
            layers += [nn.Linear(width, hidden), nn.LeakyReLU(0.01)]
            width = hidden
        layers.append(nn.Linear(width, 2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MirrorSymmetric(nn.Module):
    """G(s) = (N(s) + P N(R s)) / 2, so a mirrored flow gets the mirrored flux exactly.

    Takes and returns normalized values; the mirror itself acts in physical units.
    """

    def __init__(self, base, x_norm, y_norm):
        super().__init__()
        self.base = base
        f32 = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
        self.register_buffer("x_mean", f32(x_norm.mean))
        self.register_buffer("x_std", f32(x_norm.std))
        self.register_buffer("y_mean", f32(y_norm.mean))
        self.register_buffer("y_std", f32(y_norm.std))
        self.register_buffer("in_sign", f32(MIRROR_INPUT_SIGN))
        self.register_buffer("out_sign", f32(MIRROR_OUTPUT_SIGN))

    def forward(self, x):
        s = x * self.x_std + self.x_mean
        x_mirror = (s[..., MIRROR_INPUT] * self.in_sign - self.x_mean) / self.x_std
        # one batched pass over both stencils is much cheaper than two calls
        y, y_mirror = (self.base(torch.cat([x, x_mirror])) * self.y_std + self.y_mean).chunk(2)
        g = 0.5 * (y + self.out_sign * y_mirror)
        return (g - self.y_mean) / self.y_std


def build_net(arch, x_norm, y_norm):
    """arch = dict(hidden=..., symmetric=...)."""
    net = SubgridNet(arch["hidden"])
    if arch.get("symmetric"):
        net = MirrorSymmetric(net, x_norm, y_norm)
    return net


def n_params(net):
    return sum(p.numel() for p in net.parameters())


class Normalizer:
    """Per-feature standardization, stored alongside the weights."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, np.float64)
        self.std = np.asarray(std, np.float64)

    @classmethod
    def fit(cls, data, max_rows=2_000_000, seed=0):
        data = data.reshape(-1, data.shape[-1])
        if len(data) > max_rows:
            rows = np.random.default_rng(seed).choice(len(data), max_rows, replace=False)
            data = data[rows]
        data = data.astype(np.float64)
        return cls(data.mean(axis=0), data.std(axis=0))

    def encode(self, a):
        return (a - self.mean) / self.std

    def decode(self, a):
        return a * self.std + self.mean


class NeuralClosure:
    """Callable (H, Q) -> (G_h, G_q) used by the coarse solver."""

    def __init__(self, net, x_norm, y_norm, arch):
        self.net = net.eval()
        self.x_norm = x_norm
        self.y_norm = y_norm
        self.arch = arch
        self.meta = {}

    def __call__(self, H, Q):
        x = self.x_norm.encode(stencil(H, Q))
        with torch.no_grad():
            y = self.net(torch.as_tensor(x, dtype=torch.float32)).numpy()
        y = self.y_norm.decode(y)
        return y[..., 0], y[..., 1]

    def save(self, path, **meta):
        torch.save(dict(
            state=self.net.state_dict(), arch=self.arch,
            x_mean=self.x_norm.mean, x_std=self.x_norm.std,
            y_mean=self.y_norm.mean, y_std=self.y_norm.std,
            meta=meta,
        ), path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        x_norm = Normalizer(ckpt["x_mean"], ckpt["x_std"])
        y_norm = Normalizer(ckpt["y_mean"], ckpt["y_std"])
        # older checkpoints only stored the hidden width
        arch = ckpt.get("arch") or dict(hidden=ckpt["hidden"], symmetric=False)
        net = build_net(arch, x_norm, y_norm)
        net.load_state_dict(ckpt["state"])
        closure = cls(net, x_norm, y_norm, arch)
        closure.meta = ckpt["meta"]
        return closure
