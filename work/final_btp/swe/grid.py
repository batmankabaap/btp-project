"""Meshes, initial conditions and coarse averaging."""

import numpy as np

from .config import L


def cell_centers(n, length=L):
    dx = length / n
    return (np.arange(n) + 0.5) * dx


def sine_ic(x, H0, Ah, kh, phih, V0, Av, kv, phiv, length=L):
    """Eq. (15). Returns (h, q) with q = h v."""
    h = H0 + Ah * np.sin(2 * np.pi * kh * x / length + phih)
    v = V0 + Av * np.sin(2 * np.pi * kv * x / length + phiv)
    return h, h * v


def step_ic(x, H0, a, length=L):
    """Eq. (32): raised middle third, fluid at rest."""
    middle = (x > length / 3) & (x < 2 * length / 3)
    h = np.where(middle, H0 + a, H0 - a / 2)
    return h, np.zeros_like(h)


def coarsen(u, k):
    """Box average over k consecutive cells, eq. (9)."""
    n = u.shape[-1]
    assert n % k == 0, "fine cell count must be divisible by k"
    return u.reshape(*u.shape[:-1], n // k, k).mean(axis=-1)
