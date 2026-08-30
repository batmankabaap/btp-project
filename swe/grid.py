"""Grid construction, initial states, and coarsening."""

import numpy as np

from .config import L as L0


def make_grid(n, L=L0):
    """Uniform periodic grid of n cells on (0, L). Returns (x, dx)."""

    h = L / n
    x = (np.arange(n) + 0.5) * h

    return x, h


def ic(x, H0, Ah, kh, phih, V0, Av, kv, phiv, L=L0):
    """Sinusoidal initial state: h = H0 + Ah*sin(2*pi*kh*x/L + phih), q = h*v."""

    h = H0 + Ah * np.sin(2 * np.pi * kh * x / L + phih)
    v = V0 + Av * np.sin(2 * np.pi * kv * x / L + phiv)

    return h, h * v


def ic_step(x, H0, a, L=L0):
    """Non-smooth 'step' initial state: h = H0+a in the middle third, H0-a/2 elsewhere, v=0."""

    h = np.where((x > L / 3) & (x < 2 * L / 3), H0 + a, H0 - a / 2)

    return h, np.zeros_like(x)


def coarsen(h, q, k):
    """Average k neighboring fine cells into one coarse cell."""

    n = len(h)
    assert n % k == 0

    return h.reshape(n // k, k).mean(axis=1), q.reshape(n // k, k).mean(axis=1)
