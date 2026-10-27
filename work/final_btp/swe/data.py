"""Training data: random DNS trajectories and the exact subgrid fluxes (11)."""

import time

import numpy as np

from . import config as cfg
from .grid import cell_centers, coarsen, sine_ic
from .physics import interface_flux
from .solver import FVModel


def random_ic_params(rng, n):
    """n parameter sets drawn from the distributions below eq. (15)."""
    lo, hi = cfg.TRAIN_AMP
    k_lo, k_hi = cfg.TRAIN_WAVENUMBER
    return dict(
        H0=np.full(n, cfg.TRAIN_H0),
        Ah=rng.uniform(lo, hi, n),
        kh=rng.integers(k_lo, k_hi + 1, n),
        phih=rng.uniform(0, 2 * np.pi, n),
        V0=rng.uniform(*cfg.TRAIN_V0, n),
        Av=rng.uniform(lo, hi, n),
        kv=rng.integers(k_lo, k_hi + 1, n),
        phiv=rng.uniform(0, 2 * np.pi, n),
    )


def subgrid_flux(h, q, k=cfg.K, g=cfg.G):
    """Coarse averages U and G_{i+1/2} = F(u_ki, u_ki+1) - F(U_i, U_i+1).

    With 0-based indexing, coarse cell i holds fine cells ki .. ki+k-1, so
    its right interface separates fine cells ki+k-1 and ki+k.
    """
    H, Q = coarsen(h, k), coarsen(q, k)
    fh_fine, fq_fine = interface_flux(h, q, g)
    fh_fine, fq_fine = fh_fine[..., k - 1::k], fq_fine[..., k - 1::k]
    fh_bar, fq_bar = interface_flux(H, Q, g)
    return H, Q, fh_fine - fh_bar, fq_fine - fq_bar


def stencil(H, Q):
    """Network input for interface i+1/2: [U_{i-1}, U_i, U_{i+1}, U_{i+2}]."""
    cols = []
    for shift in (1, 0, -1, -2):
        cols += [np.roll(H, shift, axis=-1), np.roll(Q, shift, axis=-1)]
    return np.stack(cols, axis=-1)


def generate(n_traj, t_train, stride=1, seed=0, dt=cfg.DT, log=print):
    """Run n_traj DNS trajectories in one batch and collect (X, Y) samples.

    Samples are taken every `stride` fine steps. Returns X of shape
    (n_traj, n_samples, N_c, 8) and Y of shape (n_traj, n_samples, N_c, 2)
    so the caller can split by trajectory.
    """
    rng = np.random.default_rng(seed)
    params = random_ic_params(rng, n_traj)
    x = cell_centers(cfg.N_FINE)
    h, q = sine_ic(x[None, :], **{k: v[:, None] for k, v in params.items()})

    fine = FVModel(cfg.DX)
    n_steps = int(round(t_train / dt))
    n_samples = n_steps // stride
    X = np.empty((n_traj, n_samples, cfg.N_COARSE, 8), np.float32)
    Y = np.empty((n_traj, n_samples, cfg.N_COARSE, 2), np.float32)

    start = time.time()
    for n in range(1, n_steps + 1):
        h, q = fine.heun_step(h, q, dt)
        if n % stride == 0:
            s = n // stride - 1
            H, Q, gh, gq = subgrid_flux(h, q)
            X[:, s] = stencil(H, Q)
            Y[:, s] = np.stack([gh, gq], axis=-1)
        if n % (n_steps // 20) == 0:
            log(f"  t = {n * dt:6.2f} / {t_train}   min h = {h.min():.3f}   "
                f"{time.time() - start:.0f} s")
    return X, Y, params
