"""Helpers shared by the evaluation scripts: DNS references and reduced runs."""

import numpy as np

from . import config as cfg
from .grid import cell_centers, coarsen, sine_ic, step_ic
from .solver import FVModel


def fine_ics(cases):
    """Stack fine-mesh initial states. Each case is a Table-1 style dict or
    ('step', H0, a) for the non-smooth IC (32)."""
    x = cell_centers(cfg.N_FINE)
    states = [step_ic(x, *c[1:]) if isinstance(c, tuple) else sine_ic(x, **c) for c in cases]
    h, q = map(np.stack, zip(*states))
    return h, q


def dns(cases, t_end, save_at=(), k=cfg.K):
    """Fully resolved runs, averaged over k fine cells: {t: (H, Q)}."""
    h, q = fine_ics(cases)
    out = FVModel(cfg.DX).run(h, q, cfg.DT, t_end, save_at)
    return {t: (coarsen(h, k), coarsen(q, k)) for t, (h, q) in out.items()}


def reduced(cases, t_end, closure=None, limiter=False, dt=cfg.DT_COARSE,
            save_at=(), k=cfg.K):
    """Coarse run started from the box-averaged fine initial state."""
    h, q = fine_ics(cases)
    H, Q = coarsen(h, k), coarsen(q, k)
    model = FVModel(cfg.L / H.shape[-1], closure=closure, limiter=limiter)
    return model.run(H, Q, dt, t_end, save_at)


def rel_error(u, ref):
    """Relative l2 error per row; NaN if the run broke down."""
    return np.linalg.norm(u - ref, axis=-1) / np.linalg.norm(ref, axis=-1)


def total_variation(u):
    return np.abs(np.diff(u, axis=-1, append=u[..., :1])).sum(axis=-1)
