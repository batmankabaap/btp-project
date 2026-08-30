"""Fine-mesh LLF + Heun solver (the 'full model' / DNS reference)."""

import numpy as np

from .config import dx as dx0, dt as dt0, g as g0
from .physics import llf


def rhs(h, q, dx=dx0, g=g0):
    """Fine-grid spatial RHS (LLF fluxes, periodic BCs)."""

    hr, qr = np.roll(h, -1), np.roll(q, -1)
    fh, fq = llf(h, q, hr, qr, g=g)

    return (np.roll(fh, 1) - fh) / dx, (np.roll(fq, 1) - fq) / dx


def step(h, q, dt, dx=dx0, g=g0):
    """One Heun (SSP-RK2) step."""

    dh1, dq1 = rhs(h, q, dx=dx, g=g)
    hp, qp = h + dt * dh1, q + dt * dq1

    dh2, dq2 = rhs(hp, qp, dx=dx, g=g)

    return 0.5 * (h + hp + dt * dh2), 0.5 * (q + qp + dt * dq2)


def run_fine(h0, q0, t_end, dt=dt0, dx=dx0, g=g0, at=None, verbose=False):
    """
    Advance (h0, q0) to t_end. `at`: optional times to also snapshot
    along the way. Returns (h, q, snapshots) where snapshots is a
    dict {time: (h, q)}.
    """

    n = int(round(t_end / dt))
    h, q = h0.copy(), q0.copy()

    snaps = {}
    pending = sorted(at) if at else []

    for i in range(n):
        h, q = step(h, q, dt, dx=dx, g=g)
        t = (i + 1) * dt

        while pending and t >= pending[0] - 1e-9:
            snaps[pending[0]] = (h.copy(), q.copy())
            pending.pop(0)

        if verbose and i % max(n // 10, 1) == 0:
            print(f"  fine {i}/{n}  min(h)={np.min(h):.4f}")

    return h, q, snaps
