"""Coarse-mesh solver: LLF baseline + subgrid correction (net or hi-order) + MCL + Heun."""

import numpy as np

from .config import g as g0
from .physics import llf, hi_flux
from .mcl import mcl
from .net import predict


def rhs(h, q, dX, use_net=True, use_mcl=True, net=None, x_scaler=None, y_scaler=None, g=g0):
    """
    dh/dt, dq/dt for the current coarse state.

    use_net=False: plain coarse LLF.
    use_net=True:  LLF + net correction, optionally passed through
                   MCL (use_mcl=True) or added raw (use_mcl=False).
    """

    hr, qr = np.roll(h, -1), np.roll(q, -1)
    fh, fq = llf(h, q, hr, qr, g=g)

    if use_net:
        gh, gq = predict(net, x_scaler, y_scaler, h, q)
        if use_mcl:
            gh, gq = mcl(h, q, gh, gq, g=g)
        fh, fq = fh + gh, fq + gq

    return (np.roll(fh, 1) - fh) / dX, (np.roll(fq, 1) - fq) / dX


def step(h, q, dt, dX, **kw):
    """One Heun (SSP-RK2) step; kw forwarded to rhs (use_net, use_mcl, net, x_scaler, y_scaler, g)."""

    dh1, dq1 = rhs(h, q, dX, **kw)
    hp, qp = h + dt * dh1, q + dt * dq1

    dh2, dq2 = rhs(hp, qp, dX, **kw)

    return 0.5 * (h + hp + dt * dh2), 0.5 * (q + qp + dt * dq2)


def run_coarse(h0, q0, t_end, dt, dX, at=None, verbose=False, **kw):
    """Advance (h0, q0) to t_end. Returns (h, q, snapshots) as in fine.run_fine."""

    n = int(round(t_end / dt))
    h, q = h0.copy(), q0.copy()

    snaps = {}
    pending = sorted(at) if at else []

    for i in range(n):
        h, q = step(h, q, dt, dX, **kw)
        t = (i + 1) * dt

        while pending and t >= pending[0] - 1e-9:
            snaps[pending[0]] = (h.copy(), q.copy())
            pending.pop(0)

        if verbose and i % max(n // 10, 1) == 0:
            print(f"  coarse {i}/{n}  min(h)={np.min(h):.4f}")

    return h, q, snaps


# ----------------------------------------------------------------
# Higher-order (Lax-Wendroff-style) + MCL baseline, for the Fig. 6
# comparison. Reuses mcl() exactly as above -- MCL only ever sees a
# raw correction, whatever its source.
# ----------------------------------------------------------------

def hi_rhs(h, q, dX, g=g0):

    hr, qr = np.roll(h, -1), np.roll(q, -1)
    fh, fq = llf(h, q, hr, qr, g=g)
    fh_hi, fq_hi = hi_flux(h, q, g=g)

    gh, gq = mcl(h, q, fh_hi - fh, fq_hi - fq, g=g)
    fh, fq = fh + gh, fq + gq

    return (np.roll(fh, 1) - fh) / dX, (np.roll(fq, 1) - fq) / dX


def hi_step(h, q, dt, dX, g=g0):

    dh1, dq1 = hi_rhs(h, q, dX, g=g)
    hp, qp = h + dt * dh1, q + dt * dq1

    dh2, dq2 = hi_rhs(hp, qp, dX, g=g)

    return 0.5 * (h + hp + dt * dh2), 0.5 * (q + qp + dt * dq2)


def run_hi(h0, q0, t_end, dt, dX, g=g0, verbose=False):
    """Advance the coarse state under the MCL + higher-order-flux baseline to t_end."""

    n = int(round(t_end / dt))
    h, q = h0.copy(), q0.copy()

    for i in range(n):
        h, q = hi_step(h, q, dt, dX, g=g)
        if verbose and i % max(n // 10, 1) == 0:
            print(f"  hi {i}/{n}  min(h)={np.min(h):.4f}")

    return h, q
