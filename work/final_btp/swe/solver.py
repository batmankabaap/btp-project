"""Finite-volume semi-discretization and Heun time stepping.

The same code serves the fine DNS (no closure) and the reduced coarse
models (a closure adds subgrid fluxes G to the LLF flux, optionally
passed through MCL), i.e. eqs. (3), (10) and (16).
"""

import numpy as np

from .config import G
from .mcl import mcl
from .physics import interface_flux, left


class FVModel:
    """dU/dt = (F_{i-1/2} - F_{i+1/2}) / dx with F = LLF + G*."""

    def __init__(self, dx, closure=None, limiter=False, g=G):
        self.dx = dx
        self.closure = closure
        self.limiter = limiter
        self.g = g

    def fluxes(self, h, q):
        fh, fq = interface_flux(h, q, self.g)
        if self.closure is not None:
            gh, gq = self.closure(h, q)
            if self.limiter:
                gh, gq = mcl(h, q, gh, gq, self.g)
            fh, fq = fh + gh, fq + gq
        return fh, fq

    def rhs(self, h, q):
        fh, fq = self.fluxes(h, q)
        return (left(fh) - fh) / self.dx, (left(fq) - fq) / self.dx

    def heun_step(self, h, q, dt):
        dh, dq = self.rhs(h, q)
        h1, q1 = h + dt * dh, q + dt * dq
        dh, dq = self.rhs(h1, q1)
        return 0.5 * (h + h1 + dt * dh), 0.5 * (q + q1 + dt * dq)

    def run(self, h, q, dt, t_end, save_at=()):
        """Integrate to t_end. Returns {t: (h, q)} for t in save_at and t_end."""
        times = sorted(set(save_at) | {t_end})
        steps = {int(round(t / dt)): t for t in times}
        n_steps = max(steps)

        out = {}
        if 0 in steps:
            out[steps[0]] = (h.copy(), q.copy())
        # an unlimited closure may produce h <= 0; that run then just turns NaN
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            for n in range(1, n_steps + 1):
                h, q = self.heun_step(h, q, dt)
                if n in steps:
                    out[steps[n]] = (h.copy(), q.copy())
        return out
