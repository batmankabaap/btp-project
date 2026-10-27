"""SWE flux, wave speeds, LLF flux and bar states (eqs. 2, 4-6, 19).

All functions work on arrays whose last axis is space, so a batch of
trajectories can be advanced at once. Interface i+1/2 is stored at index i
and the domain is periodic.
"""

import numpy as np

from .config import G


def right(a):
    return np.roll(a, -1, axis=-1)


def left(a):
    return np.roll(a, 1, axis=-1)


def flux(h, q, g=G):
    return q, q * q / h + 0.5 * g * h * h


def wave_speed(hl, ql, hr, qr, g=G):
    return np.maximum(np.abs(ql / hl) + np.sqrt(g * hl),
                      np.abs(qr / hr) + np.sqrt(g * hr))


def llf_flux(hl, ql, hr, qr, g=G):
    fhl, fql = flux(hl, ql, g)
    fhr, fqr = flux(hr, qr, g)
    lam = wave_speed(hl, ql, hr, qr, g)
    fh = 0.5 * (fhl + fhr) - 0.5 * lam * (hr - hl)
    fq = 0.5 * (fql + fqr) - 0.5 * lam * (qr - ql)
    return fh, fq


def interface_flux(h, q, g=G):
    """LLF flux at every interface i+1/2 of a periodic mesh."""
    return llf_flux(h, q, right(h), right(q), g)


def bar_states(h, q, g=G):
    """Bar states and wave speeds at every interface i+1/2."""
    hr, qr = right(h), right(q)
    lam = wave_speed(h, q, hr, qr, g)
    fh, fq = flux(h, q, g)
    fhr, fqr = flux(hr, qr, g)
    hb = 0.5 * (h + hr) - (fhr - fh) / (2 * lam)
    qb = 0.5 * (q + qr) - (fqr - fq) / (2 * lam)
    return hb, qb, lam
