"""SWE physical flux, the LLF numerical flux, and a bar-state-based higher-order flux."""

import numpy as np

from .config import g as g0


def flux(h, q, g=g0):
    """Physical flux f(u), u=[h,q]: f_h=q, f_q=q^2/h + g*h^2/2."""

    return q, q**2 / h + 0.5 * g * h**2


def wavespeed(hl, ql, hr, qr, g=g0):
    """Max wave speed at an interface: max(|v_l|+sqrt(g*h_l), |v_r|+sqrt(g*h_r))."""

    return np.maximum(
        np.abs(ql / hl) + np.sqrt(g * hl),
        np.abs(qr / hr) + np.sqrt(g * hr),
    )


def llf(hl, ql, hr, qr, g=g0):
    """Local Lax-Friedrichs numerical flux through one interface."""

    fhl, fql = flux(hl, ql, g=g)
    fhr, fqr = flux(hr, qr, g=g)
    lam = wavespeed(hl, ql, hr, qr, g=g)

    return 0.5 * (fhl + fhr) - 0.5 * lam * (hr - hl), 0.5 * (fql + fqr) - 0.5 * lam * (qr - ql)


def bar_state(h, q, g=g0):
    """LLF bar (Riemann-averaged) state at every interface. Returns h_bar, q_bar, v_bar, lam."""

    hr, qr = np.roll(h, -1), np.roll(q, -1)

    lam = np.maximum(
        np.abs(q / h) + np.sqrt(g * h),
        np.abs(qr / hr) + np.sqrt(g * hr),
    )

    fh, fq = flux(h, q, g=g)
    fhr, fqr = flux(hr, qr, g=g)

    hb = 0.5 * (h + hr) - (fhr - fh) / (2.0 * lam)
    qb = 0.5 * (q + qr) - (fqr - fq) / (2.0 * lam)

    return hb, qb, qb / hb, lam


def hi_flux(h, q, g=g0):
    """
    Delta-t-independent higher-order target flux used only for the
    Lax-Wendroff-style baseline: the physical flux evaluated at the
    bar state, f(u_bar). Substituting a local Courant number of 1
    (dt/dx -> 1/lambda) into the classical two-step Richtmyer
    Lax-Wendroff predictor reduces it to exactly this bar state, so
    this is parameter-free like the LLF flux itself.
    """

    hb, qb, _, _ = bar_state(h, q, g=g)

    return flux(hb, qb, g=g)
