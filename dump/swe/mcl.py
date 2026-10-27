"""Monolithic Convex Limiting: bound-preserving projection of a raw flux correction."""

import numpy as np

from .config import g as g0
from .physics import bar_state


def limit_h(h, hb, lam, gh):
    """Limit the raw height correction gh so the bar states stay within local h bounds."""

    hbl = np.roll(hb, 1)
    hmin, hmax = np.minimum(hbl, hb), np.maximum(hbl, hb)
    hmin_r, hmax_r = np.roll(hmin, -1), np.roll(hmax, -1)

    lo = lam * np.maximum(hb - hmax, hmin_r - hb)
    hi = lam * np.minimum(hb - hmin, hmax_r - hb)

    gh_star = np.clip(gh, lo, hi)

    return gh_star, hb - gh_star / lam, hb + gh_star / lam


def limit_q(hb, vb, lam, gq, gh_star, h_minus, h_plus):
    """Limit the residual discharge correction so bar velocities stay within local v bounds."""

    vbl = np.roll(vb, 1)
    vmin, vmax = np.minimum(vbl, vb), np.maximum(vbl, vb)
    vmin_r, vmax_r = np.roll(vmin, -1), np.roll(vmax, -1)

    dq = gq - gh_star * vb

    lo = np.maximum(lam * h_minus * (vb - vmax), lam * h_plus * (vmin_r - vb))
    hi = np.minimum(lam * h_minus * (vb - vmin), lam * h_plus * (vmax_r - vb))

    dq_star = np.clip(dq, lo, hi)

    return gh_star * vb + dq_star


def mcl(h, q, gh, gq, g=g0):
    """Raw correction [gh, gq] -> limited correction [gh_star, gq_star]."""

    hb, qb, vb, lam = bar_state(h, q, g=g)

    gh_star, h_minus, h_plus = limit_h(h, hb, lam, gh)
    gq_star = limit_q(hb, vb, lam, gq, gh_star, h_minus, h_plus)

    return gh_star, gq_star
