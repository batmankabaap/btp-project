"""Sequential monolithic convex limiting of subgrid fluxes (Section 3.2).

Given raw corrections (gh, gq) at every interface, returns limited
corrections such that the bar states (18) satisfy the local bounds (24).
"""

import numpy as np

from .config import G
from .physics import bar_states, left, right


def _local_bounds(ub):
    """Bounds (22)/(23) for cell i from the bar states at i-1/2 and i+1/2."""
    lo = np.minimum(left(ub), ub)
    hi = np.maximum(left(ub), ub)
    return lo, hi


def limit_height(gh, hb, lam):
    """Eq. (27). Stored index i is interface i+1/2 between cells i and i+1."""
    hmin, hmax = _local_bounds(hb)
    upper = lam * np.minimum(hb - hmin, right(hmax) - hb)
    lower = lam * np.maximum(hb - hmax, right(hmin) - hb)
    return np.where(gh >= 0, np.minimum(gh, upper), np.maximum(gh, lower))


def limit_discharge(gq, gh_star, hb, vb, lam):
    """Eqs. (28) and (31)."""
    vmin, vmax = _local_bounds(vb)
    h_minus = hb - gh_star / lam
    h_plus = hb + gh_star / lam

    dgq = gq - gh_star * vb
    upper = lam * np.minimum(h_minus * (vb - vmin), h_plus * (right(vmax) - vb))
    lower = lam * np.maximum(h_minus * (vb - vmax), h_plus * (right(vmin) - vb))
    dgq_star = np.where(dgq >= 0, np.minimum(dgq, upper), np.maximum(dgq, lower))

    return gh_star * vb + dgq_star


def mcl(h, q, gh, gq, g=G):
    hb, qb, lam = bar_states(h, q, g)
    vb = qb / hb
    gh_star = limit_height(gh, hb, lam)
    gq_star = limit_discharge(gq, gh_star, hb, vb, lam)
    return gh_star, gq_star
