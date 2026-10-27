"""Table 1 test cases and the per-case (full model / net / net+MCL) driver."""

from .config import k as k0
from .grid import ic, coarsen
from .fine import run_fine
from .coarse import run_coarse

# Sinusoidal test cases: h0=H0+Ah*sin(2*pi*kh*x/L+phih), v0=V0+Av*sin(2*pi*kv*x/L+phiv)
CASES = {
    1: dict(H0=2.0, Ah=0.20, kh=3, phih=0.00, V0=1.0, Av=0.0, kv=0, phiv=0.0),
    2: dict(H0=2.0, Ah=0.45, kh=4, phih=2.78, V0=1.1, Av=0.5, kv=3, phiv=4.5),
    3: dict(H0=2.0, Ah=0.45, kh=4, phih=2.78, V0=1.1, Av=0.5, kv=4, phiv=4.5),
    4: dict(H0=2.0, Ah=0.20, kh=3, phih=0.00, V0=1.1, Av=0.1, kv=2, phiv=0.0),
}


def init_case(params, x_fine, x_coarse):
    """Fine-grid and coarse-grid initial states for one case."""

    return (*ic(x_fine, **params), *ic(x_coarse, **params))


def run_case(params, x_fine, x_coarse, t_end, dt, dx, dT, dX, k=k0, at=None, **net_kw):
    """
    Run one case's full model (coarsened), net-only, and net+MCL
    models to t_end. net_kw: net, x_scaler, y_scaler (see coarse.rhs).

    Returns dict(full=..., net=..., mcl=...), each a dict {time: (h, q)}.
    """

    h0, q0, H0, Q0 = init_case(params, x_fine, x_coarse)

    h, q, snaps = run_fine(h0, q0, t_end, dt=dt, dx=dx, at=at)
    full = {t_end: coarsen(h, q, k)}
    full.update({t: coarsen(*hq, k) for t, hq in snaps.items()})

    Hn, Qn, snaps_n = run_coarse(H0, Q0, t_end, dT, dX, use_net=True, use_mcl=False, at=at, **net_kw)
    net = {t_end: (Hn, Qn), **snaps_n}

    Hm, Qm, snaps_m = run_coarse(H0, Q0, t_end, dT, dX, use_net=True, use_mcl=True, at=at, **net_kw)
    mcl = {t_end: (Hm, Qm), **snaps_m}

    return dict(full=full, net=net, mcl=mcl)
