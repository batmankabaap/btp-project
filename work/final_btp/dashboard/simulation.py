"""Run one test case on the fine DNS, plain coarse LLF and every network.

Used by the live server and by scripts/precompute.py, so both produce the
same result layout:

    curves["llf"], curves["<model>/nn"], curves["<model>/mcl"] -> {h, q}
    metrics[<same keys>] and metrics["fine"]                   -> error, cost, ...
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swe import config as cfg
from swe.experiments import dns, reduced, rel_error, total_variation
from swe.grid import cell_centers, coarsen

SINE_KEYS = ("H0", "Ah", "kh", "phih", "V0", "Av", "kv", "phiv")
MAX_T_END = 400.0


def parse_case(case):
    """Validate a test case; returns (ic, t_end, dt)."""
    t_end = float(case.get("t_end", 200.0))
    dt = float(case.get("dt", cfg.DT_COARSE))
    if not 0 < t_end <= MAX_T_END:
        raise ValueError(f"t_end must be in (0, {MAX_T_END:g}]")
    if not 0.005 - 1e-12 <= dt <= 0.1:
        raise ValueError("dt must be between 0.005 and 0.1")

    if case.get("type", "sine") == "step":
        H0, a = float(case["H0"]), float(case["a"])
        if H0 - a / 2 <= 0:
            raise ValueError("step IC needs H0 - a/2 > 0")
        return ("step", H0, a), t_end, dt

    missing = [k for k in SINE_KEYS if k not in case]
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")
    ic = {k: float(case[k]) for k in SINE_KEYS}
    if ic["H0"] - abs(ic["Ah"]) <= 0:
        raise ValueError("water height must stay positive: need H0 > |Ah|")
    return ic, t_end, dt


def to_list(a):
    return [None if not np.isfinite(v) else round(float(v), 5) for v in a]


def finite_or_none(v):
    return round(float(v), 6) if np.isfinite(v) else None


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def run_metrics(h, q, h_ref, q_ref, seconds, dt):
    """Error against the coarse-averaged DNS plus cost; None where the run broke down."""
    ok = bool(np.all(np.isfinite(h)) and np.all(np.isfinite(q)))
    return dict(
        h_err=finite_or_none(rel_error(h, h_ref)),
        q_err=finite_or_none(rel_error(q, q_ref)),
        tv_ratio=finite_or_none(total_variation(q) / total_variation(q_ref)),
        h_min=round(float(h.min()), 4) if ok else None,
        seconds=round(seconds, 3), cells=cfg.N_COARSE, dt=dt,
    )


def simulate(closures, case):
    """closures: {name: NeuralClosure}. Returns the result dict described above."""
    ic, t_end, dt = parse_case(case)

    fine, fine_seconds = timed(lambda: dns([ic], t_end, k=1)[t_end])
    h_fine, q_fine = fine[0][0], fine[1][0]
    if not np.all(np.isfinite(h_fine)):
        raise ValueError("the reference DNS itself is unstable for this case: its fixed "
                         "time step 0.005 violates the CFL condition (flow too fast)")
    h_ref, q_ref = coarsen(h_fine, cfg.K), coarsen(q_fine, cfg.K)

    variants = {"llf": dict(closure=None)}
    for name, closure in closures.items():
        variants[f"{name}/nn"] = dict(closure=closure, limiter=False)
        variants[f"{name}/mcl"] = dict(closure=closure, limiter=True)

    initial = reduced([ic], 0.0)[0.0]
    out = dict(
        models=list(closures), t_end=t_end, dt=dt,
        x=to_list(cell_centers(cfg.N_COARSE)), x_fine=to_list(cell_centers(cfg.N_FINE)),
        initial=dict(h=to_list(initial[0][0]), q=to_list(initial[1][0])),
        fine=dict(h=to_list(h_fine), q=to_list(q_fine)),
        dns=dict(h=to_list(h_ref), q=to_list(q_ref)),
        curves={},
        metrics=dict(fine=dict(seconds=round(fine_seconds, 3), cells=cfg.N_FINE, dt=cfg.DT,
                               h_min=round(float(h_fine.min()), 4))),
    )
    for key, kw in variants.items():
        run, seconds = timed(lambda: reduced([ic], t_end, dt=dt, **kw)[t_end])
        h, q = run[0][0], run[1][0]
        out["curves"][key] = dict(h=to_list(h), q=to_list(q))
        out["metrics"][key] = run_metrics(h, q, h_ref, q_ref, seconds, dt)
    return out
