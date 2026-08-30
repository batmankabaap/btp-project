"""
Baseline comparison: MCL applied to a Delta-t-independent higher-order
("Lax-Wendroff-style") target flux instead of the learned net, at two
coarse resolutions (Nx=100, 500), vs the net+MCL model (Nx=100) and
the full model, at t=20, case #1.

    python scripts/hi_order.py

See swe/physics.hi_flux's docstring for why this flux is built from
the LLF bar state rather than the classical two-step Richtmyer form.

Saves artifacts/figures/hi_order_baseline.png.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swe.config import L, N_f, dt as dt0
from swe.grid import make_grid, ic
from swe.fine import run_fine
from swe.coarse import run_coarse, run_hi
from swe.net import load_net
from swe.cases import CASES
from swe.plot import save

T_END = 20.0
CASE_ID = 1


def main():

    params = CASES[CASE_ID]

    x_fine, dx = make_grid(N_f, L=L)
    h0, q0 = ic(x_fine, **params)

    t0 = time.time()
    h, q, _ = run_fine(h0, q0, T_END, dt=dt0, dx=dx)
    print(f"full model done ({time.time() - t0:.1f} s)")

    x100, dX100 = make_grid(100, L=L)
    H0, Q0 = ic(x100, **params)
    net, xs, ys = load_net("main")

    t0 = time.time()
    H_net, Q_net, _ = run_coarse(H0, Q0, T_END, dt0, dX100, use_net=True, use_mcl=True,
                                  net=net, x_scaler=xs, y_scaler=ys)
    print(f"net+MCL (Nx=100) done ({time.time() - t0:.1f} s)")

    t0 = time.time()
    H_hi100, Q_hi100 = run_hi(H0, Q0, T_END, dt0, dX100)
    print(f"hi-order+MCL (Nx=100) done ({time.time() - t0:.1f} s)")

    x500, dX500 = make_grid(500, L=L)
    H0_500, Q0_500 = ic(x500, **params)

    t0 = time.time()
    H_hi500, Q_hi500 = run_hi(H0_500, Q0_500, T_END, dt0, dX500)
    print(f"hi-order+MCL (Nx=500) done ({time.time() - t0:.1f} s)")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_fine, h, "k-", linewidth=1.5, label="Full model (Nf=2000)")
    ax.plot(x500, H_hi500, "b-.", linewidth=1.5, label="hi-order+MCL (Nx=500)")
    ax.plot(x100, H_hi100, "b--", linewidth=1.5, label="hi-order+MCL (Nx=100)")
    ax.plot(x100, H_net, "r:", linewidth=2.0, label="net+MCL (Nx=100)")
    ax.set_xlabel("x")
    ax.set_ylabel("h")
    ax.set_title(f"water height h(x,t) at t={T_END}, case #{CASE_ID}")
    ax.legend(loc="best")
    ax.grid(True)
    fig.tight_layout()

    save(fig, "hi_order_baseline")


if __name__ == "__main__":
    main()
