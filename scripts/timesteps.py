"""
Net+MCL solutions at t=20,50,100 for cases #2 and #3, integrated with
three coarse time steps dT=0.005, 0.05, 0.1 (the fine model's own
step is dt=0.005).

    python scripts/timesteps.py

Saves artifacts/figures/timesteps_case{2,3}.png (rows=h/q, cols=t).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swe.config import L, N_f, N_c, dt as dt0, dX
from swe.grid import make_grid, ic, coarsen
from swe.fine import run_fine
from swe.coarse import run_coarse
from swe.net import load_net
from swe.cases import CASES
from swe.plot import save

TIMES = [20.0, 50.0, 100.0]
STEPS = [0.005, 0.05, 0.1]
STYLE = {0.005: ("b--", "dT=0.005"), 0.05: ("r-", "dT=0.05"), 0.1: ("m-.", "dT=0.1")}


def run_case(case_id, net, xs, ys, x_fine, x_coarse, dx):

    params = CASES[case_id]
    h0, q0 = ic(x_fine, **params)
    H0, Q0 = ic(x_coarse, **params)

    t_end = max(TIMES)

    _, _, snaps = run_fine(h0, q0, t_end, dt=dt0, dx=dx, at=TIMES)
    full = {t: coarsen(*hq, N_f // N_c) for t, hq in snaps.items()}

    reduced = {}
    for dT in STEPS:
        t0 = time.time()
        _, _, snaps = run_coarse(
            H0, Q0, t_end, dT, dX, at=TIMES,
            use_net=True, use_mcl=True, net=net, x_scaler=xs, y_scaler=ys,
        )
        reduced[dT] = snaps
        print(f"  case #{case_id} dT={dT}: {time.time() - t0:.1f} s")

    return full, reduced


def plot_case(case_id, full, reduced, x_coarse, out_name, fig_no):

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))

    for col, t in enumerate(TIMES):

        H, Q = full[t]
        ax_h, ax_q = axes[0, col], axes[1, col]

        ax_h.plot(x_coarse, H, "k-", linewidth=2, label="Full model")
        ax_q.plot(x_coarse, Q, "k-", linewidth=2, label="Full model")

        for dT in STEPS:
            style, label = STYLE[dT]
            Hs, Qs = reduced[dT][t]
            ax_h.plot(x_coarse, Hs, style, linewidth=1.3, label=label)
            ax_q.plot(x_coarse, Qs, style, linewidth=1.3, label=label)

        ax_h.set_title(f"h(x,t), t={t}")
        ax_q.set_title(f"q(x,t), t={t}")
        ax_h.set_xlabel("x")
        ax_q.set_xlabel("x")
        ax_h.grid(True)
        ax_q.grid(True)

    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(f"net+MCL, case #{case_id}, dT=0.005/0.05/0.1")
    fig.tight_layout()

    save(fig, out_name)


def main():

    x_fine, dx = make_grid(N_f, L=L)
    x_coarse, _ = make_grid(N_c, L=L)

    net, xs, ys = load_net("main")

    for case_id, out_name in [(2, "timesteps_case2"), (3, "timesteps_case3")]:
        print(f"=== case #{case_id} ===")
        full, reduced = run_case(case_id, net, xs, ys, x_fine, x_coarse, dx)
        plot_case(case_id, full, reduced, x_coarse, out_name, case_id)


if __name__ == "__main__":
    main()
