"""
Generalization outside the training regime: all 4 "_dev"-scale
networks are trained on H0=2.0; test them, unmodified, on H0=1.5 and
H0=2.5 (a +/-25% fluctuation), same wave shape as case #2 otherwise,
at t=100, dT=0.05.

    python scripts/generalize.py

Saves artifacts/figures/gen_H1p5.png and artifacts/figures/gen_H2p5.png.
"""

import sys
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

DT = 0.05
T_END = 100.0
TAGS = {128: "n128", 64: "n64", 32: "n32", 16: "n16"}


def case_with_H0(H0):
    p = dict(CASES[2])
    p["H0"] = H0
    return p


def run(H0, out_name):

    params = case_with_H0(H0)

    x_fine, dx = make_grid(N_f, L=L)
    x_coarse, _ = make_grid(N_c, L=L)

    h0, q0 = ic(x_fine, **params)
    h, q, _ = run_fine(h0, q0, T_END, dt=dt0, dx=dx)
    H_ref, _ = coarsen(h, q, N_f // N_c)

    H0_grid, Q0_grid = ic(x_coarse, **params)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    pos = {128: (0, 0), 64: (0, 1), 32: (1, 0), 16: (1, 1)}

    for hidden, (row, col) in pos.items():
        net, xs, ys = load_net(TAGS[hidden], hidden=hidden)
        H, _, _ = run_coarse(H0_grid, Q0_grid, T_END, DT, dX, use_net=True, use_mcl=True,
                              net=net, x_scaler=xs, y_scaler=ys)
        ax = axes[row, col]
        ax.plot(x_coarse, H_ref, "k-", linewidth=2, label="Full model")
        ax.plot(x_coarse, H, "r--", linewidth=1.5, label="net+MCL")
        ax.set_title(f"hidden={hidden}")
        ax.set_xlabel("x")
        ax.set_ylabel("h")
        ax.grid(True)
        print(f"  H0={H0} hidden={hidden} done")

    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(f"h(x,t) at t={T_END}, H0={H0} (trained on H0=2.0), dT={DT}")
    fig.tight_layout()
    save(fig, out_name)


def main():

    print("=== H0=1.5 ===")
    run(1.5, "gen_H1p5")

    print("=== H0=2.5 ===")
    run(2.5, "gen_H2p5")


if __name__ == "__main__":
    main()
