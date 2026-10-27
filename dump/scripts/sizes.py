"""
Effect of network size (hidden=128,64,32,16), same training data:

  size_t200:  h(x,t) at t=200, dT=0.05, case #2, all 4 sizes, with MCL.
  size16_mcl: h(x,t) at t=20, case #2 -- hidden=128,64,32 with MCL,
              and hidden=16 both with and without MCL (shows the
              oscillatory behavior MCL corrects for the smallest net).

    python scripts/sizes.py

Saves artifacts/figures/{size_t200, size16_mcl}.png.
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

CASE_ID = 2
DT = 0.05
TAGS = {128: "n128", 64: "n64", 32: "n32", 16: "n16"}


def reference(t_end, x_fine, dx):
    h0, q0 = ic(x_fine, **CASES[CASE_ID])
    h, q, _ = run_fine(h0, q0, t_end, dt=dt0, dx=dx)
    return coarsen(h, q, N_f // N_c)


def reduced(t_end, x_coarse, hidden, use_mcl):
    H0, Q0 = ic(x_coarse, **CASES[CASE_ID])
    net, xs, ys = load_net(TAGS[hidden], hidden=hidden)
    H, Q, _ = run_coarse(H0, Q0, t_end, DT, dX, use_net=True, use_mcl=use_mcl,
                          net=net, x_scaler=xs, y_scaler=ys)
    return H, Q


def size_t200(x_fine, x_coarse, dx):

    t_end = 200.0
    H_ref, _ = reference(t_end, x_fine, dx)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    pos = {128: (0, 0), 64: (0, 1), 32: (1, 0), 16: (1, 1)}

    for hidden, (row, col) in pos.items():
        H, _ = reduced(t_end, x_coarse, hidden, use_mcl=True)
        ax = axes[row, col]
        ax.plot(x_coarse, H_ref, "k-", linewidth=2, label="Full model")
        ax.plot(x_coarse, H, "r--", linewidth=1.5, label="net+MCL")
        ax.set_title(f"hidden={hidden}")
        ax.set_xlabel("x")
        ax.set_ylabel("h")
        ax.grid(True)
        print(f"  hidden={hidden} done")

    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(f"h(x,t) at t={t_end}, dT={DT}, case #{CASE_ID}, network sizes")
    fig.tight_layout()
    save(fig, "size_t200")


def size16_mcl(x_fine, x_coarse, dx):

    t_end = 20.0
    H_ref, _ = reference(t_end, x_fine, dx)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for ax, hidden in zip(axes.flat[:3], [128, 64, 32]):
        H, _ = reduced(t_end, x_coarse, hidden, use_mcl=True)
        ax.plot(x_coarse, H_ref, "k-", linewidth=2, label="Full model")
        ax.plot(x_coarse, H, "r--", linewidth=1.5, label="with MCL")
        ax.set_title(f"hidden={hidden}, with MCL")
        ax.set_xlabel("x")
        ax.set_ylabel("h")
        ax.grid(True)

    ax = axes.flat[3]
    H_mcl, _ = reduced(t_end, x_coarse, 16, use_mcl=True)
    H_raw, _ = reduced(t_end, x_coarse, 16, use_mcl=False)
    ax.plot(x_coarse, H_ref, "k-", linewidth=2, label="Full model")
    ax.plot(x_coarse, H_mcl, "r--", linewidth=1.5, label="hidden=16, with MCL")
    ax.plot(x_coarse, H_raw, "b-.", linewidth=1.0, label="hidden=16, no MCL")
    ax.set_title("hidden=16, with vs without MCL")
    ax.set_xlabel("x")
    ax.set_ylabel("h")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"h(x,t) at t={t_end}, dT={DT}, case #{CASE_ID}")
    fig.tight_layout()
    save(fig, "size16_mcl")


def main():

    x_fine, dx = make_grid(N_f, L=L)
    x_coarse, _ = make_grid(N_c, L=L)

    print("=== size_t200 ===")
    size_t200(x_fine, x_coarse, dx)

    print("=== size16_mcl ===")
    size16_mcl(x_fine, x_coarse, dx)


if __name__ == "__main__":
    main()
