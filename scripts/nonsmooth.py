"""
Non-smooth "step" initial state (H0=2.4, a=0.5), even though all
networks were trained only on smooth data. Shows h(x,t) at
t=10,20,30,40 for hidden=128 and hidden=32 (with MCL), plus hidden=128
*without* MCL to show the spurious oscillations MCL corrects.

    python scripts/nonsmooth.py

Saves artifacts/figures/nonsmooth.png.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swe.config import L, N_c, dX, dT0
from swe.grid import make_grid, ic_step
from swe.coarse import run_coarse
from swe.net import load_net
from swe.plot import save

H0, A = 2.4, 0.5
TIMES = [10.0, 20.0, 30.0, 40.0]
TAGS = {128: "n128", 32: "n32"}


def main():

    x_coarse, _ = make_grid(N_c, L=L)
    H0_grid, Q0_grid = ic_step(x_coarse, H0, A, L=L)

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))

    for row, hidden in enumerate([128, 32]):

        net, xs, ys = load_net(TAGS[hidden], hidden=hidden)
        _, _, snaps = run_coarse(
            H0_grid, Q0_grid, max(TIMES), dT0, dX, at=TIMES,
            use_net=True, use_mcl=True, net=net, x_scaler=xs, y_scaler=ys,
        )

        for col, t in enumerate(TIMES):
            ax = axes[row, col]
            H, _ = snaps[t]
            ax.plot(x_coarse, H, "r-", linewidth=1.3)
            ax.set_title(f"hidden={hidden}, with MCL, t={t}")
            ax.set_xlabel("x")
            ax.set_ylabel("h")
            ax.grid(True)

        print(f"  hidden={hidden} with MCL done")

    net, xs, ys = load_net("n128", hidden=128)
    _, _, snaps = run_coarse(
        H0_grid, Q0_grid, max(TIMES), dT0, dX, at=TIMES,
        use_net=True, use_mcl=False, net=net, x_scaler=xs, y_scaler=ys,
    )
    for col, t in enumerate(TIMES):
        ax = axes[2, col]
        H, _ = snaps[t]
        ax.plot(x_coarse, H, "b-", linewidth=1.0)
        ax.set_title(f"hidden=128, no MCL, t={t}")
        ax.set_xlabel("x")
        ax.set_ylabel("h")
        ax.grid(True)

    print("  hidden=128 without MCL done")

    fig.suptitle(f"non-smooth IC (H0={H0}, a={A}), networks trained on smooth data only")
    fig.tight_layout()
    save(fig, "nonsmooth")


if __name__ == "__main__":
    main()
