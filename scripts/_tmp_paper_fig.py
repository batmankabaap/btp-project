"""TEMPORARY asset generator -- deleted after use.

cases_t200 figure (all 4 cases at t=200, exact vs NN vs NN+MCL) using the
paper-scale network (artifacts/models/paper/), at the paper's own coarse
time step dT = 0.005.

Output: artifacts/assets/cases_t200_paper.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swe.config import L, N_f, N_c, dt, dX
from swe.grid import make_grid
from swe.net import load_net
from swe.cases import CASES, run_case
from swe.plot import save

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

T_END = 200.0


def main():

    x_fine, dx = make_grid(N_f, L=L)
    x_coarse, _ = make_grid(N_c, L=L)

    net, xs, ys = load_net("paper")  # artifacts/models/paper/

    results = {}
    for cid, params in CASES.items():
        print(f"  case #{cid} ...", flush=True)
        results[cid] = run_case(
            params, x_fine, x_coarse, t_end=T_END,
            dt=dt, dx=dx, dT=dt, dX=dX, at=[T_END],   # dT = 0.005, as in the paper
            net=net, x_scaler=xs, y_scaler=ys,
        )

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    for cid, r in results.items():
        for row, (var, name) in enumerate([("h", "water height"), ("q", "discharge")]):
            ax = axes[row, cid - 1]
            hq_full = r["full"][T_END]
            hq_net = r["net"][T_END]
            hq_mcl = r["mcl"][T_END]
            idx = 0 if var == "h" else 1

            ax.plot(x_coarse, hq_full[idx], "k-", linewidth=2, label="exact (coarsened fine DNS)")
            ax.plot(x_coarse, hq_net[idx], "b-.", linewidth=1.5, label="paper-scale NN (no MCL)")
            ax.plot(x_coarse, hq_mcl[idx], "r--", linewidth=1.5, label="paper-scale NN + MCL")

            ax.set_title(f"case #{cid}, {name}, t={T_END:.0f}")
            ax.set_xlabel("x")
            ax.set_ylabel(var)
            ax.grid(True)

    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(f"exact vs paper-scale-NN vs paper-scale-NN+MCL at t={T_END:.0f} "
                 f"(coarse N=100, dT={dt}; fine DNS N=2000, dt={dt})")
    fig.tight_layout()
    save(fig, "cases_t200_paper", out_dir=OUT)


if __name__ == "__main__":
    main()
