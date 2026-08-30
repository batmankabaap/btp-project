"""
Effect of training-data quantity: the "n128_lo" network (much less
training data than "n128") at t=200, all 4 test cases, net-only vs
net+MCL vs full model. Expect net-only to show spurious oscillations
that net+MCL corrects.

    python scripts/undertrained.py

Saves artifacts/figures/q_t200_undertrained.png.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

from swe.config import L, N_f, N_c, dt, dX
from swe.grid import make_grid
from swe.net import load_net
from swe.cases import CASES, run_case
from swe.plot import plot_cases, save

T_END = 200.0


def main():

    x_fine, dx = make_grid(N_f, L=L)
    x_coarse, _ = make_grid(N_c, L=L)

    net, xs, ys = load_net("n128_lo")

    t0 = time.time()
    results = {}

    for cid, params in CASES.items():
        print(f"=== case #{cid} (elapsed {time.time() - t0:.0f} s) ===")
        results[cid] = run_case(
            params, x_fine, x_coarse, t_end=T_END,
            dt=dt, dx=dx, dT=dt, dX=dX,
            net=net, x_scaler=xs, y_scaler=ys,
        )

    print(f"finished in {(time.time() - t0) / 60:.1f} min")

    save(
        plot_cases(
            results, x_coarse, "q", T_END,
            "under-trained network (n128_lo), discharge at t=200", show=False,
        ),
        "q_t200_undertrained",
    )


if __name__ == "__main__":
    main()
