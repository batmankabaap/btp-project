"""
Full model vs net-only vs net+MCL, at t=200 and t=400, across all 4
test cases, using the "main" network (hidden=128, the one shipped
with this project).

    python scripts/compare.py

Saves artifacts/figures/{q_t200, h_t200, q_t400}.png.
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


def main():

    x_fine, dx = make_grid(N_f, L=L)
    x_coarse, dX_chk = make_grid(N_c, L=L)
    assert abs(dX_chk - dX) < 1e-12

    net, xs, ys = load_net("main")

    t0 = time.time()
    results = {}

    for cid, params in CASES.items():
        print(f"=== case #{cid} (elapsed {time.time() - t0:.0f} s) ===")
        results[cid] = run_case(
            params, x_fine, x_coarse, t_end=400.0,
            dt=dt, dx=dx, dT=dt, dX=dX, at=[200.0],
            net=net, x_scaler=xs, y_scaler=ys,
        )

    print(f"\nsweep finished in {(time.time() - t0) / 60:.1f} min")

    save(plot_cases(results, x_coarse, "q", 200.0, "discharge q(x,t) at t=200", show=False), "q_t200")
    save(plot_cases(results, x_coarse, "h", 200.0, "water height h(x,t) at t=200", show=False), "h_t200")
    save(plot_cases(results, x_coarse, "q", 400.0, "discharge q(x,t) at t=400", show=False), "q_t400")


if __name__ == "__main__":
    main()
