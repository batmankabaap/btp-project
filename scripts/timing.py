"""
Running time for a T=500 simulation: the fine LLF solver vs the
net+MCL coarse solver for each network size (128, 64, 32, 16).

    python scripts/timing.py

Saves artifacts/data/timing.json and prints a table. Absolute seconds
depend heavily on this machine's core count; only the relative
pattern (fine solver slowest, smaller nets faster) is expected to
carry over from the paper's Table A.2.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swe.config import L, N_f, N_c, dt, dX
from swe.grid import make_grid, ic
from swe.fine import run_fine
from swe.coarse import run_coarse
from swe.net import load_net
from swe.cases import CASES

T_END = 500.0
DT_COARSE = 0.05
TAGS = {128: "n128", 64: "n64", 32: "n32", 16: "n16"}

DATA_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "data"


def main():

    params = CASES[2]

    x_fine, dx = make_grid(N_f, L=L)
    x_coarse, _ = make_grid(N_c, L=L)

    h0, q0 = ic(x_fine, **params)
    H0, Q0 = ic(x_coarse, **params)

    times = {}

    print(f"=== fine (full model, N_f={N_f}, dt={dt}) ===")
    t0 = time.time()
    run_fine(h0, q0, T_END, dt=dt, dx=dx)
    times["fine"] = time.time() - t0
    print(f"  {times['fine']:.1f} s")

    for hidden, tag in TAGS.items():
        print(f"=== net{hidden}+MCL (Nx={N_c}, dT={DT_COARSE}) ===")
        net, xs, ys = load_net(tag, hidden=hidden)
        t0 = time.time()
        run_coarse(H0, Q0, T_END, DT_COARSE, dX, use_net=True, use_mcl=True,
                   net=net, x_scaler=xs, y_scaler=ys)
        times[f"net{hidden}"] = time.time() - t0
        print(f"  {times[f'net{hidden}']:.1f} s")

    print("\n" + "=" * 46)
    print(f"{'model':<10}{'time (s)':>10}{'speedup':>16}")
    for name, seconds in times.items():
        print(f"{name:<10}{seconds:>10.1f}{times['fine'] / seconds:>16.2f}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "timing.json"
    with open(out, "w") as f:
        json.dump(times, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
