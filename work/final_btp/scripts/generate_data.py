"""Generate the training set: n DNS trajectories up to T_train."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swe.data import generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-traj", type=int, default=200)
    p.add_argument("--t-train", type=float, default=80.0)
    p.add_argument("--stride", type=int, default=10,
                   help="keep every stride-th fine step (paper: 1)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/data_n200_T80.npz")
    args = p.parse_args()

    start = time.time()
    X, Y, params = generate(args.n_traj, args.t_train, args.stride, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X, Y=Y, **{f"ic_{k}": v for k, v in params.items()})
    print(f"saved {X.shape[0] * X.shape[1] * X.shape[2]:,} samples to {out} "
          f"in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
