"""
Train every additional network needed for the size/generalization/
non-smooth/timing scripts, in one batch: n128, n64, n32, n16 (same
dataset, four architectures), and n128_lo (much less data, same
architecture -- for the under-trained comparison).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_one import main as train_one  # noqa: E402

RUNS = [
    dict(tag="n128", hidden=128),
    dict(tag="n64", hidden=64),
    dict(tag="n32", hidden=32),
    dict(tag="n16", hidden=16),
    dict(tag="n128_lo", hidden=128, n_traj=3, t_train=3.0, skip=10),
]


def main():

    t0 = time.time()

    for i, run in enumerate(RUNS):
        print(f"\n{'=' * 60}\nRUN {i + 1}/{len(RUNS)}: {run}\n{'=' * 60}")

        argv = ["--tag", run["tag"], "--hidden", str(run["hidden"])]
        if "n_traj" in run:
            argv += ["--n-traj", str(run["n_traj"])]
        if "t_train" in run:
            argv += ["--t-train", str(run["t_train"])]
        if "skip" in run:
            argv += ["--skip", str(run["skip"])]

        sys.argv = ["train_one.py"] + argv
        train_one()

        print(f"[{time.time() - t0:.0f} s elapsed total]")

    print(f"\nall {len(RUNS)} networks trained in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
