"""
Train one subgrid-flux network and save it to artifacts/models/<tag>/
as {net.pt, in.pkl, out.pkl, meta.json}.

    python scripts/train_one.py --tag n64 --hidden 64

Uses a reduced ("dev") default scale (n=10 trajectories, T_train=10)
rather than the much larger dataset a from-scratch, fully converged
network would need -- see this repo's README for why, and pass
--n-traj/--t-train/--skip/--epochs1/--epochs2 for a larger run.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from swe.train import gen_data, train_net

MODEL_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "models"


def main():

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-traj", type=int, default=10)
    p.add_argument("--t-train", type=float, default=10.0)
    p.add_argument("--skip", type=int, default=10)
    p.add_argument("--epochs1", type=int, default=30)
    p.add_argument("--lr1", type=float, default=0.05)
    p.add_argument("--epochs2", type=int, default=100)
    p.add_argument("--lr2", type=float, default=0.001)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = MODEL_DIR / args.tag
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== data: n={args.n_traj} t_train={args.t_train} skip={args.skip} seed={args.seed} ===")
    t0 = time.time()
    X, y = gen_data(args.n_traj, args.t_train, seed=args.seed, skip=args.skip)
    print(f"{X.shape[0]} samples in {time.time() - t0:.1f} s")

    print(f"\n=== training Net(hidden={args.hidden}) ===")
    t0 = time.time()
    net, xs, ys, hist = train_net(
        X, y, hidden=args.hidden,
        epochs1=args.epochs1, lr1=args.lr1, epochs2=args.epochs2, lr2=args.lr2,
        batch=args.batch, seed=args.seed,
    )
    print(f"trained in {time.time() - t0:.1f} s")

    torch.save(net.state_dict(), out / "net.pt")
    with open(out / "in.pkl", "wb") as f:
        pickle.dump(xs, f)
    with open(out / "out.pkl", "wb") as f:
        pickle.dump(ys, f)
    with open(out / "meta.json", "w") as f:
        json.dump(
            dict(
                hidden=args.hidden, n_traj=args.n_traj, t_train=args.t_train,
                skip=args.skip, n_samples=int(X.shape[0]),
                epochs1=args.epochs1, lr1=args.lr1, epochs2=args.epochs2, lr2=args.lr2,
                seed=args.seed, train_loss=hist[-1]["train"], val_loss=hist[-1]["val"],
            ),
            f, indent=2,
        )

    print(f"\nsaved {out}  (train={hist[-1]['train']:.8f} val={hist[-1]['val']:.8f})")


if __name__ == "__main__":
    main()
