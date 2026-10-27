"""Train the subgrid-flux network on a generated dataset.

Stage 1: up to 500 epochs at lr 0.05, stage 2: up to 2000 epochs at
lr 0.001, early stopping in both. The last 20% of trajectories are held
out for validation (skorch's default ValidSplit(5) is also an unshuffled
80/20 split).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swe.network import NeuralClosure, Normalizer, build_net, n_params
from swe.training import evaluate, train_stage


def to_tensor(a, norm):
    out = torch.empty(a.shape[0], a.shape[1], dtype=torch.float32)
    for i in range(0, len(a), 1 << 21):
        out[i:i + (1 << 21)] = torch.from_numpy(norm.encode(a[i:i + (1 << 21)]).astype(np.float32))
    return out


def load_split(path, every):
    """Train/valid arrays, keeping every `every`-th time sample.

    Arrays are loaded one at a time and subsampled straight away so the
    full dataset never sits in memory next to its copies.
    """
    data = np.load(path)
    parts = {}
    for key, width in (("X", 8), ("Y", 2)):
        a = np.ascontiguousarray(data[key][:, ::every])
        n_valid = a.shape[0] // 5
        parts[key] = a[:-n_valid].reshape(-1, width), a[-n_valid:].reshape(-1, width)
        del a
    return parts["X"], parts["Y"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/data_n200_T80.npz")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--symmetric", action="store_true",
                   help="build mirror symmetry (x -> -x) into the network")
    p.add_argument("--sample-every", type=int, default=1,
                   help="keep every n-th stored time sample (memory)")
    p.add_argument("--lr1", type=float, default=0.05)
    p.add_argument("--epochs1", type=int, default=500)
    p.add_argument("--lr2", type=float, default=0.001)
    p.add_argument("--epochs2", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/model_h128.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    log_file = open(Path(args.out).with_suffix(".log"), "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    (X_train, X_valid), (Y_train, Y_valid) = load_split(args.data, args.sample_every)
    x_norm, y_norm = Normalizer.fit(X_train), Normalizer.fit(Y_train)
    train = to_tensor(X_train, x_norm), to_tensor(Y_train, y_norm)
    valid = to_tensor(X_valid, x_norm), to_tensor(Y_valid, y_norm)
    del X_train, Y_train, X_valid, Y_valid
    log(f"train {len(train[0]):,} samples, valid {len(valid[0]):,} samples")

    arch = dict(hidden=args.hidden, symmetric=args.symmetric)
    net = build_net(arch, x_norm, y_norm)
    log(f"architecture {arch}, {n_params(net):,} parameters")
    history = []
    best1 = train_stage(net, train, valid, args.lr1, args.epochs1,
                        name="stage1", log=log, history=history)
    best2 = train_stage(net, train, valid, args.lr2, args.epochs2,
                        name="stage2", log=log, history=history)

    train_loss = evaluate(net, *train)
    meta = dict(arch=arch, sample_every=args.sample_every, data=args.data, lr1=args.lr1, lr2=args.lr2,
                best_valid_stage1=best1, best_valid_stage2=best2,
                final_train_loss=train_loss,
                history=history)
    NeuralClosure(net, x_norm, y_norm, arch).save(args.out, **meta)
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=1))
    log(f"saved {args.out}  (train loss {train_loss:.3e} normalized)")


if __name__ == "__main__":
    main()
