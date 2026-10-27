"""TEMPORARY stage-2 resume script -- deleted after the run completes.

Resumes stage-2 training (lr=0.001, up to 2000 epochs, early stopping
patience=20) from the best checkpoint in artifacts/models/paper/ (which was
stage-2 epoch 6, val=5.95e-4 when the previous process was stopped).

Data is regenerated with the same seed as the original run, so the
train/val split and scaling are identical.
"""

import json
import pickle
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from swe.train import gen_data
from swe.net import Net

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "models" / "paper"

N_TRAJ, T_TRAIN, SKIP, SEED = 200, 80.0, 10, 0
HIDDEN, BATCH = 128, 128
EPOCHS2, LR2 = 2000, 0.001
PATIENCE, THRESH = 20, 1e-8
VAL_FRAC = 0.1
VAL_EVAL_MAX = 1_000_000
START_EPOCH = 7  # history already contains stage2 epochs 1-6

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():

    # ---------------- load best checkpoint + scalers ----------------
    net = Net(hidden=HIDDEN)
    net.load_state_dict(torch.load(OUT / "net.pt", map_location="cpu"))
    net.eval()
    with open(OUT / "in.pkl", "rb") as f:
        x_scaler = pickle.load(f)
    with open(OUT / "out.pkl", "rb") as f:
        y_scaler = pickle.load(f)
    with open(OUT / "history.json") as f:
        history = json.load(f)
    best = min(e["val"] for e in history)
    log(f"resumed from checkpoint: best val so far = {best:.6f}")

    # ---------------- regenerate identical data ----------------
    t0 = time.time()
    log(f"regenerating data (seed={SEED}, n={N_TRAJ}, T={T_TRAIN}, skip={SKIP})")
    X, y = gen_data(N_TRAJ, T_TRAIN, seed=SEED, skip=SKIP)
    log(f"{X.shape[0]} samples in {(time.time()-t0)/60:.1f} min")

    n = X.shape[0]
    perm = rng.permutation(n)
    n_val = round(VAL_FRAC * n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    del perm

    def to_tensor(idx):
        Xt, yt = [], []
        for s in range(0, len(idx), 2_000_000):
            chunk = idx[s:s + 2_000_000]
            Xt.append(torch.tensor(x_scaler.transform(X[chunk]), dtype=torch.float32))
            yt.append(torch.tensor(y_scaler.transform(y[chunk]), dtype=torch.float32))
        return torch.cat(Xt), torch.cat(yt)

    log("tensorizing train split ...")
    Xt, yt = to_tensor(train_idx)
    log("tensorizing val split ...")
    Xv, yv = to_tensor(val_idx)
    del X, y, train_idx, val_idx
    log(f"train: {tuple(Xt.shape)}, val: {tuple(Xv.shape)}")

    if Xv.shape[0] > VAL_EVAL_MAX:
        sub = torch.randperm(Xv.shape[0])[:VAL_EVAL_MAX]
        Xv_e, yv_e = Xv[sub].clone(), yv[sub].clone()
    else:
        Xv_e, yv_e = Xv, yv

    # ---------------- stage 2 ----------------
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.SGD(net.parameters(), lr=LR2)
    bad = 0

    def save_best():
        torch.save(net.state_dict(), OUT / "net.pt")

    for e in range(START_EPOCH, EPOCHS2 + 1):
        net.train()
        order = torch.randperm(Xt.shape[0])
        run, seen = 0.0, 0
        for s in range(0, Xt.shape[0], BATCH):
            idx = order[s:s + BATCH]
            opt.zero_grad()
            loss = loss_fn(net(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            run += loss.item() * len(idx)
            seen += len(idx)

        net.eval()
        with torch.no_grad():
            vl = 0.0
            for s in range(0, Xv_e.shape[0], 65536):
                vl += loss_fn(net(Xv_e[s:s+65536]), yv_e[s:s+65536]).item() * len(Xv_e[s:s+65536])
            vl /= Xv_e.shape[0]

        history.append(dict(stage="stage2", epoch=e, train=run/seen, val=vl))
        improved = vl < best - THRESH
        marker = ""
        if improved:
            best, bad = vl, 0
            save_best()
            marker = " *saved*"
        else:
            bad += 1
        log(f"stage2 epoch {e}/{EPOCHS2}: train={run/seen:.6f} val={vl:.6f}{marker}")

        with open(OUT / "history.json", "w") as f:
            json.dump(history, f)

        if bad >= PATIENCE:
            log(f"stage2: early stopping (no improvement for {PATIENCE} epochs)")
            break

    meta_path = OUT / "meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    meta.update(dict(
        hidden=HIDDEN, n_traj=N_TRAJ, t_train=T_TRAIN, skip=SKIP, seed=SEED,
        n_samples=int(n), epochs2=EPOCHS2, lr2=LR2, batch=BATCH,
        patience=PATIENCE, thresh=THRESH, best_val_final=best,
        note="paper-scale (Timofeyev et al. 2025): n=200, T=80; skip=10; "
             "SGD batch 128; stage1 500ep@0.05 (early-stopped ~ep121), "
             "stage2 2000ep@0.001 with early stopping; resumed once",
    ))
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log(f"DONE. best val = {best:.6f}. saved {OUT}")


if __name__ == "__main__":
    main()
