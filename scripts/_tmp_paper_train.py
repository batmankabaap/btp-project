"""TEMPORARY paper-scale training script -- deleted after the run completes.

Trains the subgrid-flux network exactly per the paper (Timofeyev et al.,
Math. Comput. Simul. 238 (2025) 163-178, Section 3.1 / 4):

  - n = 200 random-IC trajectories, T_train = 80, on the fine grid
    (N_f=2000, dt=0.005). Paper samples every fine step (skip=1, 320M
    samples); we use skip=10 (32M samples) -- information-equivalent
    deduplication required to fit in 8 GB RAM.
  - Architecture: 8 -> 128 x3 (LeakyReLU 0.01) -> 2  (swe.Net)
  - SGD, batch size 128
  - Stage 1: up to 500 epochs, lr = 0.05
  - Stage 2: up to 2000 epochs, lr = 0.001
  - Early stopping in both stages (patience=20, threshold=1e-8)
  - StandardScaler on inputs and outputs

Saves continuously-updated best-val checkpoint to artifacts/models/paper/
(net.pt, in.pkl, out.pkl, meta.json, history.json).
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
OUT.mkdir(parents=True, exist_ok=True)

N_TRAJ, T_TRAIN, SKIP, SEED = 200, 80.0, 10, 0
HIDDEN, BATCH = 128, 128
EPOCHS1, LR1 = 500, 0.05
EPOCHS2, LR2 = 2000, 0.001
PATIENCE, THRESH = 20, 1e-8
VAL_FRAC = 0.1
VAL_EVAL_MAX = 1_000_000   # val-loss evaluated on a random 1M subset (of 3.2M)
SCALER_FIT_MAX = 2_000_000  # scaler statistics from a 2M subsample

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():

    # ---------------- data ----------------
    t0 = time.time()
    log(f"generating data: n={N_TRAJ}, T={T_TRAIN}, skip={SKIP}, seed={SEED}")
    X, y = gen_data(N_TRAJ, T_TRAIN, seed=SEED, skip=SKIP)
    log(f"{X.shape[0]} samples in {(time.time()-t0)/60:.1f} min "
        f"({X.nbytes/1e9:.2f} GB X, {y.nbytes/1e9:.2f} GB y)")

    # ---------------- split / scale ----------------
    n = X.shape[0]
    perm = rng.permutation(n)
    n_val = round(VAL_FRAC * n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    del perm

    fit_idx = train_idx[:SCALER_FIT_MAX]
    x_scaler = StandardScaler().fit(X[fit_idx])
    y_scaler = StandardScaler().fit(y[fit_idx])
    del fit_idx

    def to_tensor(idx, scaler_x, scaler_y):
        Xt, yt = [], []
        for s in range(0, len(idx), 2_000_000):
            chunk = idx[s:s + 2_000_000]
            Xt.append(torch.tensor(scaler_x.transform(X[chunk]), dtype=torch.float32))
            yt.append(torch.tensor(scaler_y.transform(y[chunk]), dtype=torch.float32))
        return torch.cat(Xt), torch.cat(yt)

    log("scaling + tensorizing train split ...")
    Xt, yt = to_tensor(train_idx, x_scaler, y_scaler)
    log("scaling + tensorizing val split ...")
    Xv, yv = to_tensor(val_idx, x_scaler, y_scaler)
    del X, y, train_idx, val_idx
    log(f"train: {tuple(Xt.shape)}, val: {tuple(Xv.shape)}")

    # val evaluated on a fixed random subset (capped) for speed
    if Xv.shape[0] > VAL_EVAL_MAX:
        sub = torch.randperm(Xv.shape[0])[:VAL_EVAL_MAX]
        Xv_e, yv_e = Xv[sub].clone(), yv[sub].clone()
    else:
        Xv_e, yv_e = Xv, yv

    net = Net(hidden=HIDDEN)
    loss_fn = torch.nn.MSELoss()
    history = []

    def save_best(state):
        torch.save(state, OUT / "net.pt")
        with open(OUT / "in.pkl", "wb") as f:
            pickle.dump(x_scaler, f)
        with open(OUT / "out.pkl", "wb") as f:
            pickle.dump(y_scaler, f)

    def run_stage(name, epochs, lr):

        opt = torch.optim.SGD(net.parameters(), lr=lr)
        best, bad = float("inf"), 0

        for e in range(1, epochs + 1):
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

            history.append(dict(stage=name, epoch=e, train=run/seen, val=vl))
            improved = vl < best - THRESH
            marker = ""
            if improved:
                best, bad = vl, 0
                save_best({k: v.clone() for k, v in net.state_dict().items()})
                marker = " *saved*"
            else:
                bad += 1
            log(f"{name} epoch {e}/{epochs}: train={run/seen:.6f} val={vl:.6f}{marker}")

            with open(OUT / "history.json", "w") as f:
                json.dump(history, f)

            if bad >= PATIENCE:
                log(f"{name}: early stopping (no improvement for {PATIENCE} epochs)")
                break

        # restore best weights before the next stage
        net.load_state_dict(torch.load(OUT / "net.pt", map_location="cpu"))
        return best

    t0 = time.time()
    log(f"=== stage 1: up to {EPOCHS1} epochs, lr={LR1} ===")
    best1 = run_stage("stage1", EPOCHS1, LR1)
    log(f"stage 1 best val: {best1:.6f} ({(time.time()-t0)/60:.1f} min)")

    t0 = time.time()
    log(f"=== stage 2: up to {EPOCHS2} epochs, lr={LR2} ===")
    best2 = run_stage("stage2", EPOCHS2, LR2)
    log(f"stage 2 best val: {best2:.6f} ({(time.time()-t0)/60:.1f} min)")

    meta = dict(
        hidden=HIDDEN, n_traj=N_TRAJ, t_train=T_TRAIN, skip=SKIP, seed=SEED,
        n_samples=int(n), epochs1=EPOCHS1, lr1=LR1, epochs2=EPOCHS2, lr2=LR2,
        batch=BATCH, patience=PATIENCE, thresh=THRESH,
        best_val_stage1=best1, best_val_stage2=best2,
        note="paper-scale (Timofeyev et al. 2025): n=200, T=80; skip=10 "
             "(paper: skip=1) for 8GB RAM; SGD batch 128; 500ep@0.05 + "
             "2000ep@0.001 with early stopping",
    )
    with open(OUT / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log(f"DONE. best val = {best2:.6f}. saved {OUT}")


if __name__ == "__main__":
    main()
