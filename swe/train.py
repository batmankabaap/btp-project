"""Data generation + training for the subgrid-flux network."""

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from .config import L, dt as dt0, dx as dx0, g as g0, k as k0
from .grid import make_grid, ic, coarsen
from .fine import step as fine_step
from .physics import llf
from .net import Net, stencil


def rand_ic(rng, H0=2.0):
    """One random initial condition (same distributions used across cases)."""

    return dict(
        H0=H0,
        Ah=rng.uniform(0.2, 0.6), kh=rng.integers(1, 7), phih=rng.uniform(0, 2 * np.pi),
        V0=rng.uniform(1.0, 2.0),
        Av=rng.uniform(0.2, 0.6), kv=rng.integers(1, 7), phiv=rng.uniform(0, 2 * np.pi),
    )


def true_flux(h, q, k=k0, g=g0):
    """'True' subgrid flux correction at every coarse interface, from the fine state (h, q)."""

    H, Q = coarsen(h, q, k)

    hl, ql = h[k - 1 :: k], q[k - 1 :: k]
    hr, qr = np.roll(hl, -1), np.roll(ql, -1)
    fh_fine, fq_fine = llf(hl, ql, hr, qr, g=g)

    Hr, Qr = np.roll(H, -1), np.roll(Q, -1)
    fh_bar, fq_bar = llf(H, Q, Hr, Qr, g=g)

    return H, Q, fh_fine - fh_bar, fq_fine - fq_bar


def gen_data(n_traj, t_train, seed=0, skip=10, dt=dt0, dx=dx0, g=g0, k=k0, verbose=True):
    """
    Generate (stencil, true_correction) training pairs from n_traj
    random-IC fine trajectories of length t_train, sampled every
    `skip` fine steps. Returns (X, y) arrays.
    """

    n_f = int(round(L / dx))
    x, _ = make_grid(n_f, L=L)
    n_steps = int(round(t_train / dt))

    rng = np.random.default_rng(seed)
    xs, ys = [], []
    t0 = time.time()

    for traj in range(n_traj):

        h, q = ic(x, **rand_ic(rng), L=L)

        for i in range(n_steps):
            h, q = fine_step(h, q, dt, dx=dx, g=g)
            if i % skip == 0:
                H, Q, gh, gq = true_flux(h, q, k=k, g=g)
                xs.append(stencil(H, Q))
                ys.append(np.column_stack([gh, gq]))

        if verbose:
            print(f"  traj {traj + 1}/{n_traj} done ({time.time() - t0:.1f} s)")

    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32)


def train_net(
    X, y, hidden=128,
    epochs1=30, lr1=0.05, epochs2=100, lr2=0.001,
    batch=128, val_frac=0.1, patience=20, thresh=1e-8,
    seed=0, verbose=True,
):
    """
    Train a Net(hidden) on (X, y): SGD, two learning-rate stages,
    StandardScaler on both input and output fit on the train split.

    Returns (net, x_scaler, y_scaler, history).
    """

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    perm = rng.permutation(X.shape[0])
    n_val = max(1, round(val_frac * X.shape[0]))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    x_scaler = StandardScaler().fit(X[train_idx])
    y_scaler = StandardScaler().fit(y[train_idx])

    Xt = torch.tensor(x_scaler.transform(X[train_idx]).astype(np.float32))
    yt = torch.tensor(y_scaler.transform(y[train_idx]).astype(np.float32))
    Xv = torch.tensor(x_scaler.transform(X[val_idx]).astype(np.float32))
    yv = torch.tensor(y_scaler.transform(y[val_idx]).astype(np.float32))

    net = Net(hidden=hidden)
    loss_fn = nn.MSELoss()
    history = []

    def stage(n_epochs, lr):

        opt = torch.optim.SGD(net.parameters(), lr=lr)
        best, bad = float("inf"), 0

        for e in range(n_epochs):

            net.train()
            order = torch.randperm(Xt.shape[0])
            for s in range(0, Xt.shape[0], batch):
                idx = order[s : s + batch]
                opt.zero_grad()
                loss_fn(net(Xt[idx]), yt[idx]).backward()
                opt.step()

            net.eval()
            with torch.no_grad():
                tl = loss_fn(net(Xt), yt).item()
                vl = loss_fn(net(Xv), yv).item()
            history.append(dict(lr=lr, train=tl, val=vl))

            if verbose and e % max(n_epochs // 10, 1) == 0:
                print(f"    lr={lr:g} epoch {e}/{n_epochs} train={tl:.8f} val={vl:.8f}")

            if vl < best - thresh:
                best, bad = vl, 0
            else:
                bad += 1
                if bad >= patience:
                    if verbose:
                        print(f"    early stop at epoch {e} (lr={lr:g})")
                    break

    if verbose:
        print(f"  stage 1: {epochs1} epochs @ lr={lr1}")
    stage(epochs1, lr1)

    if verbose:
        print(f"  stage 2: {epochs2} epochs @ lr={lr2}")
    stage(epochs2, lr2)

    net.eval()

    return net, x_scaler, y_scaler, history
