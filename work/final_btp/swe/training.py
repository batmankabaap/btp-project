"""Two-stage SGD training with early stopping (Section 3.1).

The paper trains with skorch's NeuralNet defaults: plain SGD, batch size
128, MSE loss, and skorch's EarlyStopping (monitor validation loss,
patience 5, relative threshold 1e-4). Those defaults are reproduced here
in plain PyTorch.
"""

import copy
import time

import torch
from torch import nn


class EarlyStopping:
    def __init__(self, patience=5, threshold=1e-4):
        self.patience = patience
        self.threshold = threshold
        self.best = float("inf")
        self.best_state = None
        self.bad_epochs = 0

    def update(self, loss, model):
        """Record an epoch; returns True when training should stop."""
        if loss < self.best * (1 - self.threshold):
            self.best = loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def evaluate(model, X, Y, chunk=1 << 16):
    loss_fn = nn.MSELoss(reduction="sum")
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            total += loss_fn(model(X[i:i + chunk]), Y[i:i + chunk]).item()
    return total / Y.numel()


def train_stage(model, train, valid, lr, max_epochs, batch_size=128,
                patience=5, threshold=1e-4, name="", log=print, history=None):
    X, Y = train
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    stopper = EarlyStopping(patience, threshold)

    for epoch in range(1, max_epochs + 1):
        start = time.time()
        model.train()
        perm = torch.randperm(len(X))
        Xs, Ys = X[perm], Y[perm]
        running = 0.0
        for i in range(0, len(Xs), batch_size):
            loss = loss_fn(model(Xs[i:i + batch_size]), Ys[i:i + batch_size])
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(Xs[i:i + batch_size])
        del Xs, Ys

        model.eval()
        train_loss = running / len(X)
        valid_loss = evaluate(model, *valid)
        stop = stopper.update(valid_loss, model)
        mark = "*" if stopper.bad_epochs == 0 else " "
        log(f"{name} epoch {epoch:4d}  lr {lr:g}  train {train_loss:.3e}  "
            f"valid {valid_loss:.3e} {mark}  {time.time() - start:.0f} s")
        if history is not None:
            history.append(dict(stage=name, epoch=epoch, lr=lr,
                                train=train_loss, valid=valid_loss))
        if stop:
            log(f"{name} early stop after {epoch} epochs")
            break

    model.load_state_dict(stopper.best_state)
    return stopper.best
