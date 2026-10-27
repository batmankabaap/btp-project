# NN subgrid fluxes + MCL for the 1D shallow-water equations

Reimplementation of Timofeyev, Schwarzmann & Kuzmin, *Application of machine
learning and convex limiting to subgrid flux modeling in the shallow-water
equations*, Math. Comput. Simul. 238 (2025) 163–178.

## Code ↔ paper

| Module | Paper | Contents |
|---|---|---|
| `swe/config.py` | Sec. 4, Table 1, eq. (15) | domain, meshes, time steps, training-IC ranges, test cases |
| `swe/physics.py` | Sec. 2–3, eqs. (2), (4)–(6), (19) | flux, wave speed, LLF flux, bar states |
| `swe/grid.py` | eqs. (8), (9), (15), (32) | cell centres, sine / step ICs, box averaging |
| `swe/solver.py` | eqs. (3), (10), (16), Heun | finite-volume model (fine DNS or coarse + closure ± MCL) |
| `swe/data.py` | eqs. (11)–(13), Sec. 3.1 | exact subgrid flux G, 4-point stencil, batched DNS data generation |
| `swe/network.py` | Sec. 3.1, Fig. 1 | 8 → N_hid ×3 (leaky ReLU 0.01) → 2, normalization, closure wrapper; optional mirror-symmetric wrapper (our addition) |
| `swe/training.py` | Sec. 3.1 | SGD, batch 128, MSE (14), two stages, early stopping |
| `swe/mcl.py` | Sec. 3.2, eqs. (22)–(31) | sequential MCL: height limiter (27), discharge limiter (28), (31) |
| `swe/baselines.py` | Sec. 4, Fig. 6 | Lax–Wendroff target flux for MCL-LW |
| `swe/experiments.py` | Sec. 4 | DNS references, reduced runs, error measures |

| Script | Purpose |
|---|---|
| `scripts/generate_data.py` | n = 200 trajectories, T_train = 80 |
| `scripts/train.py` | stage 1: ≤500 epochs, lr 0.05; stage 2: ≤2000 epochs, lr 0.001 |
| `scripts/evaluate.py` | Figs. 2–4, 6, 7–8, 11–12, 13 and `metrics.json` |
| `scripts/precompute.py` | runs every dashboard sample on every trained network and stores the results |
| `tests/test_compliance.py` | checks every equation above against the paper |
| `dashboard/` | local web app: run DNS / NN / MCL-NN on any test case, paper-vs-ours figures |

A 3-page written report (the maths, training and inference, results) is in `../report/report.pdf`.

## Run

```bash
pip install -r requirements.txt
python -m pytest tests                      # compliance checks
python scripts/generate_data.py             # ~7 min, 1.3 GB
python scripts/train.py                     # ~1-3 h on CPU
python scripts/evaluate.py --model runs/model_h128.pt --out results
```

## Other architectures

Besides the paper's main 3 × 128 network, `scripts/train.py` can train

- `--hidden 32`: one of the paper's four sizes (128, 64, 32, 16), about 10× cheaper per call
  (we trained it on half the data; 64 and 16 were not trained);
- `--symmetric`: a mirror-symmetric network (**not in the paper, our addition**), G(s) = ½ [N(s) + P N(R s)], where
  R reverses the stencil and flips the sign of q and P = diag(−1, 1). Mirroring
  the flow (x → −x) then mirrors the predicted flux exactly. The exact subgrid
  flux has this symmetry (checked in the tests), but training data only
  contains flow to the right.

`--sample-every 2` halves the dataset so training stays under about 1.3 GB of RAM.

```bash
python scripts/train.py --data ../artifacts/final/data_n200_T80.npz --sample-every 2 --hidden 32 --out ../artifacts/final/model_small32.pt
python scripts/train.py --data ../artifacts/final/data_n200_T80.npz --sample-every 2 --symmetric --out ../artifacts/final/model_mirror128.pt
python scripts/precompute.py        # all samples x all models -> ../artifacts/final/precomputed/
```

Results on the 20 dashboard samples (the dashboard's *Model comparison* tab has
the full matrix), q error with MCL:

| | paper 3×128 | small 3×32 | mirror 3×128 |
|---|---|---|---|
| parameters / µs per call | 34k / 113 | 2.5k / 92 | 34k / 175 |
| Table 1 cases | 0.31–0.51% | 0.45–0.86% | 0.40–0.79% |
| flow to the left | 4.27% | 3.78% | **0.69%** |
| dam breaks (3 cases) | 8.1–25.6% | 5.0–13.8% | **3.4–8.4%** |
| shallow / near-dry (3 cases) | 1.1–9.2% | 1.0–9.8% | 0.9–7.1% |

- The mirror network is the most robust: it fixes left-moving flow and roughly
  halves the error on dam breaks, at a slightly higher cost inside the training
  regime (it saw half the data of the paper network).
- The small network is a little less accurate and, without MCL, often goes
  unstable (e.g. 1161% q error on the step case), so the limiter is essential
  for it. It is barely faster: per-call cost is dominated by PyTorch overhead.

## Dashboard

```bash
python dashboard/server.py          # then open http://localhost:8000
```

Every `model_*.pt` in `../artifacts/final/` gets its own panel, side by side, on
every case. Sample cases load instantly from `../artifacts/final/precomputed/`
(`python scripts/precompute.py` rebuilds them, ~10 min); an uploaded or edited
case is run live on all networks. The *Model comparison* tab has a case × model
matrix (q error, h error, oscillation or run time), a per-group summary and a
model table. `http://localhost:8000/?case=13_shallow_H0_1` opens a sample directly.

`dashboard/samples/` holds 20 cases, each with a setup, what to expect and
what actually happens, in four groups: paper cases,
outside-training cases that should still work, tough cases where the NN alone
blows up but MCL-NN survives, and cases where the method breaks (shocks from
strong dam breaks, waves too short for the coarse mesh). Pick one or upload a JSON case of the same form
(`"type": "sine"` with the eq. (15) parameters, or `"type": "step"` with `H0`, `a`;
plus `t_end` ≤ 400 and coarse `dt` in [0.005, 0.1]). It plots the 2000-cell DNS,
plain coarse LLF, and NN-reduced and MCL-NN-reduced h and q for every network and
reports errors and costs. A live t = 200 run at dt = 0.005 takes about 2 minutes.

The trained N_hid = 128 model, its training log, figures and `metrics.json`
from our run are kept outside this folder in `../artifacts/final/`.

## Choices where the paper leaves room

- **Sampling.** The paper samples every fine step (320 M samples). We keep every
  10th step (32 M); consecutive samples are 0.005 apart and nearly identical.
- **skorch defaults.** SGD, batch 128 and early stopping with patience 5 and a
  relative threshold of 1e-4, which are skorch's defaults. Validation is an
  unshuffled 80/20 split, done here by holding out whole trajectories. The
  best weights are restored at the end of each stage.
- **Coarse initial state** is the box average of the fine IC, as eq. (9) defines U.
- **MCL-LW** uses the two-step Richtmyer Lax–Wendroff flux as the target.
