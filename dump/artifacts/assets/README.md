# artifacts/assets — Generated Figures: Specification & Setup

All figures produced from the four Table-1 test cases (`swe/cases.py`) using the
**shipped pre-trained "main" network** (`artifacts/models/main/`, hidden=128).
Common setup for every figure below.

---

## Common Experimental Setup

| Item | Value | Source |
|---|---|---|
| Domain | L = 100.0, periodic, x ∈ (0, L) | `swe/config.py` |
| Gravity | g = 9.812 | `swe/config.py` |
| Fine grid (exact/reference solver) | N_f = 2000 cells, dx = 0.05, dt = 0.005, LLF flux + Heun (SSP-RK2) | `swe/fine.py`, `swe/physics.py` |
| Coarse grid (learned solver) | N_c = 100 cells, dX = 1.0 (coarsening factor k = 20) | `swe/config.py` |
| Coarse time step | **dT = 0.05** (the long-horizon step also used by `scripts/sizes.py` and `scripts/generalize.py`) | — |
| Coarse scheme | LLF flux + NN subgrid correction (± MCL projection), Heun (SSP-RK2) | `swe/coarse.py`, `swe/mcl.py` |
| Network | `main` checkpoint: 8 → 128×3 (LeakyReLU 0.01) → 2, StandardScaler in/out | `swe/net.py::load_net("main")` |
| Test cases | 4 sinusoidal ICs of Table 1 (cases #1–#4) | `swe/cases.py::CASES` |

**Line legend used in the comparison figures:**

| Line | Meaning | Definition |
|---|---|---|
| black solid | **exact** | Fine-mesh DNS (2000 cells, dt=0.005) cell-averaged by k=20 onto the 100-cell coarse grid — the ground truth |
| blue dash-dot | **main NN (no MCL)** | Coarse LLF flux + raw network correction added directly (no limiting) |
| red dashed | **main NN + MCL** | Coarse LLF flux + network correction projected through Monolithic Convex Limiting (`swe/mcl.py::mcl`) |

All curves are plotted on the same 100-cell coarse grid (x = cell centers), so the
comparison is apples-to-apples: "exact" is the best possible 100-cell representation
of the true dynamics.

---

## 1. `ics_t0.png` — Initial Conditions at t = 0

**Purpose:** reference sheet of what each of the 4 test cases looks like before any
evolution.

**Layout:** 2 × 2 grid, one panel per test case (#1 top-left, #2 top-right,
#3 bottom-left, #4 bottom-right).

**Per panel:** 2 lines on the fine grid (N = 2000):
- `h(x, 0)` — water height (blue solid)
- `q(x, 0) = h·v` — discharge (green dashed)

**Panel title** records the case parameters: H0, Ah, kh (height wave) and
V0, Av, kv (velocity wave); phases are in `swe/cases.py`.

**No solver is involved** — this is purely the analytic initial condition
evaluated at fine cell centers, x_i = (i + 0.5)·dx.

**Test-case summary:**

| Case | H0 | Ah | kh | phih | V0 | Av | kv | phiv | Character |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.0 | 0.20 | 3 | 0.00 | 1.0 | 0.0 | 0 | 0.0 | smooth, easy (constant velocity) |
| 2 | 2.0 | 0.45 | 4 | 2.78 | 1.1 | 0.5 | 3 | 4.5 | hardest smooth case (h/v decorrelated) |
| 3 | 2.0 | 0.45 | 4 | 2.78 | 1.1 | 0.5 | 4 | 4.5 | case 2 with velocity wave locked to height wave |
| 4 | 2.0 | 0.20 | 3 | 0.00 | 1.1 | 0.1 | 2 | 0.0 | mild, mixed ripples |

---

## 2. `cases_t200.png` — Exact vs Main-NN vs Main-NN+MCL at t = 200

**Purpose:** the core accuracy comparison at a moderate horizon — can a 100-cell
coarse solver with the learned subgrid flux reproduce the fine-mesh solution?

**Layout:** 2 × 4 grid.
- **Columns** = test cases #1–#4 (left to right).
- **Row 1** = water height h(x, t=200).
- **Row 2** = discharge q(x, t=200).

**Per panel (3 lines):** exact (black solid), main NN no-MCL (blue dash-dot),
main NN + MCL (red dashed) — see the common line legend above.

**How each curve is computed:**
- *exact:* `run_fine` to t=200 with dt=0.005 (40,000 fine steps), then
  `coarsen(h, q, k=20)` → 100 cells.
- *NN / NN+MCL:* `run_coarse` on the 100-cell grid to t=200 with **dT=0.05**
  (4,000 coarse steps, Heun), with `use_net=True` and `use_mcl=False` / `True`
  respectively (`swe/cases.py::run_case`).

**Initial states:** the sinusoidal IC sampled at cell centers of each grid
(`swe/grid.py::ic` on `x_fine` and `x_coarse` respectively).

---

## 3. `cases_t1000.png` — Exact vs Main-NN vs Main-NN+MCL at t = 1000

**Purpose:** long-time stability & accuracy — same three-way comparison after
1,000 seconds of simulated time (200,000 fine steps / 20,000 coarse steps),
which is 5× beyond the longest horizon in the shipped figures (t=400 in
`artifacts/figures/q_t400.png`).

**Layout, lines, and setup:** identical to `cases_t200.png` (2 × 4 grid, columns
= cases #1–#4, rows = h and q, same 3-line legend), except every run is
integrated to **t = 1000**.

**What to look for:**
- Whether NN+MCL still tracks "exact" (no drift, no phase error accumulation).
- Whether raw NN (no MCL) develops spurious oscillations or loses stability at
  long times — the expected demonstration that the convex limiting is what
  carries the method over long horizons.

---

## 3b. `cases_t200_n128.png` — Same as (2) but Using the `n128` Network

**Purpose:** identical experiment to `cases_t200.png` (all 4 cases at t=200,
exact vs NN vs NN+MCL, same layout and line legend), but with the **`n128`
network** (dev-scale: 10 trajectories × 10 s, 200k samples) in place of the
shipped `main` checkpoint.

**Why:** diagnostic comparison against the paper (Fig. 2/3 of
SWE-MatCom). Measured raw subgrid-flux MSE: `n128` = 2.0e-4 (essentially
paper-level; the paper's n=200/T=80 network reports 1.5e-4), while `main` =
9.2e-2 (~600× worse). The dT=0.05 vs 0.005 choice was verified numerically
to change the case-#2 relative L2 error by only ~0.03%, so the network —
not the solver settings — is the dominant factor in any disagreement with
the paper's figures.

**Setup:** identical to the common setup above (coarse N=100, dX=1.0,
dT=0.05, Heun; exact = fine DNS coarsened by k=20). Lines: black solid =
exact, blue dash-dot = n128 NN no-MCL, red dashed = n128 NN + MCL.

---

## 3c. `cases_t200_trained_net.png` and `cases_t200_main_net.png` — Per-Network Regeneration

**Purpose:** the t=200 all-cases comparison (exact vs NN vs NN+MCL) generated
separately for two specific networks, with explicit file names:

| File | Network | Checkpoint | Provenance |
|---|---|---|---|
| `cases_t200_trained_net.png` | **our trained paper-scale network** | `artifacts/models/paper/` | trained in this repo per Timofeyev et al. 2025 spec: n=200 trajectories, T_train=80, skip=10 (32M samples), SGD batch 128, 500 ep @ lr=0.05 (early-stopped ~ep 121) + 2000 ep @ lr=0.001 (early-stopped ~ep 102), final val loss 5.52e-4 |
| `cases_t200_main_net.png` | **shipped "main" network** | `artifacts/models/main/` | byte-identical to the paper-repo's `cls1/data/reduced-llf-2/` checkpoint — i.e. Schwarzmann's *thesis* model, trained at k=40/dX=2 (NOT the paper's Fig. 2/3 configuration) |

**Layout & lines (both files):** identical to `cases_t200.png` — 2×4 grid
(columns = cases #1–#4, rows = h and q), each panel 3 lines: exact (black
solid), NN no-MCL (blue dash-dot), NN+MCL (red dashed).

**Setup:** coarse N=100, dX=1.0, **dT=0.005** (the paper's own Fig. 2/3 time
step, unlike the dT=0.05 used in `cases_t200.png` / `cases_t200_n128.png`),
Heun; exact = fine DNS (N=2000, dt=0.005) coarsened by k=20.

Note: the intermediate `cases_t200_paper.png` (generated mid-training with
the epoch-82 checkpoint) was removed as superseded by
`cases_t200_trained_net.png` (final weights).

---

## 4. `models_case{1,2,3,4}_t200.png` — All Models vs Exact at t = 200 (one file per case)

**Purpose:** cross-model comparison at t=200 — every available network
checkpoint plotted against the exact solution on the same axes, to compare
model provenance/capacity/training-data quantity in one view.

**Layout:** one file per test case (`models_case1_t200.png` ... `models_case4_t200.png`).
Each file: 2 stacked panels — top = water height h(x, t=200), bottom =
discharge q(x, t=200).

**Per panel (13 lines):**

| Line | Meaning |
|---|---|
| black solid (thick) | **exact** — fine DNS (N=2000, dt=0.005) coarsened by k=20 onto the 100-cell grid |
| colored solid | that model **with MCL** |
| colored dotted (same color) | that model **raw NN, no MCL** |

Models included (color coding):

| Label | Checkpoint | Hidden | Trained on | Color |
|---|---|---|---|---|
| `main` | `artifacts/models/main/` | 128 | shipped pre-trained checkpoint (paper-scale, no meta.json) | red |
| `n128` | `artifacts/models/n128/` | 128 | dev-scale: 10 traj × 10 s, 200k samples | blue |
| `n64` | `artifacts/models/n64/` | 64 | same data as n128 | green |
| `n32` | `artifacts/models/n32/` | 32 | same data as n128 | orange |
| `n16` | `artifacts/models/n16/` | 16 | same data as n128 | purple |
| `n128_lo` | `artifacts/models/n128_lo/` | 128 | under-trained: 3 traj × 3 s, 18k samples | brown |

**Solver setup:** identical to the common setup above (coarse N=100, dT=0.05,
Heun; every model run twice per case: `use_mcl=True` and `use_mcl=False`).

**What to look for:** how closely each solid (+MCL) line tracks black
(accuracy per model), and how much the dotted (no-MCL) lines oscillate or
drift away (the stabilizing effect of MCL per model) — expected worst for
`n16` and `n128_lo` without MCL.

**Compatibility note:** the n-family scaler pickles (`in.pkl`/`out.pkl`) were
written under NumPy 2.x and scikit-learn 1.9, so on a NumPy 1.x environment
they must be unpickled behind a temporary `numpy._core -> numpy.core` module
alias (installed **only** around `pickle.load` — leaving it in `sys.modules`
segfaults `torch.load`). The generation script handled this via a local
`load_model()` wrapper.

---

## Reproducibility Notes

- All runs use the **`main`** network — the same checkpoint as
  `artifacts/figures/{q_t200, h_t200, q_t400}.png` (byte-identical to the
  notebook's root-level `model.pkl` + scalers).
- Coarse solver uses **dT = 0.05**, not the dT = 0.005 of `scripts/compare.py`.
  This matches the repo's own long-horizon convention (`sizes.py`,
  `generalize.py`, `timing.py`), keeps t=200 and t=1000 figures mutually
  consistent, and is itself a supported regime (see
  `artifacts/figures/timesteps_case{2,3}.png`).
- Fine reference always uses its native dt = 0.005.
- Figures were generated with the temporary script `scripts/_tmp_assets.py`,
  which was deleted after generation; this README is the permanent record of
  the setup.
