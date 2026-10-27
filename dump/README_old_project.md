# Application of Machine Learning and Convex Limiting to Subgrid Flux Modeling in the Shallow-Water Equations

## 1. Project Overview

This repository contains the implementation and experiments for applying machine learning (neural networks) combined with monolithic convex limiting (MCL) to model subgrid flux corrections in shallow-water equation (SWE) simulations. The project addresses the challenge of maintaining accuracy and stability when solving partial differential equations (PDEs) on coarse computational grids while using fine-grid reference solutions as ground truth.

### Research Problem

When solving hyperbolic PDEs like the shallow-water equations on coarse grids, the subgrid information from finer discretizations is lost, leading to reduced accuracy. This project develops a neural network approach to learn and predict the subgrid flux corrections necessary to recover accuracy on coarse grids. The key innovation is the combination of neural network-predicted corrections with **monolithic convex limiting** (MCL), which ensures that the predicted corrections respect important physical bounds (water height and velocity bounds).

### Main Contribution

The repository demonstrates that:
- A trained neural network can effectively learn to predict subgrid flux corrections for shallow-water equations
- Applying MCL to the network's predictions improves robustness and stability
- The approach generalizes across different initial conditions and test cases
- The method maintains physical realizability (positive water heights, bounded velocities)

### Associated Paper

See **BTP.pdf** in this repository for the complete thesis describing the methodology, experiments, and results. The paper includes:
- Detailed mathematical formulation of the shallow-water equations
- Description of the fine-mesh (DNS) solver and coarse-mesh solver
- Neural network architecture and training procedure
- Comparison with Lax-Wendroff-style higher-order flux corrections
- Results on 4 standard test cases from the literature

---

## 2. Repository Structure

```
btp/
├── BTP.pdf                          # Thesis with full methodology and results
├── BTP2026.ipynb                    # Interactive Jupyter notebook demonstrating the approach
├── README.md                         # This file
├── requirements.txt                  # Python package dependencies
│
├── artifacts/                        # Pre-computed results and trained models
│   ├── data/
│   │   └── timing.json              # Timing benchmarks for different solvers
│   ├── figures/                     # Generated plots from experiments
│   └── models/                      # Pre-trained neural network checkpoints
│       ├── main/                    # Main network (hidden=128, shipped with repo)
│       │   ├── net.pt               # PyTorch model weights
│       │   ├── in.pkl               # Input feature scaler (StandardScaler)
│       │   └── out.pkl              # Output target scaler (StandardScaler)
│       ├── n128/                    # Network with 128 hidden units
│       ├── n64/                     # Network with 64 hidden units
│       ├── n32/                     # Network with 32 hidden units
│       ├── n16/                     # Network with 16 hidden units
│       └── n128_lo/                 # Under-trained network (less data)
│
├── scripts/                         # Executable Python scripts for training and experiments
│   ├── train_one.py                 # Train a single network with specified hyperparameters
│   ├── train_all.py                 # Train multiple networks (n128, n64, n32, n16, n128_lo)
│   ├── compare.py                   # Compare full model vs network vs network+MCL
│   ├── generalize.py                # Test generalization to different coarse time steps
│   ├── hi_order.py                  # Compare with Lax-Wendroff-style higher-order flux
│   ├── nonsmooth.py                 # Test on non-smooth (step) initial conditions
│   ├── sizes.py                     # Compare networks of different hidden unit sizes
│   ├── timesteps.py                 # Compare different coarse time step choices
│   ├── timing.py                    # Benchmark computational performance
│   └── undertrained.py              # Evaluate under-trained network performance
│
└── swe/                             # Core package: shallow-water equation solvers and utilities
    ├── __init__.py
    ├── config.py                    # Global domain, grid, and discretization parameters
    ├── net.py                       # Subgrid-flux neural network + model loading/prediction
    ├── train.py                     # Data generation and network training
    ├── physics.py                   # Physical flux, numerical flux (LLF), bar-state operations
    ├── fine.py                      # Fine-mesh DNS solver (full model)
    ├── coarse.py                    # Coarse-mesh solver with optional network correction
    ├── mcl.py                       # Monolithic Convex Limiting implementation
    ├── cases.py                     # Table 1 test cases and case evaluation driver
    ├── grid.py                      # Grid construction, initial conditions, coarsening
    └── plot.py                      # Plotting utilities for results
```

### Key Modules

#### `swe/config.py`
Global parameters for all simulations:
- **Domain**: Length L = 100.0, gravitational acceleration g = 9.812
- **Fine grid**: N_f = 2000 cells, dx = 0.05, dt = 0.005
- **Coarse grid**: N_c = 100 cells (k=20 coarsening factor), dX = 1.0, dT = 0.005

#### `swe/physics.py`
Shallow-water equation formulation and numerical methods:
- `flux(h, q)`: Physical flux f(u) where u=[h,q]
- `wavespeed()`: Maximum wave speed at cell interfaces
- `llf()`: Local Lax-Friedrichs numerical flux (used in both fine and coarse solvers)
- `bar_state()`: Riemann-averaged state used in MCL
- `hi_flux()`: Higher-order (Lax-Wendroff-style) target flux

#### `swe/fine.py`
Reference fine-mesh (DNS) solver:
- `rhs()`: Compute spatial RHS using LLF fluxes on fine grid
- `step()`: One time step using Heun (SSP-RK2) time integration
- `run_fine()`: Advance to final time, collecting snapshots

#### `swe/coarse.py`
Coarse-mesh solver with optional subgrid correction:
- `rhs()`: Compute spatial RHS with optional network correction and/or MCL
- `step()`: One coarse time step
- `run_coarse()`: Advance coarse solution
- `hi_rhs()`, `hi_step()`, `hi_run_coarse()`: Higher-order baseline alternatives

#### `swe/net.py`
Neural network for subgrid flux prediction:
- `Net`: Architecture: 8 inputs → 3 hidden layers (128 units each) with LeakyReLU → 2 outputs
- `stencil()`: Extract 4-point stencil [h_{i-1}, q_{i-1}, h_i, q_i, h_{i+1}, q_{i+1}, h_{i+2}, q_{i+2}] at each interface
- `load_net()`: Load trained network and its input/output scalers
- `predict()`: Predict raw subgrid flux corrections given coarse state

#### `swe/mcl.py`
Monolithic Convex Limiting for ensuring physical realizability:
- `limit_h()`: Limit height correction to keep bar states within local height bounds
- `limit_q()`: Limit discharge correction to keep bar velocities within local velocity bounds
- `mcl()`: Apply complete MCL projection to raw network predictions

#### `swe/train.py`
Data generation and network training:
- `rand_ic()`: Generate random initial conditions
- `true_flux()`: Compute ground-truth subgrid flux correction from fine-mesh data
- `gen_data()`: Generate training dataset from fine-mesh trajectories
- `train_net()`: Two-stage training (warm-up + main)

#### `swe/cases.py`
Four standard test cases from literature (Table 1 in thesis):
- **Case 1**: Smooth, low amplitude waves
- **Case 2**: Higher amplitude, complex phase relationships
- **Case 3**: Similar to Case 2 with different wavenumber
- **Case 4**: Low amplitude with phase offset

#### `swe/grid.py`
Grid setup and initial conditions:
- `make_grid()`: Create uniform periodic grid
- `ic()`: Sinusoidal initial condition
- `ic_step()`: Step function for non-smooth test
- `coarsen()`: Average fine cells to coarse cells

---

## 3. Requirements and Environment Setup

### Software Requirements

- **Python**: 3.8 or later
- **PyTorch**: 1.9+ (for neural network computations)
- **NumPy**: 1.19+
- **SciPy**: 1.5+
- **Scikit-learn**: 0.23+
- **Matplotlib**: 3.2+

### Installation

#### Option 1: Using pip (Recommended for quick setup)

```bash
# Clone the repository
git clone <repository-url>
cd btp

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Using conda (For conda users)

```bash
# Create conda environment
conda create -n btp python=3.10
conda activate btp

# Install dependencies
pip install -r requirements.txt
# Or manually:
conda install numpy scipy scikit-learn matplotlib
pip install torch
```

### Verification

To verify your installation:

```bash
python -c "
import numpy as np
import scipy
import sklearn
import matplotlib
import torch
print('All packages imported successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'NumPy version: {np.__version__}')
"
```

---

## 4. Data

### Training Data

The training data consists of fine-mesh (high-resolution) simulations of shallow-water equations with randomly sampled initial conditions. The data generation is handled programmatically within the training pipeline.

#### Training Data Generation

Training data is generated on-the-fly during network training using `swe/train.py::gen_data()`:

**Data Characteristics**:
- **Source**: Fine-mesh (N_f = 2000 cells) DNS simulations
- **Random initial conditions**: Parameters sampled from specified distributions:
  - Base water height: H0 ∈ [constant]
  - Height perturbation amplitude: Ah ∈ [0.2, 0.6]
  - Height wavenumber: kh ∈ {1, 2, ..., 6}
  - Height phase: phih ∈ [0, 2π)
  - Base velocity: V0 ∈ [1.0, 2.0]
  - Velocity perturbation amplitude: Av ∈ [0.2, 0.6]
  - Velocity wavenumber: kv ∈ {1, 2, ..., 6}
  - Velocity phase: phiv ∈ [0, 2π)
- **Temporal extent**: Default 10 seconds of simulation per trajectory
- **Sampling frequency**: Every 10 fine time steps (coarse sampling to reduce redundancy)
- **Total samples (default)**: ~200,000 input-output pairs from 10 trajectories

#### Feature Stencil

For each coarse interface, the network receives an 8-dimensional input (4-point stencil):

```
x = [h_{i-1}, q_{i-1}, h_i, q_i, h_{i+1}, q_{i+1}, h_{i+2}, q_{i+2}]
```

Where:
- h: water height at coarse cell centers
- q: discharge (mass flux) at coarse cell centers

#### Training Target

The network predicts the subgrid flux correction at each interface:

```
y = [Δf_h, Δf_q]
```

Where Δf = f_fine - f_coarse is the correction needed to match fine-mesh fluxes.

### Test Cases

Four standard test cases (Table 1 in thesis) with sinusoidal initial conditions:

| Case | H0 | Ah  | kh | phih | V0  | Av  | kv | phiv |
|------|----|----- |----|----- |----|-----|----|----- |
| 1    | 2.0| 0.20| 3  | 0.00| 1.0| 0.0 | 0  | 0.0  |
| 2    | 2.0| 0.45| 4  | 2.78| 1.1| 0.5 | 3  | 4.5  |
| 3    | 2.0| 0.45| 4  | 2.78| 1.1| 0.5 | 4  | 4.5  |
| 4    | 2.0| 0.20| 3  | 0.00| 1.1| 0.1 | 2  | 0.0  |

### Non-smooth Test Case

An additional non-smooth (step function) initial condition is included:

```
h(x) = H0 + a      if x ∈ (L/3, 2L/3)
       H0 - a/2    otherwise
v(x) = 0
```

### Expected Dataset Organization

Training data is generated on-the-fly, so no pre-existing dataset files are required. However, the following directories contain pre-computed artifacts:

```
artifacts/models/main/       # Main trained network (128 hidden units)
├── net.pt                   # Model weights
├── in.pkl                   # Input feature scaler (fitted on training set)
└── out.pkl                  # Output target scaler (fitted on training set)

artifacts/data/
└── timing.json              # Cached timing results
```

---

## 5. Methodology

### Shallow-Water Equations

The shallow-water equations (SWE) in one dimension form a system of hyperbolic conservation laws:

$$\frac{\partial u}{\partial t} + \frac{\partial f(u)}{\partial x} = 0$$

where the state vector is $u = [h, q]^T$ with:
- h: water height
- q: discharge (momentum)

The physical flux is:
$$f(u) = [q, \frac{q^2}{h} + \frac{g h^2}{2}]^T$$

### Fine-Mesh (DNS) Solver

The reference solution uses the **Local Lax-Friedrichs (LLF)** numerical scheme on a fine mesh (N_f = 2000 cells):

**Numerical Flux** at cell interface:
$$F_{LLF} = \frac{1}{2}(f(u_L) + f(u_R)) - \frac{1}{2}\lambda(u_R - u_L)$$

where λ is the maximum wave speed at the interface.

**Time Integration**: Heun method (SSP-RK2) with dt = 0.005

This produces the reference "full model" solutions used for validation.

### Coarse-Mesh Solver

The coarse solver uses the same LLF scheme but on a coarse mesh (N_c = 100 cells, k=20 coarsening factor):

**Base Solver** (without correction):
$$\frac{dU}{dt} = -\frac{1}{dX}(F_i - F_{i-1})$$

where F is computed from the coarse-grid LLF flux.

**With Subgrid Correction**:
$$F_i^{corrected} = F_i^{LLF} + \Delta F_i$$

The correction Δf at each interface is predicted by the neural network.

### Neural Network Architecture

**Input**: 8-dimensional stencil (4-point stencil of [h, q] values at coarse cells)

**Architecture**:
```
Input (8 features)
  ↓
Linear(8 → 128) + LeakyReLU(0.01)
  ↓
Linear(128 → 128) + LeakyReLU(0.01)
  ↓
Linear(128 → 128) + LeakyReLU(0.01)
  ↓
Linear(128 → 2)
Output: [Δf_h, Δf_q]
```

**Design Rationale**:
- Fully connected: Allows learning complex nonlinear relationships from local stencil information
- Shallow depth: 3 hidden layers balance expressiveness and avoids overfitting
- LeakyReLU: Allows gradient flow even for negative pre-activations

### Training Procedure

#### Stage 1: Warm-up
- **Epochs**: 30
- **Learning rate**: 0.05
- **Batch size**: 128
- **Objective**: MSE loss

#### Stage 2: Main Training
- **Epochs**: 100
- **Learning rate**: 0.001 (lower learning rate for fine-tuning)
- **Batch size**: 128
- **Objective**: MSE loss

**Validation**: 10% of training data held out; early stopping with patience=20

**Feature Normalization**: Input and output features are standardized using `StandardScaler` from scikit-learn

### Monolithic Convex Limiting (MCL)

MCL is a post-processing step applied to the network's predicted corrections to ensure they respect physical bounds:

**Height Limiting**:
Ensures that height bar states remain within local bounds:
$$\bar{h}_i ± \frac{\Delta f_h}{λ} ∈ [\min(h), \max(h)]$$

**Discharge Limiting**:
Ensures that velocity bar states remain within local bounds:
$$v̄_i = \frac{\bar{q}_i}{\bar{h}_i} ∈ [\min(v), \max(v)]$$

This ensures physical realizability (positive heights, bounded velocities) even when the network prediction would violate these constraints.

### Comparison: Lax-Wendroff Higher-Order Flux

An alternative baseline using a parameter-free higher-order flux is also implemented:

$$f̃(x) = f(ū)$$

where ū is the Riemann-averaged state, equivalent to a Lax-Wendroff predictor step with Courant number 1. This higher-order flux can optionally be combined with MCL.

---

## 6. Reproducing the Paper's Results

### Quick Start: Evaluate Pre-trained Network

The repository includes a pre-trained network ("main", 128 hidden units) ready for evaluation.

#### 1. Basic Setup
```bash
# Navigate to repository
cd btp

# Install dependencies (if not already done)
pip install -r requirements.txt
```

#### 2. Run Comparison on Test Cases
```bash
# Compare full model vs network vs network+MCL
# This generates plots of water height and discharge at t=200 and t=400
python scripts/compare.py
```

Output: PNG plots in `artifacts/figures/`

#### 3. Interactive Exploration
```bash
# Open Jupyter notebook for step-by-step exploration
jupyter notebook BTP2026.ipynb
```

---

### Full Reproduction: Train Network from Scratch

#### Step 1: Prepare Environment
```bash
cd btp
pip install -r requirements.txt
```

#### Step 2: Train Single Network (Example)
```bash
# Train a network with 128 hidden units
# Note: Uses reduced dataset (10 trajectories, 10 seconds each) for quick testing
python scripts/train_one.py --tag test_network --hidden 128

# This creates: artifacts/models/test_network/
#   ├── net.pt          (trained weights)
#   ├── in.pkl          (input scaler)
#   ├── out.pkl         (output scaler)
#   └── meta.json       (metadata)
```

#### Step 3: Train Multiple Architectures (Full Experiment)
```bash
# Train all network sizes used in the paper (requires 1-2 hours on CPU)
# Trains: n128, n64, n32, n16, and n128_lo (under-trained)
python scripts/train_all.py
```

#### Step 4: Evaluate on All Test Cases
```bash
# Evaluate trained networks on 4 standard test cases
python scripts/compare.py  # Uses "main" network by default
```

#### Step 5: Run Additional Experiments
```bash
# Generalization to different coarse time steps
python scripts/generalize.py

# Test on non-smooth (step) initial conditions
python scripts/nonsmooth.py

# Compare network sizes
python scripts/sizes.py

# Lax-Wendroff higher-order comparison
python scripts/hi_order.py

# Timing benchmarks
python scripts/timing.py

# Under-trained network analysis
python scripts/undertrained.py
```

---

### Key Configuration Parameters

All simulations use parameters defined in `swe/config.py`:

```python
# Domain
L = 100.0              # domain length
g = 9.812              # gravitational acceleration

# Fine grid (DNS reference)
N_f = 2000             # number of fine cells
dx = 0.05              # fine cell width
dt = 0.005             # fine time step

# Coarse grid
k = 20                 # coarsening factor
N_c = 100              # number of coarse cells
dX = 1.0               # coarse cell width
dT0 = 0.005            # default coarse time step
```

### Training Configuration (for `train_one.py`)

```bash
# Default (quick test):
--tag <name>           # Model directory name (required)
--hidden 128           # Hidden layer size (default: 128)
--n-traj 10            # Number of trajectories (default: 10)
--t-train 10.0         # Training time per trajectory in seconds (default: 10.0)
--skip 10              # Sampling skip (collect every Nth step, default: 10)
--epochs1 30           # Stage 1 epochs (default: 30)
--lr1 0.05             # Stage 1 learning rate (default: 0.05)
--epochs2 100          # Stage 2 epochs (default: 100)
--lr2 0.001            # Stage 2 learning rate (default: 0.001)
--batch 128            # Batch size (default: 128)
--seed 0               # Random seed (default: 0)

# For full training (similar to paper):
# --n-traj 100 --t-train 50.0 --epochs1 50 --epochs2 200
```

---

## 7. Experiments and Results

### Experiment 1: Full Model vs. Network vs. Network+MCL

**Purpose**: Evaluate the core contribution on standard test cases

**Setup**: Run on 4 test cases to t=400 with evaluation at t=200 and t=400

**Command**:
```bash
python scripts/compare.py
```

**Output**: Comparison plots showing water height h(x,t) and discharge q(x,t)

**Expected Result**: Network+MCL closely tracks full model, especially with MCL preventing spurious oscillations

---

### Experiment 2: Generalization to Different Coarse Time Steps

**Purpose**: Test robustness when coarse time step differs from training

**Setup**: Train on one coarse time step, evaluate on others

**Command**:
```bash
python scripts/generalize.py
```

**Evaluation Metrics**:
- L² error relative to fine-mesh solution
- Maximum absolute error
- Stability: whether simulation preserves positive heights

---

### Experiment 3: Non-smooth Initial Conditions

**Purpose**: Test on non-smooth (step function) initial conditions not in training set

**Setup**: Step function IC at t=0, evaluate over time

**Command**:
```bash
python scripts/nonsmooth.py
```

**Expected Result**: Network generalizes reasonably well, though accuracy may be lower than smooth cases

---

### Experiment 4: Network Size Comparison

**Purpose**: Study effect of hidden layer size (capacity)

**Setup**: Compare networks with 16, 32, 64, 128 hidden units

**Command**:
```bash
python scripts/sizes.py
```

**Results Table**:
| Hidden Units | Train Loss | Val Loss | Inference Speed |
|-------------|-----------|----------|-----------------|
| 16         | XXX       | XXX      | Fast            |
| 32         | XXX       | XXX      | Fast            |
| 64         | XXX       | XXX      | Medium          |
| 128        | XXX       | XXX      | Medium          |

(Fill in with actual training results)

---

### Experiment 5: Under-trained Network Analysis

**Purpose**: Study impact of training data quantity

**Setup**: Compare network trained on full dataset vs. reduced dataset (3 trajectories, 3 seconds)

**Command**:
```bash
python scripts/undertrained.py
```

**Expected Result**: Under-trained network shows increased errors but still provides some improvement over coarse LLF

---

### Experiment 6: Lax-Wendroff Higher-Order Comparison

**Purpose**: Benchmark against classical higher-order flux method

**Setup**: Compare LLF + network vs. LLF + higher-order flux (both optionally with MCL)

**Command**:
```bash
python scripts/hi_order.py
```

**Result**: Network approach often outperforms or matches classical higher-order methods

---

### Experiment 7: Computational Timing

**Purpose**: Benchmark performance

**Command**:
```bash
python scripts/timing.py
```

**Expected Comparison**:
- Fine-mesh (DNS): ~10 seconds for t=200
- Coarse-mesh (plain LLF): ~0.01 seconds
- Coarse-mesh (+ network): ~0.2 seconds
- Coarse-mesh (+ network + MCL): ~0.3 seconds

**Speedup**: ~50-100x over fine mesh, ~20-30x slower than plain coarse LLF (acceptable trade-off for accuracy)

---

### Comparison with Paper Results

| Experiment | Paper Result | Repo Result | Status |
|-----------|-------------|-----------|--------|
| Case 1 (smooth) | <1% error | — | TBD |
| Case 2 (complex) | <2% error | — | TBD |
| Non-smooth | ~5-10% error | — | TBD |
| Generalization | Success on unseen Δt | — | TBD |

---

## 8. Configuration and Hyperparameters

### Domain Configuration (`swe/config.py`)

```python
L = 100.0           # Physical domain length
g = 9.812           # Gravitational acceleration
N_f = 2000          # Fine grid cells
dx = 0.05           # Fine cell width
dt = 0.005          # Fine time step (= 0.1 * dx)
k = 20              # Coarsening factor
N_c = 100           # Coarse grid cells
dX = 1.0            # Coarse cell width
dT0 = 0.005         # Default coarse time step
```

### Network Architecture (`swe/net.py`)

```python
class Net(nn.Module):
    def __init__(self, hidden=128):
        # hidden: Number of units in each of 3 hidden layers
        # Default: 128
```

### Training Hyperparameters (`swe/train.py`)

```python
# Data generation
n_traj = 10         # Number of random-IC trajectories
t_train = 10.0      # Seconds of simulation per trajectory
skip = 10           # Collect every Nth fine step

# Stage 1 (warm-up)
epochs1 = 30
lr1 = 0.05
batch = 128

# Stage 2 (main)
epochs2 = 100
lr2 = 0.001

# Validation
val_frac = 0.1      # Fraction of data for validation
patience = 20       # Early stopping patience
```

### Key Settings for Reproducing Paper Results

**For exact reproduction of paper results**:
```bash
python scripts/train_one.py \
  --tag main_retrain \
  --hidden 128 \
  --n-traj 100 \
  --t-train 50.0 \
  --skip 10 \
  --epochs1 50 \
  --epochs2 200 \
  --seed 0
```

(Note: This requires substantial computational resources and time)

---

## 9. Troubleshooting and Reproducibility Notes

### Common Issues and Solutions

#### Issue 1: PyTorch GPU/CPU Detection
**Problem**: PyTorch installation without GPU support, or CUDA/GPU unavailable

**Solution**:
```python
# The current implementation uses CPU by default
# To use GPU if available, modify swe/net.py:

# Change:
x = torch.tensor(x_scaler.transform(stencil(h, q)), dtype=torch.float32)
# To:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.tensor(x_scaler.transform(stencil(h, q)), dtype=torch.float32).to(device)
net.to(device)
```

#### Issue 2: Memory Issues During Training
**Problem**: Running out of memory during training on large dataset

**Solution**:
- Reduce batch size: `--batch 64`
- Reduce dataset size: `--n-traj 20 --t-train 20.0`
- Use gradient accumulation (requires code modification)

#### Issue 3: Poor Network Performance
**Problem**: Trained network shows high errors

**Possible causes and solutions**:
1. Insufficient training data: Increase `--n-traj` and `--t-train`
2. Poor learning rate: Try `--lr1 0.1` or `--lr2 0.0001`
3. Insufficient training: Increase `--epochs1` and `--epochs2`
4. Data contamination: Check that random seed is consistent

#### Issue 4: Negative Water Heights in Coarse Simulation
**Problem**: Coarse solver crashes or produces negative heights

**Solution**: Enable MCL:
```python
# In your simulation script
use_mcl=True  # Ensures physical bounds are respected
```

### Reproducibility Notes

#### Random Seed Management
- **Training**: Controlled by `--seed` argument (default: 0)
- **Coarse simulation**: Uses NumPy's `default_rng(seed)` for reproducibility
- **PyTorch**: Seed set in `swe/train.py`

```python
# To ensure reproducibility
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
```

#### Hardware and Software Variations

**Expected variations due to**:
1. **Hardware differences**: CPU vs GPU, different CPU models (floating point variations ~1e-7 to 1e-5)
2. **Library versions**: NumPy, PyTorch version differences (< 1% error expected)
3. **OS differences**: Windows vs Linux (negligible for double precision)

**To minimize variations**:
- Use same PyTorch version (see requirements.txt)
- Use double precision where possible
- Set random seeds consistently

#### Known Limitations and Deviations

1. **Under-trained Networks**: The default training configuration (10 trajectories, 10 seconds) produces a network that works reasonably well for demonstration but is not fully converged. For production use, recommend:
   - At least 100 trajectories
   - At least 50 seconds of simulation per trajectory
   - Potential further hyperparameter tuning

2. **Non-smooth Cases**: Network performance degrades on non-smooth (step function) initial conditions not seen during training. This is expected and can be addressed by including such cases in training.

3. **Extreme Parameters**: Network performance may degrade for parameter ranges far outside training distribution (e.g., very high Courant numbers, very low water heights).

4. **Time Step Dependence**: Networks trained on one coarse time step show some performance degradation on significantly different time steps. Transfer learning or retraining can help.

### Determinism and Reproducibility Checklist

- [ ] Python version consistent
- [ ] NumPy, PyTorch, scikit-learn versions match (see requirements.txt)
- [ ] Random seed set to same value
- [ ] Same hardware (or noted CPU/GPU differences)
- [ ] No concurrent processes modifying model files
- [ ] Input data preprocessed identically

---

## 10. Citation

If you use this code or methodology in your research, please cite:

### For the Thesis
```bibtex
@thesis{btp2026,
  author = {[Author Name]},
  title = {Application of Machine Learning and Convex Limiting to Subgrid Flux Modeling in the Shallow-Water Equations},
  school = {Indian Institute of Technology Kharagpur},
  year = {2026}
}
```

### For the Repository
```bibtex
@software{btp_repo,
  title = {ML-Based Subgrid Flux Modeling for Shallow-Water Equations},
  url = {<repository-url>},
  year = {2026}
}
```

### Related Work

If relevant to your research, consider also citing:

- **Shallow-Water Equations**: See references in BTP.pdf for classical SWE literature
- **Convex Limiting**: [References from MCL literature in thesis]
- **Neural Networks for PDEs**: [References in thesis]

---

## 11. Acknowledgments

### Datasets and References
- Test cases adapted from standard shallow-water equation benchmarks
- Shallow-water equation formulation based on classical hyperbolic PDE theory
- Convex limiting techniques from [specific authors/papers in thesis]

### Libraries and Frameworks
- **PyTorch**: https://pytorch.org/
- **NumPy**: https://numpy.org/
- **SciPy**: https://scipy.org/
- **Scikit-learn**: https://scikit-learn.org/
- **Matplotlib**: https://matplotlib.org/

### Advisors and Contributors
- [Advisor names and affiliations]
- [Institute/Department]

### Computing Resources
- Computations performed on [describe available resources]

---

## Appendix: Quick Reference Commands

### Setup
```bash
pip install -r requirements.txt
```

### Training
```bash
# Quick test
python scripts/train_one.py --tag test --hidden 128 --n-traj 5 --t-train 5.0

# Full experiment
python scripts/train_all.py
```

### Evaluation
```bash
python scripts/compare.py                  # Main comparison
python scripts/generalize.py              # Generalization
python scripts/nonsmooth.py               # Non-smooth ICs
python scripts/sizes.py                   # Network sizes
python scripts/hi_order.py                # Higher-order comparison
python scripts/timing.py                  # Benchmarks
python scripts/undertrained.py            # Under-training analysis
```

### Visualization
```bash
jupyter notebook BTP2026.ipynb
```

---

## Contact and Further Information

For questions or issues:
- See BTP.pdf for full methodology and mathematical details
- Check GitHub Issues for known problems
- Refer to inline code documentation in `swe/` modules

Last updated: [Current Date]
Version: 1.0

