# dump/: superseded material

Earlier implementation and its outputs. Everything here is replaced by `../work/`,
which does not depend on any of it. Kept only for reference; safe to delete.

| Path | What it is | Why it was retired |
|---|---|---|
| `swe/` | First version of the solver/training package | `train.py` built the training target G from the wrong fine cell (used cell 2k−1 instead of the neighbour across the interface) |
| `scripts/` | Experiment scripts for the old package, incl. `_tmp_*` leftovers | All import the old `swe/` |
| `artifacts/models/` | Networks n16/32/64/128, `paper`, `main` | Trained with the wrong target (except `main`, the notebook's original model) |
| `artifacts/figures/`, `artifacts/assets/`, `artifacts/data/` | Figures and timings from the old models | Produced by the old code |
| `artifacts/final_redundant/` | 1.3 GB training dataset and duplicate run logs from the new pipeline | Regenerable with `work/final_btp/scripts/generate_data.py` (~7 min) |
| `model.pkl`, `input_scaler.pkl`, `output_scaler.pkl` | The notebook's saved model and scalers | Only used by the old notebook; no training code behind them |
| `README_old_project.md`, `requirements.txt` | Documentation of the old code | Describes the retired implementation |
