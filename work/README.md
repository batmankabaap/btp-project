# work/: current project

| Path | What it is |
|---|---|
| `final_btp/` | Clean, modular code: solver, MCL, network, training, tests, dashboard. See [`final_btp/README.md`](final_btp/README.md) for the module ↔ paper mapping |
| `artifacts/final/` | Outputs the code reads and writes (see below) |
| `report/report.pdf` | 3-page explainer: the maths, training and inference, results |
| `report_latex/` | LaTeX report for the professor: paper network vs our mirror-symmetric network |
| `SWE-MatCom (1).pdf` | The paper being reproduced |
| `BTP.pdf` | Write-up of the end-to-end inference procedure |
| `BTP2026.ipynb` | Original notebook (numerics correct; superseded by `final_btp/`) |

`final_btp/` finds its models and results through the relative path `../artifacts/final`,
so keep the two folders side by side.

## artifacts/final

| Path | What it is |
|---|---|
| `model_h128.pt` | The paper's network, 3 × 128 MLP (trained on 32 M samples) |
| `model_mirror128.pt` | Our mirror-symmetric 3 × 128 network, not in the paper (16 M samples) |
| `model_small32.pt` | The paper's 3 × 32 size variant (16 M samples) |
| `model_*.log`, `model_*.json` | Training logs and loss history |
| `precomputed/` | All 20 dashboard test cases × all networks, used by the dashboard |
| `results/` | Our versions of the paper's figures and `metrics.json` |
| `paper_figures/` | Figures extracted from the paper, for side-by-side comparison |

The training dataset is not here: `python final_btp/scripts/generate_data.py` rebuilds it
in about 7 minutes (1.3 GB).

## Common commands (from `final_btp/`)

```bash
python -m pytest tests                 # 33 checks against the paper's equations
python dashboard/server.py             # dashboard at http://localhost:8000
python scripts/precompute.py           # refresh the dashboard's stored results (~10 min)
```

Reports:

```bash
cd report && python make_report.py                                  # report/report.pdf (Chrome)
cd report_latex && python make_assets.py && tectonic report.tex     # report_latex/report.pdf
```
