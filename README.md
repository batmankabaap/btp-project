# BTP: Learned subgrid fluxes with convex limiting for shallow water

Reimplementation of Timofeyev, Schwarzmann & Kuzmin, *Application of machine learning
and convex limiting to subgrid flux modeling in the shallow-water equations*,
Math. Comput. Simul. 238 (2025) 163–178, plus a mirror-symmetric network extension.

A neural network predicts the subgrid flux of a cheap 100-cell finite-volume solver,
and a convex limiter (MCL) keeps the result physical. The target is a 2000-cell
reference simulation.

## Layout

| Folder | What it is |
|---|---|
| [`work/`](work/README.md) | Everything current: clean code, trained models, results, reports, the paper |
| [`dump/`](dump/README.md) | The earlier implementation and its outputs, superseded by `work/`; kept for reference only |

Start with `work/`. Nothing in `work/` depends on `dump/`.
