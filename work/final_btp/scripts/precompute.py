"""Pre-compute every sample case for every network, for the dashboard.

Writes one <case>.json (all curves + metrics) per sample and a summary.json
(model info + metrics of all cases) into --out.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
from server import load_closures
from simulation import parse_case, simulate
from swe import config as cfg
from swe.network import n_params


def call_cost(closure, repeats=2000):
    """Microseconds per closure call on a 100-cell state."""
    H = 2 + 0.1 * np.random.default_rng(0).random(cfg.N_COARSE)
    closure(H, H)
    start = time.perf_counter()
    for _ in range(repeats):
        closure(H, H)
    return round(1e6 * (time.perf_counter() - start) / repeats, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default=str(ROOT.parent / "artifacts" / "final"))
    p.add_argument("--samples", default=str(ROOT / "dashboard" / "samples"))
    p.add_argument("--cases", nargs="*", help="only these sample names (default: all)")
    args = p.parse_args()

    artifacts = Path(args.artifacts)
    out = artifacts / "precomputed"
    out.mkdir(exist_ok=True)
    closures = load_closures(artifacts.glob("model_*.pt"))

    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else dict(cases={})
    summary["models"] = {
        name: dict(arch=c.arch, params=n_params(c.net), us_per_call=call_cost(c),
                   best_valid=c.meta.get("best_valid_stage2"))
        for name, c in closures.items()
    }

    for path in sorted(Path(args.samples).glob("*.json")):
        if args.cases and path.stem not in args.cases:
            continue
        case = json.loads(path.read_text())
        start = time.time()
        result = simulate(closures, case)
        ic, t_end, dt = parse_case(case)
        (out / f"{path.stem}.json").write_text(
            json.dumps(dict(name=case["name"], category=case.get("category"), **result)))
        summary["cases"][path.stem] = dict(name=case["name"], category=case.get("category"),
                                           t_end=t_end, dt=dt, metrics=result["metrics"])
        summary_path.write_text(json.dumps(summary))
        print(f"{path.stem}: {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
