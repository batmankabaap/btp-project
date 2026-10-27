"""Reproduce the numerical experiments of Section 4 for one trained network.

Writes figures and metrics.json into --out. Line styles follow the paper:
DNS black solid, NN-reduced blue dash-dot, MCL-NN-reduced red dashed.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swe import config as cfg
from swe.baselines import LaxWendroffClosure
from swe.experiments import dns, reduced, rel_error, total_variation
from swe.grid import cell_centers, coarsen
from swe.network import NeuralClosure

CASES = [cfg.TEST_CASES[i] for i in (1, 2, 3, 4)]
X = cell_centers(cfg.N_COARSE)
XF = cell_centers(cfg.N_FINE)
DNS_STYLE = dict(color="black", lw=1.5, label="DNS, 2000 cells")
LLF_STYLE = dict(color="tab:olive", ls=":", lw=1.4, label="plain coarse LLF")
NN_STYLE = dict(color="tab:blue", ls="-.", lw=1.2, label="NN-reduced")
MCL_STYLE = dict(color="tab:red", ls="--", lw=1.2, label="MCL-NN-reduced")


def grid_2x2(title):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    fig.suptitle(title)
    return fig, axes.ravel()


def fine_and_coarse(cases, t_end, save_at=()):
    """DNS on the fine mesh (for plotting) and its coarse average (for errors)."""
    fine = dns(cases, t_end, save_at, k=1)
    coarse = {t: (coarsen(h, cfg.K), coarsen(q, cfg.K)) for t, (h, q) in fine.items()}
    return fine, coarse


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


def cases_experiment(net, out, metrics):
    """Figs. 2-4: Table 1 cases at t = 200 and t = 400, dT = 0.005."""
    times = (200.0, 400.0)
    fine, ref = fine_and_coarse(CASES, 400.0, times)
    runs = {
        "llf": reduced(CASES, 400.0, save_at=times),
        "nn": reduced(CASES, 400.0, net, limiter=False, save_at=times),
        "mcl": reduced(CASES, 400.0, net, limiter=True, save_at=times),
    }

    for t in times:
        H, Q = ref[t]
        for name, run in runs.items():
            metrics[f"cases_t{t:g}_{name}"] = dict(
                h_rel_l2=rel_error(run[t][0], H).tolist(),
                q_rel_l2=rel_error(run[t][1], Q).tolist(),
                h_min=np.nanmin(run[t][0], axis=-1).tolist(),
                q_total_variation=total_variation(run[t][1]).tolist(),
            )
        metrics[f"cases_t{t:g}_dns"] = dict(q_total_variation=total_variation(Q).tolist())
        metrics[f"cases_t{t:g}_nn_vs_mcl_q_rel_l2"] = \
            rel_error(runs["nn"][t][1], runs["mcl"][t][1]).tolist()

    panels = [("fig2_q_t200.png", 200.0, 1, "q"), ("fig3_h_t200.png", 200.0, 0, "h"),
              ("fig4_q_t400.png", 400.0, 1, "q")]
    for fname, t, comp, label in panels:
        fig, axes = grid_2x2(f"{label}(x, t={t:g}), Table 1 cases 1-4")
        for i, ax in enumerate(axes):
            ax.plot(XF, fine[t][comp][i], **DNS_STYLE)
            ax.plot(X, runs["llf"][t][comp][i], **LLF_STYLE)
            ax.plot(X, runs["nn"][t][comp][i], **NN_STYLE)
            ax.plot(X, runs["mcl"][t][comp][i], **MCL_STYLE)
            ax.set_title(f"IC {i + 1}")
            ax.set_ylabel(label)
        axes[0].legend()
        for ax in axes[2:]:
            ax.set_xlabel("x")
        save(fig, out / fname)


def lax_wendroff_experiment(net, out, metrics):
    """Fig. 6: MCL-LW on 100 and 500 cells vs MCL-NN on 100 cells, IC 1, t = 20."""
    t, dt = 20.0, cfg.DT_COARSE
    case = [cfg.TEST_CASES[1]]
    fine = dns(case, t, k=1)[t][0][0]
    ref100, ref500 = coarsen(fine, cfg.K), coarsen(fine, 4)
    lw100 = reduced(case, t, LaxWendroffClosure(dt, 1.0), limiter=True, dt=dt)[t][0][0]
    lw500 = reduced(case, t, LaxWendroffClosure(dt, 0.2), limiter=True, dt=dt, k=4)[t][0][0]
    mcl = reduced(case, t, net, limiter=True, dt=dt)[t][0][0]

    metrics["fig6_t20"] = dict(
        mcl_lw_100_h_rel_l2=float(rel_error(lw100, ref100)),
        mcl_lw_500_h_rel_l2=float(rel_error(lw500, ref500)),
        mcl_nn_100_h_rel_l2=float(rel_error(mcl, ref100)),
        h_min=dict(dns=float(fine.min()), mcl_lw_100=float(lw100.min()),
                   mcl_lw_500=float(lw500.min()), mcl_nn=float(mcl.min())),
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(cell_centers(cfg.N_FINE), fine, **DNS_STYLE)
    ax.plot(X, lw100, color="tab:blue", ls="--", label="MCL-LW, 100 cells")
    ax.plot(cell_centers(500), lw500, color="tab:blue", ls="-.", label="MCL-LW, 500 cells")
    ax.plot(X, mcl, color="tab:red", ls=":", lw=2, label="MCL-NN, 100 cells")
    ax.set(xlabel="x", ylabel="h", title="h(x, t=20), IC 1")
    ax.legend()
    save(fig, out / "fig6_lax_wendroff.png")


def timestep_experiment(net, out, metrics):
    """Figs. 7-8: MCL-NN with dT = 0.005, 0.05, 0.1 for IC 2 and 3."""
    times, steps = (20.0, 50.0, 100.0), (0.005, 0.05, 0.1)
    cases = [cfg.TEST_CASES[2], cfg.TEST_CASES[3]]
    fine, ref = fine_and_coarse(cases, 100.0, times)
    runs = {dt: reduced(cases, 100.0, net, limiter=True, dt=dt, save_at=times) for dt in steps}
    colors = {0.005: "tab:blue", 0.05: "tab:red", 0.1: "tab:purple"}

    for c, fig_no in ((0, 7), (1, 8)):
        fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
        fig.suptitle(f"MCL-NN with different time steps, IC {c + 2}")
        for r, t in enumerate(times):
            for comp, label in ((0, "h"), (1, "q")):
                ax = axes[r, comp]
                ax.plot(XF, fine[t][comp][c], **DNS_STYLE)
                for dt in steps:
                    ax.plot(X, runs[dt][t][comp][c], ls="--", color=colors[dt], label=f"dT = {dt}")
                ax.set_ylabel(f"{label}, t={t:g}")
        axes[0, 0].legend()
        save(fig, out / f"fig{fig_no}_timesteps_ic{c + 2}.png")

    metrics["timesteps"] = {
        f"dT{dt}_t{t:g}_h_rel_l2": rel_error(runs[dt][t][0], ref[t][0]).tolist()
        for dt in steps for t in times
    }


def generalization_experiment(net, out, metrics):
    """Figs. 11-12: IC 2 with H0 = 1.5 and 2.5 (outside training), t = 100."""
    t, dt = 100.0, 0.05
    cases = [dict(cfg.TEST_CASES[2], H0=H0) for H0 in (1.5, 2.0, 2.5)]
    fine, ref = fine_and_coarse(cases, t)
    fine, ref = fine[t], ref[t]
    runs = {lim: reduced(cases, t, net, limiter=lim, dt=dt)[t] for lim in (False, True)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for i, (ax, H0) in enumerate(zip(axes, (1.5, 2.0, 2.5))):
        ax.plot(XF, fine[0][i], **DNS_STYLE)
        ax.plot(X, runs[True][0][i], **MCL_STYLE)
        ax.set(title=f"H0 = {H0}", xlabel="x", ylabel="h")
    axes[0].legend()
    save(fig, out / "fig11_12_generalization.png")

    metrics["generalization_t100"] = dict(
        H0=[1.5, 2.0, 2.5],
        mcl_h_rel_l2=rel_error(runs[True][0], ref[0]).tolist(),
        nn_h_rel_l2=rel_error(runs[False][0], ref[0]).tolist(),
    )


def nonsmooth_experiment(net, out, metrics):
    """Fig. 13: step IC (32) with H0 = 2.4, a = 0.5, dT = 0.05."""
    times, dt = (10.0, 20.0, 30.0, 40.0), 0.05
    case = [("step", 2.4, 0.5)]
    fine, ref = fine_and_coarse(case, 40.0, times)
    mcl = reduced(case, 40.0, net, limiter=True, dt=dt, save_at=times)
    nn = reduced(case, 10.0, net, limiter=False, dt=dt)[10.0]

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for ax, t in zip(axes[:2].ravel(), times):
        ax.plot(XF, fine[t][0][0], **DNS_STYLE)
        ax.plot(X, mcl[t][0][0], **MCL_STYLE)
        ax.set_title(f"MCL-NN, t = {t:g}")
    axes[2, 0].plot(XF, fine[10.0][0][0], **DNS_STYLE)
    axes[2, 0].plot(X, nn[0][0], **NN_STYLE)
    axes[2, 0].set_title("NN without MCL, t = 10")
    axes[2, 1].axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("h")
    for ax in (axes[2, 0], axes[1, 1]):
        ax.set_xlabel("x")
        ax.tick_params(labelbottom=True)
    axes[0, 0].legend()
    save(fig, out / "fig13_nonsmooth.png")

    metrics["nonsmooth"] = dict(
        mcl_h_rel_l2={t: float(rel_error(mcl[t][0], ref[t][0])[0]) for t in times},
        nn_t10_h_rel_l2=float(rel_error(nn[0], ref[10.0][0])[0]),
        dns_t10_h_tv=float(total_variation(ref[10.0][0])[0]),
        mcl_t10_h_tv=float(total_variation(mcl[10.0][0])[0]),
        nn_t10_h_tv=float(total_variation(nn[0])[0]),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/model_h128.pt")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    net = NeuralClosure.load(args.model)
    metrics = {}
    for experiment in (cases_experiment, lax_wendroff_experiment, timestep_experiment,
                       generalization_experiment, nonsmooth_experiment):
        start = time.time()
        print(f"{experiment.__name__} ...")
        experiment(net, out, metrics)
        print(f"  done in {time.time() - start:.0f} s")
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))


if __name__ == "__main__":
    main()
