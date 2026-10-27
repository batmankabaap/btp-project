"""Results figure for the report, drawn from the pre-computed dashboard curves."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PRE = Path(__file__).resolve().parents[1] / "artifacts" / "final" / "precomputed"
nan = lambda a: np.array([np.nan if v is None else v for v in a])

BLACK, OLIVE, BLUE, RED, GREEN = "#111111", "#9a7b1f", "#1f77b4", "#d62728", "#2a9d5c"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 7.6, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "font.family": "DejaVu Sans"})


def load(name):
    d = json.loads((PRE / f"{name}.json").read_text())
    return d, nan(d["x"]), nan(d["x_fine"])


fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.35))

# (a) a paper case: the net restores what plain coarse LLF smears away
d, x, xf = load("02_table1_ic2")
ax = axes[0]
ax.plot(xf, nan(d["fine"]["q"]), color=BLACK, lw=1.4, label="fine DNS (2000 cells)")
ax.plot(x, nan(d["curves"]["llf"]["q"]), color=OLIVE, lw=1.3, ls=":", label="coarse LLF, no net")
ax.plot(x, nan(d["curves"]["h128/nn"]["q"]), color=BLUE, lw=1.1, ls="-.", label="coarse + net")
ax.plot(x, nan(d["curves"]["h128/mcl"]["q"]), color=RED, lw=1.1, ls="--", label="coarse + net + MCL")
ax.set(title="(a) q at t = 200, paper case IC 2", xlabel="x", ylabel="q", ylim=(1.62, 2.5))
ax.legend(frameon=False, fontsize=6.2, loc="lower left")

# (b) shallow water: the net alone breaks, MCL keeps h positive
d, x, xf = load("13_shallow_H0_1")
ax = axes[1]
ax.plot(xf, nan(d["fine"]["h"]), color=BLACK, lw=1.4)
ax.plot(x, nan(d["curves"]["llf"]["h"]), color=OLIVE, lw=1.3, ls=":")
ax.plot(x, nan(d["curves"]["h128/mcl"]["h"]), color=RED, lw=1.1, ls="--")
ax.set(title="(b) h at t = 100, mean depth 1.0", xlabel="x", ylabel="h", ylim=(0.895, 1.04))
ax.text(0.04, 0.05, "net alone: h < 0, run blows up\n(training depths were >= 1.4)", transform=ax.transAxes,
        color=BLUE, fontsize=6.5, va="bottom")

# (c) flow to the left: the mirror-symmetric net fixes what the plain one misses
d, x, xf = load("11_negative_V0")
ax = axes[2]
ax.plot(xf, nan(d["fine"]["q"]), color=BLACK, lw=1.4, label="fine DNS")
ax.plot(x, nan(d["curves"]["h128/mcl"]["q"]), color=RED, lw=1.1, ls="--", label="paper net + MCL")
ax.plot(x, nan(d["curves"]["mirror128/mcl"]["q"]), color=GREEN, lw=1.3, label="mirror-symmetric net + MCL")
ax.set(title="(c) q at t = 100, flow to the left", xlabel="x", ylabel="q", ylim=(-3.5, -2.35))
ax.legend(frameon=False, fontsize=6.2, loc="upper right", ncol=1)

fig.tight_layout(pad=0.6, w_pad=1.0)
fig.savefig("fig_results.png", dpi=220)
print("saved")
