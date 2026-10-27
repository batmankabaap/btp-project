"""Plotting helpers: full/net/MCL comparison lines, 2x2 case layout, PNG export."""

from pathlib import Path

import matplotlib.pyplot as plt

POS = {1: (0, 0), 2: (0, 1), 3: (1, 0), 4: (1, 1)}  # 2x2 layout for CASES 1-4

FIG_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "figures"


def pick(h, q, var):
    """var: 'h' (height) or 'q' (discharge)."""

    return h if var == "h" else q


def save(fig, name, out_dir=None):
    """Save a figure as PNG under artifacts/figures/ (created if needed)."""

    out_dir = Path(out_dir) if out_dir is not None else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")

    return path


def plot_cases(results, x, var, t, title, show=True):
    """2x2 grid (one panel per case) comparing full/net/mcl at time t."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for case_id, r in results.items():
        row, col = POS[case_id]
        ax = axes[row, col]

        ax.plot(x, pick(*r["full"][t], var), "k-", linewidth=2, label="Full model")
        ax.plot(x, pick(*r["net"][t], var), "b-.", linewidth=1.5, label="Net-only")
        ax.plot(x, pick(*r["mcl"][t], var), "r--", linewidth=1.5, label="Net+MCL")

        ax.set_title(f"case #{case_id}, t={t}")
        ax.set_xlabel("x")
        ax.set_ylabel(var)
        ax.grid(True)

    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()

    if show:
        plt.show()

    return fig
