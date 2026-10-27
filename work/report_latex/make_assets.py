"""Figures and tables for report.tex, generated from the dashboard's pre-computed results.

Paper network = "h128", our mirror-symmetric network = "mirror128". Both are compared with MCL on
(the full method) unless a column says "net alone".
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PRE = HERE.parent / "artifacts" / "final" / "precomputed"
PAPER, OURS = "h128", "mirror128"

BLACK, GREY, VERMILION, BLUE = "#111111", "#8a8a82", "#d55e00", "#0072b2"
CATEGORIES = [("paper", "Paper cases"), ("generalization", "Outside the training range"),
              ("tough", "Tough cases"), ("breaks", "Cases where the method breaks")]

# LaTeX-ready names (also valid matplotlib mathtext)
NAMES = {
    "01_table1_ic1": "Table 1, IC 1 (small waves)", "02_table1_ic2": "Table 1, IC 2",
    "03_table1_ic3": "Table 1, IC 3", "04_table1_ic4": "Table 1, IC 4",
    "05_ic1_long_t400": "IC 1, long run $t{=}400$", "06_step_fig13": "Step IC, $a{=}0.5$",
    "07_H0_2p5": "Depth $H_0{=}2.5$", "08_H0_1p25": "Depth $H_0{=}1.25$", "09_H0_3p5": "Depth $H_0{=}3.5$",
    "10_fast_V0_4": "Fast flow $V_0{=}4$", "11_negative_V0": "Flow to the left $V_0{=}{-}1.5$",
    "12_large_dt_0p1": "Short waves, $\\Delta T{=}0.1$",
    "13_shallow_H0_1": "Shallow water $H_0{=}1.0$", "14_very_shallow_H0_0p6": "Very shallow $H_0{=}0.6$",
    "15_near_dry": "Near-dry troughs, $A_h{=}1.7$", "16_dam_break_a2p5": "Strong dam break, $a{=}2.5$",
    "17_dam_break_shallow": "Dam break into shallow water", "18_dam_break_a1p5": "Medium dam break, $a{=}1.5$",
    "19_short_waves_k15": "Short waves, $k{=}15$", "20_short_waves_k20": "Short waves, $k{=}20$",
}

summary = json.loads((PRE / "summary.json").read_text())
cases = summary["cases"]


def err(case, model, kind, comp="q_err"):
    return cases[case]["metrics"][f"{model}/{kind}"][comp]


def pct(v):
    return None if v is None else f"{100 * v:.2f}\\%"


def median(values):
    v = sorted(np.inf if x is None else x for x in values)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def cell(v, bold=False):
    if v is None or not np.isfinite(v):
        return "\\textcolor{gray}{\\textit{blew up}}"
    s = f"{100 * v:.2f}\\%"
    return f"\\textbf{{{s}}}" if bold else s


# ------------------------------------------------------------------ headline numbers (printed for checking)
ids = list(cases)
paper_mcl = [err(c, PAPER, "mcl") for c in ids]
ours_mcl = [err(c, OURS, "mcl") for c in ids]
wins_ours = sum(o < p for o, p in zip(ours_mcl, paper_mcl))
print("cases where ours+MCL has lower q error:", wins_ours, "of", len(ids))
for key, _ in CATEGORIES:
    sel = [c for c in ids if cases[c]["category"] == key]
    w = sum(err(c, OURS, "mcl") < err(c, PAPER, "mcl") for c in sel)
    print(f"  {key}: ours better in {w} of {len(sel)}; medians paper {100*median([err(c,PAPER,'mcl') for c in sel]):.2f}%"
          f" ours {100*median([err(c,OURS,'mcl') for c in sel]):.2f}%")
print("overall medians", 100 * median(paper_mcl), 100 * median(ours_mcl))
print("net alone blows up: paper", sum(err(c, PAPER, "nn") is None for c in ids),
      "ours", sum(err(c, OURS, "nn") is None for c in ids))
print("dam breaks reduction:", [round(100 * (1 - err(c, OURS, "mcl") / err(c, PAPER, "mcl"))) for c in
                                ("16_dam_break_a2p5", "17_dam_break_shallow", "18_dam_break_a1p5")])
print("left flow:", err("11_negative_V0", PAPER, "mcl"), err("11_negative_V0", OURS, "mcl"))
print("models:", summary["models"][PAPER], summary["models"][OURS])

# ------------------------------------------------------------------ tables
rows = ""
for key, title in CATEGORIES:
    sel = [c for c in ids if cases[c]["category"] == key]
    llf = median([err(c, "h128", "mcl") if False else cases[c]["metrics"]["llf"]["q_err"] for c in sel])
    p, o = median([err(c, PAPER, "mcl") for c in sel]), median([err(c, OURS, "mcl") for c in sel])
    bp = sum(err(c, PAPER, "nn") is None for c in sel)
    bo = sum(err(c, OURS, "nn") is None for c in sel)
    w = sum(err(c, OURS, "mcl") < err(c, PAPER, "mcl") for c in sel)
    rows += (f"{title} ({len(sel)}) & {cell(llf)} & {cell(p, p < o)} & {cell(o, o <= p)} & {w} of {len(sel)} & {bp} / {bo} \\\\\n")
llf_all = median([cases[c]["metrics"]["llf"]["q_err"] for c in ids])
p, o = median(paper_mcl), median(ours_mcl)
rows += ("\\midrule\n" + f"All cases ({len(ids)}) & {cell(llf_all)} & {cell(p, p < o)} & {cell(o, o <= p)} & "
         f"{wins_ours} of {len(ids)} & {sum(err(c, PAPER, 'nn') is None for c in ids)} / "
         f"{sum(err(c, OURS, 'nn') is None for c in ids)} \\\\\n")
HEAD_SUMMARY = ("\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
                "Category (cases) & Coarse LLF & \\paper{} + MCL & \\ours{} + MCL & \\ours{} lower in & Net alone blows up \\\\\n"
                " & (no network) & & & (cases) & (\\paper{} / \\ours{}) \\\\ \\midrule\n")
(HERE / "tab_summary.tex").write_text(HEAD_SUMMARY + rows + "\\bottomrule\n\\end{tabular}\n")

rows = ""
for key, title in CATEGORIES:
    rows += f"\\multicolumn{{6}}{{l}}{{\\textit{{{title}}}}} \\\\\n"
    for c in [c for c in ids if cases[c]["category"] == key]:
        p, o = err(c, PAPER, "mcl"), err(c, OURS, "mcl")
        rows += (f"{NAMES[c]} & {cell(cases[c]['metrics']['llf']['q_err'])} & {cell(err(c, PAPER, 'nn'))} & "
                 f"{cell(p, p < o)} & {cell(err(c, OURS, 'nn'))} & {cell(o, o <= p)} \\\\\n")
    rows += "\\addlinespace\n" if key != "breaks" else ""
HEAD_FULL = ("\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
             "Case & Coarse LLF & \\paper{} alone & \\paper{} + MCL & \\ours{} alone & \\ours{} + MCL \\\\ \\midrule\n")
(HERE / "tab_full.tex").write_text(HEAD_FULL + rows + "\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ sample-output figures
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "axes.titlesize": 7.8})


def nan(a):
    return np.array([np.nan if v is None else v for v in a], float)


def panel(ax, case, tag):
    d = json.loads((PRE / f"{case}.json").read_text())
    x, xf = nan(d["x"]), nan(d["x_fine"])
    ax.plot(xf, nan(d["fine"]["q"]), color=BLACK, lw=1.3, label="fine DNS (2000 cells)")
    ax.plot(x, nan(d["curves"]["llf"]["q"]), color=GREY, lw=0.9, ls=":", label="plain coarse LLF")
    ax.plot(x, nan(d["curves"][f"{PAPER}/mcl"]["q"]), color=VERMILION, lw=1.2, ls="--", label="Paper network + MCL")
    ax.plot(x, nan(d["curves"][f"{OURS}/mcl"]["q"]), color=BLUE, lw=1.3, label="Ours (mirror) + MCL")
    pool = np.concatenate([nan(d["fine"]["q"]), nan(d["curves"][f"{PAPER}/mcl"]["q"]),
                           nan(d["curves"][f"{OURS}/mcl"]["q"])])
    lo, hi = np.nanmin(pool), np.nanmax(pool)
    pad = 0.12 * (hi - lo or 1)
    ax.set_ylim(lo - pad, hi + pad)
    p, o = err(case, PAPER, "mcl"), err(case, OURS, "mcl")
    ax.set_title(f"{tag}  {NAMES[case]}\nerror in $q$: Paper {100*p:.2f}%, Ours {100*o:.2f}%", loc="left", pad=3)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$q$")


def figure(selection, name):
    fig, axes = plt.subplots(2, 2, figsize=(6.85, 4.1))
    for ax, (case, tag) in zip(axes.ravel(), selection):
        panel(ax, case, tag)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7.2, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=1.0, w_pad=1.2)
    fig.savefig(HERE / name)
    plt.close(fig)


figure([("01_table1_ic1", "(a)"), ("02_table1_ic2", "(b)"), ("11_negative_V0", "(c)"), ("08_H0_1p25", "(d)")],
       "fig_samples_A.pdf")
figure([("13_shallow_H0_1", "(a)"), ("16_dam_break_a2p5", "(b)"), ("18_dam_break_a1p5", "(c)"),
        ("19_short_waves_k15", "(d)")], "fig_samples_B.pdf")
print("assets written")
