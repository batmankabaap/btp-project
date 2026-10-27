"""Build report.html and print it to report.pdf with headless Chrome.

Tables are filled from the dashboard's pre-computed results, so the numbers in
the report are the numbers on the dashboard. Maths is typeset by KaTeX (CDN).
"""

import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SUMMARY = HERE.parent / "artifacts" / "final" / "precomputed" / "summary.json"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


# ------------------------------------------------------------------ numbers
def median(values):
    v = sorted(np.inf if x is None else x for x in values)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def accuracy_table():
    s = json.loads(SUMMARY.read_text())
    cases = list(s["cases"].values())
    groups = ["paper", "generalization", "tough", "breaks"]
    fmt = lambda v: "blew up" if not np.isfinite(v) else f"{100 * v:.2f}%"

    def row(label, params, key, nn_key=None):
        cells = [fmt(median([c["metrics"][key]["q_err"] for c in cases if c["category"] == g]))
                 for g in groups]
        overall = fmt(median([c["metrics"][key]["q_err"] for c in cases]))
        blow = "–" if nn_key is None else \
            f"{sum(c['metrics'][nn_key]['q_err'] is None for c in cases)} of {len(cases)}"
        return (f"<tr><td>{label}</td><td>{params}</td>"
                + "".join(f"<td>{c}</td>" for c in cells) + f"<td><b>{overall}</b></td><td>{blow}</td></tr>")

    m = s["models"]
    rows = [row("plain coarse LLF (no net)", "–", "llf")]
    for name, label in (("h128", "paper net 3×128 + MCL"), ("mirror128", "mirror-symmetric 3×128 + MCL"),
                        ("small32", "small net 3×32 + MCL")):
        rows.append(row(label, f"{m[name]['params']:,}", f"{name}/mcl", f"{name}/nn"))
    return "".join(rows), len(cases)


# ------------------------------------------------------------------ diagrams
def tspan_sub(main, sub, size=11):
    return f'{main}<tspan dy="3" font-size="{size}">{sub}</tspan><tspan dy="-3"> </tspan>'


def fig_mesh():
    cells = ""
    for i in range(30):
        x = 40 + 26 * i
        hot = i in (9 + 10, 10 + 10)          # the two fine cells either side of interface i+1/2
        fill = "#cfe0ff" if hot else "#f3f3ee"
        cells += f'<rect x="{x}" y="52" width="26" height="26" fill="{fill}" stroke="#9a9a92" stroke-width="1"/>'
    brackets = ""
    for k, name in enumerate(("U<tspan dy=\"3\" font-size=\"11\">i−1</tspan>", "U<tspan dy=\"3\" font-size=\"11\">i</tspan>",
                              "U<tspan dy=\"3\" font-size=\"11\">i+1</tspan>")):
        x0, x1 = 40 + 260 * k + 3, 40 + 260 * (k + 1) - 3
        brackets += (f'<path d="M{x0} 90 V96 H{x1} V90" fill="none" stroke="#444" stroke-width="1.4"/>'
                     f'<text x="{(x0 + x1) / 2}" y="118" text-anchor="middle" font-size="15">{name}</text>')
    return f"""
<svg viewBox="0 0 900 178" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica, Arial, sans-serif" fill="#1c1c1a">
  {cells}
  <line x1="560" y1="40" x2="560" y2="102" stroke="#1f5fbf" stroke-width="2" stroke-dasharray="4 3"/>
  <text x="560" y="30" text-anchor="middle" font-size="14" fill="#1f5fbf">fine flux {tspan_sub("F(u", "ki")}, {tspan_sub("u", "ki+1")}) — uses the two fine cells beside the interface</text>
  {brackets}
  <text x="560" y="146" text-anchor="middle" font-size="14">coarse flux {tspan_sub("F̄(U", "i")}, {tspan_sub("U", "i+1")}) — sees only the two averages</text>
  <text x="560" y="169" text-anchor="middle" font-size="15" font-weight="700" fill="#c0392b">subgrid flux {tspan_sub("G", "i+½", 12)} = fine flux − coarse flux</text>
</svg>"""


def box(x, y, w, h, lines, fill, stroke, bold=False, color="#1c1c1a"):
    weight = ' font-weight="700"' if bold else ""
    top = y + h / 2 - (len(lines) - 1) * 8.5 + 5
    text = "".join(
        f'<text x="{x + w / 2}" y="{top + 17 * i}" text-anchor="middle" font-size="14"{weight} fill="{color}">{t}</text>'
        for i, t in enumerate(lines))
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="1.5"/>' + text)


def arrow(x0, y0, x1, y1, color="#555"):
    return f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{color}" stroke-width="1.6" marker-end="url(#ar)"/>'


def fig_pipeline():
    out = ['<svg viewBox="0 0 900 262" width="100%" xmlns="http://www.w3.org/2000/svg" '
           'font-family="Helvetica, Arial, sans-serif" fill="#1c1c1a"><defs>'
           '<marker id="ar" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
           '<path d="M0,0 L8,4 L0,8 z" fill="#555"/></marker></defs>']

    out.append('<text x="0" y="14" font-size="14" font-weight="700" fill="#1f5fbf">TRAINING — offline, once (about 1 hour)</text>')
    train = [("200 random sine", "initial", "conditions"), ("Fine DNS,", "2000 cells,", "T = 80 each"),
             ("Average to 100", "cells; exact G", "per interface"), ("Stencil → G:", "8 inputs,", "2 targets"),
             ("Standardise,", "SGD, 2 stages,", "early stopping"), ("Trained net", "34,434", "parameters")]
    w, gap = 130, 24
    for i, lines in enumerate(train):
        x = i * (w + gap)
        last = i == len(train) - 1
        out.append(box(x, 24, w, 60, lines, "#1f5fbf" if last else "#eaf1fd", "#1f5fbf",
                       bold=last, color="#ffffff" if last else "#1c1c1a"))
        if i:
            out.append(arrow(x - gap + 1, 54, x - 1, 54))

    out.append('<text x="0" y="118" font-size="14" font-weight="700" fill="#c0392b">INFERENCE — online, every time step (no fine data needed)</text>')
    inf = [("Coarse state U", "100 cells"), ("Net on each", "4-cell stencil", "→ raw G̃"),
           ("MCL limiter", "G̃ → G̃*", "(keeps h > 0)"), ("F = F̄ + G̃*", "at every", "interface"),
           ("Heun update", "(2 stages)", "→ next U")]
    w, gap = 150, 37.5
    for i, lines in enumerate(inf):
        x = i * (w + gap)
        hot = i == 2
        out.append(box(x, 138, w, 60, lines, "#c0392b" if hot else "#fdeeea", "#c0392b",
                       bold=hot, color="#ffffff" if hot else "#1c1c1a"))
        if i:
            out.append(arrow(x - gap + 1, 168, x - 1, 168))
    out.append(box(187.5, 218, 150, 36, ("LLF flux F̄ from U",), "#f3f3ee", "#8a8a82"))
    out.append('<path d="M75 199 V236 H186" fill="none" stroke="#555" stroke-width="1.6" marker-end="url(#ar)"/>')
    out.append('<path d="M338 236 H562 V199" fill="none" stroke="#555" stroke-width="1.6" marker-end="url(#ar)"/>')
    out.append('<path d="M825 134 V126 H75 V134" fill="none" stroke="#555" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#ar)"/>')
    out.append('<text x="680" y="130" text-anchor="middle" font-size="12" fill="#6a6a64" '
               'style="paint-order:stroke" stroke="#fff" stroke-width="4">repeat for every time step</text>')
    out.append("</svg>")
    return "".join(out)


# ------------------------------------------------------------------ page
HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Neural subgrid fluxes with convex limiting — report</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'\\(',right:'\\)',display:false}]})"></script>
<style>
  @page { size: A4; margin: 11mm 13mm 11mm 13mm; }
  * { box-sizing: border-box; }
  body { font: 9.3pt/1.4 "Helvetica Neue", Helvetica, Arial, sans-serif; color: #1c1c1a; margin: 0; }
  h1 { font-size: 17pt; line-height: 1.15; margin: 0 0 2px; }
  .sub { color: #5d5d57; font-size: 8.6pt; margin: 0 0 7px; }
  h2 { font-size: 11.3pt; margin: 9px 0 3px; color: #1f5fbf; border-bottom: 1px solid #d9e2f3; padding-bottom: 1px; }
  h3 { font-size: 9.2pt; margin: 6px 0 1px; }
  h2, h3 { break-after: avoid; }
  p { margin: 3px 0; }
  .katex { font-size: 1.03em; }
  .katex-display { margin: 3px 0 !important; }
  .callout { background: #eef3fb; border-left: 3px solid #1f5fbf; padding: 4px 8px; margin: 5px 0; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 0 14px; }
  ol, ul { margin: 2px 0 2px 15px; padding: 0; } li { margin: 1.2px 0; }
  figure { margin: 4px 0; } figcaption { color: #4a4a45; font-size: 8pt; margin-top: 1px; }
  table { border-collapse: collapse; width: 100%; font-size: 8pt; margin: 3px 0; }
  th, td { border-bottom: 1px solid #dcdcd5; padding: 2px 4px; text-align: right; }
  th:first-child, td:first-child { text-align: left; } th { color: #55554f; font-weight: 600; }
  code { font: 7.6pt ui-monospace, Menlo, monospace; background: #f1f1ec; padding: 0 2px; border-radius: 2px; }
  .small { font-size: 8pt; color: #4a4a45; }
  table.map td:last-child, table.map th:last-child { text-align: left; }
  .gloss { display: grid; grid-template-columns: 1fr 1fr; gap: 1px 16px; font-size: 8.3pt; margin: 3px 0; }
  .gloss div { break-inside: avoid; } .gloss b { color: #1f5fbf; }
  .pagebreak { break-before: page; }
  .nobreak { break-inside: avoid; }
</style></head><body>

<h1>Neural subgrid fluxes with convex limiting for the shallow-water equations</h1>
<p class="sub">What we built, the maths behind it, and how training and inference work. Reimplementation of
Timofeyev, Schwarzmann &amp; Kuzmin, <i>Math. Comput. Simul.</i> 238 (2025) 163–178.</p>

<h2>1. The idea</h2>
<p>The shallow-water equations describe a water surface (a river, a coastal sea) by two fields: the depth
\(h(x,t)\) and the discharge \(q=hv\) (depth × velocity). Resolving every wave takes a <b>fine mesh</b>: here
2000 cells and a time step of 0.005. A <b>coarse mesh</b> of 100 cells is 20× smaller and tolerates a much larger
time step, but the standard scheme (local Lax–Friedrichs, LLF) is numerically diffusive and smears the waves away.</p>
<p>The paper's remedy: keep the cheap coarse solver and add a <b>correction flux</b> \(\tilde G\) predicted by a small
neural network that was trained on fine-mesh data. A network can output anything, including a correction that drives
the depth negative, so a limiter (<b>MCL</b>, monolithic convex limiting) clips \(\tilde G\) to a range in which the
update is provably safe.</p>
<div class="callout"><b>One line:</b> coarse state \(U\) → coarse LLF flux \(\bar F\) + network correction \(\tilde G\) → MCL clips it to
\(\tilde G^{*}\) → flux \(F=\bar F+\tilde G^{*}\) → time step → next coarse state.</div>
<div class="gloss">
<div><b>DNS</b> — direct numerical simulation: the fine 2000-cell run, treated as the truth.</div>
<div><b>Flux</b> — how much \(h\) or \(q\) crosses a cell interface per unit time.</div>
<div><b>LLF</b> — local Lax–Friedrichs: a robust but diffusive way to compute the flux between two cells.</div>
<div><b>Stencil</b> — the neighbouring cells a formula looks at (here 4 cells: \(i-1,\dots,i+2\)).</div>
<div><b>Bar state</b> — the averaged solution of the two-cell problem; the building block of MCL.</div>
<div><b>CFL condition</b> — the time step must be small enough that no wave crosses a whole cell per step.</div>
<div><b>MCL</b> — monolithic convex limiting: clips a correction so bounds such as \(h>0\) hold.</div>
<div><b>Heun</b> — a second-order, two-stage time-stepping method.</div>
<div><b>Standardise</b> — shift and scale data to zero mean, unit variance before the network sees it.</div>
<div><b>Early stopping</b> — stop training when the validation loss stops improving (guards against over-fitting).</div>
</div>
<figure>@@FIG_MESH@@<figcaption><b>Fig. 1.</b> Fine and coarse mesh (k = 20 fine cells per macrocell in the real setup, 10 drawn).
The exact subgrid flux \(G\) is what the coarse scheme gets wrong at an interface: the flux the fine solution really carries across it
minus the flux the coarse solver computes from its averages alone.</figcaption></figure>

<h2>2. The maths, step by step</h2>
<h3>2.1 Equations</h3>
<p>With \(u=[h,q]^T\) on a periodic domain \((0,L)\), \(L=100\), \(g=9.812\):</p>
$$u_t+f(u)_x=0,\qquad f(u)=\Big[\,q,\ \tfrac{q^2}{h}+\tfrac12 g h^2\Big]^T .$$
<h3>2.2 The fine scheme (the "truth", DNS)</h3>
<p>Finite volumes on \(N_f=2000\) cells (\(\Delta x=0.05\)). The LLF flux through interface \(i+\tfrac12\), with the largest local wave speed \(\lambda\):</p>
$$F_{i+1/2}=\tfrac12\big(f(u_i)+f(u_{i+1})\big)-\tfrac12\lambda_{i+1/2}(u_{i+1}-u_i),\quad
\lambda_{i+1/2}=\max\!\big(|v_i|+\sqrt{gh_i},\ |v_{i+1}|+\sqrt{gh_{i+1}}\big),\quad
\tfrac{d}{dt}u_i=\tfrac{F_{i-1/2}-F_{i+1/2}}{\Delta x}.$$
<p>Time stepping is Heun's method (two Euler stages averaged). The <i>bar state</i>
\(\bar u_{i+1/2}=\tfrac12(u_i+u_{i+1})-\tfrac{f(u_{i+1})-f(u_i)}{2\lambda_{i+1/2}}\) rewrites each Euler stage as a convex combination of \(u_i\) and bar states
whenever \(\tfrac{\Delta t}{\Delta x}(\lambda_{i-1/2}+\lambda_{i+1/2})\le1\) (the CFL condition), which is why the depth stays positive.</p>
<h3>2.3 Coarse variables and the exact subgrid flux</h3>
<p>Group \(k=20\) fine cells into a macrocell and average: \(U_i=\tfrac1k\sum_{j\in\text{cell }i}u_j\) (\(N_c=100\), \(\Delta X=1\)). Averaging the fine equation gives an <b>exact</b> coarse equation:</p>
$$\tfrac{d}{dt}U_i=\frac{(\bar F_{i-1/2}+G_{i-1/2})-(\bar F_{i+1/2}+G_{i+1/2})}{\Delta X},\qquad
G_{i+1/2}=F(u_{ki},u_{ki+1})-\bar F(U_i,U_{i+1}),$$
<p>where \(\bar F\) is the same LLF formula applied to the coarse averages. \(G\) is "anti-diffusive": coarse LLF smooths, \(G\) sharpens it back
(we checked this identity to \(10^{-16}\)). The catch: \(G\) needs the fine states, which a coarse run does not have.</p>
<h3>2.4 Learning \(G\)</h3>
<p>Approximate \(\tilde G_{i+1/2}=\mathcal N(U_{i-1},U_i,U_{i+1},U_{i+2};\omega)\): four neighbouring cells × \((h,q)\) = 8 inputs, 2 outputs \((G^h,G^q)\),
three hidden layers of 128 neurons with leaky-ReLU (slope 0.01), 34,434 parameters, trained on the mean-squared error</p>
$$\mathcal L(\omega)=\frac1M\sum_{j=1}^{M}\big\|\mathcal N(U^{(j)}_{i-1},\dots,U^{(j)}_{i+2};\omega)-G^{(j)}_{i+1/2}\big\|_2^2 .$$
<h3>2.5 Limiting with MCL</h3>
<p>Write the corrected coarse update with limited bar states \(\bar U^{*,\pm}_{i+1/2}=\bar U_{i+1/2}\pm\tilde G^{*}_{i+1/2}/\Lambda_{i+1/2}\):</p>
$$\tfrac{d}{dt}U_i=\tfrac1{\Delta X}\Big[\Lambda_{i-1/2}\big(\bar U^{*,+}_{i-1/2}-U_i\big)+\Lambda_{i+1/2}\big(\bar U^{*,-}_{i+1/2}-U_i\big)\Big].$$
<p>MCL demands a <i>local maximum principle</i>: the new bar states must stay within the range of their neighbours,
\(h_i^{\min}\le\bar h^{*,-}_{i+1/2}\le h_i^{\max}\) and \(h_{i+1}^{\min}\le\bar h^{*,+}_{i+1/2}\le h_{i+1}^{\max}\), with
\(h_i^{\min/\max}=\min/\max(\bar h_{i-1/2},\bar h_{i+1/2})\), and the same for the velocity \(\bar v=\bar q/\bar h\).
It is applied in two steps. <b>Height first</b> (eq. 27 of the paper):</p>
$$\tilde G^{h,*}=\begin{cases}\min\!\big(\tilde G^h,\ \Lambda\min(\bar h-h_i^{\min},\ h_{i+1}^{\max}-\bar h)\big),&\tilde G^h\ge0\\[2pt]
\max\!\big(\tilde G^h,\ \Lambda\max(\bar h-h_i^{\max},\ h_{i+1}^{\min}-\bar h)\big),&\tilde G^h<0\end{cases}$$
<p><b>Then discharge:</b> limit \(\Delta\tilde G^q=\tilde G^q-\tilde G^{h,*}\bar v\) to the analogous velocity interval (eq. 31) and set
\(\tilde G^{q,*}=\tilde G^{h,*}\bar v+\Delta\tilde G^{q,*}\). Under the CFL condition the updated \(U_i\) is then a convex combination of admissible states,
so <b>\(h>0\) whatever the network outputs</b>. If the network is accurate the limits are inactive and MCL changes nothing.</p>

<h2>3. How it runs</h2>
<figure class="nobreak">@@FIG_PIPE@@<figcaption><b>Fig. 2.</b> Training happens once offline; inference needs only the coarse state.</figcaption></figure>
<div class="cols">
<div><h3>Training</h3><ol>
<li><b>Initial conditions.</b> 200 random sine waves: \(h=2+A_h\sin(2\pi k_hx/L+\varphi_h)\), \(v=V_0+A_v\sin(2\pi k_vx/L+\varphi_v)\), with \(V_0\sim U(1,2)\), \(A\sim U(0.2,0.6)\), \(k\in\{1..6\}\), \(\varphi\sim U(0,2\pi)\).</li>
<li><b>Labels.</b> Run the fine DNS to \(T=80\). Every 10th step, average to 100 cells and compute the exact \(G\) at all 100 interfaces:
32 million (stencil, \(G\)) pairs. (The paper keeps every step, 320 M.)</li>
<li><b>Standardise</b> inputs and outputs (zero mean, unit variance); the last 20 % of trajectories are held out for validation.</li>
<li><b>Optimise.</b> SGD, batch 128. Stage 1: learning rate 0.05, up to 500 epochs; stage 2: 0.001, up to 2000; early stopping in both.
Ours stopped after 32 and 18 epochs, validation loss \(3.96\cdot10^{-4}\) (standardised units).</li></ol></div>
<div><h3>Inference (each time step)</h3><ol>
<li><b>Start.</b> Box-average the fine initial condition to the 100-cell state \(U\); the fine solver is never used again.</li>
<li><b>LLF flux</b> \(\bar F\) at the 100 interfaces from \(U\).</li>
<li><b>Network.</b> Build each 4-cell stencil, standardise, evaluate \(\mathcal N\), undo the standardisation: raw \(\tilde G\).</li>
<li><b>MCL</b> (optional switch): \(\tilde G\to\tilde G^{*}\) by the two limiters of §2.5.</li>
<li><b>Update.</b> \(F=\bar F+\tilde G^{*}\), \(\dot U_i=(F_{i-1/2}-F_{i+1/2})/\Delta X\). Heun's method repeats steps 2–5 for its two stages.
The time step is the paper's 0.005, or up to 0.1.</li></ol></div></div>

<h2 class="">4. What we get</h2>
<figure class="nobreak"><img src="fig_results.png" style="width:100%"><figcaption><b>Fig. 3.</b> (a) A 100-cell model with the net follows the 2000-cell truth;
plain coarse LLF flattens it. (b) Far outside the training range the net alone blows up; MCL keeps the run alive. (c) Training only contained flow to the right;
building mirror symmetry into the network fixes flow to the left.</figcaption></figure>
<p class="nobreak"><b>Table 1.</b> Relative \(\ell^2\) error of \(q\) against the fine solution averaged to 100 cells, median per group of the @@NCASES@@ test cases
(paper cases, outside training range, tough, where the method breaks).</p>
<table class="nobreak"><tr><th>model</th><th>params</th><th>paper cases</th><th>outside training</th><th>tough</th><th>breaks</th><th>all</th><th>net alone blows up</th></tr>@@TABLE@@</table>
<ul>
<li><b>It works as the paper claims.</b> On the paper's own cases the 100-cell net + MCL is 0.3–0.5 % off in \(q\) against 1.5–7 % for plain coarse LLF,
and it keeps working at mean depths \(\pm25\) % from the training value and for runs 5× longer than the training trajectories (\(t=400\) against 80, \(q\) error 0.4 %).</li>
<li><b>MCL is a safety net, not an accuracy booster.</b> Without it the net alone blows up in 5 of 20 cases (very shallow water, near-dry troughs, strong dam breaks);
with it, in none. Errors there stay large (dam breaks 5–26 %) because the network never saw shocks.</li>
<li><b>MCL can also hurt.</b> For waves resolved by only 5–7 coarse cells, clipping flattens real crests: \(q\) error 15 % with MCL against 2 % without.</li>
<li><b>Our addition (not in the paper).</b> A mirror-symmetric network, with a built-in \(x\to-x\) symmetry that the exact \(G\) obeys:
it fixes left-moving flow (4.3 % → 0.7 %) and cuts dam-break errors by 40–80 %.</li>
<li><b>From the paper: smaller networks.</b> The paper also trains 3×64, 3×32 and 3×16 networks and finds 32 neurons often as good as 128.
We trained only 3×32 (on half the data) and agree: with MCL its median error is 2.5 % against 2.9 % for the 3×128 network, with 14× fewer parameters.</li>
<li><b>The speed-up is modest.</b> One network call costs about 90 µs in PyTorch, more than a whole 2000-cell DNS right-hand side in NumPy (about 60 µs).
At the paper's \(\Delta t=0.005\) the reduced models are therefore <i>slower</i> than the DNS here; with \(\Delta t=0.05\) they are about 2–5× faster (plain coarse LLF: 17×).</li>
</ul>
<p class="small"><b>Differences from the paper.</b> Training uses every 10th time step (memory). Our Fig. 6 Lax–Wendroff baseline is less diffusive than the paper's (their variant is not specified).
The network-size study (their Figs. 9–10) is only partly reproduced: we trained 3×128 and 3×32 but not 3×64 or 3×16. An earlier version of our code built the training target \(G\) from the wrong fine cell; this report uses the corrected code
(a test checks the exact-\(G\) identity of §2.3).</p>

<h2>5. Where things are</h2>
<div class="cols"><table class="map">
<tr><th>module (<code>work/final_btp/swe/</code>)</th><th>implements</th></tr>
<tr><td><code>physics.py</code>, <code>solver.py</code></td><td>§2.1–2.2: flux, LLF, bar states, Heun</td></tr>
<tr><td><code>data.py</code></td><td>§2.3 and training data: exact \(G\), stencil</td></tr>
<tr><td><code>network.py</code>, <code>training.py</code></td><td>§2.4: network, SGD, early stopping</td></tr>
<tr><td><code>mcl.py</code></td><td>§2.5: height and discharge limiters</td></tr>
<tr><td><code>tests/</code></td><td>33 checks against the paper's equations</td></tr></table>
<div><p class="small" style="margin-top:0"><b>Run it</b> (from <code>work/final_btp</code>):</p>
<p class="small"><code>python scripts/generate_data.py</code> → <code>python scripts/train.py</code> → <code>python scripts/precompute.py</code> → <code>python dashboard/server.py</code>, then open <code>localhost:8000</code>.
The dashboard shows any of 20 cases (or your own upload) for all three networks side by side, with the comparison matrix of Table 1.</p></div></div>
</body></html>
"""


def main():
    table, n_cases = accuracy_table()
    html = (HTML.replace("@@FIG_MESH@@", fig_mesh()).replace("@@FIG_PIPE@@", fig_pipeline())
            .replace("@@TABLE@@", table).replace("@@NCASES@@", str(n_cases)))
    (HERE / "report.html").write_text(html)
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=20000", f"--print-to-pdf={HERE / 'report.pdf'}",
                    f"file://{HERE / 'report.html'}"], check=True, capture_output=True)
    print("wrote", HERE / "report.pdf")


if __name__ == "__main__":
    main()
