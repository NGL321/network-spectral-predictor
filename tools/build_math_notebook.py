"""Build notebooks/00_math_walkthrough.ipynb: the math of nsp, walked through alongside the library's own code.

    python tools/build_math_notebook.py          # write the notebook
    python tools/build_math_notebook.py --html   # ... and export notebooks/html/00_math_walkthrough.html

The notebook runs no Python: every cell is Markdown. What is derived from the library at build time:
  * every code block (copied verbatim from src/nsp with `ast`, so it is always the current code);
  * every number in a worked example (computed by calling the library);
  * the diagrams (drawn with nsp.viz and embedded as cell attachments);
  * the list of tests at the end.
The prose and equations are hand-written here. They explain the code; they are not a source of truth.
"""

from __future__ import annotations

import argparse
import ast
import base64
import datetime as dt
import io
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nbformat as nbf  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "nsp"
OUT = ROOT / "notebooks"
sys.path.insert(0, str(ROOT / "src"))

from nsp import detect, fit, graphs, model, sense, viz, world as W  # noqa: E402
from nsp.results import git_version  # noqa: E402

WORLD = W.World()


# ------------------------------------------------------------------------------------------- source extraction

def _node(tree: ast.Module, name: str):
    parts = name.split(".")
    body = tree.body
    node = None
    for part in parts:
        node = next((n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
                     and n.name == part), None)
        if node is None:
            raise KeyError(name)
        body = getattr(node, "body", [])
    return node


def source(module: str, name: str) -> str:
    """Verbatim source of a function, class or Class.method in src/nsp/<module>.py, decorators included."""
    text = (SRC / f"{module}.py").read_text()
    node = _node(ast.parse(text), name)
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    lines = text.splitlines()[start - 1: node.end_lineno]
    return textwrap.dedent("\n".join(lines))


def code(module: str, *names: str, note: str = "") -> str:
    """A Markdown block showing library code, labelled with its file and line numbers."""
    text = (SRC / f"{module}.py").read_text()
    tree = ast.parse(text)
    blocks = []
    for name in names:
        node = _node(tree, name)
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        blocks.append((name, start, node.end_lineno, source(module, name)))
    where = ", ".join(f"`{n}` (lines {a}–{b})" for n, a, b, _ in blocks)
    body = "\n\n".join(s for *_, s in blocks)
    return f"**In the library**, `src/nsp/{module}.py`: {where}.{(' ' + note) if note else ''}\n\n```python\n{body}\n```"


def tests_table() -> str:
    rows = ["| Test | What it pins down |", "| --- | --- |"]
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                rows.append(f"| `{path.name}::{node.name}` | {TEST_NOTES.get(node.name, '')} |")
    return "\n".join(rows)


TEST_NOTES = {
    "test_spectrum_identity": "spec(I⊗K − L⊗M) = ∪ spec(K − λM), on a random gated ring of 8 (Part 4)",
    "test_K0_closed_form": "K₀ = e^{A₀Δt} equals the cos/sin matrix (Part 1)",
    "test_M0_series_to_dt_cubed": "the Fréchet-derivative M₀ matches the closed form to O(Δt⁴) (Part 5)",
    "test_two_hop_term_is_what_one_hop_misses": "exact step − one-hop model = L²⊗N₀ to 2·10⁻⁵, for a path, ring and star (Part 5)",
    "test_growth_and_frequency_formulas": "per-step growth and frequency error of the series match their formulas (Part 5)",
    "test_two_angles_small_swings_are_two_planar_copies": "with two angles, the exact step is the planar one ⊗ I₂ (Part 11)",
    "test_complete_graph_two_frequencies": "a complete graph of 8 has only the graph frequencies 0 and 8 (Part 2)",
    "test_mode_frequencies_match_exact_step": "ω_k = √(ω₀² + κλ_k), read off the exact step's eigenvalues (Part 3)",
    "test_verlet_matches_exact_when_linear": "velocity Verlet agrees with the exact step on a linear world (Part 8)",
    "test_expm_sanity": "the matrix exponential of zero is the identity",
    "test_joint_fit_recovers_series": "least squares on every gate pattern of a path of 3 recovers K₀ and M₀ (Part 6)",
    "test_frozen_K_is_exact": "K fitted on lone pendulums equals K₀ to 10⁻¹⁰ (Part 6)",
    "test_gradient_descent_reaches_closed_form": "gradient descent ends at the least-squares answer (Part 6)",
    "test_parameter_recovery": "ω₀² and κ read back from the fitted K and M (Part 6)",
    "test_kronecker_fit_two_angles": "the two-angle fit is the planar fit ⊗ I₂ (Part 11)",
    "test_detection_one_spring_needs_two_trajectories_or_nnls": "the blind spot exists with one trajectory; NNLS or three trajectories close it (Part 9)",
    "test_recall_and_relock": "the binding vertex recalls a stored structure and re-locks after a switch (Part 10)",
    "test_design_steps_stack_into_the_window_design": "per-step detection blocks stack into the window's design (Part 9, E7)",
    "test_transition_priors": "the uniform and one-flip priors over wiring changes are proper transition matrices (Part 10)",
    "test_relaxation_settles_on_the_clamp": "gradient descent on a body's energy ends at the clamped estimate (Part 12)",
    "test_mode_radius_matches_the_whole_error_map": "the clamp's per-mode spectral radius equals the whole network's (Part 12)",
    "test_delay_line_is_exact_on_clean_data": "two readings determine the state when the model is the world (Part 12)",
    "test_clamped_filter_and_kalman_converge_on_clean_data": "the clamped sense and the Kalman filter converge (Part 12)",
    "test_growing_precision_settles_on_the_steady_gain_and_converges_fast": "growing precision ends at the steady gain, and converges faster from a cold start (Part 12)",
}


# ------------------------------------------------------------------------------------------------- formatting

def bmat(A, digits: int = 5) -> str:
    """A LaTeX bmatrix with the numbers of A."""
    def f(x):
        x = float(x)
        if abs(x) < 1e-12:
            return "0"
        if abs(x) < 1e-3:
            m, e = f"{x:.{digits - 2}e}".split("e")
            return rf"{m}\times10^{{{int(e)}}}"
        return f"{x:.{digits}g}"
    rows = [" & ".join(f(x) for x in row) for row in np.atleast_2d(A)]
    return r"\begin{bmatrix}" + r"\\ ".join(rows) + r"\end{bmatrix}"


def imat(A) -> str:
    return r"\begin{bmatrix}" + r"\\ ".join(" & ".join(f"{int(round(x))}" for x in row) for row in A) + r"\end{bmatrix}"


def fill(text: str, **vals) -> str:
    for k, v in vals.items():
        text = text.replace(f"<<{k}>>", str(v))
    assert "<<" not in text, text[text.index("<<"): text.index("<<") + 40]
    return text


# ---------------------------------------------------------------------------------------------------- diagrams

viz.style()


def png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fig_world() -> str:
    fig, ax = plt.subplots(figsize=(8.8, 3.4))
    th = np.array([0.36, -0.14, 0.22])
    viz.draw_row(ax, th, None, graphs.path(3), [1, 1], limits=((-0.75, 2.85), (-1.62, 0.5)))
    C = viz.COLORS
    ax.plot([0, 0], [0, -1.05], color=C["muted"], lw=0.9, ls=(0, (3, 3)))
    arc = np.linspace(0, th[0], 30)
    ax.plot(0.66 * np.sin(arc), -0.66 * np.cos(arc), color=C["nominal"], lw=1.6)
    ax.text(0.07, -0.86, r"$\theta_1$", color=C["nominal"], fontsize=13, ha="center")
    labels = ["body 1", "body 2\nstate $x_2=(\\theta_2,\\ \\dot{\\theta}_2)$", "body 3"]
    for i in range(3):
        b = (i + np.sin(th[i]), -np.cos(th[i]))
        ax.text(b[0], b[1] - 0.2, labels[i], ha="center", va="top", fontsize=10, color="#57606A")
    ax.annotate("spring $e=(1,2)$: stiffness $\\kappa$, gate $g_e\\in\\{0,1\\}$", xy=(0.55, -0.43),
                xytext=(-0.6, 0.3), fontsize=10.5, ha="left",
                arrowprops=dict(arrowstyle="->", color=C["spring"], lw=1, relpos=(0.45, 0.0)))
    b3 = (2 + np.sin(th[2]), -np.cos(th[2]))
    ax.add_patch(FancyArrowPatch((b3[0] + 0.32, b3[1] + 0.25), (b3[0] + 0.32, b3[1] - 0.2), arrowstyle="-|>",
                                 mutation_scale=12, color=C["out"], lw=1.4))
    ax.text(b3[0] + 0.4, b3[1] + 0.05, "gravity\n$\\omega_0^2 = g/\\ell$", fontsize=10, color=C["out"], va="center")
    ax.set_title("The world of step 1: pendulums on a shared bar, joined by springs that can be switched off",
                 fontsize=11.5, loc="left")
    return png(fig)


def fig_blocks() -> str:
    C = viz.COLORS
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={"width_ratios": [1, 1.55]})
    L = graphs.laplacian(3, graphs.path(3))
    for ax in (a1, a2):
        ax.set_aspect("equal")
        ax.axis("off")
    for i in range(3):
        for j in range(3):
            col = "#DDEBF7" if i == j else ("#FBE3D6" if L[i, j] else "white")
            a1.add_patch(plt.Rectangle((j, 2 - i), 1, 1, fc=col, ec="#8C959F", lw=0.8))
            a1.text(j + 0.5, 2.5 - i, f"{int(L[i, j])}".replace("-", "−"), ha="center", va="center", fontsize=13)
            lab = (f"$K-{int(L[i, i])}M$" if L[i, i] != 1 else "$K-M$") if i == j else ("$+M$" if L[i, j] else "$0$")
            a2.add_patch(plt.Rectangle((j, 2 - i), 1, 1, fc=col, ec="#8C959F", lw=0.8))
            a2.text(j + 0.5, 2.5 - i, lab, ha="center", va="center", fontsize=13)
    for ax, title in ((a1, "$L$ for a path of 3 (bodies × bodies)"),
                      (a2, "$P = I_3\\otimes K - L\\otimes M$ (each cell a $2\\times2$ block)")):
        ax.set_xlim(-0.1, 3.1)
        ax.set_ylim(-0.1, 3.1)
        ax.set_title(title, fontsize=11, loc="left")
    a2.text(3.15, 2.5, "body 1: one spring", va="center", fontsize=9.5, color="#57606A")
    a2.text(3.15, 1.5, "body 2: two springs", va="center", fontsize=9.5, color="#57606A")
    a2.text(3.15, 0.5, "body 3: one spring", va="center", fontsize=9.5, color="#57606A")
    a2.set_xlim(-0.1, 4.4)
    fig.suptitle("Blue: each body's own block, K less one M per live spring. Orange: a live spring pulls with +M. "
                 "White: no spring, no coupling.", x=0.01, ha="left", fontsize=10, color="#57606A")
    return png(fig)


def fig_modes() -> str:
    L = graphs.laplacian(3, graphs.path(3))
    lam, U = np.linalg.eigh(L)
    om = np.sqrt(WORLD.omega0**2 + WORLD.kappa * np.clip(lam, 0, None))
    names = ["all together", "ends opposite, middle still", "middle against the ends"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.9))
    for k, ax in enumerate(axes):
        u = U[:, k] * np.sign(U[np.argmax(np.abs(U[:, k])), k])
        viz.draw_row(ax, 0.32 * u / np.abs(u).max(), None, graphs.path(3), limits=((-0.8, 2.8), (-1.25, 0.35)))
        ax.set_title(f"mode {k + 1}: {names[k]}\n$\\lambda_{k + 1} = {lam[k]:.0f}$,   "
                     f"$\\omega_{k + 1} = \\sqrt{{\\omega_0^2+\\kappa\\lambda_{k + 1}}} = {om[k]:.3f}$ rad/s",
                     fontsize=10.5, loc="left")
    fig.suptitle(f"Modes of a path of 3 (ω₀ = {WORLD.omega0:g}, κ = {WORLD.kappa:g}): the more a mode stretches "
                 "its springs (larger λ), the faster it swings", x=0.01, ha="left", fontsize=11.5)
    return png(fig)


def fig_reach(kappa: float = 4.0, n: int = 8) -> str:
    w = WORLD.with_(kappa=kappa)
    L = graphs.laplacian(n, graphs.path(n))
    exact = W.exact_step(w, L)
    one = model.series(w).assemble(L)

    def blocks(P):
        return np.array([[np.linalg.norm(P[2 * i:2 * i + 2, 2 * j:2 * j + 2]) for j in range(n)] for i in range(n)])

    panels = (("exact step $e^{A\\Delta t}$", blocks(exact)), ("one-hop model $I\\otimes K_0 - L\\otimes M_0$", blocks(one)),
              ("what one hop misses", blocks(exact - one)))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    norm = LogNorm(1e-9, 1.5)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#FFFFFF", "#9CC6E4", viz.COLORS["nominal"],
                                                                     "#08306B"])
    for ax, (title, B) in zip(axes, panels):
        im = ax.imshow(np.where(B > 1e-12, B, np.nan), cmap=cmap, norm=norm)
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xticks(range(n), [str(i + 1) for i in range(n)], fontsize=8)
        ax.set_yticks(range(n), [str(i + 1) for i in range(n)], fontsize=8)
        ax.grid(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("to body")
    for ax in axes:
        ax.set_xlabel("from body")
    fig.colorbar(im, ax=axes, shrink=0.8, label="size of each 2×2 block (blank = 0)")
    fig.suptitle(f"Path of {n}, κ = {kappa:g}: the exact step reaches every body a little; the model stops at one "
                 "spring. Its miss is the two-hop term $L^2\\otimes N_0$, which also reaches bodies two springs away", x=0.01,
                 ha="left", fontsize=11.5)
    return png(fig)


def fig_binding() -> str:
    C = viz.COLORS
    fig, ax = plt.subplots(figsize=(11, 3.9))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12", fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.2)
        return (x, y, w, h)

    def arrow(p, q, text="", dy=0.12, color="#57606A"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, color=color, lw=1.3,
                                     shrinkA=2, shrinkB=2))
        if text:
            ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, text, ha="center", va="bottom", fontsize=9.2,
                    color="#57606A")

    box(0.2, 2.45, 2.3, 1.2, "structural vertex\nbank $G$: $C$ stored wirings\n(one row $G_c$ each)", "#F6F8FA", "#8C959F")
    box(3.3, 2.45, 2.3, 1.2, "binding vertex\nbelief $\\pi(t)$ over\nthe $C$ wirings", "#DDEBF7", C["nominal"])
    box(6.4, 2.45, 2.0, 1.2, "gates\n$g(t)=G^\\top\\pi(t)$", "#F6F8FA", "#8C959F")
    box(8.9, 2.45, 1.95, 1.2, "network step\n$P(g)=I\\otimes K$\n$-\\,L(g)\\otimes M$", "#F6F8FA", "#8C959F")
    box(8.9, 0.35, 1.95, 1.2, "prediction\n$P(G_c)\\,x(t)$\nfor every $c$", "#F6F8FA", "#8C959F")
    box(5.7, 0.35, 2.5, 1.2, "evidence $\\ell_c$:\nhow well wiring $c$\npredicted $x(t{+}1)$",
        "#FBE3D6", C["out"])
    box(2.6, 0.35, 2.3, 1.2, "observation\n$x(t+1)$ (actual)", "#F6F8FA", "#8C959F")
    arrow((2.5, 3.05), (3.3, 3.05), "recall")
    arrow((5.6, 3.05), (6.4, 3.05), "mix")
    arrow((8.4, 3.05), (8.9, 3.05))
    arrow((9.875, 2.45), (9.875, 1.55), "")
    arrow((8.9, 0.95), (8.2, 0.95))
    arrow((4.9, 0.95), (5.7, 0.95))
    arrow((6.95, 1.55), (4.45, 2.45), "")
    ax.text(6.3, 1.93, "correct: $\\alpha \\leftarrow \\mathrm{diag}(\\ell)\\,T^\\top\\alpha$", fontsize=9.6,
            color=C["nominal"], ha="left", va="bottom")
    ax.set_title("Transfer by recall (E6): the binding vertex predicts with its belief, then corrects the belief by how "
                 "well each stored wiring predicted", fontsize=11.2, loc="left")
    return png(fig)


# ------------------------------------------------------------------------------------------------------ content

def numbers() -> dict:
    """Worked-example numbers, computed by the library."""
    w = WORLD
    L3 = graphs.laplacian(3, graphs.path(3))
    L3cut = graphs.laplacian(3, graphs.path(3), [0, 1])
    s = model.series(w)
    M0cf, N0cf = model.series_closed_form(w)
    w16 = w.with_(kappa=16.0)
    rec = fit.recover_parameters(s, w)
    star = w16
    growth = float(model.predicted_growth(star, 8.0))
    return dict(
        w0=f"{w.omega0:g}", kap=f"{w.kappa:g}", dt=f"{w.dt:g}", w0dt=f"{w.omega0 * w.dt:g}",
        K0=bmat(s.K), M0=bmat(s.M, 4), M0cf=bmat(M0cf, 4), N0cf=bmat(N0cf, 3),
        dM0=viz.fmt(np.abs(s.M - M0cf).max(), 2),
        D3=imat(graphs.incidence(3, graphs.path(3))), L3=imat(L3), L3cut=imat(L3cut),
        lam3=", ".join(f"{x:.0f}" for x in graphs.graph_frequencies(3, graphs.path(3))),
        lam3cut=", ".join(f"{abs(x):.0f}" for x in graphs.graph_frequencies(3, graphs.path(3), [0, 1])),
        om3=", ".join(f"{x:.4f}" for x in W.mode_frequencies(w, L3)),
        lamstar=", ".join(sorted({f"{x:.0f}" for x in graphs.graph_frequencies(8, graphs.star(8))}, key=float)),
        lamcomp=", ".join(sorted({f"{x:.0f}" for x in graphs.graph_frequencies(8, graphs.complete(8))}, key=float)),
        r_path=viz.fmt(float(model.two_hop_ratio(w, 3.0)), 3),
        r_star=viz.fmt(float(model.two_hop_ratio(star, 8.0)), 3),
        g_star=viz.fmt(growth, 3), g_star_200=f"{(1 + growth) ** 200:.2f}",
        f_star=viz.pct(float(model.predicted_frequency_error(star, 8.0)), 2),
        w2hat=f"{rec['omega0_sq']:.6f}", khat=f"{rec['kappa']:.6f}", kbare=f"{rec['kappa_bare']:.6f}",
        kbare_pct=viz.pct(1 - rec["kappa_bare"], 2),
        swing2=f"{W.swing_frequency(w, 2.0) / w.omega0:.3f}",
        swing2_pct=viz.pct(1 - W.swing_frequency(w, 2.0) / w.omega0, 2),
        **_sense_numbers(w, s),
    )


def _sense_numbers(w, s) -> dict:
    """Part 12's worked example: s = 1% of a 0.1 rad swing, q = 1e-8 (rad/s)^2 per step."""
    sd, q = 0.01 * 0.1, 1e-8
    k, _ = sense.lone_gain(s.K, sd**2, q)
    rho = float(sense.mode_radius(s, k, [0.0])[0])
    return dict(kq=bmat(k[None], 4), kth=f"{k[0]:.4f}", rho=f"{rho:.4f}", sqrt1k=f"{np.sqrt(1 - k[0]):.4f}",
                halving=f"{sense.halving_steps(rho):.1f}", dline=f"{np.sqrt(2) * sd / w.dt:.4f}",
                dline_pct=viz.pct(np.sqrt(2) * sd / w.dt / (0.1 * w.omega0), 2),
                rho4=f"{float(sense.mode_radius(s, k, [4.0])[0]):.4f}")


def cells(N: dict, version: str, today: str) -> list:
    md = nbf.v4.new_markdown_cell
    out = []

    out.append(md(fill(r"""# The math of network-spectral-predictor, walked through with the code

*Step 1 · pendulums on springs · built <<today>> from the library at code version `<<version>>` by `tools/build_math_notebook.py`*

> **This notebook runs no Python, and it is not a source of truth.** Every code block is copied verbatim from `src/nsp` when the notebook is built, and every number in a worked example is computed by the library at that moment, so neither can drift from the code. The prose and the equations are hand-written: they explain the code, they do not define it. Where they disagree, the code and its tests win (the tests are listed at the end).

**How to read it.** Each part opens with one plain sentence, then the equations with every symbol defined, then a worked example, then the code that implements it. Parts 1 to 5 are the core (the world, the model, and what the model can and cannot represent); 6 and 7 are learning and measuring; 8 to 11 follow the experiments (large swings, detecting springs, recall, two angles); 12 opens step 2, where only angles are measured.

| Part | Topic | Experiments |
| --- | --- | --- |
| 0 | The picture | |
| 1 | One pendulum | E0 |
| 2 | Springs and the graph | E1–E3 |
| 3 | The whole world, and its modes | E0, E2 |
| 4 | The model: one body map, one spring map | E1–E3 |
| 5 | What one hop misses | E2 |
| 6 | Learning $K$ and $M$ | E1 |
| 7 | Measuring: actual, nominal, deviation | all |
| 8 | Large swings | E4 |
| 9 | Detecting the springs | E5 |
| 10 | Recall: structural and binding vertices | E6 |
| 11 | Preview of stage 1.5: two angles per body | stage 1.5 |
| 12 | Hidden information: a clamped sense | step 2, H1 |""", today=today, version=version)))

    out.append(md(r"""## Notation

| Symbol | Meaning | In the code |
| --- | --- | --- |
| $n$; $v, u$ | number of bodies; body indices | `n`; `v`, `u` |
| $e=(u,v)$; $m$ | a spring between bodies $u$ and $v$; number of springs | `edges`; `len(edges)` |
| $d$ | position coordinates per body: 1 = one swing angle (planar), 2 = two tilts (stage 1.5) | `World.d` |
| $\theta_v$, $\dot\theta_v$ | swing angle of body $v$ (rad) and its rate (rad/s) | `X[..., 0::2]`, `X[..., 1::2]` |
| $x_v=(\theta_v,\dot\theta_v)$; $x$ | state of body $v$; all bodies stacked, $x=(x_1,\dots,x_n)$ | one row `X[traj, t]` |
| $t$ | sample index; one step is $\Delta t$ seconds | axis 1 of `X` |
| $\omega_0$ | a lone pendulum's frequency, $\sqrt{g/\ell}$ (rad/s) | `World.omega0` |
| $\kappa$ | spring stiffness ÷ inertia (1/s²) | `World.kappa` |
| $\Delta t$ | sampling step (s) | `World.dt` |
| $g_e\in\{0,1\}$ | gate of spring $e$: live or cut | `gates` |
| $\delta$ | incidence matrix, $m\times n$: row $e=(u,v)$ has $-1$ at $u$, $+1$ at $v$ | `graphs.incidence` |
| $L(g)$ | gated graph Laplacian, $n\times n$ | `graphs.laplacian` |
| $\lambda_k$, $u_k$ | graph frequencies (eigenvalues of $L$) and their patterns (eigenvectors) | `graphs.graph_frequencies` |
| $A_0$, $E$ | a lone body's generator; the $2\times2$ matrix that turns a position into an acceleration | `World.A0`, `World.E` |
| $A$ | the whole world's generator, $2n\times2n$ | `world.generator` |
| $K$, $M$ | the model's body map and spring map, $2\times2$ each | `OneHop.K`, `OneHop.M` |
| $K_0$, $M_0$, $N_0$ | the exact step's expansion in $\lambda$: order 0, 1 and 2 | `model.series`, `model.series_closed_form` |
| $P(g)$ | the assembled model step $I_n\otimes K-L(g)\otimes M$ | `OneHop.assemble` |
| $s_v$ | spring input of body $v$: $\sum_u g_{uv}(x_v-x_u)$ | `fit.body_samples` |
| $\psi$, $W$ | a body lift (features for large swings) and its learned read-out | `lifts.features`, `LiftedOneHop.W` |
| $G$, $\pi$, $T$, $h$, $\Sigma$ | stored gate patterns, belief over them, persistence matrix, hazard, residual covariance | `context` |
| $\otimes$ | Kronecker product: $A\otimes B$ puts a copy of $B$, scaled by $a_{ij}$, in block $(i,j)$ | `np.kron` |
| actual, nominal | the generated world, the model's prediction; deviation = actual − nominal | `metrics`, `viz` |"""))

    # ---- Part 0
    out.append(md(r"""## 0 · The picture

*In one sentence:* a row of pendulums joined by springs is predicted with one small map for a pendulum and one small map for a spring, copied over the graph of springs.

In 1665 Huygens noticed that two pendulum clocks hanging from the same beam end up swinging in lockstep, in opposite directions: each clock is a lone oscillator, and the beam couples them (Bennett, Schatz, Rockwood & Wiesenfeld, *Proc. R. Soc. A* 2002, reconstruct the experiment). Here each pendulum is a lone oscillator, each spring a coupling, and a graph says which couplings exist. The bet of step 1 is that you can learn the lone oscillator ($K$) and the coupling ($M$) once, then assemble them for any graph, including graphs and switch patterns never seen in training.

![the world](attachment:world.png)

Everything that follows is about three questions: what the exact answer is (Parts 1 to 3), what the model can represent of it (4 and 5), and how the model is learned and judged (6 and 7)."""))
    out[-1]["attachments"] = {"world.png": {"image/png": fig_world()}}

    # ---- Part 1
    out.append(md(fill(r"""## 1 · One pendulum

*In one sentence:* for small swings a lone pendulum is a two-number linear system, and its exact one-step map is a rotation in the $(\theta,\ \dot\theta/\omega_0)$ plane.

For small swings $\sin\theta\approx\theta$, so $\ddot\theta=-\omega_0^2\theta$. As a first-order system:

$$\frac{d}{dt}\underbrace{\begin{bmatrix}\theta\\ \dot\theta\end{bmatrix}}_{x_v}=\underbrace{\begin{bmatrix}0 & 1\\ -\omega_0^2 & 0\end{bmatrix}}_{A_0}\begin{bmatrix}\theta\\ \dot\theta\end{bmatrix}\qquad\Longrightarrow\qquad x_v(t+1)=\underbrace{e^{A_0\Delta t}}_{K_0}\,x_v(t)$$

The exponential has a closed form (sum the series, or check that it solves the equation):

$$K_0=\begin{bmatrix}\cos\omega_0\Delta t & \tfrac{1}{\omega_0}\sin\omega_0\Delta t\\ -\omega_0\sin\omega_0\Delta t & \cos\omega_0\Delta t\end{bmatrix},\qquad \operatorname{spec}K_0=\{e^{\pm i\omega_0\Delta t}\}$$

Its eigenvalues have modulus 1 (the swing neither grows nor decays) and angle $\omega_0\Delta t$: the phase advance per step. Every later "frequency" is read the same way, as (angle of an eigenvalue) ÷ $\Delta t$.

**Worked example** ($\omega_0=<<w0>>$ rad/s, $\Delta t=<<dt>>$ s, so $\omega_0\Delta t=<<w0dt>>$ rad per step):

$$K_0=<<K0>>$$

""", **N) + code("world", "World")))

    # ---- Part 2
    out.append(md(fill(r"""## 2 · Springs and the graph

*In one sentence:* the graph Laplacian turns "who is joined to whom" into a matrix whose row $v$ adds up the stretch of every live spring on body $v$.

The incidence matrix $\delta$ reads the stretch of each spring; the gates switch springs on and off; the Laplacian sums them back onto bodies:

$$\underbrace{(\delta\theta)_e}_{\text{stretch of spring }e=(u,v)}=\theta_v-\theta_u,\qquad L(g)=\delta^\top\operatorname{diag}(g)\,\delta=\sum_e g_e\,\underbrace{b_eb_e^\top}_{\text{spring }e\text{'s own Laplacian}},\qquad (L\theta)_v=\sum_u g_{uv}\,(\theta_v-\theta_u)$$

where $b_e^\top$ is row $e$ of $\delta$. $L$ is symmetric, its rows sum to zero ($L\mathbf 1=0$: moving every body together stretches nothing), and $\theta^\top L\theta=\sum_e g_e(\theta_v-\theta_u)^2\ge 0$, so its eigenvalues $\lambda_k$ are real and non-negative. They are the **graph frequencies**: how much each pattern of motion stretches the springs.

**Worked example: a path of 3** (springs 1–2 and 2–3, both live):

$$\delta=<<D3>>,\qquad L=\delta^\top\delta=<<L3>>,\qquad \lambda=<<lam3>>$$

Cut spring 1–2 ($g=(0,1)$) and body 1 hangs free: $L=<<L3cut>>$, with $\lambda=<<lam3cut>>$. A star of 8 has graph frequencies $\{<<lamstar>>\}$; a complete graph of 8 only $\{<<lamcomp>>\}$. Their largest value, 8, is far above anything in a path or ring of 3 or 5, which is what E2 tests.

""", **N) + code("graphs", "incidence", "laplacian", "graph_frequencies", "all_gate_patterns")))

    # ---- Part 3
    out.append(md(fill(r"""## 3 · The whole world, and its modes

*In one sentence:* stacked together the bodies form one linear system, and in the graph's own coordinates it falls apart into one lone pendulum per graph frequency, stiffened by $\kappa\lambda$.

Each body feels gravity and its springs: $\ddot\theta_v=-\omega_0^2\theta_v-\kappa\,(L\theta)_v$. The matrix $E=\begin{bmatrix}0&0\\1&0\end{bmatrix}$ routes a position into an acceleration, so in state form:

$$\dot x=\underbrace{\Big(\underbrace{I_n\otimes A_0}_{\text{each body alone}}\;-\;\underbrace{\kappa\,L(g)\otimes E}_{\text{springs}}\Big)}_{A}\,x,\qquad\text{e.g. path of 3: }A=\begin{bmatrix}A_0-\kappa E & \kappa E & 0\\ \kappa E & A_0-2\kappa E & \kappa E\\ 0 & \kappa E & A_0-\kappa E\end{bmatrix}$$

**Modes.** Diagonalise the graph, $L=U\Lambda U^\top$ with orthonormal $U=[u_1\ \cdots\ u_n]$. The Kronecker mixed-product rule $(A\otimes B)(C\otimes D)=AC\otimes BD$ gives

$$(U\otimes I_2)^\top A\,(U\otimes I_2)=\underbrace{U^\top U}_{I}\otimes A_0-\kappa\,\underbrace{U^\top LU}_{\Lambda}\otimes E=\operatorname{blockdiag}_k\big(\underbrace{A_0-\kappa\lambda_kE}_{\text{a lone pendulum with }\omega_0^2\to\omega_0^2+\kappa\lambda_k}\big)$$

$$\Longrightarrow\qquad\omega_k=\sqrt{\omega_0^2+\kappa\lambda_k}$$

So a network of $n$ pendulums is $n$ independent pendulums in disguise, one per pattern $u_k$, each as stiff as its pattern stretches the springs. The exact step is $e^{A\Delta t}$, block-diagonal in the same basis.

**Worked example** (path of 3, $\omega_0=<<w0>>$, $\kappa=<<kap>>$): $\lambda=<<lam3>>$ gives $\omega=<<om3>>$ rad/s, i.e. $2$, $\sqrt5$ and $\sqrt7$.

![modes of a path of 3](attachment:modes.png)

""", **N) + code("world", "generator", "exact_step", "mode_frequencies")))
    out[-1]["attachments"] = {"modes.png": {"image/png": fig_modes()}}

    # ---- Part 4
    out.append(md(r"""## 4 · The model: one body map, one spring map

*In one sentence:* the model copies one learned body map $K$ onto every body and one learned spring map $M$ onto every live spring, so its size never depends on the graph.

$$\underbrace{x_v(t+1)}_{\text{next state of }v}=\underbrace{K\,x_v(t)}_{\text{body alone}}\;-\;\underbrace{M}_{\text{spring map}}\;\underbrace{\sum_u g_{uv}\big(x_v(t)-x_u(t)\big)}_{s_v(t)\text{: spring input}}\qquad\Longleftrightarrow\qquad x(t+1)=\underbrace{\big(I_n\otimes K-L(g)\otimes M\big)}_{P(g)}\,x(t)$$

$K$ and $M$ are $2\times2$: eight numbers in all, whatever the graph. For a path of 3, $P$ has the same block pattern as $A$:

![block structure](attachment:blocks.png)

**The spectrum identity.** The argument of Part 3 applies word for word, with $K$ in place of $A_0$ and $M$ in place of $\kappa E$:

$$\operatorname{spec}P(g)=\bigcup_k\operatorname{spec}\big(\underbrace{K-\lambda_kM}_{2\times2,\text{ one per graph frequency}}\big)$$

So the model meets a graph only through its frequencies $\lambda_k$: two graphs with the same $\lambda$'s pose it the same per-mode problems. That is why transfer (E2) should follow $\lambda$, not the number of bodies, and why a model only has to be right over the range of $\lambda$ it will meet.

**In the plan's flat-model language**, $P=\operatorname{blockdiag}(K)+C\,\Gamma\,F$ with restriction map $F=\delta\otimes I_2$ (read the difference across each spring), gated edge map $\Gamma=\operatorname{diag}(g)\otimes M$, and write-back $C=-\delta^\top\otimes I_2$. Check: $C\Gamma F=-(\delta^\top\operatorname{diag}(g)\,\delta)\otimes M=-L(g)\otimes M$.

""" + code("model", "OneHop")))
    out[-1]["attachments"] = {"blocks.png": {"image/png": fig_blocks()}}

    # ---- Part 5
    out.append(md(fill(r"""## 5 · What the exact answer is, and what one hop misses

*In one sentence:* expanding the exact step in $\lambda$ shows that the best one-hop model is the first two terms of a series, and the first term it cannot represent reaches two springs away, with relative size $\kappa\lambda\Delta t^2/6$.

The exact step of mode $k$ is $e^{(A_0-\kappa\lambda_kE)\Delta t}$. Expand it in $\lambda$:

$$e^{(A_0-\kappa\lambda E)\Delta t}=\underbrace{K_0}_{e^{A_0\Delta t}}\;-\;\lambda\,\underbrace{M_0}_{\text{first order}}\;+\;\lambda^2\,\underbrace{N_0}_{\text{second order}}\;-\;\cdots$$

Since $\lambda_k^j$ in every mode is $L^j$ on the whole graph, the exact step is $I\otimes K_0-L\otimes M_0+L^2\otimes N_0-\cdots$. The model represents the first two terms exactly. $L^2$ acts on each body, on its neighbours and on bodies two springs apart: a fitted model can soak up part of the first two, but nothing one-hop can reach the third. For an interior body of a path, $L^2$ has $6$ on the diagonal, $-4$ to each neighbour and $1$ two springs away.

$M_0$ is a directional derivative of the matrix exponential (a Fréchet derivative), which the library computes exactly: $M_0=\mathcal D\exp(A_0\Delta t)\,[\kappa E\Delta t]$. Expanded in $\Delta t$:

$$M_0=\kappa\begin{bmatrix}\tfrac{\Delta t^2}{2} & \tfrac{\Delta t^3}{6}\\[2pt] \Delta t-\tfrac{\omega_0^2\Delta t^3}{3} & \tfrac{\Delta t^2}{2}\end{bmatrix}+O(\Delta t^4),\qquad N_0=\kappa^2\begin{bmatrix}\tfrac{\Delta t^4}{24} & 0\\[2pt] \tfrac{\Delta t^3}{6} & \tfrac{\Delta t^4}{24}\end{bmatrix}+O(\Delta t^5)$$

Read the entries physically: a spring's pull first changes the *rate* (entry 2,1, size $\kappa\Delta t$) and only then the *angle* (entry 1,1, size $\kappa\Delta t^2/2$).

Three consequences, from the dominant entries (the ones in row 2, column 1):

$$\underbrace{\frac{\|\lambda^2N_0\|}{\|\lambda M_0\|}\approx\frac{\kappa\lambda\,\Delta t^2}{6}}_{\text{size of the two-hop miss}},\qquad\underbrace{|\mu|-1\approx\frac{(\kappa\lambda)^2\Delta t^4}{24}}_{\text{growth per step of }K_0-\lambda M_0},\qquad\underbrace{\frac{\hat\omega-\omega}{\omega}\approx\frac{(\kappa\lambda\Delta t)^2}{12\,\omega^2}}_{\text{frequency error}}$$

with $\mu$ an eigenvalue of $K_0-\lambda M_0$ and $\hat\omega=|\arg\mu|/\Delta t$.

**Worked example** ($\omega_0=<<w0>>$, $\kappa=<<kap>>$, $\Delta t=<<dt>>$):

$$M_0=<<M0>>,\qquad\text{closed form above}=<<M0cf>>\quad(\text{differ by }<<dM0>>),\qquad N_0\approx<<N0cf>>$$

For a path of 3 ($\lambda\le3$) the two-hop miss is only <<r_path>> of the one-hop term. For a star of 8 ($\lambda=8$) at $\kappa=16$ it is <<r_star>>; the series model also grows by <<g_star>> per step (×<<g_star_200>> over 200 steps, 10 s) and plays that mode <<f_star>> fast. These are the *series* model's numbers; a fitted model trades them off against each other, which E2 measures.

![what one hop misses](attachment:reach.png)

""", **N) + code("model", "series", "series_closed_form", "two_hop_ratio", "predicted_growth",
                  "predicted_frequency_error")))
    out[-1]["attachments"] = {"reach.png": {"image/png": fig_reach()}}

    # ---- Part 6
    out.append(md(fill(r"""## 6 · Learning $K$ and $M$

*In one sentence:* every body at every step is one sample of the same small regression, so $K$ and $M$ come from a single least-squares solve whose size does not grow with the graph.

Each sample $i$ is one (configuration, trajectory, step, body):

$$\underbrace{y_i}_{x_v(t+1)}=\underbrace{\begin{bmatrix}K & -M\end{bmatrix}}_{W^\top,\ 2\times4}\;\underbrace{\begin{bmatrix}x_v(t)\\ s_v(t)\end{bmatrix}}_{\varphi_i,\ 4\times1},\qquad s_v(t)=\big[(L(g)\otimes I_2)\,x(t)\big]_v$$

Least squares minimises $\mathcal E(W)=\frac1N\sum_i\|y_i-W^\top\varphi_i\|^2$. Setting its gradient to zero gives the normal equations:

$$W^{*\top}=\Big(\underbrace{\sum_i y_i\varphi_i^\top}_{2\times4}\Big)\Big(\underbrace{\sum_i\varphi_i\varphi_i^\top}_{4\times4}\Big)^{-1}$$

A $4\times4$ solve, for three bodies or three thousand. On clean small-swing data the answer is the series $(K_0,M_0)$ of Part 5, up to the small two-hop term the model cannot hold.

**Gradient descent on the same energy is local.** With each body's own error $\varepsilon_i=y_i-(Kx_v-Ms_v)$:

$$\Delta K\;\propto\;+\sum_i\varepsilon_i\,x_v^\top,\qquad\Delta M\;\propto\;-\sum_i\varepsilon_i\,s_v^\top$$

Each body needs only its own error and its own inputs: a delta rule. With step $\eta=1/\lambda_{\max}(H)$, $H=\frac1N\sum_i\varphi_i\varphi_i^\top$, it converges to $W^*$ (E1 checks this).

**Reading the physics back.** Since $K_0=e^{A_0\Delta t}$, the matrix logarithm returns $A_0$, and the $(2,1)$ entry of $M_0$ carries $\kappa$:

$$\hat\omega_0^2=-\Big[\tfrac{1}{\Delta t}\log K\Big]_{21},\qquad\hat\kappa=\frac{M_{21}}{\Delta t-\hat\omega_0^2\Delta t^3/3}$$

**Worked example**, applied to the exact series $(K_0,M_0)$: $\hat\omega_0^2=<<w2hat>>$ and $\hat\kappa=<<khat>>$. Without the $\Delta t^3$ correction, $M_{21}/\Delta t=<<kbare>>$: <<kbare_pct>> low.

""", **N) + code("fit", "body_samples", "fit_joint", "fit_frozen", "gradient_descent", "recover_parameters")))

    # ---- Part 7
    out.append(md(r"""## 7 · Measuring: actual, nominal, deviation

*In one sentence:* the generated world is **actual**, the model's prediction **nominal**, and every result is a deviation judged against a tolerance, as on an inspection report.

This inverts the usual metrology roles on purpose. On a shop floor the nominal is the design and the actual is the part that was made; here the world is generated exactly, so it plays the measured part, and the prediction plays the design that the part is checked against.

$$\text{deviation}(t)=x^{\text{actual}}(t)-x^{\text{nominal}}(t),\qquad e(t)=\frac{\big\|X^{\text{nominal}}(t)-X^{\text{actual}}(t)\big\|}{\big\|X^{\text{actual}}(t)\big\|}$$

with norms over every trajectory and body at step $t$ (a free-running rollout from the same start). **Status:** *in* when $|\text{deviation}|\le\text{tolerance}$, *near* within twice it, *out* beyond.

For detection (Part 9) the score is the **AUROC**: the chance that a live spring scores above a missing one, ties counting half, computed from ranks (the Mann–Whitney statistic).

""" + code("metrics", "relative", "rollout_curve", "status", "auroc")))

    # ---- Part 8
    out.append(md(fill(r"""## 8 · Large swings: when $\sin\theta\neq\theta$

*In one sentence:* a wide swing is slower than a narrow one, which no fixed linear map can reproduce, so each body's map gets extra features (a lift) while the springs stay linear.

The world is now $\ddot\theta_v=-\omega_0^2\sin\theta_v-\kappa(L\theta)_v$. A lone pendulum released from angle $a$ swings at

$$\omega(a)=\frac{\pi\,\omega_0}{2\,\mathcal K\!\big(\sin^2\tfrac a2\big)},\qquad\mathcal K(m)=\int_0^{\pi/2}\frac{d\phi}{\sqrt{1-m\sin^2\phi}}$$

($\mathcal K$ is the complete elliptic integral of the first kind). At $a=2$ rad, $\omega/\omega_0=<<swing2>>$: <<swing2_pct>> slower. A linear map has fixed eigenvalues, so it plays one frequency at every amplitude.

**Which features a step needs.** Expand one step of the rate:

$$\dot\theta(t+\Delta t)=\dot\theta+\Delta t\,\underbrace{\ddot\theta}_{-\omega_0^2\sin\theta-\kappa(L\theta)_v}+\frac{\Delta t^2}{2}\,\underbrace{\theta^{(3)}}_{-\omega_0^2\,\dot\theta\cos\theta-\kappa(L\dot\theta)_v}+O(\Delta t^3)$$

($\theta^{(3)}$ is the third time derivative.) So the body map needs $\sin\theta$ at first order and its time derivative $\dot\theta\cos\theta$ at second; the spring terms stay linear in differences. The lifted model:

$$x_v(t+1)=W^\top\begin{bmatrix}\psi\big(x_v(t)\big)\\ s_v(t)\end{bmatrix},\qquad\psi\in\Big\{(\theta,\dot\theta),\ (\theta,\dot\theta,\sin\theta),\ (\theta,\dot\theta,\sin\theta,\dot\theta\cos\theta)\Big\}$$

It is still one least-squares solve, and a rollout re-lifts the model's own output every step.

**The ground truth for large swings** is velocity Verlet with $h=\Delta t/50$, where $a(\theta)=-\omega_0^2\sin\theta-\kappa L\theta$:

$$\theta_{j+1}=\theta_j+h\dot\theta_j+\tfrac{h^2}{2}a(\theta_j),\qquad\dot\theta_{j+1}=\dot\theta_j+\tfrac h2\big(a(\theta_j)+a(\theta_{j+1})\big)$$

It is symplectic, so the energy $H=\sum_v\big[\tfrac12\dot\theta_v^2+\omega_0^2(1-\cos\theta_v)\big]+\tfrac\kappa2\theta^\top L\theta$ stays within $O(h^2)$ of its start instead of drifting (E0 checks this).

""", **N) + code("lifts", "features") + "\n\n" + code("fit", "fit_lifted") + "\n\n" +
        code("world", "swing_frequency", "simulate_verlet", "energy")))

    # ---- Part 9
    out.append(md(r"""## 9 · Detecting the springs (step 1b)

*In one sentence:* with $K$ and $M$ known the step is linear in the gates, so finding the live springs is one more least-squares problem, with one blind spot that first principles predict exactly.

Each body's residual after its own map is the spring term: $r_v(t)=x_v(t+1)-Kx_v(t)=-M\sum_u g_{uv}\,(x_v-x_u)$. Give every candidate pair $(a,b)$ an unknown weight:

$$\underbrace{r}_{\text{all bodies, all steps}}=\sum_{(a,b)}g_{ab}\,\underbrace{G_{ab}}_{\text{one column}},\qquad G_{ab}\big|_{\text{body }b}=-M(x_b-x_a),\qquad G_{ab}\big|_{\text{body }a}=+M(x_b-x_a)$$

and solve by least squares, or by non-negative least squares (NNLS), since a gate is an on/off switch and so $g\ge0$.

**The blind spot, from first principles.** A combination of gates is invisible when it changes no residual: $\sum_u g_{uv}M(x_v-x_u)=0$ for every body and step. $M$ is invertible, so this says $(L(g)\otimes I_2)\,x(t)=0$ at every $t$: the weighted Laplacian must annihilate every spatial pattern the trajectory ever shows.

Take E5's case: a ring of 5 with only spring 4–5 live. Bodies 1, 2 and 3 swing freely at $\omega_0$, and so does the in-phase motion of the pair 4–5. All four of those motions follow the same map $K_0$, so one trajectory sweeps out only a two-dimensional slice of their four-dimensional pattern space (the state of a body is two numbers, so an $n\times2$ snapshot has rank at most 2). Add the pattern $\mathbf 1$, which every Laplacian annihilates, and the pair's out-of-phase pattern, and $L(g)$ must vanish on a four-dimensional subspace $S\subset\mathbb R^5$. A symmetric matrix that vanishes on $S$ is $\alpha\,vv^\top$ for the one direction $v\perp S$, and $v\perp\mathbf 1$, so it is a valid (signed) Laplacian. Exactly one invisible direction:

$$L(g_{\text{blind}})=\alpha\,vv^\top\qquad\Longleftrightarrow\qquad(g_{\text{blind}})_{ab}=-\alpha\,v_av_b$$

Because $v\perp\mathbf 1$ has entries of both signs, $g_{\text{blind}}$ does too: moving along it from the true gates pushes some zero gates negative, which non-negativity forbids. A second trajectory shows a second slice, $S$ becomes all of $\mathbb R^5$, and the blind spot closes. The count generalises: with all five bodies free, one trajectory leaves the $2\times2$ symmetric matrices on a two-dimensional complement, three invisible directions (`detect.design_singular_values` confirms both counts).

""" + code("detect", "design", "detect_gates", "design_singular_values")))

    # ---- Part 10
    out.append(md(r"""## 10 · Recall: the structural vertex and the binding vertex (E6)

*In one sentence:* instead of re-deriving the springs from scratch, keep a bank of known wirings and a running belief over which one is in force, and update the belief by how well each wiring predicts the next step.

Think of a librarian with a shelf of known wiring diagrams who keeps a running bet on which one is in force and adjusts it after every observation. The **structural vertex** is the shelf: a bank $G\in\{0,1\}^{C\times m}$, one stored gate pattern $G_c$ per row. The **binding vertex** (the hippocampal analog) is the bet: a belief $\pi(t)$ over the $C$ rows.

![binding loop](attachment:binding.png)

$$\underbrace{\alpha(t+1)}_{\text{unnormalised belief}}=\underbrace{\operatorname{diag}\big(\ell(t+1)\big)}_{\text{evidence}}\;\underbrace{T^\top}_{\text{persistence}}\;\alpha(t),\qquad\pi=\frac{\alpha}{\mathbf 1^\top\alpha},\qquad g(t)=G^\top\pi(t)$$

$$\ell_c(t+1)=\exp\!\Big(-\tfrac12\big\|x(t+1)-P(G_c)\,x(t)\big\|^2_{\Sigma^{-1}}\Big),\qquad T=(1-h)\,I+\frac{h}{C-1}\big(\mathbf 1\mathbf 1^\top-I\big)$$

Here $h$ is the hazard (the chance per step that the wiring changes) and $\Sigma$ the diagonal covariance of one-step residuals, measured on a stretch with known gates. This is the forward pass of a hidden Markov model (Rabiner, *Proc. IEEE* 1989), the same machinery as the COIN model of motor memory (Heald, Lengyel & Wolpert, *Nature* 2021), and the minimal form of the Tolman-Eichenbaum-style binding proposed for the end of step 1 (Whittington et al., *Cell* 2020): binding a context to stored structure, without yet learning the structure itself. Unnormalised, it is a linear predictor $T^\top$ whose output is gated element by element by evidence: the same algebra as a gated edge of the network, which is why it can live as one more vertex.

The persistence matrix is a prior over how wirings change. The default spreads $h$ evenly over every other stored wiring; E7 also tries one that expects a single spring to flip, spreading $h$ over the wirings one gate away. It helps when changes really are single flips and hurts when whole patterns change, so the right $T$ depends on the world.

**Recall** over a window picks $\hat c=\arg\min_c\sum_t\mathrm{NLL}_c(t)$, with $\mathrm{NLL}_c=-\log\ell_c$.

**Is this wiring new?** Compare the best stored wiring with one detected (Part 9) on the same window:

$$\Lambda=\min_c\sum_t\mathrm{NLL}_c(t)-\sum_t\mathrm{NLL}_{\hat g}(t)\;>\;\tau\quad\Longrightarrow\quad\text{store }\hat g\text{ as a new row of }G$$

with $\tau$ set so that 1% of windows from stored wirings exceed it. Both terms carry the same noise energy (about $\tfrac12$ per coordinate per step), so the difference cancels it; an absolute test on $\min_c\sum_t\mathrm{NLL}_c$ alone has to set its threshold above that noise.

""" + code("context", "ContextBank", "residual_scale", "uniform_transitions", "flip_transitions", "BindingVertex")))
    out[-1]["attachments"] = {"binding.png": {"image/png": fig_binding()}}

    # ---- Part 11
    out.append(md(r"""## 11 · Preview of stage 1.5: two angles per body

*In one sentence:* with two tilt angles per body, small swings are two independent copies of the planar problem, so everything above carries over with $\otimes I_2$; large swings need lifts that do not prefer a direction.

Order each body's state phase-major, $x_v=(q_x,q_y,\dot q_x,\dot q_y)$. Then $A_0\to A_0\otimes I_d$ and $E\to E\otimes I_d$, so for small swings

$$K=K_2\otimes I_d,\qquad M=M_2\otimes I_d\qquad(\text{still 8 learned numbers})$$

where $K_2,M_2$ are the planar maps. Springs pointing in different directions read only their own component, a restriction map $I_2\otimes n_e^\top$ with $n_e$ the spring's unit direction, and the planar $M$ transfers unchanged.

For large swings the lift must treat every direction alike, so it uses only rotation-invariant combinations: $q\|q\|^2,\ q\|\dot q\|^2,\ q\,(q\cdot\dot q),\ \dot q\|q\|^2,\ \dot q\|\dot q\|^2,\ \dot q\,(q\cdot\dot q)$. With $d=2$ that is 16 features instead of the 24 of a full cubic lift.

""" + code("model", "OneHop.per_direction") + "\n\n" + code("fit", "fit_kronecker") + "\n\n" +
        code("lifts", "invariant_features")))

    # ---- Part 12
    out.append(md(fill(r"""## 12 · Hidden information: a clamped sense (step 2)

*In one sentence:* when only angles are measured, each body predicts with step 1's maps and corrects itself through its clamped angle; that correction is where predictive-coding relaxation settles, and its stability splits per graph frequency.

Each body senses its angle with noise of standard deviation $s$; its rate is never measured:

$$\underbrace{y_v(t)}_{\text{clamped sense}}=\underbrace{c}_{[1,\;0]}\,x_v(t)+\varepsilon_v(t),\qquad\varepsilon_v\sim\mathcal N(0,s^2)$$

**The delay line** reads the rate off two readings, $\hat{\dot\theta}(t-1)=A_{\theta\dot\theta}^{-1}\big(y(t)-A_{\theta\theta}y(t-1)\big)\approx\big(y(t)-y(t-1)\big)/\Delta t$, so its rate noise is about $\sqrt2\,s/\Delta t$ at every step.

**The clamped sense.** Each body predicts, $\hat x^-_v=K\hat x_v-M\sum_u g_{uv}(\hat x_v-\hat x_u)$, then relaxes on an energy that weighs its sense against its prediction:

$$E_v(x)=\underbrace{\frac{(y_v-c\,x)^2}{2s^2}}_{\text{sense error}}+\underbrace{\tfrac12(x-\hat x^-_v)^{\top}\,\Pi^{-1}(x-\hat x^-_v)}_{\text{prediction error}}\quad\Longrightarrow\quad\hat x_v=\hat x^-_v+\underbrace{\frac{\Pi c^\top}{c\,\Pi c^\top+s^2}}_{k}\big(y_v-c\,\hat x^-_v\big)$$

$\Pi$ is the prior covariance: fixed at the lone body's steady state (from the Riccati equation, given $q$, the rate variance per step the body allows for unmodelled forces), or growing from a cold start through each body's own $2\times2$ Riccati recursion. Neither shares covariance between bodies.

**Stability per graph frequency.** The error obeys $e(t)=\big(I-(I_n\otimes k)C\big)P\,e(t-1)$, which splits by the spectrum identity into one block $(I-kc)(K-\lambda M)$ per graph frequency. When that block's eigenvalues are a complex pair, $|\mu|^2=\det$, so

$$\rho=\sqrt{(1-k_\theta)\,\det(K-\lambda M)}\approx\sqrt{1-k_\theta}\qquad(\det(K-\lambda M)\approx1\text{: the step nearly preserves area})$$

and every graph mode converges at the rate the angle gain sets.

**Why learning (H2) needs care.** With only $\theta$ sensed, $K\to TKT^{-1}$, $M\to TMT^{-1}$ with $T=\begin{bmatrix}1&0\\ \alpha&\beta\end{bmatrix}$ predicts the same angles, so the learned hidden coordinate can be any mix $\alpha\theta+\beta\dot\theta$; physics has to be read from eigenvalues, which do not change.

**Worked example** ($s$ = 1% of a 0.1 rad swing, $q=10^{-8}$): the steady gain is $k=<<kq>>^{\top}$, so $\rho=<<rho>>$ at $\lambda=0$ and $<<rho4>>$ at $\lambda=4$, against $\sqrt{1-k_\theta}=<<sqrt1k>>$: the error halves every <<halving>> steps. The delay line's rate noise is $\sqrt2\,s/\Delta t=<<dline>>$ rad/s, <<dline_pct>> of the rate scale $a\,\omega_0$.

""", **N) + code("sense", "observe", "lone_gain", "lone_gains", "relax", "clamped_filter", "delay_line", "mode_radius")))

    # ---- tests
    out.append(md(r"""## Where the truth lives: the tests

The equations above are explanations. These tests are what the code is held to (`python -m pytest`):

""" + tests_table() + r"""

**Rebuild this notebook** after changing the library: `python tools/build_math_notebook.py --html`. The code blocks, the numbers and the diagrams follow the code; the prose does not, so read the diff of the prose when the math changes."""))
    return out


def build(html: bool = False) -> Path:
    version = git_version(ROOT)
    today = dt.date.today().isoformat()
    nb = nbf.v4.new_notebook()
    nb.cells = cells(numbers(), version, today)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    OUT.mkdir(exist_ok=True)
    path = OUT / "00_math_walkthrough.ipynb"
    nbf.write(nb, path)
    print("wrote", path.relative_to(ROOT))
    if html:
        (OUT / "html").mkdir(exist_ok=True)
        subprocess.run([sys.executable, "-m", "nbconvert", "--to", "html", "--output-dir", str(OUT / "html"), str(path)],
                       check=True)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", action="store_true")
    build(ap.parse_args().html)
