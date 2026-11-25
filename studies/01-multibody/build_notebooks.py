"""Build the study's presentation notebooks (one per experiment), then execute and export them.

    python build_notebooks.py                 # write the .ipynb files
    python build_notebooks.py --execute       # ... and run them (results are made if missing)
    python build_notebooks.py --execute --html  # ... and export presentation HTML (code hidden) to notebooks/html/

Notebooks only display and trigger: the machinery is in `nsp`, the experiment scripts and `figures.py`.
Every number in a finding is computed from the results when the notebook runs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

STUDY = Path(__file__).resolve().parent
OUT = STUDY / "notebooks"

TAG = "*Step 1 of network-spectral-predictor · pendulums on springs · actual = generated world, nominal = prediction*"


def md(text: str, slide: bool = False):
    c = nbf.v4.new_markdown_cell(text.strip())
    if slide:
        c.metadata["slideshow"] = {"slide_type": "slide"}
    return c


def code(src: str):
    return nbf.v4.new_code_cell(src.strip())


def headers(at: str, now: str, want: str, how: str):
    return md(f"""**Where we're at.** {at}

**Where the system is after the last experiments.** {now}

**Where we want it.** {want}

**How this set moves it.** {how}""")


def setup(exp: str):
    return code(f"""%config InlineBackend.figure_format = 'retina'
import sys
from pathlib import Path
import numpy as np
from IPython.display import display
STUDY = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(STUDY))
import figures as F
from nsp import config, graphs, viz, world as W
F.setup()
cfg = F.cfg("{exp}")
world = config.world_from(cfg)
p = cfg["{exp}"]""")


PLAYER_NOTE = ("Press **Play** to watch at real speed, or drag the slider. Actual bobs are filled, nominal bobs are "
               "hollow, and the whisker between them is the deviation, magnified (pick the factor) and coloured against "
               "the tolerance, like a CMM deviation plot. The trace underneath is the largest deviation over time.")


# --------------------------------------------------------------------------------------------------------------- E0

def e0():
    return [
        md(f"""# E0 · Calibrate the simulator
{TAG}

Every later experiment measures the model against this simulator, so the simulator is checked first."""),
        headers("Step 1, rung 1 (small swings). The plan's math has had a cursory review; this is the first code that runs.",
                "Only scratch checks so far: the closed forms hold, and the one-hop model misses a two-hop term of "
                "relative size κλΔt²/6.",
                "A ground truth trusted to ten times below every model deviation we will report.",
                "Checks velocity Verlet against the exact step for small swings, and against itself at 4× the "
                "substeps for large swings, which have no exact step."),
        setup("e0"),
        code('F.ledger("e0")'),
        md(r"""## The question

Small swings have an exact step; large swings (where $\sin\theta$ matters) only have an integrator:

$$\underbrace{x(t+1)}_{\text{actual, small swings}} = \underbrace{e^{A\Delta t}}_{\text{exact step}}\,x(t), \qquad A = \underbrace{I_n\otimes A_0}_{\text{each body}} - \underbrace{\kappa\,L(g)\otimes E}_{\text{springs}}$$

Here the integrator is under test, so it plays the **nominal** role and the exact step (or a 4× finer integrator) the **actual** one.

**Prediction.** Verlet with 50 substeps ($h = 1$ ms) has a phase error per sample of order $(\omega h)^3$ and a bounded energy error of order $(\omega h)^2$: parts per million.
**Pass if** the deviation at 10 s is under $10^{-4}$ of the swing, ten times below the smallest model deviation reported (about $10^{-3}$).""", slide=True),
        code('runs = F.ensure_results("e0")'),
        md(r"""## The world

Three pendulums in a line, joined by springs. Small swings move as a sum of modes, one per graph frequency $\lambda_k$ (an eigenvalue of the Laplacian $L$), each a lone pendulum stiffened by $\kappa\lambda_k$:

$$\omega_k = \sqrt{\omega_0^2 + \kappa\lambda_k}$$""", slide=True),
        code("F.e0_modes(world, runs[0]);"),
        md(f"""## Nominal vs actual

{PLAYER_NOTE} At ×1 the two worlds coincide exactly; turn the magnification up to see any deviation at all.""",
           slide=True),
        code("""a = runs[0]["arrays"]
display(viz.player(a["theta_exact"], a["theta_verlet"], graphs.path(3), dt=world.dt, tol=1e-4 * p["amplitude"],
                   title="Small swings, 20 s: Verlet (nominal) against the exact step (actual)",
                   subtitle="Tolerance: 10⁻⁴ of the 0.1 rad swing."))
display(viz.player(a["theta_large_ref"], a["theta_large"], graphs.path(3), dt=world.dt,
                   tol=1e-4 * p["large_amplitude"], title="Released from up to 2 rad: 50 substeps (nominal) against 200 (actual)",
                   subtitle="Large swings have no exact step, so the reference is the same integrator, 4× finer."))"""),
        code("""viz.film_strip(a["theta_large_ref"], a["theta_large"], graphs.path(3), dt=world.dt, tol=1e-4 * p["large_amplitude"],
               title="Large swings: the two integrations agree to the width of a line");"""),
        md("## Deviation and energy over 100 s", slide=True),
        code("F.e0_error_growth(runs, world);"),
        code("""m = [r["metrics"] for r in runs]
F.say(f"At 10 s the simulator deviates by {F.spread([x['error_10s'] for x in m])} of the swing for small swings and "
      f"{F.spread([x['error_10s_large'] for x in m])} at 2 rad (mean, then min to max over seeds). Energy drifts by at "
      f"most {viz.fmt(max(x['energy_drift_small'] for x in m))} and {viz.fmt(max(x['energy_drift_large'] for x in m))}. "
      "Any deviation above 10⁻⁴ in later experiments therefore belongs to the model, not the simulator.")"""),
        md("## Inspection report", slide=True),
        code("""mean = lambda k: float(np.mean([x[k] for x in m]))
viz.inspection_report([
    {"characteristic": "Deviation at 10 s, small swings (fraction of swing)", "deviation": mean("error_10s"), "tolerance": 1e-4},
    {"characteristic": "Deviation at 10 s, 2 rad (fraction of swing)", "deviation": mean("error_10s_large"), "tolerance": 1e-4},
    {"characteristic": "Deviation at 100 s, small swings", "deviation": mean("error_end"), "tolerance": 1e-3},
    {"characteristic": "Energy drift over 100 s, small swings", "deviation": mean("energy_drift_small"), "tolerance": 1e-4},
    {"characteristic": "Energy drift over 100 s, 2 rad", "deviation": mean("energy_drift_large"), "tolerance": 1e-4},
])"""),
        md("## Log"),
        code("F.log(runs)"),
    ]


# --------------------------------------------------------------------------------------------------------------- E1

def e1():
    return [
        md(f"""# E1 · Decompose
{TAG}

Do the two learned maps mean something physical: K a lone pendulum, M a spring?"""),
        headers("Step 1, rung 1. The simulator is calibrated (E0).",
                "The ground truth agrees with the exact step to about 10⁻⁵ at 10 s. Nothing has been learned yet.",
                "K and M that are the body and the spring, with ω₀² and κ readable from them.",
                "Fits K and M jointly on every gate pattern of a path of 3; checks them against the closed form, the "
                "frozen-bodies fit and gradient descent; then asks how much data a decomposed model needs compared "
                "with one global map."),
        setup("e1"),
        code('F.ledger("e1")'),
        md(r"""## The question

One least-squares solve over every body, step and gate pattern:

$$\underbrace{x_v(t+1)}_{\text{next state of body } v} = \underbrace{K\,x_v(t)}_{\text{body alone}} \;-\; \underbrace{M}_{\text{spring map}}\sum_{u\sim v}\underbrace{g_{uv}}_{\text{gate}}\,\big(x_v(t)-x_u(t)\big)$$

The closed form says what the answer should be (from expanding the exact step in $\lambda$):

$$K_0 = e^{A_0\Delta t}, \qquad M_0 = \kappa\begin{bmatrix}\tfrac{\Delta t^2}{2} & \tfrac{\Delta t^3}{6}\\[2pt] \Delta t-\tfrac{\omega_0^2\Delta t^3}{3} & \tfrac{\Delta t^2}{2}\end{bmatrix} + O(\Delta t^4)$$

**Prediction.** $K$ within $10^{-4}$ of $K_0$ and $\hat\kappa$ within 0.2%; the frozen-bodies $K$ equals $K_0$ exactly; gradient descent lands on the least-squares answer.
**Pass if** $K$ and $M$ are within 1% of $K_0$ and $M_0$, and $\hat\omega_0^2$ and $\hat\kappa$ within 1%.""", slide=True),
        code('runs = F.ensure_results("e1")'),
        md("## The world: every gate pattern of a path of 3", slide=True),
        code("""e = graphs.path(3)
F.schematics([(f"gates {int(g[0])}{int(g[1])}", 3, e, g) for g in graphs.all_gate_patterns(2)],
             title="Training data: all four ways to switch the two springs (dashed = off)");"""),
        md(f"""## Nominal vs actual

{PLAYER_NOTE}

First, a lone pendulum predicted by **K alone**; training never showed the model a single isolated body. Then the path of 3, predicted by the assembled model $I_3\\otimes K - L\\otimes M$ for 10 s with no correction.""",
           slide=True),
        code("""a = runs[0]["arrays"]
tol = 0.01 * p["amplitude"]
display(viz.player(a["theta_act_lone"].T, a["theta_nom_lone"].T, [], dt=world.dt, tol=tol,
                   title="Three lone pendulums, each predicted by K alone",
                   subtitle="Tolerance: 1% of the 0.1 rad swing."))
display(viz.player(a["theta_act"], a["theta_nom"], graphs.path(3), dt=world.dt, tol=tol,
                   title="Path of 3, all springs on: assembled model (nominal) against the world (actual)",
                   subtitle="Free-running for 10 s from the first state."))"""),
        code("""viz.film_strip(a["theta_act"], a["theta_nom"], graphs.path(3), dt=world.dt, tol=tol,
               title="Path of 3 over 10 s");"""),
        md("## K and M against the closed form", slide=True),
        code("""m = runs[0]["metrics"]
K, M, K0, M0 = (np.array(m[k]) for k in ("K", "M", "K0", "M0"))
rows = []
for name, A, A0 in (("K", K, K0), ("M", M, M0)):
    tol = 0.01 * np.abs(A0).max()
    for i in range(2):
        for j in range(2):
            rows.append({"characteristic": f"{name}{i + 1}{j + 1}", "nominal": A[i, j], "actual": A0[i, j],
                         "deviation": A0[i, j] - A[i, j], "tolerance": tol})
viz.inspection_report(rows, title="Seed 0: fitted (nominal) against closed form (actual); tolerance 1% of the largest entry")"""),
        code("""F.say(f"Across seeds K is within {F.spread([r['metrics']['dK'] for r in runs])} of K₀ and M within "
      f"{F.spread([r['metrics']['dM'] for r in runs])} of M₀ (largest entry). The frozen-bodies K matches K₀ to "
      f"{F.spread([r['metrics']['dK_frozen'] for r in runs])}, and its M agrees with the joint fit's to "
      f"{F.spread([r['metrics']['dM_frozen_joint'] for r in runs])}: the joint fit does not trade body against spring.")"""),
        md(r"""## The physics, read back

$$\hat\omega_0^2 = -\Big[\tfrac{1}{\Delta t}\log K\Big]_{21}, \qquad \hat\kappa = \frac{M_{21}}{\Delta t - \hat\omega_0^2\Delta t^3/3}$$""",
           slide=True),
        code("""mean = lambda k: float(np.mean([r["metrics"][k] for r in runs]))
viz.inspection_report([
    {"characteristic": "ω₀² (rad²/s²)", "nominal": mean("omega0_sq"), "actual": world.omega0**2,
     "deviation": world.omega0**2 - mean("omega0_sq"), "tolerance": 0.01 * world.omega0**2},
    {"characteristic": "κ (1/s²)", "nominal": mean("kappa"), "actual": world.kappa,
     "deviation": world.kappa - mean("kappa"), "tolerance": 0.01 * world.kappa},
    {"characteristic": "κ without the Δt³ correction", "nominal": mean("kappa_bare"), "actual": world.kappa,
     "deviation": world.kappa - mean("kappa_bare"), "tolerance": 0.01 * world.kappa},
], title="Physical parameters, mean over seeds")"""),
        code("F.e1_frequencies(runs[0], world);"),
        md("## Three ways to fit, one answer", slide=True),
        code("F.e1_gradient_descent(runs);"),
        code("""F.say(f"Gradient descent, using only each body's error times its own inputs, reaches the closed-form answer to 10⁻⁸ "
      f"in {F.spread([r['metrics']['gd_iterations'] for r in runs])} steps; the condition number of the data is "
      f"{F.spread([r['metrics']['condition'] for r in runs])}.")"""),
        md(r"""## Why decompose: data efficiency

A ring of 8 fitted from **one trajectory**: nsp's 8 shared numbers against one global map of $16\times16 = 256$. A ring has repeated graph frequencies (pairs of modes with the same $\lambda$), and one trajectory only ever excites one direction of each pair, so six of the sixteen state directions are never seen. A global map cannot learn what it never sees; nsp's numbers are pinned down by every body at once.""",
           slide=True),
        code("F.e1_scarcity(runs);"),
        code("""a = [r["arrays"] for r in runs]
F.say(f"From 2,000 clean steps the global map still misses by {F.spread([x['global_roll'][-1, 0] for x in a])} at 10 s, "
      f"while nsp misses by {F.spread([x['nsp_roll'][-1, 0] for x in a])}. At 1% noise nsp settles near "
      f"{F.spread([x['nsp_roll'][-1, -1] for x in a])}: noise in the inputs biases least squares, and more data does not remove "
      "that bias (errors in variables).")"""),
        md("## Inspection report", slide=True),
        code("""viz.inspection_report([
    {"characteristic": "max |K − K₀|", "deviation": mean("dK"), "tolerance": 0.01 * np.abs(K0).max()},
    {"characteristic": "max |M − M₀|", "deviation": mean("dM"), "tolerance": 0.01 * np.abs(M0).max()},
    {"characteristic": "ω₀² recovered (relative)", "deviation": 1 - mean("omega0_sq") / world.omega0**2, "tolerance": 0.01},
    {"characteristic": "κ recovered (relative)", "deviation": 1 - mean("kappa") / world.kappa, "tolerance": 0.01},
    {"characteristic": "Path of 3, deviation at 10 s", "deviation": float(np.mean([r["metrics"]["path3"]["200"] for r in runs])),
     "tolerance": 0.01},
    {"characteristic": "Lone pendulum by K alone, deviation at 10 s",
     "deviation": float(np.mean([r["metrics"]["lone"]["200"] for r in runs])), "tolerance": 0.01},
])"""),
        md("## Log"),
        code("F.log(runs)"),
    ]


# --------------------------------------------------------------------------------------------------------------- E2

def e2():
    return [
        md(f"""# E2 · Transfer across body counts
{TAG}

Fit on 3 and 5 pendulums; predict 8, wired four different ways, with no refit."""),
        headers("Step 1, rung 1. K and M decompose into body and spring (E1).",
                "A fit on one small graph recovers ω₀² and κ to about 0.1%.",
                "One fit that predicts any number of bodies, wired any way.",
                "Fits on 3- and 5-body paths and rings (random gates), then predicts 8-body paths, rings, stars and "
                "complete graphs without refitting, at three spring stiffnesses κ."),
        setup("e2"),
        code('F.ledger("e2")'),
        md(r"""## The question

The assembled model splits into one $2\times 2$ problem per graph frequency:

$$\operatorname{spec}\big(I_n\otimes K - L(g)\otimes M\big) \;=\; \bigcup_k \operatorname{spec}\big(\underbrace{K - \lambda_k M}_{\text{mode } k}\big)$$

So the model only ever sees a graph through its frequencies $\lambda_k$, and transfer should follow $\lambda$, not the body count. What it misses is the two-hop term, of relative size

$$\frac{\text{two-hop}}{\text{one-hop}} \approx \frac{\kappa\lambda\Delta t^2}{6}$$

**Prediction.** Paths and rings of 8 ($\lambda\le 4$) sit at the edge of the training range ($\lambda\le 3.6$) and transfer; stars and complete graphs reach $\lambda = 8$ and come out several times worse, at every $\kappa$.
**Pass if** path-8 and ring-8 deviations at 10 s are within 1.5× of a model fitted on those graphs themselves.""",
           slide=True),
        code('runs = F.ensure_results("e2")'),
        md("## The worlds", slide=True),
        code("""F.schematics([("train: path of 3", 3, graphs.path(3), None), ("train: ring of 3", 3, graphs.ring(3), None),
              ("train: path of 5", 5, graphs.path(5), None), ("train: ring of 5", 5, graphs.ring(5), None)],
             title="Fitted on these (plus random gate patterns of each)", cols=4);
F.schematics([("test: path of 8", 8, graphs.path(8), None), ("test: ring of 8", 8, graphs.ring(8), None),
              ("test: star of 8", 8, graphs.star(8), None), ("test: complete graph of 8", 8, graphs.complete(8), None)],
             title="Predicted with no refit (springs between non-neighbours drawn as arcs)", cols=2);"""),
        code("F.e2_graph_frequencies(runs[0]);"),
        md(f"""## Nominal vs actual, at κ = 4

{PLAYER_NOTE} The ring stays inside the training frequencies; the star does not.""", slide=True),
        code("""a = runs[0]["arrays"]
tol = 0.01 * p["amplitude"]
for nm, edges in (("ring", graphs.ring(8)), ("star", graphs.star(8))):
    display(viz.player(a[f"theta_act_k4_{nm}"], a[f"theta_nom_k4_{nm}"], edges, dt=world.dt, tol=tol,
                       title=f"{nm.capitalize()} of 8, κ = 4: transferred model (nominal) against the world (actual)",
                       subtitle="Fitted only on 3- and 5-body paths and rings. Tolerance: 1% of the swing."))"""),
        code("""viz.film_strip(a["theta_act_k4_star"], a["theta_nom_k4_star"], graphs.star(8), dt=world.dt, tol=tol,
               title="Star of 8, κ = 4: the hub's springs drive the deviation");"""),
        md("## Where transfer holds", slide=True),
        code("F.e2_frequency_error(runs);"),
        code("F.e2_ratios(runs);"),
        code("F.e2_rollouts(runs[0], 4.0, world);"),
        code("""g = lambda k, nm, key: [r["metrics"]["by_kappa"][k]["graphs"][nm][key] for r in runs]
ratio = lambda k, nm: [v for v in g(k, nm, "ratio") if not isinstance(v, str)]
F.say(f"Paths and rings of 8 transfer within {F.spread(ratio('1', 'path') + ratio('1', 'ring'))}× of their own fits at "
      f"κ = 1, and closer at stiffer springs. Stars land at {F.spread(ratio('1', 'star') + ratio('4', 'star') + ratio('16', 'star'))}× "
      "whatever the stiffness: the ratio tracks the graph frequencies, not the body count. A complete graph has only two "
      "graph frequencies (0 and 8), so a model fitted on it alone is exact; that is why its ratio is undefined.")"""),
        md("## Inspection report", slide=True),
        code("""rows = []
for k in ("1", "4", "16"):
    for nm in ("path", "ring", "star"):
        v = float(np.median(ratio(k, nm)))
        rows.append({"characteristic": f"{nm} of 8, κ = {k}: deviation at 10 s ÷ own fit's", "nominal": v,
                     "actual": 1.0, "deviation": v - 1, "tolerance": 0.5,
                     "status": viz.status(v - 1, 0.5) if nm != "star" else ("out" if v > 1.5 else "in")})
viz.inspection_report(rows, title="Transfer ratios (median over seeds); the plan's criterion covers paths and rings")"""),
        md("## Log"),
        code("F.log(runs)"),
    ]


# --------------------------------------------------------------------------------------------------------------- E3

def e3():
    return [
        md(f"""# E3 · Reconfigure by gating
{TAG}

Switch springs on and off, even mid-run, and keep predicting with no refit."""),
        headers("Step 1, rung 1. Transfer across body counts works inside the training graph frequencies (E2).",
                "Stars and complete graphs, whose graph frequencies lie outside the training range, transfer about "
                "6× worse.",
                "Springs switched on and off, even mid-run, with no refit.",
                "Fits on half of the 32 gate patterns of a ring of 5, predicts the other half, and cuts a spring "
                "halfway through a run."),
        setup("e3"),
        code('F.ledger("e3")'),
        md(r"""## The question

The step is linear in the gates, one spring at a time:

$$L(g) = \delta^\top \operatorname{diag}(g)\,\delta = \sum_e g_e\,\underbrace{b_e b_e^\top}_{\text{spring } e\text{'s own Laplacian}}$$

so a model fitted on some patterns has nothing pattern-specific to forget. Cutting a stretched spring removes its energy $\tfrac{\kappa}{2}(\theta_v-\theta_u)^2$: a real jump in the world, not a model error.

**Prediction.** Held-out patterns are predicted as well as seen ones, and the deviation does not jump when a spring is cut.
**Pass if** held-out deviations, and the deviation just after the cut, are within 1.5× of their references.""",
           slide=True),
        code('runs = F.ensure_results("e3")'),
        md("## The world: a ring of 5, before and after the cut", slide=True),
        code("""e = graphs.ring(5)
g_off = np.ones(5); g_off[p["switch_edge"]] = 0
F.schematics([("before: all five springs", 5, e, None), ("after: spring 1–2 cut", 5, e, g_off)], cols=2);"""),
        md(f"""## Nominal vs actual: cutting a spring at 5 s, κ = 4

{PLAYER_NOTE} The spring between bodies 1 and 2 turns dashed at the cut; the model swaps $L(g)$ at the same moment.""",
           slide=True),
        code("""a = runs[0]["arrays"]
tol = 0.01 * p["amplitude"]
display(viz.player(a["switch_act_k4"], a["switch_nom_k4"], graphs.ring(5), a["switch_gates_k4"], dt=world.dt, tol=tol,
                   switch_steps=[p["switch_step"]], title="Ring of 5, κ = 4: spring 1–2 cut at 5 s",
                   subtitle="Free-running prediction for 10 s; tolerance 1% of the swing."))
viz.film_strip(a["switch_act_k4"], a["switch_nom_k4"], graphs.ring(5), a["switch_gates_k4"], dt=world.dt, tol=tol,
               times=[0, 4.5, 4.95, 5.05, 7.5, 10], switch_steps=[p["switch_step"]],
               title="Around the cut");"""),
        md("## Seen and unseen patterns", slide=True),
        code("F.e3_patterns(runs);"),
        code("F.e3_switch(runs[0], 4.0, world, p['switch_step']);"),
        code("""b = lambda k, key: [r["metrics"]["by_kappa"][k][key] for r in runs]
sw = lambda k, key: [r["metrics"]["by_kappa"][k]["switch"][key] for r in runs]
F.say(f"At κ = 1, held-out patterns deviate by {F.spread(b('1', 'held_rollout'))} at 10 s against "
      f"{F.spread(b('1', 'seen_rollout'))} for seen ones. Cutting a spring changes the world's energy by "
      f"{F.spread([100 * x for x in b('4', 'energy_jump_mean')])}% at κ = 4, while the model's deviation goes from "
      f"{F.spread(sw('4', 'before'))} just before the cut to {F.spread(sw('4', 'just_after'))} just after.")"""),
        md("## Inspection report", slide=True),
        code("""rows = []
for k in ("1", "4"):
    held = np.mean(b(k, "held_rollout")) / np.mean(b(k, "seen_rollout"))
    jump = np.mean(sw(k, "just_after")) / np.mean(sw(k, "before"))
    rows += [{"characteristic": f"κ = {k}: held-out ÷ seen deviation at 10 s", "nominal": held, "actual": 1.0,
              "deviation": held - 1, "tolerance": 0.5},
             {"characteristic": f"κ = {k}: deviation just after ÷ just before the cut", "nominal": jump, "actual": 1.0,
              "deviation": jump - 1, "tolerance": 0.5},
             {"characteristic": f"κ = {k}: world's energy change at the cut (physics)",
              "actual": viz.pct(float(np.mean(b(k, "energy_jump_mean")))), "status": None}]
viz.inspection_report(rows, title="Reconfiguration, means over seeds")"""),
        md("## Log"),
        code("F.log(runs)"),
    ]


# --------------------------------------------------------------------------------------------------------------- E4

def e4():
    return [
        md(f"""# E4 · Large swings (rung 2)
{TAG}

When the swing is large enough that $\\sin\\theta \\neq \\theta$, which body lift keeps the model on track?"""),
        headers("Step 1, moving from rung 1 (small swings) to rung 2 (sin θ matters).",
                "In the linear world the model transfers across body counts and gate patterns (E2, E3).",
                "A body map that holds at large swings, with the springs unchanged.",
                "Releases a path of 3 from 0.1 to 2 rad and compares three body lifts: none, + $\\sin\\theta$, + $\\dot\\theta\\cos\\theta$."),
        setup("e4"),
        code('F.ledger("e4")'),
        md(r"""## The question

The world is no longer linear:

$$\ddot\theta_v = -\omega_0^2\sin\theta_v - \kappa\sum_{u\sim v} g_{uv}\,(\theta_v-\theta_u)$$

Expanding one step of length $\Delta t$ shows which features the body map needs:

$$\dot\theta(t+\Delta t) = \dot\theta + \Delta t\,\underbrace{\ddot\theta}_{-\omega_0^2\sin\theta\,-\,\kappa L\theta} + \tfrac{\Delta t^2}{2}\,\underbrace{\theta^{(3)}}_{-\omega_0^2\,\dot\theta\cos\theta\,-\,\kappa L\dot\theta} + \dots$$

($\theta^{(3)}$ is the third time derivative.) So the lift $\psi(x_v)$ needs $\sin\theta$ and its time derivative $\dot\theta\cos\theta$; the springs keep reading plain differences:

$$x_v(t+1) = W^\top\big[\,\psi(x_v(t))\,;\; s_v(t)\,\big], \qquad s_v(t) = \sum_{u\sim v} g_{uv}\big(x_v(t)-x_u(t)\big)$$

**Prediction.** Each added term cuts the deviation about 10× at large swings.
**Pass if** the best lift stays under 1% deviation at 10 s up to 2 rad.""", slide=True),
        code('runs = F.ensure_results("e4")'),
        md("## The world gets slower as it swings wider", slide=True),
        code("F.e4_frequency(world, p['amplitudes']);"),
        md(f"""## Nominal vs actual, released from up to 2 rad

{PLAYER_NOTE} The first player uses no lift; the second uses the full lift.""", slide=True),
        code("""a = runs[0]["arrays"]
tol = 0.01 * p["player_amplitude"]
display(viz.player(a["theta_act"], a["theta_nom_x"], graphs.path(3), dt=world.dt, tol=tol, height=330,
                   title="No lift: a linear model of a nonlinear pendulum",
                   subtitle="Tolerance: 1% of the 2 rad release angle."))
display(viz.player(a["theta_act"], a["theta_nom_sin+cos"], graphs.path(3), dt=world.dt, tol=tol, height=330,
                   title="Full lift: sin θ and its rate of change added to each body", subtitle="Same world, same springs."))"""),
        code("""viz.film_strip(a["theta_act"], a["theta_nom_x"], graphs.path(3), dt=world.dt, tol=tol, amplify=1,
               title="No lift at 2 rad, unmagnified: the nominal world falls out of step");"""),
        md("## Deviation by lift and amplitude", slide=True),
        code("F.e4_errors(runs);"),
        code("F.e4_phase(runs[0]);"),
        code("""from nsp.lifts import LABELS
r2 = lambda lift: [r["metrics"]["by_amplitude"]["2"]["lifts"][lift]["rollout"]["200"] for r in runs]
F.say(f"At 2 rad the deviation at 10 s is {F.spread(r2('x'))} with no lift, {F.spread(r2('sin'))} with sin θ and "
      f"{F.spread(r2('sin+cos'))} with $\\\\dot\\\\theta\\\\cos\\\\theta$ added. The full lift costs 4 body features instead of 2; the springs are "
      "untouched.")"""),
        md("## Inspection report", slide=True),
        code("""rows = []
for lift in p["lifts"]:
    for amp in ("1", "2"):
        v = float(np.mean([r["metrics"]["by_amplitude"][amp]["lifts"][lift]["rollout"]["200"] for r in runs]))
        rows.append({"characteristic": f"{LABELS[lift]} at {amp} rad: deviation at 10 s", "deviation": v, "tolerance": 0.01})
viz.inspection_report(rows, title="Large swings, means over seeds (the criterion applies to the best lift)")"""),
        md("## Log"),
        code("F.log(runs)"),
    ]


# --------------------------------------------------------------------------------------------------------------- E5

def e5():
    return [
        md(f"""# E5 · Detect the springs (step 1b)
{TAG}

The gates are no longer given. Can the learned body and spring maps find which springs are live?"""),
        headers("Step 1b: the springs are no longer given.",
                "With the gates known, the model predicts every configuration tested so far (E2 to E4).",
                "The gates recovered from data, with no spurious springs.",
                "Solves for all 10 candidate springs of 5 bodies, by least squares with and without non-negativity, "
                "from one or three trajectories, across noise levels and window lengths."),
        setup("e5"),
        code('F.ledger("e5")'),
        md(r"""## The question

With $K$ and $M$ known, the step is linear in the gates. Every pair $(a, b)$ is a candidate spring:

$$\underbrace{x_v(t+1) - K x_v(t)}_{\text{residual of body } v} \;=\; \sum_{(a,b)} g_{ab}\,G_{ab}, \qquad G_{ab} = \begin{cases} -M\,(x_b - x_a) & \text{at } b\\ +M\,(x_b - x_a) & \text{at } a\end{cases}$$

**Prediction.** Clean data give the gates exactly, with one exception: when several identical pendulums hang free, they all swing at $\omega_0$, and from one trajectory some combinations of springs among them change no prediction. Non-negativity ($g\ge 0$) or a second trajectory removes that blind spot.
**Pass if** the AUROC is at least 0.95 at 1% noise and no never-wired pair scores above 0.5.""", slide=True),
        code('runs = F.ensure_results("e5")'),
        md("## The world: a ring of 5, and 10 candidate springs", slide=True),
        code("""ring, pairs = graphs.ring(5), graphs.complete(5)
g_one = np.array([0, 0, 0, 1, 0.0])
F.schematics([("the ring's 5 springs, here one live", 5, ring, g_one),
              ("all 10 candidates (5 never wired)", 5, pairs, None)], cols=2);"""),
        code("F.e5_blind_spot(runs[0]);"),
        md(f"""## Nominal vs actual: rollouts with the detected gates

The blind-spot case: one live spring, 50 steps of one noisy trajectory. Each nominal world runs on the gates its method detected. {PLAYER_NOTE}""",
           slide=True),
        code("""a = runs[0]["arrays"]
tol = 0.01 * p["amplitude"]
for mt, label in (("lstsq", "plain least squares"), ("nnls", "non-negative least squares")):
    g = a[f"g_{mt}"]
    keep = [k for k in range(len(pairs)) if a["truth"][k] > 0.5 or abs(g[k]) > 0.1]
    display(viz.player(a["theta_act"], a[f"theta_nom_{mt}"], [pairs[k] for k in keep], a["truth"][keep], dt=world.dt,
                       tol=tol, nominal_gates=(np.abs(g[keep]) > 0.1).astype(float),
                       title=f"Gates from {label}",
                       subtitle="Free-running for 10 s on the detected gates (all of them, at their estimated strengths). "
                                "Blue dashed springs: gates above 0.1 that the actual world does not have."))"""),
        code("""g = a["g_nnls"]
spur = [k for k in range(len(pairs)) if a["truth"][k] < 0.5]
k = max(spur, key=lambda j: g[j])
u, v = pairs[k]
end = np.abs(a["theta_nom_nnls"][-1] - a["theta_act"][-1])
rest = max(end[i] for i in range(5) if i not in (u, v))
F.say(f"Even non-negative least squares leaves a spring of strength {viz.fmt(g[k])} between bodies {u + 1} and {v + 1}. By "
      f"10 s they deviate by {viz.fmt(end[u])} and {viz.fmt(end[v])} rad, against at most {viz.fmt(rest)} for the others. "
      "Pendulums with nothing else attached all swing at ω₀, so they are in resonance with each other: a weak spring "
      "between two of them trades swing back and forth steadily instead of averaging out.")"""),
        code("F.e5_matrices(runs[0]);"),
        md("## Detection across windows, noise and trajectories", slide=True),
        code("F.e5_exact(runs, dt=world.dt);"),
        code("""rows = [row for r in runs for row in r["metrics"]["rows"]]
pick = lambda **kw: [row for row in rows if all(row[k] == v for k, v in kw.items())]
sel = lambda mt, k: pick(kappa=1.0, window=50, noise=0.01, method=mt, trajectories=k)
F.say(f"At κ = 1, 1% noise and 50 steps, plain least squares on one trajectory is right on all 10 pairs in "
      f"{viz.pct(np.mean([r['exact'] for r in sel('lstsq', 1)]))} of trials, with spurious gates up to "
      f"{viz.fmt(max(r['max_spurious'] for r in sel('lstsq', 1)))}. Non-negativity raises that to "
      f"{viz.pct(np.mean([r['exact'] for r in sel('nnls', 1)]))}, and three trajectories to "
      f"{viz.pct(np.mean([r['exact'] for r in sel('lstsq', 3)]))}. Stiffer springs (κ = 16) are easier to detect: the "
      "spring's effect per step grows with κΔt.")"""),
        md("## Inspection report", slide=True),
        code("""rep = []
for mt, k in (("lstsq", 1), ("nnls", 1), ("lstsq", 3), ("nnls", 3)):
    s = sel(mt, k)
    au = float(np.mean([r["auroc"] for r in s]))
    sp = float(max(r["max_spurious"] for r in s))
    name = f"{'least squares' if mt == 'lstsq' else 'non-negative'}, {k} traj."
    rep += [{"characteristic": f"{name}: AUROC", "nominal": au, "status": "in" if au >= 0.95 else "out", "tolerance": "≥ 0.95"},
            {"characteristic": f"{name}: largest never-wired gate", "nominal": sp, "status": "in" if sp <= 0.5 else "out",
             "tolerance": "≤ 0.5"}]
viz.inspection_report(rep, title="κ = 1, 1% noise, 50 steps, across seeds")"""),
        md("## Log"),
        code("F.log(runs)"),
    ]


# --------------------------------------------------------------------------------------------------------------- E6

def e6():
    return [
        md(f"""# E6 · Transfer by recall
{TAG}

The hippocampal-analog test: recall which stored structure is active, rather than re-derive it."""),
        headers("Step 1b, last experiment: the alternative way to transfer, by recall.",
                "Detection works but needs about 50 steps and has a blind spot (E5).",
                "Known structures recalled in a few steps, and new ones recognized as new.",
                "Stores 16 of the 32 ring patterns in a bank, filters a belief over them, and compares recall with "
                "detection: accuracy by window, re-locking after a switch, and the novelty test."),
        setup("e6"),
        code('F.ledger("e6")'),
        md(r"""## The question

A **structural vertex** stores a bank $G$ of gate patterns, one row $G_c$ per context. A **binding vertex** holds a belief $\pi(t)$ over them and, like any vertex, predicts and corrects:

$$\underbrace{\alpha(t+1)}_{\text{unnormalised belief}} = \underbrace{\operatorname{diag}\big(\ell(t+1)\big)}_{\text{evidence gate}}\;\underbrace{T^\top}_{\text{contexts persist}}\;\alpha(t), \qquad \pi = \frac{\alpha}{\mathbf 1^\top\alpha}, \qquad g(t) = G^\top\pi(t)$$

$$\ell_c(t+1) = \exp\!\Big(-\tfrac12\big\|\,x(t+1) - \big(I\otimes K - L(G_c)\otimes M\big)\,x(t)\big\|^2_{\Sigma^{-1}}\Big), \qquad T = (1-h)\,I + \tfrac{h}{C-1}\,(\mathbf 1\mathbf 1^\top - I)$$

A structure is new when the best stored one loses to one detected on the same window by more than a threshold $\tau$:

$$\Lambda = \min_c \sum_t \mathrm{NLL}_c(t) - \sum_t \mathrm{NLL}_{\hat g}(t) > \tau$$

**Prediction.** Recall reaches 95% within 10 steps where detection needs about 50; after a switch the binding vertex re-locks in a few steps; the likelihood ratio flags new structures.
**Pass if** recall ≥ 95% within 10 steps, re-locking is 5× faster than detection, and the error after a switch is within 1.5× of known gates.""",
           slide=True),
        code('runs = F.ensure_results("e6")'),
        md("## The world: a ring of 5 and a bank of stored structures", slide=True),
        code("""bank = runs[0]["arrays"]["bank_patterns"]
ring = graphs.ring(5)
F.schematics([(f"stored context {c}", 5, ring, bank[c]) for c in range(6)], cols=3,
             title=f"Six of the {len(bank)} stored structures (dashed = spring off)");"""),
        md(f"""## Nominal vs actual: a switch between two stored contexts at 7.5 s

Here the nominal world is the **one-step prediction** made with the binding vertex's current best structure; blue dashed springs mark where its structure differs from the world's. {PLAYER_NOTE}""",
           slide=True),
        code("""a = runs[0]["arrays"]
display(viz.player(a["theta_act"], a["theta_nom"], ring, a["gates_act"], dt=world.dt, tol=0.05 * p["amplitude"],
                   nominal_gates=a["gates_nom"], switch_steps=[p["switch_step"]],
                   title="Contexts switch at 7.5 s; the binding vertex follows",
                   subtitle="One-step predictions. Measurement noise is 1% of the swing, so one-step deviations of a few "
                            "10⁻³ rad are noise; tolerance 5% of the swing. Watch the blue dashed springs after the switch."))"""),
        code("F.e6_belief(runs[0], world, p['switch_step']);"),
        md("## Recall against detection", slide=True),
        code("F.e6_recall_vs_detect(runs);"),
        code("F.e6_relock(runs);"),
        code("""mm = [r["metrics"] for r in runs]
w = list(mm[0]["windows"])
at = lambda key, win: [x[key][w.index(win)] for x in mm]
F.say(f"From 5 steps, recall picks the right structure {viz.pct(np.mean(at('recall_accuracy', 5)))} of the time; detecting "
      f"all 10 pairs from scratch gets {viz.pct(np.mean(at('detect_accuracy', 5)))}. After a switch the binding vertex "
      f"re-locks in a median of {F.spread([x['relock_filter_median'] for x in mm])} steps, against "
      f"{F.spread([x['relock_detect_median'] for x in mm])} for a sliding 50-step detector.")"""),
        md("## Is this structure new?", slide=True),
        code("F.e6_novelty(runs, window=20);"),
        code("""nv = lambda win, key: [x["novelty"][str(win)][key] for x in mm]
F.say(f"With 20 steps and 1% false alarms, the likelihood ratio flags {viz.pct(np.mean(nv(20, 'glr_flagged')))} of new "
      f"structures; the absolute fit test flags {viz.pct(np.mean(nv(20, 'abs_flagged')))}. Comparing against a freshly "
      "detected structure cancels the noise that an absolute threshold has to sit above.")"""),
        md("## Inspection report", slide=True),
        code("""err = lambda k: float(np.mean([x["error_after_switch"][k] for x in mm]))
rec10 = float(np.mean(at("recall_accuracy", 10)))
speed = float(np.mean([x["relock_detect_median"] for x in mm]) / np.mean([x["relock_filter_median"] for x in mm]))
viz.inspection_report([
    {"characteristic": "Recall within 10 steps", "nominal": rec10, "tolerance": "≥ 0.95", "status": "in" if rec10 >= 0.95 else "out"},
    {"characteristic": "Re-lock speed-up over sliding detection", "nominal": speed, "tolerance": "≥ 5×",
     "status": "in" if speed >= 5 else "out"},
    {"characteristic": "Error after a switch ÷ known gates", "nominal": err("filter") / err("oracle"), "actual": 1.0,
     "deviation": err("filter") / err("oracle") - 1, "tolerance": 0.5},
    {"characteristic": "New structures flagged in 20 steps (likelihood ratio)", "nominal": float(np.mean(nv(20, "glr_flagged"))),
     "tolerance": "at 1% false alarms", "status": None},
], title="Transfer by recall, across seeds")"""),
        md("## Log"),
        code("F.log(runs)"),
    ]


# --------------------------------------------------------------------------------------------------------------- E7

def e7():
    return [
        md(f"""# E7 · Everything together: eight pendulums, rewired every 2.5 s
{TAG}

The capstone of step 1: what passed one experiment at a time (E1 to E6), all at once, on a larger system that keeps changing."""),
        headers("Step 1, the capstone. Each piece passed on its own: decomposition (E1), transfer across body counts (E2), "
                "gating (E3), detection with non-negativity (E5) and recall (E6).",
                "Every test so far used 3 to 5 bodies and one change at a time. Recall beat detection on a ring of 5 "
                "with half of its 32 wirings stored (E6).",
                "Eight pendulums rewired every few seconds, predicted by maps learned only on 3 and 5 bodies, with each "
                "new wiring picked from all 256 possible ones, and predictions within 1.5× of knowing the wiring.",
                "One 30 s run of a ring of 8, rewired every 2.5 s to a random wiring. The binding vertex over the "
                "complete bank is compared with known gates, sliding-window detection and a model that never updates. "
                "A sweep then varies how often and how the wiring changes, and the noise."),
        setup("e7"),
        code('F.ledger("e7")'),
        md(r"""## The question

Nothing new is learned here. The step is E2's, learned on 3- and 5-body graphs: $P(g)=I_8\otimes K-L(g)\otimes M$. The binding vertex is E6's, but its bank $G$ now holds every one of the $2^8=256$ wirings of the ring, so whatever the world does, the right answer is stored.

What changes is the cost of switching. Each step the vertex expects to stay with probability $1-h$ and spreads $h$ over the other $C-1$ wirings, with $h=1/50$, the rewiring rate. To move its belief to a new wiring, the evidence has to pay for that prior:

$$\underbrace{\log\frac{(1-h)\,(C-1)}{h}}_{\text{evidence needed to switch}}=\log\frac{0.98\times255}{0.02}\approx 9.4\ \text{nats}$$

The evidence per step is the gap in negative log-likelihood between the old and new wiring, and it grows with how much the springs that changed are stretched:

$$\Delta\mathrm{NLL}(t)\approx\tfrac12\Big\|\big(\underbrace{(L_{\text{new}}-L_{\text{old}})}_{\text{the springs that changed}}\otimes M\big)\,x(t)\Big\|^2_{\Sigma^{-1}}$$

So re-locking should take a few steps when the changed springs are well stretched, and longer when they are barely stretched. Detection has no prior to pay, but it waits for its window to fill with the new wiring.

**Prediction.** Recall re-locks in a few steps, as in E6, so the right wiring is in force at least 90% of the time when the wiring changes every 50 steps. Detection is held back by its window (about 20 steps for a 20-step window). Predictions one second ahead stay within 1.5× of those made with known gates.
**Pass if** the right wiring is in force on at least 90% of steps, the median re-lock is at most 5 steps and at least 5× faster than the best detector, and the deviation one second ahead is within 1.5× of known gates.""", slide=True),
        code('runs = F.ensure_results("e7")'),
        md("## The world: a ring of 8, and every wiring it can take", slide=True),
        code("""F.schematics([("a ring of 8, all springs live", 8, graphs.ring(8), None)], cols=1);
F.e7_bank(runs[0]);"""),
        md(f"""## Nominal vs actual: one second ahead

Here the nominal world is the prediction made **one second earlier**, from what was measured then, with the wiring the method believed then. Even with known gates, a prediction made just before a rewiring misses after it; what differs between methods is how quickly they recover. Blue dashed springs mark where the wiring behind the prediction differs from the actual one. {PLAYER_NOTE}""",
           slide=True),
        code("""a = runs[0]["arrays"]
tol = 0.05 * p["amplitude"]
sw = list(range(p["switch_every"], p["total_steps"], p["switch_every"]))
for name, label in (("recall", "Binding vertex over the complete bank"), ("detect-20", "Sliding detection, 20-step window")):
    display(viz.player(a["theta_act"], a[f"theta_nom_{name}"], graphs.ring(8), a["gates_act"], dt=world.dt, tol=tol,
                       nominal_gates=a[f"gates_nom_{name}"], switch_steps=sw,
                       title=f"{label}: the world as predicted one second earlier",
                       subtitle="Ring of 8, rewired every 2.5 s (orange marks). Tolerance: 5% of the 0.1 rad swing."))"""),
        code("""viz.film_strip(a["theta_act"], a["theta_nom_recall"], graphs.ring(8), a["gates_act"], dt=world.dt, tol=tol,
               times=[7.0, 7.5, 8.0, 8.5, 9.0, 9.5], switch_steps=sw,
               title="Binding vertex around the rewiring at 7.5 s (gates shown are the actual ones)");"""),
        md("## Tracking the wiring", slide=True),
        code("F.e7_timeline(runs[0], world, p['switch_every']);"),
        code("F.e7_event(runs);"),
        code("""mm = [r["metrics"] for r in runs]
c = lambda k: [x["correct"][k] for x in mm]
rl = lambda k: [x["relock_median"][k] for x in mm]
F.say(f"The binding vertex has the right wiring in force on {F.spread([100 * v for v in c('recall')])}% of steps and "
      f"re-locks in a median of {F.spread(rl('recall'))} steps. Detection with a 20-step window manages "
      f"{F.spread([100 * v for v in c('detect-20')])}% and {F.spread(rl('detect-20'))} steps; with a 50-step window, "
      f"{F.spread([100 * v for v in c('detect-50')])}% and {F.spread(rl('detect-50'))} steps. Each window has to fill with "
      "the new wiring before detection can see it; the vertex only has to collect about 9 nats of evidence.")"""),
        md("## What it does to prediction", slide=True),
        code("F.e7_deviation(runs[0], runs, world, p['switch_every'], p['horizon']);"),
        code("""d = lambda k: [x["dev_ratio"][k] for x in mm]
offs = runs[0]["arrays"]["aligned_offsets"]
back = np.mean([r["arrays"]["aligned_dev_recall"] for r in runs], 0)
back_at = int(offs[(offs > 0) & (back < 1.5)][0])
F.say(f"One second ahead, the binding vertex's predictions deviate {F.spread(d('recall'))}× as much as predictions made "
      f"with known gates, averaged over the run. Detection with a 20-step window reaches {F.spread(d('detect-20'))}×, and a "
      f"model that never updates its wiring {F.spread(d('stale'))}×. Right after a rewiring every method is about as wrong "
      "as the others, because none of them has seen the new wiring yet; the vertex is back within 1.5× of known gates "
      f"{back_at} steps later.")"""),
        md(r"""## Why some rewirings are slow to see

A spring only shows itself through the force it carries, and that force is proportional to its stretch $\theta_v-\theta_u$. When the springs that changed connect pendulums that happen to swing together, they carry almost no force, the evidence per step is small, and the vertex needs more steps to pay its 9.4-nat prior.""", slide=True),
        code("F.e7_visibility(runs);"),
        code("""ev = [e for x in mm for e in x["events"]]
st = np.array([e["stretch"] for e in ev])
rr = np.array([e["relock_recall"] if e["relock_recall"] is not None else np.nan for e in ev], float)
lo, hi = np.nanpercentile(st, [33, 67])
F.say(f"For the third of rewirings whose changed springs were least stretched, the vertex needed a median of "
      f"{viz.fmt(np.nanmedian(rr[st <= lo]))} steps; for the most stretched third, {viz.fmt(np.nanmedian(rr[st >= hi]))}. "
      "Detection takes about 16 steps either way, because its window, not the evidence, sets its pace.")"""),
        md(r"""## How far it stretches

The sweep repeats the run with the wiring changing every 1, 2.5 or 5 s, by a whole new pattern or by one spring at a time, at 1% and 5% noise. It adds a second vertex whose prior expects single-spring flips: it spreads $h$ over the 8 wirings one spring away instead of all 255.""", slide=True),
        code("F.e7_sweep(runs);"),
        code("""sw_ = lambda kind, noise, every, k: float(np.mean([next(cc for cc in x["sweep"] if cc["kind"] == kind and cc["noise"] == noise
                                                          and cc["every"] == every)["correct"][k] for x in mm]))
F.say(f"The vertex wins where its prior matches the world. With whole-pattern rewiring every 2.5 s it is right "
      f"{viz.pct(sw_('pattern', 0.01, 50, 'recall'))} of the time at 1% noise and {viz.pct(sw_('pattern', 0.05, 50, 'recall'))} at 5%, "
      f"against {viz.pct(sw_('pattern', 0.01, 50, 'detect-20'))} and {viz.pct(sw_('pattern', 0.05, 50, 'detect-20'))} for detection. "
      f"When one spring flips at a time under 5% noise it falls behind: {viz.pct(sw_('toggle', 0.05, 50, 'recall'))} against "
      f"{viz.pct(sw_('toggle', 0.05, 50, 'detect-20'))}. A prior that expects single flips recovers part of the gap "
      f"({viz.pct(sw_('toggle', 0.05, 50, 'recall-flip'))}) but collapses on whole-pattern rewiring "
      f"({viz.pct(sw_('pattern', 0.05, 50, 'recall-flip'))} at 5% noise). How the wiring tends to change is itself something "
      "to learn, which is step 3's question: structure that generates the bank's transitions rather than a fixed prior.")"""),
        md("## Inspection report", slide=True),
        code("""mean = lambda xs: float(np.mean(xs))
speed = mean(rl("detect-20")) / mean(rl("recall"))
viz.inspection_report([
    {"characteristic": "Right wiring in force (binding vertex)", "nominal": mean(c("recall")), "tolerance": "≥ 0.90",
     "status": "in" if mean(c("recall")) >= 0.9 else "out"},
    {"characteristic": "Median re-lock (binding vertex), steps", "nominal": mean(rl("recall")), "tolerance": "≤ 5",
     "status": "in" if mean(rl("recall")) <= 5 else "out"},
    {"characteristic": "Re-lock speed-up over the best detector", "nominal": speed, "tolerance": "≥ 5×",
     "status": "in" if speed >= 5 else "out"},
    {"characteristic": "Deviation 1 s ahead ÷ known gates (binding vertex)", "nominal": mean(d("recall")), "actual": 1.0,
     "deviation": mean(d("recall")) - 1, "tolerance": 0.5},
    {"characteristic": "Deviation 1 s ahead ÷ known gates (detection, 20 steps)", "nominal": mean(d("detect-20")), "actual": 1.0,
     "deviation": mean(d("detect-20")) - 1, "tolerance": 0.5},
    {"characteristic": "One spring flips, 5% noise: right wiring, vertex vs detection",
     "nominal": f"{viz.pct(sw_('toggle', 0.05, 50, 'recall'))} vs {viz.pct(sw_('toggle', 0.05, 50, 'detect-20'))}",
     "tolerance": "finding", "status": None},
], title="Everything together, means over seeds (main condition: whole-pattern rewiring every 2.5 s, 1% noise)")"""),
        md("## Log"),
        code("F.log(runs)"),
    ]


NOTEBOOKS = {"e0_calibrate": e0, "e1_decompose": e1, "e2_transfer": e2, "e3_reconfigure": e3,
             "e4_large_swings": e4, "e5_detect": e5, "e6_recall": e6, "e7_capstone": e7}


def build(names=None) -> list[Path]:
    OUT.mkdir(exist_ok=True)
    paths = []
    for name, cells in NOTEBOOKS.items():
        if names and name not in names and name.split("_")[0] not in names:
            continue
        nb = nbf.v4.new_notebook()
        nb.cells = cells()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.metadata["language_info"] = {"name": "python"}
        path = OUT / f"{name}.ipynb"
        nbf.write(nb, path)
        paths.append(path)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--html", action="store_true")
    args = ap.parse_args()
    for path in build(args.names):
        print("wrote", path.relative_to(STUDY))
        if args.execute:
            subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                            "--ExecutePreprocessor.timeout=600", str(path)], check=True, cwd=OUT)
        if args.html:
            (OUT / "html").mkdir(exist_ok=True)
            subprocess.run([sys.executable, "-m", "nbconvert", "--to", "html", "--no-input",
                            "--output-dir", str(OUT / "html"), str(path)], check=True, cwd=OUT)


if __name__ == "__main__":
    main()
