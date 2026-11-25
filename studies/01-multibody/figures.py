"""Figures for study 01 (pendulums on springs). Notebooks call these; the machinery stays in `nsp`.

Vocabulary, as in metrology but inverted: *actual* = the generated world, *nominal* = the model's prediction.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display
from matplotlib.colors import LinearSegmentedColormap

from nsp import config, graphs, viz, world as W
from nsp.results import load_runs

STUDY = Path(__file__).resolve().parent
RESULTS = STUDY / "results"
SCRIPTS = {"e0": "e0_calibrate", "e1": "e1_decompose", "e2": "e2_transfer", "e3": "e3_reconfigure",
           "e4": "e4_large_swings", "e5": "e5_detect", "e6": "e6_recall"}
C = viz.COLORS

DESCRIBE = {
    "world.omega0": "a lone pendulum's frequency (rad/s)", "world.kappa": "spring stiffness ÷ inertia (1/s²)",
    "world.dt": "sampling step (s)", "world.d": "position coordinates per body (1 = planar)",
    "run.seeds": "seeds run (kept low on purpose)", "graph": "how the springs are wired", "n": "number of bodies",
    "amplitude": "starting swing size (rad)", "large_amplitude": "starting swing size, rung 2 (rad)",
    "samples": "steps simulated", "substeps": "Verlet substeps per sample", "reference_substeps": "reference substeps",
    "trajectories": "trajectories per run", "player_steps": "steps shown in the player",
    "train_trajectories": "trajectories per training configuration", "train_steps": "steps per training trajectory",
    "test_trajectories": "trajectories per test", "test_steps": "steps per test rollout (200 = 10 s)",
    "gd_tolerance": "gradient descent stops this close to the closed form",
    "scarcity_graph": "graph for the data-efficiency check", "scarcity_n": "its bodies",
    "scarcity_steps": "training steps tried (one trajectory)", "noise": "measurement noise, fraction of the swing",
    "train_sizes": "training body counts", "train_graphs": "training graphs",
    "train_patterns": "random gate patterns per training graph", "test_n": "test body count",
    "test_graphs": "test graphs", "kappas": "spring stiffnesses swept (1/s²)", "horizons": "rollout horizons (steps)",
    "lambda_grid_max": "largest graph frequency plotted", "seen": "gate patterns used for fitting",
    "steps": "steps per trajectory", "switch_step": "step at which a spring is cut",
    "switch_edge": "spring cut at the switch", "amplitudes": "release angles (rad)", "lifts": "body lifts compared",
    "player_amplitude": "release angle shown in the player (rad)", "windows": "window lengths (steps)",
    "methods": "detection methods", "trials": "trials per condition and seed", "p_on": "chance each spring is live",
    "bank_size": "stored structures", "hazard": "chance per step that the context changes",
    "static_trials": "trials per window", "switch_trials": "switch trials", "total_steps": "steps per switch trial",
    "sliding_window": "window of the sliding detector (steps)", "novelty_windows": "novelty test windows (steps)",
    "novelty_trials": "novelty trials per window", "false_alarm": "false-alarm rate the novelty threshold allows",
}


# ------------------------------------------------------------------------------------------------------- plumbing

def setup():
    viz.style()
    plt.rcParams["figure.dpi"] = 100


def cfg(exp: str) -> dict:
    return config.load(STUDY / "configs" / f"{exp}.toml")


def ensure_results(exp: str, rerun: bool = False) -> list[dict]:
    """Run the experiment script if its results are missing (or rerun=True), then load every seed."""
    seeds = cfg(exp)["run"]["seeds"]
    have = all((RESULTS / exp / f"seed-{s}" / "metrics.json").exists() for s in seeds)
    if rerun or not have:
        r = subprocess.run([sys.executable, str(STUDY / "experiments" / f"{SCRIPTS[exp]}.py")], capture_output=True,
                           text=True, cwd=STUDY)
        print(r.stdout.strip() or r.stderr.strip())
    runs = load_runs(RESULTS, exp)
    print(f"{exp}: {len(runs)} seeds loaded (code {runs[0]['meta']['code']}, run {runs[0]['meta']['time_utc']})")
    return runs


def ledger(exp: str):
    c = cfg(exp)
    rows = [(f"world.{k}", v, DESCRIBE.get(f"world.{k}", "")) for k, v in c["world"].items()]
    rows.append(("run.seeds", c["run"]["seeds"], DESCRIBE["run.seeds"]))
    rows += [(k, v, DESCRIBE.get(k, "")) for k, v in c[exp].items()]
    return viz.ledger(rows)


def log(runs: list[dict]):
    lines = ["| Seed | Code version | Run at (UTC) | Python | NumPy | SciPy |", "| --- | --- | --- | --- | --- | --- |"]
    for r in runs:
        m = r["meta"]
        lines.append(f"| {r['dir'].name.split('-')[1]} | `{m['code']}` | {m['time_utc']} | {m['python']} | "
                     f"{m['numpy']} | {m['scipy']} |")
    return Markdown("\n".join(lines))


def spread(values) -> str:
    """'mean (min to max)' across seeds."""
    v = np.asarray(values, float)
    if len(v) == 1:
        return viz.fmt(v[0])
    return f"{viz.fmt(v.mean())} ({viz.fmt(v.min())} to {viz.fmt(v.max())})"


def say(text: str):
    display(viz.finding(text))


# -------------------------------------------------------------------------------------------------- schematics

def schematics(items, title: str | None = None, cols: int | None = None):
    """Static pictures of worlds: items are (label, n, edges, gates or None)."""
    cols = cols or len(items)
    rows = int(np.ceil(len(items) / cols))
    nmax = max(it[1] for it in items)
    fig, axes = plt.subplots(rows, cols, figsize=(11, (1.9 if nmax > 5 else 2.1) * rows + 0.4), squeeze=False)
    for ax, (label, n, edges, gates) in zip(axes.flat, items):
        viz.draw_row(ax, np.zeros(n), None, edges, gates, limits=((-0.7, nmax - 1 + 0.7), (-1.25, 0.6)))
        ax.set_title(label, fontsize=11, loc="left")
    for ax in list(axes.flat)[len(items):]:
        ax.axis("off")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    return fig


# ------------------------------------------------------------------------------------------------------------- E0

def e0_modes(world: W.World, run: dict):
    U = run["arrays"]["modes"]
    lam = np.asarray(run["metrics"]["lambdas"])
    om = np.asarray(run["metrics"]["mode_frequencies"])
    names = ["all together", "ends opposite, middle still", "middle against both ends"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.9))
    edges = graphs.path(3)
    for k, ax in enumerate(axes):
        u = U[:, k] * np.sign(U[np.argmax(np.abs(U[:, k])), k])
        viz.draw_row(ax, 0.32 * u / np.abs(u).max(), None, edges, limits=((-0.8, 2.8), (-1.25, 0.35)))
        ax.set_title(f"mode {k + 1}: {names[k] if len(lam) == 3 else ''}\nλ = {lam[k]:.0f},  "
                     f"ω = √(ω₀² + κλ) = {om[k]:.3f} rad/s", fontsize=10.5, loc="left")
    fig.suptitle("The path of 3 swings in three modes; stiffer modes (larger λ) swing faster", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    return fig


def e0_error_growth(runs: list[dict], world: W.World):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    t = np.arange(len(runs[0]["arrays"]["curve"])) * world.dt
    for i, r in enumerate(runs):
        lab = (lambda s: s) if i == 0 else (lambda s: None)
        a1.semilogy(t[1:], r["arrays"]["curve"][1:], color=C["nominal"], lw=1, alpha=0.8,
                    label=lab("small swings: Verlet vs exact step"))
        a1.semilogy(t[1:], r["arrays"]["curve_large"][1:], color=C["out"], lw=1, alpha=0.8,
                    label=lab("2 rad: Verlet vs 4× substeps"))
        for key, col, text in (("energy_small", C["nominal"], "small swings"), ("energy_large", C["out"], "2 rad, with sin θ")):
            worst = np.maximum.accumulate(np.abs(r["arrays"][key][:, 1:]).max(0))
            a2.semilogy(t[1:], worst, color=col, lw=1.4, alpha=0.85, label=lab(text))
    a1.axhline(1e-4, color=C["in"], ls="--", lw=1)
    a1.text(t[-1], 1.2e-4, "tolerance at 10 s", ha="right", va="bottom", fontsize=9, color=C["in"])
    a1.axvline(10, color=C["muted"], lw=0.8, ls=":")
    a1.set(xlabel="time (s)", ylabel="relative deviation", title="Simulator deviation grows slowly")
    a2.axhline(1e-4, color=C["in"], ls="--", lw=1)
    a2.text(t[-1], 1.2e-4, "tolerance", ha="right", va="bottom", fontsize=9, color=C["in"])
    a2.set(xlabel="time (s)", ylabel="worst |H − H(0)| / H(0) so far",
           title="Energy error never exceeds a few parts per million")
    a1.legend(loc="lower right")
    a2.legend(loc="lower right")
    return fig


# ------------------------------------------------------------------------------------------------------------- E1

def e1_frequencies(run: dict, world: W.World):
    m = run["metrics"]["modes"]
    lam_grid = np.linspace(0, 3.5, 100)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    a1.plot(lam_grid, np.sqrt(world.omega0**2 + world.kappa * lam_grid), color=C["actual"], lw=1.4,
            label="actual: √(ω₀² + κλ)")
    a1.plot(m["lambda"], m["omega_nominal"], "o", ms=9, mfc="none", mec=C["nominal"], mew=2,
            label="nominal: arg eig(K − λM) / Δt")
    a1.set(xlabel="graph frequency λ", ylabel="ω (rad/s)", title="Each graph mode swings at its own frequency")
    a1.legend(loc="upper left")
    a2.bar(np.arange(len(m["lambda"])), np.abs(m["frequency_error"]), color=C["nominal"], width=0.55)
    a2.set_xticks(np.arange(len(m["lambda"])), [f"λ = {v:.0f}" for v in m["lambda"]])
    a2.set_yscale("log")
    a2.set(ylabel="|frequency deviation| / ω", title="Nominal frequencies are off by parts in 10⁵")
    return fig


def e1_gradient_descent(runs: list[dict]):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    for i, r in enumerate(runs):
        a = r["arrays"]
        ax.loglog(np.maximum(a["gd_iters"], 1), a["gd_distance"], color=viz.SERIES[i], lw=1.6,
                  label=f"seed {i}: {int(r['metrics']['gd_iterations'])} steps")
    ax.axhline(1e-8, color=C["in"], ls="--", lw=1)
    ax.set(xlabel="gradient-descent step", ylabel="‖W − W*‖ / ‖W*‖",
           title="Gradient descent lands on the closed-form answer (local delta rule, step 1/λmax)")
    ax.legend()
    return fig


def e1_scarcity(runs: list[dict]):
    met = runs[0]["metrics"]
    steps, noise = np.array(met["scarcity_steps"]), met["noise"]
    fig, axes = plt.subplots(1, len(noise), figsize=(11, 3.5), sharey=True)
    for j, (ax, sig) in enumerate(zip(axes, noise)):
        for key, col, lab in (("nsp_roll", C["nominal"], "nsp: 8 shared numbers"),
                              ("global_roll", C["out"], "one global map: 256 numbers")):
            vals = np.array([r["arrays"][key][:, j] for r in runs])
            ax.loglog(steps, np.median(vals, 0), "o-", color=col, label=lab, ms=4)
            ax.fill_between(steps, vals.min(0), vals.max(0), color=col, alpha=0.15, lw=0)
        ax.set_ylim(5e-4, 2e3)
        ax.set_title(f"noise {viz.pct(sig) if sig else 'none'}", fontsize=11)
        ax.set_xlabel("training steps (one trajectory)")
    axes[0].set_ylabel("rollout deviation at 10 s")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Ring of 8, fitted from one trajectory: the global map never pins down the ring's repeated modes",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    return fig


# ------------------------------------------------------------------------------------------------------------- E2

def e2_graph_frequencies(run: dict):
    res = run["metrics"]["by_kappa"]["1"]
    lam_train = np.array(res["lambda_train"])
    fig, ax = plt.subplots(figsize=(11, 2.9))
    ax.axvspan(0, lam_train.max(), color=C["train"], alpha=0.12, lw=0)
    ax.text(lam_train.max(), 4.55, " training range", color=C["train"], va="center", fontsize=10)
    ax.scatter(lam_train, np.full(len(lam_train), 4), color=C["train"], s=26, label="training graphs (3 and 5 bodies)")
    names = list(res["graphs"])
    for i, nm in enumerate(names):
        lam = np.array(res["graphs"][nm]["lambdas"])
        out = lam > lam_train.max() + 1e-9
        ax.scatter(lam[~out], np.full((~out).sum(), 3 - i), color=C["test"], s=40, edgecolor=C["actual"], lw=0.6)
        ax.scatter(lam[out], np.full(out.sum(), 3 - i), color=C["out"], s=55, marker="D", edgecolor=C["actual"], lw=0.6)
    ax.set_yticks([4, 3, 2, 1, 0], ["training", *[f"{nm} of 8" for nm in names]])
    ax.set_xlabel("graph frequency λ (eigenvalues of the Laplacian)")
    ax.set_xlim(-0.2, 8.6)
    ax.set_ylim(-0.6, 5)
    ax.set_title("Stars and complete graphs reach λ = 8, far outside what training saw (diamonds)")
    ax.grid(axis="y", visible=False)
    return fig


def e2_frequency_error(runs: list[dict]):
    grid = runs[0]["arrays"]["lambda_grid"]
    lam_max = runs[0]["metrics"]["by_kappa"]["1"]["lambda_train_max"]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.axvspan(0, lam_max, color=C["train"], alpha=0.12, lw=0)
    ax.text(lam_max / 2, 0.97, "training range", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=9.5, color=C["train"])
    for i, k in enumerate(runs[0]["metrics"]["kappas"]):
        vals = np.array([np.abs(r["arrays"][f"freq_err_k{k:g}"]) for r in runs])
        ax.semilogy(grid, np.median(vals, 0), color=viz.SERIES[i], label=f"κ = {k:g}")
    for x, lab in ((4, "ring and path of 8 reach here"), (8, "star and complete graph of 8")):
        ax.axvline(x, color=C["muted"], ls=":", lw=1)
        ax.text(x, 0.03, f" {lab}", transform=ax.get_xaxis_transform(), fontsize=9, color=C["muted"], va="bottom",
                ha="left" if x < 8 else "right")
    ax.set(xlabel="graph frequency λ", ylabel="|frequency deviation| / ω",
           title="The fitted model's frequencies drift away outside the training range")
    ax.legend(title="spring\nstiffness", loc="center left", bbox_to_anchor=(1.0, 0.5))
    return fig


def e2_ratios(runs: list[dict]):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullFormatter
    kappas = runs[0]["metrics"]["kappas"]
    names = list(runs[0]["metrics"]["by_kappa"]["1"]["graphs"])
    fig, ax = plt.subplots(figsize=(11, 3.8))
    off = (np.arange(len(kappas)) - (len(kappas) - 1) / 2) * 0.24
    minor_pos, minor_lab = [], []
    for j, nm in enumerate(names):
        undefined = False
        for i, k in enumerate(kappas):
            x = j + off[i]
            vals = []
            for r in runs:
                v = r["metrics"]["by_kappa"][f"{k:g}"]["graphs"][nm]["ratio"]
                vals.append(np.inf if isinstance(v, str) else v)
            vals = np.array(vals, float)
            minor_pos.append(x)
            minor_lab.append(f"κ={k:g}")
            if not np.isfinite(vals).all():
                undefined = True
                continue
            med = float(np.median(vals))
            col = C[viz.status(med - 1, 0.5)]
            ax.vlines(x, vals.min(), vals.max(), color=col, lw=3, alpha=0.35)
            ax.plot(x, med, "o", ms=9, color=col, mec="white", mew=1.2, zorder=3)
            ax.text(x + 0.035, med, f"{med:.1f}×", va="center", fontsize=9, color=C["actual"])
        if undefined:
            ax.text(j, 1.25, "a model fitted on this graph\nis exact, so the ratio\nis undefined", ha="center",
                    va="center", fontsize=9, color=C["muted"])
    ax.axhline(1, color=C["muted"], lw=0.8)
    ax.axhline(1.5, color=C["in"], ls="--", lw=1)
    ax.text(-0.45, 1.53, "1.5× limit", va="bottom", ha="left", color=C["in"], fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(0.9, 10)
    ax.set_yticks([1, 1.5, 2, 4, 8], ["1×", "1.5×", "2×", "4×", "8×"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(range(len(names)), [f"{nm} of 8" for nm in names])
    ax.xaxis.remove_overlapping_locs = False  # the middle κ sits on the major tick; keep its label
    ax.set_xticks(minor_pos, minor_lab, minor=True)
    ax.tick_params(axis="x", which="minor", labelsize=8.5, labelcolor=C["muted"], length=0)
    ax.tick_params(axis="x", which="major", pad=17, length=0, labelsize=11)
    ax.set_xlim(-0.55, len(names) - 0.45)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("deviation at 10 s ÷ own fit's")
    ax.set_title("Transfer holds where training saw the graph frequencies, and fails where it did not")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C["in"], ms=8, label="within 1.5×"),
                       Line2D([], [], marker="o", ls="", color=C["out"], ms=8, label="beyond 1.5×"),
                       Line2D([], [], color=C["muted"], lw=3, alpha=0.5, label="range over seeds")],
              loc="upper left", ncol=3)
    return fig


def e2_rollouts(run: dict, kappa: float, world: W.World):
    names = list(run["metrics"]["by_kappa"]["1"]["graphs"])
    fig, ax = plt.subplots(figsize=(11, 3.2))
    t = np.arange(len(run["arrays"][f"curve_k{kappa:g}_{names[0]}"])) * world.dt
    shown = np.concatenate([run["arrays"][f"curve_k{kappa:g}_{nm}"][1:] for nm in names])
    bottom = 10 ** np.floor(np.log10(shown[shown > 0].min()) - 0.5)
    for i, nm in enumerate(names):
        own = run["arrays"][f"curve_own_k{kappa:g}_{nm}"][1:]
        below = own.max() < bottom
        ax.semilogy(t[1:], run["arrays"][f"curve_k{kappa:g}_{nm}"][1:], color=viz.SERIES[i],
                    label=f"{nm} of 8" + (" (own fit exact, off scale)" if below else ""))
        if not below:
            ax.semilogy(t[1:], own, color=viz.SERIES[i], ls=":", lw=1.2)
    ax.set_ylim(bottom, None)
    ax.set(xlabel="time (s)", ylabel="relative deviation",
           title=f"κ = {kappa:g}: solid = transferred model, dotted = a model fitted on that 8-body graph")
    ax.legend(ncol=4, loc="lower right")
    return fig


# ------------------------------------------------------------------------------------------------------------- E3

def e3_patterns(runs: list[dict]):
    kappas = runs[0]["metrics"]["kappas"]
    pats = runs[0]["arrays"]["patterns"]
    fig, axes = plt.subplots(1, len(kappas), figsize=(11, 3.4), sharey=False)
    for ax, k in zip(np.atleast_1d(axes), kappas):
        for r_i, r in enumerate(runs):
            seen = r["arrays"][f"seen_k{k:g}"].astype(bool)
            roll = r["arrays"][f"roll_k{k:g}"]
            jit = (r_i - 1) * 0.08
            ax.scatter(np.flatnonzero(seen) * 0 + 0 + jit + np.random.default_rng(r_i).uniform(-0.15, 0.15, seen.sum()),
                       roll[seen], s=14, color=C["train"], alpha=0.7, label="fitted on" if r_i == 0 else None)
            ax.scatter(1 + jit + np.random.default_rng(10 + r_i).uniform(-0.15, 0.15, (~seen).sum()), roll[~seen], s=14,
                       color=C["test"], edgecolor=C["actual"], lw=0.4, alpha=0.9,
                       label="held out" if r_i == 0 else None)
        ax.set_xticks([0, 1], ["16 patterns fitted on", "16 held out"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_yscale("log")
        ax.set_title(f"κ = {k:g}", fontsize=11)
        ax.set_ylabel("deviation at 10 s")
    np.atleast_1d(axes)[0].legend(loc="upper left")
    fig.suptitle(f"Unseen gate patterns of a ring of 5 ({len(pats)} in all) are predicted as well as seen ones",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    return fig


def e3_switch(run: dict, kappa: float, world: W.World, switch_step: int):
    a = run["arrays"]
    curve, H = a[f"switch_curve_k{kappa:g}"], a[f"energy_k{kappa:g}"]
    t = np.arange(len(curve)) * world.dt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4))
    a1.plot(t[1:], curve[1:], color=C["actual"])
    a1.axvline(switch_step * world.dt, color=C["out"], ls=":", lw=1.4)
    a1.text(switch_step * world.dt, 0.97, " spring cut", transform=a1.get_xaxis_transform(), color=C["out"],
            va="top", fontsize=9)
    a1.set(xlabel="time (s)", ylabel="relative deviation", title="No jump at the cut beyond the usual swing")
    a2.plot(t, H / H[0], color=C["actual"])
    a2.axvline(switch_step * world.dt, color=C["out"], ls=":", lw=1.4)
    a2.set(xlabel="time (s)", ylabel="H(t) / H(0)", title="The world loses the energy the spring was holding")
    return fig


# ------------------------------------------------------------------------------------------------------------- E4

def e4_frequency(world: W.World, amplitudes):
    a = np.linspace(0.01, 2.6, 200)
    fig, ax = plt.subplots(figsize=(11, 3.1))
    ax.plot(a, [W.swing_frequency(world, x) / world.omega0 for x in a], color=C["actual"],
            label="actual: π / (2 K(sin²(a/2)))")
    ax.axhline(1, color=C["nominal"], ls="--", lw=1.4, label="any linear model: fixed frequency ω₀")
    for x in amplitudes:
        y = W.swing_frequency(world, x) / world.omega0
        ax.plot(x, y, "o", color=C["out"], ms=7)
        ax.text(x, y - 0.035, f"{x:g} rad\n{viz.pct(1 - y)} slower", ha="center", va="top", fontsize=9)
    ax.set(xlabel="release angle a (rad)", ylabel="ω(a) / ω₀", ylim=(0.62, 1.05),
           title="A pendulum slows down as it swings wider, which no linear map can do")
    ax.legend(loc="lower left")
    return fig


from nsp.lifts import LABELS as MATH_LIFT  # noqa: E402  (LaTeX labels shared with the reports)


def e4_errors(runs: list[dict]):
    amps = runs[0]["metrics"]["amplitudes"]
    lifts = runs[0]["metrics"]["lifts"]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for i, lift in enumerate(lifts):
        vals = np.array([[r["metrics"]["by_amplitude"][f"{a:g}"]["lifts"][lift]["rollout"]["200"] for a in amps]
                         for r in runs])
        ax.semilogy(amps, np.median(vals, 0), "o-", color=viz.SERIES[i], label=MATH_LIFT[lift], ms=6)
        ax.fill_between(amps, vals.min(0), vals.max(0), color=viz.SERIES[i], alpha=0.15, lw=0)
    ax.axhline(0.01, color=C["in"], ls="--", lw=1)
    ax.text(amps[-1], 0.0115, "1% at 10 s", color=C["in"], fontsize=9, va="bottom", ha="right")
    ax.set(xlabel="release angle (rad)", ylabel="deviation at 10 s",
           title="Each added term cuts the deviation about 10× at large swings")
    ax.legend(title="body lift", loc="upper left")
    return fig


def e4_phase(run: dict):
    a = run["arrays"]
    act = a["phase_act"]
    lifts = run["metrics"]["lifts"]
    fig, axes = plt.subplots(1, len(lifts), figsize=(11, 3.7), sharex=True, sharey=True)
    for ax, lift in zip(axes, lifts):
        nom = a[f"phase_nom_{lift}"]
        ax.plot(act[:, 0], act[:, 1], color=C["actual"], lw=2.2, label="actual (generated)")
        ax.plot(nom[:, 0], nom[:, 1], color=C["nominal"], lw=1.3, ls="--", label="nominal (predicted)")
        ax.set_title(MATH_LIFT[lift], fontsize=11)
        ax.set_xlabel(r"$\theta$ (rad)")
    axes[0].set_ylabel(r"$\dot\theta$ (rad/s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=10)
    fig.suptitle("Body 1 in phase space at 2 rad, 10 s: only the full lift stays on the actual orbit",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    return fig


# ------------------------------------------------------------------------------------------------------------- E5

def e5_blind_spot(run: dict):
    a = run["arrays"]
    fig, ax = plt.subplots(figsize=(11, 3.1))
    s1, s3 = np.asarray(a["blind_sv_1"]), np.asarray(a["blind_sv_3"])
    x = np.arange(1, len(s1) + 1)
    sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    ax.semilogy(x, s3, "s-", color=C["nominal"], ms=7, label="three trajectories")
    ax.semilogy(x, s1, "o-", color=C["out"], ms=7, label="one trajectory")
    ax.annotate(f"σ{str(len(s1)).translate(sub)} = {viz.fmt(s1[-1], 2)}: a mix of springs\nthat changes no prediction",
                xy=(x[-1], s1[-1]), xytext=(x[-1] - 0.3, 3e-10), ha="right", fontsize=9.5, color=C["out"],
                arrowprops=dict(arrowstyle="->", color=C["out"], lw=1.2))
    ax.set_ylim(min(s1.min(), s3.min()) / 30, 3)
    ax.set_xticks(x, [f"σ{str(i).translate(sub)}" for i in x])
    ax.set(ylabel="singular value ÷ largest", xlabel="directions in the space of the 10 candidate gates",
           title="One live spring, three free pendulums: one trajectory leaves a direction the data cannot see")
    ax.legend(loc="lower left")
    return fig


def _pair_matrix(pairs, values, n):
    Mx = np.full((n, n), np.nan)
    for (u, v), g in zip(pairs, values):
        Mx[u, v] = Mx[v, u] = g
    return Mx


def e5_matrices(run: dict, tol: float = 0.1):
    a, met = run["arrays"], run["metrics"]
    pairs, n = [tuple(p) for p in met["pairs"]], 5
    cmap = LinearSegmentedColormap.from_list("g", ["#FFFFFF", C["nominal"]])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    for ax, (key, title) in zip(axes, (("truth", "actual springs"), ("g_lstsq", "least squares"),
                                       ("g_nnls", "non-negative least squares"))):
        Mx = _pair_matrix(pairs, a[key], n)
        ax.imshow(np.clip(np.nan_to_num(Mx, nan=0), 0, 1), cmap=cmap, vmin=0, vmax=1)
        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                val = Mx[u, v]
                actual = a["truth"][pairs.index(tuple(sorted((u, v))))]
                st = viz.status(val - actual, tol)
                box = None if st == "in" else dict(boxstyle="round,pad=0.18", fc="white", ec=C[st], lw=0.8)
                ax.text(v, u, f"{val:.2f}".replace("-", "−"), ha="center", va="center", fontsize=8.5, bbox=box,
                        color="white" if val > 0.6 and st == "in" else (C["actual"] if st == "in" else C[st]),
                        fontweight="normal" if st == "in" else "bold")
        ax.set_xticks(range(n), [f"{i + 1}" for i in range(n)])
        ax.set_yticks(range(n), [f"{i + 1}" for i in range(n)])
        ax.grid(False)
        ax.tick_params(length=0)
        ax.set_title(title, fontsize=11)
    fig.suptitle("Estimated gates for all 10 pairs (50 steps, 1% noise, one trajectory)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.supxlabel(f"Deviation from the actual gate: plain = in tolerance (±{tol:g}); boxed amber = near (±{2 * tol:g}); "
                  "boxed vermillion = out.", fontsize=10, color=C["muted"])
    return fig


def e5_exact(runs: list[dict], noise: float = 0.01, dt: float = 0.05):
    rows = [row for r in runs for row in r["metrics"]["rows"]]
    kappas = sorted({row["kappa"] for row in rows})
    windows = sorted({row["window"] for row in rows})
    combos = [("lstsq", 1), ("nnls", 1), ("lstsq", 3), ("nnls", 3)]
    names = ["least squares, 1 trajectory", "non-negative, 1 trajectory", "least squares, 3 trajectories",
             "non-negative, 3 trajectories"]
    cmap = LinearSegmentedColormap.from_list("q", ["#FFFFFF", C["nominal"]])
    fig, axes = plt.subplots(1, len(kappas), figsize=(11, 3.3), sharey=True)
    for ax, k in zip(np.atleast_1d(axes), kappas):
        Z = np.array([[np.mean([row["exact"] for row in rows if row["kappa"] == k and row["window"] == wn
                                and row["noise"] == noise and row["method"] == mt and row["trajectories"] == ntr])
                       for wn in windows] for mt, ntr in combos])
        ax.imshow(Z, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                ax.text(j, i, viz.pct(Z[i, j], 3) if Z[i, j] < 1 else "100%", ha="center", va="center", fontsize=10.5,
                        color="white" if Z[i, j] > 0.6 else C["actual"])
        ax.set_xticks(range(len(windows)), [f"{w} steps ({round(w * dt, 2):g} s)" for w in windows])
        ax.set_yticks(range(len(combos)), names)
        ax.set_title(f"κ = {k:g}", fontsize=11)
        ax.grid(False)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle(f"Trials with all 10 pairs right, at {viz.pct(noise)} noise: non-negativity or a second trajectory "
                 "removes the blind spot", x=0.01, ha="left", fontsize=13, fontweight="bold")
    return fig


# ------------------------------------------------------------------------------------------------------------- E6

def e6_recall_vs_detect(runs: list[dict]):
    win = np.array(runs[0]["metrics"]["windows"])
    fig, ax = plt.subplots(figsize=(11, 3.5))
    for key, col, lab in (("recall_accuracy", C["nominal"], "recall from the bank of 16"),
                          ("detect_accuracy", C["out"], "detect all 10 pairs from scratch (E5)")):
        vals = np.array([r["metrics"][key] for r in runs])
        ax.semilogx(win, vals.mean(0), "o-", color=col, label=lab)
        ax.fill_between(win, vals.min(0), vals.max(0), color=col, alpha=0.15, lw=0)
    ax.set_xticks(win, [str(x) for x in win])
    ax.set(xlabel="window (steps of 0.05 s)", ylabel="right structure (fraction)", ylim=(0, 1.04),
           title="Recall needs a handful of steps; detection from scratch needs about fifty")
    ax.legend(loc="lower right")
    return fig


def e6_belief(run: dict, world: W.World, switch_step: int):
    a, met = run["arrays"], run["metrics"]
    pi = a["pi"]
    ca, cb = met["example_contexts"]
    C_ = len(pi[0])
    fig, ax = plt.subplots(figsize=(11, 3.6))
    cmap = LinearSegmentedColormap.from_list("b", ["#FFFFFF", C["nominal"]])
    ax.imshow(pi.T, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
              extent=(0, len(pi) * world.dt, C_ - 0.5, -0.5))
    ax.axvline(switch_step * world.dt, color=C["out"], ls=":", lw=1.5)
    ax.text(switch_step * world.dt, 0.02, " contexts switch", transform=ax.get_xaxis_transform(), color=C["out"],
            va="bottom", fontsize=9)
    labels = [str(c) for c in range(C_)]
    labels[ca], labels[cb] = f"{ca}: before", f"{cb}: after"
    ax.set_yticks(range(C_), labels, fontsize=8)
    for c in (ca, cb):
        ax.get_yticklabels()[c].set_fontweight("bold")
        ax.get_yticklabels()[c].set_color(C["actual"])
    ax.grid(False)
    ax.set(xlabel="time (s)", ylabel="stored structure",
           title="The binding vertex's belief over the 16 stored structures (darker = more belief)")
    return fig


def e6_relock(runs: list[dict]):
    f = np.array([x for r in runs for x in r["metrics"]["relock_filter"] if x is not None], float)
    d = np.array([x for r in runs for x in r["metrics"]["relock_detect"] if x is not None], float)
    missed = sum(x is None for r in runs for x in r["metrics"]["relock_detect"])
    fig, ax = plt.subplots(figsize=(11, 3.0))
    rng = np.random.default_rng(0)
    for vals, y, col, lab in ((f, 1, C["nominal"], "binding vertex (recall)"), (d, 0, C["out"], "sliding 50-step detection")):
        ax.scatter(vals, y + rng.uniform(-0.22, 0.22, len(vals)), color=col, s=22, alpha=0.85, label=lab)
        med = float(np.median(vals))
        ax.plot([med, med], [y - 0.32, y + 0.32], color=C["actual"], lw=2)
        ax.text(med, y + 0.36, f"median {med:g}", ha="center", va="bottom", fontsize=9)
    ticks = [1, 2, 5, 10, 20, 50]
    ax.set_xscale("log")
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_yticks([1, 0], ["recall", "detection"])
    ax.set_ylim(-0.5, 1.7)
    ax.set(xlabel="steps after the switch until the right structure is in force" +
           (f" ({missed} detection runs never locked)" if missed else ""),
           title="After a switch, recall re-locks in a few steps; detection waits for its window to fill")
    ax.grid(axis="y", visible=False)
    return fig


def e6_novelty(runs: list[dict], window: int = 20):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, (key, title) in zip(axes, (("abs", "absolute fit of the best stored structure"),
                                       ("glr", "likelihood ratio: best stored vs detected"))):
        shift = 0 if key == "abs" else 1.0
        inb = np.clip(np.concatenate([r["arrays"][f"{key}_in_{window}"] for r in runs]) + shift, 1e-2, None)
        outb = np.clip(np.concatenate([r["arrays"][f"{key}_out_{window}"] for r in runs]) + shift, 1e-2, None)
        bins = np.geomspace(min(inb.min(), outb.min()), max(inb.max(), outb.max()), 40)
        ax.hist(inb, bins=bins, color=C["nominal"], alpha=0.6, label="stored structure")
        ax.hist(outb, bins=bins, color=C["out"], alpha=0.6, label="new structure")
        thr = np.quantile(inb, 0.99)
        ax.axvline(thr, color=C["actual"], ls="--", lw=1.2)
        ax.text(thr, ax.get_ylim()[1] * 0.92, " 1% false alarms", fontsize=9)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("statistic" + (" + 1" if shift else "") + " (log scale)")
    axes[0].set_ylabel("trials")
    axes[1].legend(loc="upper right")
    fig.suptitle(f"Is this structure new? Window of {window} steps", x=0.01, ha="left", fontsize=13,
                 fontweight="bold")
    return fig
