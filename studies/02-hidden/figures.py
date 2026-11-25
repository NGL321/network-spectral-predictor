"""Figures for study 02 (hidden information with a clamped sense). Notebooks call these; the machinery stays in `nsp`.

Vocabulary, as in metrology but inverted: *actual* = the generated world, *nominal* = the model's prediction.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from nsp import study, viz
from nsp.results import load_runs  # noqa: F401  (handy in notebooks)

STUDY = Path(__file__).resolve().parent
RESULTS = STUDY / "results"
C = viz.COLORS
DESCRIBE = {
    "world.omega0": "a lone pendulum's frequency (rad/s)", "world.kappa": "spring stiffness ÷ inertia (1/s²)",
    "world.dt": "sampling step (s)", "world.d": "position coordinates per body (1 = planar)",
    "run.seeds": "seeds run (kept low on purpose)", "graph": "how the springs are wired", "n": "number of bodies",
    "amplitude": "starting swing size (rad)", "p_on": "chance each spring is live in a run's wiring (known here)",
    "steps": "steps per run", "trials": "runs per seed and condition",
    "noise": "sense noise on angles, fraction of the swing", "noise_levels": "sense noise levels swept",
    "horizon": "how far ahead the nominal world predicts (steps)",
    "q_levels": "rate noise the clamp allows for, (rad/s)² per step", "lambda_grid_max": "largest graph frequency plotted",
    "train_sizes": "body counts step 1's maps were learned on", "train_graphs": "graphs step 1's maps were learned on",
    "train_patterns": "random gate patterns per training graph", "train_trajectories": "trajectories per training graph",
    "train_steps": "steps per training trajectory",
}
S = study.Study(STUDY, {"h1": "h1_clamped"}, DESCRIBE)
cfg, ensure_results, ledger = S.cfg, S.ensure_results, S.ledger
log, spread, say = study.log, study.spread, study.say

STYLE = {"full": (C["muted"], ":", "angles and rates measured (step 1)"),
         "delay": (C["out"], "-", "delay line"),
         "clamped": (C["near"], "--", "clamped sense, fixed precision"),
         "precision": (C["nominal"], "-", "clamped sense, growing precision"),
         "kalman": (C["actual"], "-", "Kalman filter (centralised)")}


def setup():
    viz.style()
    plt.rcParams["figure.dpi"] = 100


def schematic():
    """One body's step: predict with K and the spring messages, then clamp to the sense."""
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.55, 3.8)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#F6F8FA", ec="#8C959F"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12", fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.2)

    def arrow(p, q, text="", dy=0.1):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, color="#57606A", lw=1.3,
                                     shrinkA=2, shrinkB=2))
        if text:
            ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, text, ha="center", va="bottom", fontsize=9, color="#57606A")

    box(0.2, 2.3, 2.2, 1.2, "last estimate\n$\\hat x_v(t-1)$, and the\nneighbours' $\\hat x_u(t-1)$")
    box(3.2, 2.3, 2.6, 1.2, "predict (step 1's maps)\n$\\hat x^-_v=K\\hat x_v$\n$-M\\sum_u g_{uv}(\\hat x_v-\\hat x_u)$")
    box(6.6, 2.3, 2.0, 1.2, "prediction\n$\\hat x^-_v(t)$:\nangle and rate")
    box(6.6, 0.25, 2.0, 1.2, "sense, clamped\n$y_v(t)=\\theta_v(t)$\n$+$ noise (angle only)", fc="#FBE3D6", ec=C["out"])
    box(9.1, 1.25, 1.75, 1.3, "relax to\n$\\hat x_v(t)=\\hat x^-_v$\n$+k\\,(y_v-\\hat\\theta^-_v)$", fc="#DDEBF7",
        ec=C["nominal"])
    arrow((2.4, 2.9), (3.2, 2.9))
    arrow((5.8, 2.9), (6.6, 2.9))
    arrow((8.6, 2.9), (9.5, 2.55))
    arrow((8.6, 0.85), (9.5, 1.25))
    ax.plot([9.975, 9.975, 1.3], [1.25, -0.3, -0.3], color=C["nominal"], lw=1.2)
    ax.add_patch(FancyArrowPatch((1.3, -0.3), (1.3, 2.3), arrowstyle="-|>", mutation_scale=13, color=C["nominal"],
                                 lw=1.2, shrinkA=0, shrinkB=2))
    ax.text(4.0, -0.22, "next step: this estimate becomes the last one", color=C["nominal"], fontsize=9.5)
    ax.set_title("One body, one step: the rate is never measured; it is predicted, then corrected through the angle",
                 fontsize=11.5, loc="left")
    return fig


def rate_trace(run: dict, world, body: int = 0, seconds: float = 10.0):
    a = run["arrays"]
    steps = int(seconds / world.dt) + 1
    t = np.arange(steps) * world.dt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    for ax, (lo, hi) in zip(axes, ((0, steps), (0, int(2.0 / world.dt) + 1))):
        ax.plot(t[lo:hi], a["rate_act"][lo:hi, body], color=C["actual"], lw=2.6, alpha=0.35, label="actual rate")
        for m in ("delay", "clamped", "precision"):
            col, ls, lab = STYLE[m]
            ax.plot(t[lo:hi], a[f"rate_est_{m}"][lo:hi, body], color=col, ls=ls, lw=1.2, label=lab)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel(r"$\dot\theta$ of body 1 (rad/s)")
    axes[0].set_title("The hidden rate, inferred from angles alone")
    axes[1].set_title("First 2 s, from a cold start")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9)
    return fig


def convergence(runs: list[dict], world):
    T = len(runs[0]["arrays"]["err_rate_full"])
    t = np.arange(T) * world.dt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.7), gridspec_kw={"width_ratios": [1.6, 1]})
    for m in ("full", "delay", "clamped", "precision", "kalman"):
        col, ls, lab = STYLE[m]
        v = np.mean([r["arrays"][f"err_rate_{m}"] for r in runs], 0)
        a1.semilogy(t[1:], v[1:], color=col, ls=ls, lw=1.5, label=lab)
    a1.set(xlabel="time since a cold start (s)", ylabel="rate error ÷ rate scale",
           title="How fast and how well each method finds the hidden rate")
    a1.legend(loc="center right", fontsize=8.5, bbox_to_anchor=(1.0, 0.62))
    names = ("delay", "full", "clamped", "precision", "kalman")
    vals = [np.mean([r["metrics"]["steady"][m]["rate"] for r in runs]) for m in names]
    settle = [np.mean([r["metrics"]["steady"][m]["settle"] for r in runs]) for m in names]
    y = np.arange(len(names))
    a2.barh(y, vals, color=[STYLE[m][0] for m in names], alpha=0.85, height=0.6)
    for yi, v, s_, m in zip(y, vals, settle, names):
        a2.text(v * 1.15, yi, f"{viz.pct(v, 2)}" + (f", settles in {s_:.0f} steps" if m in ("clamped", "precision", "kalman")
                                                       else ""), va="center", fontsize=8.5)
    a2.set_xscale("log")
    a2.set_xlim(min(vals) / 2, max(vals) * 40)
    a2.set_yticks(y, [STYLE[m][2] for m in names], fontsize=8.5)
    a2.invert_yaxis()
    a2.set(xlabel="steady rate error (last 10 s)", title="Steady state, all seeds")
    a2.grid(axis="y", visible=False)
    return fig


def modes(run: dict):
    a, m = run["arrays"], run["metrics"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    a1.plot(a["lambda_grid"], a["rho_grid"], color=C["nominal"], lw=1.8, label="clamp's error map, per graph frequency")
    a1.scatter(a["lambda_example"], a["rho_example"], color=C["out"], zorder=3, s=26, label="this run's ring of 8")
    a1.axhline(1, color=C["out"], ls="--", lw=1)
    a1.text(a["lambda_grid"][-1], 1.0, "unstable above", ha="right", va="bottom", fontsize=9, color=C["out"])
    a1.set(xlabel="graph frequency λ", ylabel=r"spectral radius of $(I-kc)(K-\lambda M)$",
           title="One 2×2 check per graph frequency")
    lo = min(a["rho_grid"].min(), 0.9)
    a1.set_ylim(lo - 0.01, 1.02)
    a1.legend(loc="lower left", fontsize=9)
    T = len(a["clean_error_model"])
    t = np.arange(T)
    a2.semilogy(t, a["clean_error_model"], color=C["nominal"], lw=1.5, label="world = the model itself")
    a2.semilogy(t, a["clean_error_world"], color=C["actual"], lw=1.2, ls="--", label="the real world")
    rho = m["clean_rho"]
    ref = a["clean_error_model"][60] * rho ** (t - 60)
    a2.semilogy(t, ref, color=C["out"], lw=1, ls=":", label=f"predicted: ρ_max^t, ρ_max = {rho:.4f}")
    a2.set_ylim(max(1e-7, min(a["clean_error_world"].min(), a["clean_error_model"].min()) / 3), 3)
    a2.set(xlabel="steps (noiseless sense)", ylabel="relative state error",
           title="Measured convergence follows the slowest mode")
    a2.legend(loc="lower left", fontsize=8.5)
    return fig


def relaxation(run: dict):
    a = run["arrays"]
    xs, Es = a["relax_states"], a["relax_energy"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4))
    a1.semilogy(np.arange(len(Es)), Es - Es.min() + 1e-12, color=C["nominal"], lw=1.8)
    a1.set(xlabel="relaxation iteration", ylabel="energy above its minimum",
           title="Inside one step: the body relaxes on its energy")
    a2.plot(xs[:, 0], xs[:, 1], "o-", color=C["nominal"], ms=3, lw=1)
    a2.plot(xs[0, 0], xs[0, 1], "o", color=C["muted"], ms=8, label="start: the prediction")
    a2.plot(xs[-1, 0], xs[-1, 1], "*", color=C["out"], ms=13, label="end: the clamped estimate")
    a2.set(xlabel=r"$\theta$ (rad)", ylabel=r"$\dot\theta$ (rad/s)",
           title="The angle error also moves the hidden rate")
    a2.legend(loc="best", fontsize=9)
    return fig


def tradeoff(runs: list[dict]):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.7))
    qs = [c["q"] for c in runs[0]["metrics"]["by_q"]]
    for m, mk in (("clamped", "o"), ("precision", "s"), ("kalman", "^")):
        col, ls, lab = STYLE[m]
        err = [np.mean([r["metrics"]["by_q"][i]["steady"][m]["rate"] for r in runs]) for i in range(len(qs))]
        st = [np.mean([r["metrics"]["by_q"][i]["steady"][m]["settle"] for r in runs]) for i in range(len(qs))]
        a1.plot(np.maximum(st, 1), err, marker=mk, color=col, ls=ls, label=lab)
        if m == "clamped":
            for q, x, y in zip(qs, st, err):
                a1.annotate(f"q = {viz.fmt(q, 1)}", (max(x, 1), y), textcoords="offset points", xytext=(6, 4), fontsize=8,
                            color=C["muted"])
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set(xlabel="steps to settle from a cold start", ylabel="steady rate error",
           title="Fixed precision trades speed for accuracy")
    a1.legend(loc="upper right", fontsize=8.5)
    noises = [c["noise"] for c in runs[0]["metrics"]["by_noise"]]
    for m in ("delay", "full", "precision", "kalman"):
        col, ls, lab = STYLE[m]
        v = [np.mean([r["metrics"]["by_noise"][i]["steady"][m]["rate"] for r in runs]) for i in range(len(noises))]
        a2.loglog(noises, v, marker="o", color=col, ls=ls, label=lab)
    a2.set(xlabel="sense noise (fraction of the swing)", ylabel="steady rate error",
           title="Errors scale with the noise; the gaps stay")
    a2.legend(loc="upper left", fontsize=8.5)
    return fig
