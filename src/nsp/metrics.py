"""Metrics: errors relative to the actual world, frequencies, growth, energy and AUROC.

Vocabulary (metrology, inverted on purpose): the *actual* world is the generated one; the *nominal* world is
the model's prediction. Deviation = actual - nominal.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from .model import OneHop
from .world import World, mode_frequencies


def relative(pred: np.ndarray, true: np.ndarray) -> float:
    """||pred - true|| / ||true|| over everything given."""
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


def one_step_error(X: np.ndarray, P: np.ndarray) -> float:
    """Relative one-step error of a linear step P on trajectories X (ntraj, T + 1, N)."""
    return relative(X[:, :-1] @ P.T, X[:, 1:])


def rollout_curve(X_actual: np.ndarray, X_nominal: np.ndarray) -> np.ndarray:
    """Relative error at each step of a free-running rollout, pooled over trajectories."""
    num = np.linalg.norm(X_nominal - X_actual, axis=(0, 2))
    den = np.linalg.norm(X_actual, axis=(0, 2))
    return num / den


def at_horizons(curve: np.ndarray, horizons) -> dict:
    return {int(h): float(curve[h]) for h in horizons}


def mode_report(model: OneHop, L: np.ndarray, world: World) -> dict:
    """Per graph mode: lambda, the actual frequency, the nominal frequency and modulus, and their errors."""
    lams = np.clip(np.linalg.eigvalsh(L), 0.0, None)
    w_act = np.sqrt(world.omega0**2 + world.kappa * lams)
    w_nom, mod = model.mode_frequencies(lams, world.dt)
    return {"lambda": lams, "omega_actual": w_act, "omega_nominal": w_nom,
            "frequency_error": (w_nom - w_act) / w_act, "growth": mod - 1.0}


def energy_drift(H: np.ndarray) -> float:
    """max_t |H(t) - H(0)| / |H(0)|, worst trajectory. H: (..., T)."""
    H = np.atleast_2d(H)
    return float(np.max(np.abs(H - H[..., :1]) / np.abs(H[..., :1])))


def auroc(scores, labels) -> float:
    """Area under the ROC curve: chance a true spring outscores a missing one (ties count half)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels).astype(bool)
    ranks = rankdata(scores)
    npos, nneg = labels.sum(), (~labels).sum()
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def status(deviation: float, tolerance: float) -> str:
    """Inspection status: 'in' within tolerance, 'near' within twice it, else 'out'."""
    a = abs(deviation)
    return "in" if a <= tolerance else ("near" if a <= 2 * tolerance else "out")


__all__ = ["relative", "one_step_error", "rollout_curve", "at_horizons", "mode_report", "energy_drift", "auroc",
           "status", "mode_frequencies"]
