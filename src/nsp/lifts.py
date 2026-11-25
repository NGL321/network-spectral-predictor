"""Lifts: features of one body's state, for swings large enough that sin(theta) matters (E4).

A lifted body step predicts the next base state from features of the current one, then re-lifts:

    x_v(t+1) = W^T [ psi(x_v(t)) ; s_v(t) ],   s_v(t) = sum_u g_uv (x_v(t) - x_u(t))

The springs still read base-state differences; only the body's own map sees the lift.
"""

from __future__ import annotations

import numpy as np

KINDS = ("x", "sin", "sin+cos")

# LaTeX, for Markdown tables and figure labels (plain-text dots over θ render unreliably).
LABELS = {
    "x": r"$(\theta,\ \dot\theta)$",
    "sin": r"$(\theta,\ \dot\theta,\ \sin\theta)$",
    "sin+cos": r"$(\theta,\ \dot\theta,\ \sin\theta,\ \dot\theta\cos\theta)$",
    "invariant": r"$(q,\ \dot q)$ and $q,\ \dot q$ times $\lVert q\rVert^2,\ \lVert \dot q\rVert^2,\ q\cdot\dot q$",
}


def features(xv: np.ndarray, kind: str) -> np.ndarray:
    """Planar lifts (d = 1). xv: (N, 2) rows of (theta, theta'). Returns (N, p)."""
    th, thd = xv[:, :1], xv[:, 1:2]
    if kind == "x":
        return xv
    if kind == "sin":
        return np.hstack([th, thd, np.sin(th)])
    if kind == "sin+cos":
        # theta' cos(theta) is the time derivative of sin(theta): one step needs it at order dt^2
        return np.hstack([th, thd, np.sin(th), thd * np.cos(th)])
    raise ValueError(f"unknown lift {kind!r}")


def invariant_features(xv: np.ndarray) -> np.ndarray:
    """Rotation-invariant cubic lift for two tilts (d = 2, stage 1.5). xv: (N, 4) rows of (q, q')."""
    q, v = xv[:, :2], xv[:, 2:]
    qq = (q**2).sum(1, keepdims=True)
    vv = (v**2).sum(1, keepdims=True)
    qv = (q * v).sum(1, keepdims=True)
    return np.hstack([xv, q * qq, q * vv, q * qv, v * qq, v * vv, v * qv])
