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


def invariant_features(xv: np.ndarray, order: int = 3) -> np.ndarray:
    """Rotation-equivariant lift for two tilts (d = 2, stage 1.5). xv: (N, 4) rows of (q, q').

    q and q' times every invariant up to the given order: 1; |q|^2, |q'|^2, q.q' (cubic, 16 numbers); and their six
    pairwise products (quintic, 28 numbers). Equivariant: rotating the input rotates the features the same way.
    """
    q, v = xv[:, :2], xv[:, 2:]
    qq = (q**2).sum(1, keepdims=True)
    vv = (v**2).sum(1, keepdims=True)
    qv = (q * v).sum(1, keepdims=True)
    inv = [qq, vv, qv]
    if order >= 5:
        inv += [qq * qq, vv * vv, qv * qv, qq * vv, qq * qv, vv * qv]
    return np.hstack([xv] + [q * s for s in inv] + [v * s for s in inv])
