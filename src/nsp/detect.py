"""Gate detection (step 1b): which springs are live, given the body and spring maps.

With K and M known the step is linear in the gates, so for candidate pairs (a, b) the residual
r_v = x_v(t+1) - K x_v(t) satisfies

    r = sum_(a,b) g_ab G_ab,   G_ab = -M (x_b - x_a) at b and +M (x_b - x_a) at a.

Plain least squares has an exact blind spot when several identical pendulums hang free (they all swing at
omega0, so some spring combinations among them change no prediction). Non-negativity (g >= 0) or a second
trajectory removes it.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from .graphs import Edge
from .model import OneHop


def design(X: np.ndarray, model: OneHop, pairs: list[Edge], n: int, d: int = 1):
    """Design matrix (one column per candidate pair) and residual vector. X: (ntraj, T + 1, n * 2d)."""
    D = 2 * d
    xs = X[:, :-1].reshape(-1, n, D)
    ys = X[:, 1:].reshape(-1, n, D)
    r = ys - xs @ model.K.T
    cols = []
    for a, b in pairs:
        col = np.zeros_like(xs)
        dv = (xs[:, b] - xs[:, a]) @ model.M.T
        col[:, b], col[:, a] = -dv, dv
        cols.append(col.reshape(-1))
    return np.stack(cols, 1), r.reshape(-1)


def detect_gates(X: np.ndarray, model: OneHop, pairs: list[Edge], n: int, d: int = 1,
                 method: str = "nnls") -> np.ndarray:
    """Estimated gate per candidate pair: 'nnls' (g >= 0) or 'lstsq' (plain least squares)."""
    G, r = design(X, model, pairs, n, d)
    if method == "nnls":
        return nnls(G, r)[0]
    if method == "lstsq":
        return np.linalg.lstsq(G, r, rcond=None)[0]
    raise ValueError(f"unknown method {method!r}")


def truth_vector(pairs: list[Edge], edges: list[Edge], gates) -> np.ndarray:
    """True gate for each candidate pair (0 for pairs with no declared spring)."""
    on = {tuple(sorted(e)): g for e, g in zip(edges, gates)}
    return np.array([on.get(tuple(sorted(p)), 0.0) for p in pairs])


def exact_match(g_hat: np.ndarray, truth: np.ndarray, threshold: float = 0.5) -> bool:
    """Every pair right after thresholding."""
    return bool(np.array_equal((g_hat > threshold).astype(float), (truth > threshold).astype(float)))


def design_singular_values(X: np.ndarray, model: OneHop, pairs: list[Edge], n: int, d: int = 1) -> np.ndarray:
    """Singular values of the design, largest first, normalised by the largest (a near-zero one = blind spot)."""
    G, _ = design(X, model, pairs, n, d)
    s = np.linalg.svd(G, compute_uv=False)
    return s / s[0]


def design_steps(X: np.ndarray, model: OneHop, pairs: list[Edge], n: int, d: int = 1):
    """Per-step design blocks and residuals for one trajectory X of shape (T + 1, n * 2d).

    Returns G of shape (T, n * 2d, m) and r of shape (T, n * 2d): stacking steps a..b gives the design and residual
    of the window a..b, so a sliding detector builds each step once.
    """
    D = 2 * d
    xs, ys = X[:-1].reshape(-1, n, D), X[1:].reshape(-1, n, D)
    r = ys - xs @ model.K.T
    G = np.zeros((len(xs), n, D, len(pairs)))
    for k, (a, b) in enumerate(pairs):
        dv = (xs[:, b] - xs[:, a]) @ model.M.T
        G[:, b, :, k], G[:, a, :, k] = -dv, dv
    return G.reshape(len(xs), n * D, len(pairs)), r.reshape(len(xs), n * D)


def sliding_gates(G: np.ndarray, r: np.ndarray, window: int, method: str = "nnls") -> np.ndarray:
    """Gate estimate after each step t from the window of the last `window` steps ending at t. Returns (T, m)."""
    T, _, m = G.shape
    out = np.empty((T, m))
    for t in range(T):
        a = max(0, t + 1 - window)
        A, b = G[a:t + 1].reshape(-1, m), r[a:t + 1].reshape(-1)
        out[t] = nnls(A, b)[0] if method == "nnls" else np.linalg.lstsq(A, b, rcond=None)[0]
    return out
