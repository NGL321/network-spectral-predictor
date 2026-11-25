"""Graphs: incidence matrices, gated Laplacians, named graphs and gate patterns.

The incidence matrix delta has one row per spring e = (u, v), with -1 at u and +1 at v, so that
(delta theta)_e = theta_v - theta_u is the twist of spring e. Gates g_e in {0, 1} switch springs on and off:

    L(g) = delta^T diag(g) delta
"""

from __future__ import annotations

import numpy as np

Edge = tuple[int, int]


def path(n: int) -> list[Edge]:
    """Bodies 0..n-1 in a line, with a spring between each neighbouring pair."""
    return [(i, i + 1) for i in range(n - 1)]


def ring(n: int) -> list[Edge]:
    """A path closed into a loop (n >= 3)."""
    return path(n) + [(n - 1, 0)] if n >= 3 else path(n)


def star(n: int) -> list[Edge]:
    """Body 0 is the hub, sprung to every other body."""
    return [(0, i) for i in range(1, n)]


def complete(n: int) -> list[Edge]:
    """A spring between every pair of bodies."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


NAMED = {"path": path, "ring": ring, "star": star, "complete": complete}


def named(name: str, n: int) -> list[Edge]:
    """One of the named graphs: path, ring, star or complete."""
    return NAMED[name](n)


def incidence(n: int, edges: list[Edge]) -> np.ndarray:
    """The incidence matrix delta (springs x bodies)."""
    D = np.zeros((len(edges), n))
    for e, (u, v) in enumerate(edges):
        D[e, u], D[e, v] = -1.0, 1.0
    return D


def laplacian(n: int, edges: list[Edge], gates=None) -> np.ndarray:
    """The gated graph Laplacian L(g) = delta^T diag(g) delta."""
    D = incidence(n, edges)
    g = np.ones(len(edges)) if gates is None else np.asarray(gates, float)
    return D.T @ (g[:, None] * D)


def graph_frequencies(n: int, edges: list[Edge], gates=None) -> np.ndarray:
    """The eigenvalues lambda_k of L(g), in ascending order (graph frequencies)."""
    return np.clip(np.linalg.eigvalsh(laplacian(n, edges, gates)), 0.0, None)


def all_gate_patterns(m: int) -> np.ndarray:
    """Every on/off pattern of m springs, shape (2^m, m); row k is the binary expansion of k."""
    return np.array([[(k >> i) & 1 for i in range(m)] for k in range(2**m)], float)


def random_gate_patterns(m: int, k: int, rng, p_on: float = 0.7, include_all_on: bool = True) -> list[np.ndarray]:
    """k random gate patterns with at least one live spring, after the all-on pattern if asked."""
    pats = [np.ones(m)] if include_all_on else []
    while len(pats) < k + int(include_all_on):
        p = (rng.random(m) < p_on).astype(float)
        if p.sum() > 0:
            pats.append(p)
    return pats


def pair_laplacian(n: int, pairs: list[Edge], weights) -> np.ndarray:
    """Laplacian over an arbitrary candidate pair list with real weights (used by gate detection)."""
    return laplacian(n, pairs, weights)
