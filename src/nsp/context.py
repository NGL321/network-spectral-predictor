"""Transfer by recall (E6): a structural vertex that stores gate patterns, and a binding vertex that recalls one.

Structural vertex: the bank G, one row per stored context, G_c in {0, 1}^m over m candidate springs.
Binding vertex (the hippocampal analog): a belief pi over stored contexts, predicted and corrected like any vertex:

    alpha(t+1) = diag(l(t+1)) T^T alpha(t),   pi = alpha / (1^T alpha),   g(t) = G^T pi(t)
    l_c(t+1)   = exp(-1/2 || x(t+1) - (I (x) K - L(G_c) (x) M) x(t) ||^2_{Sigma^-1})
    T          = (1 - h) I + h / (C - 1) (1 1^T - I)

Unnormalised, it is a linear predictor T^T gated by evidence: the same algebra as a gated edge.
A new structure is created when the best stored one loses to one detected on the same window by more than tau:

    Lambda = min_c sum_t NLL_c(t) - sum_t NLL_detected(t) > tau
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detect import detect_gates
from .graphs import Edge, laplacian
from .model import OneHop


@dataclass
class ContextBank:
    """Stored structures over a fixed edge list."""

    n: int
    edges: list[Edge]
    patterns: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    def laplacians(self) -> list[np.ndarray]:
        return [laplacian(self.n, self.edges, p) for p in self.patterns]

    def append(self, pattern) -> int:
        """Store a new structure; returns its index."""
        p = np.asarray(pattern, float)[None]
        self.patterns = p if self.patterns.size == 0 else np.vstack([self.patterns, p])
        return len(self.patterns) - 1


def residual_scale(X: np.ndarray, P: np.ndarray, d: int = 1) -> np.ndarray:
    """Per-coordinate residual standard deviation (positions, rates), tiled over bodies. X: (ntraj, T + 1, N)."""
    R = (X[:, 1:] - X[:, :-1] @ P.T).reshape(-1, X.shape[-1] // (2 * d), 2 * d)
    s = np.r_[np.full(d, R[..., :d].std()), np.full(d, R[..., d:].std())]
    return np.tile(s, X.shape[-1] // (2 * d))


def uniform_transitions(C: int, hazard: float) -> np.ndarray:
    """Stay with probability 1 - h; otherwise jump to any other stored structure, all equally likely."""
    return (1 - hazard) * np.eye(C) + hazard / max(C - 1, 1) * (1 - np.eye(C))


def flip_transitions(patterns: np.ndarray, hazard: float) -> np.ndarray:
    """Stay with probability 1 - h; otherwise flip exactly one spring (structures one gate apart, all equally likely)."""
    G = np.asarray(patterns) > 0.5
    one_apart = (G[:, None, :] != G[None, :, :]).sum(-1) == 1
    T = one_apart / np.maximum(one_apart.sum(1, keepdims=True), 1) * hazard
    return T + (1 - hazard) * np.eye(len(G))


class BindingVertex:
    """Belief over the bank's stored structures, filtered step by step.

    `transitions` is the prior over how structures change, T[c, c'] = P(c' next | c now); the default is uniform jumps.
    """

    def __init__(self, bank: ContextBank, model: OneHop, scale: np.ndarray, hazard: float = 0.01,
                 transitions: np.ndarray | None = None):
        self.bank, self.model, self.scale, self.hazard = bank, model, np.asarray(scale), hazard
        self.P = np.array([model.assemble(L) for L in bank.laplacians()])
        C = len(self.P)
        self.T = uniform_transitions(C, hazard) if transitions is None else np.asarray(transitions)

    def nll_steps(self, X: np.ndarray) -> np.ndarray:
        """Per-step negative log-likelihood (up to a constant) of each stored context. X: (T + 1, N) -> (T, C)."""
        R = X[None, 1:] - np.einsum("cij,tj->cti", self.P, X[:-1])
        return 0.5 * ((R / self.scale) ** 2).sum(-1).T

    def filter(self, X: np.ndarray, prior=None) -> np.ndarray:
        """Belief after each step, starting from a uniform prior. Returns (T, C)."""
        nl = self.nll_steps(X)
        C = nl.shape[1]
        pi = np.full(C, 1.0 / C) if prior is None else np.asarray(prior, float)
        out = np.empty_like(nl)
        for t in range(len(nl)):
            pi = np.exp(-(nl[t] - nl[t].min())) * (self.T.T @ pi)
            pi /= pi.sum()
            out[t] = pi
        return out

    def recall(self, X: np.ndarray) -> int:
        """The stored structure that best explains a window."""
        return int(np.argmin(self.nll_steps(X).sum(0)))

    def likelihood_ratio(self, X: np.ndarray, pairs: list[Edge]) -> float:
        """Lambda: best stored structure's NLL minus that of a structure detected on the same window."""
        best = self.nll_steps(X).sum(0).min()
        n, d = self.bank.n, self.model.K.shape[0] // 2
        g = detect_gates(X[None], self.model, pairs, n, d)
        P = self.model.assemble(laplacian(n, pairs, g))
        free = 0.5 * (((X[1:] - X[:-1] @ P.T) / self.scale) ** 2).sum()
        return float(best - free)
