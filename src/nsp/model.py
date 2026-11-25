"""The model: one body map K and one spring map M, assembled over a gated graph.

    x_v(t+1) = K x_v(t) - M sum_u g_uv (x_v(t) - x_u(t))
    x(t+1)   = (I_n (x) K - L(g) (x) M) x(t)

This is the flat model K_flat = blockdiag K_v + C Gamma(t) F with identity restriction maps: F = delta (x) I reads
differences across springs, Gamma(t) = diag(g) (x) M is the gated edge map, C = -delta^T (x) I writes back.
In the eigenbasis of L(g) the assembled step splits into one block K - lambda_k M per graph frequency lambda_k.

The exact small-swing step, expanded in lambda, is

    e^{(A0 - kappa lambda E) dt} = K0 - lambda M0 + lambda^2 N0 + O(lambda^3)

so the one-hop model misses mainly the two-hop term L(g)^2 (x) N0, of relative size kappa lambda dt^2 / 6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, expm_frechet

from . import lifts as _lifts
from .world import Schedule, World


@dataclass
class OneHop:
    """Shared body map K and spring map M, each 2d x 2d."""

    K: np.ndarray
    M: np.ndarray

    def assemble(self, L: np.ndarray) -> np.ndarray:
        """The whole network's step: I_n (x) K - L (x) M."""
        n = L.shape[0]
        return np.kron(np.eye(n), self.K) - np.kron(L, self.M)

    def mode_block(self, lam: float) -> np.ndarray:
        """The step of one graph mode: K - lambda M."""
        return self.K - lam * self.M

    def spectrum(self, L: np.ndarray):
        """Graph frequencies of L and, for each, the eigenvalues of K - lambda M."""
        lams = np.clip(np.linalg.eigvalsh(L), 0.0, None)
        return lams, [np.linalg.eigvals(self.mode_block(lam)) for lam in lams]

    def mode_frequencies(self, lams, dt: float):
        """Frequency (|arg mu| / dt) and per-step modulus |mu| the model plays at each graph frequency."""
        w, r = [], []
        for lam in np.atleast_1d(lams):
            mu = np.linalg.eigvals(self.mode_block(lam))
            w.append(np.abs(np.angle(mu)).max() / dt)
            r.append(np.abs(mu).max())
        return np.array(w), np.array(r)

    def predict(self, X: np.ndarray, L: np.ndarray) -> np.ndarray:
        """One-step predictions of X[:, 1:] from X[:, :-1]."""
        return X[:, :-1] @ self.assemble(L).T

    def rollout(self, x0: np.ndarray, schedule: Schedule) -> np.ndarray:
        """Free-running prediction from x0 under a schedule of (steps, L) segments."""
        x0 = np.atleast_2d(x0)
        T = sum(steps for steps, _ in schedule)
        X = np.empty((x0.shape[0], T + 1, x0.shape[1]))
        X[:, 0] = x0
        t = 0
        for steps, L in schedule:
            P = self.assemble(L)
            for _ in range(steps):
                X[:, t + 1] = X[:, t] @ P.T
                t += 1
        return X

    def per_direction(self, d: int) -> "OneHop":
        """Apply a planar model once per direction: K (x) I_d, M (x) I_d (stage 1.5a)."""
        I = np.eye(d)
        return OneHop(np.kron(self.K, I), np.kron(self.M, I))


def series(world: World) -> OneHop:
    """The one-hop series: K0 = e^{A0 dt} and M0 = the first-order coefficient in lambda (a Frechet derivative)."""
    B = world.A0 * world.dt
    K0 = expm(B)
    M0 = expm_frechet(B, world.kappa * world.E * world.dt)[1]
    return OneHop(K0, M0)


def series_closed_form(world: World):
    """Closed forms from the plan: M0 to O(dt^3) and N0 to O(dt^4), each lifted by (x) I_d."""
    k, dt, w2 = world.kappa, world.dt, world.omega0**2
    M0 = k * np.array([[dt**2 / 2, dt**3 / 6], [dt - w2 * dt**3 / 3, dt**2 / 2]])
    N0 = k**2 * np.array([[dt**4 / 24, 0.0], [dt**3 / 6, dt**4 / 24]])
    I = np.eye(world.d)
    return np.kron(M0, I), np.kron(N0, I)


def two_hop_ratio(world: World, lam):
    """Size of the missed two-hop term relative to one hop: kappa lambda dt^2 / 6."""
    return world.kappa * np.asarray(lam) * world.dt**2 / 6


def predicted_growth(world: World, lam):
    """Per-step growth of the one-hop series at graph frequency lambda: (kappa lambda)^2 dt^4 / 24."""
    return (world.kappa * np.asarray(lam)) ** 2 * world.dt**4 / 24


def predicted_frequency_error(world: World, lam):
    """Relative frequency error of the one-hop series: (kappa lambda dt)^2 / (12 omega^2)."""
    lam = np.asarray(lam)
    return (world.kappa * lam * world.dt) ** 2 / (12 * (world.omega0**2 + world.kappa * lam))


@dataclass
class LiftedOneHop:
    """Planar body step from lifted features, re-lifted every step (E4): x_v(t+1) = W^T [psi(x_v), s_v]."""

    W: np.ndarray
    lift: str

    def _next(self, x: np.ndarray, L: np.ndarray) -> np.ndarray:
        xv = x.reshape(-1, 2)
        s = (x @ np.kron(L, np.eye(2)).T).reshape(-1, 2)
        return (np.hstack([_lifts.features(xv, self.lift), s]) @ self.W).reshape(x.shape)

    def predict(self, X: np.ndarray, L: np.ndarray) -> np.ndarray:
        B = X[:, :-1].reshape(-1, X.shape[-1])
        return self._next(B, L).reshape(X.shape[0], X.shape[1] - 1, X.shape[-1])

    def rollout(self, x0: np.ndarray, L: np.ndarray, steps: int) -> np.ndarray:
        x0 = np.atleast_2d(x0)
        X = np.empty((x0.shape[0], steps + 1, x0.shape[1]))
        X[:, 0] = x0
        for t in range(steps):
            X[:, t + 1] = self._next(X[:, t], L)
        return X
