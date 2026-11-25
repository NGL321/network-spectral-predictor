"""The world: pendulums on springs, generated exactly (small swings) or integrated (large swings).

Each body has d position coordinates q (d = 1: one swing angle; d = 2: two tilts) and their rates.
The state of body v is x_v = (q_v, q_v'), ordered phase (x) direction, so planar maps lift as A0 (x) I_d.
For small swings the whole system is linear:

    x' = A x,   A = I_n (x) (A0 (x) I_d) - kappa L(g) (x) (E (x) I_d),
    A0 = [[0, 1], [-omega0^2, 0]],   E = [[0, 0], [1, 0]]

Its modes oscillate at omega_k = sqrt(omega0^2 + kappa lambda_k), one per graph frequency lambda_k.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import expm
from scipy.special import ellipk

Schedule = list[tuple[int, np.ndarray]]  # segments of (steps, L): the graph Laplacian in force for that many steps


@dataclass(frozen=True)
class World:
    """Physical constants. omega0 in rad/s, kappa in 1/s^2 (stiffness / inertia), dt in s."""

    omega0: float = 2.0
    kappa: float = 1.0
    dt: float = 0.05
    d: int = 1

    @property
    def A0(self) -> np.ndarray:
        """One body's generator, A0 (x) I_d."""
        return np.kron(np.array([[0.0, 1.0], [-self.omega0**2, 0.0]]), np.eye(self.d))

    @property
    def E(self) -> np.ndarray:
        """Routes a position into an acceleration: E (x) I_d."""
        return np.kron(np.array([[0.0, 0.0], [1.0, 0.0]]), np.eye(self.d))

    @property
    def body_size(self) -> int:
        """Numbers per body: 2d."""
        return 2 * self.d

    def with_(self, **changes) -> "World":
        return replace(self, **changes)


def generator(world: World, L: np.ndarray) -> np.ndarray:
    """A = I_n (x) A0 - kappa L (x) E."""
    n = L.shape[0]
    return np.kron(np.eye(n), world.A0) - world.kappa * np.kron(L, world.E)


def exact_step(world: World, L: np.ndarray) -> np.ndarray:
    """The exact small-swing step e^{A dt}."""
    return expm(generator(world, L) * world.dt)


def mode_frequencies(world: World, L: np.ndarray) -> np.ndarray:
    """omega_k = sqrt(omega0^2 + kappa lambda_k) for each graph frequency lambda_k of L."""
    lam = np.clip(np.linalg.eigvalsh(L), 0.0, None)
    return np.sqrt(world.omega0**2 + world.kappa * lam)


def swing_frequency(world: World, amplitude: float) -> float:
    """Frequency of a lone planar pendulum swinging to +-amplitude rad: omega0 pi / (2 K(sin^2(a/2)))."""
    return world.omega0 * np.pi / (2 * ellipk(np.sin(amplitude / 2) ** 2))


def initial_states(world: World, n: int, ntraj: int, amplitude: float, rng, kind: str = "normal") -> np.ndarray:
    """Random starting states, shape (ntraj, n * 2d).

    "normal": positions ~ N(0, amplitude^2), rates ~ N(0, (omega0 amplitude)^2).
    "uniform": positions ~ U(-amplitude, amplitude), released from rest.
    """
    d = world.d
    x0 = np.zeros((ntraj, n, 2 * d))
    if kind == "normal":
        x0[..., :d] = rng.normal(0.0, amplitude, (ntraj, n, d))
        x0[..., d:] = rng.normal(0.0, amplitude * world.omega0, (ntraj, n, d))
    elif kind == "uniform":
        x0[..., :d] = rng.uniform(-amplitude, amplitude, (ntraj, n, d))
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return x0.reshape(ntraj, n * 2 * d)


def simulate_exact(world: World, schedule: Schedule, x0: np.ndarray) -> np.ndarray:
    """Small-swing world, exact: x(t+1) = e^{A dt} x(t), switching L between schedule segments.

    x0: (ntraj, N). Returns X: (ntraj, T + 1, N).
    """
    x0 = np.atleast_2d(x0)
    T = sum(steps for steps, _ in schedule)
    X = np.empty((x0.shape[0], T + 1, x0.shape[1]))
    X[:, 0] = x0
    t = 0
    for steps, L in schedule:
        P = exact_step(world, L)
        for _ in range(steps):
            X[:, t + 1] = X[:, t] @ P.T
            t += 1
    return X


def simulate_verlet(world: World, schedule: Schedule, x0: np.ndarray, nonlinear: bool = True,
                    substeps: int = 50) -> np.ndarray:
    """Planar pendulums (d = 1) by velocity Verlet, `substeps` per sample; gravity uses sin(theta) when nonlinear."""
    if world.d != 1:
        raise NotImplementedError("the Verlet world is planar (d = 1); stage 1.5 adds the spherical one")
    x0 = np.atleast_2d(x0)
    n = x0.shape[1] // 2
    th, thd = x0[:, 0::2].copy(), x0[:, 1::2].copy()
    T = sum(steps for steps, _ in schedule)
    X = np.empty((x0.shape[0], T + 1, 2 * n))
    X[:, 0] = x0
    h = world.dt / substeps
    w2, kap = world.omega0**2, world.kappa
    t = 0
    for steps, L in schedule:
        def acc(q, L=L):
            return -w2 * (np.sin(q) if nonlinear else q) - kap * q @ L  # L is symmetric

        a = acc(th)
        for _ in range(steps):
            for _ in range(substeps):
                th_new = th + h * thd + 0.5 * h * h * a
                a_new = acc(th_new)
                thd = thd + 0.5 * h * (a + a_new)
                th, a = th_new, a_new
            t += 1
            X[:, t, 0::2], X[:, t, 1::2] = th, thd
    return X


def body_view(world: World, X: np.ndarray) -> np.ndarray:
    """Reshape (..., n * 2d) to (..., n, 2d)."""
    return X.reshape(*X.shape[:-1], X.shape[-1] // (2 * world.d), 2 * world.d)


def positions(world: World, X: np.ndarray) -> np.ndarray:
    """Position coordinates q: shape (..., n) for d = 1, (..., n, d) otherwise."""
    q = body_view(world, X)[..., : world.d]
    return q[..., 0] if world.d == 1 else q


def energy(world: World, X: np.ndarray, L: np.ndarray, nonlinear: bool = False) -> np.ndarray:
    """Energy per unit inertia: sum_v [ |q_v'|^2 / 2 + gravity ] + (kappa / 2) sum_dirs q^T L q.

    Gravity is omega0^2 (1 - cos theta) when nonlinear (planar only), omega0^2 |q|^2 / 2 otherwise.
    """
    d = world.d
    Xb = body_view(world, X)
    q, qd = Xb[..., :d], Xb[..., d:]
    if nonlinear:
        if d != 1:
            raise NotImplementedError("nonlinear energy is planar (d = 1) for now")
        grav = world.omega0**2 * (1 - np.cos(q[..., 0])).sum(-1)
    else:
        grav = 0.5 * world.omega0**2 * (q**2).sum((-1, -2))
    kin = 0.5 * (qd**2).sum((-1, -2))
    spring = 0.5 * world.kappa * np.einsum("...id,ij,...jd->...", q, L, q)
    return kin + grav + spring


def add_noise(world: World, X: np.ndarray, rel: float, amplitude: float, rng) -> np.ndarray:
    """Measurement noise: rel x amplitude on positions, rel x amplitude x omega0 on rates."""
    if rel == 0:
        return X.copy()
    d = world.d
    n = X.shape[-1] // (2 * d)
    scale = np.tile(np.r_[np.full(d, rel * amplitude), np.full(d, rel * amplitude * world.omega0)], n)
    return X + rng.normal(0.0, 1.0, X.shape) * scale
