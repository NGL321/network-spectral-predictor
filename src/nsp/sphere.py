"""Spherical pendulums joined by real springs, for swings of any size (stage 1.5, part A): RATTLE.

Each bob b_v in R^3 hangs on a rigid rod of length l from the pivot p_v = (P_v, 0), and gravity pulls it down with
g = omega0^2 l. A spring joins bobs u and v with natural length s_e = |P_v - P_u|, so at rest (every bob straight below
its pivot) every spring is at its natural length. Per unit mass:

    b_v'' = -g z  -  kappa sum_e g_e (|b_v - b_u| - s_e) (b_v - b_u) / |b_v - b_u|  +  lambda_v (b_v - p_v)

where lambda_v is the rod's tension, whatever keeps |b_v - p_v| = l. RATTLE (Andersen 1983) is velocity Verlet with
that constraint solved exactly at every substep, once for positions and once for velocities. It is symplectic, and it
keeps the rods rigid to rounding error.

The network sees the chart q_v = (horizontal bob offset) / l and its rate, as in `sheaf`. For small swings a spring's
length changes by l n_e . (q_v - q_u) to first order, and the world reduces to the oriented generator of `sheaf`.
At large swings each spring's axis turns with its bobs, and the bobs' own paths precess.
"""

from __future__ import annotations

import numpy as np

from .graphs import Edge
from .sheaf import GateSchedule
from .world import World


def to_bobs(P: np.ndarray, X: np.ndarray, rod: float = 1.0):
    """Chart states (..., 4n) to bob positions and velocities, each (..., n, 3)."""
    n = len(P)
    Xb = X.reshape(*X.shape[:-1], n, 2, 2)
    q, qd = Xb[..., 0, :], Xb[..., 1, :]
    c = np.sqrt(np.clip(1.0 - (q**2).sum(-1), 1e-12, None))
    b = np.concatenate([P + rod * q, (-rod * c)[..., None]], -1)
    v = np.concatenate([rod * qd, (rod * (q * qd).sum(-1) / c)[..., None]], -1)
    return b, v


def to_chart(P: np.ndarray, b: np.ndarray, v: np.ndarray, rod: float = 1.0) -> np.ndarray:
    """Bob positions and velocities (..., n, 3) to chart states (..., 4n), ordered phase (x) direction per body."""
    q = (b[..., :2] - P) / rod
    qd = v[..., :2] / rod
    return np.stack([q, qd], -2).reshape(*b.shape[:-2], -1)


def _accel(b: np.ndarray, edges: list[Edge], rest: np.ndarray, gates: np.ndarray, kappa: float, g: float):
    a = np.zeros_like(b)
    a[..., 2] = -g
    for e, (u, v) in enumerate(edges):
        if gates[e] == 0:
            continue
        D = b[..., v, :] - b[..., u, :]
        L = np.linalg.norm(D, axis=-1, keepdims=True)
        f = -kappa * gates[e] * (L - rest[e]) / L * D
        a[..., v, :] += f
        a[..., u, :] -= f
    return a


def simulate(world: World, P: np.ndarray, edges: list[Edge], schedule: GateSchedule, x0: np.ndarray,
             substeps: int = 50, rod: float = 1.0) -> np.ndarray:
    """RATTLE, `substeps` per sample. x0: (ntraj, 4n) chart states -> X: (ntraj, T + 1, 4n)."""
    x0 = np.atleast_2d(x0)
    n = len(P)
    piv = np.concatenate([P, np.zeros((n, 1))], 1)
    rest = np.array([np.linalg.norm(P[v] - P[u]) for u, v in edges])
    g = world.omega0**2 * rod
    h = world.dt / substeps
    b, vel = to_bobs(P, x0, rod)
    T = sum(steps for steps, _ in schedule)
    X = np.empty((x0.shape[0], T + 1, 4 * n))
    X[:, 0] = to_chart(P, b, vel, rod)
    t = 0
    for steps, gates in schedule:
        gates = np.ones(len(edges)) if gates is None else np.asarray(gates, float)
        a = _accel(b, edges, rest, gates, world.kappa, g)
        for _ in range(steps):
            for _ in range(substeps):
                r = b - piv
                u = r + h * vel + 0.5 * h * h * a  # unconstrained step, relative to the pivot
                ur, rr, uu = (u * r).sum(-1), (r * r).sum(-1), (u * u).sum(-1)
                c = (-ur + np.sqrt(ur**2 - rr * (uu - rod**2))) / rr  # the rod's pull that lands on the sphere
                r_new = u + c[..., None] * r
                b_new = piv + r_new
                w = (b_new - b) / h
                a = _accel(b_new, edges, rest, gates, world.kappa, g)
                w = w + 0.5 * h * a
                vel = w - ((r_new * w).sum(-1) / rod**2)[..., None] * r_new  # velocity tangent to the sphere
                b = b_new
            t += 1
            X[:, t] = to_chart(P, b, vel, rod)
    return X


def energy(world: World, P: np.ndarray, edges: list[Edge], X: np.ndarray, gates=None, rod: float = 1.0) -> np.ndarray:
    """Energy per unit mass: kinetic + g z + (kappa / 2) sum_e g_e (|b_v - b_u| - s_e)^2. X: (..., 4n)."""
    gates = np.ones(len(edges)) if gates is None else np.asarray(gates, float)
    b, v = to_bobs(P, X, rod)
    E = 0.5 * (v**2).sum((-1, -2)) + world.omega0**2 * rod * b[..., 2].sum(-1)
    for e, (u, w) in enumerate(edges):
        L = np.linalg.norm(b[..., w, :] - b[..., u, :], axis=-1)
        E = E + 0.5 * world.kappa * gates[e] * (L - np.linalg.norm(P[w] - P[u])) ** 2
    return E


def initial_states(n: int, ntraj: int, amplitude: float, rng, omega0: float = 2.0, max_tilt: float = 0.8) -> np.ndarray:
    """Random chart states: tilts ~ N(0, amplitude^2) per direction, rates ~ N(0, (omega0 amplitude)^2).

    Tilts are redrawn until each bob sits within max_tilt of straight down (|q| = 1 would put it level with its pivot).
    """
    x0 = np.zeros((ntraj, n, 2, 2))
    q = rng.normal(0.0, amplitude, (ntraj, n, 2))
    bad = np.linalg.norm(q, axis=-1) > max_tilt
    while bad.any():
        q[bad] = rng.normal(0.0, amplitude, (int(bad.sum()), 2))
        bad = np.linalg.norm(q, axis=-1) > max_tilt
    x0[..., 0, :] = q
    x0[..., 1, :] = rng.normal(0.0, amplitude * omega0, (ntraj, n, 2))
    return x0.reshape(ntraj, 4 * n)


# --------------------------------------------------------------------------------------- large swings: the model side

def spring_reads(P: np.ndarray, edges: list[Edge], X: np.ndarray, gates=None, rod: float = 1.0, eps: float = 1e-6):
    """What the springs tell each body, read from the current geometry, per phase. X: (N, 4n) -> (N, n, 2, 2).

    Position phase: the springs' pull on bob v, f_v = sum_e g_e (|b_v - b_u| - s_e) (b_v - b_u) / |b_v - b_u| (the force
    is -kappa f_v), as the chart feels it: the rod takes the part along itself, so a tilt q_v moves by the tangential part,

        Q_v = f_v,xy - (r_v . f_v) q_v,    r_v = (q_v, -sqrt(1 - |q_v|^2))  (the rod's unit vector),

    in rod lengths. Rate phase: dQ_v/dt along the current rates. For small swings these are the fixed-axis reads
    sum_e n_e n_e^T (q_v - q_u) and sum_e n_e n_e^T (q_v' - q_u') of `sheaf`.
    """
    gates = np.ones(len(edges)) if gates is None else np.asarray(gates, float)
    rest = np.array([np.linalg.norm(P[v] - P[u]) for u, v in edges])
    n = len(P)

    def pos_read(Xs):
        b, _ = to_bobs(P, Xs, rod)
        f = np.zeros((len(Xs), n, 3))
        for e, (u, v) in enumerate(edges):
            if gates[e] == 0:
                continue
            D = b[:, v] - b[:, u]
            L = np.linalg.norm(D, axis=-1, keepdims=True)
            fe = gates[e] * (L - rest[e]) * D / L / rod
            f[:, v] += fe
            f[:, u] -= fe
        q = Xs.reshape(len(Xs), n, 2, 2)[:, :, 0]
        r = np.concatenate([q, -np.sqrt(np.clip(1 - (q**2).sum(-1, keepdims=True), 1e-12, None))], -1)
        return f[..., :2] - (r * f).sum(-1, keepdims=True) * q

    X = np.atleast_2d(X)
    Xb = X.reshape(len(X), n, 2, 2)
    shift = np.zeros_like(Xb)
    shift[:, :, 0] = eps * Xb[:, :, 1]  # move the tilts along their rates
    up, down = (Xb + shift).reshape(len(X), -1), (Xb - shift).reshape(len(X), -1)
    return np.stack([pos_read(X), (pos_read(up) - pos_read(down)) / (2 * eps)], 2)


class LiftedSheafModel:
    """Large swings: a lifted body map and the shared spring map M, the springs read from fixed or current geometry.

        x_v(t+1) = W^T psi(x_v(t)) - sum_q M_pq Q_v,(q, .)

    psi is the state itself (lift "none") or the rotation-equivariant lift of `lifts.invariant_features` ("cubic", 16
    numbers; "quintic", 28). reads = "fixed": the springs' axes at rest (A2's model); "moving": the current geometry
    (`spring_reads`).
    """

    def __init__(self, W: np.ndarray, M: np.ndarray, lift: str, reads: str):
        self.W, self.M, self.lift, self.reads = W, M, lift, reads

    @staticmethod
    def features(xv: np.ndarray, lift: str) -> np.ndarray:
        from .lifts import invariant_features
        return xv if lift == "none" else invariant_features(xv, {"cubic": 3, "quintic": 5}[lift])

    @staticmethod
    def reads_of(P, edges, X, gates, reads: str) -> np.ndarray:
        if reads == "moving":
            return spring_reads(P, edges, X, gates)
        from .sheaf import _spring_reads
        Xp = np.concatenate([X[:, None], X[:, None]], 1)  # _spring_reads reads every step but the last
        return _spring_reads(Xp, P, edges, gates)

    def step(self, X: np.ndarray, P, edges, gates=None) -> np.ndarray:
        n = len(P)
        Q = self.reads_of(P, edges, X, gates, self.reads)
        out = self.features(X.reshape(-1, 4), self.lift) @ self.W
        out = out.reshape(len(X), n, 2, 2) - np.einsum("pq,bnqi->bnpi", self.M, Q)
        return out.reshape(X.shape)

    def rollout(self, x0: np.ndarray, P, edges, steps: int, gates=None) -> np.ndarray:
        x0 = np.atleast_2d(x0)
        X = np.empty((x0.shape[0], steps + 1, x0.shape[1]))
        X[:, 0] = x0
        for t in range(steps):
            X[:, t + 1] = self.step(X[:, t], P, edges, gates)
        return X


def fit_lifted(sets, lift: str, reads: str) -> LiftedSheafModel:
    """W and M by one least-squares solve. sets: list of (X (ntraj, T + 1, 4n), P, edges, gates)."""
    rows = {0: [], 1: []}
    rhs = {0: [], 1: []}
    for X, P, edges, gates in sets:
        n = len(P)
        xs = X[:, :-1].reshape(-1, 4 * n)
        ys = X[:, 1:].reshape(-1, n, 2, 2).reshape(-1, 2, 2)
        F = LiftedSheafModel.features(xs.reshape(-1, 4), lift)
        Q = LiftedSheafModel.reads_of(P, edges, xs, gates, reads).reshape(-1, 2, 2)
        f = F.shape[1]
        for p in (0, 1):
            for i in (0, 1):
                Wc = np.zeros((len(F), 2 * f))
                Wc[:, i * f:(i + 1) * f] = F
                rows[p].append(np.hstack([Wc, -Q[:, :, i]]))
                rhs[p].append(ys[:, p, i])
    f = LiftedSheafModel.features(np.zeros((1, 4)), lift).shape[1]
    W = np.zeros((f, 4))
    M = np.zeros((2, 2))
    for p in (0, 1):
        w = np.linalg.lstsq(np.vstack(rows[p]), np.concatenate(rhs[p]), rcond=None)[0]
        W[:, 2 * p + 0], W[:, 2 * p + 1] = w[:f], w[f:2 * f]
        M[p] = w[2 * f:]
    return LiftedSheafModel(W, M, lift, reads)
