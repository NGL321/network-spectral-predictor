"""Oriented springs on spherical pendulums (stage 1.5, part A): a sheaf over the spring graph.

Each body tilts in two directions, q_v = (q_x, q_y) = horizontal bob offset / rod length, with state
x_v = (q_v, q_v') ordered phase (x) direction, as in `world` with d = 2. Bobs hang below pivots P_v in a horizontal
plane. The spring between bodies u and v lies along the unit axis n_e = (P_v - P_u) / |P_v - P_u| and, at its natural
length, resists only stretch along that axis. So each spring reads the difference of its two bodies through the
projector n_e n_e^T, its restriction map:

    x' = A x,   A = I_n (x) A0 (x) I_2 - kappa sum_e g_e (b_e b_e^T) (x) E (x) n_e n_e^T

with b_e the spring's incidence vector (-1 at u, +1 at v). The sum over springs of g_e (b_e b_e^T) (x) n_e n_e^T is the
sheaf Laplacian L_F = R^T diag(g) R: stalks R^2 at bodies, R on springs, restriction maps n_e^T, and R the rigidity
matrix of the bar framework (row e: -n_e^T at u, +n_e^T at v). The planar model is the case n_e n_e^T = 1.

The sheaf model keeps step 1's maps and lets the geometry place them:

    x(t+1) = ( I_n (x) K - sum_e g_e (b_e b_e^T) (x) M (x) n_e n_e^T ) x(t)

with K a 4 x 4 body map (K2 (x) I_2 for an isotropic body) and M one 2 x 2 spring map shared by every spring. With
K = K2 (x) I_2 the spectrum identity carries over with L_F in place of L: the step splits into one 2 x 2 block
K2 - mu_k M per eigenvalue mu_k of L_F. A graph model, with one 4 x 4 map for every spring, cannot place them: it
sees the same coupling whatever the axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, null_space

from .graphs import Edge
from .model import OneHop
from .world import World

GateSchedule = list[tuple[int, np.ndarray]]  # segments of (steps, gates) over a fixed layout and spring list
E2 = np.array([[0.0, 0.0], [1.0, 0.0]])  # routes a position into an acceleration (phase space of one direction)


# --------------------------------------------------------------------------------------------------- geometry

def layout(n: int, rng, box: float = 2.0, min_sep: float = 0.45) -> np.ndarray:
    """n pivots drawn uniformly in a box x box square, no two closer than min_sep. Shape (n, 2)."""
    while True:
        P = rng.uniform(0.0, box, (n, 2))
        d = np.linalg.norm(P[:, None] - P[None], axis=-1) + np.eye(n) * 1e9
        if d.min() > min_sep:
            return P


def circle_layout(n: int, radius: float = 1.0, phase: float = 0.0) -> np.ndarray:
    """n pivots evenly spaced on a circle (a regular polygon)."""
    a = phase + 2 * np.pi * np.arange(n) / n
    return radius * np.stack([np.cos(a), np.sin(a)], 1)


def angular_ring(P: np.ndarray) -> list[Edge]:
    """Springs joining the pivots in angular order around their centroid: a ring that never crosses itself."""
    c = P - P.mean(0)
    order = np.argsort(np.arctan2(c[:, 1], c[:, 0]))
    n = len(P)
    return [(int(order[i]), int(order[(i + 1) % n])) for i in range(n)]


def add_chords(P: np.ndarray, edges: list[Edge], k: int, rng) -> list[Edge]:
    """The springs plus k chords drawn at random among the pairs not yet sprung."""
    have = {tuple(sorted(e)) for e in edges}
    others = [(i, j) for i in range(len(P)) for j in range(i + 1, len(P)) if (i, j) not in have]
    pick = rng.choice(len(others), size=min(k, len(others)), replace=False) if k else []
    return list(edges) + [others[i] for i in pick]


def axes(P: np.ndarray, edges: list[Edge]) -> np.ndarray:
    """Unit axis of each spring, from u to v. Shape (m, 2)."""
    D = np.array([P[v] - P[u] for u, v in edges], float).reshape(-1, 2)
    return D / np.linalg.norm(D, axis=1, keepdims=True)


def projectors(P: np.ndarray, edges: list[Edge]) -> np.ndarray:
    """Each spring's restriction map as a projector n_e n_e^T. Shape (m, 2, 2)."""
    n = axes(P, edges)
    return np.einsum("ei,ej->eij", n, n)


def rigidity_matrix(P: np.ndarray, edges: list[Edge], gates=None) -> np.ndarray:
    """R, one row per live spring: -n_e^T at u, +n_e^T at v (live springs x 2n)."""
    g = np.ones(len(edges)) if gates is None else np.asarray(gates, float)
    n = axes(P, edges)
    R = np.zeros((len(edges), 2 * len(P)))
    for e, (u, v) in enumerate(edges):
        R[e, 2 * u:2 * u + 2], R[e, 2 * v:2 * v + 2] = -n[e], n[e]
    return R[g > 0]


def sheaf_laplacian(P: np.ndarray, edges: list[Edge], gates=None) -> np.ndarray:
    """L_F = sum_e g_e (b_e b_e^T) (x) n_e n_e^T = R^T diag(g) R, acting on positions ordered body (x) direction."""
    g = np.ones(len(edges)) if gates is None else np.asarray(gates, float)
    R = rigidity_matrix(P, edges)
    return R.T @ (g[:, None] * R)


def rigid_motions(P: np.ndarray) -> np.ndarray:
    """The plane's rigid motions as columns: two translations and the rotation about the centroid. Shape (2n, 3)."""
    n = len(P)
    c = P - P.mean(0)
    T = np.zeros((2 * n, 3))
    T[0::2, 0], T[1::2, 1] = 1.0, 1.0
    T[0::2, 2], T[1::2, 2] = -c[:, 1], c[:, 0]
    return T


def zero_modes(P: np.ndarray, edges: list[Edge], gates=None) -> int:
    """Motions no live spring resists: 2n - rank(R). At least the 3 rigid motions (the springs never feel them)."""
    R = rigidity_matrix(P, edges, gates)
    return 2 * len(P) - (np.linalg.matrix_rank(R) if len(R) else 0)


def mechanisms(P: np.ndarray, edges: list[Edge], gates=None) -> np.ndarray:
    """Floppy mechanisms: zero modes that are not rigid motions, as orthonormal columns (2n x count)."""
    R = rigidity_matrix(P, edges, gates)
    Z = null_space(R) if len(R) else np.eye(2 * len(P))
    T = rigid_motions(P)
    Q, _ = np.linalg.qr(T)
    Z = Z - Q @ (Q.T @ Z)
    if Z.shape[1] == 0:
        return Z
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    return U[:, s > 1e-9]


# --------------------------------------------------------------------------------------------------- the world

def _coupling(n: int, edges: list[Edge], gates, maps: np.ndarray, phase_map: np.ndarray) -> np.ndarray:
    """sum_e g_e (b_e b_e^T) (x) phase_map (x) maps[e], with state ordered body (x) phase (x) direction."""
    g = np.ones(len(edges)) if gates is None else np.asarray(gates, float)
    C = np.zeros((4 * n, 4 * n))
    for e, (u, v) in enumerate(edges):
        if g[e] == 0:
            continue
        block = g[e] * np.kron(phase_map, maps[e])
        for a, b, s in ((u, u, 1.0), (v, v, 1.0), (u, v, -1.0), (v, u, -1.0)):
            C[4 * a:4 * a + 4, 4 * b:4 * b + 4] += s * block
    return C


def generator(world: World, P: np.ndarray, edges: list[Edge], gates=None, oriented: bool = True) -> np.ndarray:
    """A = I_n (x) A0 (x) I_2 - kappa sum_e g_e (b b^T) (x) E (x) (n n^T, or I_2 when not oriented)."""
    if world.d != 2:
        raise ValueError("sheaf worlds are spherical: world.d must be 2")
    n = len(P)
    maps = projectors(P, edges) if oriented else np.repeat(np.eye(2)[None], len(edges), 0)
    return np.kron(np.eye(n), world.A0) - world.kappa * _coupling(n, edges, gates, maps, E2)


def exact_step(world: World, P: np.ndarray, edges: list[Edge], gates=None, oriented: bool = True) -> np.ndarray:
    """The exact small-swing step e^{A dt}."""
    return expm(generator(world, P, edges, gates, oriented) * world.dt)


def simulate_exact(world: World, P: np.ndarray, edges: list[Edge], schedule: GateSchedule, x0: np.ndarray,
                   oriented: bool = True) -> np.ndarray:
    """Small-swing world, exact, switching gates between schedule segments. x0: (ntraj, 4n) -> (ntraj, T + 1, 4n)."""
    x0 = np.atleast_2d(x0)
    T = sum(steps for steps, _ in schedule)
    X = np.empty((x0.shape[0], T + 1, x0.shape[1]))
    X[:, 0] = x0
    t = 0
    for steps, gates in schedule:
        Phi = exact_step(world, P, edges, gates, oriented)
        for _ in range(steps):
            X[:, t + 1] = X[:, t] @ Phi.T
            t += 1
    return X


def mode_frequencies(world: World, P: np.ndarray, edges: list[Edge], gates=None) -> np.ndarray:
    """omega_k = sqrt(omega0^2 + kappa mu_k), one per eigenvalue mu_k of L_F (2n of them)."""
    mu = np.clip(np.linalg.eigvalsh(sheaf_laplacian(P, edges, gates)), 0.0, None)
    return np.sqrt(world.omega0**2 + world.kappa * mu)


# --------------------------------------------------------------------------------------------------- the model

@dataclass
class SheafModel:
    """A 4 x 4 body map K and one 2 x 2 spring map M, placed along each spring's axis by its projector."""

    K: np.ndarray
    M: np.ndarray

    @classmethod
    def from_planar(cls, planar: OneHop) -> "SheafModel":
        """Step 1's planar maps, unchanged: K2 (x) I_2 for each body, M for each spring."""
        return cls(np.kron(planar.K, np.eye(2)), planar.M.copy())

    def coupling(self, P: np.ndarray, edges: list[Edge], gates=None, maps: np.ndarray | None = None) -> np.ndarray:
        """sum_e g_e (b_e b_e^T) (x) M (x) n_e n_e^T (maps overrides the projectors, e.g. learned ones)."""
        maps = projectors(P, edges) if maps is None else maps
        return _coupling(len(P), edges, gates, maps, self.M)

    def assemble(self, P: np.ndarray, edges: list[Edge], gates=None, maps: np.ndarray | None = None) -> np.ndarray:
        """The whole network's step: I_n (x) K - coupling."""
        return np.kron(np.eye(len(P)), self.K) - self.coupling(P, edges, gates, maps)

    def mode_blocks(self, P: np.ndarray, edges: list[Edge], gates=None):
        """For an isotropic body (K = K2 (x) I_2): eigenvalues mu_k of L_F and the 2 x 2 blocks K2 - mu_k M."""
        K2 = self.K[0::2, 0::2]
        mu = np.clip(np.linalg.eigvalsh(sheaf_laplacian(P, edges, gates)), 0.0, None)
        return mu, [K2 - m * self.M for m in mu]

    def rollout(self, x0: np.ndarray, P: np.ndarray, edges: list[Edge], schedule: GateSchedule,
                maps: np.ndarray | None = None) -> np.ndarray:
        """Free-running prediction from x0 under a schedule of (steps, gates) segments."""
        x0 = np.atleast_2d(x0)
        T = sum(steps for steps, _ in schedule)
        X = np.empty((x0.shape[0], T + 1, x0.shape[1]))
        X[:, 0] = x0
        t = 0
        for steps, gates in schedule:
            S = self.assemble(P, edges, gates, maps)
            for _ in range(steps):
                X[:, t + 1] = X[:, t] @ S.T
                t += 1
        return X


SheafSet = tuple[np.ndarray, np.ndarray, list[Edge], np.ndarray]  # (X (ntraj, T + 1, 4n), P, edges, gates)


def _spring_reads(X: np.ndarray, P: np.ndarray, edges: list[Edge], gates) -> np.ndarray:
    """Q_v = sum over the springs at v of g_e n_e n_e^T (x_v - x_u), per phase. Returns (N, n, 2 phases, 2 dirs)."""
    n = len(P)
    xs = X[:, :-1].reshape(-1, n, 2, 2)
    Pe = projectors(P, edges)
    g = np.ones(len(edges)) if gates is None else np.asarray(gates, float)
    Q = np.zeros_like(xs)
    for e, (u, v) in enumerate(edges):
        if g[e] == 0:
            continue
        d = xs[:, v] - xs[:, u]
        Q[:, v] += g[e] * d @ Pe[e].T
        Q[:, u] -= g[e] * d @ Pe[e].T
    return Q


def fit_sheaf(sets: list[SheafSet], isotropic_body: bool = False):
    """K and the shared spring map M by one least-squares solve over every body, step and layout.

    y_(p,i) = sum_b K_(p,i),b x_b - sum_q M_pq Q_(q,i), with Q_v the spring reads at body v. Free K: 16 + 4 numbers;
    isotropic_body: K = K2 (x) I_2, 4 + 4 numbers. Returns (SheafModel, info).
    """
    rows = {p: [] for p in (0, 1)}
    rhs = {p: [] for p in (0, 1)}
    for X, P, edges, gates in sets:
        n = len(P)
        xs = X[:, :-1].reshape(-1, n, 2, 2).reshape(-1, 2, 2)  # (N, phase, dir)
        ys = X[:, 1:].reshape(-1, n, 2, 2).reshape(-1, 2, 2)
        Q = _spring_reads(X, P, edges, gates).reshape(-1, 2, 2)
        for p in (0, 1):
            for i in (0, 1):
                if isotropic_body:
                    feats = np.hstack([xs[:, :, i], -Q[:, :, i]])  # K2_p0, K2_p1, M_p0, M_p1
                else:
                    Kc = np.zeros((len(xs), 8))
                    Kc[:, 4 * i:4 * i + 4] = xs.reshape(-1, 4)  # row (p, i) of K, against the whole body state
                    feats = np.hstack([Kc, -Q[:, :, i]])
                rows[p].append(feats)
                rhs[p].append(ys[:, p, i])
    K = np.zeros((4, 4))
    M = np.zeros((2, 2))
    samples = 0
    for p in (0, 1):
        A, b = np.vstack(rows[p]), np.concatenate(rhs[p])
        w = np.linalg.lstsq(A, b, rcond=None)[0]
        samples = len(A)
        if isotropic_body:
            for i in (0, 1):
                K[2 * p + i, i], K[2 * p + i, 2 + i] = w[0], w[1]
            M[p] = w[2:]
        else:
            K[2 * p + 0], K[2 * p + 1] = w[0:4], w[4:8]
            M[p] = w[8:]
    return SheafModel(K, M), {"samples": samples}


def design_steps(X: np.ndarray, model: SheafModel, P: np.ndarray, pairs: list[Edge], maps: np.ndarray | None = None):
    """Per-step design blocks and residuals for gate detection along known axes. X: (T + 1, 4n).

    r_v = x_v(t+1) - K x_v(t) = sum_(a,b) g_ab G_ab with G_ab = -(M (x) n n^T)(x_b - x_a) at b and the opposite at a.
    maps overrides the projectors n n^T (for axes learned rather than given). Returns G (T, 4n, m) and r (T, 4n), as
    `detect.design_steps` does for the plane.
    """
    n = len(P)
    xs, ys = X[:-1].reshape(-1, n, 4), X[1:].reshape(-1, n, 4)
    r = ys - xs @ model.K.T
    Pe = projectors(P, pairs) if maps is None else maps
    G = np.zeros((len(xs), n, 4, len(pairs)))
    for k, (a, b) in enumerate(pairs):
        dv = (xs[:, b] - xs[:, a]) @ np.kron(model.M, Pe[k]).T
        G[:, b, :, k], G[:, a, :, k] = -dv, dv
    return G.reshape(len(xs), 4 * n, len(pairs)), r.reshape(len(xs), 4 * n)
