"""Learning: one least-squares solve over every body, step and configuration.

Each sample i is one (configuration, trajectory, step, body):

    y_i = x_v(t+1) = [K, -M] [x_v(t); s_v(t)] = W phi_i,   s_v(t) = sum_u g_uv (x_v(t) - x_u(t))
    W* = (sum_i y_i phi_i^T) (sum_i phi_i phi_i^T)^{-1}

The solve is (4d^2)-sized whatever the number of bodies. Gradient descent on the same energy is local
(dE/dK = -2/N sum eps x^T, dE/dM = +2/N sum eps s^T) and ends at W*.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import logm

from . import lifts as _lifts
from .model import LiftedOneHop, OneHop
from .world import World

Dataset = tuple[np.ndarray, np.ndarray]  # (X of shape (ntraj, T + 1, n * 2d), L)


def body_samples(X: np.ndarray, L: np.ndarray, d: int = 1):
    """Per-body samples x_v(t), s_v(t) and y_v = x_v(t+1), each of shape (N, 2d)."""
    D = 2 * d
    S = X @ np.kron(L, np.eye(D)).T
    return X[:, :-1].reshape(-1, D), S[:, :-1].reshape(-1, D), X[:, 1:].reshape(-1, D)


def _stack(datasets: list[Dataset], d: int):
    xs, ss, ys = zip(*(body_samples(X, L, d) for X, L in datasets))
    return np.vstack(xs), np.vstack(ss), np.vstack(ys)


def _condition(P: np.ndarray) -> float:
    ev = np.linalg.eigvalsh(P.T @ P / len(P))
    return float(ev.max() / ev.min())


def fit_joint(datasets: list[Dataset], d: int = 1):
    """K and M together, by one least-squares solve. Returns (OneHop, info)."""
    x, s, y = _stack(datasets, d)
    P = np.hstack([x, s])
    W = np.linalg.lstsq(P, y, rcond=None)[0]
    D = 2 * d
    return OneHop(W[:D].T, -W[D:].T), {"samples": len(P), "condition": _condition(P)}


def fit_frozen(lone_X: np.ndarray, datasets: list[Dataset], d: int = 1) -> OneHop:
    """Decision 3's cross-check: K from lone bodies, then M on coupled data with K held fixed."""
    D = 2 * d
    K = np.linalg.lstsq(lone_X[:, :-1].reshape(-1, D), lone_X[:, 1:].reshape(-1, D), rcond=None)[0].T
    x, s, y = _stack(datasets, d)
    M = -np.linalg.lstsq(s, y - x @ K.T, rcond=None)[0].T
    return OneHop(K, M)


def split_directions(X: np.ndarray, d: int) -> list[np.ndarray]:
    """Split (ntraj, T + 1, n * 2d) states into d planar arrays, one per direction."""
    ntraj, T1, N = X.shape
    n = N // (2 * d)
    Xb = X.reshape(ntraj, T1, n, 2, d)
    return [Xb[..., k].reshape(ntraj, T1, 2 * n) for k in range(d)]


def fit_kronecker(datasets: list[Dataset], d: int):
    """K = K2 (x) I_d and M = M2 (x) I_d: 8 numbers whatever d (stage 1.5a)."""
    planar = [(Xk, L) for X, L in datasets for Xk in split_directions(X, d)]
    m, info = fit_joint(planar, 1)
    return m.per_direction(d), info


def fit_global(X: np.ndarray) -> np.ndarray:
    """One global linear map for one configuration (the global DMD baseline)."""
    N = X.shape[-1]
    return np.linalg.lstsq(X[:, :-1].reshape(-1, N), X[:, 1:].reshape(-1, N), rcond=None)[0].T


def gradient_descent(datasets: list[Dataset], d: int = 1, tol: float = 1e-6, max_iter: int = 200_000):
    """Plain gradient descent from zero, step 1/lambda_max; records the distance to the closed form."""
    x, s, y = _stack(datasets, d)
    P = np.hstack([x, s])
    N = len(P)
    H, b = P.T @ P / N, P.T @ y / N
    W_star = np.linalg.solve(H, b)
    eta = 1.0 / np.linalg.eigvalsh(H).max()
    W = np.zeros_like(W_star)
    norm = np.linalg.norm(W_star)
    marks = set(np.unique(np.geomspace(1, max_iter, 200).astype(int)))
    its, dist = [0], [1.0]
    it = 0
    for it in range(1, max_iter + 1):
        W -= eta * (H @ W - b)
        err = np.linalg.norm(W - W_star) / norm
        if it in marks or err < tol:
            its.append(it)
            dist.append(err)
        if err < tol:
            break
    D = 2 * d
    return OneHop(W[:D].T, -W[D:].T), {"iterations": it, "iters": np.array(its), "distance": np.array(dist),
                                        "condition": _condition(P)}


def recover_parameters(model: OneHop, world: World) -> dict:
    """Read the physics back: omega0^2 = -[log(K) / dt]_21, kappa = M_21 / (dt - omega0^2 dt^3 / 3)."""
    d = model.K.shape[0] // 2
    G = np.real(logm(model.K)) / world.dt
    w2 = -G[d, 0]
    return {"omega0_sq": float(w2),
            "kappa": float(model.M[d, 0] / (world.dt - w2 * world.dt**3 / 3)),
            "kappa_bare": float(model.M[d, 0] / world.dt)}


def fit_lifted(datasets: list[Dataset], lift: str) -> LiftedOneHop:
    """Planar lifted body step (E4): least squares on [psi(x_v), s_v] -> x_v(t+1)."""
    feats, ys = [], []
    for X, L in datasets:
        xv, sv, yv = body_samples(X, L, 1)
        feats.append(np.hstack([_lifts.features(xv, lift), sv]))
        ys.append(yv)
    W = np.linalg.lstsq(np.vstack(feats), np.vstack(ys), rcond=None)[0]
    return LiftedOneHop(W, lift)
