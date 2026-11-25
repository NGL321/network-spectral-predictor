"""Hidden information with a clamped sense (step 2): angles are measured, velocities are not.

Each body's sense is clamped to its measured angle, y_v(t) = theta_v(t) + noise. A body predicts with the step-1 maps
and the spring messages from its neighbours' estimates, then corrects itself with its own clamped sense:

    x-_v(t) = [P(g) x(t-1)]_v,        x_v(t) = x-_v(t) + k (y_v(t) - c x-_v(t)),        c = [1, 0]

The gain k is where predictive-coding relaxation settles. Each body relaxes its state on the energy

    E_v(x) = (y_v - c x)^2 / (2 s^2) + (x - x-_v)^T Pi^-1 (x - x-_v) / 2

(sensory error weighted by the sense's precision 1/s^2, prior error by the prior's precision Pi^-1), whose minimum is
x* = x-_v + Pi c^T (c Pi c^T + s^2)^-1 (y_v - c x-_v). Pi is the lone body's steady-state prior covariance for process
noise Q (how much the body doubts its own prediction), from the discrete Riccati equation.

Stability is checked per graph frequency: the error obeys e(t) = (I - G C) P e(t-1) with G = I (x) k, so by the spectrum
identity it splits into one 2 x 2 block (I - k c)(K - lambda M) per graph frequency lambda.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are

from .model import OneHop
from .world import World

C_ANGLE = np.array([[1.0, 0.0]])  # a body's sense reads its angle


def observe(world: World, X: np.ndarray, rel: float, amplitude: float, rng) -> np.ndarray:
    """Measured angles only: (..., n * 2) states -> (..., n) angles plus noise of rel x amplitude (planar)."""
    if world.d != 1:
        raise NotImplementedError("the clamped sense is planar for now")
    th = X[..., 0::2]
    return th + rng.normal(0.0, rel * amplitude, th.shape) if rel else th.copy()


def process_noise(q: float) -> np.ndarray:
    """Q for one body: unmodelled forces disturb the rate only, with variance q per step ((rad/s)^2)."""
    return np.array([[0.0, 0.0], [0.0, q]])


def lone_gain(K: np.ndarray, s2: float, q: float):
    """Steady-state clamp gain k (2,) and prior covariance Pi (2, 2) for one body with sense noise s2 and rate noise q."""
    Pi = solve_discrete_are(K.T, C_ANGLE.T, process_noise(q), np.array([[s2]]))
    k = (Pi @ C_ANGLE.T / (C_ANGLE @ Pi @ C_ANGLE.T + s2)).ravel()
    return k, Pi


def lone_gains(K: np.ndarray, s2: float, q: float, v0: float, steps: int) -> np.ndarray:
    """Gain per step for one body whose prior precision grows from a cold start: the 2 x 2 Riccati recursion.

    Each body runs it alone (no covariance shared with neighbours), starting from angle variance s2 and rate variance
    v0^2; the gains fall from large (trust the sense) to the steady-state lone gain (trust the prediction).
    Returns (steps + 1, 2); row 0 is unused.
    """
    c = C_ANGLE
    S, Q = np.diag([s2, v0**2]), process_noise(q)
    ks = np.zeros((steps + 1, 2))
    for t in range(1, steps + 1):
        S = K @ S @ K.T + Q
        k = (S @ c.T / (c @ S @ c.T + s2)).ravel()
        S = (np.eye(2) - np.outer(k, c.ravel())) @ S
        ks[t] = k
    return ks


def relax(x_prior: np.ndarray, y: float, Pi: np.ndarray, s2: float, steps: int = 200, eta: float | None = None):
    """Gradient descent on one body's energy E(x), from x = x_prior. Returns the states and energies visited."""
    Pinv = np.linalg.inv(Pi)
    c = C_ANGLE.ravel()
    H = np.outer(c, c) / s2 + Pinv
    eta = 1.0 / np.linalg.eigvalsh(H).max() if eta is None else eta
    x = np.array(x_prior, float)
    xs, Es = [x.copy()], []
    for _ in range(steps):
        e_sense, e_prior = y - c @ x, x - x_prior
        Es.append(0.5 * e_sense**2 / s2 + 0.5 * e_prior @ Pinv @ e_prior)
        x = x - eta * (-c * e_sense / s2 + Pinv @ e_prior)
        xs.append(x.copy())
    e_sense, e_prior = y - c @ x, x - x_prior
    Es.append(0.5 * e_sense**2 / s2 + 0.5 * e_prior @ Pinv @ e_prior)
    return np.array(xs), np.array(Es)


def clamped_filter(model: OneHop, L: np.ndarray, Y: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Local clamped-sense filter for one run. Y: (T + 1, n) measured angles. Returns estimates (T + 1, 2n).

    k is one gain (2,) for every step, or one per step (T + 1, 2) as from `lone_gains`. Cold start: angles from the
    first reading, rates zero.
    """
    P = model.assemble(L)
    n = Y.shape[1]
    ks = np.broadcast_to(np.asarray(k, float), (len(Y), 2)) if np.ndim(k) == 1 else np.asarray(k, float)
    Xh = np.zeros((len(Y), 2 * n))
    Xh[0, 0::2] = Y[0]
    for t in range(1, len(Y)):
        x = P @ Xh[t - 1]
        x += np.tile(ks[t], n) * np.repeat(Y[t] - x[0::2], 2)
        Xh[t] = x
    return Xh


def delay_line(model: OneHop, L: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """The last two readings determine the state: solve theta(t) = A_tt y(t-1) + A_tv v(t-1) for the rate.

    Exact for the linear model on clean data; on noisy data it differentiates the noise. Returns (T + 1, 2n).
    """
    P = model.assemble(L)
    n = Y.shape[1]
    Att, Atv = P[0::2, 0::2], P[0::2, 1::2]
    Xh = np.zeros((len(Y), 2 * n))
    Xh[0, 0::2] = Y[0]
    for t in range(1, len(Y)):
        v = np.linalg.solve(Atv, Y[t] - Att @ Y[t - 1])
        prev = np.empty(2 * n)
        prev[0::2], prev[1::2] = Y[t - 1], v
        Xh[t] = P @ prev
    return Xh


def kalman(model: OneHop, L: np.ndarray, Y: np.ndarray, s2: float, q: float, v0: float) -> np.ndarray:
    """Centralised Kalman filter (every body's covariance shared): the optimal linear reference for the same Q and s2.

    Starts like the clamped filter: angles from the first reading with variance s2, rates zero with variance v0^2.
    """
    P = model.assemble(L)
    n = Y.shape[1]
    Cm = np.kron(np.eye(n), C_ANGLE)
    Q = np.kron(np.eye(n), process_noise(q))
    R = s2 * np.eye(n)
    x = np.zeros(2 * n)
    x[0::2] = Y[0]
    S = np.kron(np.eye(n), np.diag([s2, v0**2]))
    Xh = np.zeros((len(Y), 2 * n))
    Xh[0] = x
    for t in range(1, len(Y)):
        x, S = P @ x, P @ S @ P.T + Q
        Kg = np.linalg.solve(Cm @ S @ Cm.T + R, Cm @ S).T
        x = x + Kg @ (Y[t] - Cm @ x)
        S = (np.eye(2 * n) - Kg @ Cm) @ S
        Xh[t] = x
    return Xh


def mode_radius(model: OneHop, k: np.ndarray, lams) -> np.ndarray:
    """Spectral radius of the clamped error map (I - k c)(K - lambda M), one per graph frequency lambda."""
    A = np.eye(2) - np.outer(k, C_ANGLE.ravel())
    return np.array([np.abs(np.linalg.eigvals(A @ model.mode_block(lam))).max() for lam in np.atleast_1d(lams)])


def halving_steps(rho: float) -> float:
    """Steps for an error shrinking by rho per step to halve."""
    return float(np.log(0.5) / np.log(rho)) if 0 < rho < 1 else float("inf")
