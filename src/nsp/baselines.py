"""Baselines that bracket the model.

- The exact step e^{A dt}: the ceiling, nothing learned.
- The one-hop series K0, M0: closed form, nothing learned (`nsp.model.series`).
- One global linear map per configuration (global DMD): 4 d^2 n^2 numbers, no sharing, refit for every
  configuration (`nsp.fit.fit_global`).
"""

from __future__ import annotations

import numpy as np

from .fit import fit_global
from .model import series
from .world import exact_step

__all__ = ["exact_step", "series", "fit_global", "rollout_matrix"]


def rollout_matrix(P: np.ndarray, x0: np.ndarray, steps: int) -> np.ndarray:
    """Free-running rollout of any fixed linear step P."""
    x0 = np.atleast_2d(x0)
    X = np.empty((x0.shape[0], steps + 1, x0.shape[1]))
    X[:, 0] = x0
    for t in range(steps):
        X[:, t + 1] = X[:, t] @ P.T
    return X
