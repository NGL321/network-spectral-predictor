"""E0: calibrate the simulator.

Small swings: velocity Verlet (nominal) against the exact step e^(A dt) (actual) on a path of 3, for 100 s.
Large swings have no exact step, so Verlet at `substeps` (nominal) is checked against Verlet at 4x the
substeps (actual), and energy drift is measured in both regimes.
"""

import numpy as np
from common import main

from nsp import config, graphs, metrics, world as W


def run(cfg: dict, seed: int):
    p = cfg["e0"]
    w = config.world_from(cfg)
    rng = np.random.default_rng(seed)
    n, edges = p["n"], graphs.named(p["graph"], p["n"])
    L = graphs.laplacian(n, edges)
    sched = [(p["samples"], L)]

    x0 = W.initial_states(w, n, p["trajectories"], p["amplitude"], rng)
    X_exact = W.simulate_exact(w, sched, x0)
    X_verlet = W.simulate_verlet(w, sched, x0, nonlinear=False, substeps=p["substeps"])
    curve = metrics.rollout_curve(X_exact, X_verlet)
    H = W.energy(w, X_verlet, L)

    xb = W.initial_states(w, n, p["trajectories"], p["large_amplitude"], rng, kind="uniform")
    Xb = W.simulate_verlet(w, sched, xb, nonlinear=True, substeps=p["substeps"])
    Xb_ref = W.simulate_verlet(w, sched, xb, nonlinear=True, substeps=p["reference_substeps"])
    curve_big = metrics.rollout_curve(Xb_ref, Xb)
    Hb = W.energy(w, Xb, L, nonlinear=True)

    lams, U = np.linalg.eigh(L)
    s10 = int(round(10 / w.dt))
    m = {
        "duration_s": p["samples"] * w.dt,
        "one_sample_error": curve[1], "error_10s": curve[s10], "error_end": curve[-1],
        "energy_drift_small": metrics.energy_drift(H),
        "error_10s_large": curve_big[s10], "error_end_large": curve_big[-1],
        "energy_drift_large": metrics.energy_drift(Hb),
        "lambdas": np.clip(lams, 0, None), "mode_frequencies": W.mode_frequencies(w, L),
    }
    k = p["player_steps"] + 1
    arrays = {
        "theta_exact": X_exact[0, :k, 0::2], "theta_verlet": X_verlet[0, :k, 0::2],
        "theta_large_ref": Xb_ref[0, :k, 0::2], "theta_large": Xb[0, :k, 0::2],
        "curve": curve, "curve_large": curve_big,
        "energy_small": H / H[:, :1] - 1, "energy_large": Hb / Hb[:, :1] - 1, "modes": U,
    }
    return m, arrays


if __name__ == "__main__":
    main("e0", run)
