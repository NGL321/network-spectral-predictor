"""E4: large swings (rung 2). Which lift holds when sin(theta) matters?

Path of 3, released from rest at 0.1, 0.5, 1 and 2 rad; data from velocity Verlet with sin(theta).
Lifts: x = (theta, theta'), + sin(theta), + theta' cos(theta). The springs still read base-state differences.
"""

import numpy as np
from common import main

from nsp import config, fit, graphs, metrics, world as W


def run(cfg: dict, seed: int):
    p = cfg["e4"]
    w = config.world_from(cfg)
    n, edges, T = p["n"], graphs.named(p["graph"], p["n"]), p["steps"]
    L = graphs.laplacian(n, edges)
    out = {"amplitudes": p["amplitudes"], "lifts": p["lifts"], "by_amplitude": {}}
    arrays = {}
    for a in p["amplitudes"]:
        rng = np.random.default_rng(seed * 100 + int(round(10 * a)))
        sim = lambda k: W.simulate_verlet(w, [(T, L)], W.initial_states(w, n, k, a, rng, kind="uniform"),
                                          nonlinear=True, substeps=p["substeps"])
        Xtr, Xte = sim(p["train_trajectories"]), sim(p["test_trajectories"])
        res = {"frequency_ratio": W.swing_frequency(w, a) / w.omega0, "lifts": {}}
        for lift in p["lifts"]:
            m = fit.fit_lifted([(Xtr, L)], lift)
            Xn = m.rollout(Xte[:, 0], L, T)
            curve = metrics.rollout_curve(Xte, Xn)
            res["lifts"][lift] = {"one_step": metrics.relative(m.predict(Xte, L), Xte[:, 1:]),
                                  "rollout": metrics.at_horizons(curve, [10, 50, 200]), "features": m.W.shape[0]}
            arrays[f"curve_a{a:g}_{lift}"] = curve
            if a == p["player_amplitude"]:
                arrays[f"theta_nom_{lift}"] = Xn[0, :, 0::2]
                arrays[f"phase_nom_{lift}"] = Xn[0, :, :2]
        if a == p["player_amplitude"]:
            arrays["theta_act"] = Xte[0, :, 0::2]
            arrays["phase_act"] = Xte[0, :, :2]
        out["by_amplitude"][f"{a:g}"] = res
    return out, arrays


if __name__ == "__main__":
    main("e4", run)
