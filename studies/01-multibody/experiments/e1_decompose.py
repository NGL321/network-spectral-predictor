"""E1: decompose. Do K and M come out as the body and the spring?

Fit jointly on every gate pattern of a path of 3. Compare with the one-hop series (K0, M0), the frozen-bodies
fit and gradient descent; read omega0^2 and kappa back; predict a lone pendulum with K alone and the path of 3
with the assembled model. Then the data-efficiency check on a ring of 8: nsp (8 numbers) against one global map.
"""

import numpy as np
from common import main

from nsp import baselines, config, fit, graphs, metrics, model, world as W


def run(cfg: dict, seed: int):
    p = cfg["e1"]
    w = config.world_from(cfg)
    rng = np.random.default_rng(seed)
    n, edges = p["n"], graphs.named(p["graph"], p["n"])
    amp, T = p["amplitude"], p["test_steps"]

    train = []
    for g in graphs.all_gate_patterns(len(edges)):
        L = graphs.laplacian(n, edges, g)
        x0 = W.initial_states(w, n, p["train_trajectories"], amp, rng)
        train.append((W.simulate_exact(w, [(p["train_steps"], L)], x0), L))
    joint, info = fit.fit_joint(train)
    ser = model.series(w)
    lone_X = W.simulate_exact(w, [(p["train_steps"], np.zeros((n, n)))],
                              W.initial_states(w, n, p["train_trajectories"], amp, rng))
    frozen = fit.fit_frozen(lone_X, train)
    gd, gd_info = fit.gradient_descent(train, tol=p["gd_tolerance"])
    params = fit.recover_parameters(joint, w)

    L = graphs.laplacian(n, edges)
    x0 = W.initial_states(w, n, p["test_trajectories"], amp, rng)
    Xa, Xn = W.simulate_exact(w, [(T, L)], x0), joint.rollout(x0, [(T, L)])
    curve = metrics.rollout_curve(Xa, Xn)
    Z = np.zeros((1, 1))
    x1 = W.initial_states(w, 1, p["test_trajectories"], amp, rng)
    Xa1, Xn1 = W.simulate_exact(w, [(T, Z)], x1), joint.rollout(x1, [(T, Z)])
    curve1 = metrics.rollout_curve(Xa1, Xn1)
    modes = metrics.mode_report(joint, L, w)

    ns = p["scarcity_n"]
    Ls = graphs.laplacian(ns, graphs.named(p["scarcity_graph"], ns))
    test = W.simulate_exact(w, [(T, Ls)], W.initial_states(w, ns, 10, amp, np.random.default_rng(10_000 + seed)))
    steps, noise = p["scarcity_steps"], p["noise"]
    eff = {k: np.full((len(steps), len(noise)), np.nan) for k in ("nsp_one", "nsp_roll", "global_one", "global_roll")}
    for i, s in enumerate(steps):
        for j, sig in enumerate(noise):
            r = np.random.default_rng(seed * 1000 + 10 * i + j)
            X = W.add_noise(w, W.simulate_exact(w, [(s, Ls)], W.initial_states(w, ns, 1, amp, r)), sig, amp, r)
            m_nsp, _ = fit.fit_joint([(X, Ls)])
            P_glob = fit.fit_global(X)
            for name, P in (("nsp", m_nsp.assemble(Ls)), ("global", P_glob)):
                eff[f"{name}_one"][i, j] = metrics.one_step_error(test, P)
                roll = baselines.rollout_matrix(P, test[:, 0], T)
                with np.errstate(all="ignore"):
                    eff[f"{name}_roll"][i, j] = min(metrics.relative(roll[:, T], test[:, T]), 1e6)

    m = {
        "K": joint.K, "M": joint.M, "K0": ser.K, "M0": ser.M, "K_frozen": frozen.K, "M_frozen": frozen.M,
        "K_gd": gd.K, "M_gd": gd.M,
        "dK": np.abs(joint.K - ser.K).max(), "dM": np.abs(joint.M - ser.M).max(),
        "dK_frozen": np.abs(frozen.K - ser.K).max(), "dM_frozen_joint": np.abs(frozen.M - joint.M).max(),
        "gd_iterations": gd_info["iterations"], "condition": info["condition"], "samples": info["samples"],
        "omega0_sq": params["omega0_sq"], "kappa": params["kappa"], "kappa_bare": params["kappa_bare"],
        "true_omega0_sq": w.omega0**2, "true_kappa": w.kappa,
        "path3": metrics.at_horizons(curve, [10, 50, 200]), "lone": metrics.at_horizons(curve1, [10, 50, 200]),
        "one_step_path3": metrics.one_step_error(Xa, joint.assemble(L)),
        "modes": modes, "scarcity_steps": steps, "noise": noise,
    }
    arrays = {"theta_act": Xa[0, :, 0::2], "theta_nom": Xn[0, :, 0::2],
              "theta_act_lone": Xa1[:3, :, 0], "theta_nom_lone": Xn1[:3, :, 0],
              "curve": curve, "curve_lone": curve1,
              "gd_iters": gd_info["iters"], "gd_distance": gd_info["distance"], **eff}
    return m, arrays


if __name__ == "__main__":
    main("e1", run)
