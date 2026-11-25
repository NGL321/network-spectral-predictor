"""E2: transfer across body counts.

Fit on paths and rings of 3 and 5 bodies (random gate patterns); predict 8-body paths, rings, stars and complete
graphs with no refit, for kappa = 1, 4, 16. Reference: a model fitted on each 8-body graph itself.
"""

import numpy as np
from common import main

from nsp import config, fit, graphs, metrics, world as W


def run(cfg: dict, seed: int):
    p = cfg["e2"]
    base = config.world_from(cfg)
    amp, T, n8 = p["amplitude"], p["test_steps"], p["test_n"]
    lam_grid = np.linspace(0, p["lambda_grid_max"], 91)
    out, arrays = {"kappas": p["kappas"], "graphs": p["test_graphs"], "by_kappa": {}}, {"lambda_grid": lam_grid}
    for kappa in p["kappas"]:
        w = base.with_(kappa=kappa)
        rng = np.random.default_rng(seed)
        train, lam_train = [], []
        for n in p["train_sizes"]:
            for name in p["train_graphs"]:
                edges = graphs.named(name, n)
                for g in graphs.random_gate_patterns(len(edges), p["train_patterns"], rng):
                    L = graphs.laplacian(n, edges, g)
                    lam_train += list(graphs.graph_frequencies(n, edges, g))
                    x0 = W.initial_states(w, n, p["train_trajectories"], amp, rng)
                    train.append((W.simulate_exact(w, [(p["train_steps"], L)], x0), L))
        model_, info = fit.fit_joint(train)
        w_nom, mod = model_.mode_frequencies(lam_grid, w.dt)
        w_act = np.sqrt(w.omega0**2 + kappa * lam_grid)
        arrays[f"freq_err_k{kappa:g}"] = (w_nom - w_act) / w_act
        arrays[f"growth_k{kappa:g}"] = mod - 1
        res = {"lambda_train_max": max(lam_train), "lambda_train": sorted(set(np.round(lam_train, 4))),
               "K": model_.K, "M": model_.M, "graphs": {}}
        for name in p["test_graphs"]:
            edges = graphs.named(name, n8)
            L = graphs.laplacian(n8, edges)
            r_test = np.random.default_rng(50_000 + seed)
            x0 = W.initial_states(w, n8, p["test_trajectories"], amp, r_test)
            Xa, Xn = W.simulate_exact(w, [(T, L)], x0), model_.rollout(x0, [(T, L)])
            curve = metrics.rollout_curve(Xa, Xn)
            X8 = W.simulate_exact(w, [(T, L)], W.initial_states(w, n8, p["test_trajectories"], amp,
                                                                    np.random.default_rng(60_000 + seed)))
            own, _ = fit.fit_joint([(X8, L)])
            curve_own = metrics.rollout_curve(Xa, own.rollout(x0, [(T, L)]))
            modes = metrics.mode_report(model_, L, w)
            ratio = curve[T] / curve_own[T] if curve_own[T] > 1e-9 else float("inf")
            res["graphs"][name] = {
                "lambda_max": float(graphs.graph_frequencies(n8, edges).max()),
                "lambdas": graphs.graph_frequencies(n8, edges),
                "one_step": metrics.one_step_error(Xa, model_.assemble(L)),
                "rollout": metrics.at_horizons(curve, p["horizons"]),
                "rollout_own": metrics.at_horizons(curve_own, p["horizons"]),
                "ratio": ratio, "freq_err_max": float(np.abs(modes["frequency_error"]).max()),
                "growth_max": float(modes["growth"].max()),
            }
            arrays[f"curve_k{kappa:g}_{name}"] = curve
            arrays[f"curve_own_k{kappa:g}_{name}"] = curve_own
            arrays[f"theta_act_k{kappa:g}_{name}"] = Xa[0, :, 0::2]
            arrays[f"theta_nom_k{kappa:g}_{name}"] = Xn[0, :, 0::2]
        out["by_kappa"][f"{kappa:g}"] = res
    return out, arrays


if __name__ == "__main__":
    main("e2", run)
