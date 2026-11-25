"""E3: reconfigure by gating.

Fit on half of the 32 gate patterns of a ring of 5; predict the other half with no refit. Then cut one spring
halfway through a run: the model swaps L(g) at the same moment.
"""

import numpy as np
from common import main

from nsp import config, fit, graphs, metrics, world as W


def run(cfg: dict, seed: int):
    p = cfg["e3"]
    base = config.world_from(cfg)
    n, edges, amp, T = p["n"], graphs.named(p["graph"], p["n"]), p["amplitude"], p["steps"]
    pats = graphs.all_gate_patterns(len(edges))
    out, arrays = {"kappas": p["kappas"], "by_kappa": {}}, {"patterns": pats}
    for kappa in p["kappas"]:
        w = base.with_(kappa=kappa)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(pats))
        seen = np.zeros(len(pats), bool)
        seen[order[: p["seen"]]] = True
        train = []
        for k in np.flatnonzero(seen):
            L = graphs.laplacian(n, edges, pats[k])
            train.append((W.simulate_exact(w, [(T, L)], W.initial_states(w, n, p["trajectories"], amp, rng)), L))
        model_, _ = fit.fit_joint(train)
        one, roll = np.zeros(len(pats)), np.zeros(len(pats))
        r_test = np.random.default_rng(70_000 + seed)
        for k, g in enumerate(pats):
            L = graphs.laplacian(n, edges, g)
            x0 = W.initial_states(w, n, p["test_trajectories"], amp, r_test)
            Xa = W.simulate_exact(w, [(T, L)], x0)
            one[k] = metrics.one_step_error(Xa, model_.assemble(L))
            roll[k] = metrics.rollout_curve(Xa, model_.rollout(x0, [(T, L)]))[T]

        g_off = np.ones(len(edges))
        g_off[p["switch_edge"]] = 0
        L_on, L_off = graphs.laplacian(n, edges), graphs.laplacian(n, edges, g_off)
        s = p["switch_step"]
        sched = [(s, L_on), (T - s, L_off)]
        x0 = W.initial_states(w, n, p["test_trajectories"], amp, r_test)
        Xa, Xn = W.simulate_exact(w, sched, x0), model_.rollout(x0, sched)
        curve = metrics.rollout_curve(Xa, Xn)
        H_before, H_after = W.energy(w, Xa[:, s], L_on), W.energy(w, Xa[:, s], L_off)
        jump = (H_after - H_before) / H_before
        out["by_kappa"][f"{kappa:g}"] = {
            "seen_one_step": one[seen].mean(), "held_one_step": one[~seen].mean(),
            "seen_rollout": roll[seen].mean(), "held_rollout": roll[~seen].mean(),
            "switch": {"before": curve[s], "just_after": curve[s + 1], "mid_after": curve[s + (T - s) // 2],
                       "end": curve[T], "pre_switch_mean": curve[1:s + 1].mean(),
                       "post_switch_mean": curve[s + 1:].mean()},
            "energy_jump_mean": jump.mean(), "energy_jump_min": jump.min(), "energy_jump_max": jump.max(),
        }
        gates_t = np.vstack([np.tile(np.ones(len(edges)), (s + 1, 1)), np.tile(g_off, (T - s, 1))])
        arrays.update({f"seen_k{kappa:g}": seen, f"one_k{kappa:g}": one, f"roll_k{kappa:g}": roll,
                       f"switch_curve_k{kappa:g}": curve, f"switch_act_k{kappa:g}": Xa[0, :, 0::2],
                       f"switch_nom_k{kappa:g}": Xn[0, :, 0::2], f"switch_gates_k{kappa:g}": gates_t,
                       f"energy_k{kappa:g}": np.r_[W.energy(w, Xa[0, : s + 1], L_on),
                                                   W.energy(w, Xa[0, s + 1:], L_off)]})
    return out, arrays


if __name__ == "__main__":
    main("e3", run)
