"""E5: detect the springs (step 1b). Can K and M find which springs are live?

Ring of 5 with random gates; every pair of the 5 bodies is a candidate (10 pairs, 5 of them never wired).
Detection by plain least squares and by non-negative least squares, from 1 or 3 trajectories, across window
lengths and noise levels, for kappa = 1 and 16. K and M are the one-hop series (a fitted model matches it to 1e-4).
"""

import numpy as np
from common import main

from nsp import config, detect, graphs, metrics, model, world as W


def run(cfg: dict, seed: int):
    p = cfg["e5"]
    base = config.world_from(cfg)
    n, ring, amp = p["n"], graphs.named(p["graph"], p["n"]), p["amplitude"]
    pairs = graphs.complete(n)
    rows = []
    for kappa in p["kappas"]:
        w = base.with_(kappa=kappa)
        m = model.series(w)
        for win in p["windows"]:
            for sig in p["noise"]:
                for k in p["trajectories"]:
                    rng = np.random.default_rng([seed, int(kappa), win, int(1000 * sig), k])
                    scores = {mt: [] for mt in p["methods"]}
                    exact = {mt: 0 for mt in p["methods"]}
                    labels = []
                    for _ in range(p["trials"]):
                        g = (rng.random(len(ring)) < p["p_on"]).astype(float)
                        L = graphs.laplacian(n, ring, g)
                        X = W.add_noise(w, W.simulate_exact(w, [(win, L)], W.initial_states(w, n, k, amp, rng)),
                                        sig, amp, rng)
                        truth = detect.truth_vector(pairs, ring, g)
                        labels += list(truth)
                        for mt in p["methods"]:
                            gh = detect.detect_gates(X, m, pairs, n, method=mt)
                            scores[mt] += list(gh)
                            exact[mt] += detect.exact_match(gh, truth)
                    y = np.array(labels)
                    for mt in p["methods"]:
                        s = np.array(scores[mt])
                        rows.append({"kappa": kappa, "window": win, "noise": sig, "trajectories": k, "method": mt,
                                     "auroc": metrics.auroc(s, y), "exact": exact[mt] / p["trials"],
                                     "max_spurious": float(np.abs(s[y < 0.5]).max())})

    w = base
    m = model.series(w)
    g_one = np.array([0, 0, 0, 1, 0.0])
    L1 = graphs.laplacian(n, ring, g_one)
    rb = np.random.default_rng(1000 + seed)
    blind = {}
    for k in (1, 3):
        X = W.simulate_exact(w, [(200, L1)], W.initial_states(w, n, k, amp, rb))
        blind[k] = detect.design_singular_values(X, m, pairs, n)
    re = np.random.default_rng(2000 + seed)
    Xd = W.add_noise(w, W.simulate_exact(w, [(50, L1)], W.initial_states(w, n, 1, amp, re)), 0.01, amp, re)
    gh = {mt: detect.detect_gates(Xd, m, pairs, n, method=mt) for mt in ("lstsq", "nnls")}
    truth = detect.truth_vector(pairs, ring, g_one)
    x0 = W.initial_states(w, n, 1, amp, re)
    Xa = W.simulate_exact(w, [(200, L1)], x0)
    arrays = {"blind_sv_1": blind[1], "blind_sv_3": blind[3], "truth": truth, "theta_act": Xa[0, :, 0::2]}
    for mt, g in gh.items():
        arrays[f"g_{mt}"] = g
        arrays[f"theta_nom_{mt}"] = m.rollout(x0, [(200, graphs.laplacian(n, pairs, g))])[0, :, 0::2]
    return {"rows": rows, "pairs": pairs, "ring": ring, "example_window": 50, "example_noise": 0.01}, arrays


if __name__ == "__main__":
    main("e5", run)
