"""E6: transfer by recall. Does recalling a stored structure beat detecting it from scratch?

Ring of 5, 1% noise. A bank of 16 of the 32 gate patterns (the structural vertex) and a binding vertex that
filters a belief over them. Three tests: recall vs detection by window length; re-locking after a mid-run switch;
deciding when a structure is new (likelihood ratio vs an absolute fit test).
"""

import numpy as np
from common import main

from nsp import config, context, detect, graphs, model, world as W


def run(cfg: dict, seed: int):
    p = cfg["e6"]
    w = config.world_from(cfg)
    n, ring, amp, sig = p["n"], graphs.named(p["graph"], p["n"]), p["amplitude"], p["noise"]
    pairs = graphs.complete(n)
    rng = np.random.default_rng(seed)
    pats = graphs.all_gate_patterns(len(ring))
    bank_idx = np.sort(rng.choice(len(pats), p["bank_size"], replace=False))
    outside = np.setdiff1d(np.arange(len(pats)), bank_idx)
    bank = context.ContextBank(n, ring, pats[bank_idx])
    m = model.series(w)
    Ls = bank.laplacians()
    C = len(Ls)

    def traj(segments):
        x0 = W.initial_states(w, n, 1, amp, rng)
        return W.add_noise(w, W.simulate_exact(w, segments, x0), sig, amp, rng)[0]

    scale = context.residual_scale(traj([(2000, Ls[0])])[None], m.assemble(Ls[0]))
    bv = context.BindingVertex(bank, m, scale, hazard=p["hazard"])

    recall_acc, detect_acc = [], []
    for win in p["windows"]:
        ok_r = ok_d = 0
        for _ in range(p["static_trials"]):
            c = int(rng.integers(C))
            X = traj([(win, Ls[c])])
            ok_r += bv.recall(X) == c
            ok_d += detect.exact_match(detect.detect_gates(X[None], m, pairs, n),
                                       detect.truth_vector(pairs, ring, bank.patterns[c]))
        recall_acc.append(ok_r / p["static_trials"])
        detect_acc.append(ok_d / p["static_trials"])

    s, T, Wsl = p["switch_step"], p["total_steps"], p["sliding_window"]
    first_lock, relock_f, relock_d = [], [], []
    err = {"oracle": [], "filter": [], "stale": []}
    example = {}
    for trial in range(p["switch_trials"]):
        a, b = rng.choice(C, 2, replace=False)
        X = traj([(s, Ls[a]), (T - s, Ls[b])])
        pi = bv.filter(X)
        first_lock.append(next((t + 1 for t in range(s) if pi[t, a] > 0.5), np.inf))
        relock_f.append(next((t + 1 - s for t in range(s, T) if pi[t, b] > 0.5), np.inf))
        prev = np.vstack([np.full(C, 1 / C), pi[:-1]]).argmax(1)  # belief before seeing x(t+1)
        for t in range(s, s + 20):
            err["oracle"].append(np.linalg.norm(bv.P[b] @ X[t] - X[t + 1]))
            err["filter"].append(np.linalg.norm(bv.P[prev[t]] @ X[t] - X[t + 1]))
            err["stale"].append(np.linalg.norm(bv.P[a] @ X[t] - X[t + 1]))
        truth_b = detect.truth_vector(pairs, ring, bank.patterns[b])
        hit = np.inf
        for t in range(s, T):
            if detect.exact_match(detect.detect_gates(X[max(0, t + 1 - Wsl): t + 2][None], m, pairs, n), truth_b):
                hit = t + 1 - s
                break
        relock_d.append(hit)
        if trial == 0:
            nominal = np.vstack([X[:1], np.einsum("tij,tj->ti", bv.P[prev], X[:-1])])
            example = {"pi": pi, "a": int(a), "b": int(b), "theta_act": X[:, 0::2], "theta_nom": nominal[:, 0::2],
                       "gates_act": np.vstack([np.tile(bank.patterns[a], (s + 1, 1)),
                                               np.tile(bank.patterns[b], (T - s, 1))]),
                       "gates_nom": np.vstack([bank.patterns[prev[0]], bank.patterns[prev]])}

    novelty = {}
    nov_arrays = {}
    for win in p["novelty_windows"]:
        glr_in, glr_out, abs_in, abs_out = [], [], [], []
        for _ in range(p["novelty_trials"]):
            X = traj([(win, Ls[int(rng.integers(C))])])
            glr_in.append(bv.likelihood_ratio(X, pairs))
            abs_in.append(bv.nll_steps(X).sum(0).min())
            g = pats[outside[int(rng.integers(len(outside)))]]
            X = traj([(win, graphs.laplacian(n, ring, g))])
            glr_out.append(bv.likelihood_ratio(X, pairs))
            abs_out.append(bv.nll_steps(X).sum(0).min())
        q = 1 - p["false_alarm"]
        thr_g, thr_a = np.quantile(glr_in, q), np.quantile(abs_in, q)
        novelty[str(win)] = {"glr_flagged": float(np.mean(np.array(glr_out) > thr_g)),
                             "abs_flagged": float(np.mean(np.array(abs_out) > thr_a)),
                             "glr_threshold": float(thr_g)}
        nov_arrays.update({f"glr_in_{win}": glr_in, f"glr_out_{win}": glr_out, f"abs_in_{win}": abs_in,
                           f"abs_out_{win}": abs_out})

    f = lambda v: [x if np.isfinite(x) else None for x in v]
    out = {"bank": bank_idx, "windows": p["windows"], "recall_accuracy": recall_acc, "detect_accuracy": detect_acc,
           "first_lock": f(first_lock), "relock_filter": f(relock_f), "relock_detect": f(relock_d),
           "relock_filter_median": float(np.median(relock_f)), "relock_detect_median": float(np.median(relock_d)),
           "relock_filter_p90": float(np.percentile(relock_f, 90)),
           "error_after_switch": {k: float(np.mean(v)) for k, v in err.items()},
           "noise_floor": float(np.linalg.norm(scale)), "novelty": novelty,
           "example_contexts": [example["a"], example["b"]]}
    arrays = {"pi": example["pi"], "theta_act": example["theta_act"], "theta_nom": example["theta_nom"],
              "gates_act": example["gates_act"], "gates_nom": example["gates_nom"], "bank_patterns": bank.patterns,
              **nov_arrays}
    return out, arrays


if __name__ == "__main__":
    main("e6", run)
