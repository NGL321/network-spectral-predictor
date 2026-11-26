"""A5: the capstone in 3-D. Eight spherical pendulums on a circle, their ring of oriented springs rewired every 2.5 s,
the wiring picked from the complete bank of 2^8 = 256 patterns. The maps are learned only on 3- and 5-body layouts.

E7 again, with the sheaf: every stored wiring's step is the sheaf model's, each spring along its own axis.
  known      the true wiring, known the moment it changes
  recall     the binding vertex over the complete bank: its most believed wiring
  detect-W   non-negative least squares along known axes on the last W steps, thresholded at 0.5
  stale      the first wiring, never updated
The nominal world is the prediction made `horizon` steps earlier, from the measured state, with the wiring then chosen.
"""

import numpy as np
from common import main

from nsp import config, context, detect, graphs, sheaf, sphere, world as W


def learn(p, w, rng):
    sets = []
    for n, chords in zip(p["train_sizes"], p["train_chords"]):
        for _ in range(p["train_layouts"]):
            P = sheaf.layout(n, rng, p["box"], p["train_min_sep"])
            edges = sheaf.add_chords(P, sheaf.angular_ring(P), chords, rng)
            X = sheaf.simulate_exact(w, P, edges, [(p["train_steps"], None)],
                                     sphere.initial_states(n, p["train_trajectories"], p["amplitude"], rng, w.omega0))
            sets.append((X, P, edges, None))
    return sheaf.fit_sheaf(sets)[0]


def calibrate(p, w, model_, n, rng):
    P5 = sheaf.circle_layout(5, p["radius"])
    ring5 = graphs.ring(5)
    X = sheaf.simulate_exact(w, P5, ring5, [(p["calibration_steps"], None)],
                             sphere.initial_states(5, 1, p["amplitude"], rng, w.omega0))
    X = W.add_noise(w, X, p["noise"], p["amplitude"], rng)
    return np.tile(context.residual_scale(X, model_.assemble(P5, ring5), d=2)[:4], n)


def schedule(kind, C, m, segments, rng):
    seq = [int(rng.integers(C))]
    for _ in range(segments - 1):
        if kind == "pattern":
            c = int(rng.integers(C - 1))
            seq.append(c + (c >= seq[-1]))
        else:
            seq.append(seq[-1] ^ (1 << int(rng.integers(m))))
    return seq


def trial(w, p, Pv, ring, pats, S, Sk, vertex, model_, seq, rng):
    n, m, every, k = p["n"], len(ring), p["switch_every"], p["horizon"]
    T = every * len(seq)
    x0 = sphere.initial_states(n, 1, p["amplitude"], rng, w.omega0)
    Xc = sheaf.simulate_exact(w, Pv, ring, [(every, pats[c]) for c in seq], x0)[0]
    X = W.add_noise(w, Xc[None], p["noise"], p["amplitude"], rng)[0]
    truth = np.repeat(seq, every)
    est = {"known": truth, "stale": np.full(T, seq[0])}
    after = {}
    b = vertex.filter(X)
    est["recall"], after["recall"] = np.r_[0, b[:-1].argmax(1)], b.argmax(1)
    G, r = sheaf.design_steps(X, model_, Pv, ring)
    for win in p["windows"]:
        idx = (detect.sliding_gates(G, r, win) > 0.5).astype(int) @ (1 << np.arange(m))
        est[f"detect-{win}"], after[f"detect-{win}"] = np.r_[0, idx[:-1]], idx
    correct = {name: e == truth for name, e in est.items()}
    relock = {name: [] for name in after}
    for s in range(every, T, every):
        for name, a in after.items():
            hit = np.flatnonzero(a[s:s + every] == truth[s])
            relock[name].append(int(hit[0]) + 1 if len(hit) else None)
    dev = {}
    for name, e in est.items():
        pred = np.einsum("tij,tj->ti", Sk[e[:T - k + 1]], X[:T - k + 1])
        dev[name] = np.linalg.norm(pred - Xc[k:], axis=1) / np.linalg.norm(Xc[k:], axis=1)
    return correct, relock, dev


def run(cfg: dict, seed: int):
    p = cfg["a5"]
    w = config.world_from(cfg)
    rng = np.random.default_rng(seed)
    n = p["n"]
    Pv = sheaf.circle_layout(n, p["radius"])
    ring = graphs.ring(n)
    pats = graphs.all_gate_patterns(len(ring))
    model_ = learn(p, w, rng)
    S = np.array([model_.assemble(Pv, ring, g) for g in pats])
    Sk = np.array([np.linalg.matrix_power(s, p["horizon"]) for s in S])
    bank = context.ContextBank(n, ring, pats)
    scale = calibrate(p, w, model_, n, rng)
    vertex = context.BindingVertex(bank, None, scale, hazard=1 / p["switch_every"], steps=S)
    segments = p["total_steps"] // p["switch_every"]
    out = {}
    for kind in p["kinds"]:
        pooled = {"correct": {}, "relock": {}, "dev": {}}
        for _ in range(p["trials"]):
            seq = schedule(kind, len(pats), len(ring), segments, rng)
            correct, relock, dev = trial(w, p, Pv, ring, pats, S, Sk, vertex, model_, seq, rng)
            for key, d in (("correct", correct), ("relock", relock), ("dev", dev)):
                for name, v in d.items():
                    pooled[key].setdefault(name, []).extend(list(v))
        known_dev = np.mean(pooled["dev"]["known"])
        out[kind] = {
            "right_wiring": {name: float(np.mean(v)) for name, v in pooled["correct"].items()},
            "relock_median": {name: (float(np.median([x for x in v if x is not None])) if any(x is not None for x in v)
                                     else None) for name, v in pooled["relock"].items()},
            "relock_never": {name: int(sum(x is None for x in v)) for name, v in pooled["relock"].items()},
            "ahead_deviation": {name: float(np.mean(v)) for name, v in pooled["dev"].items()},
            "ahead_ratio_to_known": {name: float(np.mean(v) / known_dev) for name, v in pooled["dev"].items()},
        }
    out["zero_modes_all_on"] = sheaf.zero_modes(Pv, ring)
    return out, {}


if __name__ == "__main__":
    main("a5", run)
