"""E7: everything together. Eight pendulums on a ring, rewired every few seconds, with the wiring selected from the
complete bank of all 2^8 = 256 patterns. K and M are learned only on 3- and 5-body graphs (E2's training set).

Each method picks the wiring used to predict step t -> t+1 from what it has measured up to x(t):
  known      the true wiring, known the moment it changes (E3's gates)
  recall     the binding vertex over the complete bank (E6): its most believed wiring
  detect-W   non-negative least squares on the last W steps (E5), thresholded at 0.5
  stale      the first wiring, never updated
The nominal world is the prediction made `horizon` steps earlier, from the measured state, with the wiring then chosen.
"""

import numpy as np
from common import main

from nsp import config, context, detect, fit, graphs, world as W

METHODS = ("known", "recall", "stale")


def index(g) -> int:
    """Row of graphs.all_gate_patterns for a binary gate vector."""
    return int(sum(int(round(x)) << i for i, x in enumerate(g)))


def learn(p, w, rng):
    """E2's training set: paths and rings of 3 and 5 bodies with random gates, clean."""
    train = []
    for n in p["train_sizes"]:
        for name in p["train_graphs"]:
            edges = graphs.named(name, n)
            for g in graphs.random_gate_patterns(len(edges), p["train_patterns"], rng):
                L = graphs.laplacian(n, edges, g)
                x0 = W.initial_states(w, n, p["train_trajectories"], p["amplitude"], rng)
                train.append((W.simulate_exact(w, [(p["train_steps"], L)], x0), L))
    return fit.fit_joint(train)[0]


def calibrate(p, w, model_, noise, n, rng):
    """Residual scale per coordinate, measured on a noisy ring of 5 with known gates, tiled over n bodies."""
    L5 = graphs.laplacian(5, graphs.ring(5))
    X = W.add_noise(w, W.simulate_exact(w, [(p["calibration_steps"], L5)],
                                        W.initial_states(w, 5, 1, p["amplitude"], rng)), noise, p["amplitude"], rng)
    return np.tile(context.residual_scale(X, model_.assemble(L5))[:2], n)


def schedule(kind: str, C: int, m: int, segments: int, rng) -> list[int]:
    """Wiring per segment: a fresh random pattern each time, or one spring flipped."""
    seq = [int(rng.integers(C))]
    for _ in range(segments - 1):
        if kind == "pattern":
            c = int(rng.integers(C - 1))
            seq.append(c + (c >= seq[-1]))
        else:
            seq.append(seq[-1] ^ (1 << int(rng.integers(m))))
    return seq


def trial(w, p, ring, pats, P, Pk, vertices, model_, seq, every, noise, windows, rng, keep=False):
    n, m, C = p["n"], len(ring), len(pats)
    T, k = every * len(seq), p["horizon"]
    x0 = W.initial_states(w, n, 1, p["amplitude"], rng)
    Xc = W.simulate_exact(w, [(every, graphs.laplacian(n, ring, pats[c])) for c in seq], x0)[0]
    X = W.add_noise(w, Xc[None], noise, p["amplitude"], rng)[0]
    truth = np.repeat(seq, every)
    est = {"known": truth, "stale": np.full(T, seq[0])}
    after = {}  # wiring believed once x(t + 1) is in
    for name, bv in vertices.items():
        b = bv.filter(X)  # belief after seeing x(t + 1)
        est[name], after[name] = np.r_[0, b[:-1].argmax(1)], b.argmax(1)
        if name == "recall":
            pi = b
    G, r = detect.design_steps(X, model_, ring, n)
    for win in windows:
        idx = (detect.sliding_gates(G, r, win) > 0.5).astype(int) @ (1 << np.arange(m))
        est[f"detect-{win}"], after[f"detect-{win}"] = np.r_[0, idx[:-1]], idx
    correct = {name: e == truth for name, e in est.items()}

    switches = [s for s in range(every, T, every)]
    relock = {name: [] for name in after}
    for s in switches:
        new = truth[s]
        for name, a in after.items():
            hit = np.flatnonzero(a[s:s + every] == new)
            relock[name].append(int(hit[0]) + 1 if len(hit) else None)
    lock = int(np.flatnonzero(after["recall"] == truth[0])[0]) + 1 if (after["recall"] == truth[0]).any() else None

    # how stretched the springs that changed are, just after each switch (clean states)
    th = Xc[:, 0::2]
    stretch = []
    for s in switches:
        old, new = pats[truth[s - 1]], pats[truth[s]]
        ch = [e for e in range(m) if old[e] != new[e]]
        u, v = np.array([ring[e][0] for e in ch]), np.array([ring[e][1] for e in ch])
        stretch.append(float(((th[s:s + 10, v] - th[s:s + 10, u]) ** 2).sum(1).mean()))

    # the nominal world: the k-step-ahead prediction from the measured state, with the wiring chosen then
    dev = {}
    for name, e in est.items():
        pred = np.einsum("tij,tj->ti", Pk[e[:T - k + 1]], X[:T - k + 1])
        dev[name] = np.linalg.norm(pred - Xc[k:], axis=1) / np.linalg.norm(Xc[k:], axis=1)
    out = {"correct": correct, "relock": relock, "lock": lock, "stretch": stretch, "dev": dev, "switches": switches}
    if keep:
        nominal = {}
        for name, e in est.items():
            th_nom = np.empty((T + 1, n))
            x = X[0].copy()
            for t in range(min(k, T + 1)):  # first k steps: predicted from x(0) with the wiring chosen at 0
                th_nom[t] = x[0::2]
                x = P[e[0]] @ x
            th_nom[k:] = np.einsum("tij,tj->ti", Pk[e[:T - k + 1]], X[:T - k + 1])[:, 0::2]
            g_nom = np.vstack([np.tile(pats[e[0]], (k, 1)), pats[e[:T - k + 1]]])
            nominal[name] = (th_nom, g_nom)
        out["example"] = {"theta_act": th, "gates_act": np.vstack([pats[truth], pats[truth[-1]]]), "pi": pi,
                          "seq": np.array(seq), "nominal": nominal}
    return out


def run(cfg: dict, seed: int):
    p = cfg["e7"]
    w = config.world_from(cfg)
    rng = np.random.default_rng(seed)
    n, ring = p["n"], graphs.named(p["graph"], p["n"])
    pats = graphs.all_gate_patterns(len(ring))
    C = len(pats)
    model_ = learn(p, w, rng)
    bank = context.ContextBank(n, ring, pats)
    P = np.array([model_.assemble(L) for L in bank.laplacians()])
    Pk = np.array([np.linalg.matrix_power(A, p["horizon"]) for A in P])
    scales = {}

    def vertex(noise, every):
        if noise not in scales:
            scales[noise] = calibrate(p, w, model_, noise, n, rng)
        return context.BindingVertex(bank, model_, scales[noise], hazard=1.0 / every)

    # main condition
    every, T = p["switch_every"], p["total_steps"]
    vertices = {"recall": vertex(p["noise"], every)}
    names = list(METHODS) + [f"detect-{win}" for win in p["windows"]]
    pooled = {"correct": {k: [] for k in names}, "relock": {}, "dev": {k: [] for k in names}}
    aligned_c, aligned_d = {k: [] for k in names}, {k: [] for k in names}
    events, locks, example, example_dev = [], [], None, None
    for i in range(p["trials"]):
        seq = schedule(p["switch_kind"], C, len(ring), T // every, rng)
        r = trial(w, p, ring, pats, P, Pk, vertices, model_, seq, every, p["noise"], p["windows"], rng, keep=(i == 0))
        if i == 0:
            example, example_dev = r["example"], r["dev"]
        locks.append(r["lock"])
        for name in names:
            pooled["correct"][name].append(r["correct"][name])
            pooled["dev"][name].append(r["dev"][name])
            for s in r["switches"]:
                aligned_c[name].append(r["correct"][name][s - 5: s + every - 5])
                if s - 5 + every - 5 <= len(r["dev"][name]):
                    aligned_d[name].append(r["dev"][name][s - 5: s + every - 5] / r["dev"]["known"][s - 5: s + every - 5])
        for j, s in enumerate(r["switches"]):
            events.append({"stretch": r["stretch"][j], **{f"relock_{k}": v[j] for k, v in r["relock"].items()}})
        for k, v in r["relock"].items():
            pooled["relock"].setdefault(k, []).extend(v)

    def frac(name, skip_first=False):
        arr = np.array(pooled["correct"][name])
        return float(arr[:, every:].mean() if skip_first else arr.mean())

    def med(xs):
        v = [x for x in xs if x is not None]
        return float(np.median(v)) if v else None

    dev_mean = {k: float(np.mean(np.concatenate(v))) for k, v in pooled["dev"].items()}
    metrics = {
        "names": names, "bank_size": C, "switch_every": every, "total_steps": T, "hazard": 1.0 / every,
        "noise_floor": float(np.linalg.norm(scales[p["noise"]])),
        "correct": {k: frac(k) for k in names}, "correct_after_first": {k: frac(k, True) for k in names},
        "relock_median": {k: med(v) for k, v in pooled["relock"].items()},
        "relock_p90": {k: float(np.percentile([x for x in v if x is not None], 90)) for k, v in pooled["relock"].items()},
        "relock_never": {k: int(sum(x is None for x in v)) for k, v in pooled["relock"].items()},
        "first_lock_median": med(locks),
        "dev_mean": dev_mean, "dev_ratio": {k: dev_mean[k] / dev_mean["known"] for k in names},
        "events": events,
    }

    # sweep: switch interval x noise x kind of rewiring; also a vertex whose prior expects one-spring flips
    sweep = []
    names_s = names + ["recall-flip"]
    for kind in p["sweep_kinds"]:
        for noise in p["sweep_noise"]:
            for ev in p["sweep_every"]:
                bv_s = vertex(noise, ev)
                bv_f = context.BindingVertex(bank, model_, scales[noise], hazard=1.0 / ev,
                                             transitions=context.flip_transitions(pats, 1.0 / ev))
                acc = {k: [] for k in names_s}
                rel = {"recall": [], "recall-flip": [], "detect-20": []}
                dv = {k: [] for k in names_s}
                for _ in range(p["sweep_trials"]):
                    seq = schedule(kind, C, len(ring), T // ev, rng)
                    r = trial(w, p, ring, pats, P, Pk, {"recall": bv_s, "recall-flip": bv_f}, model_, seq, ev, noise,
                              p["windows"], rng)
                    for k in names_s:
                        acc[k].append(r["correct"][k].mean())
                        dv[k].append(r["dev"][k].mean())
                    for k in rel:
                        rel[k].extend(r["relock"][k])
                sweep.append({"kind": kind, "noise": noise, "every": ev,
                              "correct": {k: float(np.mean(v)) for k, v in acc.items()},
                              "relock_median": {k: med(v) for k, v in rel.items()},
                              "dev_ratio": {k: float(np.mean(dv[k]) / np.mean(dv["known"])) for k in names_s}})
    metrics["sweep"] = sweep

    arrays = {"theta_act": example["theta_act"], "gates_act": example["gates_act"], "pi": example["pi"].astype(np.float32),
              "seq": example["seq"], "bank_patterns": pats,
              "aligned_offsets": np.arange(-5, every - 5)}
    for name, (th_nom, g_nom) in example["nominal"].items():
        arrays[f"theta_nom_{name}"], arrays[f"gates_nom_{name}"] = th_nom, g_nom
    for name in names:
        arrays[f"aligned_correct_{name}"] = np.mean(aligned_c[name], 0)
        arrays[f"aligned_dev_{name}"] = np.mean(aligned_d[name], 0)
        arrays[f"dev_example_{name}"] = example_dev[name]
    return metrics, arrays


if __name__ == "__main__":
    main("e7", run)
