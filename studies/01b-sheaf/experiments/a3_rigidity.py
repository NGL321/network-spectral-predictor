"""A3: rigidity and detection. A spring at its natural length feels only motion along its own axis: which gates can
detection see, and when is a gate invisible?

Eight pivots at random, wired as a ring, a ring plus 3 chords, and a ring plus 5 chords (13 = 2n - 3 springs).
  counting   motions no spring resists (2n - rank R: 3 rigid motions plus the floppy mechanisms), against the number of
             the world's modes that swing at exactly omega0, as a free pendulum would
  detection  non-negative least squares along known axes over all 28 pairs, from one noisy window, started three ways:
             random      every mode excited
             floppy      only the motions no spring resists, plus a small leak of random motion
             rigid       every bob displaced alike (a translation: no spring stretches)
  visibility each spring's load in the window: rms of the relative motion along its axis, per swing
Step 1's planar maps are carried through the axes; nothing is fitted in 3-D.
"""

import numpy as np
from common import main
from maps import planar_maps
from scipy.optimize import nnls

from nsp import config, detect, graphs, metrics, sheaf, sphere, world as W

KINDS = ("random", "floppy", "rigid")


def start(kind, P, edges, amp, omega0, leak, rng):
    n = len(P)
    x = sphere.initial_states(n, 1, amp, rng, omega0)[0].reshape(n, 2, 2)
    if kind == "random":
        return x.reshape(-1)
    if kind == "rigid":
        q, v = rng.normal(0, amp, 2), rng.normal(0, amp * omega0, 2)
        out = np.zeros((n, 2, 2))
        out[:, 0], out[:, 1] = q, v
        return out.reshape(-1)
    Z = np.hstack([sheaf.rigid_motions(P), sheaf.mechanisms(P, edges)])
    Z, _ = np.linalg.qr(Z)
    q = Z @ rng.normal(0, 1, Z.shape[1])
    v = Z @ rng.normal(0, 1, Z.shape[1])
    out = np.zeros((n, 2, 2))
    out[:, 0] = q.reshape(n, 2) * amp * np.sqrt(n) / np.linalg.norm(q)
    out[:, 1] = v.reshape(n, 2) * amp * omega0 * np.sqrt(n) / np.linalg.norm(v)
    return (out + leak * x).reshape(-1)


def visibility(X, P, pairs, amp):
    n = len(P)
    q = X[:, :4 * n].reshape(-1, n, 2, 2)[:, :, 0]
    Pe = sheaf.projectors(P, pairs)
    return np.array([np.sqrt((np.einsum("ij,tj->ti", Pe[k], q[:, b] - q[:, a]) ** 2).sum(1).mean()) / amp
                     for k, (a, b) in enumerate(pairs)])


def run(cfg: dict, seed: int):
    p = cfg["a3"]
    w = config.world_from(cfg)
    rng = np.random.default_rng(seed)
    n, amp = p["n"], p["amplitude"]
    model_ = sheaf.SheafModel.from_planar(planar_maps(p, w.with_(d=1), rng))
    pairs = graphs.complete(n)
    counts, det, live_vis = [], [], []
    for _ in range(p["layouts"]):
        P = sheaf.layout(n, rng, p["box"], p["min_sep"])
        edges = sheaf.angular_ring(P)
        wirings = []
        for c in p["chords"]:
            edges = sheaf.add_chords(P, edges, c - (len(edges) - n), rng) if c else edges
            wirings.append((c, list(edges)))
        for c, edges in wirings:
            Phi = sheaf.exact_step(w, P, edges)
            freq = np.abs(np.angle(np.linalg.eigvals(Phi))) / w.dt
            at_w0 = int(np.sum(np.abs(freq - w.omega0) < 1e-8) // 2)
            counts.append({"chords": c, "springs": len(edges), "zero_modes": sheaf.zero_modes(P, edges),
                           "mechanisms": int(sheaf.mechanisms(P, edges).shape[1]), "modes_at_omega0": at_w0})
            truth = detect.truth_vector(pairs, edges, np.ones(len(edges)))
            for kind in KINDS:
                for win in p["windows"]:
                    x0 = start(kind, P, edges, amp, w.omega0, p["floppy_leak"], rng)
                    X = sheaf.simulate_exact(w, P, edges, [(win, None)], x0[None])[0]
                    Xn = W.add_noise(w, X, p["noise"], amp, rng)
                    G, r = sheaf.design_steps(Xn, model_, P, pairs)
                    g = nnls(G.reshape(-1, len(pairs)), r.reshape(-1))[0]
                    vis = visibility(X, P, pairs, amp)
                    det.append({"chords": c, "kind": kind, "window": win, "auroc": metrics.auroc(g, truth),
                                "exact": detect.exact_match(g, truth), "spurious_max": float(g[truth == 0].max()),
                                "live_missed": int(((g < 0.5) & (truth > 0.5)).sum())})
                    if win == p["windows"][-1]:
                        live_vis += [(kind, float(v), float(gh)) for v, gh, t in zip(vis, g, truth) if t > 0.5]

    summary = {}
    for kind in KINDS:
        for win in p["windows"]:
            rows = [d for d in det if d["kind"] == kind and d["window"] == win]
            summary[f"{kind}_{win}"] = {"auroc": float(np.mean([d["auroc"] for d in rows])),
                                        "exact": float(np.mean([d["exact"] for d in rows])),
                                        "live_missed_per_layout": float(np.mean([d["live_missed"] for d in rows])),
                                        "spurious_max": float(np.max([d["spurious_max"] for d in rows]))}
    match = float(np.mean([c["zero_modes"] == c["modes_at_omega0"] for c in counts]))
    by_chords = {c: {k: float(np.mean([r[k] for r in counts if r["chords"] == c]))
                     for k in ("springs", "zero_modes", "mechanisms")} for c in p["chords"]}
    lv = np.array([(KINDS.index(k), v, g) for k, v, g in live_vis])
    metrics_ = {"zero_modes_match_modes_at_omega0": match, "by_chords": by_chords, "detection": summary}
    return metrics_, {"live_visibility": lv}


if __name__ == "__main__":
    main("a3", run)
