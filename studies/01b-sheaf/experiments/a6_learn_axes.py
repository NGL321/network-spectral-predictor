"""A6: learning the springs' axes. Nothing about the geometry is given: for every pair of bodies, learn a 2 x 2
restriction map S_ab from motion alone, with step 1's planar maps carried over.

A scalar gate g_ab becomes a map gate: the coupling of pair (a, b) is M (x) S_ab with S_ab symmetric (3 numbers),
linear in the residual, so one least-squares solve learns all 28 at once:

    r_v = x_v(t+1) - (K (x) I_2) x_v(t) = - sum_(a,b) at v (M (x) S_ab)(x_v - x_other)

A real spring at its natural length should come out rank one, S_ab = g n n^T: its top eigenvector the spring's axis, its
top eigenvalue the gate, its second eigenvalue zero. A pair with no spring should come out zero.
  map              the least-squares S_ab as they come out (they can be indefinite: a negative stiffness)
  map, then axis   each pair's axis read off its S_ab, then the gates along those axes by non-negative least squares
                   (A3's detector), so every learned coupling is a spring g n n^T with g >= 0
  ... and select   the pairs whose gate clears 0.5 kept as springs (gate 1) along their learned axes; the rest dropped
Data: one window of `window` steps from `runs` independent runs, with measurement noise.
"""

import numpy as np
from common import main
from maps import planar_maps

from scipy.optimize import nnls

from nsp import config, detect, graphs, metrics, sheaf, sphere, world as W

BASIS = (np.array([[1.0, 0], [0, 0]]), np.array([[0, 1.0], [1, 0]]), np.array([[0, 0], [0, 1.0]]))


def learn_maps(X, K, M, n, pairs):
    """Least-squares S_ab for every candidate pair. X: (ntraj, T + 1, 4n). Returns (m, 2, 2)."""
    xs, ys = X[:, :-1].reshape(-1, n, 4), X[:, 1:].reshape(-1, n, 4)
    r = (ys - xs @ K.T).reshape(-1)
    cols = []
    for a, b in pairs:
        d = xs[:, b] - xs[:, a]
        for B in BASIS:
            col = np.zeros_like(xs)
            dv = d @ np.kron(M, B).T
            col[:, b], col[:, a] = -dv, dv
            cols.append(col.reshape(-1))
    w = np.linalg.lstsq(np.stack(cols, 1), r, rcond=None)[0].reshape(len(pairs), 3)
    return np.einsum("mk,kij->mij", w, np.array(BASIS))


def run(cfg: dict, seed: int):
    p = cfg["a6"]
    w = config.world_from(cfg)
    rng = np.random.default_rng(seed)
    n, amp, H = p["n"], p["amplitude"], p["horizons"]
    planar = planar_maps(p, w.with_(d=1), rng)
    model_ = sheaf.SheafModel.from_planar(planar)
    pairs = graphs.complete(n)
    rows = []
    for _ in range(p["layouts"]):
        P = sheaf.layout(n, rng, p["box"], p["min_sep"])
        edges = sheaf.add_chords(P, sheaf.angular_ring(P), p["chords"], rng)
        truth = detect.truth_vector(pairs, edges, np.ones(len(edges)))
        true_axes = sheaf.axes(P, pairs)
        Xt = sheaf.simulate_exact(w, P, edges, [(p["steps"], None)],
                                  sphere.initial_states(n, p["test_trajectories"], amp, rng, w.omega0))
        known = metrics.rollout_curve(Xt, model_.rollout(Xt[:, 0], P, edges, [(p["steps"], None)]))
        live = truth > 0.5
        for win in p["windows"]:
            for runs in p["runs"]:
                for noise in p["noise_levels"]:
                    X = sheaf.simulate_exact(w, P, edges, [(win, None)],
                                             sphere.initial_states(n, runs, amp, rng, w.omega0))
                    Xn = W.add_noise(w, X, noise, amp, rng)
                    S = learn_maps(Xn, model_.K, model_.M, n, pairs)
                    ev, evec = np.linalg.eigh(S)  # ascending
                    top, second, axis = ev[:, 1], ev[:, 0], evec[:, :, 1]
                    angle = np.degrees(np.arccos(np.clip(np.abs((axis * true_axes).sum(1)), 0, 1)))
                    Pa = np.einsum("mi,mj->mij", axis, axis)
                    GA, RA = zip(*(sheaf.design_steps(x, model_, P, pairs, maps=Pa) for x in Xn))
                    g = nnls(np.concatenate(GA).reshape(-1, len(pairs)), np.concatenate(RA).reshape(-1))[0]
                    row = {"window": win, "runs": runs, "noise": noise,
                           "axis_error_deg_median": float(np.median(angle[live])),
                           "second_over_top_median": float(np.median(np.abs(second[live]) / np.abs(top[live]))),
                           "empty_pairs_top_max": float(np.abs(ev[~live]).max())}
                    sel = (g > 0.5).astype(float)
                    for name, score, maps in (("map", top, S), ("map, then axis", g, g[:, None, None] * Pa),
                                              ("and select", g, sel[:, None, None] * Pa)):
                        with np.errstate(all="ignore"):
                            c = metrics.rollout_curve(Xt, model_.rollout(Xt[:, 0], P, pairs, [(p["steps"], None)],
                                                                         maps=maps))
                        row[name] = {"auroc": metrics.auroc(score, truth), "exact": detect.exact_match(score, truth),
                                     "gate_error_median": float(np.median(np.abs(score[live] - 1))),
                                     "vs_known_20": float(c[H[0]] / known[H[0]]) if np.isfinite(c[H[0]]) else np.inf,
                                     "vs_known_200": float(c[H[-1]] / known[H[-1]]) if np.isfinite(c[H[-1]]) else np.inf}
                    rows.append(row)
    summary = {}
    for win in p["windows"]:
        for runs in p["runs"]:
            for noise in p["noise_levels"]:
                rr = [r for r in rows if r["window"] == win and r["runs"] == runs and r["noise"] == noise]
                out = {"axis_error_deg_median": float(np.median([r["axis_error_deg_median"] for r in rr])),
                       "second_over_top_median": float(np.median([r["second_over_top_median"] for r in rr])),
                       "empty_pairs_top_max": float(np.max([r["empty_pairs_top_max"] for r in rr]))}
                for name in ("map", "map, then axis", "and select"):
                    out[name] = {"auroc": float(np.mean([r[name]["auroc"] for r in rr])),
                                 "exact": float(np.mean([r[name]["exact"] for r in rr])),
                                 "gate_error_median": float(np.median([r[name]["gate_error_median"] for r in rr])),
                                 "vs_known_20_median": float(np.median([r[name]["vs_known_20"] for r in rr])),
                                 "vs_known_200_median": float(np.median([r[name]["vs_known_200"] for r in rr])),
                                 "vs_known_200_worst": float(np.max([r[name]["vs_known_200"] for r in rr]))}
                summary[f"{win} steps x {runs} runs, noise {noise}"] = out
    return {"summary": summary}, {}


if __name__ == "__main__":
    main("a6", run)
