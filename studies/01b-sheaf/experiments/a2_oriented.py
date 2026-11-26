"""A2: oriented springs. Each spring resists only stretch along its own axis: does step 1 transfer through the maps?

Pivots at random in a plane, joined in a ring plus chords, each spring along its own axis n_e. Training layouts of 3 and
5 bodies; test layouts of 8 bodies with 3 chords, never seen.
  graph      one 4 x 4 body map and one 4 x 4 spring map shared by every spring, blind to axes (fitted on training)
  sheaf      a 4 x 4 body map and one 2 x 2 spring map placed along each axis by n n^T (fitted on the same data)
  sheaf-iso  the same with an isotropic body, K2 (x) I_2: 8 numbers
  planar     step 1's planar maps pushed through the axes, with no 3-D fitting at all
  own        the sheaf model fitted on the test layout itself, from separate runs (the reference, as in E2)
"""

import numpy as np
from common import main
from maps import planar_maps

from nsp import config, fit, graphs, metrics, model, sheaf, sphere

MODELS = ("graph", "sheaf", "sheaf-iso", "planar", "own")


def layout(n, chords, min_sep, p, rng):
    P = sheaf.layout(n, rng, p["box"], min_sep)
    return P, sheaf.add_chords(P, sheaf.angular_ring(P), chords, rng)


def run(cfg: dict, seed: int):
    p = cfg["a2"]
    w = config.world_from(cfg)
    rng = np.random.default_rng(seed)
    steps, H, amp = p["steps"], p["horizons"], p["amplitude"]

    sets, graph_sets = [], []
    for n, chords in zip(p["train_sizes"], p["train_chords"]):
        for _ in range(p["train_layouts"]):
            P, edges = layout(n, chords, p["train_min_sep"], p, rng)
            X = sheaf.simulate_exact(w, P, edges, [(p["train_steps"], None)],
                                     sphere.initial_states(n, p["train_trajectories"], amp, rng, w.omega0))
            sets.append((X, P, edges, None))
            graph_sets.append((X, graphs.laplacian(n, edges)))
    graph_m = fit.fit_joint(graph_sets, d=2)[0]
    sheaf_m = sheaf.fit_sheaf(sets)[0]
    iso_m = sheaf.fit_sheaf(sets, isotropic_body=True)[0]
    planar = sheaf.SheafModel.from_planar(planar_maps(p, w.with_(d=1), rng))
    ref = model.series(w.with_(d=1))

    per_layout, curves, example = [], {m: [] for m in MODELS}, None
    for k in range(p["test_layouts"]):
        n = p["test_n"]
        P, edges = layout(n, p["test_chords"], p["test_min_sep"], p, rng)
        X = sheaf.simulate_exact(w, P, edges, [(steps, None)],
                                 sphere.initial_states(n, p["test_trajectories"], amp, rng, w.omega0))
        Xo = sheaf.simulate_exact(w, P, edges, [(p["train_steps"], None)],
                                  sphere.initial_states(n, p["own_trajectories"], amp, rng, w.omega0))
        own = sheaf.fit_sheaf([(Xo, P, edges, None)])[0]
        L = graphs.laplacian(n, edges)
        nominal = {"graph": graph_m.rollout(X[:, 0], [(steps, L)])}
        for name, m in (("sheaf", sheaf_m), ("sheaf-iso", iso_m), ("planar", planar), ("own", own)):
            nominal[name] = m.rollout(X[:, 0], P, edges, [(steps, None)])
        row = {"layout": k, "springs": len(edges), "zero_modes": sheaf.zero_modes(P, edges)}
        for name in MODELS:
            c = metrics.rollout_curve(X, nominal[name])
            curves[name].append(c)
            row[name] = metrics.at_horizons(c, H)
        row["ratio_to_own"] = {name: row[name][H[-1]] / row["own"][H[-1]] for name in MODELS if name != "own"}
        per_layout.append(row)
        if k == 0:
            mu = np.clip(np.linalg.eigvalsh(sheaf.sheaf_laplacian(P, edges)), 0, None)
            example = {"P": P, "edges": np.array(edges), "actual": X[0], "mu": mu,
                       **{f"nominal_{name}": nominal[name][0] for name in ("graph", "sheaf", "planar")}}

    K0 = np.kron(ref.K, np.eye(2))
    fitted = {name: {"K_vs_K0": float(np.abs(m.K - K0).max()), "M_vs_M0": float(np.abs(m.M - ref.M).max()),
                     "M": m.M} for name, m in (("sheaf", sheaf_m), ("sheaf-iso", iso_m), ("planar", planar))}
    fitted["graph"] = {"K_vs_K0": float(np.abs(graph_m.K - K0).max()), "M": graph_m.M}
    worst = {name: max(r["ratio_to_own"][name] for r in per_layout) for name in MODELS if name != "own"}
    metrics_ = {"per_layout": per_layout, "worst_ratio_to_own": worst, "fitted": fitted, "M0": ref.M}
    arrays = {f"curve_{name}": np.array(curves[name]) for name in MODELS}
    arrays.update({f"example_{k}": v for k, v in example.items()})
    return metrics_, arrays


if __name__ == "__main__":
    main("a2", run)
