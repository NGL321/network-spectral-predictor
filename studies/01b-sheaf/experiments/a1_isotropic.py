"""A1: an isotropic control. If springs pull equally in every direction, is 3-D just the planar step twice?

Spherical pendulums (two tilts per body) on a ring of 8, with springs that couple both tilts alike (L (x) E (x) I_2).
  identity   the exact 3-D step against the planar step (x) I_2
  transfer   step 1's planar maps, learned on 3- and 5-body paths and rings, run once per tilt on the 3-D ring of 8;
             against the same maps on planar data, and a 3-D model (4 x 4 K and M) fitted from scratch on 3-D data
  data cost  ring-8 data only, scarce or noisy: the Kronecker fit (K2 (x) I_2 and M2 (x) I_2, 8 numbers), the
             unconstrained fit (4 x 4 K and M, 32 numbers) and one global linear map (DMD, 1024 numbers)
"""

import numpy as np
from common import main

from nsp import config, fit, graphs, metrics, world as W


def training(p, w, rng):
    sets = []
    for n in p["train_sizes"]:
        for name in p["train_graphs"]:
            L = graphs.laplacian(n, graphs.named(name, n))
            x0 = W.initial_states(w, n, p["train_trajectories"], p["amplitude"], rng)
            sets.append((W.simulate_exact(w, [(p["train_steps"], L)], x0), L))
    return sets


def curve_of(Xa, step, steps):
    Xn = np.empty_like(Xa)
    Xn[:, 0] = Xa[:, 0]
    for t in range(steps):
        Xn[:, t + 1] = Xn[:, t] @ step.T
    return metrics.rollout_curve(Xa, Xn), Xn


def run(cfg: dict, seed: int):
    p = cfg["a1"]
    w3 = config.world_from(cfg)
    w2 = w3.with_(d=1)
    rng = np.random.default_rng(seed)
    n, steps, H = p["n"], p["steps"], p["horizons"]
    L = graphs.laplacian(n, graphs.ring(n))

    identity = float(np.abs(W.exact_step(w3, L) - np.kron(W.exact_step(w2, L), np.eye(2))).max())

    planar = fit.fit_joint(training(p, w2, rng))[0]
    scratch = fit.fit_joint(training(p, w3, rng), d=2)[0]
    x0 = W.initial_states(w3, n, p["trajectories"], p["amplitude"], rng)
    X3 = W.simulate_exact(w3, [(steps, L)], x0)
    X2 = W.simulate_exact(w2, [(steps, L)], W.initial_states(w2, n, p["trajectories"], p["amplitude"], rng))
    c_transfer, Xn_transfer = curve_of(X3, planar.per_direction(2).assemble(L), steps)
    c_planar, _ = curve_of(X2, planar.assemble(L), steps)
    c_scratch, _ = curve_of(X3, scratch.assemble(L), steps)
    K0 = np.kron(W.exact_step(w2, np.zeros((1, 1))), np.eye(2))
    cross = float(max(np.abs(scratch.K[np.ix_([0, 2], [1, 3])]).max(), np.abs(scratch.M[np.ix_([0, 2], [1, 3])]).max()))

    cost = []
    for T, ntraj in p["data_cost"]:
        for noise in p["noise_levels"]:
            r = np.random.default_rng(seed * 1000 + T + ntraj)
            Xd = W.simulate_exact(w3, [(T, L)], W.initial_states(w3, n, ntraj, p["amplitude"], r))
            Xd = W.add_noise(w3, Xd, noise, p["amplitude"], r)
            kron, _ = fit.fit_kronecker([(Xd, L)], 2)
            free = fit.fit_joint([(Xd, L)], d=2)[0]
            glob = fit.fit_global(Xd)
            row = {"steps": T, "trajectories": ntraj, "noise": noise}
            for name, S in (("kronecker", kron.assemble(L)), ("unconstrained", free.assemble(L)), ("global", glob)):
                with np.errstate(over="ignore", invalid="ignore"):
                    c, _ = curve_of(X3, S, steps)
                row[name] = float(c[steps]) if np.isfinite(c[steps]) else float("inf")
            cost.append(row)

    metrics_ = {
        "identity_max_abs": identity,
        "transfer": {"planar_maps_on_3d": metrics.at_horizons(c_transfer, H),
                     "planar_maps_on_planar": metrics.at_horizons(c_planar, H),
                     "3d_from_scratch": metrics.at_horizons(c_scratch, H)},
        "scratch_K_vs_K0": float(np.abs(scratch.K - K0).max()), "scratch_cross_direction_max": cross,
        "data_cost": cost,
    }
    arrays = {"curve_transfer": c_transfer, "curve_planar": c_planar, "curve_scratch": c_scratch,
              "example_actual": X3[0], "example_nominal": Xn_transfer[0]}
    return metrics_, arrays


if __name__ == "__main__":
    main("a1", run)
