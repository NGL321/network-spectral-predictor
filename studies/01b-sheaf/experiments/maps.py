"""Step 1's planar maps, learned the way E2 learned them, for the stage 1.5 experiments to carry over."""

from __future__ import annotations

from nsp import fit, graphs, world as W


def planar_maps(p: dict, w2: W.World, rng):
    """K and M from all-on paths and rings of p["train_sizes"] bodies (p["planar_graphs"]), planar world w2."""
    sets = []
    for n in p["train_sizes"]:
        for name in p.get("planar_graphs", ["path", "ring"]):
            L = graphs.laplacian(n, graphs.named(name, n))
            x0 = W.initial_states(w2, n, p["train_trajectories"], p["amplitude"], rng)
            sets.append((W.simulate_exact(w2, [(p["train_steps"], L)], x0), L))
    return fit.fit_joint(sets)[0]
