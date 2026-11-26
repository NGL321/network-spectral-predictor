"""A4: large swings. The bobs' own paths bend and precess, and the springs' axes turn with the bobs: which matters?

The world is RATTLE (`nsp.sphere`): rigid rods, real springs between the bobs, natural length at rest.
  precession  one spherical pendulum started on a narrow ellipse (tilt a, sideways rate b omega0): its apsides turn at
              (3/8) omega0 a b, a rotation no per-direction model can make
  lifts       one pendulum, lifts compared: none (4 numbers), per direction (12), rotation-equivariant cubic (16), every
              odd cubic (24), rotation-equivariant quintic (28)
  networks    8-body layouts at swings of 0.1, 0.3 and 0.5 rod lengths, maps learned on 3 and 5 bodies at the same swing:
              body lift (equivariant quintic; cubic as a check) or none, springs read along fixed axes (A2) or from the
              current geometry, projected on each bob's sphere
"""

import itertools

import numpy as np
from common import main

from nsp import config, metrics, sheaf, sphere

MODELS = {"linear, fixed axes": ("none", "fixed"), "lifted, fixed axes": ("quintic", "fixed"),
          "linear, moving axes": ("none", "moving"), "lifted, moving axes": ("quintic", "moving"),
          "cubic lift, moving axes": ("cubic", "moving")}
LIFTS = ("none", "per direction", "equivariant", "every odd cubic", "equivariant quintic")


def lone_features(x, kind):
    q, v = x[:, :2], x[:, 2:]
    if kind == "none":
        return x
    if kind == "per direction":
        cols = [x]
        for i in range(2):
            qi, vi = q[:, i:i + 1], v[:, i:i + 1]
            cols += [qi**3, qi**2 * vi, qi * vi**2, vi**3]
        return np.hstack(cols)
    if kind == "equivariant":
        return sphere.LiftedSheafModel.features(x, "cubic")
    if kind == "equivariant quintic":
        return sphere.LiftedSheafModel.features(x, "quintic")
    mons = [np.prod(x[:, list(c)], axis=1, keepdims=True) for c in itertools.combinations_with_replacement(range(4), 3)]
    return np.hstack([x] + mons)


def lone_runs(w, amp, count, steps, sub, rng):
    x0 = np.zeros((count, 4))
    r1, r2 = rng.uniform(0.3, 1, count), rng.uniform(0.3, 1, count)
    a1, a2 = rng.uniform(0, 2 * np.pi, count), rng.uniform(0, 2 * np.pi, count)
    x0[:, 0], x0[:, 1] = amp * r1 * np.cos(a1), amp * r1 * np.sin(a1)
    x0[:, 2], x0[:, 3] = w.omega0 * amp * r2 * np.cos(a2), w.omega0 * amp * r2 * np.sin(a2)
    return sphere.simulate(w, np.zeros((1, 2)), [], [(steps, None)], x0, sub)


def precession_rate(X, dt):
    q = X[:, :2]
    r2 = (q**2).sum(1)
    pk = [i for i in range(1, len(r2) - 1) if r2[i] > r2[i - 1] and r2[i] >= r2[i + 1]]
    ang = np.unwrap(2 * np.arctan2(q[pk, 1], q[pk, 0])) / 2
    return float(np.polyfit(np.array(pk) * dt, ang, 1)[0])


def layout(n, chords, p, rng):
    P = sheaf.layout(n, rng, p["box"], p["min_sep"])
    return P, sheaf.add_chords(P, sheaf.angular_ring(P), chords, rng)


def run(cfg: dict, seed: int):
    p = cfg["a4"]
    w = config.world_from(cfg)
    rng = np.random.default_rng(seed)
    sub, steps, H = p["substeps"], p["steps"], p["horizons"]

    prec = []
    if seed == cfg["run"]["seeds"][0]:  # deterministic: computed once
        for a, b in p["precession"]:
            X = sphere.simulate(w, np.zeros((1, 2)), [], [(p["precession_steps"], None)],
                                np.array([[a, 0.0, 0.0, w.omega0 * b]]), sub)[0]
            prec.append({"a": a, "b": b, "measured": precession_rate(X, w.dt), "predicted": 3 / 8 * w.omega0 * a * b})

    lone = []
    for amp in p["lone_amplitudes"]:
        Xtr = lone_runs(w, amp, p["lone_train"], steps, sub, rng)
        Xte = lone_runs(w, amp, p["lone_test"], steps, sub, rng)
        row = {"amplitude": amp}
        for kind in LIFTS:
            Wl = np.linalg.lstsq(lone_features(Xtr[:, :-1].reshape(-1, 4), kind), Xtr[:, 1:].reshape(-1, 4), rcond=None)[0]
            Xn = np.empty_like(Xte)
            Xn[:, 0] = Xte[:, 0]
            for t in range(steps):
                Xn[:, t + 1] = lone_features(Xn[:, t], kind) @ Wl
            row[kind] = metrics.at_horizons(metrics.rollout_curve(Xte, Xn), H)
        lone.append(row)

    nets, curves = [], {}
    for amp in p["amplitudes"]:
        sets = []
        for n, chords in zip(p["train_sizes"], p["train_chords"]):
            for _ in range(p["train_layouts"]):
                P, edges = layout(n, chords, p, rng)
                x0 = sphere.initial_states(n, p["train_trajectories"], amp, rng, w.omega0)
                sets.append((sphere.simulate(w, P, edges, [(p["train_steps"], None)], x0, sub), P, edges, None))
        models = {name: sphere.fit_lifted(sets, lift, reads) for name, (lift, reads) in MODELS.items()}
        rows = {name: [] for name in list(MODELS) + ["floor"]}
        closest = []
        for k in range(p["test_layouts"]):
            P, edges = layout(p["test_n"], p["test_chords"], p, rng)
            x0 = sphere.initial_states(p["test_n"], p["test_trajectories"], amp, rng, w.omega0)
            X = sphere.simulate(w, P, edges, [(steps, None)], x0, sub)
            b, _ = sphere.to_bobs(P, X)
            gap = np.linalg.norm(b[:, :, :, None] - b[:, :, None, :], axis=-1) + 1e9 * np.eye(len(P))
            closest += list(gap.min(axis=(1, 2, 3)))
            # the chaos floor: the world itself, restarted from the state as measured with noise floor_noise x the swing
            scale = np.tile([amp, amp, amp * w.omega0, amp * w.omega0], p["test_n"])
            xf = x0 + p["floor_noise"] * scale * rng.normal(size=x0.shape)
            nominal = {name: m.rollout(X[:, 0], P, edges, steps) for name, m in models.items()}
            nominal["floor"] = sphere.simulate(w, P, edges, [(steps, None)], xf, sub)
            for name, Xn in nominal.items():
                with np.errstate(all="ignore"):
                    c = np.linalg.norm(Xn - X, axis=2) / np.linalg.norm(X, axis=2)  # per trajectory
                rows[name].append(np.where(np.isfinite(c), c, np.inf))
        row = {"amplitude": amp, "M_lifted_moving": models["lifted, moving axes"].M,
               "closest_bobs_median": float(np.median(closest)), "closest_bobs_min": float(np.min(closest))}
        for name in rows:
            C = np.vstack(rows[name])  # (trajectories, steps + 1)
            curves[f"{amp}_{name}"] = np.median(C, 0)
            row[name] = {int(h): float(np.median(C[:, h])) for h in H}
            row[name]["p90_200"] = float(np.quantile(C[:, H[-1]], 0.9))
            row[name]["diverged"] = int(np.sum(~np.isfinite(C[:, H[-1]])))
        nets.append(row)

    arrays = {f"curve_{k.replace(' ', '_').replace(',', '')}": v for k, v in curves.items()}
    return {"precession": prec, "lone_lifts": lone, "networks": nets}, arrays


if __name__ == "__main__":
    main("a4", run)
