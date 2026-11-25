"""H1: a clamped sense. Only angles are measured; can each body infer its hidden rate from its own clamped sense?

Step 1's maps K and M (learned on 3- and 5-body graphs, as in E2) are carried over frozen; nothing is learned here.
A ring of 8 with a random known wiring per run. Each estimator turns measured angles into a full state estimate:
  full      step 1's situation: angles and rates both measured (a reference, not available in step 2)
  delay     the last two readings determine the state (the plan's delay line)
  clamped   each body predicts with K and the spring messages, then corrects with its own clamped angle (fixed gain)
  precision the same clamp, with each body's prior precision growing from a cold start (local 2 x 2 Riccati)
  kalman    the centralised Kalman filter: the optimal linear reference for the same noise assumptions
The clamp's allowance for unmodelled forces, q, is chosen on rings of 5 (small graphs), then used on the ring of 8.
"""

import numpy as np
from common import main

from nsp import config, fit, graphs, sense, world as W

METHODS = ("full", "delay", "clamped", "precision", "kalman")


def learn(p, w, rng):
    train = []
    for n in p["train_sizes"]:
        for name in p["train_graphs"]:
            edges = graphs.named(name, n)
            for g in graphs.random_gate_patterns(len(edges), p["train_patterns"], rng):
                L = graphs.laplacian(n, edges, g)
                x0 = W.initial_states(w, n, p["train_trajectories"], p["amplitude"], rng)
                train.append((W.simulate_exact(w, [(p["train_steps"], L)], x0), L))
    return fit.fit_joint(train)[0]


def wiring(n, p_on, rng):
    edges = graphs.ring(n)
    g = (rng.random(len(edges)) < p_on).astype(float)
    return edges, g, graphs.laplacian(n, edges, g)


def ahead(model_, L, Xh, Xc, k):
    """Relative deviation of the k-step-ahead prediction from each estimate, against the actual (clean) state."""
    Pk = np.linalg.matrix_power(model_.assemble(L), k)
    pred = Xh[:len(Xh) - k] @ Pk.T
    return np.linalg.norm(pred - Xc[k:], axis=1) / np.linalg.norm(Xc[k:], axis=1), pred


def estimates(model_, L, w, Xc, noise, amp, q, rng):
    s2 = (noise * amp) ** 2
    Y = sense.observe(w, Xc, noise, amp, rng)
    k_gain, _ = sense.lone_gain(model_.K, s2, q)
    k_grow = sense.lone_gains(model_.K, s2, q, amp * w.omega0, len(Y) - 1)
    full = W.add_noise(w, Xc[None], noise, amp, rng)[0]
    return {"full": full, "delay": sense.delay_line(model_, L, Y), "clamped": sense.clamped_filter(model_, L, Y, k_gain),
            "precision": sense.clamped_filter(model_, L, Y, k_grow),
            "kalman": sense.kalman(model_, L, Y, s2, q, amp * w.omega0)}


def errors(Xh, Xc, amp, omega0):
    """Rate and angle errors per step: rms over bodies, relative to the rate and angle scales."""
    rate = np.sqrt(((Xh[:, 1::2] - Xc[:, 1::2]) ** 2).mean(1)) / (amp * omega0)
    angle = np.sqrt(((Xh[:, 0::2] - Xc[:, 0::2]) ** 2).mean(1)) / amp
    return rate, angle


def run(cfg: dict, seed: int):
    p = cfg["h1"]
    w = config.world_from(cfg)
    rng = np.random.default_rng(seed)
    n, amp, T, k = p["n"], p["amplitude"], p["steps"], p["horizon"]
    model_ = learn(p, w, rng)

    def simulate(nb, L):
        return W.simulate_exact(w, [(T, L)], W.initial_states(w, nb, 1, amp, rng))[0]

    def choose_q(noise):
        """The q with the smallest steady deviation 1 s ahead on rings of 5 (small graphs only)."""
        score = []
        for q in p["q_levels"]:
            s = []
            for _ in range(4):
                _, _, L5 = wiring(5, p["p_on"], rng)
                Xc = simulate(5, L5)
                Y = sense.observe(w, Xc, noise, amp, rng)
                kq, _ = sense.lone_gain(model_.K, (noise * amp) ** 2, q)
                s.append(ahead(model_, L5, sense.clamped_filter(model_, L5, Y, kq), Xc, k)[0][T // 2:].mean())
            score.append(float(np.mean(s)))
        return p["q_levels"][int(np.argmin(score))], score

    def batch(noise, q, trials, keep=False):
        curves = {m: {"rate": [], "angle": [], "ahead": []} for m in METHODS}
        example = None
        for i in range(trials):
            edges, g, L = wiring(n, p["p_on"], rng)
            Xc = simulate(n, L)
            est = estimates(model_, L, w, Xc, noise, amp, q, rng)
            for m, Xh in est.items():
                rate, angle = errors(Xh, Xc, amp, w.omega0)
                dev, pred = ahead(model_, L, Xh, Xc, k)
                curves[m]["rate"].append(rate)
                curves[m]["angle"].append(angle)
                curves[m]["ahead"].append(dev)
                if keep and i == 0:
                    example = example or {"theta_act": Xc[:, 0::2], "rate_act": Xc[:, 1::2], "gates": g, "L": L}
                    th_nom = np.vstack([np.tile(Xh[0, 0::2], (k, 1)), pred[:, 0::2]])
                    example[f"theta_nom_{m}"] = th_nom
                    example[f"rate_est_{m}"] = Xh[:, 1::2]
        mean = {m: {key: np.mean(v, 0) for key, v in c.items()} for m, c in curves.items()}
        steady = {m: {key: float(v[T // 2:].mean()) for key, v in c.items()} for m, c in mean.items()}
        for m in METHODS:  # steps until the rate error first falls below twice its steady value
            steady[m]["settle"] = int(np.argmax(mean[m]["rate"] < 2 * steady[m]["rate"]))
        return mean, steady, example

    # main condition
    q_main, q_score = choose_q(p["noise"])
    k_main, Pi_main = sense.lone_gain(model_.K, (p["noise"] * amp) ** 2, q_main)
    mean, steady, example = batch(p["noise"], q_main, p["trials"], keep=True)
    lam_grid = np.linspace(0, p["lambda_grid_max"], 161)
    rho_grid = sense.mode_radius(model_, k_main, lam_grid)
    lam_ex = np.clip(np.linalg.eigvalsh(example["L"]), 0, None)
    rho_ex = sense.mode_radius(model_, k_main, lam_ex)

    # convergence on a noiseless sense: does the slowest graph mode set the rate, as predicted? Checked in the model's
    # own world (exact match) and in the real one (the two-hop term the model misses keeps feeding the error)
    _, _, Lc = wiring(n, p["p_on"], rng)
    x0 = W.initial_states(w, n, 1, amp, rng)
    rho_c = float(sense.mode_radius(model_, k_main, np.clip(np.linalg.eigvalsh(Lc), 0, None)).max())
    seg = slice(60, 300)
    clean, slope = {}, {}
    for label, Xw in (("model", model_.rollout(x0, [(T, Lc)])[0]), ("world", W.simulate_exact(w, [(T, Lc)], x0)[0])):
        e = np.linalg.norm(sense.clamped_filter(model_, Lc, Xw[:, 0::2], k_main) - Xw, axis=1) / np.linalg.norm(Xw, axis=1)
        clean[label] = e
        slope[label] = float(np.polyfit(np.arange(T + 1)[seg], np.log(e[seg]), 1)[0])

    # one body's relaxation, at the first reading after a cold start
    xs, Es = sense.relax(np.array([example["theta_act"][0, 0], 0.0]), float(example["theta_act"][1, 0]), Pi_main,
                         (p["noise"] * amp) ** 2, steps=60)

    # noise sweep (q chosen afresh on small graphs at each level) and q sweep at the main noise
    by_noise = []
    for noise in p["noise_levels"]:
        qn, _ = choose_q(noise)
        _, st, _ = batch(noise, qn, max(3, p["trials"] // 2))
        by_noise.append({"noise": noise, "q": qn, "steady": st})
    by_q = []
    for q in p["q_levels"]:
        kq, _ = sense.lone_gain(model_.K, (p["noise"] * amp) ** 2, q)
        mq, st, _ = batch(p["noise"], q, max(3, p["trials"] // 2))
        rho = float(sense.mode_radius(model_, kq, lam_grid[lam_grid <= 4]).max())
        rate = mq["clamped"]["rate"]
        settle = int(np.argmax(rate < 2 * st["clamped"]["rate"]))
        by_q.append({"q": q, "gain": kq, "rho": rho, "halving": sense.halving_steps(rho), "settle": settle,
                     "steady": {m: st[m] for m in ("clamped", "precision", "kalman")}})

    metrics = {"q_main": q_main, "q_score": q_score, "gain": k_main, "steady": steady,
               "rho_max_ring": float(rho_ex.max()), "halving_ring": sense.halving_steps(float(rho_ex.max())),
               "clean_rho": rho_c, "clean_slope": slope, "predicted_slope": float(np.log(rho_c)),
               "settle": {m: steady[m]["settle"] for m in METHODS},
               "by_noise": by_noise, "by_q": by_q, "example_lambdas": lam_ex}
    arrays = {"lambda_grid": lam_grid, "rho_grid": rho_grid, "lambda_example": lam_ex, "rho_example": rho_ex,
              "clean_error_model": clean["model"], "clean_error_world": clean["world"], "relax_states": xs,
              "relax_energy": Es, "gains_grow": sense.lone_gains(model_.K, (p["noise"] * amp) ** 2, q_main,
                                                                 amp * w.omega0, T),
              "theta_act": example["theta_act"], "rate_act": example["rate_act"], "gates": example["gates"]}
    for m in METHODS:
        arrays[f"theta_nom_{m}"] = example[f"theta_nom_{m}"]
        arrays[f"rate_est_{m}"] = example[f"rate_est_{m}"]
        for key in ("rate", "angle", "ahead"):
            arrays[f"err_{key}_{m}"] = mean[m][key]
    return metrics, arrays


if __name__ == "__main__":
    main("h1", run)
