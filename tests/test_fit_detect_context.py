"""Learning, detection and recall on clean data."""

import numpy as np

from nsp import context, detect, fit, graphs, model, world as W

WORLD = W.World()


def _path3_data(seed=0, d=1):
    w = WORLD.with_(d=d)
    rng = np.random.default_rng(seed)
    edges = graphs.path(3)
    sets = []
    for g in graphs.all_gate_patterns(len(edges)):
        L = graphs.laplacian(3, edges, g)
        sets.append((W.simulate_exact(w, [(200, L)], W.initial_states(w, 3, 10, 0.1, rng)), L))
    return sets


def test_joint_fit_recovers_series():
    m, info = fit.fit_joint(_path3_data())
    s = model.series(WORLD)
    assert np.abs(m.K - s.K).max() < 1e-4
    assert np.abs(m.M - s.M).max() < 2e-4
    assert info["condition"] < 1e3


def test_frozen_K_is_exact():
    rng = np.random.default_rng(3)
    lone = W.simulate_exact(WORLD, [(200, np.zeros((3, 3)))], W.initial_states(WORLD, 3, 10, 0.1, rng))
    m = fit.fit_frozen(lone, _path3_data())
    assert np.abs(m.K - model.series(WORLD).K).max() < 1e-10


def test_gradient_descent_reaches_closed_form():
    data = _path3_data()
    m_ls, _ = fit.fit_joint(data)
    m_gd, info = fit.gradient_descent(data, tol=1e-7)
    assert info["distance"][-1] < 1e-7
    assert np.abs(m_gd.K - m_ls.K).max() < 1e-6


def test_parameter_recovery():
    m, _ = fit.fit_joint(_path3_data())
    p = fit.recover_parameters(m, WORLD)
    assert abs(p["omega0_sq"] - 4.0) < 1e-3
    assert abs(p["kappa"] - 1.0) < 1e-2


def test_kronecker_fit_two_angles():
    m, _ = fit.fit_kronecker(_path3_data(d=2), d=2)
    assert np.allclose(m.K, np.kron(model.series(WORLD).K, np.eye(2)), atol=1e-4)


def test_detection_one_spring_needs_two_trajectories_or_nnls():
    n, ring, pairs = 5, graphs.ring(5), graphs.complete(5)
    g = np.array([0, 0, 0, 1, 0.0])
    L = graphs.laplacian(n, ring, g)
    m = model.series(WORLD)
    truth = detect.truth_vector(pairs, ring, g)
    X1 = W.simulate_exact(WORLD, [(200, L)], W.initial_states(WORLD, n, 1, 0.1, np.random.default_rng(5)))
    assert detect.design_singular_values(X1, m, pairs, n)[-1] < 1e-10  # the blind spot
    assert detect.exact_match(detect.detect_gates(X1, m, pairs, n, method="nnls"), truth)
    X3 = W.simulate_exact(WORLD, [(200, L)], W.initial_states(WORLD, n, 3, 0.1, np.random.default_rng(5)))
    assert detect.design_singular_values(X3, m, pairs, n)[-1] > 1e-2
    assert detect.exact_match(detect.detect_gates(X3, m, pairs, n, method="lstsq"), truth)


def test_recall_and_relock():
    n, ring = 5, graphs.ring(5)
    rng = np.random.default_rng(0)
    pats = graphs.all_gate_patterns(5)[[3, 9, 14, 21, 27, 31]]
    bank = context.ContextBank(n, ring, pats)
    m = model.series(WORLD)
    cal = W.add_noise(WORLD, W.simulate_exact(WORLD, [(500, bank.laplacians()[0])],
                                              W.initial_states(WORLD, n, 2, 0.1, rng)), 0.01, 0.1, rng)
    scale = context.residual_scale(cal, m.assemble(bank.laplacians()[0]))
    bv = context.BindingVertex(bank, m, scale)
    Ls = bank.laplacians()
    X = W.add_noise(WORLD, W.simulate_exact(WORLD, [(100, Ls[1]), (100, Ls[4])],
                                            W.initial_states(WORLD, n, 1, 0.1, rng)), 0.01, 0.1, rng)[0]
    pi = bv.filter(X)
    assert np.argmax(pi[90]) == 1 and np.argmax(pi[130]) == 4


def test_design_steps_stack_into_the_window_design():
    n, ring, pairs = 5, graphs.ring(5), graphs.complete(5)
    L = graphs.laplacian(n, ring, [1, 0, 1, 1, 0])
    m = model.series(WORLD)
    X = W.simulate_exact(WORLD, [(60, L)], W.initial_states(WORLD, n, 1, 0.1, np.random.default_rng(2)))
    G, r = detect.design(X, m, pairs, n)
    Gs, rs = detect.design_steps(X[0], m, pairs, n)
    assert np.allclose(Gs.reshape(-1, len(pairs)), G) and np.allclose(rs.reshape(-1), r)
    g = detect.sliding_gates(Gs, rs, 20)
    assert detect.exact_match(g[-1], detect.truth_vector(pairs, ring, [1, 0, 1, 1, 0]))


def test_transition_priors():
    pats = graphs.all_gate_patterns(8)
    for T in (context.uniform_transitions(len(pats), 0.02), context.flip_transitions(pats, 0.02)):
        assert np.allclose(T.sum(1), 1) and np.allclose(np.diag(T), 0.98)
    F = context.flip_transitions(pats, 0.02)
    assert np.all((F > 0).sum(1) == 9)  # stay, or flip one of the 8 springs
