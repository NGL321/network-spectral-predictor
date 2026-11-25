"""The closed-form identities the plan's math rests on (rung 1, small swings)."""

import numpy as np
import pytest
from scipy.linalg import expm

from nsp import graphs, model, world as W

WORLD = W.World()


def test_spectrum_identity():
    rng = np.random.default_rng(1)
    edges = graphs.ring(8)
    L = graphs.laplacian(8, edges, (rng.random(len(edges)) < 0.6).astype(float))
    m = model.series(WORLD)
    big = np.linalg.eigvals(m.assemble(L))
    lams, mus = m.spectrum(L)
    small = np.concatenate(mus)
    assert max(np.abs(small - z).min() for z in big) < 1e-12


def test_K0_closed_form():
    w0, dt = WORLD.omega0, WORLD.dt
    K0 = model.series(WORLD).K
    cf = np.array([[np.cos(w0 * dt), np.sin(w0 * dt) / w0], [-w0 * np.sin(w0 * dt), np.cos(w0 * dt)]])
    assert np.allclose(K0, cf, atol=1e-14)


def test_M0_series_to_dt_cubed():
    M0 = model.series(WORLD).M
    M0_cf, _ = model.series_closed_form(WORLD)
    assert np.abs(M0 - M0_cf).max() < 3e-6


@pytest.mark.parametrize("name,n", [("path", 3), ("ring", 8), ("star", 8)])
def test_two_hop_term_is_what_one_hop_misses(name, n):
    L = graphs.laplacian(n, graphs.named(name, n))
    exact = W.exact_step(WORLD, L)
    one_hop = model.series(WORLD).assemble(L)
    _, N0 = model.series_closed_form(WORLD)
    residual = exact - one_hop
    assert np.abs(residual - np.kron(L @ L, N0)).max() < 2e-5
    assert np.abs(residual).max() > 1e-4  # the two-hop term itself is clearly there


def test_growth_and_frequency_formulas():
    for kappa in (1.0, 4.0):
        w = WORLD.with_(kappa=kappa)
        m = model.series(w)
        for lam in (1.0, 4.0, 8.0):
            wn, mod = m.mode_frequencies([lam], w.dt)
            w_true = np.sqrt(w.omega0**2 + kappa * lam)
            assert mod[0] - 1 == pytest.approx(model.predicted_growth(w, lam), rel=0.05)
            assert (wn[0] - w_true) / w_true == pytest.approx(model.predicted_frequency_error(w, lam), rel=0.1)


def test_two_angles_small_swings_are_two_planar_copies():
    L = graphs.laplacian(8, graphs.ring(8))
    planar = W.exact_step(WORLD, L)
    spherical = W.exact_step(WORLD.with_(d=2), L)
    assert np.allclose(spherical, np.kron(planar, np.eye(2)), atol=1e-13)


def test_complete_graph_two_frequencies():
    lams = graphs.graph_frequencies(8, graphs.complete(8))
    assert np.allclose(np.unique(np.round(lams, 9)), [0.0, 8.0])


def test_mode_frequencies_match_exact_step():
    L = graphs.laplacian(3, graphs.path(3))
    ev = np.linalg.eigvals(W.exact_step(WORLD, L))
    w_exact = np.sort(np.abs(np.angle(ev)))[::2] / WORLD.dt
    assert np.allclose(np.sort(w_exact), np.sort(W.mode_frequencies(WORLD, L)), atol=1e-12)
    assert np.allclose(np.sort(W.mode_frequencies(WORLD, L)), [2.0, np.sqrt(5), np.sqrt(7)])


def test_verlet_matches_exact_when_linear():
    L = graphs.laplacian(3, graphs.path(3))
    x0 = W.initial_states(WORLD, 3, 2, 0.1, np.random.default_rng(0))
    sched = [(100, L)]
    Xe = W.simulate_exact(WORLD, sched, x0)
    Xv = W.simulate_verlet(WORLD, sched, x0, nonlinear=False)
    assert np.abs(Xv - Xe).max() / np.abs(Xe).max() < 1e-5


def test_expm_sanity():
    assert np.allclose(expm(np.zeros((2, 2))), np.eye(2))
