"""Stage 1.5, part A: oriented springs (the sheaf), the spherical world, the sheaf model and its fits."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import nnls

from nsp import graphs, model, sheaf, sphere, world as W

PLANAR = W.World()
SPHERE = W.World(d=2)


def _layout(n=6, seed=0, chords=1):
    rng = np.random.default_rng(seed)
    P = sheaf.layout(n, rng)
    return P, sheaf.add_chords(P, sheaf.angular_ring(P), chords, rng)


def test_isotropic_springs_are_the_planar_step_twice():
    P, edges = _layout()
    L = graphs.laplacian(len(P), edges)
    iso = sheaf.exact_step(SPHERE, P, edges, oriented=False)
    assert np.allclose(iso, np.kron(W.exact_step(PLANAR, L), np.eye(2)), atol=1e-13)


def test_sheaf_laplacian_is_the_rigidity_gram_and_the_spectrum_splits():
    P, edges = _layout()
    g = np.array([1.0, 0, 1, 1, 0, 1, 1])
    R = sheaf.rigidity_matrix(P, edges, g)
    assert np.allclose(sheaf.sheaf_laplacian(P, edges, g), R.T @ R, atol=1e-13)
    m = sheaf.SheafModel.from_planar(model.series(PLANAR))
    big = np.linalg.eigvals(m.assemble(P, edges, g))
    mu, blocks = m.mode_blocks(P, edges, g)
    small = np.concatenate([np.linalg.eigvals(B) for B in blocks])
    assert max(np.abs(small - z).min() for z in big) < 1e-12


def test_oriented_world_is_the_sheaf_model_to_one_hop_accuracy():
    P, edges = _layout()
    exact = sheaf.exact_step(SPHERE, P, edges)
    one_hop = sheaf.SheafModel.from_planar(model.series(PLANAR)).assemble(P, edges)
    lam = np.linalg.eigvalsh(sheaf.sheaf_laplacian(P, edges)).max()
    assert np.abs(exact - one_hop).max() < 2 * (PLANAR.kappa * lam) ** 2 * PLANAR.dt**3 / 6


def test_fit_sheaf_recovers_step_ones_maps():
    rng = np.random.default_rng(3)
    sets = []
    for n in (3, 5, 3, 5):
        P = sheaf.layout(n, rng)
        edges = sheaf.add_chords(P, sheaf.angular_ring(P), 1 if n == 5 else 0, rng)
        X = sheaf.simulate_exact(SPHERE, P, edges, [(200, None)], sphere.initial_states(n, 10, 0.1, rng))
        sets.append((X, P, edges, None))
    ref = model.series(PLANAR)
    for iso in (True, False):
        m, _ = sheaf.fit_sheaf(sets, isotropic_body=iso)
        assert np.abs(m.K - np.kron(ref.K, np.eye(2))).max() < 1e-4
        assert np.abs(m.M - ref.M).max() < 0.01 * np.abs(ref.M).max()


def test_zero_modes_count_the_frameworks_motions():
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.9]])
    assert sheaf.zero_modes(tri, [(0, 1), (1, 2), (2, 0)]) == 3
    P = sheaf.layout(5, np.random.default_rng(3))
    ring = sheaf.angular_ring(P)
    assert sheaf.zero_modes(P, ring) == 5 and sheaf.mechanisms(P, ring).shape[1] == 2
    braced = ring + [(ring[0][0], ring[1][1]), (ring[2][0], ring[3][1])]
    assert sheaf.zero_modes(P, braced) == 3 and sheaf.mechanisms(P, braced).shape[1] == 0
    Z = sheaf.mechanisms(P, ring)
    assert np.abs(sheaf.sheaf_laplacian(P, ring) @ Z).max() < 1e-12  # no spring feels a mechanism


def test_rattle_reduces_to_the_oriented_world_for_small_swings():
    """The springs' geometry is nonlinear at first order in swing / spacing (a sideways offset stretches a spring
    quadratically), so the gap to the linear world shrinks in proportion to the amplitude."""
    P, edges = _layout()
    gaps = []
    for amp in (1e-3, 1e-4):
        x0 = sphere.initial_states(len(P), 2, amp, np.random.default_rng(1))
        Xr = sphere.simulate(SPHERE, P, edges, [(200, None)], x0)
        Xe = sheaf.simulate_exact(SPHERE, P, edges, [(200, None)], x0)
        gaps.append(np.abs(Xr - Xe).max() / np.abs(Xe).max())
    assert gaps[1] < 2e-4 and 5 < gaps[0] / gaps[1] < 20


def test_rattle_conserves_energy_at_large_swings():
    P, edges = _layout()
    x0 = sphere.initial_states(len(P), 2, 0.4, np.random.default_rng(2))
    X = sphere.simulate(SPHERE, P, edges, [(400, None)], x0)
    E = sphere.energy(SPHERE, P, edges, X)
    E0 = E[:, :1]
    assert np.abs(E - E0).max() < 1e-5 * np.abs(E0 - sphere.energy(SPHERE, P, edges, 0 * X[:, :1])).max()


def test_lone_spherical_pendulum_matches_the_chart_equation():
    w2 = SPHERE.omega0**2

    def rhs(t, s):
        q, v = s[:2], s[2:]
        c = np.sqrt(1 - q @ q)
        return np.concatenate([v, -(w2 * c + v @ v + (q @ v) ** 2 / c**2) * q])

    s0 = np.array([0.3, 0.0, 0.0, 0.3])
    T = 200
    sol = solve_ivp(rhs, (0, T * SPHERE.dt), s0, method="DOP853", rtol=1e-12, atol=1e-14,
                    t_eval=np.arange(T + 1) * SPHERE.dt).y.T
    X = sphere.simulate(SPHERE, np.zeros((1, 2)), [], [(T, None)], s0[[0, 1, 2, 3]][None])[0]
    assert np.abs(X - sol).max() < 1e-5


def test_detection_along_known_axes_is_exact_on_clean_data():
    rng = np.random.default_rng(4)
    P = sheaf.layout(5, rng)
    pairs = graphs.complete(5)
    edges = sheaf.add_chords(P, sheaf.angular_ring(P), 2, rng)
    X = sheaf.simulate_exact(SPHERE, P, edges, [(60, None)], sphere.initial_states(5, 1, 0.1, rng))[0]
    G, r = sheaf.design_steps(X, sheaf.SheafModel.from_planar(model.series(PLANAR)), P, pairs)
    g = nnls(G.reshape(-1, len(pairs)), r.reshape(-1))[0]
    truth = np.array([1.0 if tuple(sorted(p)) in {tuple(sorted(e)) for e in edges} else 0.0 for p in pairs])
    assert np.array_equal(g > 0.5, truth > 0.5)
