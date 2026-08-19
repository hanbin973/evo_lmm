import numpy as np

from evo_lmm import EvolutionaryLmmOps, FullPrior, SimplifiedPrior, exact_reml_score, fit_reml
from evo_lmm.reml import _profile_objective_dense, _quantities, haseman_elston_initialization


def test_dense_reml_score_matches_profiled_objective_derivative():
    rng = np.random.default_rng(8)
    x = rng.binomial(2, 0.35, size=(28, 8)).astype(float)
    frequencies = x.mean(axis=0) / 2.0
    ops = EvolutionaryLmmOps.from_dense(x, frequencies, model="simplified")
    y = rng.normal(size=ops.n)
    coordinates = np.array([0.2, -0.4])
    values = _quantities(ops, y, coordinates, np.ones((ops.n, 5)), exact=True)
    for index in range(2):
        step = np.zeros(2)
        step[index] = 1e-5
        numerical = (_profile_objective_dense(ops, y, coordinates + step) - _profile_objective_dense(ops, y, coordinates - step)) / 2e-5
        assert np.isclose(numerical, -values.score[index], rtol=2e-5, atol=2e-6)


def test_full_r_one_has_same_kernel_and_fit_shape_as_simplified():
    rng = np.random.default_rng(9)
    x = rng.binomial(2, 0.25, size=(24, 7)).astype(float)
    frequencies = x.mean(axis=0) / 2.0
    simplified = EvolutionaryLmmOps.from_dense(x, frequencies, model="simplified")
    full = EvolutionaryLmmOps.from_dense(x, frequencies, model="full")
    p_s = SimplifiedPrior(1.0, 0.6)
    p_f = FullPrior(1.0, 0.6, 1.0)
    assert np.array_equal(simplified.dense_kernel(p_s), full.dense_kernel(p_f))
    y = rng.normal(size=x.shape[0])
    fit_s = fit_reml(simplified, y, initial=p_s, max_iter=8, exact=True)
    fit_f = fit_reml(full, y, initial=p_f, max_iter=8, exact=True)
    assert np.isclose(fit_s.sigma_b2, fit_f.sigma_b2, rtol=1e-6)
    assert np.isclose(fit_s.sigma_e2, fit_f.sigma_e2, rtol=1e-6)


def test_matrix_free_projected_solve_and_seeded_trace_are_reproducible():
    rng = np.random.default_rng(10)
    x = rng.binomial(2, 0.4, size=(22, 6)).astype(float)
    ops = EvolutionaryLmmOps.from_dense(x, x.mean(axis=0) / 2.0)
    prior = SimplifiedPrior(1.0, 0.9)
    phi = prior.coordinates(0.7)
    y = rng.normal(size=ops.n)
    exact = ops.solve_ph(y, phi)
    approx = ops.solve_ph(y, phi, tol=1e-11)
    assert np.allclose(exact, approx, atol=1e-8)
    rhs = np.column_stack((y, np.roll(y, 1), np.ones(ops.n)))
    batch = ops.solve_ph(rhs, phi, tol=1e-11)
    assert np.allclose(batch, np.column_stack([ops.solve_ph(rhs[:, i], phi, tol=1e-11) for i in range(rhs.shape[1])]), atol=1e-8)


def test_haseman_elston_initialization_is_available_as_fit_mode():
    rng = np.random.default_rng(11)
    x = rng.binomial(2, 0.35, size=(26, 9)).astype(float)
    ops = EvolutionaryLmmOps.from_dense(x, x.mean(axis=0) / 2.0, model="simplified")
    y = rng.normal(size=ops.n)
    sigma_b2, sigma_e2, delta = haseman_elston_initialization(
        ops, y, SimplifiedPrior(1.0, 0.3), seed=3
    )
    assert sigma_b2 > 0.0 and sigma_e2 > 0.0 and delta > 0.0
    fit = fit_reml(ops, y, initialization="he", max_iter=2, exact=True)
    assert fit.diagnostics.initialization == "he"
    assert np.isfinite(fit.delta) and fit.delta > 0.0


def test_unknown_initialization_mode_is_rejected():
    x = np.ones((8, 3), dtype=float)
    ops = EvolutionaryLmmOps.from_dense(x, np.full(3, 0.5))
    with np.testing.assert_raises(ValueError):
        fit_reml(ops, np.arange(8, dtype=float), initialization="nonsense")
