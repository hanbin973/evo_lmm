import numpy as np

from evo_lmm import EvolutionaryLmmOps, SimplifiedPrior, fit_reml
from evo_lmm.trace import spherical_gaussian_probes, xtrace


def test_spherical_gaussian_probes_have_fixed_radius_and_seed():
    first = spherical_gaussian_probes(31, 5, seed=17)
    second = spherical_gaussian_probes(31, 5, seed=17)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=0), np.sqrt(31.0))


def test_xtrace_is_exact_for_full_rank_diagonal_sketch():
    matrix = np.diag(np.arange(1.0, 11.0))
    probes = spherical_gaussian_probes(matrix.shape[0], 10, seed=3)
    estimate = xtrace(lambda values: matrix @ values, probes)
    np.testing.assert_allclose(estimate.value, np.trace(matrix), rtol=1e-10, atol=1e-10)
    assert estimate.estimator == "xtrace"


def test_xtrace_rank_fallback_remains_unbiased_shape():
    matrix = np.ones((12, 12))
    probes = spherical_gaussian_probes(12, 5, seed=4)
    estimate = xtrace(lambda values: matrix @ values, probes)
    assert np.isfinite(estimate.value)
    assert np.isfinite(estimate.standard_error)
    assert estimate.estimator == "xtrace-rank-fallback"


def test_projected_cg_warm_start_matches_cold_solution_and_rejects_poor_guess():
    rng = np.random.default_rng(21)
    dosage = rng.binomial(2, 0.35, size=(26, 9)).astype(float)
    ops = EvolutionaryLmmOps.from_dense(dosage, dosage.mean(axis=0) / 2.0)
    phi = SimplifiedPrior(1.0, 0.7).coordinates(0.4)
    rhs = rng.normal(size=(ops.n, 3))
    cold_stats = {}
    cold = ops.solve_ph(rhs, phi, tol=1e-11, stats=cold_stats)
    warm_stats = {}
    warm = ops.solve_ph(rhs, phi, tol=1e-11, initial=cold, stats=warm_stats)
    np.testing.assert_allclose(warm, cold, atol=1e-9)
    assert warm_stats["warm_used"] == rhs.shape[1]

    poor_stats = {}
    poor = ops.solve_ph(rhs, phi, tol=1e-11, initial=rng.normal(size=rhs.shape) * 1e6, stats=poor_stats)
    np.testing.assert_allclose(poor, cold, atol=1e-9)
    assert poor_stats["warm_rejected"] == rhs.shape[1]

    shape_stats = {}
    shaped = ops.solve_ph(rhs, phi, tol=1e-11, initial=cold[:, :1], stats=shape_stats)
    np.testing.assert_allclose(shaped, cold, atol=1e-9)
    assert shape_stats["warm_rejected"] == rhs.shape[1]


def test_matrix_free_fit_reports_xtrace_query_budget():
    rng = np.random.default_rng(22)
    dosage = rng.binomial(2, 0.4, size=(24, 8)).astype(float)
    ops = EvolutionaryLmmOps.from_dense(dosage, dosage.mean(axis=0) / 2.0)
    fit = fit_reml(
        ops,
        rng.normal(size=ops.n),
        initial=SimplifiedPrior(1.0, 0.5),
        exact=False,
        trace_probes=5,
        max_iter=1,
        cg_tol=1e-7,
        trace_method="xtrace",
    )
    assert fit.diagnostics.trace_estimator == "xtrace"
    assert fit.diagnostics.trace_probes == 5
    assert fit.diagnostics.trace_operator_queries == 10
