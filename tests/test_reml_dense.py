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


def test_convergence_statistics_are_scale_free_and_reject_rank_deficiency():
    """The criterion must be sample-size stable and never converge on a null AI.

    ``||score||_inf`` grows with ``n`` at a fixed statistical distance from the
    optimum, which is why it is not the gate.  The rank-deficient case is a
    regression: ``numpy.linalg.pinv`` inverts the small *negative* eigenvalues a
    stochastic average-information matrix carries, producing negative variances
    that read as zero standard errors and therefore as convergence.
    """
    from evo_lmm.reml import convergence_statistics, psd_pseudo_inverse

    # A well-conditioned problem: the statistic is the Newton step measured in
    # standard errors, and both statistics are invariant to a rescaling of the
    # information that leaves score/information consistent.
    ai = np.array([[4.0, 1.0], [1.0, 2.0]])
    score = np.array([0.4, -0.2])
    inverse = np.linalg.inv(ai)
    expected_step = inverse @ score
    expected_se = np.sqrt(np.diag(inverse))
    step_se, decrement = convergence_statistics(score, ai)
    np.testing.assert_allclose(step_se, np.max(np.abs(expected_step) / expected_se), rtol=1e-12)
    np.testing.assert_allclose(decrement, np.sqrt(score @ inverse @ score), rtol=1e-12)

    # The statistic is invariant under a diagonal reparameterization
    # ``theta -> D theta`` (score -> D^-1 s, AI -> D^-1 AI D^-1), which the raw
    # score is not: a coordinate rescaling changes ||score||_inf at will.
    scaling = np.array([1e3, 1e-2])
    rescaled_score = score / scaling
    rescaled_ai = ai / np.outer(scaling, scaling)
    rescaled_step_se, rescaled_decrement = convergence_statistics(rescaled_score, rescaled_ai)
    np.testing.assert_allclose(rescaled_step_se, step_se, rtol=1e-10)
    np.testing.assert_allclose(rescaled_decrement, decrement, rtol=1e-10)
    assert np.max(np.abs(rescaled_score)) > 10.0 * np.max(np.abs(score))

    # Rank-one information with a numerically negative null eigenvalue, as a
    # symmetrised stochastic AI produces when two coordinates become
    # indistinguishable (this one is taken from a fit whose tau ran away).
    identified = np.array([1.0, 1.0]) / np.sqrt(2.0)
    null = np.array([1.0, -1.0]) / np.sqrt(2.0)
    singular = (2.2625e-4 * np.outer(identified, identified)
                - 2.0577e-15 * np.outer(null, null))
    eigenvalues = np.linalg.eigvalsh(singular)
    assert eigenvalues[0] < 0.0 < eigenvalues[1]
    assert np.all(np.diag(np.linalg.pinv(singular)) < 0.0), "fixture no longer trips numpy pinv"
    inverse, retained = psd_pseudo_inverse(singular)
    assert retained == 1
    assert np.all(np.diag(inverse) >= 0.0)
    singular_step_se, singular_decrement = convergence_statistics(
        np.array([7.1e-3, 7.1e-3]), singular
    )
    assert singular_step_se > 1e-2, "a rank-deficient AI must not read as converged"
    assert np.isfinite(singular_decrement) and singular_decrement > 0.0

    # No usable direction at all is reported as unjudgeable, not as converged.
    assert convergence_statistics(np.array([1.0, 1.0]), np.zeros((2, 2))) == (
        float("inf"), float("inf")
    )
    assert convergence_statistics(np.array([1.0]), np.array([[np.nan]])) == (
        float("inf"), float("inf")
    )


def test_score_norm_grows_with_sample_size_while_the_criterion_does_not():
    """The empirical reason the gate moved off ``||score||_inf``.

    Both quantities are evaluated at the *generating* parameters, i.e. at a
    fixed statistical distance from the optimum.  There ``||score||_inf`` grows
    roughly like ``sqrt(n)`` -- so a fixed absolute score tolerance demands
    getting ``sqrt(n)`` times closer in standard-error units as data are added
    -- while the step/standard-error statistic stays order one.
    """
    from evo_lmm.reml import _quantities, convergence_statistics
    from evo_lmm.trace import rademacher_probes

    prior = SimplifiedPrior(0.02, 20.0)
    residual_variance = 1.0
    coordinates = np.array([np.log(residual_variance / prior.sigma_b2), np.log(prior.tau)])
    raw, scaled = [], []
    for n in (250, 1000):
        rng = np.random.default_rng(4)
        population = np.geomspace(0.01, 0.5, 120)
        dosage = rng.binomial(2, population, size=(n, 120)).astype(float)
        frequencies = dosage.mean(axis=0) / 2.0
        ops = EvolutionaryLmmOps.from_dense(dosage, frequencies)
        y = dosage @ rng.normal(0.0, np.sqrt(prior.effect_variances(frequencies))) + rng.normal(
            0.0, np.sqrt(residual_variance), n
        )
        quantities = _quantities(ops, y, coordinates, rademacher_probes(n, 8, 1), exact=True)
        raw.append(float(np.linalg.norm(quantities.score, ord=np.inf)))
        scaled.append(convergence_statistics(quantities.score, quantities.ai)[0])
    assert raw[1] > 1.3 * raw[0], f"raw score did not grow with n: {raw}"
    assert 0.25 < scaled[1] / scaled[0] < 4.0, f"criterion is not distance-stable: {scaled}"
