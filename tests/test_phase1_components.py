import numpy as np

from evo_lmm import (
    MultiComponentOps,
    MultiComponentPrior,
    SimplifiedPrior,
    boundary_lrt_pvalue,
    collapse_mac,
    genic_variance_by_maf,
    heritability_conventions,
    joint_mom_initialization,
    fit_multicomponent_reml,
    fit_genes,
    fit_report,
    fit_parameter_profiles,
    fit_tau_profiles,
    fit_rare_effect_baseline,
    rare_effect_mom_ratio,
)


def _fixture():
    rng = np.random.default_rng(123)
    g = {"lof": rng.binomial(2, 0.1, (20, 4)).astype(float),
         "missense": rng.binomial(2, 0.3, (20, 5)).astype(float)}
    f = {key: value.mean(axis=0) / 2 for key, value in g.items()}
    ops = MultiComponentOps.from_dense(g, f)
    prior = MultiComponentPrior(ops.labels, (SimplifiedPrior(1.0, 0.3), SimplifiedPrior(0.7, 0.8)))
    return g, f, ops, prior


def test_mac_collapse_recomputes_frequency_and_keeps_provenance():
    g = {"c": np.array([[1, 1, 0], [1, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=float)}
    result = collapse_mac(g, mac_threshold=2)
    assert result.genotypes["c"].shape == (4, 2)
    assert result.source_indices["c"] == ((0,), (1, 2))
    np.testing.assert_allclose(result.frequencies["c"], [0.25, 0.25])


def test_joint_mom_reports_raw_and_truncated_estimates():
    _, _, ops, prior = _fixture()
    y = np.random.default_rng(2).normal(size=ops.n)
    result = joint_mom_initialization(ops, y, prior)
    assert result.system.shape == (3, 3)
    assert result.raw_component_scales.shape == (2,)
    projected = ops.project(y)
    kernels = ops.component_kernels(prior)
    traces = np.asarray([np.trace(kernel) for kernel in kernels.values()])
    expected_system = np.block([
        [np.asarray([[np.trace(left @ right) for right in kernels.values()]
                     for left in kernels.values()]), traces[:, None]],
        [traces[None, :], np.asarray([[ops.dim]])],
    ])
    expected_rhs = np.asarray(
        [projected @ kernel @ projected for kernel in kernels.values()]
        + [projected @ projected]
    )
    expected_raw = np.linalg.solve(expected_system, expected_rhs)
    np.testing.assert_allclose(result.system, expected_system, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.raw_component_scales, expected_raw[:-1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.residual_variance, max(expected_raw[-1], np.finfo(float).tiny))
    np.testing.assert_array_equal(result.component_scales, np.maximum(expected_raw[:-1], 0.0))
    xtrace_result = joint_mom_initialization(ops, np.random.default_rng(2).normal(size=ops.n), prior,
                                             trace_method="xtrace", trace_probes=4, seed=4)
    assert xtrace_result.trace_standard_errors is not None


def test_production_ai_reml_supports_hutchinson_and_xtrace():
    _, _, ops, _ = _fixture()
    y = np.random.default_rng(9).normal(size=ops.n)
    for trace_method in ("hutchinson", "xtrace"):
        fit = fit_multicomponent_reml(ops, y, max_iter=2, trace_method=trace_method, trace_probes=4)
        assert np.isfinite(fit.objective)
        assert fit.ai_covariance is not None
        assert fit.standard_errors is not None
        assert fit.prior.labels == ops.labels
        assert all(np.isfinite(value) for value in fit.standard_errors.values())
        report = fit_report(fit, maf_bins=[0.0, 0.1, 0.5])
        assert report.maf_decomposition is not None
        assert np.isfinite(report.heritability_se)
        assert set(report.component_standard_errors) == {
            "sigma_b2[lof]", "tau[lof]", "sigma_b2[missense]", "tau[missense]"
        }
        assert all(np.isfinite(value) for value in report.component_standard_errors.values())
        profiles = fit_parameter_profiles(
            fit,
            {"sigma_b2[lof]": [0.5, 1.0, 2.0], "tau[missense]": [0.0, 0.3, 0.8]},
        )
        assert set(profiles) == {"sigma_b2[lof]", "tau[missense]"}
        for profile in profiles.values():
            assert np.all(np.isfinite(profile.objective))
            assert np.ptp(profile.objective) > 1e-10


def test_gene_reporting_pools_tau_and_fits_category_scales():
    _, _, ops, _ = _fixture()
    prior = MultiComponentPrior(
        ops.labels, (SimplifiedPrior(1.0, 0.3), SimplifiedPrior(0.7, 0.8))
    )
    reports = fit_genes({"gene1": ops}, np.random.default_rng(12).normal(size=ops.n),
                        {label: component.tau for label, component in zip(prior.labels, prior.components)},
                        max_iter=1, trace_probes=4)
    assert reports["gene1"].gene == "gene1"
    assert set(reports["gene1"].pooled_tau) == set(ops.labels)
    assert reports["gene1"].pooled_tau == {
        label: component.tau for label, component in zip(prior.labels, prior.components)
    }
    assert set(reports["gene1"].sigma_b2_by_category) == set(ops.labels)
    assert all(value > 0.0 for value in reports["gene1"].sigma_b2_by_category.values())


def test_rare_effect_ratio_matches_independent_dense_formula_and_fallback():
    marginal = np.array([2.0, 3.0])
    marginal_mom = np.array([1.0, -1.0])
    joint_mom = np.array([1.5, 2.0])
    result = rare_effect_mom_ratio(marginal, marginal_mom, joint_mom)
    np.testing.assert_allclose(result.adjusted_scales, [3.0, 3.0])
    np.testing.assert_array_equal(result.negative_mom_fallback, [False, True])


def _error_contrast_rare_effect(genotypes, basis, y):
    """RareEffect's baseline, reimplemented through the error contrasts.

    Deliberately shares nothing with ``evo_lmm.baselines``: the restricted
    likelihood is built from the eigenvalues of ``Q' X X' Q``, where ``Q``
    spans the orthogonal complement of the covariate basis, rather than from a
    projected full-rank inverse; the moment systems are solved as explicit
    normal equations on the reduced kernels; and the ratio rule with its
    non-positive-MoM fallback is applied inline.  Only the raw genotype
    matrices and the covariate basis are taken from the fixture.
    """
    from scipy.linalg import null_space
    from scipy.optimize import minimize_scalar

    contrasts = null_space(basis.T)
    reduced_y = contrasts.T @ y
    dimension = reduced_y.size
    reduced = {label: (contrasts.T @ matrix) @ (contrasts.T @ matrix).T
               for label, matrix in genotypes.items()}
    labels = list(genotypes)
    marginal, marginal_mom = [], []
    for label in labels:
        kernel = reduced[label]
        eigenvalues, vectors = np.linalg.eigh(kernel)
        rotated = vectors.T @ reduced_y

        def objective(log_delta, eigenvalues=eigenvalues, rotated=rotated):
            delta = np.exp(log_delta)
            quadratic = float(np.sum(rotated ** 2 / (eigenvalues + delta)))
            return 0.5 * (float(np.sum(np.log(eigenvalues + delta)))
                          + dimension * np.log(quadratic / dimension))

        best = minimize_scalar(objective, bounds=(-30.0, 30.0), method="bounded",
                               options={"xatol": 1e-12})
        delta = float(np.exp(best.x))
        marginal.append(float(np.sum(rotated ** 2 / (eigenvalues + delta))) / dimension)
        system = np.array([[float(np.sum(eigenvalues ** 2)), float(np.sum(eigenvalues))],
                           [float(np.sum(eigenvalues)), float(dimension)]])
        rhs = np.array([float(reduced_y @ kernel @ reduced_y), float(reduced_y @ reduced_y)])
        marginal_mom.append(float(np.linalg.solve(system, rhs)[0]))
    count = len(labels)
    system = np.empty((count + 1, count + 1))
    rhs = np.empty(count + 1)
    for i, left in enumerate(labels):
        for j, right in enumerate(labels):
            system[i, j] = float(np.trace(reduced[left] @ reduced[right]))
        system[i, count] = system[count, i] = float(np.trace(reduced[left]))
        rhs[i] = float(reduced_y @ reduced[left] @ reduced_y)
    system[count, count] = float(dimension)
    rhs[count] = float(reduced_y @ reduced_y)
    joint = np.linalg.solve(system, rhs)[:count]
    marginal = np.asarray(marginal)
    marginal_mom = np.asarray(marginal_mom)
    fallback = (marginal_mom <= 0.0) | (joint <= 0.0)
    adjusted = marginal.copy()
    adjusted[~fallback] = (marginal * joint / marginal_mom)[~fallback]
    return marginal, marginal_mom, joint, adjusted, fallback


def test_rare_effect_baseline_matches_an_error_contrast_reimplementation():
    """Clause 4: an independent implementation, plus frozen values.

    The previous version of this test rebuilt the production algorithm from the
    production helpers and computed its expectation by calling
    ``rare_effect_mom_ratio`` -- the function under test -- so it could not
    fail.  Here the comparison is against a different formulation (error
    contrasts and eigenvalues) and against literals recorded from this fixture,
    so drift in either implementation is caught.
    """
    genotypes, _, ops, _ = _fixture()
    rng = np.random.default_rng(19)
    y = (genotypes["lof"] @ rng.normal(0.0, 0.8, genotypes["lof"].shape[1])
         + genotypes["missense"] @ rng.normal(0.0, 0.5, genotypes["missense"].shape[1])
         + rng.normal(0.0, 1.5, ops.n))

    result = fit_rare_effect_baseline(ops, y)
    marginal, marginal_mom, joint, adjusted, fallback = _error_contrast_rare_effect(
        genotypes, ops.basis, y
    )
    assert not np.any(fallback), "fixture no longer keeps both categories interior"
    np.testing.assert_allclose(result.marginal_scales, marginal, rtol=1e-6)
    np.testing.assert_allclose(result.marginal_mom_scales, marginal_mom, rtol=1e-9)
    np.testing.assert_allclose(result.joint_mom_scales, joint, rtol=1e-9)
    np.testing.assert_allclose(result.adjusted_scales, adjusted, rtol=1e-6)
    np.testing.assert_array_equal(result.negative_mom_fallback, fallback)

    # Recorded from this fixture; both implementations must keep reproducing them.
    np.testing.assert_allclose(result.marginal_scales, [0.5312628449, 0.2067134205], rtol=1e-6)
    np.testing.assert_allclose(result.marginal_mom_scales, [0.610498666, 0.09999417018], rtol=1e-8)
    np.testing.assert_allclose(result.joint_mom_scales, [0.5970611186, 0.09172101803], rtol=1e-8)
    np.testing.assert_allclose(result.adjusted_scales, [0.5195693392, 0.1896107076], rtol=1e-6)


def test_rare_effect_baseline_falls_back_on_a_non_positive_moment_estimate():
    """The published fallback must fire on real data, not only on hand inputs."""
    genotypes, _, ops, _ = _fixture()
    y = 3.0 * np.random.default_rng(19).normal(size=ops.n)
    result = fit_rare_effect_baseline(ops, y)
    _marginal, marginal_mom, joint, _adjusted, fallback = _error_contrast_rare_effect(
        genotypes, ops.basis, y
    )
    assert fallback[0], "fixture no longer triggers the non-positive-MoM rule"
    np.testing.assert_array_equal(result.negative_mom_fallback, fallback)
    np.testing.assert_allclose(result.marginal_mom_scales, marginal_mom, rtol=1e-9)
    np.testing.assert_allclose(result.joint_mom_scales, joint, rtol=1e-9)
    # A fallback category keeps the unadjusted marginal estimate.
    np.testing.assert_array_equal(result.adjusted_scales[0], result.marginal_scales[0])


def test_reporting_adapters_return_both_conventions_and_maf_bins():
    _, _, ops, prior = _fixture()
    estimates = heritability_conventions(ops, prior, 0.8)
    assert 0 <= estimates.rare_effect <= 1
    assert 0 <= estimates.evolutionary <= 1
    bins = genic_variance_by_maf(ops, prior, [0.0, 0.1, 0.5])
    assert set(bins) == set(ops.labels)
    assert all(values.shape == (2,) and np.all(values >= 0) for values in bins.values())


def test_boundary_lrt_is_a_mixture_not_naive_chi_square():
    assert boundary_lrt_pvalue(0.0) == 1.0
    assert 0.0 < boundary_lrt_pvalue(3.0, added_boundaries=2) < 1.0


def test_parameter_profiles_are_evaluated_on_the_reported_scientific_scale():
    """Profiles must use the same scale the fit reports.

    Regression: the fit's scientific-scale ``sigma_b2_c`` was passed straight
    into ``profiled_reml_objective``, which profiles the residual scale and
    therefore takes ratios.  Every profile -- and every grid value -- was off
    by a factor of ``sigma_e2``.  The phenotype here is scaled so that factor
    is far from one.
    """
    from evo_lmm.multicomponent import profiled_reml_objective
    from evo_lmm.reporting import _ratio_prior

    _, _, ops, _ = _fixture()
    y = 4.0 * np.random.default_rng(31).normal(size=ops.n)
    fit = fit_multicomponent_reml(ops, y, max_iter=6, trace_probes=8)
    assert abs(fit.sigma_e2 - 1.0) > 0.5, "fixture no longer exercises the profiled scale"

    fitted_sigma_b2 = float(fit.prior.components[0].sigma_b2)
    at_fit = profiled_reml_objective(ops, y, _ratio_prior(fit, fit.prior.components))[0]
    profile = fit_parameter_profiles(fit, {"sigma_b2[lof]": [fitted_sigma_b2]})
    np.testing.assert_allclose(profile["sigma_b2[lof]"].objective, [at_fit], rtol=1e-10)

    tau_profile = fit_tau_profiles(fit, {"lof": [float(fit.prior.components[0].tau)]})
    np.testing.assert_allclose(tau_profile["lof"].objective, [at_fit], rtol=1e-10)


def test_gene_fitting_conditions_on_pooled_shapes_without_re_estimating_them():
    """A per-gene fit must not move the pooled shapes.

    Regression: ``pooled_tau`` was only an initialization, the fit re-estimated
    every ``tau_c`` freely, and the report echoed the input ``pooled_tau`` back
    as though it had been held.
    """
    _, _, ops, _ = _fixture()
    y = 2.0 * np.random.default_rng(41).normal(size=ops.n)
    pooled = {"lof": 0.35, "missense": 0.9}
    reports = fit_genes({"gene1": ops}, y, pooled, max_iter=15, trace_probes=8)
    assert reports["gene1"].pooled_tau == pooled
    initial = MultiComponentPrior(
        ops.labels, tuple(SimplifiedPrior(1.0, pooled[label]) for label in ops.labels)
    )
    conditioned = fit_multicomponent_reml(
        ops, y, initial=initial, max_iter=15, trace_probes=8, fit_tau=False
    )
    for index, label in enumerate(ops.labels):
        np.testing.assert_allclose(
            reports["gene1"].sigma_b2_by_category[label],
            conditioned.prior.components[index].sigma_b2, rtol=1e-12,
        )
    free = fit_multicomponent_reml(ops, y, initial=initial, max_iter=15, trace_probes=8)
    assert any(abs(free.prior.tau[index] - pooled[label]) > 1e-8
               for index, label in enumerate(ops.labels)), "fixture no longer moves tau when free"
    assert any(abs(free.prior.components[index].sigma_b2
                   - conditioned.prior.components[index].sigma_b2) > 1e-8
               for index in range(len(ops.labels))), "pooled and free fits are indistinguishable here"
