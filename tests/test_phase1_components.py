import numpy as np
from scipy.optimize import minimize_scalar

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
    result = joint_mom_initialization(ops, np.random.default_rng(2).normal(size=ops.n), prior)
    assert result.system.shape == (3, 3)
    assert result.raw_component_scales.shape == (2,)
    np.testing.assert_array_equal(result.truncated, result.raw_component_scales < 0)
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
        report = fit_report(fit, maf_bins=[0.0, 0.1, 0.5])
        assert report.maf_decomposition is not None
        assert np.isfinite(report.heritability_se)
        profiles = fit_parameter_profiles(
            fit,
            {"sigma_b2[lof]": [0.5, 1.0, 2.0], "tau[missense]": [0.0, 0.3, 0.8]},
        )
        assert set(profiles) == {"sigma_b2[lof]", "tau[missense]"}
        assert all(np.isfinite(profile.objective).any() for profile in profiles.values())


def test_multicomponent_convergence_uses_single_component_score_step_rule():
    _, _, ops, _ = _fixture()
    y = np.random.default_rng(10).normal(size=ops.n)
    fit = fit_multicomponent_reml(ops, y, max_iter=2, tol=1e9, max_step=2.0)
    assert fit.converged
    assert fit.score_norm <= 1e9
    assert fit.accepted_step == 0.0
    assert fit.ai_damping == 0.0


def test_gene_reporting_pools_tau_and_fits_category_scales():
    _, _, ops, prior = _fixture()
    reports = fit_genes({"gene1": ops}, np.random.default_rng(12).normal(size=ops.n),
                        {label: component.tau for label, component in zip(prior.labels, prior.components)},
                        max_iter=1, trace_probes=4)
    assert reports["gene1"].gene == "gene1"
    assert set(reports["gene1"].pooled_tau) == set(ops.labels)


def test_rare_effect_ratio_matches_independent_dense_formula_and_fallback():
    marginal = np.array([2.0, 3.0])
    marginal_mom = np.array([1.0, -1.0])
    joint_mom = np.array([1.5, 2.0])
    result = rare_effect_mom_ratio(marginal, marginal_mom, joint_mom)
    np.testing.assert_allclose(result.adjusted_scales, [3.0, 3.0])
    np.testing.assert_array_equal(result.negative_mom_fallback, [False, True])


def test_rare_effect_baseline_matches_independent_dense_reimplementation():
    _, _, ops, _ = _fixture()
    y = np.random.default_rng(19).normal(size=ops.n)
    result = fit_rare_effect_baseline(ops, y)
    marginal = []
    marginal_mom = []
    projected = ops.project(y)
    for label in ops.labels:
        component = ops.components[label]
        kernel = component.dense_kernel(SimplifiedPrior(1.0, 0.0))

        def objective(log_delta):
            from evo_lmm.reml import _dense_projection
            shape = kernel + np.exp(log_delta) * np.eye(component.n)
            ph, inv_shape, logdet = _dense_projection(shape, component.basis)
            q = y @ ph @ y
            fixed = component.basis.T @ inv_shape @ component.basis
            return 0.5 * (logdet + np.linalg.slogdet(fixed)[1]
                          + component.dim * np.log(q / component.dim))

        optimum = minimize_scalar(objective, bounds=(-30.0, 30.0), method="bounded",
                                  options={"xatol": 1e-10})
        from evo_lmm.reml import _dense_projection
        ph, _, _ = _dense_projection(kernel + np.exp(optimum.x) * np.eye(component.n), component.basis)
        marginal.append(y @ ph @ y / component.dim)
        trace = np.trace(kernel)
        system = np.array([[np.trace(kernel @ kernel), trace], [trace, component.dim]])
        rhs = np.array([projected @ kernel @ projected, projected @ projected])
        marginal_mom.append(np.linalg.lstsq(system, rhs, rcond=None)[0][0])
    flat = MultiComponentPrior.flat(ops.labels)
    kernels = ops.component_kernels(flat)
    traces = np.array([np.trace(kernel) for kernel in kernels.values()])
    cross = np.array([[np.trace(left @ right) for right in kernels.values()]
                      for left in kernels.values()])
    system = np.block([[cross, traces[:, None]], [traces[None, :], np.array([[ops.dim]])]])
    rhs = np.r_[[projected @ kernel @ projected for kernel in kernels.values()], projected @ projected]
    joint = np.linalg.lstsq(system, rhs, rcond=None)[0][:-1]
    expected = rare_effect_mom_ratio(np.array(marginal), np.array(marginal_mom), joint)
    np.testing.assert_allclose(result.marginal_scales, expected.marginal_scales, rtol=1e-8)
    np.testing.assert_allclose(result.marginal_mom_scales, expected.marginal_mom_scales, rtol=1e-8)
    np.testing.assert_allclose(result.joint_mom_scales, expected.joint_mom_scales, rtol=1e-8)
    np.testing.assert_allclose(result.adjusted_scales, expected.adjusted_scales, rtol=1e-8)


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
