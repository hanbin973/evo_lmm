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
