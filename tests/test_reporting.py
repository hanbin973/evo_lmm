import numpy as np

from evo_lmm import (
    MultiComponentOps,
    MultiComponentPrior,
    SimplifiedPrior,
    boundary_lrt_pvalue,
    fit_genes,
    fit_multicomponent_reml,
    fit_parameter_profiles,
    fit_report,
    fit_tau_profiles,
    genic_variance_by_maf,
    heritability_conventions,
)


def _fixture():
    rng = np.random.default_rng(123)
    genotypes = {
        "lof": rng.binomial(2, 0.1, (20, 4)).astype(float),
        "missense": rng.binomial(2, 0.3, (20, 5)).astype(float),
    }
    frequencies = {
        label: matrix.mean(axis=0) / 2 for label, matrix in genotypes.items()
    }
    ops = MultiComponentOps.from_dense(genotypes, frequencies)
    prior = MultiComponentPrior(
        ops.labels, (SimplifiedPrior(1.0, 0.3), SimplifiedPrior(0.7, 0.8))
    )
    return ops, prior


def test_reporting_adapters_return_both_conventions_and_maf_bins():
    ops, prior = _fixture()
    estimates = heritability_conventions(ops, prior, 0.8)
    assert 0 <= estimates.rare_effect <= 1
    assert 0 <= estimates.evolutionary <= 1
    bins = genic_variance_by_maf(ops, prior, [0.0, 0.1, 0.5])
    assert set(bins) == set(ops.labels)
    assert all(values.shape == (2,) and np.all(values >= 0) for values in bins.values())


def test_boundary_lrt_is_a_mixture_not_naive_chi_square():
    assert boundary_lrt_pvalue(0.0) == 1.0
    assert 0.0 < boundary_lrt_pvalue(3.0, added_boundaries=2) < 1.0


def test_fit_report_includes_uncertainty_and_maf_decomposition():
    ops, _ = _fixture()
    y = np.random.default_rng(9).normal(size=ops.n)
    fit = fit_multicomponent_reml(ops, y, max_iter=2, trace_probes=4)
    report = fit_report(fit, maf_bins=[0.0, 0.1, 0.5])
    assert report.maf_decomposition is not None
    assert np.isfinite(report.heritability_se)
    assert set(report.component_standard_errors) == {
        "sigma_b2[lof]",
        "tau[lof]",
        "sigma_b2[missense]",
        "tau[missense]",
    }
    assert all(
        np.isfinite(value) for value in report.component_standard_errors.values()
    )


def test_parameter_profiles_use_the_reported_scientific_scale():
    from evo_lmm.multicomponent import profiled_reml_objective
    from evo_lmm.reporting import _ratio_prior

    ops, _ = _fixture()
    y = 4.0 * np.random.default_rng(31).normal(size=ops.n)
    fit = fit_multicomponent_reml(ops, y, max_iter=6, trace_probes=8)
    assert abs(fit.sigma_e2 - 1.0) > 0.5

    fitted_sigma_b2 = float(fit.prior.components[0].sigma_b2)
    at_fit = profiled_reml_objective(
        ops, y, _ratio_prior(fit, fit.prior.components)
    )[0]
    profile = fit_parameter_profiles(fit, {"sigma_b2[lof]": [fitted_sigma_b2]})
    np.testing.assert_allclose(profile["sigma_b2[lof]"].objective, [at_fit], rtol=1e-10)

    tau_profile = fit_tau_profiles(
        fit, {"lof": [float(fit.prior.components[0].tau)]}
    )
    np.testing.assert_allclose(tau_profile["lof"].objective, [at_fit], rtol=1e-10)


def test_gene_fitting_conditions_on_pooled_shapes():
    ops, _ = _fixture()
    y = 2.0 * np.random.default_rng(41).normal(size=ops.n)
    pooled = {"lof": 0.35, "missense": 0.9}
    reports = fit_genes({"gene1": ops}, y, pooled, max_iter=15, trace_probes=8)
    assert reports["gene1"].gene == "gene1"
    assert reports["gene1"].pooled_tau == pooled
    assert set(reports["gene1"].sigma_b2_by_category) == set(ops.labels)

    initial = MultiComponentPrior(
        ops.labels, tuple(SimplifiedPrior(1.0, pooled[label]) for label in ops.labels)
    )
    conditioned = fit_multicomponent_reml(
        ops, y, initial=initial, max_iter=15, trace_probes=8, fit_tau=False
    )
    for index, label in enumerate(ops.labels):
        np.testing.assert_allclose(
            reports["gene1"].sigma_b2_by_category[label],
            conditioned.prior.components[index].sigma_b2,
            rtol=1e-12,
        )
    free = fit_multicomponent_reml(
        ops, y, initial=initial, max_iter=15, trace_probes=8
    )
    assert any(
        abs(free.prior.tau[index] - pooled[label]) > 1e-8
        for index, label in enumerate(ops.labels)
    )
    assert any(
        abs(
            free.prior.components[index].sigma_b2
            - conditioned.prior.components[index].sigma_b2
        )
        > 1e-8
        for index in range(len(ops.labels))
    )
