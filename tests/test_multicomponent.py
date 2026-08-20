import numpy as np

from evo_lmm import (
    MultiComponentOps,
    MultiComponentPrior,
    SimplifiedPrior,
    fit_multicomponent_reml,
    profiled_reml_objective,
)
from evo_lmm.reml import fit_reml


def _ops():
    rng = np.random.default_rng(41)
    genotypes = {
        "lof": rng.binomial(2, 0.08, size=(28, 5)).astype(float),
        "missense": rng.binomial(2, 0.25, size=(28, 7)).astype(float),
    }
    frequencies = {label: matrix.mean(axis=0) / 2 for label, matrix in genotypes.items()}
    return MultiComponentOps.from_dense(genotypes, frequencies), genotypes, frequencies


def test_partitioned_components_are_symmetric_psd_and_derivatives_are_analytic():
    ops, _, _ = _ops()
    prior = MultiComponentPrior(
        ops.labels, (SimplifiedPrior(1.2, 0.7), SimplifiedPrior(0.8, 1.4))
    )
    kernel = ops.dense_kernel(prior)
    assert np.allclose(kernel, kernel.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(kernel)) >= -1e-10
    for component_kernel in ops.component_kernels(prior).values():
        assert np.allclose(component_kernel, component_kernel.T, atol=1e-12)
        assert np.min(np.linalg.eigvalsh(component_kernel)) >= -1e-10
    derivatives = ops.derivative_kernels(prior)
    eps = 1e-6
    shifted = MultiComponentPrior(
        ops.labels, (SimplifiedPrior(1.2 * np.exp(eps), 0.7), SimplifiedPrior(0.8, 1.4))
    )
    numeric = (ops.dense_kernel(shifted) - kernel) / eps
    np.testing.assert_allclose(numeric, derivatives["log_sigma_b2[lof]"], rtol=2e-5, atol=1e-8)


def test_zero_tau_is_exact_flat_per_category_nesting():
    ops, genotypes, frequencies = _ops()
    flat = MultiComponentPrior(
        ops.labels, (SimplifiedPrior(1.2, 0.0), SimplifiedPrior(0.8, 0.0))
    )
    expected = sum(
        scale * ops.components[label].dense_kernel(SimplifiedPrior(1.0, 0.0))
        for label, scale in zip(ops.labels, (1.2, 0.8))
    )
    np.testing.assert_array_equal(ops.dense_kernel(flat), expected)


def test_small_dense_fit_ladder_has_exact_boundary_objectives():
    ops, _, _ = _ops()
    y = np.random.default_rng(7).normal(size=ops.n)
    flat = MultiComponentPrior.flat(ops.labels, scales=(1.2, 0.8))
    flat_reference = MultiComponentPrior(
        ops.labels, (SimplifiedPrior(1.2, 0.0), SimplifiedPrior(0.8, 0.0))
    )
    shared = MultiComponentPrior(
        ops.labels, (SimplifiedPrior(1.2, 0.7), SimplifiedPrior(0.8, 0.7))
    )
    shared_reference = MultiComponentPrior(
        ops.labels, (SimplifiedPrior(1.2, 0.7), SimplifiedPrior(0.8, 0.7))
    )
    np.testing.assert_array_equal(
        profiled_reml_objective(ops, y, flat),
        profiled_reml_objective(ops, y, flat_reference),
    )
    np.testing.assert_array_equal(
        profiled_reml_objective(ops, y, shared),
        profiled_reml_objective(ops, y, shared_reference),
    )


def test_shared_tau_and_batched_component_derivatives_are_exact():
    ops, _, _ = _ops()
    prior = MultiComponentPrior(
        ops.labels, (SimplifiedPrior(1.2, 0.7), SimplifiedPrior(0.8, 0.7))
    )
    shared = MultiComponentPrior(
        ops.labels, (SimplifiedPrior(1.2, 0.7), SimplifiedPrior(0.8, 0.7))
    ).with_shared_tau(0.7)
    np.testing.assert_array_equal(ops.dense_kernel(prior), ops.dense_kernel(shared))
    values = np.random.default_rng(11).normal(size=(ops.n, 3))
    batched = ops.apply_component_derivatives_matmat(values, prior)
    for name, derivative in batched.items():
        label = name.split("[", 1)[1][:-1]
        index = ops.labels.index(label)
        component = prior.components[index]
        expected = component.sigma_b2 * ops.components[label].dense_kernel(component) @ values
        if name.startswith("log_tau"):
            expected = ops.derivative_kernels(prior)[name] @ values
        np.testing.assert_allclose(derivative, expected, atol=1e-10)


def test_single_category_fit_delegates_to_existing_fitter():
    ops, genotypes, frequencies = _ops()
    single_ops = MultiComponentOps.from_dense({"lof": genotypes["lof"]}, {"lof": frequencies["lof"]})
    y = np.random.default_rng(33).normal(size=single_ops.n)
    multi = fit_multicomponent_reml(single_ops, y, initial=MultiComponentPrior.flat(("lof",)), max_iter=8)
    reference = fit_reml(single_ops.components["lof"], y, initial=SimplifiedPrior(1.0, 0.0), exact=True, max_iter=8)
    np.testing.assert_allclose(multi.sigma_e2, reference.sigma_e2, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(multi.h2, reference.h2, rtol=1e-10, atol=1e-12)


def test_multicomponent_reml_returns_scientific_component_scales():
    ops, _, _ = _ops()
    rng = np.random.default_rng(42)
    y = rng.normal(size=ops.n)
    fit = fit_multicomponent_reml(ops, y, max_iter=30)
    assert fit.converged or np.isfinite(fit.objective)
    assert fit.prior.labels == ops.labels
    assert np.all(fit.prior.sigma_b2 > 0)
    assert np.all(fit.prior.tau >= 0)
    assert 0 <= fit.h2 <= 1
