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


def test_max_iter_zero_returns_seeded_diagnostics_instead_of_raising():
    """A loop that never runs must still report the initial point.

    Regression: the reported state was seeded with a placeholder that reached
    ``SimplifiedPrior`` and raised "sigma_b2 must be finite and strictly
    positive" from the result-assembly step instead of returning diagnostics.
    """
    ops, _, _ = _ops()
    y = np.random.default_rng(5).normal(size=ops.n)
    initial = MultiComponentPrior(
        ops.labels, (SimplifiedPrior(0.9, 0.4), SimplifiedPrior(0.6, 1.1))
    )
    fit = fit_multicomponent_reml(ops, y, initial=initial, max_iter=0)
    assert not fit.converged
    assert np.isfinite(fit.sigma_e2) and fit.sigma_e2 > 0.0
    assert np.all(fit.prior.sigma_b2 > 0.0)
    np.testing.assert_allclose(fit.prior.tau, initial.tau)
    np.testing.assert_allclose(
        fit.prior.sigma_b2, fit.sigma_e2 * initial.sigma_b2, rtol=1e-12
    )


def _rejection_fixture():
    n = 120
    rng = np.random.default_rng(5)
    genotypes, frequencies, contribution = {}, {}, np.zeros(n)
    for label, count, scale, tau in (("lof", 15, 0.05, 60.0), ("missense", 20, 0.02, 10.0)):
        p = rng.beta(0.4, 0.4, size=count) * 0.49 + 0.01
        matrix = rng.binomial(2, p, size=(n, count)).astype(float)
        freq = matrix.mean(axis=0) / 2.0
        q = freq * (1.0 - freq)
        contribution += matrix @ rng.normal(0.0, np.sqrt(scale / (1.0 + 2.0 * tau * q)))
        genotypes[label] = matrix
        frequencies[label] = freq
    y = contribution + rng.normal(0.0, 1.0, size=n)
    ops = MultiComponentOps.from_dense(genotypes, frequencies)
    truth = MultiComponentPrior(
        ops.labels, (SimplifiedPrior(0.05, 60.0), SimplifiedPrior(0.02, 10.0))
    )
    return ops, y, truth


def test_first_iteration_line_search_rejection_reports_that_iterate():
    """A rejected first step must report the current iterate, not raise.

    Regression: the ``not accepted`` branch left the loop before the state
    assignments, so a first-iteration rejection hit the same result-assembly
    validator error as ``max_iter=0``.  This is the reachable trigger: starting
    the fit at the generating parameters is enough.
    """
    ops, y, truth = _rejection_fixture()
    fit = fit_multicomponent_reml(ops, y, initial=truth, max_iter=1, trace_probes=4, seed=1)
    assert fit.accepted_step == 0.0, "fixture no longer exercises the rejection path"
    assert not fit.converged
    assert np.isfinite(fit.sigma_e2) and fit.sigma_e2 > 0.0
    assert np.isfinite(fit.score_norm)
    np.testing.assert_allclose(fit.prior.tau, truth.tau)
