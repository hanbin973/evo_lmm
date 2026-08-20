"""Large seeded parameter-estimation tests.

These tests deliberately use more individuals than the algebraic unit tests.
The exact dense fit is the reference implementation for the small GRG oracle
comparison; the matrix-free fit is the path used for larger data.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest


# Shape optimizers can inspect singular trial matrices before rejecting them;
# those expected exploratory warnings are not failures of the fitted result.
pytestmark = pytest.mark.filterwarnings("ignore:.*slogdet.*:RuntimeWarning")

from evo_lmm import (
    EvolutionaryLmmOps,
    FullPrior,
    SimplifiedPrior,
    fit_reml,
    restricted_log_likelihood,
    simulate_grg_lmm,
)


def _seeded_dense_simulation(prior, *, seed: int, n: int = 400, m: int = 300, residual_variance: float = 0.1):
    """Create a variable-frequency raw-dosage data set with a known prior."""

    rng = np.random.default_rng(seed)
    # A broad spectrum makes the frequency-dependent shape identifiable.  The
    # low-frequency end is still observed often enough with 400 individuals.
    population_frequencies = np.geomspace(0.005, 0.5, m)
    dosage = rng.binomial(2, population_frequencies, size=(n, m)).astype(np.float64)
    frequencies = dosage.mean(axis=0) / 2.0
    ops = EvolutionaryLmmOps.from_dense(dosage, frequencies, model=prior.model_name)
    effects = rng.normal(0.0, np.sqrt(prior.effect_variances(frequencies)))
    phenotype = ops.apply_model_x(effects) + rng.normal(
        0.0,
        np.sqrt(residual_variance),
        size=n,
    )
    return dosage, frequencies, phenotype, ops


@pytest.mark.large
def test_large_simplified_matrix_free_fit_recovers_variance_components():
    """The two-parameter model recovers scale and selection shape at N=400."""

    true_prior = SimplifiedPrior(sigma_b2=0.01, tau=20.0)
    _dosage, _frequencies, phenotype, ops = _seeded_dense_simulation(
        true_prior,
        seed=13,
    )

    fit = fit_reml(
        ops,
        phenotype,
        initial=true_prior,
        exact=False,
        trace_probes=128,
        seed=8,
        max_iter=30,
        cg_tol=1e-8,
    )

    assert ops.n == 400
    assert fit.diagnostics.converged
    assert fit.diagnostics.trace_estimator == "hutchinson"
    assert fit.diagnostics.trace_probes == 128
    assert np.isclose(fit.prior.sigma_b2, true_prior.sigma_b2, rtol=0.9)
    assert np.isclose(fit.prior.tau, true_prior.tau, rtol=0.75)
    assert np.isclose(fit.sigma_e2, 0.1, rtol=0.25)
    assert fit.prior.tau > 0.0
    assert fit.sigma_b2 > 0.0
    assert fit.sigma_e2 > 0.0

    # A fitted evolutionary kernel must remain a valid covariance operator.
    kernel = ops.dense_kernel(fit.prior)
    minimum_eigenvalue = float(np.linalg.eigvalsh(kernel)[0])
    assert minimum_eigenvalue >= -1e-9 * max(float(np.trace(kernel)), 1.0)


@pytest.mark.large
def test_large_full_exact_fit_recovers_parameters_and_improves_reml():
    """The full model estimates a well-conditioned parameter set at N=400.

    ``rho`` and ``tau`` are not expected to be exact in a single-trait draw:
    their likelihood surface can be shallow.  The test consequently checks
    broad Monte Carlo recovery ranges, the fitted covariance scale, and the
    exact REML optimum relative to the generating parameters.
    """

    true_prior = FullPrior(sigma_b2=0.01, tau=20.0, rho=0.7)
    _dosage, frequencies, phenotype, ops = _seeded_dense_simulation(
        true_prior,
        seed=12,
    )

    # The optimizer evaluates deliberately invalid trial covariances while it
    # brackets the constrained shape parameters.  Those trials are not test
    # failures, but NumPy may report their failed log-determinants as warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fit = fit_reml(
            ops,
            phenotype,
            initial=true_prior,
            exact=True,
            seed=8,
            max_iter=45,
        )

    assert ops.n == 400
    # ``status`` is asserted rather than ``converged``: the latter is also set
    # by the finishing optimizer's loose ``||score||_inf < 1e-4`` back-stop.
    assert fit.diagnostics.status == "converged"
    assert fit.diagnostics.trace_estimator == "exact"
    # Stationarity is checked on the scale-free statistics.  ``score_norm`` is
    # not a criterion: it grows with the sample size, so no absolute bound on
    # it means the same thing at two different ``n``.
    assert fit.diagnostics.step_se_norm <= 1e-2
    assert fit.diagnostics.newton_decrement <= 1e-2
    assert np.isclose(fit.prior.sigma_b2, true_prior.sigma_b2, rtol=0.5)
    assert np.isclose(fit.prior.tau, true_prior.tau, rtol=0.75)
    assert np.isclose(fit.prior.rho, true_prior.rho, rtol=0.5, atol=0.25)
    assert np.isclose(fit.sigma_e2, 0.1, rtol=0.6)
    assert 0.0 <= fit.prior.rho2 <= 1.0

    fitted_covariance = fit.sigma_b2 * ops.dense_kernel(fit.prior) + fit.sigma_e2 * np.eye(ops.n)
    true_covariance = true_prior.sigma_b2 * ops.dense_kernel(true_prior) + 0.1 * np.eye(ops.n)
    fitted_reml = restricted_log_likelihood(phenotype, fitted_covariance, ops.basis)
    true_reml = restricted_log_likelihood(phenotype, true_covariance, ops.basis)
    assert fitted_reml >= true_reml - 1e-6

    # The scale-weighted effect-variance curve is more identifiable than the
    # individual tau/rho coordinates.  It should be close over the spectrum.
    log_variance_error = np.mean(
        np.abs(np.log(fit.prior.effect_variances(frequencies) / true_prior.effect_variances(frequencies)))
    )
    assert log_variance_error < 0.3


@pytest.mark.large
def test_large_full_boundary_fit_matches_simplified_parameter_estimates():
    """The full fitter remains exactly nested at ``rho^2 = 1``."""

    true_prior = SimplifiedPrior(sigma_b2=0.01, tau=2.0)
    dosage, frequencies, phenotype, simplified_ops = _seeded_dense_simulation(
        true_prior,
        seed=44,
        n=256,
        m=220,
        residual_variance=0.2,
    )
    full_ops = EvolutionaryLmmOps.from_dense(dosage, frequencies, model="full")

    simplified_fit = fit_reml(
        simplified_ops,
        phenotype,
        initial=true_prior,
        exact=True,
        seed=1,
        max_iter=30,
    )
    full_fit = fit_reml(
        full_ops,
        phenotype,
        initial=FullPrior(true_prior.sigma_b2, true_prior.tau, rho=1.0),
        exact=True,
        seed=1,
        max_iter=30,
    )

    assert simplified_ops.n == 256
    assert full_fit.prior.rho2 == 1.0
    np.testing.assert_allclose(full_fit.prior.sigma_b2, simplified_fit.prior.sigma_b2, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(full_fit.prior.tau, simplified_fit.prior.tau, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(full_fit.delta, simplified_fit.delta, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(full_fit.sigma_e2, simplified_fit.sigma_e2, rtol=1e-8, atol=1e-12)


@pytest.mark.large
def test_large_grg_matrix_free_fit_matches_dense_parameter_oracle():
    """GRG conversion and matrix-free fitting preserve estimated components.

    The dense dosage matrix below is intentionally materialized only in the
    test.  It comes from GRAPP's raw GRG operator and serves as an oracle for
    the same converted data; production GRG code remains matrix-free.
    """

    true_prior = SimplifiedPrior(sigma_b2=0.01, tau=1.5)
    simulation = simulate_grg_lmm(
        true_prior,
        n_individuals=96,
        sequence_length=100_000,
        population_size=1_000,
        mutation_rate=2e-7,
        residual_variance=0.4,
        seed=14,
    )
    grg_ops = EvolutionaryLmmOps(
        simulation.grg,
        frequencies=simulation.frequencies,
        model="simplified",
    )
    dense_dosage = np.asarray(
        grg_ops._chromosomes[0].raw.matmat(np.eye(grg_ops.n_variants)),
        dtype=np.float64,
    )
    dense_ops = EvolutionaryLmmOps.from_dense(
        dense_dosage,
        simulation.frequencies,
        model="simplified",
    )

    dense_fit = fit_reml(
        dense_ops,
        simulation.phenotype,
        initial=true_prior,
        exact=True,
        seed=3,
        max_iter=30,
    )
    grg_fit = fit_reml(
        grg_ops,
        simulation.phenotype,
        initial=true_prior,
        exact=False,
        trace_probes=512,
        seed=3,
        max_iter=40,
        cg_tol=1e-8,
    )

    assert simulation.n_individuals == 96
    assert simulation.n_variants > 200
    assert dense_dosage.shape == (96, simulation.n_variants)
    assert grg_fit.diagnostics.converged
    assert grg_fit.diagnostics.trace_estimator == "hutchinson"
    probe = np.linspace(-1.0, 1.0, grg_ops.n)
    np.testing.assert_allclose(
        grg_ops.apply_k(probe, grg_fit.prior),
        dense_ops.apply_k(probe, dense_fit.prior),
        rtol=0.3,
        atol=1e-8,
    )
    assert np.isclose(grg_fit.sigma_b2, dense_fit.sigma_b2, rtol=0.25)
    assert np.isclose(grg_fit.sigma_e2, dense_fit.sigma_e2, rtol=0.1)
    assert np.isclose(grg_fit.prior.tau, dense_fit.prior.tau, rtol=0.75)
