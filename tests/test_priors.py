import numpy as np

from evo_lmm import FullPrior, SimplifiedPrior


def test_prior_boundaries_and_nested_identity():
    frequencies = np.array([0.0, 0.1, 0.5, 1.0])
    simplified = SimplifiedPrior(2.0, 3.0)
    full = FullPrior(2.0, 3.0, 1.0)
    assert np.array_equal(simplified.weights(frequencies), full.weights(frequencies))
    assert np.all(simplified.weights(frequencies[[0, 3]]) == 1.0)
    assert np.all(FullPrior(1.0, 10.0, 0.0).weights(frequencies) == 1.0)
    assert np.all(SimplifiedPrior(1.0, 0.0).weights(frequencies) == 1.0)


def test_weights_are_positive_and_monotone():
    q_freq = np.linspace(0.01, 0.49, 25)
    low = SimplifiedPrior(1.0, 0.2).weights(q_freq)
    high = SimplifiedPrior(1.0, 2.0).weights(q_freq)
    assert np.all(low > 0.0)
    assert np.all(high > 0.0)
    assert np.all(np.diff(low) <= 0.0)
    assert np.all(high <= low)


def test_analytic_derivatives_match_finite_difference():
    frequencies = np.array([0.03, 0.2, 0.4, 0.8])
    tau = 0.7
    simplified = SimplifiedPrior(1.0, tau)
    eps = 1e-6
    numeric = (
        SimplifiedPrior(1.0, tau * np.exp(eps)).weights(frequencies)
        - SimplifiedPrior(1.0, tau * np.exp(-eps)).weights(frequencies)
    ) / (2.0 * eps)
    assert np.allclose(numeric, simplified.weight_derivatives(frequencies)["log_tau"], rtol=1e-7, atol=1e-9)

    full = FullPrior(1.0, tau, 0.6)
    numeric_tau = (
        FullPrior(1.0, tau * np.exp(eps), 0.6).weights(frequencies)
        - FullPrior(1.0, tau * np.exp(-eps), 0.6).weights(frequencies)
    ) / (2.0 * eps)
    assert np.allclose(numeric_tau, full.weight_derivatives(frequencies)["log_tau"], rtol=1e-7, atol=1e-9)
    r = full.rho2
    logit = np.log(r) - np.log1p(-r)
    r_plus = 1.0 / (1.0 + np.exp(-(logit + eps)))
    r_minus = 1.0 / (1.0 + np.exp(-(logit - eps)))
    numeric_r = (
        FullPrior(1.0, tau, np.sqrt(r_plus)).weights(frequencies)
        - FullPrior(1.0, tau, np.sqrt(r_minus)).weights(frequencies)
    ) / (2.0 * eps)
    assert np.allclose(numeric_r, full.weight_derivatives(frequencies)["logit_r"], rtol=1e-6, atol=1e-9)

