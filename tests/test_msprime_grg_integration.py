import numpy as np
import pytest

from evo_lmm import (
    FullPrior,
    SimplifiedPrior,
    fit_evolutionary_bolt_lmm,
    sample_allele_frequencies,
    simulate_grg_lmm,
)


@pytest.mark.parametrize(
    "prior",
    [SimplifiedPrior(sigma_b2=0.6, tau=0.4), FullPrior(sigma_b2=0.6, tau=0.4, rho=0.7)],
)
def test_msprime_tree_sequence_grg_fit_samples_effects_from_prior(prior):
    simulation = simulate_grg_lmm(
        prior,
        n_individuals=12,
        sequence_length=50_000,
        population_size=500,
        mutation_rate=5e-7,
        seed=91,
    )

    assert simulation.tree_sequence.num_samples == 24
    assert simulation.grg.num_individuals == 12
    assert simulation.grg.num_mutations > 0
    assert simulation.effects.shape == simulation.frequencies.shape
    assert np.all(np.isfinite(simulation.effects))
    assert np.all(simulation.prior.effect_variances(simulation.frequencies) > 0.0)
    np.testing.assert_allclose(
        simulation.frequencies,
        sample_allele_frequencies(simulation.grg),
    )

    fit = fit_evolutionary_bolt_lmm(
        [("simulated", simulation.grg)],
        simulation.phenotype,
        frequencies={"simulated": simulation.frequencies},
        model=prior.model_name,
        initial=prior,
        trace_probes=4,
        max_iter=4,
        seed=22,
    )
    assert fit.model == prior.model_name
    assert fit.ops.n == simulation.n_individuals
    assert np.isfinite(fit.sigma_b2) and fit.sigma_b2 > 0.0
    assert np.isfinite(fit.sigma_e2) and fit.sigma_e2 > 0.0

