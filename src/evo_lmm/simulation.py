"""Reproducible msprime -> GRG simulations for examples and integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any

import msprime
import numpy as np
import pygrgl

from .grg_data import sample_allele_frequencies
from .operators import EvolutionaryLmmOps
from .priors import EvolutionaryPrior


@dataclass(frozen=True)
class GrgSimulation:
    """A simulated GRG phenotype with effects sampled from an evolutionary prior."""

    tree_sequence: Any
    grg: Any
    frequencies: np.ndarray
    effects: np.ndarray
    genetic_value: np.ndarray
    phenotype: np.ndarray
    prior: EvolutionaryPrior
    residual_variance: float
    seed: int

    @property
    def n_individuals(self) -> int:
        return int(self.grg.num_individuals)

    @property
    def n_variants(self) -> int:
        return int(self.grg.num_mutations)


def simulate_grg_lmm(
    prior: EvolutionaryPrior,
    *,
    n_individuals: int = 24,
    sequence_length: float = 100_000.0,
    population_size: float = 1_000.0,
    recombination_rate: float = 1e-8,
    mutation_rate: float = 2e-7,
    residual_variance: float = 0.4,
    seed: int = 7,
) -> GrgSimulation:
    """Simulate a diploid tree sequence, convert it to a GRG, and draw a trait.

    The returned ``effects`` are raw-dosage SNP effects sampled independently
    as ``Normal(0, prior.effect_variances(frequencies))``.  The genetic value
    is computed by the GRG-backed raw operator and projected off the intercept,
    matching the fitting boundary used by :func:`fit_evolutionary_bolt_lmm`.

    The temporary ``.trees`` file is removed after GRGL conversion.  The GRG
    object and original in-memory tree sequence remain available in the result.
    """

    prior.validate()
    if int(n_individuals) < 2:
        raise ValueError("n_individuals must be at least 2")
    if float(sequence_length) <= 0.0:
        raise ValueError("sequence_length must be positive")
    if float(population_size) <= 0.0:
        raise ValueError("population_size must be positive")
    if float(recombination_rate) < 0.0 or float(mutation_rate) <= 0.0:
        raise ValueError("recombination_rate must be non-negative and mutation_rate positive")
    if float(residual_variance) <= 0.0:
        raise ValueError("residual_variance must be positive")

    tree_sequence = msprime.sim_ancestry(
        samples=int(n_individuals),
        ploidy=2,
        sequence_length=float(sequence_length),
        population_size=float(population_size),
        recombination_rate=float(recombination_rate),
        random_seed=int(seed),
    )
    tree_sequence = msprime.sim_mutations(
        tree_sequence,
        rate=float(mutation_rate),
        random_seed=int(seed) + 1,
    )
    if tree_sequence.num_mutations == 0:
        raise RuntimeError("msprime produced no mutations; increase mutation_rate or sequence_length")

    with tempfile.TemporaryDirectory(prefix="evo_lmm_msprime_") as directory:
        trees_path = Path(directory) / "simulation.trees"
        tree_sequence.dump(str(trees_path))
        grg = pygrgl.grg_from_trees(str(trees_path))

    frequencies = sample_allele_frequencies(grg)
    if frequencies.size == 0:
        raise RuntimeError("GRGL conversion produced no mutations")
    rng = np.random.default_rng(int(seed) + 2)
    effects = rng.normal(
        loc=0.0,
        scale=np.sqrt(prior.effect_variances(frequencies)),
    )
    ops = EvolutionaryLmmOps(grg, frequencies=frequencies, model=prior.model_name)
    genetic_value = ops.apply_model_x(effects)
    phenotype = genetic_value + rng.normal(
        loc=0.0,
        scale=np.sqrt(float(residual_variance)),
        size=ops.n,
    )
    return GrgSimulation(
        tree_sequence=tree_sequence,
        grg=grg,
        frequencies=frequencies,
        effects=effects,
        genetic_value=genetic_value,
        phenotype=phenotype,
        prior=prior,
        residual_variance=float(residual_variance),
        seed=int(seed),
    )

