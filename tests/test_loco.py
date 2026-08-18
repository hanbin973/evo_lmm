import numpy as np

from evo_lmm import EvolutionaryLmmOps, FullPrior


def test_loco_excludes_exact_chromosome_without_renormalisation():
    rng = np.random.default_rng(11)
    x0 = rng.binomial(2, 0.2, size=(16, 3)).astype(float)
    x1 = rng.binomial(2, 0.7, size=(16, 4)).astype(float)
    ops = EvolutionaryLmmOps.from_dense(
        [x0, x1], [x0.mean(0) / 2.0, x1.mean(0) / 2.0], model="full"
    )
    prior = FullPrior(1.0, 0.4, 0.5)
    v = rng.normal(size=16)
    expected = EvolutionaryLmmOps.from_dense(x1, x1.mean(0) / 2.0, model="full").apply_k(v, prior)
    assert np.allclose(ops.apply_k(v, prior, exclude_chrom=0), expected)

