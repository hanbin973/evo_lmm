def test_pinned_grapp_adapter_exposes_raw_operator_api():
    from evo_lmm.grapp_backend import GRAPP_COMMIT, assert_compatible

    assert len(GRAPP_COMMIT) == 40
    assert_compatible()


def test_small_grg_raw_kernel_matches_dense_dosage():
    import numpy as np
    import pygrgl

    from evo_lmm import EvolutionaryLmmOps, SimplifiedPrior

    dosage = np.array([[0, 0], [1, 0], [2, 1], [0, 2], [1, 1], [2, 0]], dtype=int)
    grg = pygrgl.MutableGRG(2 * dosage.shape[0], 2)
    for variant in range(dosage.shape[1]):
        node = grg.make_node()
        for individual in range(dosage.shape[0]):
            for haplotype in range(2):
                if dosage[individual, variant] > haplotype:
                    grg.connect(node, 2 * individual + haplotype)
        grg.add_mutation(
            pygrgl.Mutation(variant + 1, "A", "G"),
            node,
            pygrgl.INVALID_NODE,
        )
    frequencies = dosage.mean(axis=0) / 2.0
    grg_ops = EvolutionaryLmmOps(grg)
    dense_ops = EvolutionaryLmmOps.from_dense(dosage.astype(float), frequencies)
    prior = SimplifiedPrior(1.0, 0.5)
    vector = np.arange(dosage.shape[0], dtype=float)
    assert np.allclose(grg_ops.apply_k(vector, prior), dense_ops.apply_k(vector, prior))
    assert np.isclose(grg_ops.kernel_trace(prior), dense_ops.kernel_trace(prior))
