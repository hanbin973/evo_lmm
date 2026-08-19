import numpy as np

from evo_lmm import EvolutionaryLmmOps, FullPrior, SimplifiedPrior


def _data():
    rng = np.random.default_rng(4)
    x1 = rng.binomial(2, 0.3, size=(18, 4)).astype(float)
    x2 = rng.binomial(2, 0.6, size=(18, 3)).astype(float)
    f1 = x1.mean(axis=0) / 2.0
    f2 = x2.mean(axis=0) / 2.0
    ops = EvolutionaryLmmOps.from_dense([x1, x2], [f1, f2], model="full")
    return ops, x1, x2, f1, f2


def test_projected_weighted_kernel_is_dense_and_psd():
    ops, x1, x2, f1, f2 = _data()
    prior = FullPrior(1.0, 0.8, 0.7)
    rng = np.random.default_rng(5)
    vector = rng.normal(size=ops.n)
    p = ops.project(vector)
    expected = ops.dense_kernel(prior) @ p
    assert np.allclose(ops.apply_k(vector, prior), expected)
    kernel = ops.dense_kernel(prior)
    assert np.allclose(kernel, kernel.T)
    assert np.min(np.linalg.eigvalsh(kernel)) >= -1e-10


def test_derivative_and_loco_match_dense_removal():
    ops, x1, x2, f1, f2 = _data()
    prior = FullPrior(1.0, 0.8, 0.7)
    vector = np.random.default_rng(6).normal(size=ops.n)
    derivative = ops.apply_dk(vector, prior, "log_tau")
    plus = FullPrior(1.0, 0.8 * np.exp(1e-6), 0.7)
    minus = FullPrior(1.0, 0.8 * np.exp(-1e-6), 0.7)
    numeric = (ops.apply_k(vector, plus) - ops.apply_k(vector, minus)) / (2e-6)
    assert np.allclose(derivative, numeric, rtol=1e-5, atol=1e-7)

    loco = ops.apply_k(vector, prior, exclude_chrom=0)
    expected_ops = EvolutionaryLmmOps.from_dense(x2, f2, model="full")
    assert np.allclose(loco, expected_ops.apply_k(vector, prior), atol=1e-10)


def test_batched_h_derivative_matches_column_applications():
    ops, *_ = _data()
    coordinates = np.array([np.log(0.4), np.log(0.8), 0.7])
    values = np.random.default_rng(16).normal(size=(ops.n, 5))

    for parameter in ("log_delta", "log_tau", "logit_r"):
        expected = np.column_stack(
            [ops.apply_dh(values[:, column], coordinates, parameter) for column in range(values.shape[1])]
        )
        assert np.allclose(
            ops.apply_dh_matmat(values, coordinates, parameter),
            expected,
            atol=1e-10,
        )


def test_model_and_test_operator_are_distinct_when_weights_vary():
    ops, *_ = _data()
    prior = SimplifiedPrior(1.0, 1.2)
    vector = np.random.default_rng(7).normal(size=ops.n)
    model_scores = ops.model_scores(vector)
    test_scores = np.concatenate([ops.test_scores(chrom, vector) for chrom in ops.chroms])
    assert not np.allclose(model_scores, test_scores)
