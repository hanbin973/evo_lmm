import numpy as np

from evo_lmm import (
    MultiComponentOps,
    collapse_mac,
    fit_rare_effect_baseline,
    rare_effect_mom_ratio,
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
    return genotypes, MultiComponentOps.from_dense(genotypes, frequencies)


def test_mac_collapse_recomputes_frequency_and_keeps_provenance():
    genotypes = {
        "c": np.array(
            [[1, 1, 0], [1, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=float
        )
    }
    result = collapse_mac(genotypes, mac_threshold=2)
    assert result.genotypes["c"].shape == (4, 2)
    assert result.source_indices["c"] == ((0,), (1, 2))
    np.testing.assert_allclose(result.frequencies["c"], [0.25, 0.25])


def test_rare_effect_ratio_uses_non_positive_moment_fallback():
    result = rare_effect_mom_ratio(
        marginal_scales=np.array([2.0, 3.0]),
        marginal_mom_scales=np.array([1.0, -1.0]),
        joint_mom_scales=np.array([1.5, 2.0]),
    )
    np.testing.assert_allclose(result.adjusted_scales, [3.0, 3.0])
    np.testing.assert_array_equal(result.negative_mom_fallback, [False, True])


def _error_contrast_rare_effect(genotypes, basis, y):
    """Independently reproduce the baseline through error contrasts."""
    from scipy.linalg import null_space
    from scipy.optimize import minimize_scalar

    contrasts = null_space(basis.T)
    reduced_y = contrasts.T @ y
    dimension = reduced_y.size
    reduced = {
        label: (contrasts.T @ matrix) @ (contrasts.T @ matrix).T
        for label, matrix in genotypes.items()
    }
    labels = list(genotypes)
    marginal, marginal_mom = [], []
    for label in labels:
        kernel = reduced[label]
        eigenvalues, vectors = np.linalg.eigh(kernel)
        rotated = vectors.T @ reduced_y

        def objective(log_delta, eigenvalues=eigenvalues, rotated=rotated):
            delta = np.exp(log_delta)
            quadratic = float(np.sum(rotated**2 / (eigenvalues + delta)))
            return 0.5 * (
                float(np.sum(np.log(eigenvalues + delta)))
                + dimension * np.log(quadratic / dimension)
            )

        best = minimize_scalar(
            objective,
            bounds=(-30.0, 30.0),
            method="bounded",
            options={"xatol": 1e-12},
        )
        marginal.append(float(np.sum(rotated**2 / (eigenvalues + np.exp(best.x)))) / dimension)
        system = np.array(
            [
                [float(np.sum(eigenvalues**2)), float(np.sum(eigenvalues))],
                [float(np.sum(eigenvalues)), float(dimension)],
            ]
        )
        rhs = np.array(
            [float(reduced_y @ kernel @ reduced_y), float(reduced_y @ reduced_y)]
        )
        marginal_mom.append(float(np.linalg.solve(system, rhs)[0]))

    count = len(labels)
    system = np.empty((count + 1, count + 1))
    rhs = np.empty(count + 1)
    for i, left in enumerate(labels):
        for j, right in enumerate(labels):
            system[i, j] = float(np.trace(reduced[left] @ reduced[right]))
        system[i, count] = system[count, i] = float(np.trace(reduced[left]))
        rhs[i] = float(reduced_y @ reduced[left] @ reduced_y)
    system[count, count] = float(dimension)
    rhs[count] = float(reduced_y @ reduced_y)
    joint = np.linalg.solve(system, rhs)[:count]
    marginal = np.asarray(marginal)
    marginal_mom = np.asarray(marginal_mom)
    fallback = (marginal_mom <= 0.0) | (joint <= 0.0)
    adjusted = marginal.copy()
    adjusted[~fallback] = (marginal * joint / marginal_mom)[~fallback]
    return marginal, marginal_mom, joint, adjusted, fallback


def test_rare_effect_baseline_matches_error_contrast_oracle():
    genotypes, ops = _fixture()
    rng = np.random.default_rng(19)
    y = (
        genotypes["lof"] @ rng.normal(0.0, 0.8, genotypes["lof"].shape[1])
        + genotypes["missense"]
        @ rng.normal(0.0, 0.5, genotypes["missense"].shape[1])
        + rng.normal(0.0, 1.5, ops.n)
    )
    result = fit_rare_effect_baseline(ops, y)
    marginal, marginal_mom, joint, adjusted, fallback = _error_contrast_rare_effect(
        genotypes, ops.basis, y
    )
    assert not np.any(fallback)
    np.testing.assert_allclose(result.marginal_scales, marginal, rtol=1e-6)
    np.testing.assert_allclose(result.marginal_mom_scales, marginal_mom, rtol=1e-9)
    np.testing.assert_allclose(result.joint_mom_scales, joint, rtol=1e-9)
    np.testing.assert_allclose(result.adjusted_scales, adjusted, rtol=1e-6)
    np.testing.assert_array_equal(result.negative_mom_fallback, fallback)
    np.testing.assert_allclose(
        result.marginal_scales, [0.5312628449, 0.2067134205], rtol=1e-6
    )
    np.testing.assert_allclose(
        result.adjusted_scales, [0.5195693392, 0.1896107076], rtol=1e-6
    )


def test_rare_effect_baseline_falls_back_on_non_positive_moment_estimate():
    genotypes, ops = _fixture()
    y = 3.0 * np.random.default_rng(19).normal(size=ops.n)
    result = fit_rare_effect_baseline(ops, y)
    _, marginal_mom, joint, _, fallback = _error_contrast_rare_effect(
        genotypes, ops.basis, y
    )
    assert fallback[0]
    np.testing.assert_array_equal(result.negative_mom_fallback, fallback)
    np.testing.assert_allclose(result.marginal_mom_scales, marginal_mom, rtol=1e-9)
    np.testing.assert_allclose(result.joint_mom_scales, joint, rtol=1e-9)
    np.testing.assert_array_equal(result.adjusted_scales[0], result.marginal_scales[0])
