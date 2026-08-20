import numpy as np

from evo_lmm import (
    EvolutionaryLmmOps,
    MultiComponentOps,
    MultiComponentPrior,
    SimplifiedPrior,
    fit_multicomponent_reml,
    fit_parameter_profiles,
    fit_report,
)
from evo_lmm.multicomponent import profiled_reml_objective, score_and_information
from evo_lmm.reml import fit_reml


def _ops():
    rng = np.random.default_rng(41)
    genotypes = {
        "lof": rng.binomial(2, 0.08, size=(28, 5)).astype(float),
        "missense": rng.binomial(2, 0.25, size=(28, 7)).astype(float),
    }
    frequencies = {label: matrix.mean(axis=0) / 2 for label, matrix in genotypes.items()}
    return MultiComponentOps.from_dense(genotypes, frequencies), genotypes, frequencies


def _grg_ops():
    import pygrgl

    genotypes = {
        "lof": np.array([[0, 1, 0], [1, 0, 1], [2, 1, 0], [0, 0, 1],
                          [1, 2, 0], [0, 1, 1], [2, 0, 1], [1, 1, 0]], dtype=float),
        "missense": np.array([[1, 0, 1], [0, 2, 0], [1, 1, 0], [2, 0, 1],
                               [0, 1, 2], [1, 0, 0], [0, 2, 1], [2, 1, 0]], dtype=float),
    }
    frequencies = {label: matrix.mean(axis=0) / 2 for label, matrix in genotypes.items()}
    operators = {}
    for label, matrix in genotypes.items():
        grg = pygrgl.MutableGRG(2 * matrix.shape[0], 2)
        for variant in range(matrix.shape[1]):
            node = grg.make_node()
            for individual in range(matrix.shape[0]):
                for haplotype in range(2):
                    if matrix[individual, variant] > haplotype:
                        grg.connect(node, 2 * individual + haplotype)
            grg.add_mutation(
                pygrgl.Mutation(variant + 1, "A", "G"),
                node,
                pygrgl.INVALID_NODE,
            )
        operators[label] = EvolutionaryLmmOps(grg, frequencies=frequencies[label])
    return MultiComponentOps.from_operators(operators), genotypes, frequencies


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
    shifted_tau = MultiComponentPrior(
        ops.labels, (SimplifiedPrior(1.2, 0.7 * np.exp(eps)), SimplifiedPrior(0.8, 1.4))
    )
    numeric_tau = (ops.dense_kernel(shifted_tau) - kernel) / eps
    np.testing.assert_allclose(numeric_tau, derivatives["log_tau[lof]"], rtol=2e-5, atol=1e-8)


def test_zero_tau_is_exact_flat_per_category_nesting():
    ops, _, _ = _ops()
    flat = MultiComponentPrior(
        ops.labels, (SimplifiedPrior(1.2, 0.0), SimplifiedPrior(0.8, 0.0))
    )
    expected = sum(
        scale * ops.components[label].dense_kernel(SimplifiedPrior(1.0, 0.0))
        for label, scale in zip(ops.labels, (1.2, 0.8))
    )
    np.testing.assert_array_equal(ops.dense_kernel(flat), expected)


def test_shared_tau_and_batched_component_derivatives_are_exact():
    ops, _, _ = _ops()
    prior = MultiComponentPrior(
        ops.labels, (SimplifiedPrior(1.2, 0.2), SimplifiedPrior(0.8, 1.4))
    )
    shared = prior.with_shared_tau(0.7)
    np.testing.assert_array_equal(shared.sigma_b2, prior.sigma_b2)
    np.testing.assert_array_equal(shared.tau, [0.7, 0.7])
    expected_shared = sum(
        scale * ops.components[label].dense_kernel(SimplifiedPrior(1.0, 0.7))
        for label, scale in zip(ops.labels, prior.sigma_b2)
    )
    np.testing.assert_array_equal(ops.dense_kernel(shared), expected_shared)
    values = np.random.default_rng(11).normal(size=(ops.n, 3))
    batched = ops.apply_component_derivatives_matmat(values, shared)
    for name, derivative in batched.items():
        label = name.split("[", 1)[1][:-1]
        index = ops.labels.index(label)
        component = shared.components[index]
        expected = component.sigma_b2 * ops.components[label].dense_kernel(component) @ values
        if name.startswith("log_tau"):
            expected = ops.derivative_kernels(shared)[name] @ values
        np.testing.assert_allclose(derivative, expected, atol=1e-10)


def test_matrix_free_partitioned_application_matches_dense_oracle():
    matrix_free, genotypes, frequencies = _grg_ops()
    dense = MultiComponentOps.from_dense(genotypes, frequencies)
    prior = MultiComponentPrior(
        matrix_free.labels, (SimplifiedPrior(1.2, 0.7), SimplifiedPrior(0.8, 1.4))
    )
    values = np.random.default_rng(17).normal(size=(matrix_free.n, 4))
    assert all(chrom.dense is None for op in matrix_free.components.values() for chrom in op._chromosomes)
    np.testing.assert_allclose(
        matrix_free.apply_k(values, prior), dense.apply_k(values, prior), rtol=1e-12, atol=1e-12
    )
    for name, result in matrix_free.apply_component_derivatives_matmat(values, prior).items():
        np.testing.assert_allclose(
            result, dense.derivative_kernels(prior)[name] @ values, rtol=1e-12, atol=1e-12
        )
    flat = MultiComponentPrior.flat(matrix_free.labels, scales=(1.2, 0.8))
    expected_flat = sum(
        scale * matrix_free.components[label].apply_k_matmat(values, SimplifiedPrior(1.0, 0.0))
        for label, scale in zip(matrix_free.labels, (1.2, 0.8))
    )
    np.testing.assert_allclose(matrix_free.apply_k(values, flat), expected_flat, rtol=1e-12, atol=1e-12)
    shared = MultiComponentPrior(
        matrix_free.labels, (SimplifiedPrior(1.2, 0.2), SimplifiedPrior(0.8, 1.4))
    ).with_shared_tau(0.7)
    expected_shared = sum(
        scale * matrix_free.components[label].apply_k_matmat(
            values, SimplifiedPrior(1.0, 0.7)
        )
        for label, scale in zip(matrix_free.labels, shared.sigma_b2)
    )
    np.testing.assert_allclose(
        matrix_free.apply_k(values, shared), expected_shared, rtol=1e-12, atol=1e-12
    )


def test_matrix_free_multicomponent_ai_fit_uses_operator_path():
    ops, _, _ = _grg_ops()
    y = np.random.default_rng(21).normal(size=ops.n)
    strict = fit_multicomponent_reml(
        ops, y, max_iter=1, trace_probes=12
    )
    assert not strict.converged
    assert strict.score_norm > 1e-6
    for trace_method in ("hutchinson", "xtrace"):
        fit = fit_multicomponent_reml(
            ops, y, max_iter=3, trace_method=trace_method, trace_probes=12
        )
        assert np.isfinite(fit.sigma_e2) and fit.sigma_e2 > 0.0
        assert np.isfinite(fit.score_norm)
        assert 0.0 <= fit.h2 <= 1.0
        assert np.all(fit.prior.sigma_b2 > 0.0)
        assert np.all(fit.prior.tau >= 0.0)
        assert fit.trace_method == trace_method
        assert fit.trace_probes == 12
        assert fit.cg_tol == 5e-4
        report = fit_report(fit, maf_bins=[0.0, 0.25, 0.5])
        assert np.isfinite(report.heritability_se)
        assert all(np.isfinite(value) for value in report.component_standard_errors.values())
        profiles = fit_parameter_profiles(
            fit, {"sigma_b2[lof]": [0.5, 1.0, 2.0], "tau[missense]": [0.0, 0.5, 1.0]}
        )
        assert all(np.all(np.isfinite(profile.objective)) for profile in profiles.values())


def test_single_category_fit_delegates_to_existing_fitter_bit_for_bit():
    """Clause 3: one category must reproduce the single-component fit exactly.

    Every reported quantity is compared with ``assert_array_equal``, not a
    tolerance, and the GRGL-backed partition is checked too: the delegation
    used to force ``exact=True``, which would have materialised dense kernels
    from a GRG and ignored the requested solver settings.
    """
    ops, genotypes, frequencies = _ops()
    single_ops = MultiComponentOps.from_dense({"lof": genotypes["lof"]}, {"lof": frequencies["lof"]})
    y = 2.0 * np.random.default_rng(33).normal(size=single_ops.n)
    multi = fit_multicomponent_reml(
        single_ops, y, initial=MultiComponentPrior.flat(("lof",)), max_iter=8
    )
    reference = fit_reml(
        single_ops.components["lof"], y, initial=SimplifiedPrior(1.0, 0.0), max_iter=8,
        trace_method="hutchinson", trace_probes=12, seed=0, cg_tol=5e-4,
    )
    np.testing.assert_array_equal(multi.sigma_e2, reference.sigma_e2)
    np.testing.assert_array_equal(multi.h2, reference.h2)
    np.testing.assert_array_equal(multi.prior.sigma_b2, [reference.prior.sigma_b2])
    np.testing.assert_array_equal(multi.prior.tau, [reference.prior.tau])

    grg_ops, _, _ = _grg_ops()
    label = grg_ops.labels[0]
    single_grg = MultiComponentOps.from_operators({label: grg_ops.components[label]})
    assert all(chrom.dense is None for chrom in single_grg.components[label]._chromosomes)
    y_grg = np.random.default_rng(34).normal(size=single_grg.n)
    multi_grg = fit_multicomponent_reml(
        single_grg, y_grg, initial=MultiComponentPrior.flat((label,)), max_iter=8,
        trace_probes=8, seed=2,
    )
    reference_grg = fit_reml(
        single_grg.components[label], y_grg, initial=SimplifiedPrior(1.0, 0.0), max_iter=8,
        trace_method="hutchinson", trace_probes=8, seed=2, cg_tol=5e-4,
    )
    np.testing.assert_array_equal(multi_grg.sigma_e2, reference_grg.sigma_e2)
    np.testing.assert_array_equal(multi_grg.h2, reference_grg.h2)
    np.testing.assert_array_equal(multi_grg.prior.sigma_b2, [reference_grg.prior.sigma_b2])
    np.testing.assert_array_equal(multi_grg.prior.tau, [reference_grg.prior.tau])


def test_dense_and_grg_backed_multicomponent_fits_agree_at_the_default_cg_tol():
    """Clause 2: a whole fit, not only an operator application, must agree.

    The operator-level oracle test above never enters the CG solve, so it
    cannot pin the GRGL traversal through a fit at the production ``cg_tol``.
    Both fits use identical seeds and probe budgets, so the only difference is
    the traversal itself.
    """
    grg_ops, genotypes, frequencies = _grg_ops()
    dense_ops = MultiComponentOps.from_dense(genotypes, frequencies)
    assert all(chrom.dense is None
               for op in grg_ops.components.values() for chrom in op._chromosomes)
    y = np.random.default_rng(21).normal(size=grg_ops.n)
    settings = dict(max_iter=12, trace_probes=12, seed=3)
    dense_fit = fit_multicomponent_reml(dense_ops, y, **settings)
    grg_fit = fit_multicomponent_reml(grg_ops, y, **settings)
    assert dense_fit.cg_tol == 5e-4 and grg_fit.cg_tol == 5e-4
    assert dense_fit.converged == grg_fit.converged
    np.testing.assert_allclose(grg_fit.sigma_e2, dense_fit.sigma_e2, rtol=1e-9)
    np.testing.assert_allclose(grg_fit.h2, dense_fit.h2, rtol=1e-9)
    np.testing.assert_allclose(grg_fit.prior.sigma_b2, dense_fit.prior.sigma_b2, rtol=1e-8)
    np.testing.assert_allclose(grg_fit.prior.tau, dense_fit.prior.tau, rtol=1e-8)


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


def _oracle_state(ops, y, prior):
    """Independent dense profiled-REML score and average information.

    Built from an explicit projected inverse rather than from any production
    helper, so it can pin ``score_and_information``.
    """
    from evo_lmm.reml import _dense_projection

    shape = np.eye(ops.n) + ops.dense_kernel(prior)
    projected, _inverse, _logdet = _dense_projection(shape, ops.basis)
    ph_y = projected @ y
    q = float(y @ ph_y)
    sigma_e2 = q / ops.dim
    names = [f"log_{parameter}[{label}]" for label in ops.labels
             for parameter in ("sigma_b2", "tau")]
    derivatives = ops.derivative_kernels(prior)
    data = np.asarray([float(ph_y @ (derivatives[name] @ ph_y)) for name in names])
    traces = np.asarray([float(np.trace(projected @ derivatives[name])) for name in names])
    score = 0.5 * (data / sigma_e2 - traces)
    zeta = np.column_stack([projected @ (derivatives[name] @ ph_y) for name in names])
    direct = np.asarray([[0.5 * float(ph_y @ (derivatives[left] @ zeta[:, column]))
                          for column in range(len(names))] for left in names])
    ai = (direct - np.outer(data, data) / (2.0 * q)) / sigma_e2
    return score, (ai + ai.T) * 0.5, sigma_e2


def _exact_trace_probes(n):
    """Rademacher-style probes that make the Hutchinson trace exact.

    ``z_i = sqrt(n) e_i`` gives ``mean_i z_i' A z_i = tr(A)`` with no variance,
    so the stochastic estimator can be compared against a dense oracle exactly.
    """
    return np.sqrt(float(n)) * np.eye(n)


def test_score_matches_finite_difference_gradient_of_profiled_objective():
    """The score must be the gradient of ``profiled_reml_objective``.

    Regression: the data quadratic was not divided by the profiled
    ``sigma_e2``, so the score was off by that factor -- a quantity that is not
    the gradient of any objective.  The fixture keeps ``sigma_e2`` far from one
    so the omission cannot hide.
    """
    ops, _, _ = _ops()
    y = 3.0 * np.random.default_rng(77).normal(size=ops.n)
    prior = MultiComponentPrior(
        ops.labels, (SimplifiedPrior(0.6, 2.5), SimplifiedPrior(0.3, 7.0))
    )
    state = score_and_information(
        ops, y, prior, _exact_trace_probes(ops.n), cg_tol=1e-12
    )
    assert abs(state["sigma_e2"] - 1.0) > 0.5, "fixture no longer exercises the profiled scale"
    coordinates = prior.coordinates
    step = 1e-6
    gradient = np.empty(coordinates.size)
    for index in range(coordinates.size):
        shift = np.zeros(coordinates.size)
        shift[index] = step
        plus = profiled_reml_objective(
            ops, y, MultiComponentPrior.from_coordinates(ops.labels, coordinates + shift))[0]
        minus = profiled_reml_objective(
            ops, y, MultiComponentPrior.from_coordinates(ops.labels, coordinates - shift))[0]
        gradient[index] = (plus - minus) / (2.0 * step)
    np.testing.assert_allclose(state["score"], -gradient, rtol=1e-5, atol=1e-7)
    oracle_score, _oracle_ai, oracle_sigma_e2 = _oracle_state(ops, y, prior)
    np.testing.assert_allclose(state["score"], oracle_score, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(state["sigma_e2"], oracle_sigma_e2, rtol=1e-10)


def test_average_information_matches_dense_oracle_and_is_positive_semidefinite():
    """The AI matrix must equal the dense oracle and be PSD.

    Regression: the left contraction used ``P dV_i P y`` instead of ``P y``,
    inserting a third derivative factor.  The result was indefinite, so the
    solved step was not a descent direction and every step halving was
    rejected.  Both checks below fail against that construction.
    """
    ops, _, _ = _ops()
    y = 3.0 * np.random.default_rng(78).normal(size=ops.n)
    prior = MultiComponentPrior(
        ops.labels, (SimplifiedPrior(0.6, 2.5), SimplifiedPrior(0.3, 7.0))
    )
    state = score_and_information(
        ops, y, prior, _exact_trace_probes(ops.n), cg_tol=1e-12
    )
    _oracle_score, oracle_ai, _sigma_e2 = _oracle_state(ops, y, prior)
    np.testing.assert_allclose(state["ai"], oracle_ai, rtol=1e-7, atol=1e-9)
    assert np.min(np.linalg.eigvalsh(state["ai"])) >= -1e-8
    assert np.min(np.linalg.eigvalsh(oracle_ai)) >= -1e-8


def test_seeded_two_category_fit_converges_and_recovers_heritability():
    """A converged fit must reach the truth's likelihood, not merely return.

    Regression: with the score and AI defects above the fitter stalled on the
    first iteration -- it returned ``converged=False`` with ``h2`` pinned near
    one and ``tau`` collapsed to zero, at a profiled objective far worse than
    the objective at the generating parameters.  A third defect kept it there:
    a rejected line search left the loop instead of escalating the Levenberg
    damping, and an undamped step in the weakly identified ``tau`` directions
    is capped to one that barely moves the scale coordinates.
    """
    n = 600
    rng = np.random.default_rng(202)
    genotypes, frequencies, contribution = {}, {}, np.zeros(n)
    truth_parameters = (("lof", 90, 0.2, 30.0), ("missense", 120, 0.1, 5.0))
    residual_sd = 1.0
    for label, count, scale, tau in truth_parameters:
        p = rng.beta(0.4, 0.4, size=count) * 0.49 + 0.01
        matrix = rng.binomial(2, p, size=(n, count)).astype(float)
        freq = matrix.mean(axis=0) / 2.0
        q = freq * (1.0 - freq)
        contribution += matrix @ rng.normal(0.0, np.sqrt(scale / (1.0 + 2.0 * tau * q)))
        genotypes[label] = matrix
        frequencies[label] = freq
    y = contribution + rng.normal(0.0, residual_sd, size=n)
    ops = MultiComponentOps.from_dense(genotypes, frequencies)
    realized_h2 = contribution.var() / (contribution.var() + residual_sd ** 2)

    fit = fit_multicomponent_reml(ops, y, max_iter=60, trace_probes=12, seed=1)

    assert fit.converged, f"step_se_norm={fit.step_se_norm}"
    assert fit.status == "converged"
    assert fit.step_se_norm <= 1e-2
    # The old absolute score gate would not have been met here; the criterion
    # is the scale-free statistic, not ||score||_inf.
    assert fit.score_norm > 1e-6
    assert abs(fit.h2 - realized_h2) < 0.1, f"h2={fit.h2} against {realized_h2}"
    assert np.all(fit.prior.tau > 0.0), "tau collapsed to the zero boundary"
    ratio = MultiComponentPrior(
        ops.labels,
        tuple(SimplifiedPrior(p.sigma_b2 / fit.sigma_e2, p.tau) for p in fit.prior.components),
    )
    truth_ratio = MultiComponentPrior(
        ops.labels,
        tuple(SimplifiedPrior(scale / residual_sd ** 2, tau)
              for _label, _count, scale, tau in truth_parameters),
    )
    assert (profiled_reml_objective(ops, y, ratio)[0]
            <= profiled_reml_objective(ops, y, truth_ratio)[0] + 1e-6)
    assert np.isfinite(fit.trace_standard_error)


def test_pooled_shape_fit_holds_tau_fixed_while_a_free_fit_moves_it():
    """``fit_tau=False`` must search the scales only.

    The pooled-shape mode exists so per-gene scales can be conditioned on
    shapes estimated once across genes; a fit that re-estimates ``tau_c`` per
    gene is not a pooled-shape fit.
    """
    ops, _, _ = _ops()
    y = 2.0 * np.random.default_rng(91).normal(size=ops.n)
    initial = MultiComponentPrior(
        ops.labels, (SimplifiedPrior(1.0, 0.4), SimplifiedPrior(1.0, 1.3))
    )
    pooled = fit_multicomponent_reml(ops, y, initial=initial, max_iter=20, fit_tau=False)
    free = fit_multicomponent_reml(ops, y, initial=initial, max_iter=20)
    np.testing.assert_array_equal(pooled.prior.tau, initial.tau)
    assert np.any(free.prior.tau != initial.tau), "fixture no longer moves tau when free"
    assert np.all(pooled.prior.sigma_b2 > 0.0)
    assert pooled.standard_errors is not None
    for label in ops.labels:
        assert pooled.standard_errors[f"log_tau[{label}]"] == 0.0
        assert pooled.standard_errors[f"log_sigma_b2[{label}]"] > 0.0


def test_status_names_how_the_fit_ended():
    """``converged`` is a summary; ``status`` says which exit was taken.

    A single boolean previously conflated the criterion being met, the
    near-tolerance fallback after a rejected line search, an exhausted
    iteration budget, and SciPy's verdict on the exact dense method.
    """
    ops, _, _ = _ops()
    y = 2.0 * np.random.default_rng(5).normal(size=ops.n)
    initial = MultiComponentPrior(
        ops.labels, (SimplifiedPrior(0.9, 0.4), SimplifiedPrior(0.6, 1.1))
    )
    not_started = fit_multicomponent_reml(ops, y, initial=initial, max_iter=0)
    assert not_started.status == "not_started" and not not_started.converged
    assert np.isnan(not_started.step_se_norm)

    capped = fit_multicomponent_reml(ops, y, initial=initial, max_iter=1)
    assert capped.status == "iteration_cap" and not capped.converged
    assert capped.step_se_norm > 1e-2

    done = fit_multicomponent_reml(ops, y, max_iter=40)
    assert done.status == "converged" and done.converged
    assert done.step_se_norm <= 1e-2
    assert np.isfinite(done.newton_decrement)

    dense = fit_multicomponent_reml(ops, y, max_iter=200, method="dense")
    # The dense method reports SciPy's verdict, not this module's criterion.
    assert dense.status in ("optimizer_success", "optimizer_failure")
    assert dense.converged == (dense.status == "optimizer_success")

    for fit in (not_started, capped, done):
        assert fit.converged == (fit.status in ("converged", "stalled_near_tolerance"))


def test_tau_boundary_is_reported_without_changing_convergence():
    """Decision 4, report only: a boundary hit is surfaced, not acted on."""
    from evo_lmm.multicomponent import _tau_warnings

    ops, _, _ = _ops()
    y = 2.0 * np.random.default_rng(5).normal(size=ops.n)
    fit = fit_multicomponent_reml(ops, y, max_iter=40)
    assert np.all(fit.prior.tau <= 1e-6), "fixture no longer drives tau to the flat kernel"
    assert len(fit.warnings) == len(ops.labels)
    assert all("at or near zero" in message for message in fit.warnings)
    # Reported, but the fit is still a converged fit: a flat kernel is a
    # legitimate estimate, not a failure.
    assert fit.status == "converged" and fit.converged

    # The other regime: every weight crushed, so only tau * q is identified.
    saturated = MultiComponentPrior(
        ops.labels, (SimplifiedPrior(1.0, 1e12), SimplifiedPrior(1.0, 1e12))
    )
    messages = _tau_warnings(ops, saturated)
    assert len(messages) == len(ops.labels)
    assert all("saturate" in message for message in messages)

    # An interior tau is not reported at all.
    interior = MultiComponentPrior(
        ops.labels, (SimplifiedPrior(1.0, 2.0), SimplifiedPrior(1.0, 5.0))
    )
    assert _tau_warnings(ops, interior) == ()
