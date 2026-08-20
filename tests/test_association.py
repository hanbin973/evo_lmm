"""Priority 0 tests for the calibrated BOLT-style association path."""

from __future__ import annotations

import math

import numpy as np
import pytest

import pygrgl

from evo_lmm import (
    EvolutionaryLmmOps,
    FullPrior,
    SimplifiedPrior,
    association,
    association_summary,
    calibrate_association,
    fit_evolutionary_bolt_lmm,
    fit_evolutionary_lmm,
    fit_reml,
    loco_solve,
    predict_blup,
    select_calibration_variants,
    simulate_grg_lmm,
)
from evo_lmm.calibration import CALIBRATION_STD_LIMIT
from evo_lmm.grapp_backend import raw_operator_class
from evo_lmm.results import FitDiagnostics, FitResult

# The production default CG tolerance matches GRAPP's loose ``5e-4``.  Tests
# that compare against an explicit dense pseudo-inverse, or that assert the
# exact ``rho^2 = 1`` nested identity, pin a tight tolerance instead so they
# measure the formulas rather than the solver's truncation.
ORACLE_CG_TOL = 1e-9


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fixed_result(ops, y, prior, delta):
    """Build a fit result at a fixed prior, bypassing the optimizer.

    This isolates the association/calibration formulas from optimizer paths so
    two models can be compared at exactly the same covariance.
    """

    projected = ops.project(np.asarray(y, dtype=np.float64))
    coordinates = prior.coordinates(delta)
    ph_y = ops.solve_ph(projected, coordinates, tol=ORACLE_CG_TOL)
    q = float(projected @ ph_y)
    sigma_b2 = q / max(ops.dim, 1)
    diagnostics = FitDiagnostics(
        converged=True,
        iterations=0,
        objective=float("nan"),
        score_norm=0.0,
        ai_condition=1.0,
        ai_damping=0.0,
        accepted_step=0.0,
        trace_estimator="exact",
        trace_probes=0,
    )
    return FitResult(
        prior=prior.with_sigma_b2(sigma_b2),
        sigma_b2=sigma_b2,
        sigma_e2=delta * sigma_b2,
        delta=float(delta),
        h2=float("nan"),
        log_likelihood=float("nan"),
        fixed_effects=np.zeros(ops.rank),
        projected_phenotype=projected,
        ph_y=ph_y,
        diagnostics=diagnostics,
        model=prior.model_name,
        ops=ops,
    )


def _dense_genotypes(grg):
    """Materialise a small GRG as a raw dosage matrix for oracle comparisons."""

    operator = raw_operator_class(grg)(grg, pygrgl.TraversalDirection.UP, dtype=np.float64)
    identity = np.eye(int(operator.shape[1]), dtype=np.float64)
    return np.asarray(operator.matmat(identity), dtype=np.float64)


def _dense_loco_solve(ops, prior, delta, rhs, exclude_chrom):
    """Dense oracle for ``H_loco^-1 P rhs`` using an explicit pseudo-inverse."""

    projector = np.eye(ops.n) - ops.basis @ ops.basis.T
    kernel = ops.dense_kernel(prior, exclude_chrom=exclude_chrom)
    matrix = projector @ (kernel + delta * np.eye(ops.n)) @ projector
    return np.linalg.pinv(matrix, rcond=1e-12) @ (projector @ np.asarray(rhs, dtype=np.float64))


def _two_chromosome_dense(seed=3, n=60, m0=14, m1=17):
    rng = np.random.default_rng(seed)
    x0 = rng.binomial(2, rng.uniform(0.15, 0.6, size=m0), size=(n, m0)).astype(float)
    x1 = rng.binomial(2, rng.uniform(0.15, 0.6, size=m1), size=(n, m1)).astype(float)
    return {"1": x0, "2": x1}


# ---------------------------------------------------------------------------
# dense oracle
# ---------------------------------------------------------------------------


def test_association_matches_dense_pseudoinverse_oracle():
    genotypes = _two_chromosome_dense()
    rng = np.random.default_rng(17)
    y = rng.normal(size=next(iter(genotypes.values())).shape[0])
    ops = EvolutionaryLmmOps(genotypes, model="simplified")
    prior = SimplifiedPrior(sigma_b2=1.0, tau=0.8)
    delta = 1.7
    result = _fixed_result(ops, y, prior, delta)

    selected, _ = select_calibration_variants(result, count=6, seed=5)
    computed = calibrate_association(result, count=6, seed=5, cg_tol=ORACLE_CG_TOL)
    assert list(computed.selected) == list(selected)

    dim = float(ops.dim)
    shape = prior.with_sigma_b2(1.0)
    residuals = {
        chrom: _dense_loco_solve(ops, shape, delta, result.projected_phenotype, chrom)
        for chrom in ops.chroms
    }
    for chrom, value in residuals.items():
        np.testing.assert_allclose(computed.residuals[chrom], value, rtol=1e-7, atol=1e-9)

    prospective = []
    retrospective = []
    for chrom, local_idx in selected:
        column = ops.test_column(chrom, local_idx)
        solved = _dense_loco_solve(ops, shape, delta, column, chrom)
        score = float(column @ residuals[chrom])
        retrospective.append(
            dim * score * score / (float(residuals[chrom] @ residuals[chrom]) * float(column @ column))
        )
        prospective.append(
            dim
            * score
            * score
            / float(column @ solved)
            / float(result.projected_phenotype @ residuals[chrom])
        )
    pro = np.asarray(prospective)
    retro = np.asarray(retrospective)
    np.testing.assert_allclose(computed.prospective, pro, rtol=1e-6)
    np.testing.assert_allclose(computed.retrospective, retro, rtol=1e-6)

    factor = pro.sum() / retro.sum()
    jackknife = (pro.sum() - pro) / (retro.sum() - retro)
    std = math.sqrt(
        max(0.0, (np.sum(jackknife**2) - jackknife.sum() ** 2 / jackknife.size) * (jackknife.size - 1) / jackknife.size)
    )
    if std > CALIBRATION_STD_LIMIT:
        factor = float(np.median(pro) / np.median(retro))
    assert computed.factor == pytest.approx(factor, rel=1e-6)

    results = association(result, calibration=computed, cg_tol=ORACLE_CG_TOL)
    for item in results:
        chrom = item.chrom
        stats = ops.test_stats(chrom)
        inverse_scale = 1.0 / (
            math.sqrt(dim / float(residuals[chrom] @ residuals[chrom]) * factor) * result.sigma_b2
        )
        assert item.inverse_scale == pytest.approx(inverse_scale, rel=1e-6)
        raw_score = (
            np.asarray(ops.test_scores(chrom, residuals[chrom])) * stats.test_scale / result.sigma_b2
        )
        beta = raw_score / (stats.projected_norm2 * inverse_scale**2)
        se = 1.0 / (np.sqrt(stats.projected_norm2) * inverse_scale)
        mask = item.good()
        np.testing.assert_allclose(item.beta[mask], beta[mask], rtol=1e-6)
        np.testing.assert_allclose(item.se[mask], se[mask], rtol=1e-6)
        np.testing.assert_allclose(
            item.chisq[mask], (beta[mask] / se[mask]) ** 2, rtol=1e-8
        )


def test_association_beta_se_and_chisq_are_mutually_consistent():
    genotypes = _two_chromosome_dense(seed=9)
    rng = np.random.default_rng(4)
    y = rng.normal(size=next(iter(genotypes.values())).shape[0])
    ops = EvolutionaryLmmOps(genotypes, model="simplified")
    result = _fixed_result(ops, y, SimplifiedPrior(sigma_b2=1.0, tau=0.3), 2.0)
    for item in association(result, calibration_variants=5, seed=2):
        mask = item.good()
        np.testing.assert_allclose(
            item.chisq[mask], (item.beta[mask] / item.se[mask]) ** 2, rtol=1e-10
        )
        # ``score``, ``beta`` and ``se`` share one denominator: the projected
        # test-column norm scaled by the calibrated inverse scale.
        np.testing.assert_allclose(
            item.score[mask],
            item.beta[mask] / (item.se[mask] ** 2 * item.inverse_scale),
            rtol=1e-8,
        )
        assert np.all(item.se[mask] > 0.0)
        assert np.all(item.pvalue[mask] > 0.0) and np.all(item.pvalue[mask] <= 1.0)


# ---------------------------------------------------------------------------
# nested-model identity
# ---------------------------------------------------------------------------


def test_full_model_at_rho2_one_matches_simplified_association():
    genotypes = _two_chromosome_dense(seed=21)
    rng = np.random.default_rng(31)
    y = rng.normal(size=next(iter(genotypes.values())).shape[0])
    delta = 1.3
    tau = 0.65

    simplified_ops = EvolutionaryLmmOps(genotypes, model="simplified")
    full_ops = EvolutionaryLmmOps(genotypes, model="full")
    simplified = _fixed_result(simplified_ops, y, SimplifiedPrior(sigma_b2=1.0, tau=tau), delta)
    full = _fixed_result(full_ops, y, FullPrior(sigma_b2=1.0, tau=tau, rho=1.0), delta)

    assert simplified.sigma_b2 == pytest.approx(full.sigma_b2, rel=1e-10)
    left = association(simplified, calibration_variants=6, seed=13, cg_tol=ORACLE_CG_TOL)
    right = association(full, calibration_variants=6, seed=13, cg_tol=ORACLE_CG_TOL)
    assert [item.chrom for item in left] == [item.chrom for item in right]
    for a, b in zip(left, right):
        assert a.calibration_factor == pytest.approx(b.calibration_factor, rel=1e-9)
        for name in ("beta", "se", "chisq", "score", "chisq_linreg", "pvalue"):
            np.testing.assert_allclose(
                getattr(a, name), getattr(b, name), rtol=1e-9, atol=1e-12
            )
    np.testing.assert_allclose(predict_blup(simplified), predict_blup(full), rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# dense versus GRG equivalence
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_chromosome_grgs():
    prior = SimplifiedPrior(sigma_b2=0.5, tau=0.5)
    left = simulate_grg_lmm(
        prior,
        n_individuals=40,
        sequence_length=60_000,
        population_size=800,
        mutation_rate=6e-7,
        seed=101,
    )
    right = simulate_grg_lmm(
        prior,
        n_individuals=40,
        sequence_length=60_000,
        population_size=800,
        mutation_rate=6e-7,
        seed=202,
    )
    phenotype = left.genetic_value + right.genetic_value
    rng = np.random.default_rng(303)
    phenotype = phenotype + rng.normal(scale=0.6, size=phenotype.size)
    return left, right, phenotype


def test_dense_and_grg_association_agree(two_chromosome_grgs):
    left, right, phenotype = two_chromosome_grgs
    grg_chroms = [("1", left.grg), ("2", right.grg)]
    dense_chroms = {"1": _dense_genotypes(left.grg), "2": _dense_genotypes(right.grg)}
    np.testing.assert_allclose(dense_chroms["1"].mean(axis=0) / 2.0, left.frequencies, atol=1e-12)

    # Dense and GRG operators are mathematically identical, so this test pins
    # the tight CG tolerance: at the production default the two paths agree
    # only to the solver's truncation level, which would mask a real
    # operator-level discrepancy.
    settings = dict(
        model="simplified",
        initial=SimplifiedPrior(0.5, 0.5),
        max_iter=3,
        trace_probes=8,
        seed=41,
        cg_tol=ORACLE_CG_TOL,
    )
    grg_fit = fit_evolutionary_bolt_lmm(grg_chroms, phenotype, **settings)
    dense_fit = fit_evolutionary_lmm(dense_chroms, phenotype, exact=False, **settings)

    assert grg_fit.sigma_b2 == pytest.approx(dense_fit.sigma_b2, rel=1e-6)
    assert grg_fit.tau == pytest.approx(dense_fit.tau, rel=1e-6)
    assert grg_fit.delta == pytest.approx(dense_fit.delta, rel=1e-6)

    grg_assoc = association(grg_fit, calibration_variants=8, seed=7, cg_tol=ORACLE_CG_TOL)
    dense_assoc = association(dense_fit, calibration_variants=8, seed=7, cg_tol=ORACLE_CG_TOL)
    for a, b in zip(grg_assoc, dense_assoc):
        assert a.chrom == b.chrom
        np.testing.assert_array_equal(a.local_idx, b.local_idx)
        np.testing.assert_array_equal(a.good(), b.good())
        assert a.calibration_factor == pytest.approx(b.calibration_factor, rel=1e-5)
        mask = a.good()
        for name in ("beta", "se", "chisq", "chisq_linreg"):
            np.testing.assert_allclose(
                getattr(a, name)[mask], getattr(b, name)[mask], rtol=1e-5, atol=1e-8
            )


# ---------------------------------------------------------------------------
# null calibration
# ---------------------------------------------------------------------------


def _null_dense_panel(seed, n=300, m0=150, m1=160):
    rng = np.random.default_rng(seed)
    f0 = rng.uniform(0.1, 0.5, size=m0)
    f1 = rng.uniform(0.1, 0.5, size=m1)
    x0 = rng.binomial(2, f0, size=(n, m0)).astype(float)
    x1 = rng.binomial(2, f1, size=(n, m1)).astype(float)
    return rng, {"1": x0, "2": x1}


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_seeded_pure_null_statistics_are_calibrated(seed):
    rng, genotypes = _null_dense_panel(seed)
    y = rng.normal(size=genotypes["1"].shape[0])
    fit = fit_evolutionary_lmm(genotypes, y, model="simplified", max_iter=40, exact=True)
    assert fit.diagnostics.converged
    calibration = calibrate_association(fit, count=25, seed=seed)
    assert calibration.n_selected == 25
    assert np.all(calibration.prospective > 0.0)
    assert np.all(calibration.retrospective > 0.0)
    assert all(scale > 0.0 for scale in calibration.inverse_scale.values())
    # With no genetic signal the fitted covariance collapses towards the
    # identity, so the calibration factor must be close to one.
    assert calibration.factor == pytest.approx(1.0, abs=0.15)

    summary = association_summary(association(fit, calibration=calibration))
    assert summary["lmm"]["n_good"] == 310
    assert summary["lmm"]["mean"] == pytest.approx(1.0, abs=0.2)
    assert summary["lmm"]["lambda_gc"] == pytest.approx(1.0, abs=0.35)
    # A null LMM statistic must not be inflated relative to plain regression.
    assert summary["lmm"]["mean"] == pytest.approx(summary["linreg"]["mean"], abs=0.15)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_seeded_polygenic_null_chromosome_is_not_inflated(seed):
    rng, genotypes = _null_dense_panel(seed)
    prior = SimplifiedPrior(sigma_b2=0.02, tau=0.5)
    causal = genotypes["1"]
    effects = rng.normal(scale=np.sqrt(prior.effect_variances(causal.mean(axis=0) / 2.0)))
    genetic = causal @ effects
    genetic = (genetic - genetic.mean()) / genetic.std()
    y = math.sqrt(0.5) * genetic + math.sqrt(0.5) * rng.normal(size=genetic.size)

    fit = fit_evolutionary_lmm(genotypes, y, model="simplified", max_iter=40, exact=True)
    assert fit.diagnostics.converged
    calibration = calibrate_association(fit, count=25, seed=seed)
    assert calibration.factor == pytest.approx(1.0, abs=0.25)

    results = association(fit, calibration=calibration)
    null_chromosome = association_summary([item for item in results if item.chrom == "2"])
    assert null_chromosome["lmm"]["mean"] == pytest.approx(1.0, abs=0.4)
    # Chromosome 2 carries no signal, so the mixed-model statistic there must
    # stay in line with the plain-regression statistic on the same variants.
    assert null_chromosome["lmm"]["mean"] == pytest.approx(
        null_chromosome["linreg"]["mean"], abs=0.25
    )
    pooled = np.concatenate([item.pvalue[item.good()] for item in results if item.chrom == "2"])
    assert np.all((pooled > 0.0) & (pooled <= 1.0))


def test_grg_calibration_is_well_formed(two_chromosome_grgs):
    left, right, _ = two_chromosome_grgs
    rng = np.random.default_rng(555)
    null_phenotype = rng.normal(size=int(left.grg.num_individuals))
    fit = fit_evolutionary_bolt_lmm(
        [("1", left.grg), ("2", right.grg)],
        null_phenotype,
        model="simplified",
        initial=SimplifiedPrior(0.2, 0.4),
        max_iter=4,
        trace_probes=8,
        seed=61,
    )
    calibration = calibrate_association(fit, count=12, seed=71)
    assert calibration.n_selected == 12
    assert calibration.tried >= 12
    assert np.all(calibration.prospective > 0.0)
    assert np.all(calibration.retrospective > 0.0)
    assert calibration.factor > 0.0
    assert set(calibration.inverse_scale) == {"1", "2"}
    results = association(fit, calibration=calibration)
    pooled = np.concatenate([item.pvalue[item.good()] for item in results])
    assert np.all((pooled > 0.0) & (pooled <= 1.0))


def test_uncalibrated_statistics_are_available_but_flagged(two_chromosome_grgs):
    left, right, phenotype = two_chromosome_grgs
    fit = fit_evolutionary_bolt_lmm(
        [("1", left.grg), ("2", right.grg)],
        phenotype,
        model="simplified",
        initial=SimplifiedPrior(0.5, 0.5),
        max_iter=3,
        trace_probes=8,
        seed=41,
    )
    plain = association(fit, calibrate=False)
    assert all(item.calibration_factor == 1.0 for item in plain)
    in_sample = association(fit, use_loco=False)
    assert all(item.calibration_factor == 1.0 for item in in_sample)
    # LOCO removes the tested chromosome from the covariance, so the in-sample
    # statistic is systematically deflated relative to the LOCO statistic.
    loco_mean = float(np.nanmean(np.concatenate([item.chisq for item in plain])))
    in_sample_mean = float(np.nanmean(np.concatenate([item.chisq for item in in_sample])))
    assert in_sample_mean < loco_mean


# ---------------------------------------------------------------------------
# missing phenotypes and covariates end to end
# ---------------------------------------------------------------------------


def test_missing_phenotypes_and_covariates_flow_through_the_pipeline(two_chromosome_grgs):
    left, right, phenotype = two_chromosome_grgs
    n = phenotype.size
    rng = np.random.default_rng(808)
    covariates = np.column_stack(
        (rng.normal(size=n), rng.integers(0, 2, size=n).astype(float))
    )
    observed = phenotype + covariates @ np.asarray([0.8, -0.4])
    missing = np.asarray([1, 5, 9])
    with_nan = observed.copy()
    with_nan[missing] = np.nan
    keep = np.setdiff1d(np.arange(n), missing)

    fit = fit_evolutionary_bolt_lmm(
        [("1", left.grg), ("2", right.grg)],
        with_nan,
        covariates=covariates,
        model="simplified",
        initial=SimplifiedPrior(0.5, 0.5),
        max_iter=3,
        trace_probes=8,
        seed=41,
    )
    assert fit.ops.n == keep.size
    # The intercept plus two covariates are removed by projection.
    assert fit.ops.rank == 3
    assert fit.projected_phenotype.size == keep.size
    assert np.isfinite(fit.sigma_b2) and fit.sigma_b2 > 0.0
    assert fit.fixed_effects.size == fit.ops.rank

    solved = loco_solve(fit, fit.projected_phenotype, "1")
    assert solved.shape == (keep.size,)
    # Covariate directions are annihilated by the projection.
    np.testing.assert_allclose(fit.ops.basis.T @ solved, 0.0, atol=1e-8)
    blup = predict_blup(fit)
    assert blup.shape == (keep.size,)
    np.testing.assert_allclose(fit.ops.basis.T @ blup, 0.0, atol=1e-8)

    results = association(fit, calibration_variants=8, seed=13)
    assert [item.chrom for item in results] == ["1", "2"]
    for item in results:
        mask = item.good()
        assert mask.any()
        assert np.all(np.isfinite(item.beta[mask]))
        assert np.all(np.isfinite(item.se[mask]))
        assert np.all(item.se[mask] > 0.0)
        assert np.all(np.isnan(item.beta[~mask]))
        assert np.all(item.pvalue[~mask] == 1.0)
        assert item.frequencies.shape == item.local_idx.shape

    # Frequencies are recomputed on the retained sample, not carried over.
    retained_frequencies = association(fit, calibration_variants=8, seed=13)[0].frequencies
    assert retained_frequencies.shape[0] > 0
    assert not np.allclose(retained_frequencies, left.frequencies)


def test_explicit_sample_filter_matches_prefiltered_dense_fit():
    genotypes = _two_chromosome_dense(seed=44, n=50)
    rng = np.random.default_rng(45)
    y = rng.normal(size=50)
    keep = np.arange(0, 50, 2)
    filtered = fit_evolutionary_lmm(
        genotypes,
        y,
        sample_filter=keep,
        model="simplified",
        initial=SimplifiedPrior(1.0, 0.5),
        max_iter=3,
        exact=True,
    )
    manual = fit_evolutionary_lmm(
        {label: matrix[keep] for label, matrix in genotypes.items()},
        y[keep],
        model="simplified",
        initial=SimplifiedPrior(1.0, 0.5),
        max_iter=3,
        exact=True,
    )
    assert filtered.sigma_b2 == pytest.approx(manual.sigma_b2, rel=1e-9)
    left = association(filtered, calibration_variants=5, seed=6)
    right = association(manual, calibration_variants=5, seed=6)
    for a, b in zip(left, right):
        np.testing.assert_allclose(a.beta, b.beta, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(a.chisq, b.chisq, rtol=1e-8, atol=1e-12)
