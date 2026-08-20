"""End-to-end evolutionary LMM/BOLT-style fitting helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.special import erfc

from .calibration import (
    DEFAULT_CALIBRATION_VARIANTS,
    DEFAULT_SCREEN_THRESHOLD,
    CalibrationResult,
    calibrate_association,
    uncalibrated_scale,
)
from .operators import EvolutionaryLmmOps
from .results import AssociationResult, FitResult
from .reml import fit_reml

# Median of a one-degree-of-freedom chi-square, used to normalise lambda_GC.
_CHI2_1DF_MEDIAN = 0.4549364231195732


def _filter_dense(genotypes: Any, keep: np.ndarray) -> Any:
    if isinstance(genotypes, np.ndarray):
        return np.asarray(genotypes)[keep]
    if isinstance(genotypes, Mapping):
        return {label: np.asarray(matrix)[keep] for label, matrix in genotypes.items()}
    values = list(genotypes)
    if all(isinstance(value, tuple) and len(value) == 2 for value in values):
        return [(label, np.asarray(matrix)[keep]) for label, matrix in values]
    return [np.asarray(matrix)[keep] for matrix in values]


def fit_evolutionary_lmm(
    genotypes: Any,
    y: np.ndarray,
    frequencies: Any = None,
    *,
    covariates: np.ndarray | None = None,
    model: str = "simplified",
    initial: Any = None,
    sample_filter: Sequence[int] | None = None,
    trace_probes: int = 64,
    seed: int = 0,
    max_iter: int = 50,
    cg_tol: float = 1e-9,
    exact: bool | None = None,
    trace_method: str = "hutchinson",
    warm_start: bool = True,
    initialization: str = "default",
) -> FitResult:
    """Fit a simplified or full evolutionary LMM from dense matrices or GRGs.

    Missing phenotypes are removed before constructing the operator.  For GRGs,
    the corresponding individual sample filter is passed to GRAPP and allele
    frequencies are recomputed on the retained sample.
    """

    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim != 1:
        raise ValueError("y must be one-dimensional")
    missing = np.isnan(y_arr)
    keep = np.flatnonzero(~missing)
    if keep.size == 0:
        raise ValueError("phenotype has no observed individuals")
    if np.any(missing):
        genotypes = _filter_dense(genotypes, keep) if _is_dense_input(genotypes) else genotypes
        if covariates is not None:
            covariates = np.asarray(covariates)[keep]
        y_arr = y_arr[keep]
        sample_filter = keep
    elif sample_filter is not None:
        sample_filter = np.asarray(sample_filter, dtype=np.int64)
        y_arr = y_arr[sample_filter]
        if covariates is not None:
            covariates = np.asarray(covariates)[sample_filter]
        if _is_dense_input(genotypes):
            genotypes = _filter_dense(genotypes, sample_filter)

    # Frequencies are sample statistics.  Once an individual mask is applied,
    # recompute them from the retained dosage/GRG data rather than silently
    # carrying frequencies from the original sample.
    if sample_filter is not None:
        frequencies = None

    ops = EvolutionaryLmmOps(
        genotypes,
        frequencies,
        covariates,
        sample_filter=None if _is_dense_input(genotypes) else sample_filter,
        model=model,
    )
    return fit_reml(
        ops,
        y_arr,
        initial=initial,
        trace_probes=trace_probes,
        seed=seed,
        max_iter=max_iter,
        cg_tol=cg_tol,
        exact=exact,
        trace_method=trace_method,
        warm_start=warm_start,
        initialization=initialization,
    )


def fit_evolutionary_bolt_lmm(
    chrom_grgs: Sequence[tuple[Any, Any]],
    y: np.ndarray,
    *,
    frequencies: Mapping[Any, np.ndarray] | Sequence[np.ndarray] | None = None,
    covariates: np.ndarray | None = None,
    model: str = "simplified",
    initial: Any = None,
    sample_filter: Sequence[int] | None = None,
    trace_probes: int = 64,
    seed: int = 0,
    max_iter: int = 50,
    cg_tol: float = 1e-9,
    trace_method: str = "hutchinson",
    warm_start: bool = True,
    initialization: str = "default",
) -> FitResult:
    """Fit the CPU GRGL-backed evolutionary model across chromosomes."""

    return fit_evolutionary_lmm(
        list(chrom_grgs),
        y,
        frequencies,
        covariates=covariates,
        model=model,
        initial=initial,
        sample_filter=sample_filter,
        trace_probes=trace_probes,
        seed=seed,
        max_iter=max_iter,
        cg_tol=cg_tol,
        exact=False,
        trace_method=trace_method,
        warm_start=warm_start,
        initialization=initialization,
    )


def _is_dense_input(genotypes: Any) -> bool:
    if isinstance(genotypes, np.ndarray):
        return True
    if isinstance(genotypes, Mapping):
        return all(isinstance(value, np.ndarray) for value in genotypes.values())
    values = list(genotypes)
    return all(
        isinstance(value, np.ndarray)
        or (isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], np.ndarray))
        for value in values
    )


def _coordinates(result: FitResult) -> np.ndarray:
    return result.prior.coordinates(result.delta)


def loco_solve(result: FitResult, y: np.ndarray, exclude_chrom: Any) -> np.ndarray:
    """Solve the fitted projected LOCO shape system for one chromosome."""

    if result.ops is None:
        raise ValueError("fit result is not attached to an operator")
    return result.ops.solve_ph(np.asarray(y, dtype=np.float64), _coordinates(result), exclude_chrom=exclude_chrom)


def loco_solutions(result: FitResult, y: np.ndarray | None = None) -> dict[Any, np.ndarray]:
    """Return one projected ``P_H y`` solution per chromosome exclusion."""

    if result.ops is None:
        raise ValueError("fit result is not attached to an operator")
    phenotype = result.projected_phenotype if y is None else np.asarray(y, dtype=np.float64)
    return {
        chrom: loco_solve(result, phenotype, chrom)
        for chrom in result.ops.chroms
    }


def _chi2_sf_1df(values: np.ndarray) -> np.ndarray:
    """Survival function of a one-degree-of-freedom chi-square."""

    with np.errstate(invalid="ignore"):
        return erfc(np.sqrt(np.asarray(values, dtype=np.float64) * 0.5))


def association(
    result: FitResult,
    y: np.ndarray | None = None,
    *,
    use_loco: bool = True,
    calibrate: bool = True,
    calibration: CalibrationResult | None = None,
    calibration_variants: int = DEFAULT_CALIBRATION_VARIANTS,
    seed: int = 0,
    screen_threshold: float = DEFAULT_SCREEN_THRESHOLD,
    cg_tol: float = 1e-9,
) -> list[AssociationResult]:
    """Compute calibrated BOLT-style association statistics per chromosome.

    The mixed-model statistic uses the *fitted evolutionary* LOCO covariance
    ``V_loco = sigma_b2 * (K_evo,loco + delta I)`` while the tested columns keep
    GRAPP's independent BOLT normalisation.  With ``calibrate=True`` the
    prospective/retrospective moment matching of
    :func:`evo_lmm.calibrate_association` supplies the per-chromosome inverse
    scale; otherwise the uncalibrated (``factor = 1``) scale is used, which is
    only appropriate for diagnostics.

    ``beta`` and ``se`` are returned in raw diploid-dosage units.  A
    single-variant linear-regression chi-square is reported alongside the mixed
    model statistic so inflation can be compared directly.
    """

    ops = result.ops
    if ops is None:
        raise ValueError("fit result is not attached to an operator")
    phenotype = result.projected_phenotype if y is None else np.asarray(y, dtype=np.float64)
    projected_phenotype = ops.project(phenotype)
    phenotype_norm2 = float(projected_phenotype @ projected_phenotype)
    if phenotype_norm2 <= 0.0:
        raise ValueError("phenotype has nonpositive projected norm")
    stats = {chrom: ops.test_stats(chrom) for chrom in ops.chroms}
    dim = float(max(ops.dim, 1))

    if not use_loco:
        if calibration is not None or calibrate:
            # Calibration is defined by leave-one-chromosome-out moments; there
            # is no honest calibration for the in-sample statistic.
            calibrate = False
            calibration = None
        residuals = {chrom: result.ph_y for chrom in ops.chroms}
    elif calibrate:
        if calibration is None:
            calibration = calibrate_association(
                result,
                phenotype,
                count=calibration_variants,
                seed=seed,
                screen_threshold=screen_threshold,
                cg_tol=cg_tol,
                stats=stats,
            )
        residuals = calibration.residuals
        if not residuals:
            residuals = loco_solutions(result, phenotype)
    else:
        residuals = loco_solutions(result, phenotype)

    output: list[AssociationResult] = []
    for chrom in ops.chroms:
        item = stats[chrom]
        residual = residuals[chrom]
        if calibration is not None:
            inverse_scale = float(calibration.inverse_scale[chrom])
            factor = float(calibration.factor)
        else:
            inverse_scale = uncalibrated_scale(result, float(residual @ residual))
            factor = 1.0
        mask = np.asarray(item.model_mask, dtype=bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            # ``test_scores`` are BOLT-normalised; multiplying by ``test_scale``
            # returns to raw diploid-dosage score units before the 1/sigma_b2
            # factor of V^-1 is applied.
            raw_score = (
                np.asarray(ops.test_scores(chrom, residual), dtype=np.float64)
                * item.test_scale
                / float(result.sigma_b2)
            )
            score = raw_score / inverse_scale
            chisq = (score * score) / item.projected_norm2
            beta = raw_score / (item.projected_norm2 * inverse_scale * inverse_scale)
            se = 1.0 / (np.sqrt(item.projected_norm2) * inverse_scale)
            linreg_score = np.asarray(
                ops.test_scores(chrom, projected_phenotype), dtype=np.float64
            )
            chisq_linreg = (
                linreg_score * linreg_score
            ) / phenotype_norm2 / item.std_projected_norm2 * dim
        pvalue = np.where(mask, _chi2_sf_1df(np.where(mask, chisq, 0.0)), 1.0)
        pvalue_linreg = np.where(
            mask, _chi2_sf_1df(np.where(mask, chisq_linreg, 0.0)), 1.0
        )
        nan = float("nan")
        output.append(
            AssociationResult(
                chrom=chrom,
                local_idx=item.local_idx,
                score=np.where(mask, score, nan),
                beta=np.where(mask, beta, nan),
                se=np.where(mask, se, nan),
                chisq=np.where(mask, chisq, nan),
                pvalue=pvalue,
                chisq_linreg=np.where(mask, chisq_linreg, nan),
                pvalue_linreg=pvalue_linreg,
                model_mask=mask,
                frequencies=np.asarray(
                    ops.chromosome_frequencies(chrom), dtype=np.float64
                ),
                inverse_scale=inverse_scale,
                calibration_factor=factor,
            )
        )
    return output


def association_summary(results: Sequence[AssociationResult]) -> dict[str, dict[str, float]]:
    """Return mean chi-square and lambda_GC for the mixed model and linreg.

    ``lambda_gc`` divides the median chi-square by the median of a
    one-degree-of-freedom chi-square, so a calibrated null gives values near
    one for both statistics.
    """

    summary: dict[str, dict[str, float]] = {}
    for key, attribute in (("lmm", "chisq"), ("linreg", "chisq_linreg")):
        values: list[np.ndarray] = []
        for item in results:
            array = getattr(item, attribute)
            if array is None:
                continue
            array = np.asarray(array, dtype=np.float64)
            values.append(array[item.good() & np.isfinite(array)])
        pooled = np.concatenate(values) if values else np.empty(0, dtype=np.float64)
        if pooled.size:
            summary[key] = {
                "mean": float(np.mean(pooled)),
                "lambda_gc": float(np.median(pooled) / _CHI2_1DF_MEDIAN),
                "n_good": int(pooled.size),
            }
        else:
            summary[key] = {"mean": float("nan"), "lambda_gc": float("nan"), "n_good": 0}
    return summary


def predict_blup(result: FitResult) -> np.ndarray:
    """Return projected genetic-value BLUPs."""

    return result.blup()


fit_evolutionary_bolt = fit_evolutionary_bolt_lmm
