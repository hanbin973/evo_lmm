"""End-to-end evolutionary LMM/BOLT-style fitting helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.special import ndtr

from .operators import EvolutionaryLmmOps
from .results import AssociationResult, FitResult
from .reml import fit_reml


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


def association(
    result: FitResult,
    y: np.ndarray | None = None,
    *,
    use_loco: bool = True,
) -> list[AssociationResult]:
    """Compute compact score/beta/SE association results from test operators.

    The test columns are BOLT-normalised and remain separate from the
    frequency-weighted model columns.  This CPU implementation intentionally
    returns typed arrays instead of coupling the core to a DataFrame package.
    """

    if result.ops is None:
        raise ValueError("fit result is not attached to an operator")
    phenotype = result.projected_phenotype if y is None else np.asarray(y, dtype=np.float64)
    solutions = loco_solutions(result, phenotype) if use_loco else {None: result.ph_y}
    output: list[AssociationResult] = []
    for chrom in result.ops.chroms:
        solution = solutions[chrom] if use_loco else solutions[None]
        scores = result.ops.test_scores(chrom, solution)
        # The score variance uses the test-column norm on the fitted residual
        # subspace.  It is conservative but remains in the same tested-genotype
        # units as GRAPP's downstream calibration interface.
        nvar = scores.size
        se = np.empty(nvar, dtype=np.float64)
        beta = np.empty(nvar, dtype=np.float64)
        local_idx = result.ops.local_indices(chrom)
        for idx, score in enumerate(scores):
            # test_column is projected and normalised, hence its Euclidean norm
            # is an appropriate stable denominator for this score statistic.
            column = result.ops.test_column(chrom, int(local_idx[idx]))
            norm2 = max(float(column @ column), np.finfo(float).tiny)
            se[idx] = np.sqrt((result.sigma_e2 + result.sigma_b2) / norm2)
            beta[idx] = score / norm2
        chisq = (beta / np.maximum(se, np.finfo(float).tiny)) ** 2
        pvalue = 2.0 * ndtr(-np.sqrt(np.maximum(chisq, 0.0)))
        output.append(AssociationResult(chrom, local_idx, scores, beta, se, chisq, pvalue))
    return output


def predict_blup(result: FitResult) -> np.ndarray:
    """Return projected genetic-value BLUPs."""

    return result.blup()


fit_evolutionary_bolt = fit_evolutionary_bolt_lmm
