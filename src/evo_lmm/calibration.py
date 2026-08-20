"""BOLT-style calibration of evolutionary LMM association statistics.

The calibration procedure mirrors GRAPP's ``calibrate_lmm_inf`` and BOLT-LMM's
``Bolt::computeStats`` moment matching, with one deliberate substitution: the
LOCO covariance is the *fitted evolutionary* covariance

``V_loco = sigma_b2 * (K_evo,loco + delta * I)``,

where ``K_evo = P_C X diag(w(tau[, rho])) X^T P_C`` uses raw diploid dosage
columns and frequency-dependent weights.  The tested-genotype columns stay on
GRAPP's independent BOLT normalisation, so ``sigma_b2`` keeps its raw-effect
meaning and is never reinterpreted as GRAPP's standardized ``sigma_g2``.

All statistics below are therefore computed in *shape* units: the solves return
``H^-1 P_C v`` with ``H = K_evo + delta I`` and the single ``1 / sigma_b2``
factor of ``V^-1`` is applied explicitly where it belongs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .operators import TestVariantStats
from .results import FitResult

# GRAPP/BOLT reject a candidate calibration variant whose GRAMMAR retrospective
# statistic exceeds this value, so that calibration is estimated from variants
# that are plausibly null.
DEFAULT_SCREEN_THRESHOLD = 5.0
DEFAULT_CALIBRATION_VARIANTS = 30
# BOLT switches from the ratio of sums to the ratio of medians when the
# jackknife standard error of the factor is above this value.
CALIBRATION_STD_LIMIT = 0.01
_MAX_BLOCK_ATTEMPTS = 1_000_000


@dataclass(frozen=True)
class CalibrationResult:
    """Prospective/retrospective calibration of the evolutionary LMM statistic.

    Attributes
    ----------
    factor:
        The applied calibration factor: the ratio of prospective to
        retrospective statistic sums, or the ratio of medians when the
        jackknife standard error exceeds :data:`CALIBRATION_STD_LIMIT`.
    inverse_scale:
        Per-chromosome ``VinvScaleFactor`` in raw-effect units.  A tested
        chromosome's calibrated inverse-variance score is
        ``(x_j^T H_loco^-1 P y) / sigma_b2 / inverse_scale[chrom]``.
    residuals:
        ``H_loco^-1 P_C y`` for each left-out chromosome, reused by
        :func:`evo_lmm.association` so the solves are not repeated.
    """

    factor: float
    std: float
    ratio_of_medians: float
    median_of_ratios: float
    selected: tuple[tuple[Any, int], ...]
    tried: int
    prospective: np.ndarray
    retrospective: np.ndarray
    inverse_scale: dict[Any, float]
    residuals: dict[Any, np.ndarray] = field(repr=False, default_factory=dict)
    quadratic_form: dict[Any, float] = field(default_factory=dict)
    screen_threshold: float = DEFAULT_SCREEN_THRESHOLD
    seed: int = 0

    @property
    def n_selected(self) -> int:
        return len(self.selected)


def _require_ops(result: FitResult) -> Any:
    if result.ops is None:
        raise ValueError("fit result is not attached to an operator")
    return result.ops


def _phenotype(result: FitResult, y: np.ndarray | None) -> np.ndarray:
    values = result.projected_phenotype if y is None else np.asarray(y, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("y must be one-dimensional")
    return values


def _stats_by_chrom(ops: Any) -> dict[Any, TestVariantStats]:
    return {chrom: ops.test_stats(chrom) for chrom in ops.chroms}


def select_calibration_variants(
    result: FitResult,
    *,
    count: int = DEFAULT_CALIBRATION_VARIANTS,
    seed: int = 0,
    screen_threshold: float = DEFAULT_SCREEN_THRESHOLD,
    stats: dict[Any, TestVariantStats] | None = None,
) -> tuple[list[tuple[Any, int]], int]:
    """Select calibration variants with BOLT's blocked GRAMMAR pre-screen.

    Eligible variants are split into ``count`` equal blocks in chromosome
    order.  One variant is drawn uniformly from each block and accepted when its
    all-chromosome GRAMMAR retrospective statistic falls below
    ``screen_threshold``, which keeps strongly associated variants out of the
    calibration set.  Returns the selected ``(chrom, local_idx)`` pairs and the
    number of candidates examined.
    """

    ops = _require_ops(result)
    num_calib = int(count)
    if num_calib < 2:
        raise ValueError("at least two calibration variants are required")
    variant_stats = _stats_by_chrom(ops) if stats is None else stats

    segments: list[tuple[Any, int, np.ndarray, TestVariantStats]] = []
    offset = 0
    for chrom in ops.chroms:
        item = variant_stats[chrom]
        positions = np.flatnonzero(item.model_mask)
        if positions.size == 0:
            continue
        segments.append((chrom, offset, positions, item))
        offset += int(positions.size)
    model_count = offset
    if model_count <= 0:
        raise ValueError("no eligible variants are available for calibration")
    if num_calib > model_count:
        raise ValueError(
            f"requested {num_calib} calibration variants but only {model_count} are eligible"
        )

    def locate(flat: int) -> tuple[Any, np.ndarray, TestVariantStats, int]:
        for chrom, start, positions, item in segments:
            if flat < start + positions.size:
                return chrom, positions, item, flat - start
        raise IndexError(flat)

    # Equal-width blocks over the flat eligible index, matching BOLT's
    # ``mFirst`` construction.
    block_starts = [model_count] * (num_calib + 1)
    for flat in range(model_count):
        block = num_calib * flat // model_count
        if block_starts[block] == model_count:
            block_starts[block] = flat
    if any(start == model_count for start in block_starts[:-1]):
        raise RuntimeError("failed to build calibration blocks")

    all_hinv_norm2 = float(result.ph_y @ result.ph_y)
    if all_hinv_norm2 <= 0.0:
        raise RuntimeError("all-chromosome H^-1 y has nonpositive norm")
    grammar_scores = {
        chrom: np.asarray(ops.test_scores(chrom, result.ph_y), dtype=np.float64)
        for chrom, _, _, _ in segments
    }

    rng = np.random.default_rng(int(seed) + 321)
    selected: list[tuple[Any, int]] = []
    tried = 0
    dim = float(max(ops.dim, 1))
    for block in range(num_calib):
        start = int(block_starts[block])
        end = int(block_starts[block + 1])
        width = end - start
        if width <= 0:
            raise RuntimeError(f"empty calibration block {block}")
        attempts = 0
        while True:
            attempts += 1
            if attempts > _MAX_BLOCK_ATTEMPTS:
                raise RuntimeError(
                    f"could not select a calibration variant from block {block}; "
                    "every candidate exceeded the GRAMMAR screening threshold"
                )
            flat = start + int(rng.integers(width))
            chrom, positions, item, offset_in_chrom = locate(flat)
            position = int(positions[offset_in_chrom])
            tried += 1
            score = float(grammar_scores[chrom][position])
            std_norm2 = float(item.std_projected_norm2[position])
            retro = score * score / all_hinv_norm2 / std_norm2 * dim
            if retro < float(screen_threshold):
                selected.append((chrom, int(item.local_idx[position])))
                break
    return selected, tried


def calibrate_association(
    result: FitResult,
    y: np.ndarray | None = None,
    *,
    count: int = DEFAULT_CALIBRATION_VARIANTS,
    seed: int = 0,
    screen_threshold: float = DEFAULT_SCREEN_THRESHOLD,
    cg_tol: float = 1e-9,
    stats: dict[Any, TestVariantStats] | None = None,
) -> CalibrationResult:
    """Calibrate the LMM statistic against the fitted evolutionary covariance.

    For every selected variant ``j`` on its own chromosome ``c`` the
    retrospective and prospective statistics are

    ``retro = (N - C) * (x_j^T u_c)^2 / (||u_c||^2 * ||x_j||^2)``

    ``pro   = (N - C) * (x_j^T u_c)^2 / (x_j^T H_c^-1 x_j) / (y^T u_c)``

    with ``u_c = H_c^-1 P_C y`` and ``H_c`` the LOCO evolutionary shape matrix.
    Both are invariant to ``sigma_b2``, so the factor is a pure shape
    correction; ``sigma_b2`` enters only through :attr:`inverse_scale`.
    """

    ops = _require_ops(result)
    variant_stats = _stats_by_chrom(ops) if stats is None else stats
    phenotype = _phenotype(result, y)
    coordinates = result.prior.coordinates(result.delta)
    selected, tried = select_calibration_variants(
        result,
        count=count,
        seed=seed,
        screen_threshold=screen_threshold,
        stats=variant_stats,
    )

    chroms = list(ops.chroms)
    residuals: dict[Any, np.ndarray] = {}
    for chrom in chroms:
        residuals[chrom] = ops.solve_ph(
            phenotype, coordinates, exclude_chrom=chrom, tol=cg_tol
        )
    residual_norm2 = {chrom: float(value @ value) for chrom, value in residuals.items()}
    quadratic = {
        chrom: float(ops.project(phenotype) @ value) for chrom, value in residuals.items()
    }

    prospective: list[float] = []
    retrospective: list[float] = []
    dim = float(max(ops.dim, 1))
    for chrom, local_idx in selected:
        column = ops.test_column(chrom, local_idx)
        solved = ops.solve_ph(column, coordinates, exclude_chrom=chrom, tol=cg_tol)
        score = float(column @ residuals[chrom])
        column_norm2 = float(column @ column)
        if residual_norm2[chrom] <= 0.0 or quadratic[chrom] <= 0.0:
            raise RuntimeError(f"invalid LOCO H^-1 y moments for chromosome {chrom!r}")
        if column_norm2 <= 0.0:
            raise RuntimeError(
                f"calibration variant {local_idx} has nonpositive projected norm"
            )
        retro = dim * score * score / (residual_norm2[chrom] * column_norm2)
        denominator = float(column @ solved)
        if denominator <= 0.0:
            raise RuntimeError(
                f"calibration variant {local_idx} has nonpositive prospective denominator"
            )
        pro = dim * score * score / denominator / quadratic[chrom]
        if retro <= 0.0 or pro <= 0.0:
            raise RuntimeError(
                f"calibration variant {local_idx} has a nonpositive statistic"
            )
        prospective.append(pro)
        retrospective.append(retro)

    pro_arr = np.asarray(prospective, dtype=np.float64)
    retro_arr = np.asarray(retrospective, dtype=np.float64)
    total_pro = float(pro_arr.sum())
    total_retro = float(retro_arr.sum())
    if total_pro <= 0.0 or total_retro <= 0.0:
        raise RuntimeError("calibration failed: a statistic sum is nonpositive")
    factor = total_pro / total_retro
    jackknife = (total_pro - pro_arr) / (total_retro - retro_arr)
    jack_count = jackknife.size
    jack_sum = float(jackknife.sum())
    jack_sum2 = float(np.sum(jackknife * jackknife))
    std = math.sqrt(
        max(
            0.0,
            (jack_sum2 - jack_sum * jack_sum / jack_count)
            * (jack_count - 1)
            / jack_count,
        )
    )
    ratio_of_medians = float(np.median(pro_arr) / np.median(retro_arr))
    median_of_ratios = float(np.median(pro_arr / retro_arr))
    if std > CALIBRATION_STD_LIMIT:
        factor = ratio_of_medians
    if not np.isfinite(factor) or factor <= 0.0:
        raise RuntimeError(f"calibration factor is nonpositive: {factor}")

    inverse_scale = {
        chrom: _inverse_scale(residual_norm2[chrom], dim, factor, result.sigma_b2, chrom)
        for chrom in chroms
    }
    return CalibrationResult(
        factor=float(factor),
        std=float(std),
        ratio_of_medians=ratio_of_medians,
        median_of_ratios=median_of_ratios,
        selected=tuple(selected),
        tried=int(tried),
        prospective=pro_arr,
        retrospective=retro_arr,
        inverse_scale=inverse_scale,
        residuals=residuals,
        quadratic_form=quadratic,
        screen_threshold=float(screen_threshold),
        seed=int(seed),
    )


def _inverse_scale(
    residual_norm2: float, dim: float, factor: float, sigma_b2: float, chrom: Any
) -> float:
    if residual_norm2 <= 0.0:
        raise RuntimeError(f"LOCO H^-1 y has nonpositive norm for chromosome {chrom!r}")
    if sigma_b2 <= 0.0:
        raise RuntimeError("sigma_b2 must be positive to scale association statistics")
    return 1.0 / (math.sqrt(dim / residual_norm2 * factor) * float(sigma_b2))


def uncalibrated_scale(result: FitResult, residual_norm2: float) -> float:
    """Return the ``factor = 1`` inverse scale used when calibration is skipped."""

    ops = _require_ops(result)
    return _inverse_scale(
        float(residual_norm2), float(max(ops.dim, 1)), 1.0, result.sigma_b2, None
    )
