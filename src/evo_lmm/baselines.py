"""Explicit RareEffect-style baseline utilities.

These functions are deliberately separate from the evolutionary model.  In
particular, the flat prior is never silently substituted for evolutionary
frequency weighting.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .multicomponent import MultiComponentOps, MultiComponentPrior
from .priors import SimplifiedPrior


def flat_prior(labels: tuple[Any, ...], scales: Mapping[Any, float] | None = None) -> MultiComponentPrior:
    """Return named M0: a flat per-category prior with ``tau_c = 0``."""
    scales = {} if scales is None else scales
    return MultiComponentPrior(
        tuple(labels),
        tuple(SimplifiedPrior(float(scales.get(label, 1.0)), 0.0) for label in labels),
    )


@dataclass(frozen=True)
class CollapsedVariants:
    """Result of optional MAC-threshold burden construction."""

    genotypes: dict[Any, np.ndarray]
    frequencies: dict[Any, np.ndarray]
    source_indices: dict[Any, tuple[tuple[int, ...], ...]]


def collapse_mac(
    genotypes: Mapping[Any, np.ndarray],
    *,
    ploidy: int = 2,
    mac_threshold: float = 10.0,
) -> CollapsedVariants:
    """Collapse variants with MAC below ``mac_threshold`` per category.

    Filtering and collapsing occur before frequency recomputation.  The
    collapsed column is a dosage burden and its frequency is its sample mean
    divided by ``ploidy * n``.
    """
    if ploidy <= 0 or mac_threshold < 0:
        raise ValueError("ploidy must be positive and mac_threshold non-negative")
    result: dict[Any, np.ndarray] = {}
    frequencies: dict[Any, np.ndarray] = {}
    sources: dict[Any, tuple[tuple[int, ...], ...]] = {}
    for label, source in genotypes.items():
        values = np.asarray(source, dtype=np.float64)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError("genotypes must be finite two-dimensional arrays")
        mac = values.sum(axis=0)
        keep = np.flatnonzero(mac >= float(mac_threshold))
        rare = np.flatnonzero(mac < float(mac_threshold))
        columns = [values[:, index] for index in keep]
        mapping = [(int(index),) for index in keep]
        if rare.size:
            columns.append(values[:, rare].sum(axis=1))
            mapping.append(tuple(int(index) for index in rare))
        matrix = np.column_stack(columns) if columns else np.empty((values.shape[0], 0))
        result[label] = matrix
        frequencies[label] = matrix.mean(axis=0) / float(ploidy) if matrix.shape[1] else np.empty(0)
        sources[label] = tuple(mapping)
    return CollapsedVariants(result, frequencies, sources)


@dataclass(frozen=True)
class MoMResult:
    """Joint method-of-moments estimates and the negative-estimate audit."""

    component_scales: np.ndarray
    residual_variance: float
    raw_component_scales: np.ndarray
    truncated: np.ndarray
    system: np.ndarray


def joint_mom_initialization(
    ops: MultiComponentOps,
    y: np.ndarray,
    prior: MultiComponentPrior,
) -> MoMResult:
    """Solve the ``(|c|+1)`` projected Haseman--Elston moment system.

    The returned ``raw_component_scales`` are never truncated.  The
    ``component_scales`` field applies the RareEffect boundary rule for an
    explicitly requested baseline comparison.
    """
    values = np.asarray(y, dtype=np.float64)
    if values.shape != (ops.n,) or not np.all(np.isfinite(values)):
        raise ValueError("y must be a finite vector with shape (n,)")
    kernels = ops.component_kernels(prior)
    labels = prior.labels
    projected = ops.project(values)
    traces = np.asarray([np.trace(kernel) for kernel in kernels.values()])
    matrix = np.empty((len(labels) + 1, len(labels) + 1), dtype=np.float64)
    matrix[:-1, :-1] = [[np.trace(left @ right) for right in kernels.values()] for left in kernels.values()]
    matrix[:-1, -1] = traces
    matrix[-1, :-1] = traces
    matrix[-1, -1] = ops.dim
    rhs = np.asarray([projected @ kernel @ projected for kernel in kernels.values()] + [projected @ projected])
    try:
        raw = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        raw = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    truncated = raw[:-1] < 0.0
    scales = np.maximum(raw[:-1], 0.0)
    residual = max(float(raw[-1]), np.finfo(float).tiny)
    return MoMResult(scales, residual, raw[:-1].copy(), truncated, matrix)


@dataclass(frozen=True)
class RareEffectBaselineResult:
    """Marginal ML estimates and the faithfully reproduced MoM-ratio rule."""

    marginal_scales: np.ndarray
    marginal_mom_scales: np.ndarray
    joint_mom_scales: np.ndarray
    adjusted_scales: np.ndarray
    negative_mom_fallback: np.ndarray


def fit_rare_effect_baseline(
    ops: MultiComponentOps, y: np.ndarray, *, max_iter: int = 100
) -> RareEffectBaselineResult:
    """Fit the named flat baseline marginally and apply the MoM-ratio rule.

    Each category is fitted independently with the existing exact REML engine
    at the ``tau=0`` boundary.  The joint adjustment is then computed from the
    partitioned moment system; no evolutionary weighting is introduced.
    """
    from .reml import fit_reml

    marginal = []
    marginal_mom = []
    flat = flat_prior(ops.labels)
    for label in ops.labels:
        fit = fit_reml(ops.components[label], y,
                       initial=SimplifiedPrior(1.0, 0.0), exact=True,
                       max_iter=max_iter)
        marginal.append(fit.sigma_b2)
        component = ops.components[label]
        kernel = component.dense_kernel(SimplifiedPrior(1.0, 0.0))
        projected = component.project(np.asarray(y, dtype=np.float64))
        trace = float(np.trace(kernel))
        system = np.array([[float(np.trace(kernel @ kernel)), trace], [trace, component.dim]])
        rhs = np.array([projected @ kernel @ projected, projected @ projected])
        moment = np.linalg.lstsq(system, rhs, rcond=None)[0]
        marginal_mom.append(moment[0])
    joint = joint_mom_initialization(ops, y, flat)
    return rare_effect_mom_ratio(np.asarray(marginal), np.asarray(marginal_mom), joint.raw_component_scales)


def rare_effect_mom_ratio(
    marginal_scales: np.ndarray,
    marginal_mom_scales: np.ndarray,
    joint_mom_scales: np.ndarray,
) -> RareEffectBaselineResult:
    """Apply RareEffect's marginal-ML × joint-MoM/marginal-MoM adjustment.

    A non-positive marginal or joint MoM estimate triggers the published
    unadjusted-marginal fallback rather than silently truncating the result.
    """
    marginal = np.asarray(marginal_scales, dtype=np.float64)
    marginal_mom = np.asarray(marginal_mom_scales, dtype=np.float64)
    joint_mom = np.asarray(joint_mom_scales, dtype=np.float64)
    if marginal.ndim != 1 or marginal.shape != marginal_mom.shape or marginal.shape != joint_mom.shape:
        raise ValueError("all scale vectors must have the same one-dimensional shape")
    fallback = (marginal_mom <= 0.0) | (joint_mom <= 0.0)
    adjusted = marginal.copy()
    valid = ~fallback
    adjusted[valid] = marginal[valid] * joint_mom[valid] / marginal_mom[valid]
    return RareEffectBaselineResult(marginal, marginal_mom, joint_mom, adjusted, fallback)
