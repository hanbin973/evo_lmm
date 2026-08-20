"""Estimand adapters and uncertainty/reporting helpers for MC3."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import chi2

from .multicomponent import MultiComponentOps, MultiComponentPrior


@dataclass(frozen=True)
class HeritabilityEstimates:
    rare_effect: float
    evolutionary: float
    rare_genetic_variance: float
    evolutionary_genetic_variance: float


@dataclass(frozen=True)
class ProfileLikelihood:
    """One-dimensional profile likelihood for a non-negative ``tau_c``."""

    tau: np.ndarray
    objective: np.ndarray
    lower: float
    upper: float


@dataclass(frozen=True)
class GeneComponentReport:
    """Gene-level empirical-Bayes report with pooled category shapes."""

    gene: Any
    pooled_tau: dict[Any, float]
    sigma_b2_by_category: dict[Any, float]


def profile_tau(
    tau_values: Sequence[float], objective: Callable[[float], float], *, confidence: float = 0.95
) -> ProfileLikelihood:
    """Evaluate a tau profile and use the likelihood-ratio cutoff for bounds."""
    tau = np.asarray(tau_values, dtype=np.float64)
    if tau.ndim != 1 or np.any(tau < 0) or tau.size == 0:
        raise ValueError("tau_values must be a non-empty non-negative vector")
    values = np.asarray([objective(float(value)) for value in tau], dtype=np.float64)
    minimum = float(np.nanmin(values))
    cutoff = 0.5 * float(chi2.ppf(confidence, 1))
    accepted = tau[values <= minimum + cutoff]
    return ProfileLikelihood(tau, values, float(accepted.min()) if accepted.size else float(tau.min()),
                             float(accepted.max()) if accepted.size else float(tau.max()))


def gene_component_report(
    gene: Any, pooled_tau: dict[Any, float], sigma_b2_by_category: dict[Any, float]
) -> GeneComponentReport:
    """Construct the pooled-shape/per-gene-scale reporting unit."""
    if set(pooled_tau) != set(sigma_b2_by_category):
        raise ValueError("pooled tau and per-gene scales must cover the same categories")
    return GeneComponentReport(gene, dict(pooled_tau), dict(sigma_b2_by_category))


def heritability_conventions(
    ops: MultiComponentOps, prior: MultiComponentPrior, sigma_e2: float
) -> HeritabilityEstimates:
    """Return RareEffect's uncentered-``n`` and evo-lmm's projected-``d`` h²."""
    raw_trace = 0.0
    for label, component in zip(prior.labels, prior.components):
        op = ops.components[label]
        for chrom in op._chromosomes:
            raw_trace += component.sigma_b2 * float(np.sum(
                (chrom.dense * chrom.dense) * component.weights(chrom.data.frequencies)[None, :]
            ))
    evolutionary = ops.kernel_trace(prior)
    n = float(ops.n)
    d = float(ops.dim)
    return HeritabilityEstimates(
        raw_trace / max(raw_trace + n * sigma_e2, np.finfo(float).tiny),
        evolutionary / max(evolutionary + d * sigma_e2, np.finfo(float).tiny),
        raw_trace / n,
        evolutionary / d,
    )


def genic_variance_by_maf(
    ops: MultiComponentOps,
    prior: MultiComponentPrior,
    bins: Sequence[float],
) -> dict[Any, np.ndarray]:
    """Decompose projected genic variance by MAF bin and category."""
    edges = np.asarray(bins, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("bins must be strictly increasing and contain at least two edges")
    result: dict[Any, np.ndarray] = {}
    for label, component in zip(prior.labels, prior.components):
        op = ops.components[label]
        values = np.zeros(edges.size - 1, dtype=np.float64)
        for chrom in op._chromosomes:
            maf = np.minimum(chrom.data.frequencies, 1.0 - chrom.data.frequencies)
            centered = op.project(chrom.dense)
            contribution = component.sigma_b2 * component.weights(chrom.data.frequencies) * np.sum(centered * centered, axis=0)
            bucket = np.searchsorted(edges, maf, side="right") - 1
            for index, value in zip(bucket, contribution):
                if 0 <= index < values.size:
                    values[index] += value
        result[label] = values
    return result


def delta_method_se(function: Callable[[np.ndarray], float], estimate: np.ndarray, covariance: np.ndarray, step: float = 1e-5) -> float:
    """Finite-difference delta-method standard error from an AI covariance."""
    point = np.asarray(estimate, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    gradient = np.empty(point.size)
    for index in range(point.size):
        plus, minus = point.copy(), point.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (function(plus) - function(minus)) / (2.0 * step)
    return float(np.sqrt(max(gradient @ cov @ gradient, 0.0)))


def boundary_lrt_pvalue(statistic: float, added_boundaries: int = 1) -> float:
    """Mixture-null p-value for independent non-negative boundary coordinates."""
    if statistic < 0 or added_boundaries < 1:
        raise ValueError("statistic must be non-negative and boundaries positive")
    weights = np.asarray([comb(added_boundaries, j) / 2.0 ** added_boundaries
                          for j in range(added_boundaries + 1)])
    return float(sum(weight * chi2.sf(statistic, j) if j else weight * (statistic <= 0)
                     for j, weight in enumerate(weights)))
