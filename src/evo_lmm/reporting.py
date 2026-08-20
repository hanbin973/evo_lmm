"""Estimand adapters and uncertainty/reporting helpers for MC3."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import chi2

from .multicomponent import MultiComponentFit, MultiComponentOps, MultiComponentPrior, fit_multicomponent_reml
from .priors import SimplifiedPrior


@dataclass(frozen=True)
class HeritabilityEstimates:
    rare_effect: float
    evolutionary: float
    rare_genetic_variance: float
    evolutionary_genetic_variance: float


@dataclass(frozen=True)
class ProfileLikelihood:
    """One-dimensional profile likelihood on a scientific parameter scale."""

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


@dataclass(frozen=True)
class FitReport:
    """Integrated estimand report with covariance-derived uncertainty."""

    heritability: HeritabilityEstimates
    heritability_se: float
    component_standard_errors: dict[str, float]
    maf_decomposition: dict[Any, np.ndarray] | None
    tau_profiles: dict[Any, ProfileLikelihood] | None = None


def _ratio_prior(fit: MultiComponentFit, components: Sequence[SimplifiedPrior]) -> MultiComponentPrior:
    """Convert scientific-scale components to the profiled objective's scale.

    :func:`profiled_reml_objective` profiles the residual scale, so its prior
    argument is the ratio ``sigma_b2_c / sigma_e2``.  A fit reports
    ``sigma_b2_c`` on the scientific scale, so feeding a fit's prior back
    unconverted evaluates the objective at the wrong point whenever
    ``sigma_e2 != 1`` -- and takes grid values on the wrong scale with it.
    """
    scale = max(float(fit.sigma_e2), np.finfo(float).tiny)
    return MultiComponentPrior(
        fit.ops.labels,
        tuple(SimplifiedPrior(component.sigma_b2 / scale, component.tau)
              for component in components),
    )


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


def fit_report(
    fit: MultiComponentFit, *, maf_bins: Sequence[float] | None = None
) -> FitReport:
    """Convert a fit into both estimands plus delta-method h² uncertainty."""
    estimates = heritability_conventions(fit.ops, fit.prior, fit.sigma_e2)
    covariance = fit.ai_covariance
    if covariance is None:
        h2_se = float("nan")
    else:
        coordinates = fit.prior.coordinates

        def h2_at(value: np.ndarray) -> float:
            prior = MultiComponentPrior.from_coordinates(fit.ops.labels, value)
            return heritability_conventions(fit.ops, prior, fit.sigma_e2).evolutionary

        h2_se = delta_method_se(h2_at, coordinates, covariance)
    decomposition = None if maf_bins is None else genic_variance_by_maf(
        fit.ops, fit.prior, maf_bins
    )
    component_errors: dict[str, float] = {}
    for index, label in enumerate(fit.ops.labels):
        if fit.standard_errors:
            component_errors[f"sigma_b2[{label}]"] = fit.prior.components[index].sigma_b2 * fit.standard_errors.get(
                f"log_sigma_b2[{label}]", float("nan")
            )
            component_errors[f"tau[{label}]"] = fit.prior.components[index].tau * fit.standard_errors.get(
                f"log_tau[{label}]", float("nan")
            )
    return FitReport(estimates, h2_se, component_errors, decomposition)


def fit_tau_profiles(
    fit: MultiComponentFit, tau_grids: dict[Any, Sequence[float]]
) -> dict[Any, ProfileLikelihood]:
    """Profile each category's ``tau_c`` with other fitted parameters fixed."""
    from .multicomponent import profiled_reml_objective

    profiles: dict[Any, ProfileLikelihood] = {}
    for label, grid in tau_grids.items():
        if label not in fit.ops.labels:
            raise KeyError(label)
        index = fit.ops.labels.index(label)

        def objective(tau: float, index: int = index) -> float:
            components = list(fit.prior.components)
            components[index] = SimplifiedPrior(components[index].sigma_b2, tau)
            if fit.phenotype is None:
                raise ValueError("fit does not retain a phenotype for profiling")
            return profiled_reml_objective(
                fit.ops, fit.phenotype, _ratio_prior(fit, components)
            )[0]

        profiles[label] = profile_tau(grid, objective)
    return profiles


def fit_parameter_profiles(
    fit: MultiComponentFit, parameter_grids: dict[str, Sequence[float]]
) -> dict[str, ProfileLikelihood]:
    """Profile scientific-scale ``sigma_b2[label]`` and ``tau[label]`` grids."""
    from .multicomponent import profiled_reml_objective

    profiles: dict[str, ProfileLikelihood] = {}
    for name, grid in parameter_grids.items():
        try:
            parameter, raw_label = name.split("[", 1)
            label = raw_label[:-1]
            index = fit.ops.labels.index(label)
        except (ValueError, IndexError):
            raise KeyError(name) from None
        if parameter not in ("sigma_b2", "tau"):
            raise KeyError(name)

        def objective(value: float, index: int = index, parameter: str = parameter) -> float:
            if value <= 0.0 and parameter == "sigma_b2":
                return float("inf")
            components = list(fit.prior.components)
            current = components[index]
            components[index] = SimplifiedPrior(
                value if parameter == "sigma_b2" else current.sigma_b2,
                value if parameter == "tau" else current.tau,
            )
            if fit.phenotype is None:
                raise ValueError("fit does not retain a phenotype for profiling")
            # Grid values are on the scientific scale, like the reported fit.
            return profiled_reml_objective(
                fit.ops, fit.phenotype, _ratio_prior(fit, components)
            )[0]

        profiles[name] = profile_tau(grid, objective)
    return profiles


def fit_genes(
    genes: dict[Any, MultiComponentOps],
    phenotype: np.ndarray,
    pooled_tau: dict[Any, float],
    *,
    max_iter: int = 100,
    trace_method: str = "hutchinson",
    trace_probes: int = 12,
) -> dict[Any, GeneComponentReport]:
    """Fit per-gene scales with the category shapes pooled across genes.

    ``pooled_tau`` is held fixed for every gene (``fit_tau=False``); it is a
    pooled estimate, not a per-gene starting point.  Only the ``|c|`` scale
    coordinates are searched, so the reported ``pooled_tau`` is the shape the
    per-gene scales were actually conditioned on.
    """
    reports: dict[Any, GeneComponentReport] = {}
    for gene, ops in genes.items():
        if set(pooled_tau) != set(ops.labels):
            raise ValueError("pooled_tau categories must match every gene partition")
        initial = MultiComponentPrior(
            ops.labels,
            tuple(SimplifiedPrior(1.0, pooled_tau[label])
                  for label in ops.labels),
        )
        fit = fit_multicomponent_reml(
            ops, phenotype, initial=initial, max_iter=max_iter,
            trace_method=trace_method, trace_probes=trace_probes, fit_tau=False,
        )
        fitted_tau = {label: component.tau
                      for label, component in zip(fit.prior.labels, fit.prior.components)}
        if any(fitted_tau[label] != pooled_tau[label] for label in ops.labels):
            raise RuntimeError("pooled shapes moved during a pooled-shape fit")
        reports[gene] = gene_component_report(
            gene, fitted_tau,
            {label: component.sigma_b2 for label, component in zip(fit.prior.labels, fit.prior.components)},
        )
    return reports


def heritability_conventions(
    ops: MultiComponentOps, prior: MultiComponentPrior, sigma_e2: float
) -> HeritabilityEstimates:
    """Return RareEffect's uncentered-``n`` and evo-lmm's projected-``d`` h²."""
    raw_trace = 0.0
    for label, component in zip(prior.labels, prior.components):
        op = ops.components[label]
        for chrom in op._chromosomes:
            raw_norm2 = chrom.data.raw_norm2
            if raw_norm2 is None:
                identity = np.eye(op.n, dtype=np.float64)
                raw_norm2 = np.sum(op._raw_matmat(chrom, identity) ** 2, axis=0)
            raw_trace += component.sigma_b2 * float(np.dot(
                np.asarray(raw_norm2), component.weights(chrom.data.frequencies)
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
            op._ensure_projected_norms()
            contribution = component.sigma_b2 * component.weights(chrom.data.frequencies) * chrom.projected_norm2
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
