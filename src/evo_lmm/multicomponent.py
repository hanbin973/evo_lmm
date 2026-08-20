"""Annotation-partitioned simplified evolutionary random-effects models.

This module deliberately has no full-prior or coupling coordinate.  A component
``tau_c`` is the composite ``rho_ab,c**2 * sigma_a,c**2 / (2*k_c*W_S)``;
``rho_ab`` is fixed to one on this path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .operators import EvolutionaryLmmOps
from .priors import SimplifiedPrior
from .reml import _dense_projection


@dataclass(frozen=True)
class MultiComponentPrior:
    """One simplified prior per annotation category.

    ``coordinates`` are ordered ``(log_sigma_b2_c, log_tau_c)`` pairs.  A
    ``tau_c`` of zero is represented by ``-inf`` in transformed coordinates.
    """

    labels: tuple[Any, ...]
    components: tuple[SimplifiedPrior, ...]

    def __post_init__(self) -> None:
        if len(self.labels) == 0 or len(self.labels) != len(self.components):
            raise ValueError("labels and components must be non-empty and have equal length")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("component labels must be unique")

    @classmethod
    def from_parameters(
        cls, parameters: Mapping[Any, tuple[float, float]]
    ) -> "MultiComponentPrior":
        labels = tuple(parameters)
        return cls(labels, tuple(SimplifiedPrior(*parameters[label]) for label in labels))

    @property
    def sigma_b2(self) -> np.ndarray:
        return np.asarray([p.sigma_b2 for p in self.components], dtype=np.float64)

    @property
    def tau(self) -> np.ndarray:
        return np.asarray([p.tau for p in self.components], dtype=np.float64)

    @property
    def coordinates(self) -> np.ndarray:
        values = []
        for component in self.components:
            values.extend((np.log(component.sigma_b2),
                           np.log(component.tau) if component.tau > 0 else -np.inf))
        return np.asarray(values, dtype=np.float64)

    @classmethod
    def from_coordinates(cls, labels: Sequence[Any], coordinates: np.ndarray) -> "MultiComponentPrior":
        labels_tuple = tuple(labels)
        values = np.asarray(coordinates, dtype=np.float64)
        if values.shape != (2 * len(labels_tuple),):
            raise ValueError("coordinates must contain (log_sigma_b2, log_tau) per component")
        components = []
        for index in range(len(labels_tuple)):
            sigma = float(np.exp(values[2 * index]))
            tau = float(np.exp(values[2 * index + 1])) if np.isfinite(values[2 * index + 1]) else 0.0
            components.append(SimplifiedPrior(sigma, tau))
        return cls(labels_tuple, tuple(components))


class MultiComponentOps:
    """Sum of independently partitioned projected raw-dosage component operators."""

    def __init__(self, components: Mapping[Any, EvolutionaryLmmOps]):
        if not components:
            raise ValueError("at least one annotation component is required")
        self.components = dict(components)
        first = next(iter(self.components.values()))
        if any(op.n != first.n or op.rank != first.rank for op in self.components.values()):
            raise ValueError("all component operators must have the same individuals and covariate rank")
        self.labels = tuple(self.components)
        self.n = first.n
        self.rank = first.rank
        self.dim = first.dim
        self.basis = first.basis

    @property
    def n_components(self) -> int:
        return len(self.components)

    def project(self, values: np.ndarray) -> np.ndarray:
        """Project vectors or matrices off the shared covariate basis."""
        array = np.asarray(values, dtype=np.float64)
        if array.ndim not in (1, 2) or array.shape[0] != self.n:
            raise ValueError("values must have n rows")
        return array - self.basis @ (self.basis.T @ array)

    @classmethod
    def from_dense(
        cls,
        genotypes: Mapping[Any, np.ndarray],
        frequencies: Mapping[Any, np.ndarray],
        covariates: np.ndarray | None = None,
    ) -> "MultiComponentOps":
        return cls({label: EvolutionaryLmmOps(matrix, frequencies[label], covariates)
                    for label, matrix in genotypes.items()})

    def component_kernels(self, prior: MultiComponentPrior) -> dict[Any, np.ndarray]:
        if prior.labels != self.labels:
            raise ValueError("prior labels do not match the partition")
        return {label: component.sigma_b2 * self._component_kernel(label, component)
                for label, component in zip(prior.labels, prior.components)}

    def _component_kernel(self, label: Any, prior: SimplifiedPrior) -> np.ndarray:
        """Materialize one component only when an exact small fit requests it."""
        op = self.components[label]
        if all(chrom.dense is not None for chrom in op._chromosomes):
            return op.dense_kernel(prior)
        identity = np.eye(op.n, dtype=np.float64)
        return op.apply_k_matmat(identity, prior)

    def dense_kernel(self, prior: MultiComponentPrior) -> np.ndarray:
        return sum(self.component_kernels(prior).values(), start=np.zeros((self.n, self.n)))

    def kernel_trace(self, prior: MultiComponentPrior) -> float:
        """Return ``tr(P_C K P_C)`` without materialising a dense kernel."""
        if prior.labels != self.labels:
            raise ValueError("prior labels do not match the partition")
        return float(sum(
            component.sigma_b2 * self.components[label].kernel_trace(component)
            for label, component in zip(prior.labels, prior.components)
        ))

    def apply_k(self, values: np.ndarray, prior: MultiComponentPrior) -> np.ndarray:
        """Apply the summed partitioned kernel to one or many vectors."""
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            return self.apply_k(matrix[:, None], prior)[:, 0]
        if matrix.ndim != 2 or matrix.shape[0] != self.n:
            raise ValueError("values must have shape (n,) or (n, k)")
        return sum(
            component.sigma_b2 * self.components[label].apply_k_matmat(matrix, component)
            for label, component in zip(prior.labels, prior.components)
        )

    def derivative_kernels(self, prior: MultiComponentPrior) -> dict[str, np.ndarray]:
        """Return analytic derivatives in the same coordinate order as the prior."""
        result: dict[str, np.ndarray] = {}
        for index, (label, component) in enumerate(zip(self.labels, prior.components)):
            op = self.components[label]
            if any(chrom.dense is None for chrom in op._chromosomes):
                identity = np.eye(op.n, dtype=np.float64)
                result[f"log_sigma_b2[{label}]"] = component.sigma_b2 * op.apply_k_matmat(identity, component)
                result[f"log_tau[{label}]"] = component.sigma_b2 * op.apply_dh_matmat(identity, component, "log_tau")
                continue
            result[f"log_sigma_b2[{label}]"] = component.sigma_b2 * op.dense_kernel(component)
            derivative = component.weight_derivatives(op.frequencies)["log_tau"]
            matrices = []
            offset = 0
            for chrom in op._chromosomes:
                centered = op.project(chrom.dense)
                count = chrom.n_variants
                matrices.append((centered * derivative[offset:offset + count]) @ centered.T)
                offset += count
            result[f"log_tau[{label}]"] = sum(matrices, start=np.zeros((self.n, self.n)))
        return result

    def apply_dh_matmat(self, values: np.ndarray, prior: MultiComponentPrior, parameter: str) -> np.ndarray:
        """Apply a component derivative to batched vectors (dense/GRGL path)."""
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != self.n:
            raise ValueError("values must have shape (n, k)")
        label = parameter.split("[", 1)[-1][:-1]
        if label not in self.components:
            raise KeyError(parameter)
        index = self.labels.index(label)
        component = prior.components[index]
        op = self.components[label]
        if parameter.startswith("log_sigma"):
            return component.sigma_b2 * op.apply_k_matmat(matrix, component)
        return component.sigma_b2 * op.apply_dh_matmat(matrix, component, "log_tau")


@dataclass(frozen=True)
class MultiComponentFit:
    """Profiled REML result for the partitioned simplified model."""

    prior: MultiComponentPrior
    sigma_e2: float
    h2: float
    objective: float
    converged: bool
    ops: MultiComponentOps
    ai_covariance: np.ndarray | None = None
    standard_errors: dict[str, float] | None = None


def fit_multicomponent_reml(
    ops: MultiComponentOps,
    y: np.ndarray,
    *,
    initial: MultiComponentPrior | np.ndarray | None = None,
    max_iter: int = 200,
) -> MultiComponentFit:
    """Fit by exact profiled REML.

    The residual scale ``sigma_e2`` is profiled; component scales are searched
    as ratios to that scale in log coordinates.  The objective is REML, and
    the reported component ``sigma_b2`` values are returned on the scientific
    scale after profiling.
    """
    values = np.asarray(y, dtype=np.float64)
    if values.shape != (ops.n,) or not np.all(np.isfinite(values)):
        raise ValueError("y must be a finite vector with one entry per individual")
    if initial is None:
        prior = MultiComponentPrior(tuple(ops.labels), tuple(SimplifiedPrior(1.0, 0.1) for _ in ops.labels))
    elif isinstance(initial, MultiComponentPrior):
        prior = initial
    else:
        prior = MultiComponentPrior.from_coordinates(ops.labels, np.asarray(initial))
    if prior.labels != ops.labels:
        raise ValueError("initial prior labels do not match the partition")

    # The one-category model is exactly the existing single-component model;
    # delegate it so its scale/profile/numerical conventions remain identical.
    if ops.n_components == 1:
        from .reml import fit_reml
        label = ops.labels[0]
        single = fit_reml(ops.components[label], values, initial=prior.components[0], exact=True,
                          max_iter=max_iter)
        fitted = MultiComponentPrior((label,), (single.prior,))
        return MultiComponentFit(fitted, single.sigma_e2, single.h2, single.diagnostics.objective,
                                 single.diagnostics.converged, ops)
    coordinates = prior.coordinates.copy()
    finite = np.isfinite(coordinates)
    coordinates[~finite] = np.log(np.finfo(float).tiny)

    def evaluate(theta: np.ndarray) -> tuple[float, float, MultiComponentPrior]:
        current = MultiComponentPrior.from_coordinates(ops.labels, theta)
        shape = np.eye(ops.n) + ops.dense_kernel(current)
        ph, inv_shape, logdet = _dense_projection(shape, ops.basis)
        q = float(values @ ph @ values)
        if q <= 0 or not np.isfinite(q):
            return np.inf, np.nan, current
        d = ops.dim
        sigma_e2 = q / d
        fixed = ops.basis.T @ inv_shape @ ops.basis
        objective = 0.5 * (logdet + np.linalg.slogdet(fixed)[1] + d * np.log(q / d))
        scaled = tuple(SimplifiedPrior(sigma_e2 * p.sigma_b2, p.tau) for p in current.components)
        return float(objective), float(sigma_e2), MultiComponentPrior(current.labels, scaled)

    result = minimize(lambda theta: evaluate(theta)[0], coordinates, method="L-BFGS-B",
                      bounds=[(-30.0, 30.0), (-30.0, 30.0)] * ops.n_components,
                      options={"maxiter": int(max_iter), "ftol": 1e-12})
    objective, sigma_e2, fitted = evaluate(result.x)
    kernel = ops.dense_kernel(fitted)
    genetic = float(np.trace(kernel))
    h2 = genetic / max(genetic + ops.dim * sigma_e2, np.finfo(float).tiny)
    covariance = None
    standard_errors: dict[str, float] = {}
    if hasattr(result, "hess_inv"):
        try:
            covariance = np.asarray(result.hess_inv.todense(), dtype=np.float64)
            diagonal = np.sqrt(np.maximum(np.diag(covariance), 0.0))
            for index, label in enumerate(ops.labels):
                standard_errors[f"log_sigma_b2[{label}]"] = float(diagonal[2 * index])
                standard_errors[f"log_tau[{label}]"] = float(diagonal[2 * index + 1])
        except (AttributeError, ValueError):
            covariance = None
    return MultiComponentFit(fitted, sigma_e2, h2, objective, bool(result.success), ops,
                             covariance, standard_errors)
