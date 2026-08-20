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
from scipy.sparse.linalg import LinearOperator, cg

from .operators import EvolutionaryLmmOps
from .priors import SimplifiedPrior
from .reml import _dense_projection
from .trace import rademacher_probes, spherical_gaussian_probes


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

    @classmethod
    def flat(cls, labels: Sequence[Any], scales: Sequence[float] | None = None) -> "MultiComponentPrior":
        """Return the exact M0 boundary with all ``tau_c = 0``."""
        labels_tuple = tuple(labels)
        scale_values = [1.0] * len(labels_tuple) if scales is None else list(scales)
        if len(scale_values) != len(labels_tuple):
            raise ValueError("scales must match labels")
        return cls(labels_tuple, tuple(SimplifiedPrior(scale, 0.0) for scale in scale_values))

    def with_shared_tau(self, tau: float) -> "MultiComponentPrior":
        """Return M1's shared-``tau`` identifiability-crutch specification."""
        if tau < 0 or not np.isfinite(tau):
            raise ValueError("tau must be finite and non-negative")
        return MultiComponentPrior(
            self.labels, tuple(SimplifiedPrior(p.sigma_b2, tau) for p in self.components)
        )

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
    def from_operators(cls, components: Mapping[Any, EvolutionaryLmmOps]) -> "MultiComponentOps":
        """Construct a partition from already-adapted dense or GRGL operators."""
        return cls(components)

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
            result[f"log_tau[{label}]"] = component.sigma_b2 * sum(
                matrices, start=np.zeros((self.n, self.n))
            )
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

    def apply_component_derivatives_matmat(
        self, values: np.ndarray, prior: MultiComponentPrior
    ) -> dict[str, np.ndarray]:
        """Apply all component derivatives to shared batched right-hand sides.

        The projection and right-hand-side batch are shared across the returned
        derivatives; GRGL-backed component operators retain their ``matmat``
        traversal rather than falling back to one traversal per probe.
        """
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != self.n:
            raise ValueError("values must have shape (n, k)")
        return {
            name: self.apply_dh_matmat(matrix, prior, name)
            for label in self.labels
            for name in (f"log_sigma_b2[{label}]", f"log_tau[{label}]")
        }

    def apply_shape_matmat(self, values: np.ndarray, prior: MultiComponentPrior) -> np.ndarray:
        """Apply ``H = I + sum_c K_c`` to batched projected vectors."""
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != self.n:
            raise ValueError("values must have shape (n, k)")
        # H is full-rank; projection is applied to the right-hand side and in
        # the fixed-effect correction, not to the identity term itself.
        return matrix + self.apply_k(matrix, prior)


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
    trace_method: str = "hutchinson"
    trace_probes: int = 0
    cg_tol: float = float("nan")
    score_norm: float = float("nan")
    trace_standard_error: float = float("nan")
    phenotype: np.ndarray | None = None
    accepted_step: float = float("nan")
    ai_damping: float = float("nan")


def profiled_reml_objective(
    ops: MultiComponentOps, y: np.ndarray, prior: MultiComponentPrior
) -> tuple[float, float]:
    """Evaluate the exact dense profiled-REML objective for a fixed prior.

    Returns ``(objective, sigma_e2)`` for ``V = sigma_e2 * (I + K)``.  This is
    the small-dense reference used to verify the M0/M1/M2 nesting ladder.
    """
    values = np.asarray(y, dtype=np.float64)
    if values.shape != (ops.n,) or not np.all(np.isfinite(values)):
        raise ValueError("y must be a finite vector with shape (n,)")
    shape = np.eye(ops.n) + ops.dense_kernel(prior)
    ph, inv_shape, logdet = _dense_projection(shape, ops.basis)
    q = float(values @ ph @ values)
    if q <= 0.0:
        raise np.linalg.LinAlgError("profiled REML quadratic form is non-positive")
    fixed = ops.basis.T @ inv_shape @ ops.basis
    objective = 0.5 * (logdet + np.linalg.slogdet(fixed)[1] + ops.dim * np.log(q / ops.dim))
    return float(objective), float(q / ops.dim)


def fit_multicomponent_reml(
    ops: MultiComponentOps,
    y: np.ndarray,
    *,
    initial: MultiComponentPrior | np.ndarray | None = None,
    max_iter: int = 200,
    method: str = "ai",
    trace_method: str = "hutchinson",
    trace_probes: int = 12,
    seed: int = 0,
    cg_tol: float = 5e-4,
    tol: float = 1e-6,
    max_step: float = 2.0,
) -> MultiComponentFit:
    """Fit by exact profiled REML.

    The residual scale ``sigma_e2`` is profiled; component scales are searched
    as ratios to that scale in log coordinates.  The objective is REML, and
    the reported component ``sigma_b2`` values are returned on the scientific
    scale after profiling.
    """
    if method not in ("ai", "dense"):
        raise ValueError("method must be 'ai' or 'dense'")
    if trace_method not in ("hutchinson", "xtrace"):
        raise ValueError("trace_method must be 'hutchinson' or 'xtrace'")
    if trace_probes < 2 or cg_tol <= 0 or tol <= 0 or max_step <= 0:
        raise ValueError("trace_probes must be at least two; tolerances and max_step must be positive")
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

    if method == "ai" and ops.n_components > 1:
        return _fit_multicomponent_ai(
            ops, values, prior, max_iter=max_iter, trace_method=trace_method,
            trace_probes=trace_probes, seed=seed, cg_tol=cg_tol, tol=tol,
            max_step=max_step,
        )

    # The one-category model is exactly the existing single-component model;
    # delegate it so its scale/profile/numerical conventions remain identical.
    if ops.n_components == 1:
        from .reml import fit_reml
        label = ops.labels[0]
        single = fit_reml(ops.components[label], values, initial=prior.components[0], exact=True,
                          max_iter=max_iter)
        fitted = MultiComponentPrior((label,), (single.prior,))
        return MultiComponentFit(fitted, single.sigma_e2, single.h2, single.diagnostics.objective,
                                 single.diagnostics.converged, ops,
                                 trace_method=trace_method, trace_probes=trace_probes,
                                 cg_tol=cg_tol, score_norm=single.diagnostics.score_norm,
                                 phenotype=values.copy())
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
                             covariance, standard_errors, trace_method, trace_probes, cg_tol,
                             phenotype=values.copy())


def _fit_multicomponent_ai(
    ops: MultiComponentOps,
    y: np.ndarray,
    initial: MultiComponentPrior,
    *,
    max_iter: int,
    trace_method: str,
    trace_probes: int,
    seed: int,
    cg_tol: float,
    tol: float,
    max_step: float,
) -> MultiComponentFit:
    """Projected AI-REML with profiled residual scale and batched probes."""
    coords = initial.coordinates.copy()
    coords[~np.isfinite(coords)] = np.log(np.finfo(float).tiny)
    probes = (rademacher_probes(ops.n, trace_probes, seed)
              if trace_method == "hutchinson"
              else spherical_gaussian_probes(ops.n, trace_probes, seed))
    names = [name for label in ops.labels for name in
             (f"log_sigma_b2[{label}]", f"log_tau[{label}]")]
    converged = False
    covariance = None
    last_sigma_e2 = np.nan
    last_prior = initial
    last_score = np.full(len(names), np.nan)
    last_step = 0.0
    damping = 0.0

    def derivative_trace(name: str, prior: MultiComponentPrior, vectors: np.ndarray) -> float:
        def apply(values: np.ndarray) -> np.ndarray:
            derivative = ops.apply_component_derivatives_matmat(values, prior)[name]
            return projected_solve(derivative, prior)
        if trace_method == "xtrace":
            from .trace import xtrace
            return float(xtrace(apply, vectors).value)
        return float(np.mean(np.sum(vectors * apply(vectors), axis=0)))

    def projected_solve(rhs: np.ndarray, prior: MultiComponentPrior) -> np.ndarray:
        rhs_matrix = np.asarray(rhs, dtype=np.float64)
        was_vector = rhs_matrix.ndim == 1
        if was_vector:
            rhs_matrix = rhs_matrix[:, None]
        combined_rhs = np.column_stack((ops.basis, ops.project(rhs_matrix)))
        operator = LinearOperator(
            (ops.n, ops.n),
            matvec=lambda vector: ops.apply_shape_matmat(vector[:, None], prior)[:, 0],
            dtype=np.float64,
        )
        def block_cg(rhs_block: np.ndarray) -> np.ndarray:
            x = np.zeros_like(rhs_block)
            residual = rhs_block.copy()
            direction = residual.copy()
            scale = np.maximum(np.sum(residual * residual, axis=0), 1e-30)
            for _ in range(max(50, 4 * ops.n)):
                applied = np.column_stack([
                    operator.matvec(direction[:, column])
                    for column in range(direction.shape[1])
                ])
                gram = direction.T @ applied
                rr = residual.T @ residual
                alpha = np.linalg.lstsq(gram, rr, rcond=None)[0]
                x += direction @ alpha
                residual -= applied @ alpha
                if np.all(np.sum(residual * residual, axis=0) <= scale * cg_tol ** 2):
                    return x
                new_rr = residual.T @ residual
                beta = np.linalg.lstsq(rr, new_rr, rcond=None)[0]
                direction = residual + direction @ beta
            # Dependent probe columns can make block Gram systems rank-deficient;
            # retain a strict fallback for those rare cases.
            for column in range(rhs_block.shape[1]):
                solution, info = cg(operator, rhs_block[:, column], rtol=cg_tol, atol=0.0,
                                     maxiter=max(50, 4 * ops.n))
                if info != 0:
                    raise np.linalg.LinAlgError(f"multi-component CG failed with info={info}")
                x[:, column] = solution
            return x

        solutions = block_cg(combined_rhs)
        basis_solution = solutions[:, :ops.rank]
        fixed = ops.basis.T @ basis_solution
        target = solutions[:, ops.rank:]
        correction = basis_solution @ np.linalg.solve(fixed, ops.basis.T @ target)
        result = target - correction
        return result[:, 0] if was_vector else result

    for iteration in range(1, int(max_iter) + 1):
        prior = MultiComponentPrior.from_coordinates(ops.labels, coords)
        rhs = np.column_stack((y, probes))
        solved = projected_solve(rhs, prior)
        ph_y = solved[:, 0]
        ph_probes = solved[:, 1:]
        q = float(y @ ph_y)
        if q <= 0 or not np.isfinite(q):
            raise np.linalg.LinAlgError("profiled REML quadratic form is non-positive")
        derivative_values = ops.apply_component_derivatives_matmat(np.column_stack((ph_y, probes)), prior)
        score = np.empty(len(names), dtype=np.float64)
        data = np.empty(len(names), dtype=np.float64)
        solved_derivatives: list[np.ndarray] = []
        for index, name in enumerate(names):
            applied_y = derivative_values[name][:, 0]
            d_ph_y = projected_solve(applied_y, prior)
            solved_derivatives.append(d_ph_y)
            data[index] = float(ph_y @ applied_y)
            trace = derivative_trace(name, prior, probes)
            score[index] = 0.5 * (data[index] - trace)
        score_norm = float(np.linalg.norm(score, ord=np.inf))
        if score_norm <= tol and (last_step <= tol or iteration > 1):
            converged = True
            last_score = score
            last_prior = prior
            last_sigma_e2 = q / ops.dim
            break
        direct = np.empty((len(names), len(names)), dtype=np.float64)
        for i in range(len(names)):
            for j in range(len(names)):
                value = solved_derivatives[i] @ ops.apply_component_derivatives_matmat(
                    solved_derivatives[j][:, None], prior
                )[names[i]]
                direct[i, j] = 0.5 * float(value[0])
        ai = (direct - np.outer(data, data) / (2.0 * q)) / max(q / ops.dim, np.finfo(float).tiny)
        ai = (ai + ai.T) * 0.5
        if not np.all(np.isfinite(ai)) or np.linalg.cond(ai) > 1e12:
            damping = max(damping, 1e-8 * max(float(np.trace(np.abs(ai))) / max(len(names), 1), 1.0))
        trial_ai = ai + damping * np.eye(len(names))
        try:
            step = np.linalg.solve(trial_ai, score)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(trial_ai, score, rcond=None)[0]
        step_norm = float(np.max(np.abs(step))) if step.size else 0.0
        if step_norm > max_step:
            step *= max_step / step_norm
        old_norm = float(np.linalg.norm(score / np.sqrt(np.maximum(np.diag(trial_ai), 1e-12))))
        accepted = False
        for halving in range(10):
            trial_coords = np.clip(coords + step * (0.5 ** halving), -30.0, 30.0)
            trial_prior = MultiComponentPrior.from_coordinates(ops.labels, trial_coords)
            trial_ph_y = projected_solve(y, trial_prior)
            trial_score = []
            trial_rhs = np.column_stack((trial_ph_y, probes))
            trial_values = ops.apply_component_derivatives_matmat(trial_rhs, trial_prior)
            for name in names:
                trial_score.append(0.5 * (trial_ph_y @ trial_values[name][:, 0]
                                          - derivative_trace(name, trial_prior, probes)))
            new_norm = float(np.linalg.norm(np.asarray(trial_score) /
                                            np.sqrt(np.maximum(np.diag(trial_ai), 1e-12))))
            if new_norm <= old_norm or np.max(np.abs(step)) * (0.5 ** halving) <= tol:
                coords = trial_coords
                accepted = True
                last_step = float(np.max(np.abs(step)) * (0.5 ** halving))
                break
        if not accepted:
            last_step = 0.0
            damping = max(1e-6, damping * 10.0 if damping else 1e-6)
            if score_norm <= 10.0 * tol:
                converged = True
            break
        last_score = score
        last_prior = prior
        last_sigma_e2 = q / ops.dim
        covariance = np.linalg.pinv(ai + damping * np.eye(len(names)))

    fitted = MultiComponentPrior(
        ops.labels,
        tuple(SimplifiedPrior(last_sigma_e2 * p.sigma_b2, p.tau) for p in last_prior.components),
    )
    genetic = ops.kernel_trace(fitted)
    h2 = genetic / max(genetic + ops.dim * last_sigma_e2, np.finfo(float).tiny)
    errors = None if covariance is None else {
        name: float(np.sqrt(max(covariance[index, index], 0.0)))
        for index, name in enumerate(names)
    }
    # The AI path does not evaluate a stochastic log-determinant objective;
    # expose a finite score diagnostic in the common ``objective`` slot.
    objective = float(0.5 * np.dot(last_score, last_score))
    return MultiComponentFit(
        fitted, last_sigma_e2, h2, objective, converged, ops, covariance, errors,
        trace_method, trace_probes, cg_tol,
        float(np.linalg.norm(last_score, ord=np.inf)),
        float("nan"),
        y.copy(),
        last_step,
        damping,
    )
