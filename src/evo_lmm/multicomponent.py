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
from .reml import _dense_projection, convergence_statistics
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
            raise ValueError(
                "labels and components must be non-empty and have equal length"
            )
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("component labels must be unique")

    @classmethod
    def from_parameters(
        cls, parameters: Mapping[Any, tuple[float, float]]
    ) -> MultiComponentPrior:
        labels = tuple(parameters)
        return cls(
            labels, tuple(SimplifiedPrior(*parameters[label]) for label in labels)
        )

    @classmethod
    def flat(
        cls, labels: Sequence[Any], scales: Sequence[float] | None = None
    ) -> MultiComponentPrior:
        """Return the exact M0 boundary with all ``tau_c = 0``."""
        labels_tuple = tuple(labels)
        scale_values = [1.0] * len(labels_tuple) if scales is None else list(scales)
        if len(scale_values) != len(labels_tuple):
            raise ValueError("scales must match labels")
        return cls(
            labels_tuple, tuple(SimplifiedPrior(scale, 0.0) for scale in scale_values)
        )

    def with_shared_tau(self, tau: float) -> MultiComponentPrior:
        """Return M1's shared-``tau`` identifiability-crutch specification."""
        if tau < 0 or not np.isfinite(tau):
            raise ValueError("tau must be finite and non-negative")
        return MultiComponentPrior(
            self.labels,
            tuple(SimplifiedPrior(p.sigma_b2, tau) for p in self.components),
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
            values.extend(
                (
                    np.log(component.sigma_b2),
                    np.log(component.tau) if component.tau > 0 else -np.inf,
                )
            )
        return np.asarray(values, dtype=np.float64)

    @classmethod
    def from_coordinates(
        cls, labels: Sequence[Any], coordinates: np.ndarray
    ) -> MultiComponentPrior:
        labels_tuple = tuple(labels)
        values = np.asarray(coordinates, dtype=np.float64)
        if values.shape != (2 * len(labels_tuple),):
            raise ValueError(
                "coordinates must contain (log_sigma_b2, log_tau) per component"
            )
        components = []
        for index in range(len(labels_tuple)):
            sigma = float(np.exp(values[2 * index]))
            tau = (
                float(np.exp(values[2 * index + 1]))
                if np.isfinite(values[2 * index + 1])
                else 0.0
            )
            components.append(SimplifiedPrior(sigma, tau))
        return cls(labels_tuple, tuple(components))


class MultiComponentOps:
    """Sum of independently partitioned projected raw-dosage component operators."""

    def __init__(self, components: Mapping[Any, EvolutionaryLmmOps]):
        if not components:
            raise ValueError("at least one annotation component is required")
        self.components = dict(components)
        first = next(iter(self.components.values()))
        if any(
            op.n != first.n or op.rank != first.rank for op in self.components.values()
        ):
            raise ValueError(
                "all component operators must have the same individuals and covariate rank"
            )
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
    def from_operators(
        cls, components: Mapping[Any, EvolutionaryLmmOps]
    ) -> MultiComponentOps:
        """Construct a partition from already-adapted dense or GRGL operators."""
        return cls(components)

    @classmethod
    def from_dense(
        cls,
        genotypes: Mapping[Any, np.ndarray],
        frequencies: Mapping[Any, np.ndarray],
        covariates: np.ndarray | None = None,
    ) -> MultiComponentOps:
        return cls(
            {
                label: EvolutionaryLmmOps(matrix, frequencies[label], covariates)
                for label, matrix in genotypes.items()
            }
        )

    def component_kernels(self, prior: MultiComponentPrior) -> dict[Any, np.ndarray]:
        if prior.labels != self.labels:
            raise ValueError("prior labels do not match the partition")
        return {
            label: component.sigma_b2 * self._component_kernel(label, component)
            for label, component in zip(prior.labels, prior.components)
        }

    def _component_kernel(self, label: Any, prior: SimplifiedPrior) -> np.ndarray:
        """Materialize one component only when an exact small fit requests it."""
        op = self.components[label]
        if all(chrom.dense is not None for chrom in op._chromosomes):
            return op.dense_kernel(prior)
        identity = np.eye(op.n, dtype=np.float64)
        return op.apply_k_matmat(identity, prior)

    def dense_kernel(self, prior: MultiComponentPrior) -> np.ndarray:
        return sum(
            self.component_kernels(prior).values(), start=np.zeros((self.n, self.n))
        )

    def kernel_trace(self, prior: MultiComponentPrior) -> float:
        """Return ``tr(P_C K P_C)`` without materialising a dense kernel."""
        if prior.labels != self.labels:
            raise ValueError("prior labels do not match the partition")
        return float(
            sum(
                component.sigma_b2 * self.components[label].kernel_trace(component)
                for label, component in zip(prior.labels, prior.components)
            )
        )

    def apply_k(self, values: np.ndarray, prior: MultiComponentPrior) -> np.ndarray:
        """Apply the summed partitioned kernel to one or many vectors."""
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            return self.apply_k(matrix[:, None], prior)[:, 0]
        if matrix.ndim != 2 or matrix.shape[0] != self.n:
            raise ValueError("values must have shape (n,) or (n, k)")
        return sum(
            component.sigma_b2
            * self.components[label].apply_k_matmat(matrix, component)
            for label, component in zip(prior.labels, prior.components)
        )

    def derivative_kernels(self, prior: MultiComponentPrior) -> dict[str, np.ndarray]:
        """Return analytic derivatives in the same coordinate order as the prior."""
        result: dict[str, np.ndarray] = {}
        for index, (label, component) in enumerate(zip(self.labels, prior.components)):
            op = self.components[label]
            if any(chrom.dense is None for chrom in op._chromosomes):
                identity = np.eye(op.n, dtype=np.float64)
                result[f"log_sigma_b2[{label}]"] = (
                    component.sigma_b2 * op.apply_k_matmat(identity, component)
                )
                result[f"log_tau[{label}]"] = component.sigma_b2 * op.apply_dh_matmat(
                    identity, component, "log_tau"
                )
                continue
            result[f"log_sigma_b2[{label}]"] = component.sigma_b2 * op.dense_kernel(
                component
            )
            derivative = component.weight_derivatives(op.frequencies)["log_tau"]
            matrices = []
            offset = 0
            for chrom in op._chromosomes:
                centered = op.project(chrom.dense)
                count = chrom.n_variants
                matrices.append(
                    (centered * derivative[offset : offset + count]) @ centered.T
                )
                offset += count
            result[f"log_tau[{label}]"] = component.sigma_b2 * sum(
                matrices, start=np.zeros((self.n, self.n))
            )
        return result

    def apply_dh_matmat(
        self, values: np.ndarray, prior: MultiComponentPrior, parameter: str
    ) -> np.ndarray:
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

    def apply_shape_matmat(
        self, values: np.ndarray, prior: MultiComponentPrior
    ) -> np.ndarray:
        """Apply ``H = I + sum_c K_c`` to batched projected vectors."""
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != self.n:
            raise ValueError("values must have shape (n, k)")
        # H is full-rank; projection is applied to the right-hand side and in
        # the fixed-effect correction, not to the identity term itself.
        return matrix + self.apply_k(matrix, prior)


@dataclass(frozen=True)
class MultiComponentFit:
    """Profiled REML result for the partitioned simplified model.

    ``status`` names how the fit ended -- ``converged``,
    ``stalled_near_tolerance``, ``line_search_stalled``, ``iteration_cap``,
    ``not_started``, or ``optimizer_success``/``optimizer_failure`` on the exact
    dense method, where the label is SciPy's verdict and not this module's
    criterion.  ``step_se_norm`` is the criterion itself,
    ``max_i |step_i| / SE_i``; see :func:`evo_lmm.reml.convergence_statistics`.

    ``warnings`` reports each ``tau_c`` at or near the zero boundary.  It is a
    report only: a boundary hit does not change ``status`` or ``converged``.
    """

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
    status: str = "unknown"
    step_se_norm: float = float("nan")
    newton_decrement: float = float("nan")
    warnings: tuple[str, ...] = ()
    initialization: str = "default"
    mom_raw_component_scales: np.ndarray | None = None
    mom_truncated: np.ndarray | None = None


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
    objective = 0.5 * (
        logdet + np.linalg.slogdet(fixed)[1] + ops.dim * np.log(q / ops.dim)
    )
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
    step_se_tol: float = 1e-2,
    max_step: float = 2.0,
    fit_tau: bool = True,
    initialization: str = "default",
) -> MultiComponentFit:
    """Fit by profiled REML.

    The residual scale ``sigma_e2`` is profiled; component scales are searched
    as ratios to that scale in log coordinates.  The objective is REML, and
    the reported component ``sigma_b2`` values are returned on the scientific
    scale after profiling.  ``tol`` is the score-norm convergence tolerance;
    the default matches the single-component fitter.  It is no longer the
    convergence gate: convergence is declared when ``step_se_tol`` bounds
    ``max_i |step_i| / SE_i``, the same scale-free statistic the
    single-component fitter uses, and ``tol`` only accepts a vanishing
    line-search displacement.

    ``fit_tau=False`` holds every ``tau_c`` at its value in ``initial`` and
    searches only the ``|c|`` scale coordinates.  That is the pooled-shape mode
    used for per-gene fitting, where the shapes are estimated once across genes
    and must not be re-estimated per gene.

    ``initialization="he"`` applies the joint projected Haseman--Elston
    ``(|c|+1)`` moment system to the component *scale* coordinates.  The
    current category-specific ``tau_c`` values are retained: HE is linear in
    the covariance components conditional on those evolutionary weights and
    cannot identify the nonlinear shape coordinates by itself.  Raw moment
    scales and negative-scale flags are retained on the returned fit for the
    required no-truncation audit; invalid or non-positive values fall back to
    the requested/default scale for optimisation.
    """
    if method not in ("ai", "dense"):
        raise ValueError("method must be 'ai' or 'dense'")
    if trace_method not in ("hutchinson", "xtrace"):
        raise ValueError("trace_method must be 'hutchinson' or 'xtrace'")
    if initialization not in ("default", "he"):
        raise ValueError("initialization must be 'default' or 'he'")
    if (
        trace_probes < 2
        or not np.isfinite(cg_tol)
        or cg_tol <= 0
        or not np.isfinite(tol)
        or tol <= 0
        or not np.isfinite(max_step)
        or max_step <= 0
    ):
        raise ValueError(
            "trace_probes must be at least two; tolerances and max_step must be positive"
        )
    values = np.asarray(y, dtype=np.float64)
    if values.shape != (ops.n,) or not np.all(np.isfinite(values)):
        raise ValueError("y must be a finite vector with one entry per individual")
    if initial is None:
        prior = MultiComponentPrior(
            tuple(ops.labels), tuple(SimplifiedPrior(1.0, 0.1) for _ in ops.labels)
        )
    elif isinstance(initial, MultiComponentPrior):
        prior = initial
    else:
        prior = MultiComponentPrior.from_coordinates(ops.labels, np.asarray(initial))
    if prior.labels != ops.labels:
        raise ValueError("initial prior labels do not match the partition")

    mom_raw_component_scales = None
    mom_truncated = None
    if initialization == "he" and ops.n_components > 1:
        # Import lazily: baselines exposes the public moment helper and imports
        # this module's partitioned operators.
        from .baselines import joint_mom_initialization

        moment = joint_mom_initialization(
            ops,
            values,
            prior,
            trace_method=(
                "exact"
                if all(
                    chrom.dense is not None
                    for op in ops.components.values()
                    for chrom in op._chromosomes
                )
                else trace_method
            ),
            trace_probes=trace_probes,
            seed=seed,
        )
        mom_raw_component_scales = moment.raw_component_scales.copy()
        mom_truncated = moment.truncated.copy()
        residual = moment.residual_variance
        initialized_components = []
        for component, multiplier in zip(prior.components, moment.raw_component_scales):
            # The multi-component AI parameterization stores each genetic scale
            # relative to profiled sigma_e2.  HE returns scientific covariance
            # multipliers for the supplied component kernels.
            if (
                np.isfinite(multiplier)
                and multiplier > 0.0
                and np.isfinite(residual)
                and residual > 0.0
            ):
                scale = component.sigma_b2 * float(multiplier) / float(residual)
            else:
                scale = component.sigma_b2
            initialized_components.append(SimplifiedPrior(scale, component.tau))
        prior = MultiComponentPrior(prior.labels, tuple(initialized_components))

    if method == "dense" and not fit_tau:
        raise ValueError("fit_tau=False is implemented on the 'ai' method only")
    # A pooled-shape fit cannot delegate to the single-component fitter: that
    # fitter searches its own tau.
    if method == "ai" and (ops.n_components > 1 or not fit_tau):
        return _fit_multicomponent_ai(
            ops,
            values,
            prior,
            max_iter=max_iter,
            trace_method=trace_method,
            trace_probes=trace_probes,
            seed=seed,
            cg_tol=cg_tol,
            tol=tol,
            step_se_tol=step_se_tol,
            max_step=max_step,
            fit_tau=fit_tau,
            initialization=initialization,
            mom_raw_component_scales=mom_raw_component_scales,
            mom_truncated=mom_truncated,
        )

    # The one-category model is exactly the existing single-component model;
    # delegate it so its scale/profile/numerical conventions remain identical.
    # ``exact`` is left to the single fitter: a dense operator gets its exact
    # dense fit, and a GRGL-backed one stays matrix-free at the requested
    # ``cg_tol`` instead of being forced to materialise dense kernels.
    if ops.n_components == 1:
        from .reml import fit_reml

        label = ops.labels[0]
        single = fit_reml(
            ops.components[label],
            values,
            initial=prior.components[0],
            max_iter=max_iter,
            trace_method=trace_method,
            trace_probes=trace_probes,
            seed=seed,
            cg_tol=cg_tol,
            tol=tol,
            max_step=max_step,
            initialization=initialization,
        )
        fitted = MultiComponentPrior((label,), (single.prior,))
        return MultiComponentFit(
            fitted,
            single.sigma_e2,
            single.h2,
            single.diagnostics.objective,
            single.diagnostics.converged,
            ops,
            trace_method=trace_method,
            trace_probes=trace_probes,
            cg_tol=cg_tol,
            score_norm=single.diagnostics.score_norm,
            phenotype=values.copy(),
            status=single.diagnostics.status,
            step_se_norm=single.diagnostics.step_se_norm,
            newton_decrement=single.diagnostics.newton_decrement,
            warnings=_tau_warnings(ops, fitted),
            initialization=initialization,
        )
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
        scaled = tuple(
            SimplifiedPrior(sigma_e2 * p.sigma_b2, p.tau) for p in current.components
        )
        return (
            float(objective),
            float(sigma_e2),
            MultiComponentPrior(current.labels, scaled),
        )

    result = minimize(
        lambda theta: evaluate(theta)[0],
        coordinates,
        method="L-BFGS-B",
        bounds=[(-30.0, 30.0), (-30.0, 30.0)] * ops.n_components,
        options={"maxiter": int(max_iter), "ftol": 1e-12},
    )
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
    return MultiComponentFit(
        fitted,
        sigma_e2,
        h2,
        objective,
        bool(result.success),
        ops,
        covariance,
        standard_errors,
        trace_method,
        trace_probes,
        cg_tol,
        phenotype=values.copy(),
        status="optimizer_success" if result.success else "optimizer_failure",
        warnings=_tau_warnings(ops, fitted),
        initialization=initialization,
        mom_raw_component_scales=mom_raw_component_scales,
        mom_truncated=mom_truncated,
    )


def coordinate_names(labels: Sequence[Any]) -> list[str]:
    """Return the transformed coordinate names in prior order."""
    return [
        name
        for label in labels
        for name in (f"log_sigma_b2[{label}]", f"log_tau[{label}]")
    ]


def projected_solve(
    ops: MultiComponentOps,
    rhs: np.ndarray,
    prior: MultiComponentPrior,
    *,
    cg_tol: float = 5e-4,
) -> np.ndarray:
    """Apply the projected inverse ``P = H^-1 - H^-1 B (B'H^-1 B)^-1 B'H^-1``.

    One block-CG solve covers the covariate basis and every right-hand-side
    column; ``H`` is applied through the batched operator ``matmat`` so a
    GRGL-backed component keeps one traversal per block rather than one per
    column.  A rank-deficient block Gram system falls back to per-column CG.
    """
    rhs_matrix = np.asarray(rhs, dtype=np.float64)
    was_vector = rhs_matrix.ndim == 1
    if was_vector:
        rhs_matrix = rhs_matrix[:, None]
    if rhs_matrix.ndim != 2 or rhs_matrix.shape[0] != ops.n:
        raise ValueError("rhs must have shape (n,) or (n, k)")
    if not np.isfinite(cg_tol) or cg_tol <= 0:
        raise ValueError("cg_tol must be positive")
    combined_rhs = np.column_stack((ops.basis, ops.project(rhs_matrix)))
    operator = LinearOperator(
        (ops.n, ops.n),
        matvec=lambda vector: ops.apply_shape_matmat(vector[:, None], prior)[:, 0],
        matmat=lambda block: ops.apply_shape_matmat(block, prior),
        dtype=np.float64,
    )

    def block_cg(rhs_block: np.ndarray) -> np.ndarray:
        x = np.zeros_like(rhs_block)
        residual = rhs_block.copy()
        direction = residual.copy()
        scale = np.maximum(np.sum(residual * residual, axis=0), 1e-30)
        for _ in range(max(50, 4 * ops.n)):
            applied = operator.matmat(direction)
            gram = direction.T @ applied
            rr = residual.T @ residual
            alpha = np.linalg.lstsq(gram, rr, rcond=None)[0]
            x += direction @ alpha
            residual -= applied @ alpha
            if np.all(np.sum(residual * residual, axis=0) <= scale * cg_tol**2):
                return x
            new_rr = residual.T @ residual
            beta = np.linalg.lstsq(rr, new_rr, rcond=None)[0]
            direction = residual + direction @ beta
        # Dependent probe columns can make block Gram systems rank-deficient;
        # retain a strict fallback for those rare cases.
        for column in range(rhs_block.shape[1]):
            solution, info = cg(
                operator,
                rhs_block[:, column],
                rtol=cg_tol,
                atol=0.0,
                maxiter=max(50, 4 * ops.n),
            )
            if info != 0:
                raise np.linalg.LinAlgError(
                    f"multi-component CG failed with info={info}"
                )
            x[:, column] = solution
        return x

    solutions = block_cg(combined_rhs)
    basis_solution = solutions[:, : ops.rank]
    fixed = ops.basis.T @ basis_solution
    target = solutions[:, ops.rank :]
    correction = basis_solution @ np.linalg.solve(fixed, ops.basis.T @ target)
    result = target - correction
    return result[:, 0] if was_vector else result


def score_and_information(
    ops: MultiComponentOps,
    y: np.ndarray,
    prior: MultiComponentPrior,
    probes: np.ndarray,
    *,
    trace_method: str = "hutchinson",
    cg_tol: float = 5e-4,
    information: bool = True,
) -> dict[str, Any]:
    """Profiled REML score and average information at one point.

    ``score[i] = 0.5 * ((P y)' dV_i (P y) / sigma_e2 - tr(P dV_i))`` is the
    gradient of :func:`profiled_reml_objective` in the transformed coordinates,
    and

    ``AI[i, j] = 0.5 * (P y)' dV_i P dV_j (P y) / sigma_e2`` minus the
    profiling correction ``data_i data_j / (2 q sigma_e2)``.

    Both conventions match the single-component fitter.  Dropping the
    ``sigma_e2`` division leaves a quantity that is not the gradient of any
    objective, and contracting the information matrix on the left with
    ``P dV_i P y`` rather than ``P y`` inserts a third derivative factor and
    makes the matrix indefinite; either error stalls the iteration.

    Passing ``probes = sqrt(n) * I`` makes the Hutchinson trace exact, which is
    how the dense oracle tests pin this function.
    """
    values = np.asarray(y, dtype=np.float64)
    if values.shape != (ops.n,) or not np.all(np.isfinite(values)):
        raise ValueError("y must be a finite vector with shape (n,)")
    probe_matrix = np.asarray(probes, dtype=np.float64)
    if probe_matrix.ndim != 2 or probe_matrix.shape[0] != ops.n:
        raise ValueError("probes must have shape (n, k)")
    if trace_method not in ("hutchinson", "xtrace"):
        raise ValueError("trace_method must be 'hutchinson' or 'xtrace'")
    names = coordinate_names(ops.labels)
    tiny = np.finfo(float).tiny
    solved = projected_solve(
        ops, np.column_stack((values, probe_matrix)), prior, cg_tol=cg_tol
    )
    ph_y = solved[:, 0]
    ph_probes = solved[:, 1:]
    q = float(values @ ph_y)
    if q <= 0 or not np.isfinite(q):
        raise np.linalg.LinAlgError("profiled REML quadratic form is non-positive")
    sigma_e2 = q / ops.dim
    derivatives = ops.apply_component_derivatives_matmat(
        np.column_stack((ph_y, probe_matrix)), prior
    )
    data = np.asarray([float(ph_y @ derivatives[name][:, 0]) for name in names])
    traces = np.empty(len(names), dtype=np.float64)
    trace_errors: list[float] = []
    for index, name in enumerate(names):
        if trace_method == "xtrace":
            from .trace import xtrace

            def apply(block: np.ndarray, name: str = name) -> np.ndarray:
                return projected_solve(
                    ops, ops.apply_dh_matmat(block, prior, name), prior, cg_tol=cg_tol
                )

            estimate = xtrace(apply, probe_matrix)
            traces[index] = estimate.value
            trace_errors.append(float(estimate.standard_error))
            continue
        # tr(P dV) = E[z' P dV z] is evaluated as (P z)' (dV z), so the probe
        # columns of the single block solve above are reused unchanged.
        samples = np.sum(ph_probes * derivatives[name][:, 1:], axis=0)
        traces[index] = float(np.mean(samples))
        trace_errors.append(
            float(np.std(samples, ddof=1) / np.sqrt(samples.size))
            if samples.size > 1
            else 0.0
        )
    score = 0.5 * (data / max(sigma_e2, tiny) - traces)
    ai = None
    if information:
        eta = np.column_stack([derivatives[name][:, 0] for name in names])
        zeta = projected_solve(ops, eta, prior, cg_tol=cg_tol)
        direct = np.empty((len(names), len(names)), dtype=np.float64)
        for index, name in enumerate(names):
            direct[index] = 0.5 * (ph_y @ ops.apply_dh_matmat(zeta, prior, name))
        ai = (direct - np.outer(data, data) / (2.0 * q)) / max(sigma_e2, tiny)
        ai = (ai + ai.T) * 0.5
    return {
        "names": names,
        "prior": prior,
        "ph_y": ph_y,
        "q": q,
        "sigma_e2": sigma_e2,
        "data": data,
        "traces": traces,
        "score": score,
        "ai": ai,
        "trace_error": float(max(trace_errors, default=0.0)),
    }


def _tau_warnings(
    ops: MultiComponentOps, prior: MultiComponentPrior, threshold: float = 1e-6
) -> tuple[str, ...]:
    """Report each ``tau_c`` sitting in a regime where it is unidentified.

    Judged on the effect-variance weights ``w_j = 1 / (1 + 2 tau_c q_j)``, which
    is what the data actually see, rather than on a magic threshold for
    ``tau_c`` itself:

    ``flat``
        ``max_j (1 - w_j) <= threshold``: the component kernel is
        indistinguishable from the flat ``tau_c = 0`` kernel.
    ``saturated``
        ``max_j w_j <= threshold``: every weight is crushed, so the kernel is
        indistinguishable from the ``tau_c -> infinity`` shape and only the
        product ``tau_c * q_j`` is visible.

    Both are reports.  A boundary hit does not change ``status`` or
    ``converged``: a flat kernel is a legitimate estimate, and this module does
    not decide what a caller may report.
    """
    messages: list[str] = []
    for label, component in zip(prior.labels, prior.components):
        weights = component.weights(ops.components[label].frequencies)
        if weights.size == 0:
            continue
        if float(np.max(1.0 - weights)) <= threshold:
            messages.append(
                f"tau[{label}] is at or near zero; the component kernel is flat "
                f"and tau[{label}] is weakly identified"
            )
        elif float(np.max(weights)) <= threshold:
            messages.append(
                f"tau[{label}] is large enough to saturate every weight; only "
                f"tau[{label}] * q_j is identified"
            )
    return tuple(messages)


def _embed(block: np.ndarray, active: np.ndarray, size: int) -> np.ndarray:
    """Place an active-coordinate covariance block into a full-size matrix.

    Coordinates that were held fixed carry zero (co)variance: the fit did not
    estimate them.
    """
    full = np.zeros((size, size), dtype=np.float64)
    full[np.ix_(active, active)] = block
    return full


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
    step_se_tol: float = 1e-2,
    fit_tau: bool = True,
    initialization: str = "default",
    mom_raw_component_scales: np.ndarray | None = None,
    mom_truncated: np.ndarray | None = None,
) -> MultiComponentFit:
    """Projected AI-REML with profiled residual scale and batched probes.

    ``fit_tau=False`` restricts the search, the convergence test, and the
    reported covariance to the scale coordinates; the ``tau_c`` supplied in
    ``initial`` are held fixed and reported with a zero standard error, since
    they are not estimated by this fit.
    """
    coords = initial.coordinates.copy()
    coords[~np.isfinite(coords)] = np.log(np.finfo(float).tiny)
    probes = (
        rademacher_probes(ops.n, trace_probes, seed)
        if trace_method == "hutchinson"
        else spherical_gaussian_probes(ops.n, trace_probes, seed)
    )
    names = coordinate_names(ops.labels)
    active = np.arange(len(names)) if fit_tau else np.arange(0, len(names), 2)
    converged = False
    status = "not_started"
    covariance = None
    last_score = np.full(len(names), np.nan)
    last_step = 0.0
    last_rejected = False
    last_trace_error = float("nan")
    last_step_se = float("nan")
    last_decrement = float("nan")
    damping = 0.0

    def evaluate(current: np.ndarray, *, information: bool = True) -> dict[str, Any]:
        state = score_and_information(
            ops,
            y,
            MultiComponentPrior.from_coordinates(ops.labels, current),
            probes,
            trace_method=trace_method,
            cg_tol=cg_tol,
            information=information,
        )
        state["coords"] = current
        return state

    # Seed the reported state from the initial point.  Two exits can leave the
    # loop before the in-loop assignments below: ``max_iter=0``, and a
    # line-search rejection on the first iteration.  Without this seeding both
    # propagate a placeholder into ``SimplifiedPrior`` and raise "sigma_b2 must
    # be finite and strictly positive" -- a validator error from the
    # result-assembly step rather than an actionable fit diagnostic.
    last_prior = MultiComponentPrior.from_coordinates(ops.labels, coords)
    seed_quadratic = float(y @ projected_solve(ops, y, last_prior, cg_tol=cg_tol))
    if not np.isfinite(seed_quadratic) or seed_quadratic <= 0.0:
        raise np.linalg.LinAlgError(
            "profiled REML quadratic form is non-positive at the initial prior"
        )
    last_sigma_e2 = seed_quadratic / ops.dim
    pending: dict[str, Any] | None = None

    for iteration in range(1, int(max_iter) + 1):
        state = pending if pending is not None else evaluate(coords)
        pending = None
        score = state["score"][active]
        ai = state["ai"][np.ix_(active, active)]
        last_score = state["score"]
        last_prior = state["prior"]
        last_sigma_e2 = state["sigma_e2"]
        last_trace_error = state["trace_error"]
        # Convergence is judged on the scale-free step/standard-error
        # statistic.  ``score_norm`` is still reported on the result, but it is
        # no longer a criterion: it grows with the sample size.
        last_step_se, last_decrement = convergence_statistics(score, ai)
        if last_step_se <= step_se_tol:
            converged = True
            status = "converged"
            covariance = _embed(
                np.linalg.pinv(ai + damping * np.eye(active.size)), active, len(names)
            )
            break
        if not np.all(np.isfinite(ai)) or np.linalg.cond(ai) > 1e12:
            damping = max(
                damping,
                1e-8 * max(float(np.trace(np.abs(ai))) / max(active.size, 1), 1.0),
            )
        trial_ai = ai + damping * np.eye(active.size)
        try:
            step = np.linalg.solve(trial_ai, score)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(trial_ai, score, rcond=None)[0]
        step_norm = float(np.max(np.abs(step))) if step.size else 0.0
        if step_norm > max_step:
            step *= max_step / step_norm
        scaling = np.sqrt(np.maximum(np.diag(trial_ai), 1e-12))
        old_norm = float(np.linalg.norm(score / scaling))
        accepted = False
        for halving in range(10):
            trial_coords = coords.copy()
            trial_coords[active] = np.clip(
                coords[active] + step * (0.5**halving), -30.0, 30.0
            )
            trial = evaluate(trial_coords)
            new_norm = float(np.linalg.norm(trial["score"][active] / scaling))
            if new_norm <= old_norm or np.max(np.abs(step)) * (0.5**halving) <= tol:
                coords = trial_coords
                accepted = True
                pending = trial
                last_step = float(np.max(np.abs(step)) * (0.5**halving))
                break
        if not accepted:
            # The step was rejected, but this iteration's point is a valid
            # iterate: report it rather than the previous one (or, on the first
            # iteration, rather than the seed).  A rejection escalates the
            # Levenberg damping and the loop continues: the average-information
            # matrix is routinely near-singular in the weakly identified tau
            # directions, and an undamped step there is capped to one that moves
            # the well-identified scale coordinates by almost nothing.
            last_step = 0.0
            last_rejected = True
            covariance = _embed(np.linalg.pinv(trial_ai), active, len(names))
            damping = max(1e-6, damping * 10.0 if damping else 1e-6)
            if last_step_se <= 10.0 * step_se_tol:
                converged = True
                status = "stalled_near_tolerance"
                break
            continue
        last_rejected = False
        covariance = _embed(np.linalg.pinv(trial_ai), active, len(names))
        damping = 0.0 if damping <= 1e-8 else damping * 0.1

    if not converged and int(max_iter) > 0:
        status = "line_search_stalled" if last_rejected else "iteration_cap"

    # Coordinates that were held fixed are restored from ``initial`` rather
    # than reconstructed from log coordinates, so a pooled shape survives the
    # round trip exactly instead of within one ulp.
    fitted = MultiComponentPrior(
        ops.labels,
        tuple(
            SimplifiedPrior(
                last_sigma_e2 * component.sigma_b2,
                component.tau if fit_tau else initial.components[index].tau,
            )
            for index, component in enumerate(last_prior.components)
        ),
    )
    genetic = ops.kernel_trace(fitted)
    h2 = genetic / max(genetic + ops.dim * last_sigma_e2, np.finfo(float).tiny)
    errors = (
        None
        if covariance is None
        else {
            name: float(np.sqrt(max(covariance[index, index], 0.0)))
            for index, name in enumerate(names)
        }
    )
    # The AI path does not evaluate a stochastic log-determinant objective;
    # expose a finite score diagnostic in the common ``objective`` slot.
    objective = float(0.5 * np.dot(last_score[active], last_score[active]))
    return MultiComponentFit(
        fitted,
        last_sigma_e2,
        h2,
        objective,
        converged,
        ops,
        covariance,
        errors,
        trace_method,
        trace_probes,
        cg_tol,
        float(np.linalg.norm(last_score[active], ord=np.inf)),
        last_trace_error,
        y.copy(),
        last_step,
        damping,
        status,
        float(last_step_se),
        float(last_decrement),
        _tau_warnings(ops, fitted),
        initialization,
        mom_raw_component_scales,
        mom_truncated,
    )
