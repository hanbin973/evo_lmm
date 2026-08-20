"""Typed results returned by evolutionary REML and BOLT-style analyses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .priors import EvolutionaryPrior


@dataclass
class FitDiagnostics:
    """Numerical and identifiability diagnostics from a fit.

    ``converged`` is a summary; ``status`` names *how* the fit ended and is the
    field to branch on:

    ``converged``
        The step/standard-error criterion was met at an accepted iterate.
    ``stalled_near_tolerance``
        Every step halving was rejected, but the criterion was within ten times
        its tolerance, so the iterate is accepted as converged.
    ``line_search_stalled``
        The iteration budget ran out and the final iteration's step was
        rejected.
    ``iteration_cap``
        The iteration budget ran out with steps still being accepted.
    ``not_started``
        ``max_iter=0``; the reported state is the initial point.
    ``dense_finish``
        The exact dense finishing optimizer reported success.
    ``dense_finish_backstop``
        The finishing optimizer did not report success and convergence was
        declared only by the loose ``||score||_inf < 1e-4`` back-stop.
    ``dense_finish_failed``
        The finishing optimizer ran and convergence was not declared.

    ``step_se_norm`` is the convergence statistic itself:
    ``max_i |step_i| / SE_i`` with ``step = AI^-1 score`` and
    ``SE = sqrt(diag(AI^-1))`` -- the largest move the next undamped Newton
    step would make in any coordinate, in units of that coordinate's own
    standard error.  ``newton_decrement`` is ``sqrt(score' AI^-1 score)``,
    logged for comparison.  Both are dimensionless, invariant to a
    reparameterization of the coordinates, and order one at a fixed distance
    from the optimum in standard-error units -- unlike ``score_norm``, which at
    the same point grows roughly like ``sqrt(n)``.
    """

    converged: bool
    iterations: int
    objective: float
    score_norm: float
    ai_condition: float
    ai_damping: float
    accepted_step: float
    trace_estimator: str
    trace_probes: int
    initialization: str = "default"
    trace_operator_queries: int = 0
    trace_standard_errors: dict[str, float] = field(default_factory=dict)
    cg_iterations: list[int] = field(default_factory=list)
    cg_warm_start_hits: int = 0
    cg_warm_start_rejections: int = 0
    cg_initial_residual_norms: list[float] = field(default_factory=list)
    cg_final_residual_norms: list[float] = field(default_factory=list)
    random_seed: int | None = None
    boundary_hits: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    status: str = "unknown"
    step_se_norm: float = float("nan")
    newton_decrement: float = float("nan")


@dataclass
class FitResult:
    """Scientific-scale fit result for an evolutionary LMM."""

    prior: EvolutionaryPrior
    sigma_b2: float
    sigma_e2: float
    delta: float
    h2: float
    log_likelihood: float
    fixed_effects: np.ndarray
    projected_phenotype: np.ndarray
    ph_y: np.ndarray
    diagnostics: FitDiagnostics
    model: str
    ops: Any = field(repr=False, default=None)

    @property
    def tau(self) -> float:
        return float(self.prior.tau)

    @property
    def rho(self) -> float:
        return float(getattr(self.prior, "rho", 1.0))

    @property
    def rho2(self) -> float:
        return float(self.prior.rho2)

    @property
    def sigma_g2(self) -> float:
        """Compatibility alias; unlike GRAPP this is raw-effect ``sigma_b2``."""

        return self.sigma_b2

    @property
    def genetic_variance(self) -> float:
        if self.ops is None:
            return float("nan")
        trace_k = self.ops.kernel_trace(self.prior)
        return float(self.sigma_b2 * trace_k / max(self.ops.n, 1))

    def blup(self) -> np.ndarray:
        """Return the projected genetic-value BLUP ``sigma_b2 K P_V y``."""

        if self.ops is None:
            raise ValueError("fit result is not attached to an operator")
        # ``ph_y`` solves H^{-1} in raw units.  Since V=sigma_b2*H, the
        # sigma_b2 factor in G cancels the 1/sigma_b2 factor in V^{-1}.
        return self.ops.apply_k(self.ph_y, self.prior)


@dataclass(frozen=True)
class AssociationResult:
    """Compact BOLT-compatible association output for one chromosome block.

    ``beta`` and ``se`` are in raw diploid-dosage effect units, matching the
    evolutionary model's ``sigma_b2`` scale.  ``score`` is the calibrated
    inverse-variance score ``x_j^T V_loco^-1 y`` on the BOLT-normalised test
    column.  Entries excluded by ``model_mask`` (monomorphic or
    covariate-collinear columns) carry ``nan`` statistics and ``pvalue = 1``.
    """

    chrom: Any
    local_idx: np.ndarray
    score: np.ndarray
    beta: np.ndarray
    se: np.ndarray
    chisq: np.ndarray
    pvalue: np.ndarray
    chisq_linreg: np.ndarray | None = None
    pvalue_linreg: np.ndarray | None = None
    model_mask: np.ndarray | None = None
    frequencies: np.ndarray | None = None
    inverse_scale: float = float("nan")
    calibration_factor: float = float("nan")

    @property
    def n_variants(self) -> int:
        return int(self.local_idx.size)

    def good(self) -> np.ndarray:
        """Return the boolean mask of variants with reportable statistics."""

        if self.model_mask is None:
            return np.isfinite(self.chisq)
        return np.asarray(self.model_mask, dtype=bool)
