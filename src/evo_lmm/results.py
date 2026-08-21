"""Typed results returned by evolutionary REML and BOLT-style analyses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .priors import EvolutionaryPrior


@dataclass(frozen=True)
class ConvergenceReport:
    """How a fit ended, in sample-size-independent terms.

    Both fitters -- single-component :func:`evo_lmm.fit_reml` and
    :func:`evo_lmm.fit_multicomponent_reml` -- judge convergence by the same
    rule and report it through this object, so the two are comparable.

    ``step_se_norm`` **is** the criterion:
    ``max_i |step_i| / SE_i`` with ``step = AI^-1 score`` and
    ``SE = sqrt(diag(AI^-1))`` -- the largest move the next undamped Newton
    step would make in any coordinate, in units of that coordinate's own
    standard error.  Convergence is declared when it is at or below
    ``step_se_tol``.  ``newton_decrement`` is ``sqrt(score' AI^-1 score)``,
    logged alongside it.  Both are dimensionless, invariant to a
    reparameterization of the coordinates, and order one at a fixed distance
    from the optimum regardless of sample size.

    ``score_norm`` is a **diagnostic, not a criterion**: at a fixed statistical
    distance from the optimum it grows roughly like ``sqrt(n)``, so no absolute
    bound on it means the same thing at two sample sizes.  It is reported
    because it is cheap and occasionally informative, never to be thresholded.

    ``status`` names the exit, and is the field to branch on:

    ``converged``
        ``step_se_norm <= step_se_tol`` at an accepted iterate.
    ``converged_after_dense_finish``
        The criterion was met only after the exact dense finishing optimizer
        ran (single-component fitter).
    ``stalled_near_tolerance``
        Every step halving was rejected, but the criterion was within ten times
        its tolerance, so the iterate is accepted as converged.
    ``line_search_stalled``
        The iteration budget ran out and the final iteration's step was
        rejected.
    ``iteration_cap``
        The iteration budget ran out with steps still being accepted.
    ``optimizer_stalled``
        A delegated optimizer (dense L-BFGS-B) returned without meeting the
        criterion, whatever its own verdict was.
    ``unidentified``
        The criterion is met on every direction the information matrix locates,
        but at least one coordinate has a standard error wider than its own
        coordinate box, so the data do not place it anywhere.  The fit is done;
        that coordinate's value is not an estimate and must not be reported.
        A pure-null panel lands here: with no genetic variance there is no
        information about the frequency-shape parameter.
    ``not_started``
        ``max_iter=0``; the reported state is the initial point.
    ``oracle``
        Not produced by an optimizer at all -- an explicitly constructed
        covariance, used by oracle tests.

    ``converged`` is the summary ``status in``
    :data:`CONVERGED_STATUSES`; it never distinguishes these cases on its own,
    which is why ``status`` exists.
    """

    status: str = "unknown"
    converged: bool = False
    iterations: int = 0
    step_se_norm: float = float("nan")
    step_se_tol: float = float("nan")
    newton_decrement: float = float("nan")
    score_norm: float = float("nan")
    accepted_step: float = float("nan")
    ai_condition: float = float("nan")
    ai_damping: float = float("nan")


CONVERGED_STATUSES = frozenset(
    {
        "converged",
        "converged_after_dense_finish",
        "stalled_near_tolerance",
        "unidentified",
    }
)


@dataclass
class FitDiagnostics:
    """Numerical and identifiability diagnostics from a fit.

    Convergence lives in :class:`ConvergenceReport` under ``convergence``; the
    flat accessors below delegate to it so both fitters expose the same names.

    ``objective`` is the profiled REML objective **or ``nan``**.  It is never a
    stand-in for something else: the stochastic paths do not evaluate a
    log-determinant, so they report ``nan`` rather than a surrogate, and
    convergence is judged by ``convergence.step_se_norm`` in every path.
    """

    convergence: ConvergenceReport
    trace_estimator: str
    trace_probes: int
    objective: float = float("nan")
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

    @property
    def status(self) -> str:
        return self.convergence.status

    @property
    def converged(self) -> bool:
        return self.convergence.converged

    @property
    def iterations(self) -> int:
        return self.convergence.iterations

    @property
    def step_se_norm(self) -> float:
        return self.convergence.step_se_norm

    @property
    def step_se_tol(self) -> float:
        return self.convergence.step_se_tol

    @property
    def newton_decrement(self) -> float:
        return self.convergence.newton_decrement

    @property
    def score_norm(self) -> float:
        return self.convergence.score_norm

    @property
    def accepted_step(self) -> float:
        return self.convergence.accepted_step

    @property
    def ai_condition(self) -> float:
        return self.convergence.ai_condition

    @property
    def ai_damping(self) -> float:
        return self.convergence.ai_damping


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
