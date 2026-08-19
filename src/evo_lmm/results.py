"""Typed results returned by evolutionary REML and BOLT-style analyses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .priors import EvolutionaryPrior


@dataclass
class FitDiagnostics:
    """Numerical and identifiability diagnostics from a fit."""

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
    """Compact BOLT-compatible association output for one chromosome block."""

    chrom: Any
    local_idx: np.ndarray
    score: np.ndarray
    beta: np.ndarray
    se: np.ndarray
    chisq: np.ndarray
    pvalue: np.ndarray
