"""Trace estimators shared by REML and Haseman--Elston initialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class TraceEstimate:
    """A trace estimate and its probe-based standard error."""

    value: float
    standard_error: float
    estimator: str
    probes: int


def rademacher_probes(n: int, probes: int, seed: int) -> np.ndarray:
    """Create deterministic Rademacher probes with shape ``(n, probes)``."""

    if n <= 0 or probes <= 0:
        raise ValueError("n and probes must be positive")
    rng = np.random.default_rng(int(seed))
    return rng.choice(np.array([-1.0, 1.0]), size=(int(n), int(probes)))


def hutchinson_trace(
    apply: Callable[[np.ndarray], np.ndarray],
    probes: np.ndarray,
) -> TraceEstimate:
    """Estimate ``tr(A)`` from common Rademacher probes.

    ``apply`` may support a matrix input; the implementation uses one call when
    possible and falls back to independent columns for simple callables.
    """

    z = np.asarray(probes, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("probes must have shape (n, n_probes)")
    try:
        values = np.asarray(apply(z), dtype=np.float64)
        if values.shape != z.shape:
            raise ValueError
    except (TypeError, ValueError):
        values = np.column_stack(
            [np.asarray(apply(z[:, i]), dtype=np.float64) for i in range(z.shape[1])]
        )
    samples = np.sum(z * values, axis=0)
    ddof = 1 if samples.size > 1 else 0
    standard_error = float(np.std(samples, ddof=ddof) / np.sqrt(samples.size))
    return TraceEstimate(float(np.mean(samples)), standard_error, "hutchinson", z.shape[1])


def exact_trace(matrix: np.ndarray) -> TraceEstimate:
    """Return an exact trace packaged like a stochastic estimate."""

    value = float(np.trace(np.asarray(matrix, dtype=np.float64)))
    return TraceEstimate(value, 0.0, "exact", 0)


def xtrace(
    apply: Callable[[np.ndarray], np.ndarray],
    probes: np.ndarray,
) -> TraceEstimate:
    """Optional XTrace-compatible entry point.

    The first CPU implementation uses the same shared-probe estimator while
    retaining the method label and API needed for later randomized-range
    refinement.  It is exact for the small dense oracle when callers use
    :func:`exact_trace` instead.
    """

    estimate = hutchinson_trace(apply, probes)
    return TraceEstimate(estimate.value, estimate.standard_error, "xtrace", estimate.probes)

