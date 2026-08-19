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


def spherical_gaussian_probes(n: int, probes: int, seed: int) -> np.ndarray:
    """Return isotropic Gaussian columns normalized to radius ``sqrt(n)``."""

    if n <= 0 or probes <= 0:
        raise ValueError("n and probes must be positive")
    rng = np.random.default_rng(int(seed))
    values = rng.normal(size=(int(n), int(probes)))
    norms = np.linalg.norm(values, axis=0)
    if np.any(norms <= np.finfo(np.float64).tiny):
        raise FloatingPointError("could not normalize a Gaussian probe")
    return np.asarray(values * (np.sqrt(float(n)) / norms)[None, :], dtype=np.float64)


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
    """Estimate a trace with the spherical TSLMM XTrace algorithm.

    ``probes`` contains the spherical Gaussian test columns. XTrace uses two
    operator query blocks, ``A @ probes`` and ``A @ Q``; keeping the columns
    explicit lets REML reuse the same seeded vectors at every parameter point.
    """

    omega = np.asarray(probes, dtype=np.float64)
    if omega.ndim != 2 or omega.shape[1] < 2:
        raise ValueError("XTrace requires a probe matrix with at least two columns")
    n, m = omega.shape
    norms = np.linalg.norm(omega, axis=0)
    if not np.all(np.isfinite(omega)) or np.any(norms <= np.finfo(float).tiny):
        raise ValueError("XTrace probes must be finite and nonzero")
    if not np.allclose(norms, np.sqrt(float(n)), rtol=1e-10, atol=1e-10):
        raise ValueError("XTrace probes must have spherical radius sqrt(n)")

    def _apply(values: np.ndarray) -> np.ndarray:
        result = np.asarray(apply(values), dtype=np.float64)
        if result.shape != values.shape:
            raise ValueError("trace operator returned a mismatched shape")
        return result

    y = _apply(omega)
    q, r = np.linalg.qr(y, mode="reduced")
    diagonal = np.abs(np.diag(r))
    threshold = (diagonal.max() if diagonal.size else 0.0) * max(n, m) * np.finfo(float).eps
    rank = int(np.count_nonzero(diagonal > threshold))
    if rank < 2:
        samples = np.sum(omega * y, axis=0)
        error = float(np.std(samples, ddof=1) / np.sqrt(m))
        return TraceEstimate(float(np.mean(samples)), error, "xtrace-rank-fallback", int(m))
    if rank < m:
        omega = omega[:, :rank]
        y = y[:, :rank]
        q, r = np.linalg.qr(y, mode="reduced")
        m = rank

    z = _apply(q)
    w = q.T @ omega
    # S = normalize(inv(R).T) without explicitly forming an inverse.
    s = np.linalg.solve(r.T, np.eye(m, dtype=np.float64))
    s = s / np.linalg.norm(s, axis=0)[None, :]

    def diag_prod(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.sum(left * right, axis=0)

    h = q.T @ z
    hw = h @ w
    t = z.T @ omega
    d_sw = diag_prod(s, w)
    d_shs = diag_prod(s, h @ s)
    d_tw = diag_prod(t, w)
    d_whw = diag_prod(w, hw)
    d_srmhw = diag_prod(s, r - hw)
    d_tmhrs = diag_prod(t - h.T @ w, s)
    denominator = n - np.linalg.norm(w, axis=0) ** 2 + np.abs(d_sw) ** 2
    if np.any(denominator <= 0.0) or not np.all(np.isfinite(denominator)):
        raise FloatingPointError("XTrace normalization denominator is non-positive")
    scale = (float(n - m + 1) / denominator)
    estimates = (
        np.trace(h) - d_shs
        + (d_whw - d_tw + d_tmhrs * d_sw + np.abs(d_sw) ** 2 * d_shs + d_sw * d_srmhw)
        * scale
    )
    estimates = np.asarray(estimates, dtype=np.float64)
    error = float(np.std(estimates, ddof=1) / np.sqrt(m)) if m > 1 else 0.0
    return TraceEstimate(float(np.mean(estimates)), error, "xtrace", int(m))
