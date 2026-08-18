"""Frequency-dependent evolutionary random-effect priors.

The priors in this module are defined on raw diploid dosage.  Frequencies are
sample allele frequencies, not a genotype-standardisation instruction.  The
separation is intentional: :mod:`evo_lmm.operators` uses these weights for the
model covariance and keeps the BOLT-normalised test operator independent.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np


Array = np.ndarray


def _frequencies(frequencies: Array) -> Array:
    values = np.asarray(frequencies, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("frequencies must be a one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("frequencies must be finite")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("sample allele frequencies must lie in [0, 1]")
    return values


def allele_frequency_q(frequencies: Array) -> Array:
    """Return ``q = x_hat * (1 - x_hat)`` for sample allele frequencies."""

    values = _frequencies(frequencies)
    return values * (1.0 - values)


def _logit(value: float) -> float:
    if value <= 0.0:
        return -np.inf
    if value >= 1.0:
        return np.inf
    return float(np.log(value) - np.log1p(-value))


def _sigmoid(value: float) -> float:
    if value == np.inf:
        return 1.0
    if value == -np.inf:
        return 0.0
    if value >= 0.0:
        z = np.exp(-value)
        return float(1.0 / (1.0 + z))
    z = np.exp(value)
    return float(z / (1.0 + z))


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


class EvolutionaryPrior:
    """Common interface implemented by the two evolutionary priors."""

    sigma_b2: float
    tau: float

    def validate(self) -> None:
        _positive(self.sigma_b2, "sigma_b2")
        tau = float(self.tau)
        if not np.isfinite(tau) or tau < 0.0:
            raise ValueError("tau must be finite and non-negative")

    def weights(self, frequencies: Array) -> Array:
        raise NotImplementedError

    def effect_variances(self, frequencies: Array) -> Array:
        return self.sigma_b2 * self.weights(frequencies)

    def weight_derivatives(self, frequencies: Array) -> Mapping[str, Array]:
        raise NotImplementedError

    def with_sigma_b2(self, sigma_b2: float) -> "EvolutionaryPrior":
        return replace(self, sigma_b2=float(sigma_b2))

    @property
    def model_name(self) -> str:
        raise NotImplementedError

    @property
    def rho2(self) -> float:
        return 1.0

    def coordinates(self, delta: float) -> Array:
        """Return unconstrained ``(log_delta, log_tau[, logit_r])`` coordinates."""

        delta_value = _positive(delta, "delta")
        tau_value = max(float(self.tau), np.finfo(np.float64).tiny)
        values = [np.log(delta_value), np.log(tau_value)]
        if self.model_name == "full":
            values.append(_logit(float(self.rho2)))
        return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class SimplifiedPrior(EvolutionaryPrior):
    """The exact ``rho^2 = 1`` evolutionary prior.

    Parameters
    ----------
    sigma_b2:
        Per-locus focal-trait effect variance on raw dosage units.
    tau:
        Non-negative ``sigma_a^2 / W_S`` aggregate.  ``tau=0`` is the
        frequency-independent boundary and is useful for tests and diagnostics.
    """

    sigma_b2: float
    tau: float

    def __post_init__(self) -> None:
        self.validate()

    @property
    def model_name(self) -> str:
        return "simplified"

    def weights(self, frequencies: Array) -> Array:
        q = allele_frequency_q(frequencies)
        a = 2.0 * float(self.tau) * q
        return 1.0 / (1.0 + a)

    def weight_derivatives(self, frequencies: Array) -> Mapping[str, Array]:
        q = allele_frequency_q(frequencies)
        a = 2.0 * float(self.tau) * q
        denominator = 1.0 + a
        return {"log_tau": -a / (denominator * denominator)}


@dataclass(frozen=True)
class FullPrior(EvolutionaryPrior):
    """The full evolutionary prior with identifiable coupling ``rho^2``.

    Only ``rho^2`` appears in the conditional variance, so the sign of ``rho``
    is not identifiable.  The public parameter accepts ``-1 <= rho <= 1`` and
    the optimizer works directly with ``r = rho^2``.
    """

    sigma_b2: float
    tau: float
    rho: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        super().validate()
        rho = float(self.rho)
        if not np.isfinite(rho) or abs(rho) > 1.0:
            raise ValueError("rho must be finite and lie in [-1, 1]")

    @property
    def model_name(self) -> str:
        return "full"

    @property
    def rho2(self) -> float:
        return float(self.rho) ** 2

    def weights(self, frequencies: Array) -> Array:
        q = allele_frequency_q(frequencies)
        a = 2.0 * float(self.tau) * q
        if self.rho2 == 1.0:
            # Preserve the nested-model identity at the floating-point level,
            # not merely within tolerance.
            return 1.0 / (1.0 + a)
        return 1.0 - self.rho2 * a / (1.0 + a)

    def weight_derivatives(self, frequencies: Array) -> Mapping[str, Array]:
        q = allele_frequency_q(frequencies)
        a = 2.0 * float(self.tau) * q
        denominator = 1.0 + a
        r = self.rho2
        if r == 1.0:
            return {"log_tau": -a / (denominator * denominator), "logit_r": np.zeros_like(a)}
        return {
            "log_tau": -r * a / (denominator * denominator),
            "logit_r": -r * (1.0 - r) * a / denominator,
        }


def prior_from_coordinates(
    model: str,
    coordinates: Array,
    *,
    sigma_b2: float = 1.0,
) -> tuple[EvolutionaryPrior, float]:
    """Construct a unit/profiled prior and ``delta`` from optimizer coordinates.

    The returned prior has the supplied ``sigma_b2`` and coordinates are
    ``(log_delta, log_tau[, logit_r])``.  Exact boundary values are supported
    by allowing infinite transformed coordinates.
    """

    values = np.asarray(coordinates, dtype=np.float64)
    expected = 2 if model == "simplified" else 3 if model == "full" else -1
    if expected < 0:
        raise ValueError("model must be 'simplified' or 'full'")
    if values.shape != (expected,):
        raise ValueError(f"{model} coordinates must have shape ({expected},)")
    if not np.isfinite(values[0]):
        raise ValueError("log_delta must be finite")
    delta = float(np.exp(values[0]))
    tau = float(np.exp(values[1])) if np.isfinite(values[1]) else 0.0
    if model == "simplified":
        return SimplifiedPrior(float(sigma_b2), tau), delta
    if values[2] >= 20.0:
        rho2 = 1.0
    elif values[2] <= -20.0:
        rho2 = 0.0
    else:
        rho2 = _sigmoid(float(values[2]))
    return FullPrior(float(sigma_b2), tau, float(np.sqrt(rho2))), delta


def prior_from_parameters(
    model: str,
    *,
    sigma_b2: float,
    tau: float,
    rho: float = 1.0,
) -> EvolutionaryPrior:
    """Create a prior from scientific parameters with an explicit model name."""

    if model == "simplified":
        return SimplifiedPrior(sigma_b2, tau)
    if model == "full":
        return FullPrior(sigma_b2, tau, rho)
    raise ValueError("model must be 'simplified' or 'full'")
