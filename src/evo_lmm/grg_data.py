"""Genotype/frequency boundary helpers for dense arrays and GRGL graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .grapp_backend import allele_frequencies as _grapp_allele_frequencies
from .grapp_backend import wrap_grg


@dataclass(frozen=True)
class VariantData:
    """Cached per-variant information used by an evolutionary operator.

    ``frequencies`` are sample allele frequencies and ``local_idx`` are the
    mutation identifiers in the source chromosome.  ``raw_norm2`` and
    ``centered_norm2`` are optional because GRGL operators can obtain them via
    graph traversals when needed.
    """

    frequencies: np.ndarray
    local_idx: np.ndarray
    raw_norm2: np.ndarray | None = None
    centered_norm2: np.ndarray | None = None

    def __post_init__(self) -> None:
        frequencies = np.asarray(self.frequencies, dtype=np.float64)
        local_idx = np.asarray(self.local_idx, dtype=np.int64)
        if frequencies.ndim != 1 or local_idx.ndim != 1:
            raise ValueError("variant arrays must be one-dimensional")
        if frequencies.shape != local_idx.shape:
            raise ValueError("frequencies and local_idx must have equal length")
        if not np.all(np.isfinite(frequencies)) or np.any(
            (frequencies < 0.0) | (frequencies > 1.0)
        ):
            raise ValueError("frequencies must be finite and lie in [0, 1]")
        object.__setattr__(self, "frequencies", np.ascontiguousarray(frequencies))
        object.__setattr__(self, "local_idx", np.ascontiguousarray(local_idx))
        for name in ("raw_norm2", "centered_norm2"):
            value = getattr(self, name)
            if value is None:
                continue
            arr = np.asarray(value, dtype=np.float64)
            if arr.shape != frequencies.shape:
                raise ValueError(f"{name} must match frequencies")
            if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, np.ascontiguousarray(arr))

    @property
    def q(self) -> np.ndarray:
        return self.frequencies * (1.0 - self.frequencies)

    @property
    def n_variants(self) -> int:
        return int(self.frequencies.size)


def dense_variant_data(
    genotypes: np.ndarray,
    frequencies: np.ndarray | None = None,
    *,
    local_idx: Sequence[int] | None = None,
) -> VariantData:
    """Build variant metadata from an ``(N, M)`` raw dosage matrix.

    If frequencies are omitted, they are calculated as the mean dosage divided
    by ploidy two.  This default is deliberately explicit in the docstring and
    is only intended for diploid dense test/simulation inputs.
    """

    x = np.asarray(genotypes, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("genotypes must have shape (n_individuals, n_variants)")
    if not np.all(np.isfinite(x)):
        raise ValueError("genotypes must be finite raw dosages")
    if frequencies is None:
        frequencies = np.mean(x, axis=0) / 2.0
    idx = np.arange(x.shape[1], dtype=np.int64) if local_idx is None else local_idx
    raw_norm2 = np.sum(x * x, axis=0)
    centered = x - 2.0 * np.asarray(frequencies, dtype=np.float64)[None, :]
    centered_norm2 = np.sum(centered * centered, axis=0)
    return VariantData(frequencies, idx, raw_norm2, centered_norm2)


def _haplotype_filter(grg: Any, sample_filter: Sequence[int] | None) -> list[int] | None:
    if sample_filter is None:
        return None
    wrapped = wrap_grg(grg)
    ploidy = int(wrapped.ploidy)
    if ploidy <= 0:
        raise ValueError("GRG ploidy must be positive")
    values = [int(v) for v in sample_filter]
    if len(set(values)) != len(values):
        raise ValueError("sample_filter contains duplicate individuals")
    return [ploidy * individual + hap for individual in values for hap in range(ploidy)]


def sample_allele_frequencies(
    grg: Any,
    *,
    sample_filter: Sequence[int] | None = None,
    adjust_missing: bool = True,
) -> np.ndarray:
    """Extract sample allele frequencies from a GRGL-backed chromosome.

    ``sample_filter`` is expressed in individual indices, matching GRAPP's raw
    genotype operators.  Missing alleles are excluded from the denominator by
    default, so the result remains a sample frequency rather than an imputed
    dosage frequency.
    """

    hap_filter = _haplotype_filter(grg, sample_filter)
    values = _grapp_allele_frequencies(
        grg, sample_filter=hap_filter, adjust_missing=adjust_missing
    )
    return np.asarray(values, dtype=np.float64)


def grg_variant_data(
    grg: Any,
    *,
    frequencies: np.ndarray | None = None,
    sample_filter: Sequence[int] | None = None,
    mutation_filter: Sequence[int] | None = None,
) -> VariantData:
    """Build cached metadata for one GRG chromosome.

    GRGL retains mutation identifiers even when a mutation filter is supplied;
    the returned ``local_idx`` preserves those identifiers for ``test_column``.
    """

    wrapped = wrap_grg(grg)
    if frequencies is None:
        frequencies = sample_allele_frequencies(grg, sample_filter=sample_filter)
    values = np.asarray(frequencies, dtype=np.float64)
    if mutation_filter is None:
        idx = np.arange(int(wrapped.num_mutations), dtype=np.int64)
    else:
        idx = np.asarray(mutation_filter, dtype=np.int64)
        if idx.ndim != 1 or np.any(idx < 0) or np.any(idx >= wrapped.num_mutations):
            raise ValueError("mutation_filter contains an invalid mutation id")
        values = values[idx]
    if values.shape != idx.shape:
        raise ValueError("frequency count does not match GRG variants")
    return VariantData(values, idx)


def validate_sample_filter(sample_filter: Sequence[int] | None, n: int) -> np.ndarray | None:
    """Validate and return an integer individual mask/index array."""

    if sample_filter is None:
        return None
    values = np.asarray(sample_filter, dtype=np.int64)
    if values.ndim != 1 or np.any(values < 0) or np.any(values >= int(n)):
        raise ValueError("sample_filter contains an invalid individual index")
    if np.unique(values).size != values.size:
        raise ValueError("sample_filter contains duplicate individual indices")
    return values

