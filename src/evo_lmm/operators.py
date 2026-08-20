"""Projected raw-dosage and evolutionary-kernel operators.

The dense path is used as an oracle and for small analyses.  GRG chromosomes
use GRAPP's raw ``X`` LinearOperator, so no individual-by-variant matrix is
created by production code.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .grapp_backend import raw_operator_class, wrap_grg
from .grg_data import VariantData, dense_variant_data, grg_variant_data
from .priors import EvolutionaryPrior, FullPrior, SimplifiedPrior, prior_from_coordinates, prior_from_parameters


# GRAPP excludes columns whose projected raw-dosage norm falls below this
# threshold; keeping the same value preserves eligibility parity.
_MIN_PROJECTED_NORM2 = 0.1


def _as_chromosomes(chromosomes: Any) -> list[tuple[Any, Any]]:
    if isinstance(chromosomes, np.ndarray):
        return [(0, chromosomes)]
    if isinstance(chromosomes, Mapping):
        return list(chromosomes.items())
    try:
        values = list(chromosomes)
    except TypeError:
        return [(0, chromosomes)]
    if not values:
        raise ValueError("at least one chromosome is required")
    if all(isinstance(value, tuple) and len(value) == 2 for value in values):
        return [(value[0], value[1]) for value in values]
    return list(enumerate(values))


def _frequency_for(
    frequencies: Any,
    label: Any,
    index: int,
    n_variants: int,
    labels: Sequence[Any],
) -> np.ndarray | None:
    if frequencies is None:
        return None
    if isinstance(frequencies, Mapping):
        value = frequencies[label]
    elif isinstance(frequencies, np.ndarray):
        value = frequencies
    else:
        values = list(frequencies)
        try:
            one_dimensional = np.asarray(values).ndim == 1
        except ValueError:
            one_dimensional = False
        if one_dimensional and len(values) == n_variants and len(labels) == 1:
            value = values
        elif len(values) == len(labels):
            value = values[index]
        else:
            value = np.concatenate([np.asarray(v) for v in values])
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 1 or value.size != n_variants:
        raise ValueError(f"frequency count for chromosome {label!r} does not match variants")
    return value


def _orthonormal_covariates(covariates: np.ndarray | None, n: int) -> np.ndarray:
    if covariates is None:
        values = np.ones((n, 1), dtype=np.float64)
    else:
        values = np.asarray(covariates, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != n or values.shape[1] == 0:
            raise ValueError("covariates must have shape (n_individuals, n_covariates)")
        # The public boundary requires an intercept.  Add one when callers pass
        # only non-intercept covariates; do not duplicate an existing constant.
        if not np.allclose(values[:, 0], values[0, 0]) or abs(values[0, 0]) < 1e-14:
            values = np.column_stack((np.ones(n), values))
    if not np.all(np.isfinite(values)):
        raise ValueError("covariates must be finite")
    q, r = np.linalg.qr(values, mode="reduced")
    diagonal = np.abs(np.diag(r))
    if diagonal.size == 0 or diagonal[0] <= 1e-14:
        raise ValueError("covariates have no non-zero rank")
    rank = int(np.count_nonzero(diagonal > diagonal.max() * 1e-10))
    return np.ascontiguousarray(q[:, :rank], dtype=np.float64)


@dataclass(frozen=True)
class TestVariantStats:
    """Per-variant tested-genotype statistics for one chromosome block.

    All quantities are computed after the covariate projection ``P_C``.  The
    ``test_scale`` is the BOLT normalisation ``sqrt(centered_norm2 / (n - 1))``
    applied to raw diploid dosage columns, so ``norm_scale = 1 / test_scale``
    converts a raw-dosage quantity into BOLT-normalised units.  Nothing here
    depends on the fitted evolutionary prior.

    Attributes
    ----------
    projected_norm2:
        ``||P_C x_j||^2`` in raw diploid dosage units.
    std_projected_norm2:
        ``||P_C x_j||^2`` in BOLT-normalised units, i.e.
        ``projected_norm2 * norm_scale**2``.
    model_mask:
        Variants eligible for association output; monomorphic and
        covariate-collinear columns are excluded exactly as in GRAPP.
    """

    chrom: Any
    local_idx: np.ndarray
    centered_norm2: np.ndarray
    projected_norm2: np.ndarray
    test_scale: np.ndarray
    norm_scale: np.ndarray
    std_projected_norm2: np.ndarray
    model_mask: np.ndarray

    @property
    def n_variants(self) -> int:
        return int(self.local_idx.size)


@dataclass
class _Chromosome:
    label: Any
    source: Any
    data: VariantData
    dense: np.ndarray | None
    raw: Any
    local_to_pos: dict[int, int]
    test_scale: np.ndarray
    projected_norm2: np.ndarray

    @property
    def n_variants(self) -> int:
        return self.data.n_variants


class EvolutionaryLmmOps:
    """Matrix-free projected raw-dosage operators for evolutionary LMMs.

    Parameters
    ----------
    chromosomes:
        An ``(N, M)`` dense dosage matrix, a mapping of chromosome labels to
        matrices/GRGs, or a sequence of matrices/``(label, source)`` pairs.
    frequencies:
        Sample allele frequencies, supplied as one vector, one vector per
        chromosome, or a mapping.  GRG inputs may omit this and extract them
        through GRAPP's frequency traversal.
    covariates:
        Fixed-effect design.  An intercept is added when absent and the basis
        is QR-orthonormalised once.
    model:
        Coordinate interpretation for transformed ``phi`` arrays.
    """

    def __init__(
        self,
        chromosomes: Any,
        frequencies: Any = None,
        covariates: np.ndarray | None = None,
        *,
        sample_filter: Sequence[int] | None = None,
        model: str = "simplified",
        mutation_filter: Mapping[Any, Sequence[int]] | None = None,
    ) -> None:
        if model not in ("simplified", "full"):
            raise ValueError("model must be 'simplified' or 'full'")
        pairs = _as_chromosomes(chromosomes)
        labels = [label for label, _ in pairs]
        self.model_name = model
        self.sample_filter = None if sample_filter is None else np.asarray(sample_filter, dtype=np.int64)
        all_dense = all(isinstance(source, np.ndarray) or hasattr(source, "__array__") for _, source in pairs)
        if all_dense and self.sample_filter is not None:
            if np.any(self.sample_filter < 0):
                raise ValueError("sample_filter contains a negative index")
            if covariates is not None:
                covariates = np.asarray(covariates)[self.sample_filter]
        self._chromosomes: list[_Chromosome] = []

        for index, (label, source) in enumerate(pairs):
            if isinstance(source, np.ndarray) or hasattr(source, "__array__"):
                dense = np.asarray(source, dtype=np.float64)
                if dense.ndim != 2 or not np.all(np.isfinite(dense)):
                    raise ValueError("dense chromosome genotypes must be finite matrices")
                if self.sample_filter is not None:
                    if np.any(self.sample_filter >= dense.shape[0]):
                        raise ValueError("sample_filter contains an invalid dense individual index")
                    dense = dense[self.sample_filter]
                freq = _frequency_for(frequencies, label, index, dense.shape[1], labels)
                data = dense_variant_data(dense, freq)
                raw = dense
                n = dense.shape[0]
            else:
                dense = None
                wrapped = wrap_grg(source)
                selected = None if mutation_filter is None else mutation_filter.get(label)
                freq = _frequency_for(frequencies, label, index, int(wrapped.num_mutations), labels)
                data = grg_variant_data(
                    source,
                    frequencies=freq,
                    sample_filter=None if self.sample_filter is None else self.sample_filter.tolist(),
                    mutation_filter=selected,
                )
                # GRAPP's raw operator accepts individual sample filters and
                # returns an (individuals, mutations) LinearOperator.
                cls = raw_operator_class(source)
                import pygrgl

                full_freq = (
                    data.frequencies
                    if selected is None
                    else np.asarray(freq, dtype=np.float64)
                )
                kwargs: dict[str, Any] = {
                    "dtype": np.float64,
                    "mutation_filter": None if selected is None else data.local_idx.tolist(),
                    "sample_filter": None if self.sample_filter is None else self.sample_filter.tolist(),
                }
                if getattr(wrapped, "has_missing_data", False):
                    kwargs["miss_values"] = int(wrapped.ploidy) * full_freq
                raw = cls(source, pygrgl.TraversalDirection.UP, **kwargs)
                n = int(raw.shape[0])

            if self._chromosomes and n != self.n:
                raise ValueError("all chromosomes must have the same individual count")
            raw_norm2, centered_norm2, projected_norm2 = self._norms_for(data, dense, raw, n)
            test_scale = np.ones(data.n_variants, dtype=np.float64)
            positive = centered_norm2 > np.finfo(np.float64).eps
            test_scale[positive] = np.sqrt(centered_norm2[positive] / max(n - 1, 1))
            test_scale[~positive] = 1.0
            raw_norm2 = raw_norm2 if raw_norm2 is not None else data.raw_norm2
            if raw_norm2 is None:
                ploidy = float(getattr(getattr(raw, "grg", None), "ploidy", 2))
                raw_norm2 = centered_norm2 + n * ploidy * data.frequencies * data.frequencies
            data = VariantData(
                data.frequencies,
                data.local_idx,
                raw_norm2,
                centered_norm2,
            )
            self._chromosomes.append(
                _Chromosome(
                    label,
                    source,
                    data,
                    dense,
                    raw,
                    {int(v): pos for pos, v in enumerate(data.local_idx)},
                    test_scale,
                    projected_norm2,
                )
            )
        self._basis = _orthonormal_covariates(covariates, self.n)
        # The covariate basis is fixed after construction, so the projected
        # test-column norms only need one traversal per chromosome.
        self._projected_norms_ready = False

    @classmethod
    def from_dense(
        cls,
        genotypes: np.ndarray | Sequence[np.ndarray],
        frequencies: np.ndarray | Sequence[np.ndarray],
        covariates: np.ndarray | None = None,
        *,
        chrom_labels: Sequence[Any] | None = None,
        model: str = "simplified",
    ) -> "EvolutionaryLmmOps":
        """Convenience constructor for dense tests and small simulations."""

        if isinstance(genotypes, np.ndarray):
            return cls(genotypes, frequencies, covariates, model=model)
        matrices = list(genotypes)
        labels = list(range(len(matrices))) if chrom_labels is None else list(chrom_labels)
        if len(labels) != len(matrices):
            raise ValueError("chrom_labels and genotypes must have equal length")
        return cls(
            list(zip(labels, matrices)), frequencies, covariates, model=model
        )

    @staticmethod
    def _norms_for(data: VariantData, dense: np.ndarray | None, raw: Any, n: int) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
        if dense is not None:
            centered = dense - 2.0 * data.frequencies[None, :]
            raw_norm2 = np.sum(dense * dense, axis=0)
            centered_norm2 = np.sum(centered * centered, axis=0)
            # Projection is filled after the basis exists; initialize with the
            # centered norm and refresh in __init__ below.
            return raw_norm2, centered_norm2, centered_norm2.copy()
        raw_norm2 = data.raw_norm2
        if data.centered_norm2 is not None:
            centered_norm2 = data.centered_norm2.copy()
        else:
            ones = np.ones(n, dtype=np.float64)
            sum_x = np.asarray(raw.rmatvec(ones), dtype=np.float64).reshape(-1)
            mean = int(getattr(raw, "grg", None).ploidy) * data.frequencies if hasattr(getattr(raw, "grg", None), "ploidy") else 2.0 * data.frequencies
            # A GRGL xtx traversal gives the diagonal without allocating N*M.
            wrapped = getattr(raw, "grg", None)
            if wrapped is not None:
                mask = np.ones(int(wrapped.num_individuals), dtype=np.float64)
                if getattr(raw.filter, "sample_filter", None) is not None:
                    mask[:] = 0.0
                    mask[np.asarray(raw.filter.sample_filter, dtype=np.int64)] = 1.0
                try:
                    value = wrapped.matmul(
                        mask.reshape(1, -1),
                        __import__("pygrgl").TraversalDirection.UP,
                        by_individual=True,
                        init="xtx",
                    )
                    raw_norm2 = np.asarray(value, dtype=np.float64).reshape(-1)
                    mutation_filter = getattr(getattr(raw, "filter", None), "mutation_filter", None)
                    if mutation_filter is not None:
                        raw_norm2 = raw_norm2[np.asarray(mutation_filter, dtype=np.int64)]
                    mean = float(wrapped.ploidy) * data.frequencies
                    centered_norm2 = np.maximum(
                        raw_norm2 - 2.0 * mean * sum_x + n * mean * mean, 0.0
                    )
                except RuntimeError:
                    # Some mutable/minimal GRGs do not carry coalescent counts,
                    # so GRAPP's xtx initialiser is unavailable.  Query one raw
                    # column at a time instead.  This is slower, but remains
                    # O(N + M) memory and gives exact test normalisation.
                    raw_norm2 = np.empty(data.n_variants, dtype=np.float64)
                    for column in range(data.n_variants):
                        unit = np.zeros(data.n_variants, dtype=np.float64)
                        unit[column] = 1.0
                        value = np.asarray(raw.matvec(unit), dtype=np.float64).reshape(-1)
                        raw_norm2[column] = float(value @ value)
                    mean = float(wrapped.ploidy) * data.frequencies
                    centered_norm2 = np.maximum(
                        raw_norm2 - 2.0 * mean * sum_x + n * mean * mean, 0.0
                    )
            else:
                centered_norm2 = np.ones(data.n_variants)
        # The projected norm is refreshed by _refresh_projected_norms once the
        # covariate basis is known.  Returning centered norms is a safe default.
        return raw_norm2, centered_norm2, centered_norm2.copy()

    @property
    def n(self) -> int:
        if not self._chromosomes:
            return 0
        source = self._chromosomes[0]
        return int(source.dense.shape[0] if source.dense is not None else source.raw.shape[0])

    @property
    def rank(self) -> int:
        return int(self._basis.shape[1])

    @property
    def dim(self) -> int:
        return self.n - self.rank

    @property
    def basis(self) -> np.ndarray:
        return self._basis.copy()

    @property
    def chroms(self) -> list[Any]:
        return [chrom.label for chrom in self._chromosomes]

    @property
    def frequencies(self) -> np.ndarray:
        return np.concatenate([chrom.data.frequencies for chrom in self._chromosomes])

    @property
    def n_variants(self) -> int:
        return int(sum(chrom.n_variants for chrom in self._chromosomes))

    @property
    def variant_data(self) -> list[VariantData]:
        return [chrom.data for chrom in self._chromosomes]

    def _refresh_projected_norms(self) -> None:
        for chrom in self._chromosomes:
            projected = np.empty(chrom.n_variants, dtype=np.float64)
            if chrom.dense is not None:
                values = chrom.dense - self._basis @ (self._basis.T @ chrom.dense)
                projected[:] = np.sum(values * values, axis=0)
            else:
                coefficients = np.asarray(chrom.raw.rmatmat(self._basis), dtype=np.float64)
                raw_norm = chrom.data.raw_norm2
                if raw_norm is None:
                    raw_norm = chrom.data.centered_norm2
                projected[:] = np.maximum(raw_norm - np.sum(coefficients * coefficients, axis=1), 0.0)
            chrom.projected_norm2[:] = projected

    def _ensure_projected_norms(self) -> None:
        if not self._projected_norms_ready:
            self._refresh_projected_norms()
            self._projected_norms_ready = True

    def test_stats(self, chrom: Any) -> TestVariantStats:
        """Return prior-independent tested-genotype statistics for a chromosome.

        These are the quantities the BOLT-style calibration and association
        formulas need: the projected raw-dosage norms, the BOLT normalisation,
        and the eligibility mask.  They depend only on the genotypes and the
        covariate basis, never on the fitted evolutionary prior.
        """

        item = self._get_chrom(chrom)
        self._ensure_projected_norms()
        centered_norm2 = np.asarray(item.data.centered_norm2, dtype=np.float64)
        projected_norm2 = np.asarray(item.projected_norm2, dtype=np.float64).copy()
        test_scale = np.asarray(item.test_scale, dtype=np.float64).copy()
        usable = (centered_norm2 > 0.0) & (test_scale > 0.0)
        norm_scale = np.zeros_like(test_scale)
        norm_scale[usable] = 1.0 / test_scale[usable]
        std_projected_norm2 = projected_norm2 * norm_scale * norm_scale
        model_mask = (
            usable
            & (projected_norm2 >= _MIN_PROJECTED_NORM2)
            & (std_projected_norm2 > 0.0)
        )
        return TestVariantStats(
            chrom=item.label,
            local_idx=item.data.local_idx.copy(),
            centered_norm2=centered_norm2.copy(),
            projected_norm2=projected_norm2,
            test_scale=test_scale,
            norm_scale=norm_scale,
            std_projected_norm2=std_projected_norm2,
            model_mask=model_mask,
        )

    def project(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape[0] != self.n:
            raise ValueError(f"values have {arr.shape[0]} rows; expected {self.n}")
        if arr.ndim == 1:
            return arr - self._basis @ (self._basis.T @ arr)
        return arr - self._basis @ (self._basis.T @ arr)

    def _prior(self, theta: Any) -> EvolutionaryPrior:
        if isinstance(theta, EvolutionaryPrior):
            return theta
        if isinstance(theta, Mapping):
            if "prior" in theta:
                return self._prior(theta["prior"])
            if "log_tau" in theta:
                coordinates = [float(theta.get("log_delta", 0.0)), float(theta["log_tau"])]
                if self.model_name == "full":
                    coordinates.append(float(theta.get("logit_r", 0.0)))
                return prior_from_coordinates(self.model_name, np.asarray(coordinates, dtype=np.float64))[0]
            sigma = float(theta.get("sigma_b2", 1.0))
            tau = float(theta["tau"])
            rho = float(theta.get("rho", 1.0))
            return prior_from_parameters(self.model_name, sigma_b2=sigma, tau=tau, rho=rho)
        return prior_from_coordinates(self.model_name, np.asarray(theta, dtype=np.float64))[0]

    def _delta(self, phi: Any) -> float:
        if isinstance(phi, Mapping):
            if "delta" in phi:
                value = float(phi["delta"])
            elif "log_delta" in phi:
                value = float(np.exp(phi["log_delta"]))
            else:
                raise ValueError("shape mapping requires delta or log_delta")
            if value <= 0.0 or not np.isfinite(value):
                raise ValueError("delta must be finite and positive")
            return value
        if isinstance(phi, EvolutionaryPrior):
            return 1.0
        values = np.asarray(phi, dtype=np.float64)
        return float(np.exp(values[0]))

    def _chrom_weights(self, prior: EvolutionaryPrior) -> list[np.ndarray]:
        return [prior.weights(chrom.data.frequencies) for chrom in self._chromosomes]

    def _split_global(self, values: np.ndarray) -> list[np.ndarray]:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 1 or arr.size != self.n_variants:
            raise ValueError(f"expected a vector of {self.n_variants} variant values")
        result: list[np.ndarray] = []
        start = 0
        for chrom in self._chromosomes:
            result.append(arr[start : start + chrom.n_variants])
            start += chrom.n_variants
        return result

    def _sum_x(self, vectors: list[np.ndarray], exclude_chrom: Any = None) -> np.ndarray:
        result = np.zeros(self.n, dtype=np.float64)
        for chrom, vector in zip(self._chromosomes, vectors):
            if chrom.label == exclude_chrom:
                continue
            result += self._raw_matvec(chrom, vector)
        return result

    @staticmethod
    def _raw_matvec(chrom: _Chromosome, vector: np.ndarray) -> np.ndarray:
        if chrom.dense is not None:
            return np.asarray(chrom.dense @ vector, dtype=np.float64).reshape(-1)
        return np.asarray(chrom.raw.matvec(vector), dtype=np.float64).reshape(-1)

    @staticmethod
    def _raw_rmatvec(chrom: _Chromosome, vector: np.ndarray) -> np.ndarray:
        if chrom.dense is not None:
            return np.asarray(chrom.dense.T @ vector, dtype=np.float64).reshape(-1)
        return np.asarray(chrom.raw.rmatvec(vector), dtype=np.float64).reshape(-1)

    @staticmethod
    def _raw_matmat(chrom: _Chromosome, values: np.ndarray) -> np.ndarray:
        if chrom.dense is not None:
            return np.asarray(chrom.dense @ values, dtype=np.float64)
        return np.asarray(chrom.raw.matmat(values), dtype=np.float64)

    @staticmethod
    def _raw_rmatmat(chrom: _Chromosome, values: np.ndarray) -> np.ndarray:
        if chrom.dense is not None:
            return np.asarray(chrom.dense.T @ values, dtype=np.float64)
        return np.asarray(chrom.raw.rmatmat(values), dtype=np.float64)

    def apply_model_x(
        self,
        weights: np.ndarray | Mapping[Any, np.ndarray] | EvolutionaryPrior,
        theta: Any = None,
        exclude_chrom: Any = None,
    ) -> np.ndarray:
        """Apply ``P_C X diag(weights)`` to variant coefficients.

        ``weights`` is normally a vector of model coefficients.  When ``theta``
        is a prior, it applies the model operator ``P_C X diag(sqrt(w(theta)))``
        to those coefficients.  A raw coefficient vector or chromosome mapping
        may also be supplied without ``theta``.  This method deliberately
        contains no sample-standardisation or ``1/M`` normalization.
        """

        if isinstance(weights, EvolutionaryPrior):
            prior = weights
            vectors = [np.sqrt(value) for value in self._chrom_weights(prior)]
        elif theta is not None and isinstance(theta, EvolutionaryPrior):
            if isinstance(weights, Mapping):
                coefficient_blocks = [np.asarray(weights[chrom.label], dtype=np.float64) for chrom in self._chromosomes]
            else:
                coefficient_blocks = self._split_global(np.asarray(weights, dtype=np.float64))
            vectors = [coefficient * np.sqrt(prior_weight) for coefficient, prior_weight in zip(coefficient_blocks, self._chrom_weights(theta))]
        elif isinstance(weights, Mapping):
            vectors = [np.asarray(weights[chrom.label], dtype=np.float64) for chrom in self._chromosomes]
        else:
            vectors = self._split_global(np.asarray(weights, dtype=np.float64))
        return self.project(self._sum_x(vectors, exclude_chrom))

    def model_scores(self, vector: np.ndarray, theta: Any = None, exclude_chrom: Any = None) -> np.ndarray:
        """Return model scores, optionally for ``B_theta^T = diag(sqrt(w))X^T P_C``."""

        projected = self.project(vector)
        result = []
        for chrom in self._chromosomes:
            if chrom.label == exclude_chrom:
                continue
            result.append(self._raw_rmatvec(chrom, projected))
        if not result:
            return np.empty(0, dtype=np.float64)
        output = np.concatenate(result)
        if theta is None:
            return output
        prior = self._prior(theta)
        blocks = self._split_global(output) if exclude_chrom is None else []
        if exclude_chrom is not None:
            cursor = 0
            for chrom in self._chromosomes:
                if chrom.label == exclude_chrom:
                    blocks.append(np.zeros(chrom.n_variants, dtype=np.float64))
                else:
                    blocks.append(output[cursor : cursor + chrom.n_variants])
                    cursor += chrom.n_variants
        weighted = [block * np.sqrt(prior.weights(chrom.data.frequencies)) for block, chrom in zip(blocks, self._chromosomes)]
        return np.concatenate([block for chrom, block in zip(self._chromosomes, weighted) if chrom.label != exclude_chrom])

    def apply_k(self, vector: np.ndarray, theta: Any, exclude_chrom: Any = None) -> np.ndarray:
        """Apply ``K = P_C X diag(w(theta)) X^T P_C``."""

        prior = self._prior(theta)
        scores = self.model_scores(vector, exclude_chrom=exclude_chrom)
        if exclude_chrom is None:
            weighted = self._split_global(scores)
        else:
            weighted = []
            cursor = 0
            for chrom in self._chromosomes:
                if chrom.label == exclude_chrom:
                    weighted.append(np.zeros(chrom.n_variants))
                else:
                    weighted.append(scores[cursor : cursor + chrom.n_variants])
                    cursor += chrom.n_variants
        weighted = [w * prior.weights(chrom.data.frequencies) for w, chrom in zip(weighted, self._chromosomes)]
        return self.project(self._sum_x(weighted, exclude_chrom))

    def apply_dk(self, vector: np.ndarray, theta: Any, parameter: str, exclude_chrom: Any = None) -> np.ndarray:
        """Apply a first derivative of ``K`` in a transformed shape coordinate."""

        prior = self._prior(theta)
        derivatives = prior.weight_derivatives(
            np.concatenate([chrom.data.frequencies for chrom in self._chromosomes])
        )
        if parameter not in derivatives:
            raise KeyError(f"prior has no derivative parameter {parameter!r}")
        scores = self.model_scores(vector, exclude_chrom=exclude_chrom)
        if exclude_chrom is None:
            score_blocks = self._split_global(scores)
        else:
            score_blocks = []
            cursor = 0
            for chrom in self._chromosomes:
                if chrom.label == exclude_chrom:
                    score_blocks.append(np.zeros(chrom.n_variants))
                else:
                    score_blocks.append(scores[cursor : cursor + chrom.n_variants])
                    cursor += chrom.n_variants
        derivative = self._split_global(derivatives[parameter])
        return self.project(
            self._sum_x(
                [score * change for score, change in zip(score_blocks, derivative)],
                exclude_chrom,
            )
        )

    def apply_h(self, vector: np.ndarray, phi: Any, exclude_chrom: Any = None) -> np.ndarray:
        """Apply the projected shape matrix ``H = K + delta I``."""

        prior = self._prior(phi)
        projected = self.project(vector)
        return self.apply_k(projected, prior, exclude_chrom) + self._delta(phi) * projected

    def apply_dh(self, vector: np.ndarray, phi: Any, parameter: str, exclude_chrom: Any = None) -> np.ndarray:
        """Apply a first derivative of ``H`` in ``log_delta``, ``log_tau``, or ``logit_r``."""

        projected = self.project(vector)
        if parameter == "log_delta":
            return self._delta(phi) * projected
        return self.apply_dk(projected, self._prior(phi), parameter, exclude_chrom)

    def apply_dh_matmat(
        self,
        values: np.ndarray,
        phi: Any,
        parameter: str,
        exclude_chrom: Any = None,
    ) -> np.ndarray:
        """Apply a first derivative of ``H`` to several columns at once.

        This is the matrix-RHS counterpart of :meth:`apply_dh`. Keeping the
        columns batched is important for GRG inputs: one ``matmat`` traversal
        replaces one traversal per stochastic trace probe.
        """

        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != self.n:
            raise ValueError("values must have shape (n, k)")
        projected = self.project(matrix)
        if parameter == "log_delta":
            return self._delta(phi) * projected

        prior = self._prior(phi)
        derivatives = prior.weight_derivatives(self.frequencies)
        if parameter not in derivatives:
            raise KeyError(f"prior has no derivative parameter {parameter!r}")
        derivative_blocks = self._split_global(derivatives[parameter])
        result = np.zeros_like(projected)
        for chrom, derivative in zip(self._chromosomes, derivative_blocks):
            if chrom.label == exclude_chrom:
                continue
            scores = self._raw_rmatmat(chrom, projected)
            result += self._raw_matmat(chrom, scores * derivative[:, None])
        return self.project(result)

    def solve_ph(
        self,
        rhs_columns: np.ndarray,
        phi: Any,
        exclude_chrom: Any = None,
        *,
        tol: float = 1e-9,
        max_iter: int | None = None,
        initial: np.ndarray | None = None,
        stats: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Solve projected ``H z = P_C rhs`` for one or many right-hand sides.

        ``initial`` is an optional warm start. Each column is validated against
        a zero start independently; a poor or invalid cached column is reset
        without affecting the other right-hand sides. If ``stats`` is passed,
        it is populated with aggregate iteration and warm-start diagnostics.
        """

        rhs = np.asarray(rhs_columns, dtype=np.float64)
        was_vector = rhs.ndim == 1
        if was_vector:
            rhs = rhs[:, None]
        if rhs.ndim != 2 or rhs.shape[0] != self.n:
            raise ValueError("rhs_columns must have shape (n, k)")
        b = self.project(rhs)
        warm_requested = initial is not None
        x = np.zeros_like(b)
        warm_used = np.zeros(b.shape[1], dtype=bool)
        warm_rejected = np.zeros(b.shape[1], dtype=bool)
        if initial is not None:
            candidate = np.asarray(initial, dtype=np.float64)
            if candidate.shape != b.shape:
                warm_rejected[:] = True
                candidate = None
            if candidate is None:
                initial = None
            else:
                finite = np.all(np.isfinite(candidate), axis=0)
                if np.any(finite):
                    projected_candidate = self.project(candidate[:, finite])
                    x[:, finite] = projected_candidate
                    warm_used[finite] = True
                warm_rejected[~finite] = True
        residual = b - self._apply_h_matmat(x, phi, exclude_chrom)
        residual = self.project(residual)
        if initial is not None and np.any(warm_used):
            zero_residual_norm2 = np.sum(b * b, axis=0)
            candidate_residual_norm2 = np.sum(residual * residual, axis=0)
            worse = warm_used & (candidate_residual_norm2 > zero_residual_norm2)
            if np.any(worse):
                x[:, worse] = 0.0
                warm_used[worse] = False
                warm_rejected[worse] = True
                residual[:, worse] = b[:, worse]
        initial_residual_norm = float(np.sqrt(np.max(np.sum(residual * residual, axis=0)))) if residual.size else 0.0
        directions = residual.copy()
        residual_norm2 = np.sum(residual * residual, axis=0)
        target = np.maximum(residual_norm2, 1.0) * float(tol) ** 2
        active = residual_norm2 > target
        if max_iter is None:
            max_iter = max(50, 4 * self.n)
        iterations = 0
        for _ in range(int(max_iter)):
            iterations += 1
            if not np.any(active):
                break
            active_indices = np.flatnonzero(active)
            hp = self._apply_h_matmat(directions[:, active_indices], phi, exclude_chrom)
            rr = residual_norm2[active_indices]
            denom = np.sum(directions[:, active_indices] * hp, axis=0)
            safe = np.abs(denom) > np.finfo(np.float64).tiny
            alpha = np.zeros_like(rr)
            alpha[safe] = rr[safe] / denom[safe]
            x[:, active_indices] += directions[:, active_indices] * alpha
            residual[:, active_indices] -= hp * alpha
            residual[:, active_indices] = self.project(residual[:, active_indices])
            new_norm2 = np.sum(residual[:, active_indices] ** 2, axis=0)
            beta = np.zeros_like(new_norm2)
            beta[rr > 0.0] = new_norm2[rr > 0.0] / rr[rr > 0.0]
            directions[:, active_indices] = residual[:, active_indices] + directions[:, active_indices] * beta
            residual_norm2[active_indices] = new_norm2
            active[active_indices] = new_norm2 > target[active_indices]
            active[active_indices[~safe]] = False
        if np.any(active):
            max_rel = float(np.sqrt(np.max(residual_norm2[active]) / np.maximum(np.max(np.sum(b * b, axis=0)), 1e-300)))
            raise np.linalg.LinAlgError(f"projected CG did not converge; relative residual {max_rel:.3g}")
        if stats is not None:
            stats["iterations"] = int(iterations)
            stats["active_columns"] = int(rhs.shape[1])
            stats["warm_requested"] = bool(warm_requested)
            stats["warm_used"] = int(np.count_nonzero(warm_used))
            stats["warm_rejected"] = int(np.count_nonzero(warm_rejected))
            stats["initial_residual_norm"] = initial_residual_norm
            stats["final_residual_norm"] = float(np.sqrt(np.max(residual_norm2))) if residual_norm2.size else 0.0
        return x[:, 0] if was_vector else x

    def _apply_h_matmat(self, values: np.ndarray, phi: Any, exclude_chrom: Any) -> np.ndarray:
        projected = self.project(values)
        prior = self._prior(phi)
        result = self._delta(phi) * projected
        for chrom in self._chromosomes:
            if chrom.label == exclude_chrom:
                continue
            scores = self._raw_rmatmat(chrom, projected)
            weighted = scores * prior.weights(chrom.data.frequencies)[:, None]
            result += self._raw_matmat(chrom, weighted)
        return self.project(result)

    def test_scores(self, chrom: Any, vector: np.ndarray) -> np.ndarray:
        """Return BOLT-normalised test-genotype scores for one chromosome."""

        item = self._get_chrom(chrom)
        scores = self._raw_rmatvec(item, self.project(vector))
        return scores / item.test_scale

    def test_column(self, chrom: Any, local_idx: int) -> np.ndarray:
        """Return one projected, BOLT-normalised test-genotype column."""

        item = self._get_chrom(chrom)
        try:
            position = item.local_to_pos[int(local_idx)]
        except KeyError as exc:
            raise KeyError(f"unknown mutation {local_idx!r} on chromosome {chrom!r}") from exc
        coefficients = np.zeros(item.n_variants, dtype=np.float64)
        coefficients[position] = 1.0 / item.test_scale[position]
        if item.dense is not None:
            return self.project(item.dense @ coefficients)
        return self.project(self._raw_matvec(item, coefficients))

    def chromosome_frequencies(self, chrom: Any) -> np.ndarray:
        """Return sample allele frequencies for one chromosome in operator order."""

        return self._get_chrom(chrom).data.frequencies.copy()

    def local_indices(self, chrom: Any) -> np.ndarray:
        """Return mutation identifiers for a chromosome in operator order."""

        return self._get_chrom(chrom).data.local_idx.copy()

    def kernel_trace(self, theta: Any, exclude_chrom: Any = None) -> float:
        """Return ``tr(P_C K P_C)`` without constructing a dense kernel."""

        prior = self._prior(theta)
        self._ensure_projected_norms()
        return float(
            sum(
                np.dot(prior.weights(chrom.data.frequencies), chrom.projected_norm2)
                for chrom in self._chromosomes
                if chrom.label != exclude_chrom
            )
        )

    def dense_kernel(self, theta: Any, exclude_chrom: Any = None) -> np.ndarray:
        """Materialise a kernel for a small dense input or a test oracle."""

        if any(chrom.dense is None for chrom in self._chromosomes):
            raise RuntimeError("dense_kernel is only available for dense chromosomes")
        prior = self._prior(theta)
        result = np.zeros((self.n, self.n), dtype=np.float64)
        for chrom in self._chromosomes:
            if chrom.label == exclude_chrom:
                continue
            centered = self.project(chrom.dense)
            result += (centered * prior.weights(chrom.data.frequencies)[None, :]) @ centered.T
        return result

    def _get_chrom(self, label: Any) -> _Chromosome:
        for chrom in self._chromosomes:
            if chrom.label == label:
                return chrom
        raise KeyError(f"unknown chromosome {label!r}")
