"""Benchmark evo-lmm against GRAPP's GRG-backed BOLT-LMM implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pygrgl
import tskit

from evo_lmm import (
    EvolutionaryLmmOps,
    SimplifiedPrior,
    fit_reml,
    sample_allele_frequencies,
)
from evo_lmm.grapp_backend import wrap_grg


# The forward tutorial is the source of truth for the simulation configuration.
# Sphinx's plot directive executes this file without defining __file__, so add
# the tutorial directory explicitly in that execution mode.
if "__file__" in globals():
    _TUTORIAL_DIRECTORY = Path(__file__).resolve().parent
else:
    _TUTORIAL_DIRECTORY = Path("docs/tutorials").resolve()
if str(_TUTORIAL_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_TUTORIAL_DIRECTORY))

from slim_forward_simplified import (  # noqa: E402
    N_GENERATIONS,
    N_INDIVIDUALS,
    RESIDUAL_VARIANCE,
    SEED,
    SIGMA_A2,
    TRUE_TAU,
    V_S,
    W_S,
    mutation_effects,
    run_slim,
)


MAF_THRESHOLDS = np.array([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5])
N_FORWARD_BLOCKS = 2
# Match GRAPP's numerical work budget.  At N=2,000 its automatic Monte Carlo
# rule selects 15 trials and its public BOLT driver defaults to this CG
# tolerance.  Using 64 probes and 1e-8 here measures a different accuracy
# target rather than the overhead of the evolutionary kernel.
TRACE_PROBES = 15
CG_TOL = 5e-4


@dataclass(frozen=True)
class BenchmarkData:
    """One forward-simulation data set and its two physical GRG blocks."""

    tree_sequence: tskit.TreeSequence
    full_grg: Any
    full_frequencies: np.ndarray
    alpha: np.ndarray
    phenotype: np.ndarray
    blocks: tuple[tuple[str, Any], ...]
    block_frequencies: dict[str, np.ndarray]
    bolt_blocks: tuple[tuple[str, Any], ...]
    bolt_block_frequencies: dict[str, np.ndarray]


@dataclass(frozen=True)
class BenchmarkReplicateResult:
    """One fit pair, runtime pair, and data set used in the benchmark."""

    seed: int
    data: BenchmarkData
    evo_fit: Any
    bolt_fit: Any
    bolt_stats: list[Any]
    evo_seconds: float
    bolt_seconds: float


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregate benchmark result over one or more forward replicates."""

    replicates: tuple[BenchmarkReplicateResult, ...]

    def __post_init__(self) -> None:
        if not self.replicates:
            raise ValueError("benchmark requires at least one replicate")

    @property
    def seed(self) -> int:
        """Return the first seed for backwards-compatible single-run callers."""

        return self.replicates[0].seed

    @property
    def data(self) -> BenchmarkData:
        """Return the first data set for backwards-compatible callers."""

        return self.replicates[0].data

    @property
    def evo_fit(self) -> Any:
        return self.replicates[0].evo_fit

    @property
    def bolt_fit(self) -> Any:
        return self.replicates[0].bolt_fit

    @property
    def bolt_stats(self) -> list[Any]:
        return self.replicates[0].bolt_stats

    @property
    def evo_seconds(self) -> float:
        return float(np.mean([result.evo_seconds for result in self.replicates]))

    @property
    def bolt_seconds(self) -> float:
        return float(np.mean([result.bolt_seconds for result in self.replicates]))

    @property
    def n_replicates(self) -> int:
        return len(self.replicates)

    def runtime_summary(self) -> dict[str, tuple[float, float]]:
        """Return mean and sample standard deviation of each fit runtime."""

        evo = np.asarray([result.evo_seconds for result in self.replicates])
        bolt = np.asarray([result.bolt_seconds for result in self.replicates])
        ddof = 1 if self.n_replicates > 1 else 0
        return {
            "evo-lmm": (float(np.mean(evo)), float(np.std(evo, ddof=ddof))),
            "GRAPP BOLT-LMM": (float(np.mean(bolt)), float(np.std(bolt, ddof=ddof))),
        }


def _split_forward_tree_sequence(
    tree_sequence: tskit.TreeSequence,
    output_directory: Path,
) -> tuple[tuple[str, Any], ...]:
    """Convert two physical halves of one tree sequence to coalescent GRGs.

    GRAPP's top-level BOLT-LMM driver performs leave-one-chromosome-out
    calibration. Two blocks are therefore needed even though the simulation
    has one continuous chromosome. The blocks contain disjoint variants from
    the same tree sequence and preserve the original individual sample set.
    """

    midpoint = float(tree_sequence.sequence_length) / N_FORWARD_BLOCKS
    intervals = ((0.0, midpoint), (midpoint, float(tree_sequence.sequence_length)))
    blocks: list[tuple[str, Any]] = []
    for index, (left, right) in enumerate(intervals, start=1):
        block = tree_sequence.keep_intervals([[left, right]], simplify=True).trim()
        if block.num_mutations == 0:
            raise RuntimeError(f"forward block {index} contains no mutations")
        block_path = output_directory / f"slim_forward_block_{index}.trees"
        block.dump(str(block_path))
        # GRAPP's XTX traversal requires coalescent counts in the GRG.
        grg = pygrgl.grg_from_trees(str(block_path), compute_coals=True)
        blocks.append((f"block-{index}", grg))
    return tuple(blocks)


def _make_bolt_blocks(
    blocks: tuple[tuple[str, Any], ...],
    block_frequencies: dict[str, np.ndarray],
    output_directory: Path,
) -> tuple[tuple[tuple[str, Any], ...], dict[str, np.ndarray]]:
    """Remove fixed columns before GRAPP's standardized-GRM calibration."""

    bolt_blocks: list[tuple[str, Any]] = []
    bolt_frequencies: dict[str, np.ndarray] = {}
    for label, grg in blocks:
        frequencies = block_frequencies[label]
        selected = np.flatnonzero(frequencies * (1.0 - frequencies) > 0.0)
        if selected.size == 0:
            raise RuntimeError(f"forward block {label} has no segregating variants")
        bolt_path = output_directory / f"{label}.segregating.grg"
        if not pygrgl.save_subset(
            grg,
            str(bolt_path),
            pygrgl.TraversalDirection.DOWN,
            selected.tolist(),
        ):
            raise RuntimeError(f"could not write segregating GRG for {label}")
        bolt_grg = pygrgl.load_immutable_grg(str(bolt_path))
        bolt_blocks.append((label, bolt_grg))
        bolt_frequencies[label] = sample_allele_frequencies(bolt_grg)
    return tuple(bolt_blocks), bolt_frequencies


def _build_benchmark_data(output_directory: Path, seed: int) -> BenchmarkData:
    """Generate one benchmark data set into a persistent directory."""

    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    tree_path = run_slim(output_directory, seed)
    recorded = tskit.load(str(tree_path))
    alpha = mutation_effects(recorded)
    tree_sequence = recorded.simplify(filter_sites=False)
    if alpha.size != tree_sequence.num_mutations:
        raise ValueError(
            "SLiM effect order changed during simplification: "
            f"{alpha.size} effects for {tree_sequence.num_mutations} mutations"
        )

    full_path = output_directory / "slim_forward.simplified.trees"
    tree_sequence.dump(str(full_path))
    full_grg = pygrgl.grg_from_trees(str(full_path), compute_coals=True)
    full_frequencies = sample_allele_frequencies(full_grg)
    if alpha.size != full_grg.num_mutations:
        raise ValueError(
            "SLiM effect order does not match GRG mutation order: "
            f"{alpha.size} effects for {full_grg.num_mutations} mutations"
        )

    prior = SimplifiedPrior(sigma_b2=SIGMA_A2, tau=TRUE_TAU)
    full_ops = EvolutionaryLmmOps(
        full_grg,
        frequencies=full_frequencies,
        model="simplified",
    )
    genetic_value = full_ops.apply_model_x(alpha)
    rng = np.random.default_rng(seed + 1)
    phenotype = genetic_value + rng.normal(
        0.0,
        np.sqrt(RESIDUAL_VARIANCE),
        size=full_ops.n,
    )

    blocks = _split_forward_tree_sequence(tree_sequence, output_directory)
    block_frequencies = {
        label: sample_allele_frequencies(grg) for label, grg in blocks
    }
    bolt_blocks, bolt_block_frequencies = _make_bolt_blocks(
        blocks,
        block_frequencies,
        output_directory,
    )
    concatenated_frequencies = np.concatenate(
        [block_frequencies[label] for label, _ in blocks]
    )
    np.testing.assert_allclose(concatenated_frequencies, full_frequencies)
    np.save(output_directory / "alpha.npy", alpha)
    np.save(output_directory / "phenotype.npy", phenotype)
    np.save(output_directory / "full.frequencies.npy", full_frequencies)
    for label, frequencies in block_frequencies.items():
        np.save(output_directory / f"{label}.frequencies.npy", frequencies)
    for label, frequencies in bolt_block_frequencies.items():
        np.save(output_directory / f"{label}.segregating.frequencies.npy", frequencies)
    (output_directory / "seed.txt").write_text(f"{seed}\n", encoding="utf-8")

    return BenchmarkData(
        tree_sequence=tree_sequence,
        full_grg=full_grg,
        full_frequencies=full_frequencies,
        alpha=alpha,
        phenotype=phenotype,
        blocks=blocks,
        block_frequencies=block_frequencies,
        bolt_blocks=bolt_blocks,
        bolt_block_frequencies=bolt_block_frequencies,
    )


def prepare_benchmark_data(output_directory: Path, seed: int = SEED) -> BenchmarkData:
    """Run SLiM once and persist all data required by subsequent fit runs."""

    return _build_benchmark_data(Path(output_directory), seed)


def load_benchmark_data(output_directory: Path) -> BenchmarkData:
    """Load a previously prepared benchmark without rerunning SLiM."""

    directory = Path(output_directory)
    if not (directory / "slim_forward.simplified.trees").exists():
        raise FileNotFoundError(
            f"benchmark data are missing in {directory}; run prepare_bolt_benchmark.py first"
        )
    tree_sequence = tskit.load(str(directory / "slim_forward.simplified.trees"))
    full_grg = pygrgl.grg_from_trees(
        str(directory / "slim_forward.simplified.trees"), compute_coals=True
    )
    labels = ("block-1", "block-2")
    blocks = tuple(
        (
            label,
            pygrgl.grg_from_trees(
                str(directory / f"slim_forward_{label.replace('-', '_')}.trees"),
                compute_coals=True,
            ),
        )
        for label in labels
    )
    bolt_blocks = tuple(
        (
            label,
            pygrgl.load_immutable_grg(str(directory / f"{label}.segregating.grg")),
        )
        for label in labels
    )
    block_frequencies = {
        label: np.load(directory / f"{label}.frequencies.npy") for label in labels
    }
    bolt_block_frequencies = {
        label: np.load(directory / f"{label}.segregating.frequencies.npy")
        for label in labels
    }
    return BenchmarkData(
        tree_sequence=tree_sequence,
        full_grg=full_grg,
        full_frequencies=np.load(directory / "full.frequencies.npy"),
        alpha=np.load(directory / "alpha.npy"),
        phenotype=np.load(directory / "phenotype.npy"),
        blocks=blocks,
        block_frequencies=block_frequencies,
        bolt_blocks=bolt_blocks,
        bolt_block_frequencies=bolt_block_frequencies,
    )


def benchmark_data_from_forward_result(
    forward_result: dict,
    output_directory: Path,
) -> BenchmarkData:
    """Build benchmark blocks from a previously simulated forward replicate.

    ``slim_forward_simplified.run_replicates`` returns the simplified tree
    sequence, GRG, frequencies, effects, and phenotype used by its two-panel
    figure.  This adapter reuses those objects and only performs the physical
    block split needed by GRAPP; it never invokes SLiM.
    """

    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    tree_sequence = forward_result["tree_sequence"]
    full_grg = forward_result["grg"]
    full_frequencies = np.asarray(forward_result["frequencies"], dtype=np.float64)
    alpha = np.asarray(forward_result["alpha"], dtype=np.float64)
    phenotype = np.asarray(forward_result["phenotype"], dtype=np.float64)
    if alpha.size != full_grg.num_mutations:
        raise ValueError(
            "forward effect order does not match GRG mutation order: "
            f"{alpha.size} effects for {full_grg.num_mutations} mutations"
        )
    blocks = _split_forward_tree_sequence(tree_sequence, output_directory)
    block_frequencies = {
        label: sample_allele_frequencies(grg) for label, grg in blocks
    }
    bolt_blocks, bolt_block_frequencies = _make_bolt_blocks(
        blocks,
        block_frequencies,
        output_directory,
    )
    concatenated_frequencies = np.concatenate(
        [block_frequencies[label] for label, _ in blocks]
    )
    np.testing.assert_allclose(concatenated_frequencies, full_frequencies)
    return BenchmarkData(
        tree_sequence=tree_sequence,
        full_grg=full_grg,
        full_frequencies=full_frequencies,
        alpha=alpha,
        phenotype=phenotype,
        blocks=blocks,
        block_frequencies=block_frequencies,
        bolt_blocks=bolt_blocks,
        bolt_block_frequencies=bolt_block_frequencies,
    )


def simulate_forward_data(seed: int = SEED) -> BenchmarkData:
    """Run the simulation in a temporary directory for in-process callers."""

    with tempfile.TemporaryDirectory(prefix="evo_lmm_bolt_benchmark_") as directory:
        return _build_benchmark_data(Path(directory), seed)


def _fit_benchmark_data(seed: int, data: BenchmarkData) -> BenchmarkReplicateResult:
    """Fit both methods for one prepared data set and time only the fits."""

    initial = SimplifiedPrior(sigma_b2=SIGMA_A2, tau=TRUE_TAU)

    start = time.perf_counter()
    evo_ops = EvolutionaryLmmOps(
        data.blocks,
        frequencies=data.block_frequencies,
        model="simplified",
    )
    evo_fit = fit_reml(
        evo_ops,
        data.phenotype,
        initial=initial,
        trace_probes=TRACE_PROBES,
        # GRAPP's secant search makes at most seven variance-component
        # evaluations (two initial points plus five updates). Allow one extra
        # AI-REML update while keeping optimizer work comparable.
        max_iter=8,
        cg_tol=CG_TOL,
        seed=seed + 2,
        trace_method="hutchinson",
    )
    evo_seconds = time.perf_counter() - start

    # Import GRAPP's public BOLT-LMM driver here so the data-generation helper
    # remains usable for inspecting the shared data without fitting.
    from grapp.assoc.bolt_inf_core import CovariateBasis
    from grapp.assoc.bolt_lmm import bolt_lmm_inf

    grapp_blocks = [(label, wrap_grg(grg)) for label, grg in data.bolt_blocks]
    covariates = CovariateBasis.intercept_only(N_INDIVIDUALS)
    start = time.perf_counter()
    bolt_fit, _calibration, _residuals, bolt_stats = bolt_lmm_inf(
        grapp_blocks,
        data.phenotype,
        covariates,
        mc_trials=TRACE_PROBES,
        cg_tol=CG_TOL,
        seed=seed + 2,
        threads=1,
        batched_apply_x=True,
    )
    bolt_seconds = time.perf_counter() - start

    return BenchmarkReplicateResult(
        seed=seed,
        data=data,
        evo_fit=evo_fit,
        bolt_fit=bolt_fit,
        bolt_stats=bolt_stats,
        evo_seconds=evo_seconds,
        bolt_seconds=bolt_seconds,
    )


def run_benchmark(
    seed: int = SEED,
    *,
    data_directory: Path | None = None,
    forward_results: list[dict] | None = None,
) -> BenchmarkResult:
    """Fit evo-lmm and GRAPP over one or more prepared replicates.

    ``forward_results`` is the output of the two-panel SLiM tutorial and is
    preferred for the documentation benchmark: it reuses those simulations
    and phenotypes while fitting each method with the benchmark's matched
    numerical budget.  ``data_directory`` remains a single-replicate fallback
    for the fit-only command and backwards-compatible callers.
    """

    if forward_results is not None and data_directory is not None:
        raise ValueError("pass either forward_results or data_directory, not both")

    if forward_results is None:
        data = (
            load_benchmark_data(data_directory)
            if data_directory is not None
            else simulate_forward_data(seed)
        )
        return BenchmarkResult((_fit_benchmark_data(seed, data),))

    if not forward_results:
        raise ValueError("forward_results must contain at least one replicate")
    with tempfile.TemporaryDirectory(prefix="evo_lmm_benchmark_blocks_") as directory:
        root = Path(directory)
        fitted = []
        for index, forward_result in enumerate(forward_results):
            replicate_seed = int(forward_result.get("seed", seed + index))
            data = benchmark_data_from_forward_result(
                forward_result,
                root / f"seed_{replicate_seed}",
            )
            fitted.append(_fit_benchmark_data(replicate_seed, data))
    return BenchmarkResult(tuple(fitted))


def _cumulative_by_maf(
    maf: np.ndarray,
    contributions: np.ndarray,
    *,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Sum per-variant genic contributions below each MAF threshold."""

    maf = np.asarray(maf, dtype=np.float64)
    contributions = np.asarray(contributions, dtype=np.float64)
    if maf.shape != contributions.shape:
        raise ValueError("maf and contributions must have matching shapes")
    if eligible is not None:
        eligible = np.asarray(eligible, dtype=bool)
        if eligible.shape != maf.shape:
            raise ValueError("eligible must match maf")
        maf = maf[eligible]
        contributions = contributions[eligible]
    order = np.argsort(maf, kind="stable")
    sorted_maf = maf[order]
    sorted_contributions = contributions[order]
    cumulative = np.cumsum(sorted_contributions)
    positions = np.searchsorted(sorted_maf, MAF_THRESHOLDS, side="right")
    return np.where(positions > 0, cumulative[np.maximum(positions - 1, 0)], 0.0)


def _bolt_curve(result: BenchmarkReplicateResult) -> np.ndarray:
    """Return GRAPP's global-GRM cumulative variance allocation.

    BOLT-LMM uses a standardized global GRM, so its fitted genetic scale is
    allocated equally across model SNPs: each eligible marker contributes
    ``sigma_g2 / M``. This is the neutral reference curve for this comparison,
    rather than the frequency-dependent evolutionary prior used by evo-lmm.
    """

    frequencies = np.concatenate(
        [np.asarray(stats.a1freq, dtype=np.float64) for stats in result.bolt_stats]
    )
    se = np.concatenate([np.asarray(stats.se, dtype=np.float64) for stats in result.bolt_stats])
    eligible = np.isfinite(se)
    marker_count = int(np.count_nonzero(eligible))
    if marker_count == 0:
        raise RuntimeError("GRAPP BOLT-LMM returned no model markers")
    marker_contribution = np.zeros(frequencies.size, dtype=np.float64)
    marker_contribution[eligible] = result.bolt_fit.sigma_g2 / marker_count
    maf = np.minimum(frequencies, 1.0 - frequencies)
    return _cumulative_by_maf(maf, marker_contribution, eligible=eligible)


def _replicate_curves(result: BenchmarkReplicateResult) -> dict[str, np.ndarray]:
    """Compute the four cumulative genic-variance curves for one replicate."""

    data = result.data
    maf = np.minimum(data.full_frequencies, 1.0 - data.full_frequencies)
    q = data.full_frequencies * (1.0 - data.full_frequencies)
    segregating = q > 0.0
    genic_factor = 2.0 * q
    configured_prior = SimplifiedPrior(sigma_b2=SIGMA_A2, tau=TRUE_TAU)
    return {
        "SLiM realization": _cumulative_by_maf(
            maf,
            np.square(data.alpha) * genic_factor,
            eligible=segregating,
        ),
        "configured evolutionary prior": _cumulative_by_maf(
            maf,
            configured_prior.effect_variances(data.full_frequencies) * genic_factor,
            eligible=segregating,
        ),
        "evo-lmm fitted prior": _cumulative_by_maf(
            maf,
            result.evo_fit.prior.effect_variances(data.full_frequencies) * genic_factor,
            eligible=segregating,
        ),
        "GRAPP BOLT-LMM": _bolt_curve(result),
    }


def make_summary(result: BenchmarkResult) -> plt.Figure:
    """Plot mean cumulative genic variance with replicate variation bars."""

    curve_values = {
        label: np.stack([_replicate_curves(rep)[label] for rep in result.replicates])
        for label in (
            "SLiM realization",
            "configured evolutionary prior",
            "evo-lmm fitted prior",
            "GRAPP BOLT-LMM",
        )
    }
    means = {label: np.mean(values, axis=0) for label, values in curve_values.items()}
    ddof = 1 if result.n_replicates > 1 else 0
    variations = {
        label: np.std(values, axis=0, ddof=ddof)
        for label, values in curve_values.items()
    }

    figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    x = np.arange(MAF_THRESHOLDS.size)
    axis.errorbar(
        x,
        means["SLiM realization"],
        yerr=variations["SLiM realization"],
        color="tab:green",
        marker="o",
        linewidth=1.8,
        capsize=3,
        label="SLiM realization",
    )
    axis.errorbar(
        x,
        means["configured evolutionary prior"],
        yerr=variations["configured evolutionary prior"],
        color="black",
        linewidth=1.4,
        capsize=3,
        label=r"configured evolutionary prior ($\sigma_a^2=1$, $W_S=1$)",
    )
    axis.errorbar(
        x,
        means["evo-lmm fitted prior"],
        yerr=variations["evo-lmm fitted prior"],
        color="tab:red",
        linestyle="--",
        marker="s",
        linewidth=1.8,
        capsize=3,
        label="evo-lmm fitted prior",
    )
    axis.errorbar(
        x,
        means["GRAPP BOLT-LMM"],
        yerr=variations["GRAPP BOLT-LMM"],
        color="tab:blue",
        linestyle=":",
        marker="^",
        linewidth=2.0,
        capsize=3,
        label=r"GRAPP BOLT-LMM (global GRM)",
    )
    axis.set_xticks(x, [f"{threshold:g}" for threshold in MAF_THRESHOLDS], rotation=35, ha="right")
    axis.set_xlabel("cumulative MAF bin")
    axis.set_ylabel("cumulative genic variance")
    axis.set_title(
        "Cumulative genic variance across MAF bins\n"
        rf"$N={N_INDIVIDUALS}$, $V_S={V_S:g}$, $W_S={W_S:g}$, "
        rf"$\rho^2=1$, {result.n_replicates} replicates"
    )
    axis.grid(axis="y", linestyle=":", alpha=0.55)
    axis.set_ylim(bottom=0.0)
    axis.legend(loc="upper left", fontsize=8)
    axis.text(
        0.98,
        0.06,
        "runtime (mean $\\pm$ SD)\n"
        f"evo-lmm: {result.runtime_summary()['evo-lmm'][0]:.2f} "
        f"$\\pm$ {result.runtime_summary()['evo-lmm'][1]:.2f} s\n"
        f"GRAPP BOLT-LMM: {result.runtime_summary()['GRAPP BOLT-LMM'][0]:.2f} "
        f"$\\pm$ {result.runtime_summary()['GRAPP BOLT-LMM'][1]:.2f} s",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.85},
    )
    return figure


if __name__ == "__main__":
    forward_artifacts = Path("docs/_artifacts/forward_replicates")
    artifact_directory = Path("docs/_artifacts/bolt_seed_812")
    if forward_artifacts.exists():
        from slim_forward_simplified import load_simulation_replicates

        benchmark = run_benchmark(
            forward_results=load_simulation_replicates(forward_artifacts),
        )
    else:
        benchmark = run_benchmark(
            data_directory=artifact_directory if artifact_directory.exists() else None,
        )
    print(
        f"replicates={benchmark.n_replicates} "
        f"mutations_first={benchmark.data.full_grg.num_mutations} "
        f"evo_lmm_seconds={benchmark.evo_seconds:.6f} "
        f"grapp_bolt_lmm_seconds={benchmark.bolt_seconds:.6f}"
    )
    print(
        f"evo_lmm_sigma_b2={benchmark.evo_fit.prior.sigma_b2:.6g} "
        f"evo_lmm_tau={benchmark.evo_fit.prior.tau:.6g} "
        f"grapp_sigma_g2={benchmark.bolt_fit.sigma_g2:.6g}"
    )
    make_summary(benchmark)
    if "agg" not in plt.get_backend().lower():
        plt.show()
