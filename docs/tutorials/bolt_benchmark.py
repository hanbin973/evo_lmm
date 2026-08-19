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
    fit_evolutionary_bolt_lmm,
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


@dataclass(frozen=True)
class BenchmarkResult:
    """Fits, timings, and data needed for the benchmark figure."""

    seed: int
    data: BenchmarkData
    evo_fit: Any
    bolt_fit: Any
    bolt_stats: list[Any]
    evo_seconds: float
    bolt_seconds: float


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


def simulate_forward_data(seed: int = SEED) -> BenchmarkData:
    """Run the same SLiM configuration as the forward-prior tutorial.

    The phenotype is generated once from the full GRG, then both methods fit
    the same phenotype using two GRG blocks. The split is only an adapter for
    GRAPP's LOCO calibration; it does not simulate a second data set.
    """

    with tempfile.TemporaryDirectory(prefix="evo_lmm_bolt_benchmark_") as directory:
        output_directory = Path(directory)
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
        )


def run_benchmark(seed: int = SEED) -> BenchmarkResult:
    """Fit evo-lmm and GRAPP BOLT-LMM and measure their wall-clock runtimes."""

    data = simulate_forward_data(seed)
    initial = SimplifiedPrior(sigma_b2=SIGMA_A2, tau=TRUE_TAU)

    start = time.perf_counter()
    evo_fit = fit_evolutionary_bolt_lmm(
        data.blocks,
        data.phenotype,
        frequencies=data.block_frequencies,
        model="simplified",
        initial=initial,
        trace_probes=64,
        max_iter=30,
        cg_tol=1e-8,
        seed=seed + 2,
    )
    evo_seconds = time.perf_counter() - start

    # Import GRAPP's public BOLT-LMM driver here so the data-generation helper
    # remains usable for inspecting the shared data without fitting.
    from grapp.assoc.bolt_inf_core import CovariateBasis
    from grapp.assoc.bolt_lmm import bolt_lmm_inf

    grapp_blocks = [(label, wrap_grg(grg)) for label, grg in data.blocks]
    covariates = CovariateBasis.intercept_only(N_INDIVIDUALS)
    start = time.perf_counter()
    bolt_fit, _calibration, _residuals, bolt_stats = bolt_lmm_inf(
        grapp_blocks,
        data.phenotype,
        covariates,
        seed=seed + 2,
        threads=1,
    )
    bolt_seconds = time.perf_counter() - start

    return BenchmarkResult(
        seed=seed,
        data=data,
        evo_fit=evo_fit,
        bolt_fit=bolt_fit,
        bolt_stats=bolt_stats,
        evo_seconds=evo_seconds,
        bolt_seconds=bolt_seconds,
    )


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


def _bolt_curve(result: BenchmarkResult) -> np.ndarray:
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


def make_summary(result: BenchmarkResult) -> plt.Figure:
    """Make the one-panel Figure-A11-style cumulative variance comparison."""

    data = result.data
    maf = np.minimum(data.full_frequencies, 1.0 - data.full_frequencies)
    q = data.full_frequencies * (1.0 - data.full_frequencies)
    segregating = q > 0.0
    genic_factor = 2.0 * q

    realized = _cumulative_by_maf(
        maf,
        np.square(data.alpha) * genic_factor,
        eligible=segregating,
    )
    configured_prior = SimplifiedPrior(
        sigma_b2=SIGMA_A2,
        tau=TRUE_TAU,
    )
    configured = _cumulative_by_maf(
        maf,
        configured_prior.effect_variances(data.full_frequencies) * genic_factor,
        eligible=segregating,
    )
    evo = _cumulative_by_maf(
        maf,
        result.evo_fit.prior.effect_variances(data.full_frequencies) * genic_factor,
        eligible=segregating,
    )
    bolt = _bolt_curve(result)

    figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    x = np.arange(MAF_THRESHOLDS.size)
    axis.plot(x, realized, color="tab:green", marker="o", linewidth=1.8, label="SLiM realization")
    axis.plot(
        x,
        configured,
        color="black",
        linewidth=1.4,
        label=r"configured evolutionary prior ($\sigma_a^2=1$, $W_S=1$)",
    )
    axis.plot(
        x,
        evo,
        color="tab:red",
        linestyle="--",
        marker="s",
        linewidth=1.8,
        label="evo-lmm fitted prior",
    )
    axis.plot(
        x,
        bolt,
        color="tab:blue",
        linestyle=":",
        marker="^",
        linewidth=2.0,
        label=r"GRAPP BOLT-LMM (global GRM)",
    )
    axis.set_xticks(x, [f"{threshold:g}" for threshold in MAF_THRESHOLDS], rotation=35, ha="right")
    axis.set_xlabel("cumulative MAF bin")
    axis.set_ylabel("cumulative genic variance")
    axis.set_title(
        "Cumulative genic variance across MAF bins\n"
        rf"$N={N_INDIVIDUALS}$, $V_S={V_S:g}$, $W_S={W_S:g}$, "
        rf"$\rho^2=1$, seed={result.seed}"
    )
    axis.grid(axis="y", linestyle=":", alpha=0.55)
    axis.set_ylim(bottom=0.0)
    axis.legend(loc="upper left", fontsize=8)
    axis.text(
        0.98,
        0.06,
        f"runtime\nevo-lmm: {result.evo_seconds:.2f} s\n"
        f"GRAPP BOLT-LMM: {result.bolt_seconds:.2f} s",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.85},
    )
    return figure


if __name__ == "__main__":
    benchmark = run_benchmark()
    print(
        f"mutations={benchmark.data.full_grg.num_mutations} "
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
