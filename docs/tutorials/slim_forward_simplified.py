"""SLiM -> tree sequence -> GRG -> evo-lmm simplified-prior check."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pygrgl
import tskit

from evo_lmm import EvolutionaryLmmOps, SimplifiedPrior, fit_evolutionary_bolt_lmm, sample_allele_frequencies


N_INDIVIDUALS = 1_000
N_GENERATIONS = 4 * N_INDIVIDUALS
N_REPLICATES = 10
SIGMA_A2 = 1.0
V_S = 2.0 * N_INDIVIDUALS
W_S = V_S / (2.0 * N_INDIVIDUALS)
TRUE_TAU = SIGMA_A2 / W_S
RESIDUAL_VARIANCE = 0.4
SEED = 812


def run_slim(output_directory: Path, seed: int = SEED) -> Path:
    """Run the checked-in SLiM script and return its tree-sequence file."""

    output_directory.mkdir(parents=True, exist_ok=True)
    tree_path = output_directory / "slim_forward.trees"
    if "__file__" in globals():
        script = Path(__file__).with_name("slim_simplified_prior.slim")
    else:
        # matplotlib's Sphinx plot directive executes a file without defining
        # __file__. The repository-root path also works in a normal build.
        script = Path("docs/tutorials/slim_simplified_prior.slim")
        if not script.exists():
            script = Path("slim_simplified_prior.slim")
    script = script.resolve()
    subprocess.run(
        [
            "slim",
            "-s",
            str(seed),
            "-d",
            f'OUTPUT_FILE="{tree_path}"',
            "-d",
            f"BURN_IN={N_GENERATIONS}",
            str(script),
        ],
        check=True,
        cwd=output_directory,
    )
    return tree_path


def mutation_effects(tree_sequence: tskit.TreeSequence) -> np.ndarray:
    """Extract current SLiM m2 effects in tskit's mutation-row order.

    SLiM can stack recurrent mutations at one site. In that case, a tskit
    mutation row stores the current mutation first and its inherited mutation
    history after it; GRGL exposes one mutation column for the current row.
    """

    effects = []
    for mutation in tree_sequence.mutations():
        mutation_list = mutation.metadata.get("mutation_list", [])
        if not mutation_list:
            raise ValueError(
                "expected SLiM mutation metadata for every mutation row, "
                f"got none for mutation {mutation.id}"
            )
        effects.append(float(mutation_list[0]["selection_coeff"]))
    return np.asarray(effects, dtype=np.float64)


def simulate_and_fit(seed: int = SEED):
    """Run SLiM, convert its tree sequence to a GRG, and fit the prior."""

    with tempfile.TemporaryDirectory(prefix="evo_lmm_slim_") as directory:
        tree_path = run_slim(Path(directory), seed)
        tree_sequence = tskit.load(str(tree_path))
        alpha = mutation_effects(tree_sequence)
        # SLiM marks historical nodes as samples in its full recording. GRGL
        # expects the current haploid genomes to be the leaf samples, so use
        # ordinary tskit simplification before conversion. This is not PySLiM
        # annotation and preserves mutation order with filter_sites=False.
        grg_tree_path = Path(directory) / "slim_forward.simplified.trees"
        tree_sequence.simplify(filter_sites=False).dump(str(grg_tree_path))
        grg = pygrgl.grg_from_trees(str(grg_tree_path))
        frequencies = sample_allele_frequencies(grg)
        if alpha.size != grg.num_mutations:
            raise ValueError(
                "SLiM effect order does not match GRG mutation order: "
                f"{alpha.size} effects for {grg.num_mutations} mutations"
            )

        prior = SimplifiedPrior(sigma_b2=SIGMA_A2, tau=TRUE_TAU)
        ops = EvolutionaryLmmOps(grg, frequencies=frequencies, model="simplified")
        # rho^2 = 1: the focal effect beta is the selected-trait effect alpha.
        genetic_value = ops.apply_model_x(alpha)
        rng = np.random.default_rng(seed + 1)
        phenotype = genetic_value + rng.normal(
            0.0,
            np.sqrt(RESIDUAL_VARIANCE),
            size=ops.n,
        )
        fit = fit_evolutionary_bolt_lmm(
            [("slim", grg)],
            phenotype,
            frequencies={"slim": frequencies},
            model="simplified",
            initial=prior,
            trace_probes=64,
            max_iter=30,
            cg_tol=1e-8,
            seed=seed + 2,
        )
        return {
            "tree_sequence": tree_sequence,
            "grg": grg,
            "frequencies": frequencies,
            "alpha": alpha,
            "phenotype": phenotype,
            "fit": fit,
        }


def run_replicates() -> list[dict]:
    """Run independent SLiM replicates with deterministic seeds."""

    return [simulate_and_fit(SEED + replicate) for replicate in range(N_REPLICATES)]


def local_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    query: np.ndarray,
    *,
    span: float = 0.4,
) -> np.ndarray:
    """Evaluate a log-frequency local-linear regression of ``y`` on ``x``.

    The bandwidth is adaptive: each query point uses the nearest ``span``
    fraction of observations with a tricube kernel. This empirical smoother
    is plotted separately from the evolutionary prior formula.
    """

    if not 0.0 < span <= 1.0:
        raise ValueError("span must be in (0, 1]")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0)
    if not np.any(valid):
        raise ValueError("local regression requires positive finite x values")
    log_x = np.log(x[valid])
    observations = y[valid]
    neighbor_count = min(
        log_x.size,
        max(8, int(np.ceil(span * log_x.size))),
    )
    fitted = np.empty(query.size, dtype=np.float64)
    for index, point in enumerate(np.log(query)):
        distances = np.abs(log_x - point)
        bandwidth = np.partition(distances, neighbor_count - 1)[neighbor_count - 1]
        bandwidth = max(bandwidth, np.finfo(np.float64).eps)
        scaled = distances / bandwidth
        weights = np.where(scaled < 1.0, (1.0 - scaled**3) ** 3, 0.0)
        design = np.column_stack((np.ones_like(log_x), log_x - point))
        weighted_design = weights[:, None] * design
        normal = design.T @ weighted_design
        rhs = design.T @ (weights * observations)
        try:
            fitted[index] = np.linalg.solve(normal, rhs)[0]
        except np.linalg.LinAlgError:
            fitted[index] = np.average(observations, weights=weights)
    return fitted


def make_summary(results: list[dict]) -> plt.Figure:
    """Plot one effect spectrum and replicate-level fitted components."""

    representative = results[0]
    fit = representative["fit"]
    frequencies = representative["frequencies"]
    alpha = representative["alpha"]
    minor_frequencies = np.minimum(frequencies, 1.0 - frequencies)
    observed = np.square(alpha)
    segregating = minor_frequencies > 0.0
    plot_frequencies = minor_frequencies[segregating]
    plot_observed = observed[segregating]
    curve_frequencies = np.geomspace(plot_frequencies.min(), 0.5, 200)
    parametric = fit.prior.effect_variances(curve_frequencies)
    local = local_linear_regression(plot_frequencies, plot_observed, curve_frequencies)

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].scatter(
        plot_frequencies,
        plot_observed,
        s=8,
        alpha=0.25,
        label=r"SLiM $\alpha_j^2$",
    )
    axes[0].plot(
        curve_frequencies,
        parametric,
        color="tab:red",
        linewidth=2,
        label=r"parametric fitted $E[\beta_j^2\mid x_j]$",
    )
    axes[0].plot(
        curve_frequencies,
        local,
        color="tab:purple",
        linestyle="--",
        linewidth=1.5,
        label="local linear regression",
    )
    axes[0].set_xlabel("sample minor allele frequency")
    axes[0].set_ylabel(r"effect square / variance")
    axes[0].set_title("Forward-simulated effect spectrum")
    axes[0].set_xscale("log")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    labels = [r"$\sigma_b^2$", r"$\tau$", r"$\sigma_e^2$"]
    estimates = np.asarray(
        [
            [result["fit"].prior.sigma_b2, result["fit"].prior.tau, result["fit"].sigma_e2]
            for result in results
        ],
        dtype=np.float64,
    )
    true_values = [SIGMA_A2, TRUE_TAU, RESIDUAL_VARIANCE]
    axes[1].boxplot(
        estimates,
        tick_labels=labels,
        showmeans=True,
        meanprops={"marker": "^", "markerfacecolor": "tab:green", "markeredgecolor": "tab:green"},
    )
    positions = np.arange(1, len(labels) + 1, dtype=np.float64)
    for position, true_value in zip(positions, true_values):
        axes[1].hlines(
            true_value,
            position - 0.3,
            position + 0.3,
            colors="tab:red",
            linestyles=":",
            linewidth=2,
            label="generating value" if position == positions[0] else None,
        )
    axes[1].set_ylim(bottom=0.0)
    axes[1].set_title(f"N={N_INDIVIDUALS}, {N_REPLICATES} replicates")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(loc="best")
    return figure


if __name__ == "__main__":
    results = run_replicates()
    for replicate, result in enumerate(results):
        print(
            f"replicate={replicate} mutations={result['grg'].num_mutations} "
            f"prior={result['fit'].prior} sigma_e2={result['fit'].sigma_e2} "
            f"diagnostics={result['fit'].diagnostics}"
        )
    make_summary(results)
    if "agg" not in plt.get_backend().lower():
        plt.show()
