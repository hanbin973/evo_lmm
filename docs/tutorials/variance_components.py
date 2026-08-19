"""Ten-replicate, 1,000-individual variance-component tutorial."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from evo_lmm import EvolutionaryLmmOps, SimplifiedPrior, fit_reml


N_INDIVIDUALS = 1_000
N_VARIANTS = 500
N_REPLICATES = 10
TRUE_PRIOR = SimplifiedPrior(sigma_b2=0.004, tau=2.0)
TRUE_SIGMA_E2 = 0.35


def fit_one_replicate(replicate: int) -> tuple[float, float, float, bool]:
    """Simulate and fit one replicate, returning ``(sigma_b2, tau, sigma_e2, converged)``."""

    rng = np.random.default_rng(1_000 + int(replicate))
    population_frequencies = np.linspace(0.03, 0.5, N_VARIANTS)
    dosage = rng.binomial(
        2,
        population_frequencies,
        size=(N_INDIVIDUALS, N_VARIANTS),
    ).astype(np.float64)
    frequencies = dosage.mean(axis=0) / 2.0
    ops = EvolutionaryLmmOps.from_dense(dosage, frequencies, model="simplified")

    effects = rng.normal(
        0.0,
        np.sqrt(TRUE_PRIOR.effect_variances(frequencies)),
    )
    phenotype = ops.apply_model_x(effects) + rng.normal(
        0.0,
        np.sqrt(TRUE_SIGMA_E2),
        size=N_INDIVIDUALS,
    )
    fit = fit_reml(
        ops,
        phenotype,
        initial=TRUE_PRIOR,
        exact=False,
        trace_probes=64,
        seed=2_000 + int(replicate),
        max_iter=25,
        cg_tol=1e-8,
    )
    return (
        float(fit.prior.sigma_b2),
        float(fit.prior.tau),
        float(fit.sigma_e2),
        bool(fit.diagnostics.converged),
    )


def run_replicates() -> np.ndarray:
    """Return one row per replicate with estimates and a convergence flag."""

    return np.asarray(
        [fit_one_replicate(replicate) for replicate in range(N_REPLICATES)],
        dtype=np.float64,
    )


def make_box_plot(results: np.ndarray) -> plt.Figure:
    """Draw log-scale estimate boxes with the generating values overlaid."""

    estimates = results[:, :3]
    true_values = np.array(
        [TRUE_PRIOR.sigma_b2, TRUE_PRIOR.tau, TRUE_SIGMA_E2],
        dtype=np.float64,
    )
    labels = [r"$\sigma_b^2$", r"$\tau$", r"$\sigma_e^2$"]

    figure, axes = plt.subplots(1, 3, figsize=(10, 3.8), constrained_layout=True)
    for axis, label, values, true_value in zip(axes, labels, estimates.T, true_values):
        positive_values = np.maximum(values, np.finfo(np.float64).tiny)
        axis.boxplot(positive_values, tick_labels=["10 replicates"], showmeans=True)
        axis.axhline(true_value, color="tab:red", linestyle="--", label="true value")
        axis.set_yscale("log")
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="best")
    figure.suptitle("Variance-component estimates, N = 1,000")
    return figure


results = run_replicates()
figure = make_box_plot(results)

if __name__ == "__main__":
    print("converged replicates:", int(np.count_nonzero(results[:, 3])))
    print(results[:, :3])
    if "agg" not in plt.get_backend().lower():
        plt.show()
