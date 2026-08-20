"""Step 5: plot runtime and method agreement."""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

# Okabe-Ito qualitative colours: high contrast, colour-vision-deficiency safe.
MU_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]

# Methods are separated by line style and marker, mutation rates by colour.
METHOD_STYLE = {
    "tskit": {"linestyle": "-", "marker": "o"},
    "grapp": {"linestyle": "--", "marker": "s"},
}


def _set_sample_ticks(ax, df):
    """Label every simulated sample size, not just log-decade ticks."""
    sizes = sorted(df["n_samples"].unique())
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{int(n):,}" for n in sizes])
    ax.minorticks_off()


def plot_benchmark(csv_path, output_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: could not find '{csv_path}'.")
        sys.exit(1)

    df["ts_sem"] = df["ts_std_sec"] / np.sqrt(df["iterations"])
    df["grapp_sem"] = df["grapp_std_sec"] / np.sqrt(df["iterations"])

    unique_mus = sorted(df["mu"].unique())
    color_map = {
        mu: MU_COLORS[i % len(MU_COLORS)] for i, mu in enumerate(unique_mus)
    }

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    # --- Panel A: runtime ---
    ax = axes[0]
    for mu in unique_mus:
        subset = df[df["mu"] == mu].sort_values("n_samples")
        color = color_map[mu]
        for method, mean_col, sem_col in (
            ("tskit", "ts_mean_sec", "ts_sem"),
            ("grapp", "grapp_mean_sec", "grapp_sem"),
        ):
            ax.errorbar(
                subset["n_samples"],
                subset[mean_col],
                yerr=subset[sem_col],
                color=color,
                capsize=3,
                linewidth=2,
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=0.6,
                **METHOD_STYLE[method],
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(r"GRM-vector runtime ($\pm$Std. Error)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sample size (haploid)", fontsize=11)
    ax.set_ylabel("Execution time (s)", fontsize=11)
    _set_sample_ticks(ax, df)

    # --- Panel B: method agreement ---
    ax = axes[1]
    for mu in unique_mus:
        subset = df[df["mu"] == mu].sort_values("n_samples")
        ax.plot(
            subset["n_samples"],
            subset["correlation"],
            color=color_map[mu],
            linestyle="-",
            marker="o",
            linewidth=2.5,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.6,
        )
    ax.set_xscale("log")
    ax.set_title("Method agreement (Pearson correlation)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sample size (haploid)", fontsize=11)
    ax.set_ylabel("Correlation (r)", fontsize=11)
    min_corr = df["correlation"].min()
    ax.set_ylim(max(0.0, min_corr - 0.01), 1.002)
    _set_sample_ticks(ax, df)

    # --- Shared legends: methods above, mutation rates below ---
    method_handles = [
        Line2D([], [], color="0.25", linewidth=2, markersize=6,
               markerfacecolor="0.25", markeredgecolor="white", label=name,
               **METHOD_STYLE[name])
        for name in METHOD_STYLE
    ]
    fig.legend(
        handles=method_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=len(method_handles),
        frameon=False,
        fontsize=10,
        title="Method",
        title_fontsize=10,
    )

    mu_handles = [
        Line2D([], [], color=color_map[mu], linewidth=3, label=f"{mu:.0e}")
        for mu in unique_mus
    ]
    fig.legend(
        handles=mu_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=len(mu_handles),
        frameon=False,
        fontsize=10,
        title=r"Mutation rate ($\mu$)",
        title_fontsize=10,
    )

    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    print(f"Saving plot to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot GRM-vector benchmark results")
    parser.add_argument("--input", type=str, required=True, help="Aggregated summary CSV")
    parser.add_argument("--output", type=str, required=True, help="Output PNG path")
    args = parser.parse_args()
    plot_benchmark(args.input, args.output)
