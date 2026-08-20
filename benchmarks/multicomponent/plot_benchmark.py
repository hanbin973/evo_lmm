"""Aggregate replicate CSVs and plot multicomponent runtime and estimates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

CATEGORIES = ("lof", "missense", "synonymous")


def read_results(paths: list[Path]) -> list[dict[str, str]]:
    """Read and sort one-row inference results."""
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    return sorted(rows, key=lambda row: int(row["replicate"]))


def write_summary(rows: list[dict[str, str]], output: Path) -> None:
    """Combine replicate rows into a stable summary CSV."""
    if not rows:
        raise ValueError("at least one benchmark row is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows: list[dict[str, object]], output: Path) -> plt.Figure:
    """Plot runtime and parameter estimates with generating-value references."""
    if not rows:
        raise ValueError("at least one benchmark row is required")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    runtime_keys = ("simulation_seconds", "fit_seconds", "runtime_seconds")
    axes[0].boxplot(
        [[float(row[key]) for row in rows] for key in runtime_keys],
        tick_labels=("SLiM + conversion", "REML fit", "total"),
        showmeans=True,
    )
    axes[0].set_ylabel("runtime (seconds)")
    axes[0].set_title(f"Runtime across {len(rows)} replicates")
    axes[0].grid(axis="y", alpha=0.3)

    linear_specs = [("sigma_e2", "residual")]
    linear_specs += [
        (f"sigma_b2_{label}", rf"$\sigma^2_{{b,{label}}}$") for label in CATEGORIES
    ]
    estimates = [
        [float(row[f"{key}_estimate"]) for row in rows] for key, _ in linear_specs
    ]
    axes[1].boxplot(
        estimates, tick_labels=[label for _, label in linear_specs], showmeans=True
    )
    for position, (key, _) in enumerate(linear_specs, start=1):
        truth = float(rows[0][f"{key}_generating"])
        axes[1].hlines(
            truth,
            position - 0.32,
            position + 0.32,
            color="tab:red",
            linestyle=":",
            linewidth=2,
            label="generating parameter" if position == 1 else None,
        )
    axes[1].set_ylabel("parameter value")
    axes[1].set_title("Scale and residual estimates")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(loc="best")

    tau_specs = [(f"tau_{label}", rf"$\tau_{{{label}}}$") for label in CATEGORIES]
    tau_estimates = [
        [float(row[f"{key}_estimate"]) for row in rows] for key, _ in tau_specs
    ]
    if any(
        value <= 0.0
        for estimates_for_category in tau_estimates
        for value in estimates_for_category
    ):
        raise ValueError("log-scale tau panel requires strictly positive estimates")
    axes[2].boxplot(
        tau_estimates, tick_labels=[label for _, label in tau_specs], showmeans=True
    )
    for position, (key, _) in enumerate(tau_specs, start=1):
        truth = float(rows[0][f"{key}_generating"])
        if truth <= 0.0:
            raise ValueError("log-scale tau panel requires positive generating values")
        axes[2].hlines(
            truth,
            position - 0.32,
            position + 0.32,
            color="tab:red",
            linestyle=":",
            linewidth=2,
            label="generating parameter" if position == 1 else None,
        )
    axes[2].set_yscale("log")
    axes[2].set_ylabel(r"composite $\tau_c$ (log scale)")
    axes[2].set_title(r"Composite $\tau_c$ estimates")
    axes[2].grid(axis="y", which="both", alpha=0.3)
    axes[2].legend(loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    rows = read_results(args.input)
    write_summary(rows, args.summary)
    make_figure(rows, args.figure)


if __name__ == "__main__":
    main()
