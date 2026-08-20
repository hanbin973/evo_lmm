"""Fit one cached multicomponent SLiM replicate and record estimates/runtime."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pygrgl

from evo_lmm import (
    EvolutionaryLmmOps,
    MultiComponentOps,
    fit_multicomponent_reml,
    sample_allele_frequencies,
)

CATEGORIES = ("lof", "missense", "synonymous")
FIELDNAMES = (
    "replicate",
    "seed",
    "n_individuals",
    "n_mutations",
    "simulation_seconds",
    "fit_seconds",
    "runtime_seconds",
    "status",
    "converged",
    "sigma_e2_estimate",
    "sigma_e2_generating",
    *(f"sigma_b2_{label}_estimate" for label in CATEGORIES),
    *(f"sigma_b2_{label}_generating" for label in CATEGORIES),
    *(f"tau_{label}_estimate" for label in CATEGORIES),
    *(f"tau_{label}_generating" for label in CATEGORIES),
)


def infer(data_directory: Path, output: Path) -> None:
    """Load cached GRGs and phenotype, run REML, and write one CSV row."""
    metadata = json.loads(
        (data_directory / "simulation.json").read_text(encoding="utf-8")
    )
    phenotype = np.load(data_directory / "phenotype.npy")
    operators = {}
    for label in CATEGORIES:
        grg = pygrgl.load_immutable_grg(str(data_directory / f"{label}.grg"))
        operators[label] = EvolutionaryLmmOps(
            grg, frequencies=sample_allele_frequencies(grg)
        )
    ops = MultiComponentOps.from_operators(operators)
    started = perf_counter()
    fit = fit_multicomponent_reml(ops, phenotype, seed=int(metadata["seed"]) + 2)
    fit_seconds = perf_counter() - started

    row: dict[str, object] = {
        "replicate": metadata["replicate"],
        "seed": metadata["seed"],
        "n_individuals": metadata["n_individuals"],
        "n_mutations": metadata["n_mutations"],
        "simulation_seconds": metadata["simulation_seconds"],
        "fit_seconds": fit_seconds,
        "runtime_seconds": float(metadata["simulation_seconds"]) + fit_seconds,
        "status": fit.status,
        "converged": fit.converged,
        "sigma_e2_estimate": fit.sigma_e2,
        "sigma_e2_generating": metadata["residual_variance"],
    }
    for label, component in zip(CATEGORIES, fit.prior.components):
        row[f"sigma_b2_{label}_estimate"] = component.sigma_b2
        row[f"sigma_b2_{label}_generating"] = metadata["sigma_b2"][label]
        row[f"tau_{label}_estimate"] = component.tau
        row[f"tau_{label}_generating"] = metadata["tau"][label]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    infer(args.data_directory, args.output)


if __name__ == "__main__":
    main()
