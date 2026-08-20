"""Create one cached SLiM replicate and its annotation-partitioned GRGs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import pygrgl
import tskit

from evo_lmm import EvolutionaryLmmOps, sample_allele_frequencies

CATEGORIES = ("lof", "missense", "synonymous")
MUTATION_TYPE_IDS = {"lof": 2, "missense": 3, "synonymous": 4}


def mutation_data(tree_sequence: tskit.TreeSequence) -> tuple[np.ndarray, np.ndarray]:
    """Return current alpha effects and SLiM mutation-type IDs in row order."""
    effects: list[float] = []
    mutation_types: list[int] = []
    for mutation in tree_sequence.mutations():
        entries = mutation.metadata.get("mutation_list", [])
        if not entries:
            raise ValueError(f"mutation {mutation.id} has no SLiM metadata")
        effects.append(float(entries[0]["selection_coeff"]))
        mutation_types.append(int(entries[0]["mutation_type"]))
    return np.asarray(effects), np.asarray(mutation_types, dtype=np.int64)


def subset_grg(grg, selected: np.ndarray, path: Path) -> None:
    """Persist one mutation-type subset of a full GRG."""
    if selected.size == 0:
        raise RuntimeError(f"SLiM produced no mutations for component {path.stem}")
    if not pygrgl.save_subset(
        grg, str(path), pygrgl.TraversalDirection.DOWN, selected.tolist()
    ):
        raise RuntimeError(f"could not save component GRG {path}")


def simulate(args: argparse.Namespace) -> None:
    """Run SLiM, convert its output, simulate a phenotype, and cache all inputs."""
    args.output_directory = args.output_directory.resolve()
    args.metadata = args.metadata.resolve()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    trees_path = args.output_directory / "multicomponent.trees"
    simplified_path = args.output_directory / "multicomponent.simplified.trees"
    started = perf_counter()
    subprocess.run(
        [
            "slim",
            "-s",
            str(args.seed),
            "-d",
            f"N={args.individuals}",
            "-d",
            f"L={args.sequence_length}",
            "-d",
            f"MUT_RATE={args.mutation_rate}",
            "-d",
            f"BURN_IN={args.burn_in}",
            "-d",
            f"SIGMA_LOF2={args.sigma_lof2}",
            "-d",
            f"SIGMA_MISSENSE2={args.sigma_missense2}",
            "-d",
            f"SIGMA_SYNONYMOUS2={args.sigma_synonymous2}",
            "-d",
            f"V_S={2.0 * args.individuals * args.w_s}",
            "-d",
            f'OUTPUT_FILE="{trees_path}"',
            str(args.slim_script.resolve()),
        ],
        check=True,
        cwd=args.output_directory,
    )
    recorded = tskit.load(str(trees_path))
    alpha, mutation_types = mutation_data(recorded)
    simplified = recorded.simplify(filter_sites=False)
    simplified.dump(str(simplified_path))
    full_grg = pygrgl.grg_from_trees(str(simplified_path))
    if full_grg.num_mutations != alpha.size:
        raise ValueError(
            "SLiM mutation order changed during simplification/GRG conversion"
        )

    full_ops = EvolutionaryLmmOps(
        full_grg, frequencies=sample_allele_frequencies(full_grg)
    )
    genetic_value = full_ops.apply_model_x(alpha)
    rng = np.random.default_rng(args.seed + 1)
    phenotype = genetic_value + rng.normal(
        0.0, np.sqrt(args.residual_variance), size=full_ops.n
    )
    np.save(args.output_directory / "phenotype.npy", phenotype)

    component_counts = {}
    for label in CATEGORIES:
        selected = np.flatnonzero(mutation_types == MUTATION_TYPE_IDS[label])
        subset_grg(full_grg, selected, args.output_directory / f"{label}.grg")
        component_counts[label] = int(selected.size)

    metadata = {
        "replicate": args.replicate,
        "seed": args.seed,
        "n_individuals": full_ops.n,
        "n_mutations": int(alpha.size),
        "component_mutations": component_counts,
        "simulation_seconds": perf_counter() - started,
        "residual_variance": args.residual_variance,
        "sigma_b2": {
            "lof": args.sigma_lof2,
            "missense": args.sigma_missense2,
            "synonymous": args.sigma_synonymous2,
        },
        "tau": {
            "lof": args.sigma_lof2 / args.w_s,
            "missense": args.sigma_missense2 / args.w_s,
            "synonymous": args.sigma_synonymous2 / args.w_s,
        },
        "w_s": args.w_s,
    }
    args.metadata.write_text(
        json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--individuals", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--mutation-rate", type=float, required=True)
    parser.add_argument("--burn-in", type=int, required=True)
    parser.add_argument("--residual-variance", type=float, required=True)
    parser.add_argument("--sigma-lof2", type=float, required=True)
    parser.add_argument("--sigma-missense2", type=float, required=True)
    parser.add_argument("--sigma-synonymous2", type=float, required=True)
    parser.add_argument("--w-s", type=float, required=True)
    parser.add_argument("--slim-script", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    simulate(args)


if __name__ == "__main__":
    main()
