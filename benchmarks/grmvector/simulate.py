"""Step 1: simulate a tree sequence and cache it as a ``.trees`` file."""

import argparse
import json

import utils


def main():
    parser = argparse.ArgumentParser(description="Simulate and cache a tree sequence")
    parser.add_argument("--n_samples", type=int, required=True, help="Haploid samples")
    parser.add_argument("--mu", type=float, required=True, help="Mutation rate")
    parser.add_argument("--seq_len", type=float, default=1e6, help="Sequence length")
    parser.add_argument("--ne", type=float, default=1e5, help="Effective population size")
    parser.add_argument("--rho", type=float, default=1e-8, help="Recombination rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, required=True, help="Output .trees path")
    parser.add_argument("--meta", type=str, default=None, help="Optional JSON metadata path")
    args = parser.parse_args()

    ts = utils.simulate_tree_sequence(
        args.n_samples, args.seq_len, args.ne, args.rho, args.mu, seed=args.seed
    )
    ts.dump(args.output)
    print(
        f"n={args.n_samples} mu={args.mu}: "
        f"{ts.num_trees} trees, {ts.num_sites} sites, {ts.num_mutations} mutations"
    )

    if args.meta:
        with open(args.meta, "w") as handle:
            json.dump(
                {
                    "n_samples": args.n_samples,
                    "mu": args.mu,
                    "seq_len": args.seq_len,
                    "ne": args.ne,
                    "rho": args.rho,
                    "seed": args.seed,
                    "num_trees": ts.num_trees,
                    "num_sites": ts.num_sites,
                    "num_mutations": ts.num_mutations,
                },
                handle,
                indent=2,
            )


if __name__ == "__main__":
    main()
