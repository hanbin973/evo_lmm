"""Step 3: time tskit vs grapp GRM-vector products on cached inputs.

Loading the ``.trees`` and ``.grg`` artifacts is not timed; only the repeated
``G G^T v`` products are.
"""

import argparse
import time

import numpy as np
import pandas as pd
import tskit

import utils


def run_benchmark(trees_path, grg_path, mu, iterations, seed, output_file):
    ts = tskit.load(trees_path)
    grg = utils.load_grg(grg_path)
    operator = utils.make_grapp_operator(grg)

    n_samples = ts.num_samples
    if operator.shape[0] != n_samples:
        raise ValueError(
            f"sample axis mismatch: tskit={n_samples}, grapp={operator.shape[0]}"
        )

    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n_samples, 1))

    # 1. Agreement check, outside the timing loop.  tskit branch mode gives the
    # expected product under mutation rate mu; grapp gives the realized one, so
    # these agree in expectation, not exactly.
    y_ts = utils.compute_tskit_Gv(ts, v, mu=mu)
    y_grapp = utils.compute_grapp_Gv(operator, v)
    correlation = np.corrcoef(y_ts.flatten(), y_grapp.flatten())[0, 1]
    print(f"  correlation (expected vs realized)={correlation:.6f}")

    # 2. Timing loop.
    times_ts = []
    times_grapp = []
    for _ in range(iterations):
        start = time.perf_counter()
        utils.compute_tskit_Gv(ts, v)
        times_ts.append(time.perf_counter() - start)

        start = time.perf_counter()
        utils.compute_grapp_Gv(operator, v)
        times_grapp.append(time.perf_counter() - start)

    results = {
        "n_samples": n_samples,
        "num_trees": ts.num_trees,
        "num_sites": ts.num_sites,
        "num_mutations": ts.num_mutations,
        "grg_num_mutations": operator.shape[1],
        "correlation": correlation,
        "ts_mean_sec": np.mean(times_ts),
        "ts_std_sec": np.std(times_ts),
        "grapp_mean_sec": np.mean(times_grapp),
        "grapp_std_sec": np.std(times_grapp),
        "iterations": iterations,
    }
    pd.DataFrame([results]).to_csv(output_file, index=False)
    print(f"Done. Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark tskit vs grapp GRM-vector")
    parser.add_argument("--trees", type=str, required=True, help="Cached .trees path")
    parser.add_argument("--grg", type=str, required=True, help="Cached .grg path")
    parser.add_argument(
        "--mu", type=float, required=True, help="Mutation rate (scales branch mode)"
    )
    parser.add_argument("--iterations", type=int, default=10, help="Timing iterations")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the test vector")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    args = parser.parse_args()

    print(f"--- Benchmarking: {args.trees} (mu={args.mu}) ---")
    run_benchmark(
        args.trees, args.grg, args.mu, args.iterations, args.seed, args.output
    )

    # Record mu alongside the measurements for the aggregation step.
    df = pd.read_csv(args.output)
    df.insert(1, "mu", args.mu)
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
