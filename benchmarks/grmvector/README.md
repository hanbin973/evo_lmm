# `grmvector` — GRM-vector product: tskit vs grapp

Compares two ways of computing the un-normalized, un-centred GRM-vector product

```text
y = G G^T v
```

on the same simulated data, without ever forming `G`:

| Method | Call |
| --- | --- |
| tskit | `ts.genetic_relatedness_vector(v, mode="branch", span_normalise=False, centre=False)` |
| grapp | `X @ (X.T @ v)` with `SciPyXOperator(grg, TraversalDirection.UP, haploid=True)` |

This follows the layout of
[`05-arg_rhe`](https://github.com/hanbin973/tslmm_paper/tree/main/05-arg_rhe) in
the tslmm reproduction package, with `arg_needle_lib.arg_matmul` replaced by the
GRG-backed grapp operator.

## What is being compared

`mode="site"` is not implemented in tskit yet, so the tskit side uses
`mode="branch"`, exactly as `05-arg_rhe` does. That gives the *expected* GRM
under the mutation rate — shared branch length, scaled by `mu` — while the grapp
side gives the *realized* GRM from the sampled mutations. The two therefore agree
in expectation, not exactly, and `benchmark.py` reports the Pearson correlation
between them as the agreement check. The `mu` factor is applied only in that
check; the timing loop measures the bare traversal.

The grapp side uses `SciPyXOperator`, the raw allele-count operator — **not**
`SciPyStdXOperator`. No allele-frequency standardization and no `1 / M` kernel
scaling is applied, matching tskit's `span_normalise=False, centre=False`
convention.

Simulation is haploid (`ploidy=1`) and the operator is built with
`haploid=True`, so both methods act on the same sample axis. GRGs are built with
`binary_mutations=True` so that recurrent and back mutations do not distort the
realized side.

## Staged pipeline

Tree sequences and GRGs are built once and cached on disk; the timing step only
loads them. Changing `benchmark.py` or `plot_benchmark.py` therefore does not
trigger re-simulation.

| Step | Script | Output |
| --- | --- | --- |
| 1. simulate | `simulate.py` | `data/trees/ts_N{n}_mu{mu}.trees` (+ `.json` metadata) |
| 2. convert | `convert.py` | `data/grg/grg_N{n}_mu{mu}.grg` |
| 3. benchmark | `benchmark.py` | `results/raw/bench_N{n}_mu{mu}.csv` |
| 4. aggregate | `Snakefile` | `results/benchmark_summary.csv` |
| 5. plot | `plot_benchmark.py` | `results/benchmark_plot.png` |

## Running

```bash
uv sync                       # from the repository root
cd benchmarks/grmvector
uv run snakemake --cores 1    # single core: the timings are single-threaded
```

To build only the cached inputs, or to re-time without touching them:

```bash
uv run snakemake --cores 4 data/grg/grg_N2000_mu1e-08.grg
uv run snakemake --cores 1 results/benchmark_summary.csv
```

Sample sizes, mutation rates, sequence length, `Ne`, recombination rate, seed,
and the iteration count are set at the top of the `Snakefile`.

Requires `snakemake`, `pandas`, `seaborn`, and `matplotlib` in addition to the
project dependencies.

## Output columns

`results/benchmark_summary.csv` carries, per configuration: `n_samples`, `mu`,
`num_trees`, `num_sites`, `num_mutations`, `grg_num_mutations`, `correlation`,
`ts_mean_sec`, `ts_std_sec`, `grapp_mean_sec`, `grapp_std_sec`, `iterations`.

The plot shows runtime against sample size (panel A, with standard errors) and
the tskit-vs-grapp Pearson correlation (panel B).
