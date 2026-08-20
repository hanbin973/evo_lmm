# Multicomponent benchmark

This benchmark runs ten independent forward SLiM simulations, partitions the
mutations into LoF, missense, and synonymous components using distinct SLiM
mutation types, and fits the annotation-partitioned simplified REML model. It
records the simulation/conversion time, fit time, total runtime, convergence
status, residual variance, and every category's `sigma_b2_c` and composite
`tau_c` estimate in CSV format.

The three mutation types use normally distributed selected-trait effects
(`k_c = 1/2`) with `rho_ab,c = 1`. The shared selection width is `W_S = 2`, so
the generating composite is `tau_c = sigma_a,c^2 / W_S`; `tau_c` is not
interpreted as coupling alone.

Run from the repository root (SLiM must be on `PATH`):

```bash
uv run snakemake \
  -s benchmarks/multicomponent/Snakefile \
  --directory benchmarks/multicomponent \
  --cores 1
```

This writes `results/multicomponent.csv` and `results/multicomponent.png`.
Simulation artifacts are cached per replicate under `data/`; changing
`infer.py` reruns only inference and plotting, while changing
`plot_benchmark.py` reruns only aggregation and plotting. The fitter uses its
production defaults (`cg_tol=5e-4`, 12 trace probes) and starts from its generic
initial point; the benchmark does not adopt the tight small-fixture settings or
truth-valued initialization.

Individual stages can also be targeted without invalidating their inputs. For
example, rerun inference and downstream plotting for one cached replicate with
`... results/raw/replicate_0.csv`, or regenerate only the final products with
`... results/multicomponent.png`.
