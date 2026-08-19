# Evolutionary GRGL-backed LMM: status and remaining work

Last audited: 2026-08-19

## Purpose

This file is the current implementation ledger for evo-lmm. It records the
scientific and architectural decisions that remain binding, what is already in
the repository, and the next work in priority order. Historical implementation
notes have been removed now that the corresponding code exists.

The target remains a correct, testable evolutionary random-effects LMM over
GRGL-backed genotype data, followed by a calibrated BOLT-style association
pipeline. GRAPP is a pinned, read-only reference implementation.

## Stable model decisions

Let

```text
q_j = x_hat_j * (1 - x_hat_j)
tau = sigma_a^2 / W_S
```

The raw-dosage effect variances are

```text
simplified: v_j = sigma_b^2 / (1 + 2 * tau * q_j)

full:       v_j = sigma_b^2 *
              (1 - rho^2 * (2 * tau * q_j) / (1 + 2 * tau * q_j)).
```

The following choices are not open implementation questions:

- The simplified model is the exact `rho^2 = 1` specialization of the full
  model.
- `sigma_b^2` is a per-locus effect scale on raw diploid dosage. The
  evolutionary kernel has no implicit `1 / M` normalization.
- Model covariance columns and tested-genotype columns have separate scaling.
- Frequencies are sample allele frequencies and are recomputed after sample
  filtering.
- Covariates, including an intercept, are removed by an orthonormal projection.
- GRAPP and GRGL submodules remain unmodified; compatibility imports stay in
  `src/evo_lmm/grapp_backend.py`.
- Hutchinson is the production trace-estimator default. Spherical XTrace stays
  available through `trace_method="xtrace"` for experiments.
- Warm-started CG is enabled by default and must preserve the requested CG
  tolerance and accepted/trial cache isolation.

## Present in the repository

### Model and data boundaries

- [x] `SimplifiedPrior` and `FullPrior` with validation, stable transformed
  coordinates, effect variances, and analytic first derivatives.
- [x] Exact nested identity at `rho^2 = 1` and explicit `tau = 0`, `rho = 0`,
  and monomorphic boundaries.
- [x] Dense dosage and GRG variant adapters with sample-frequency extraction,
  mutation filters, sample filters, and missing-data imputation inputs.
- [x] Pinned GRAPP compatibility layer for raw `X` operators and frequency
  traversal.

Evidence:

- `src/evo_lmm/priors.py`
- `src/evo_lmm/grg_data.py`
- `src/evo_lmm/grapp_backend.py`
- `tests/test_priors.py`
- `tests/test_grapp_regression.py`

### Matrix-free covariance operators

- [x] Projected raw-dosage products for dense matrices and GRGs.
- [x] Frequency-weighted kernels and analytic derivative-kernel products.
- [x] Batched `matmat` paths and multi-chromosome summation.
- [x] LOCO exclusion without marker-count renormalization.
- [x] Separately normalized model and test operators.
- [x] Dense kernel oracle and symmetry/PSD checks.

Evidence:

- `src/evo_lmm/operators.py`
- `tests/test_operators_dense.py`
- `tests/test_loco.py`

### REML fitting

- [x] Profiled `sigma_b^2`, derived `sigma_e^2`, and transformed optimization
  of `(log_delta, log_tau[, logit_rho2])`.
- [x] Exact dense restricted likelihood, score, and average-information oracle.
- [x] Matrix-free average-information iterations with damping, step capping,
  and step halving.
- [x] Fixed seeded probes across an optimization run.
- [x] Shared-probe Hutchinson estimator as the default.
- [x] Configurable spherical XTrace with query-budget and standard-error
  diagnostics.
- [x] Synchronized projected CG with active-column convergence.
- [x] Accepted/trial warm-start caches, residual revalidation, rejection of poor
  guesses, and diagnostics.
- [x] Boundary and weak-identification warnings.
- [x] Haseman-Elston moment initializer implemented as a public helper.

Evidence:

- `src/evo_lmm/reml.py`
- `src/evo_lmm/trace.py`
- `src/evo_lmm/results.py`
- `tests/test_reml_dense.py`
- `tests/test_trace.py`
- `tests/test_parameter_estimation_large.py`

### Public fitting and prediction surface

- [x] Dense and GRG-backed simplified/full fitting entry points.
- [x] Missing-phenotype and explicit sample filtering at the public boundary.
- [x] LOCO covariance solves.
- [x] Genetic-value BLUPs.
- [x] Typed fit diagnostics and compact typed association arrays.

Evidence:

- `src/evo_lmm/bolt.py`
- `src/evo_lmm/results.py`
- `src/evo_lmm/__init__.py`
- `tests/test_msprime_grg_integration.py`

### Documentation and performance work

- [x] Sphinx documentation and public API reference.
- [x] Explicit ten-replicate SLiM tutorial with persisted, reusable artifacts.
- [x] Fit-only GRAPP versus evo-lmm benchmark separated from simulation.
- [x] Ten-replicate cumulative genic-variance plot with mean and sample-SD
  error bars.
- [x] Ten-replicate runtime aggregation.
- [x] Sequential profiles for cold/warm CG, Hutchinson, XTrace, and reduced
  probe counts.

Current benchmark context is documented in
`docs/tutorials/bolt_benchmark.rst`. The latest regenerated figure reports
mean fit times of `27.06 +/- 6.77 s` for evo-lmm and `11.74 +/- 1.77 s` for
GRAPP on this machine. These are profiling results, not portable performance
guarantees.

## Remaining work

### Priority 0: finish correctness of the BOLT-style analysis path

The variance-component fitter is implemented, but the current
`association()` helper is not yet a faithful replacement for GRAPP's full
calibration procedure. It uses projected test columns and conservative
denominators; it does not perform GRAPP's prospective/retrospective calibration
on sampled variants.

- [ ] Implement calibration using the fitted evolutionary LOCO covariance while
  retaining the independent BOLT-normalized test operator.
- [ ] Audit beta, standard-error, chi-square, and calibration-scale formulas in
  raw-effect units; do not reinterpret `sigma_b^2` as GRAPP's standardized
  `sigma_g^2`.
- [ ] Add dense-versus-GRG association equivalence tests.
- [ ] Add simplified-versus-full `rho^2 = 1` association equivalence tests.
- [ ] Add seeded null simulations that check calibration and test-statistic
  inflation.
- [ ] Add tests covering missing phenotypes and nontrivial covariates through
  fitting, LOCO, association, and BLUP.

Acceptance gate: the CPU association output is calibrated on seeded null data,
matches a dense oracle on small data, and preserves the exact nested-model
identity.

### Priority 1: harden fitting for routine production use

The documentation benchmark intentionally caps optimization at eight
iterations and reports secant/convergence warnings from the comparison path.
That setup is useful for timing but is not a production convergence policy.

- [ ] Define and test a production convergence policy independently of the
  matched-budget benchmark.
- [ ] Split stochastic REML into two trace-precision stages:
  - a **sketch stage** uses a small configurable probe budget for the first
    coarse AI-REML steps, where accurate scores are unnecessary;
  - a **refinement stage** switches to a larger configurable probe budget
    before convergence can be declared and uses that budget for final scores,
    uncertainty diagnostics, and reported estimates;
  - make the refinement probes a deterministic superset of the sketch probes
    so the switch adds information without discarding the common random
    numbers already used by the optimizer;
  - define the transition using optimization state (for example, a bounded
    step/score threshold) with a maximum sketch-iteration fallback, rather
    than relying only on a fixed iteration number;
  - reset or revalidate CG warm-start caches at the transition and record the
    stage, probe counts, transition iteration, and operator-query counts in
    diagnostics;
  - test that final convergence and estimates are judged only at refinement
    precision, including cases where the sketch-stage score appears to have
    converged by chance.
- [x] Wire `haseman_elston_initialization()` into `fit_reml()` as the optional
  `initialization="he"` mode. The default remains unchanged.
- [ ] Add a matrix-free full-model GRG recovery test away from the `rho`
  boundaries.
- [ ] Add explicit tests for trace-error-driven non-convergence and any retry or
  probe-budget escalation policy before implementing automatic escalation.
- [ ] Verify fixed-effect reporting and phenotype rescaling on nonstandardized
  phenotypes with multiple covariates.
- [ ] Add a stable serialization/reporting surface if fit results must persist
  beyond the Python process.

Acceptance gate: default CPU fits converge reliably on representative
simplified and full simulations, and failures return actionable diagnostics
rather than plausible-looking estimates.

### Priority 2: close the remaining performance gap

Warm starts removed most redundant CG work, but the ten-replicate benchmark
still shows evo-lmm slower than GRAPP. Profile before adding more machinery.

- [ ] Attribute remaining time to GRG traversals, derivative construction,
  trace queries, Python orchestration, LOCO setup, and optimizer evaluations.
- [ ] Record operator-call counts and time by inverse stage, not only aggregate
  CG iterations.
- [ ] Reduce repeated parameter-independent work across accepted iterations and
  line-search trials where profiling shows a material cost.
- [ ] Re-evaluate Hutchinson probe count using end-to-end estimate error and
  convergence, not trace microbenchmarks alone.
- [ ] Consider a changing-kernel Nyström preconditioner only if CG again becomes
  the dominant measured cost. Include sketch construction and refresh time in
  the comparison.

Acceptance gate: report end-to-end time, estimate changes, convergence status,
and operator-equivalent work over multiple persisted replicates.

### Priority 3: GPU and user-facing interfaces

- [ ] Add CuPy parity only after the calibrated CPU path and its tests are
  complete.
- [ ] Replace the temporary `main.py` entry point with a deliberately scoped CLI
  if command-line fitting is required.
- [ ] Decide whether association results need DataFrame/file-output adapters;
  keep the core typed-array API independent of a tabular dependency.
- [ ] Add curated end-to-end examples for the full model and calibrated
  association once those paths are production-ready.

## Deferred or rejected for now

- XTrace is not the default: current equal-cost experiments do not show a
  consistent error advantage over Hutchinson.
- Five trace probes are not sufficient for final refinement by default: the
  measured speedup came with large shifts in shape estimates and larger trace
  uncertainty. A similarly small budget may be evaluated for the sketch stage
  because its estimates are not reportable final results.
- Nyström preconditioning is deferred until profiling demonstrates that solve
  iterations dominate total runtime.
- GPU work is deferred until CPU statistical calibration is complete.
- No changes belong in the `grapp/` or `grgl/` submodules as part of the items
  above.

## Current validation commands

Run from the repository root:

```bash
uv run pytest -q
uv run pytest -q -m large
uv run python -c "import evo_lmm, pygrgl"
uv run sphinx-build -W -b html docs docs/_build/html
```

At the last audit, the regular suite had 25 passing tests and the large suite
had 4 passing tests. Generated simulation artifacts remain ignored under
`docs/_artifacts/`.

## Definition of the next milestone

The next milestone is not another covariance-kernel prototype. It is a
calibrated, end-to-end CPU analysis path in which:

1. simplified and full evolutionary fits converge with clear diagnostics;
2. LOCO association uses the fitted evolutionary covariance and independent
   BOLT-normalized test columns;
3. null calibration and dense/GRG equivalence tests pass;
4. BLUP and association preserve the `rho^2 = 1` nested identity; and
5. the multi-replicate benchmark reports accuracy and runtime without rerunning
   forward simulations.
