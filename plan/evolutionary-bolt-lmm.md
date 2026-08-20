# Evolutionary GRGL-backed LMM: status and remaining work

Last audited: 2026-08-20

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

The annotation-partitioned extension adds one component per annotation
category, with a category-specific `tau_c`, a shared `W_S`, and `rho_ab` fixed
at 1. Its invariants are recorded in `AGENTS.md`; its rationale is
`notes/rare_variant.md` section 2. The full model is frozen: implemented and
tested at the boundaries, but no new work targets it.

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

## Active workstream: annotation-partitioned multi-component kernel

This is the current development target and it takes precedence over the
`Priority N` backlog below, which covers the existing single-component fitter.
Items are labelled `MC0`-`MC4`; `MC0` is not `Priority 0`.

Detailed specification for every item is `notes/rare_variant.md` section 5 —
that file says what each item means, this ledger says whether it is done.
Binding model invariants are in `AGENTS.md`, "Annotation-partitioned
multi-component model". Simplified prior only; the full model is frozen.

- [ ] **MC0 — partitioned multi-component kernel.** Per-category prior objects
  with analytic derivatives in `(log sigma_b_c^2, log tau_c)`; multi-component
  AI-REML profiling one scale and searching the remaining `2|c|` shape
  coordinates; batched per-component derivative traversals reusing the shared CG
  solve and probes; PSD and symmetry tests per component and for the sum; exact
  nesting tests (all `tau_c = 0`, shared `tau`, single category).
- [ ] **MC1 — named baselines.** The flat per-category prior; RareEffect's
  marginal ML plus MoM-ratio adjustment including its negative-MoM truncation
  rule, reproduced faithfully; an optional MAC-threshold collapsing operator.
- [ ] **MC2 — joint multi-component MoM / Haseman-Elston.** Generalize
  `haseman_elston_initialization()` to the `(|c|+1)`-dimensional moment system.
  Initialization matters more here than in the single-component case: six shape
  coordinates, several weakly identified by construction.
- [ ] **MC3 — estimand adapters and reporting.** Both heritability conventions;
  per-MAF-bin genic-variance decomposition; delta-method standard errors plus
  profile likelihoods for each `tau_c`; boundary-aware likelihood-ratio tests;
  gene-level output with pooled shape parameters.
- [ ] **MC4 — WES data path.** Exome pVCF/BGEN to GRG conversion; annotation
  masks as first-class variant partitions; MAC/MAF filters applied before
  frequency recomputation; measured GRG compression on exome rare variants.
  Needed only for the access-gated Phase 3, so it is the lowest priority here.

Order: MC0, then MC1, then MC2, then MC3. MC4 is independent and deferred.

Acceptance gate: every nesting identity above holds exactly; dense-versus-GRG
equivalence holds at `cg_tol=1e-9`; the reproduced RareEffect baseline matches
an independent reimplementation on a small dense case.

## Remaining work

The sections below are the backlog for the existing single-component fitter.
They are not the active workstream.

### Priority 0: finish correctness of the BOLT-style analysis path (done)

`association()` now performs GRAPP's prospective/retrospective calibration on
sampled variants using the fitted evolutionary LOCO covariance, with the tested
columns kept on the independent BOLT normalisation.

- [x] Implement calibration using the fitted evolutionary LOCO covariance while
  retaining the independent BOLT-normalized test operator.
- [x] Audit beta, standard-error, chi-square, and calibration-scale formulas in
  raw-effect units; do not reinterpret `sigma_b^2` as GRAPP's standardized
  `sigma_g^2`.
- [x] Add dense-versus-GRG association equivalence tests.
- [x] Add simplified-versus-full `rho^2 = 1` association equivalence tests.
- [x] Add seeded null simulations that check calibration and test-statistic
  inflation.
- [x] Add tests covering missing phenotypes and nontrivial covariates through
  fitting, LOCO, association, and BLUP.

Implementation notes:

- `EvolutionaryLmmOps.test_stats()` returns the prior-independent tested-genotype
  quantities (projected raw and BOLT-normalized norms, `norm_scale`, and the
  GRAPP-compatible eligibility mask). Projected norms are computed once per
  operator, since the covariate basis is fixed after construction.
- `evo_lmm.calibration` mirrors GRAPP's `select_bolt_calibration_snps` and
  `calibrate_lmm_inf`: blocked uniform selection with a GRAMMAR pre-screen at
  `retro < 5`, prospective/retrospective moments from LOCO solves, a jackknife
  standard error, and the ratio-of-medians fallback above a standard error of
  `0.01`. The prospective and retrospective statistics are invariant to
  `sigma_b^2`, so the factor is a pure shape correction and `sigma_b^2` enters
  only through the per-chromosome inverse scale.
- `association()` returns `beta` and `se` in raw diploid-dosage units, a
  calibrated inverse-variance `score`, a single-variant linear-regression
  chi-square, and the eligibility mask. Ineligible columns carry `nan`
  statistics and `pvalue = 1`. `association_summary()` reports mean chi-square
  and `lambda_GC` for both statistics.
- `calibrate=False` and `use_loco=False` remain available as diagnostics and are
  documented as uncalibrated.

Evidence:

- `src/evo_lmm/calibration.py`
- `src/evo_lmm/bolt.py`
- `src/evo_lmm/operators.py`
- `src/evo_lmm/results.py`
- `tests/test_association.py`
- `docs/howto/index.rst`

Acceptance gate: met on CPU. The dense oracle test reproduces the residual
solves, both calibration moments, the factor, and the reported
`beta`/`se`/`chi-square` from an explicit pseudo-inverse; the seeded pure-null
panels return a calibration factor within `0.15` of one and a mixed-model mean
chi-square that matches plain regression; the `rho^2 = 1` full model reproduces
the simplified association output exactly.

Observed while validating, and now recorded as Priority 1 evidence: when a GRG
fit is stopped at a small iteration cap and leaves `h2` near its upper boundary
on null data, the calibrated statistic inflates (`lambda_GC` around 1.7). The
same null data with a converged fit gives `lambda_GC` in line with plain
regression. Calibration corrects the statistic's scale, not a variance-component
fit that has not converged, which is why the convergence-policy work, now under
Priority 2, remains a gate on any reported estimate.

### Priority 1: reporting-surface correctness

Convergence-policy work has moved to Priority 2. What remains here is the
correctness of what a fit reports once it has converged, which is independent of
how convergence is declared.

- [x] Wire `haseman_elston_initialization()` into `fit_reml()` as the optional
  `initialization="he"` mode. The default remains unchanged.
- [ ] Verify fixed-effect reporting and phenotype rescaling on nonstandardized
  phenotypes with multiple covariates.
- [ ] Add a stable serialization/reporting surface if fit results must persist
  beyond the Python process.

Acceptance gate: reported fixed effects, rescaled phenotypes, and persisted fit
results round-trip correctly on nonstandardized phenotypes with multiple
covariates.

### Priority 2: convergence policy and the remaining performance gap

Warm starts removed most redundant CG work, but the ten-replicate benchmark
still shows evo-lmm slower than GRAPP. Profile before adding more machinery.

This section also now owns the convergence-policy work, lowered from Priority 1
on the judgement that it can be tested later. Note what that costs: a
deprioritized prerequisite is still a prerequisite, and
`notes/rare_variant.md` deliberately keeps its Phase 3 gate on the convergence
policy rather than relaxing it. Either this work gets pulled forward when the
rare-variant reanalysis reaches Phase 3, or Phase 3 waits.

The stochastic defaults are the cheap end of the range: `trace_probes=12` and
`cg_tol=5e-4`, the latter matching GRAPP's solver budget so per-application work
is comparable. On a two-chromosome `N=200` GRG fit this is `8.6x` faster than
the previous `(64, 1e-9)` pair for a `2-3x` increase in trace standard error and
point estimates agreeing to three significant figures. These defaults are for
exploratory fits and benchmark parity, not for reported estimates; the
dense-oracle and dense/GRG equivalence tests pin `cg_tol=1e-9` explicitly. Until
the policy work below lands, a reportable estimate is obtained by raising the
budget by hand — `(64, 1e-9)` is the documented target pair — and reading
`FitDiagnostics.trace_standard_errors` to confirm the score is resolved above
trace noise.

The documentation benchmark intentionally caps optimization at eight iterations
and reports secant/convergence warnings from the comparison path. That setup is
useful for timing but is not a production convergence policy.

- [ ] Define and test a production convergence policy independently of the
  matched-budget benchmark.
- [ ] Add explicit tests for trace-error-driven non-convergence and any retry or
  probe-budget escalation policy before implementing automatic escalation.

- [ ] Attribute remaining time to GRG traversals, derivative construction,
  trace queries, Python orchestration, LOCO setup, and optimizer evaluations.
- [ ] Record operator-call counts and time by inverse stage, not only aggregate
  CG iterations.
- [ ] Reduce repeated parameter-independent work across accepted iterations and
  line-search trials where profiling shows a material cost.
- [ ] Re-evaluate Hutchinson probe count using end-to-end estimate error and
  convergence, not trace microbenchmarks alone.
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
  Lowered from Priority 1: the hand-raised `(64, 1e-9)` budget already
  yields correct reported estimates, so this item buys cost, not correctness.
- [ ] Consider a changing-kernel Nyström preconditioner only if CG again becomes
  the dominant measured cost. Include sketch construction and refresh time in
  the comparison.

Acceptance gate, convergence: default CPU fits converge reliably on
representative simplified simulations, and failures return actionable
diagnostics rather than plausible-looking estimates.

Acceptance gate, performance: report end-to-end time, estimate changes,
convergence status, and operator-equivalent work over multiple persisted
replicates. For the two-stage item specifically, the gate is that estimates and
convergence match a single-stage fit run entirely at the refinement budget, at
lower total cost.

### Priority 3: GPU and user-facing interfaces

- [ ] Add CuPy parity only after the calibrated CPU path and its tests are
  complete.
- [ ] Replace the temporary `main.py` entry point with a deliberately scoped CLI
  if command-line fitting is required.
- [ ] Decide whether association results need DataFrame/file-output adapters;
  keep the core typed-array API independent of a tabular dependency.
- [ ] Add curated end-to-end examples for calibrated association once that
  path is production-ready.

## Deferred or rejected for now

- The **full evolutionary model is frozen, not removed.** `FullPrior`, the
  `rho^2 = 1` nested identity, and the existing boundary tests stay exactly as
  they are and must keep passing; the simplified model remains its exact
  `rho^2 = 1` specialization. What moves to future work is all *further*
  full-model development, specifically the matrix-free full-model GRG recovery
  test away from the `rho` boundaries and curated end-to-end full-model
  examples. Rationale: under a U-shaped allele-frequency spectrum the data
  mainly identify the product `rho_ab^2 * sigma_a^2 / W_S` rather than its
  factors, so `rho^2` is weakly identified in practice. See
  `notes/rare_variant.md` section 2.2. Development effort goes to the
  annotation-partitioned simplified model instead.
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

At the last audit, `uv run pytest -q` collected 41 tests and 40 passed.
`test_large_grg_matrix_free_fit_matches_dense_parameter_oracle` fails on
aarch64 Linux with a stochastic-tolerance mismatch in the matrix-free versus
dense kernel comparison. It failed before the Priority 0 work as well, and
belongs to the Priority 1 convergence-policy item. Generated simulation artifacts remain ignored under
`docs/_artifacts/`.

## Definition of the next milestone

The next milestone is not another covariance-kernel prototype. It is a
calibrated, end-to-end CPU analysis path in which:

1. simplified evolutionary fits converge with clear diagnostics
   (open: Priority 2);
2. LOCO association uses the fitted evolutionary covariance and independent
   BOLT-normalized test columns (done);
3. null calibration and dense/GRG equivalence tests pass (done);
4. BLUP and association preserve the `rho^2 = 1` nested identity (done); and
5. the multi-replicate benchmark reports accuracy and runtime without rerunning
   forward simulations.
