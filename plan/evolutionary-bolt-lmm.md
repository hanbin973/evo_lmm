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

### Annotation-partitioned kernel (MC0 complete)

- [x] Added `MultiComponentPrior` with one `SimplifiedPrior` per annotation
  category and transformed `(log_sigma_b2_c, log_tau_c)` coordinates.
- [x] Added `MultiComponentOps` with category-specific projected kernels,
  analytic component derivatives, and batched matrix-free kernel and derivative
  application on both the dense and the GRGL-backed path.
- [x] Added an exact dense profiled-REML prototype that profiles the residual
  scale and reports category-specific `sigma_b2_c`, `tau_c`, and heritability.
- [x] Added production projected AI-REML with analytic component derivatives,
  profiled residual scale, CG solves, and selectable Hutchinson/XTrace traces.
- [x] Reuse one block-CG solve across all probe/RHS columns, with a robust
  dependent-column fallback.
- [x] Deterministic tests for component and summed PSD/symmetry, derivative
  agreement, and independent dense and GRGL-backed boundary checks for the
  all-`tau_c`-zero flat kernel and the shared-`tau` model.
- [x] Pin the stochastic score against a finite-difference gradient of
  `profiled_reml_objective` and the stochastic average-information matrix
  against a dense oracle, both at exact traces
  (`probes = sqrt(n) * I` makes the Hutchinson estimator exact).
- [x] Bit-for-bit single-category delegation on both the dense and the
  GRGL-backed partition, and whole-fit dense-versus-GRG agreement at the
  default `cg_tol=5e-4`.
- [x] Pooled-shape fitting (`fit_tau=False`): the `|c|` scale coordinates are
  searched with every `tau_c` held at its supplied value, which is what
  per-gene reporting needs.

Evidence:

- `src/evo_lmm/multicomponent.py`
- `src/evo_lmm/operators.py`
- `tests/test_multicomponent.py`

### Phase 1 supporting utilities (MC1–MC3 partial)

- [x] Named flat M0 prior and optional MAC-threshold burden collapsing with
  frequency recomputation and source-column provenance.
- [x] RareEffect MoM-ratio adjustment with the non-positive-MoM fallback
  represented explicitly in the result.
- [x] Joint projected multi-component MoM system with raw estimates and a
  separate truncation audit.
- [x] Both heritability conventions and the per-MAF-bin genic-variance
  decomposition.
- [x] Uncertainty reporting: AI covariance carried into h² through a
  delta-method SE, scientific-scale SEs and profile intervals for every
  reported `sigma_b2_c` and `tau_c`, and boundary-mixture LRT p-values.
  Profiles are evaluated on the scale the fit reports.
- [x] Pooled-shape/per-gene reporting: `fit_genes()` conditions per-gene scales
  on shapes held fixed across genes, and reports the shapes it conditioned on.
- [x] Marginal baseline fitting with Hutchinson-compatible joint trace
  estimation, reproduced against an independent error-contrast implementation
  on a small dense case.

Evidence:

- `src/evo_lmm/baselines.py`
- `src/evo_lmm/reporting.py`
- `tests/test_phase1_components.py`

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
- [x] Ten-replicate multi-component forward-simulation benchmark, cached per
  replicate: SLiM with three mutation types -> per-category GRGs -> one
  annotation-partitioned fit at the production defaults, recording runtime,
  `status`, and every `sigma_b2_c`/`tau_c` against its generating value
  (`benchmarks/multicomponent/`, with `tests/test_multicomponent_benchmark.py`
  covering the aggregation/plotting stage; results are regenerable and stay
  untracked).

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
that file says what each item means, this ledger says whether it is done. Item
names here match that file's section headings verbatim; keep them in sync when
either side is edited.
Binding model invariants are in `AGENTS.md`, "Annotation-partitioned
multi-component model". Simplified prior only; the full model is frozen.

- [x] **MC0 — Annotation-partitioned multi-component kernel (simplified prior
  only).** Per-category prior objects
  with analytic derivatives in `(log sigma_b_c^2, log tau_c)`; multi-component
  AI-REML profiling one scale and searching the remaining `2|c|` shape
  coordinates; batched per-component derivative traversals reusing the shared CG
  solve and probes; PSD and symmetry tests per component and for the sum; exact
  nesting tests (all `tau_c = 0`, shared `tau`, single category).
- [x] **MC1 — Named baselines.** The flat per-category prior; RareEffect's
  marginal fit plus MoM-ratio adjustment including its negative-MoM truncation
  rule; an optional MAC-threshold collapsing operator. Open caveat: the
  marginal step is *restricted* (REML) profiling — the objective carries the
  `slogdet(B' V^-1 B)` term — and whether that matches RareEffect's published
  convention is not verified in this repository. The docstrings no longer claim
  it does. Verifying it needs an external reference run, not another internal
  test.
- [x] **MC2 — Joint multi-component MoM / Haseman–Elston.** The
  `(|c|+1)`-dimensional moment system in `joint_mom_initialization()` is wired
  into `fit_multicomponent_reml(initialization="he")`, mirroring the
  single-component fitter. It initializes the residual-relative component
  scales conditional on the requested evolutionary weights; it does not claim
  to identify or initialize nonlinear `tau_c` shapes. Raw MoM scales and
  negative-scale flags remain available on `MultiComponentFit` for the
  no-truncation audit. Ten of ten `n=2000` fits also converge from the generic
  initial point, so this is an optional initializer rather than a convergence
  workaround.
- [ ] **MC3 — Estimand adapters and reporting.** Both heritability conventions;
  per-MAF-bin genic-variance decomposition; scientific-scale delta-method
  standard errors plus profile likelihoods for each `sigma_b2_c` and `tau_c`;
  boundary-aware likelihood-ratio tests; gene-level output with pooled shape
  parameters. All of it is implemented (see "Phase 1 supporting utilities")
  except the likelihood-ratio clause: `boundary_lrt_pvalue()` supplies the
  boundary-mixture null, but nothing assembles the statistic from two fits, and
  the AI path cannot supply one — its `objective` field is `0.5*||score||^2`,
  not a likelihood, so an LR statistic can only come from
  `profiled_reml_objective` on the exact dense path. Closing MC3 means either
  wiring that assembly for the dense path and documenting the restriction, or
  giving the AI path a usable objective.
- [ ] **MC4 — WES data path.** Exome pVCF/BGEN to GRG conversion; annotation
  masks as first-class variant partitions; MAC/MAF filters applied before
  frequency recomputation; measured GRG compression on exome rare variants.
  Needed only for the access-gated Phase 3, so it is the lowest priority here.

Order: MC0 and MC1 are done; MC2 next, then MC3. MC4 is independent and
deferred.

Acceptance gate. All four clauses, no qualifiers:

1. **Every** nesting identity holds exactly — not "every implemented" one. An
   identity that is not yet tested is an open gate, not a satisfied one.
2. **Dense-versus-GRG equivalence at the default `cg_tol=5e-4`, on a small dataset.** This
   clause was deleted once and is restored deliberately. It is a small-data
   equivalence check by design: the point is to pin the GRG traversal against a
   dense oracle where the dense oracle is computable at all.
3. Single-category fitting delegates to the existing fitter bit-for-bit.
4. The reproduced RareEffect baseline matches an independent reimplementation on
   a small dense case.

**The sketch budget is retained for tests and exploratory fits**, and the
benchmark below uses it unchanged. Priority 2 owns the solver defaults and the
convergence criterion; this section does not restate them.

Gate status, re-audited 2026-08-20 after the fixes below — **all four clauses
are met, and the tests that back them have been checked to fail against the
constructions they replaced.**

1. Nesting: all-`tau_c`-zero and shared-`tau` use independent flat/operator
   constructions and `assert_array_equal`; the single-category identity is now
   also exact, on all four reported quantities, for the dense *and* the
   GRGL-backed partition.
2. Dense-versus-GRG: in addition to the operator-application oracle, a whole
   two-category fit agrees to `rtol=1e-9` on `sigma_e2`/`h2` and `1e-8` on the
   component parameters at the default `cg_tol=5e-4`, with identical seeds and
   probe budgets. The earlier status text claimed this clause on the operator
   test alone, which never enters the CG solve.
3. Single-category delegation is bit-for-bit. The delegation no longer forces
   `exact=True`: a GRGL-backed single category stays matrix-free at the
   requested `cg_tol` instead of materialising dense kernels.
4. The RareEffect baseline is compared against an error-contrast
   reimplementation (null-space contrasts and kernel eigenvalues, sharing no
   helper with `baselines.py`) plus recorded literals. The previous test
   rebuilt the production algorithm from production helpers and computed its
   expectation by calling the function under test, so it could not fail.

The convergence criterion these fits are judged by is recorded under
Priority 2, "Convergence policy — decided"; convergence and recovery at
`n = 2000` are measured there under "Convergence at realistic data size".

**`tau_c` recovery is not a target, and the benchmark confirms why.** The
paper's identifiability result, quoted in `notes/rare_variant.md` section 2.2,
already settles this: `1 + 2 tau_c q_j` sits close to one under a U-shaped
frequency spectrum, `sigma_b_c^2` sets the overall scale while `tau_c` enters
only as a weak frequency modulation, and the conclusion recorded there is to
write the analysis around `sigma_b_c^2`, not `tau_c`. The benchmark measures
exactly that: `sigma_b_c^2` and `sigma_e2` are recovered, individual `tau_c`
are not, and the fits sit at their own optima while doing so. No work item
follows from it. What does follow is that `tau_c` is reported with the
identifiability caveat attached and never as the headline estimand, and that
the quantity to watch is the accuracy of `sigma_b_c^2` — see the
smallest-component bias recorded under Priority 2.

Defects found in the same audit, all now fixed:

- The fitter did not converge or recover parameters: at `n=2000` with two
  categories it returned `converged=False`, `h2 = 0.993` against a truth of
  `0.187`, `tau` collapsed to zero, and a profiled REML objective of `1050.6`
  against `108.6` at the truth. Three independent defects, in the order they
  had to be removed:
  1. the score omitted the division of the data quadratic by the profiled
     `sigma_e2`, so it was not the gradient of any objective (the
     single-component fitter always divided);
  2. the average-information matrix contracted on the left with `P dV_i P y`
     instead of `P y`, inserting a third derivative factor and making the
     matrix indefinite, so the solved step was not a descent direction;
  3. a rejected line search left the loop instead of escalating the Levenberg
     damping and continuing. The AI matrix is routinely near-singular in the
     weakly identified `tau` directions, so the undamped step is dominated by
     them and the uniform `max_step` cap then squashes the well-identified
     scale coordinates to nearly zero movement.
  Score and AI are now extracted into `score_and_information()` and pinned to
  dense oracles. On a seeded two-category dense case the fitter converges and
  recovers `h2` to within `0.01`, at a profiled objective better than the
  objective at the generating parameters. Individual `sigma_b2_c`/`tau_c` are
  recovered less precisely, which is the documented weak identification, not a
  fitter defect.
- Profiles were evaluated on the wrong scale: a fit's scientific-scale
  `sigma_b2_c` was passed straight into `profiled_reml_objective`, which
  profiles the residual scale and therefore takes ratios, so every profile and
  every grid value was off by a factor of `sigma_e2`. Fixed in
  `reporting._ratio_prior`.
- `fit_genes()` used `pooled_tau` only as an initialisation, re-estimated every
  `tau_c` freely, and echoed the input `pooled_tau` back as if it had been
  held. It now fits with `fit_tau=False` and reports the shapes actually
  conditioned on.
- The block-CG operator applied `H` one column at a time, so a GRGL-backed
  component performed one traversal per column per iteration; it now uses the
  batched `matmat`. The Hutchinson trace also reuses the probe columns of the
  single block solve (`z' P dV z` as `(P z)' (dV z)`) instead of taking one
  extra solve per coordinate, and the derivative right-hand sides for the AI
  matrix share one block solve. `trace_standard_error` is now reported instead
  of `nan`.
- A `nan` placeholder for `last_sigma_e2` reached `SimplifiedPrior` on any loop
  exit before the in-loop assignments, raising a validator error instead of
  returning diagnostics. The reported state is seeded from the initial point,
  and the rejection branch records its own iterate. Covered by
  `test_max_iter_zero_returns_seeded_diagnostics_instead_of_raising` and
  `test_first_iteration_line_search_rejection_reports_that_iterate`.

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
fit that has not converged. The reader of an association result therefore has to
check the fit's `status` — an `iteration_cap` fit is exactly the case that
inflated here — since no acceptance test at realistic data size stands behind
the convergence criterion.

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

#### Convergence policy — decided

The criterion, the exit taxonomy, and the `tau_c` boundary report are settled
and implemented in both fitters. What is deliberately *not* settled is listed
immediately after them.

- **The criterion is scale-free.** Convergence is declared when
  `step_se_tol` (default `1e-2`) bounds `max_i |step_i| / SE_i`, with
  `step = AI^-1 score` and `SE = sqrt(diag(AI^-1))` — the largest move the next
  undamped Newton step would make in any coordinate, in units of that
  coordinate's own standard error. `evo_lmm.convergence_statistics` computes it
  and the Newton decrement `sqrt(score' AI^-1 score)` alongside; both are
  reported in diagnostics. `tol` keeps only its two remaining roles: accepting
  a vanishing line-search displacement, and `gtol` for the exact dense
  finishing optimizer.
  Rationale: at a fixed statistical distance from the optimum, `||score||_inf`
  grows roughly like `sqrt(n)`, so an absolute score tolerance demands getting
  `sqrt(n)` times closer as data are added and means something different at
  every sample size. The new statistic is also the one the line search already
  used as its merit function, so the two are no longer inconsistent.
  Measured on a seeded two-category fixture at the generating parameters:
  `||score||_inf` went 1.31 -> 13.0 as `n` went 300 -> 2400, while
  `max_i |step_i|/SE_i` stayed between 0.5 and 2.6.
- **A rank-deficient information matrix is never scored as converged.**
  `psd_pseudo_inverse` inverts only eigenvalues above `1e-12 * lambda_max`;
  `numpy.linalg.pinv` inverts the small negative eigenvalues a stochastic AI
  carries and returns negative variances, which read as zero standard errors
  and hence as convergence. A fit whose `tau` ran away to `2.4e8` was declared
  converged that way during this work.
- **`status` replaces the single boolean.** `converged`,
  `stalled_near_tolerance`, `line_search_stalled`, `iteration_cap`,
  `not_started`, and — on paths that defer to SciPy — `optimizer_success` /
  `optimizer_failure`, plus `dense_finish`, `dense_finish_backstop`, and
  `dense_finish_failed` for the single-component exact finishing step. The
  back-stop is named separately because it declares convergence at
  `||score||_inf < 1e-4`, two orders of magnitude looser than the loop's own
  criterion.
- **`tau_c` boundary reporting, report only.** The multi-component fit reports
  each `tau_c` in a regime where it is unidentified, judged on the weights
  `w_j = 1/(1 + 2 tau_c q_j)`: `max_j (1 - w_j) <= 1e-6` (kernel
  indistinguishable from flat) or `max_j w_j <= 1e-6` (every weight saturated,
  only `tau_c * q_j` identified). A boundary hit does **not** change `status`
  or `converged`: a flat kernel is a legitimate estimate.

Deliberately **not** decided:

- Trace and CG error gate nothing, by decision. Probes are drawn once and held
  fixed, so the iteration converges to a stationary point of the *sketch*; the
  sketch-to-truth distance is a bias no tolerance on the sketch can see.
  Neither a probe-escalation stage nor a multi-seed agreement check was
  adopted, and no test is planned for trace-error-driven non-convergence.
- A near-singular AI can satisfy the criterion while the Newton decrement stays
  of order one: movement that is statistically irrelevant but an objective
  still improving. The decrement is logged and could become a second gate; it
  is not one.
- Only `tau_c` boundaries are reported. A scale coordinate pinned at its
  `+/-30` coordinate bound is not.
- The former *acceptance-gate* framing was removed by decision: this file sets
  no pass/fail bar for convergence at realistic data size. Evidence at that
  size does exist — see "Convergence at realistic data size" below — so the
  question is closed as measured, not as required.

The stochastic defaults remain the cheap end of the range: `trace_probes=12`
and `cg_tol=5e-4`, the latter matching GRAPP's solver budget so per-application
work is comparable. Larger probe budgets are optional verification settings,
not a production rule. Both fitters share the step rule — step capping,
Levenberg damping, and the near-convergence fallback — with one deliberate
difference: in the multi-component fitter a rejected step escalates the damping
and continues rather than ending the fit, because its AI matrix is near-singular
in the weakly identified `tau` directions on every fixture tried.

The documentation benchmark intentionally caps optimization at eight iterations
and reports secant/convergence warnings from the comparison path. That setup is
useful for timing and says nothing about convergence.

#### Convergence at realistic data size — measured

`benchmarks/multicomponent/` fits ten independent SLiM forward simulations at
`n = 2000` with three annotation categories and roughly 7,400-8,150 mutations,
GRGL-backed, at the production defaults (`cg_tol=5e-4`, 12 probes) from the
generic initial point. Results on this machine:

- **Convergence: 10/10 replicates return `status="converged"`.** This is the
  evidence the deleted acceptance gate asked for, and it is now the reason the
  question is closed rather than open.
- **The scales — the primary estimands — are recovered.** Mean and sample SD
  over the ten replicates, against the generating value:

  | quantity | truth | mean | SD | bias |
  | --- | --- | --- | --- | --- |
  | `sigma_e2` | `0.400` | `0.3834` | `0.0166` | `-4.1%` |
  | `sigma_b2_lof` | `2.250` | `2.2718` | `0.1173` | `+1.0%` |
  | `sigma_b2_missense` | `1.000` | `1.0795` | `0.0945` | `+8.0%` |
  | `sigma_b2_synonymous` | `0.250` | `0.3175` | `0.0946` | `+27.0%` |

  The two larger components are recovered well. The smallest one is biased
  upward by `27%` with a `30%` coefficient of variation; at ten replicates that
  is about `2.3` standard errors of the mean, so it is suggestive of a real
  small-component bias rather than established. Since `sigma_b_c^2` is the
  reported estimand, this is the number worth another look — more replicates
  would settle whether the bias is real.
- **Individual `tau_c` are not recovered — expected, and not a fitter defect.**
  Every replicate puts at least one category on the flat boundary (order
  `1e-8`) and at least one in the range `1.5` to `91`, against generating
  values of `1.125`, `0.5`, `0.125`. This is the identifiability result the
  model is built on (`notes/rare_variant.md` section 2.2), so it is recorded
  rather than chased. It was checked once, on replicates 1, 4 and 8, only to
  confirm the fitter is not masking a defect: each returned point is within
  `0.3-1.5` nats of the objective at the generating `tau_c` — better than the
  truth in replicate 8, worse in 1 and 4 — and profiling `tau[missense]` puts
  its minimum near `20`, `0` and `1` respectively. The fits sit at their own
  optima; it is the optimum's location in `tau_c` that is not stable across
  draws.

Runtime, same ten replicates: SLiM plus conversion `266-351 s`, fit
`26-137 s`. Reported as profiling, not as a performance guarantee.

- [ ] `benchmarks/multicomponent/infer.py` records `status` but not
  `fit.warnings`, so the `tau_c` boundary reports do not reach the CSV even
  though they fire on most of these replicates. Adding the column costs a rerun
  of the inference and plotting stages for all ten cached replicates.

`notes/rare_variant.md` still gates its Phase 3 on the convergence policy. That
file is not edited from here; its gate is now satisfiable by the measurement
above, but the wording is the owner's to reconcile.

#### Performance

Warm starts removed most redundant CG work, but the ten-replicate single-
component benchmark in `docs/tutorials/bolt_benchmark.rst` still shows evo-lmm
slower than GRAPP. Profile before adding more machinery. (The multi-component
benchmark under `benchmarks/multicomponent/` is a different artifact and has no
GRAPP comparison: GRAPP has no annotation-partitioned model to compare against.)

- [ ] Attribute remaining time to GRG traversals, derivative construction,
  trace queries, Python orchestration, LOCO setup, and optimizer evaluations.
- [ ] Record operator-call counts and time by inverse stage, not only aggregate
  CG iterations.
- [ ] Reduce repeated parameter-independent work across accepted iterations and
  line-search trials where profiling shows a material cost.
- [ ] Re-evaluate Hutchinson probe count using end-to-end estimate error and
  convergence, not trace microbenchmarks alone.
- [ ] Split stochastic REML into two trace-precision stages (deferred; not part
  of the retained convergence rule, and a cost/precision experiment rather than
  a correctness item). If it is ever built: a small sketch budget for the
  coarse steps and a larger refinement budget before convergence may be
  declared; refinement probes a deterministic superset of the sketch probes, so
  the switch adds information without discarding the optimizer's common random
  numbers; the transition driven by optimization state with a
  maximum-sketch-iteration fallback rather than a fixed iteration; CG
  warm-start caches revalidated at the transition; stage, probe counts,
  transition iteration, and operator-query counts recorded in diagnostics; and
  convergence judged only at refinement precision, including when the sketch
  stage appears converged by chance.
- [ ] Consider a changing-kernel Nyström preconditioner only if CG again becomes
  the dominant measured cost. Include sketch construction and refresh time in
  the comparison.

Acceptance gate, performance: report end-to-end time, estimate changes,
convergence status, and operator-equivalent work over multiple persisted
replicates. Partially met by `benchmarks/multicomponent/`, which reports
per-replicate simulation, fit, and total time, the estimates, and `status` over
ten cached replicates. Operator-equivalent work is still not recorded, so the
attribution items above stay open. For the two-stage item specifically, the gate is that estimates and
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

At the last audit `uv run pytest -q` collected 70 tests and all 70 passed on
darwin/arm64, including the `large` markers.
`tests/test_multicomponent_benchmark.py` covers the multi-component benchmark's
aggregation and plotting stage on synthetic rows; the forward simulations and
the fits themselves are run through the Snakefile, not the test suite.
`test_large_grg_matrix_free_fit_matches_dense_parameter_oracle` was once
reported to fail on aarch64 Linux with a stochastic-tolerance mismatch; it
passes here, so that report is unconfirmed on this machine. The
`sphinx-build` command cannot run in this environment: `docs/conf.py` loads the
`sphinx_design` extension, which is not in the `dev` dependency group — a
pre-existing dependency gap, not a docs error. Generated simulation artifacts
remain ignored under `docs/_artifacts/`.

## Definition of the next milestone

The next milestone is not another covariance-kernel prototype. It is a
calibrated, end-to-end CPU analysis path in which:

1. simplified evolutionary fits converge with clear diagnostics — the
   criterion and the `status`/boundary diagnostics are implemented
   (Priority 2, "Convergence policy — decided"); convergence at realistic data
   size is not demonstrated and, by decision, is not required here;
2. LOCO association uses the fitted evolutionary covariance and independent
   BOLT-normalized test columns (done);
3. null calibration and dense/GRG equivalence tests pass (done);
4. BLUP and association preserve the `rho^2 = 1` nested identity (done); and
5. the multi-replicate benchmark reports accuracy and runtime without rerunning
   forward simulations (done for the multi-component path:
   `benchmarks/multicomponent/` caches per replicate, so changing `infer.py`
   reruns inference and plotting only).
