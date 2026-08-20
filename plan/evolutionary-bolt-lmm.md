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

### Annotation-partitioned kernel (MC0 complete for the current small-dense/GRGL scope)

- [x] Added `MultiComponentPrior` with one `SimplifiedPrior` per annotation
  category and transformed `(log_sigma_b2_c, log_tau_c)` coordinates.
- [x] Added `MultiComponentOps` with category-specific projected kernels,
  analytic component derivatives, PSD/symmetry coverage, and batched
  matrix-free kernel/derivative application.
- [x] Added an exact dense profiled-REML prototype that profiles the residual
  scale and reports category-specific `sigma_b2_c`, `tau_c`, and heritability.
- [x] Added deterministic tests for component PSD, derivative agreement, the
  `tau_c = 0` flat-kernel boundary, and a seeded multi-component fit.
- [x] GRGL-backed component application/materialization and shared batched
  derivative application are available for small exact fits.
- [x] Added production projected AI-REML with analytic component derivatives,
  profiled residual scale, CG solves, and selectable Hutchinson/XTrace traces.
- [x] Reuse one block-CG solve across all probe/RHS columns, with a robust
  dependent-column fallback, and add explicit single-category fit parity tests.
- [x] Add the small-dense fit-level boundary-ladder checks for all-tau-zero and
  shared-tau models, plus exact single-category parity at the configured
  default `cg_tol=5e-4`.

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
- [x] Both heritability conventions, per-MAF-bin genic-variance decomposition,
  generic delta-method SEs, tau profile-likelihood evaluation, boundary-mixture
  LRT p-values, and pooled-shape/per-gene-scale report objects.
- [x] Added marginal baseline fitting, Hutchinson-compatible joint trace
  estimation, fit-level Hessian covariance where available, and pooled
  gene-level reporting objects.
- [x] Integrate AI covariance into h² reporting and add pooled-shape/per-gene
  fitting/reporting helpers.
- [x] Complete scientific-scale delta-method SEs/profile intervals for every
  reported `sigma_b2_c` and `tau_c`.
- [x] Reproduce the complete RareEffect baseline independently, including its
  marginal ML conventions, on a small dense case.

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

- [ ] **MC0 — Annotation-partitioned multi-component kernel (simplified prior
  only).** Per-category prior objects
  with analytic derivatives in `(log sigma_b_c^2, log tau_c)`; multi-component
  AI-REML profiling one scale and searching the remaining `2|c|` shape
  coordinates; batched per-component derivative traversals reusing the shared CG
  solve and probes; PSD and symmetry tests per component and for the sum; exact
  nesting tests (all `tau_c = 0`, shared `tau`, single category).
- [x] **MC1 — Named baselines.** The flat per-category prior; RareEffect's
  marginal ML plus MoM-ratio adjustment including its negative-MoM truncation
  rule, reproduced faithfully; an optional MAC-threshold collapsing operator.
- [ ] **MC2 — Joint multi-component MoM / Haseman–Elston.** Generalize
  `haseman_elston_initialization()` to the `(|c|+1)`-dimensional moment system.
  Initialization matters more here than in the single-component case: six shape
  coordinates, several weakly identified by construction.
- [ ] **MC3 — Estimand adapters and reporting.** Both heritability conventions;
  per-MAF-bin genic-variance decomposition; scientific-scale delta-method
  standard errors plus profile likelihoods for each `sigma_b2_c` and `tau_c`;
  boundary-aware likelihood-ratio tests;
  gene-level output with pooled shape parameters.
- [ ] **MC4 — WES data path.** Exome pVCF/BGEN to GRG conversion; annotation
  masks as first-class variant partitions; MAC/MAF filters applied before
  frequency recomputation; measured GRG compression on exome rare variants.
  Needed only for the access-gated Phase 3, so it is the lowest priority here.

Order: MC0, then MC1, then MC2, then MC3. MC4 is independent and deferred.

Acceptance gate. All four clauses, no qualifiers:

1. **Every** nesting identity holds exactly — not "every implemented" one. An
   identity that is not yet tested is an open gate, not a satisfied one.
2. **Dense-versus-GRG equivalence at `cg_tol=1e-9`, on a small dataset.** This
   clause was deleted once and is restored deliberately. It is a small-data
   equivalence check by design: the point is to pin the GRG traversal against a
   dense oracle where the dense oracle is computable at all.
3. Single-category fitting delegates to the existing fitter bit-for-bit.
4. The reproduced RareEffect baseline matches an independent reimplementation on
   a small dense case.

**Tight tolerances are test-fixture settings, never a production rule.**
`cg_tol=1e-9` and `trace_probes=64` exist to make small-dataset verification
exact. They are not the route to a trustworthy large-scale estimate and must
never be documented as one — they do not scale, and presenting them as the
interim production path was an error in an earlier revision of these documents.
A reportable large-scale estimate requires the production convergence policy
(Priority 2), which is open.

Gate status, audited 2026-08-20 — **clauses 1 and 2 are not met.** Clause 1: the
`tau_c = 0` identity is verified against the evolutionary path with the scale
factored out rather than against the named flat baseline, the shared-`tau` and
fit-ladder tests compare identically-constructed priors, and `dK/dlog tau` has
no finite-difference check. Clause 2: no dense-versus-GRG test for the
partitioned kernel exists. Clauses 3 and 4 are met and independently confirmed.

Blocking defects found in the same audit, ahead of any further checkboxes:

- The multi-component fitter does not converge or recover parameters. At
  `n=2000` with two categories it returns `converged=False`, `h2 = 0.993`
  against a truth of `0.187`, and `tau` collapsed to zero; the exact profiled
  REML objective is `1050.6` at the returned point against `108.6` at the truth.
- ~~`multicomponent.py:550` propagates the `np.nan` initialiser of
  `last_sigma_e2` into `SimplifiedPrior`, raising `ValueError: sigma_b2 must be
  finite and strictly positive` on any loop exit that precedes the in-loop
  assignment.~~ **Fixed.** The reported state is now seeded from the initial
  point before the loop, and the line-search-rejection branch records its own
  iterate instead of breaking past the assignments. Two exits triggered it:
  `max_iter=0`, and a first-iteration line-search rejection — reachable simply
  by starting the fit at the generating parameters. Covered by
  `test_max_iter_zero_returns_seeded_diagnostics_instead_of_raising` and
  `test_first_iteration_line_search_rejection_reports_that_iterate`, both of
  which fail against the pre-fix source.

Diagnostic lead for the convergence defect, recorded while fixing the above:
started at the generating parameters on the `n=2000` case, the fitter reports
`h2 = 0.178` against a realized truth of `0.187` — i.e. the truth is close to
right — but `score_norm = 2.218` there and **every one of the ten step halvings
is rejected**. The score and the average-information matrix therefore disagree
about the descent direction at a point that is very nearly optimal. Suspect the
AI construction or the 12-probe trace noise before suspecting the step logic.

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

The stochastic defaults remain the cheap end of the range: `trace_probes=12` and
`cg_tol=5e-4`, the latter matching GRAPP's solver budget so per-application work
is comparable. The multi-component fitter reuses the single-component score/step
rule, including step capping, damping escalation, and the near-convergence
fallback — but reusing that rule is not the same as having a production
convergence policy, and the item below stays open.

`trace_probes=64` and `cg_tol=1e-9` are **small-dataset verification settings**.
They belong in equivalence and oracle tests and nowhere else. They are not a
production rule and are not an interim substitute for a convergence policy on
real-sized data; an earlier revision of these documents wrongly presented them
as the route to a reportable estimate.

The documentation benchmark intentionally caps optimization at eight iterations
and reports secant/convergence warnings from the comparison path. That setup is
useful for timing and is not a convergence policy.

- [ ] Define and test a production convergence policy independently of the
  matched-budget benchmark. Re-opened by the 2026-08-20 audit: the only test
  backing this passed `tol=1e9` against a true `score_norm` of 2.92, so it
  cannot fail, and the fitter demonstrably does not converge at `n=2000`.
  Transferring the single-component score/step rule is a prerequisite, not the
  policy.
- [ ] Add explicit tests for trace-error-driven non-convergence and any retry or
  probe-budget escalation policy. These remain deferred because the chosen
  policy keeps the sketch tolerance and does not add automatic refinement.

- [ ] Attribute remaining time to GRG traversals, derivative construction,
  trace queries, Python orchestration, LOCO setup, and optimizer evaluations.
- [ ] Record operator-call counts and time by inverse stage, not only aggregate
  CG iterations.
- [ ] Reduce repeated parameter-independent work across accepted iterations and
  line-search trials where profiling shows a material cost.
- [ ] Re-evaluate Hutchinson probe count using end-to-end estimate error and
  convergence, not trace microbenchmarks alone.
- [ ] Split stochastic REML into two trace-precision stages (deferred; not part
  of the retained convergence rule):
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
  Deferred from the current target: the retained sketch budget is the selected
  convergence budget, so this item is a future cost/precision experiment.
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
