# Evolutionary priors in GRGL-backed BOLT-LMM

## 1. Goal and deliverable

Implement two frequency-dependent random-effect priors on top of the
GRGL-backed BOLT-LMM-inf design in the pinned `grapp/` submodule:

1. A simplified evolutionary prior with two prior parameters,
   `sigma_b2` and `tau`.
2. A full evolutionary prior with three prior parameters, `sigma_b2`, `tau`,
   and `rho`.

Here

```text
q_j   = x_hat_j * (1 - x_hat_j)
tau   = sigma_a2 / W_S
r     = rho^2, with 0 <= r <= 1.
```

The per-variant raw-dosage effect variances are

```text
simplified: v_j = sigma_b2 / (1 + 2 * tau * q_j)

full:       v_j = sigma_b2 *
                    (1 - r * (2 * tau * q_j) / (1 + 2 * tau * q_j)).
```

Define shape weights `w_j = v_j / sigma_b2`. The simplified prior is exactly
the `r = 1` boundary of the full prior. The counts above describe the genetic
prior parameters; residual variance `sigma_e2` remains a nuisance LMM variance
component in both models and is fitted jointly by AI-REML.

The first implementation target is a library API with deterministic tests. CLI
exposure and GPU performance follow only after the CPU implementation is
numerically validated.

## 2. Reference implementation analyzed

The analysis below is against GRAPP commit
`be6e2419d6fef51a0a0c6ebe938646ded88c98a9`.

### 2.1 Pipeline

`grapp/grapp/assoc/bolt_lmm.py::bolt_lmm_inf` is the top-level pipeline:

1. Remove missing phenotypes and build the covariate projection.
2. Compute allele frequencies and per-variant norms with
   `compute_bolt_variant_stats`.
3. Build `BoltLmmOps`, which owns implicit per-chromosome `X`, whole-genome
   `X X^T`, and LOCO operators.
4. Estimate a single kernel-to-noise ratio with Monte Carlo REML and a secant
   search in `fit_bolt_variance_components`.
5. Solve LOCO systems, calibrate on sampled SNPs, and compute association
   statistics.

The useful source anchors are:

| Concern | GRAPP source |
| --- | --- |
| Orchestration | `grapp/grapp/assoc/bolt_lmm.py:105` |
| Variant means/norms | `grapp/grapp/assoc/bolt_inf_core.py:1366` |
| GRG operator assembly | `grapp/grapp/assoc/bolt_inf_core.py:453` |
| Kinship matvec and LOCO | `grapp/grapp/assoc/bolt_inf_core.py:694` |
| Monte Carlo genetic/noise probes | `grapp/grapp/assoc/bolt_inf_core.py:789` |
| Monte Carlo REML score | `grapp/grapp/assoc/bolt_inf_core.py:846` |
| One-dimensional variance fit | `grapp/grapp/assoc/bolt_inf_core.py:942` |
| LOCO calibration | `grapp/grapp/assoc/bolt_inf_core.py:1195` |
| Association statistics | `grapp/grapp/assoc/bolt_inf_core.py:1573` |
| Raw CPU `X`, `X^T`, and `X X^T` | `grapp/grapp/linalg/ops_scipy.py:133` |
| Raw GPU `X`, `X^T`, and `X X^T` | `grapp/grapp/linalg/ops_cupy.py:190` |
| CPU standardized GRG scaling | `grapp/grapp/linalg/ops_scipy.py:338` |
| Raw/standardized operator dispatch | `grapp/grapp/grg_calculator.py:21` |
| Existing BOLT regression test | `grapp/test/assoc/test_bolt_lmm.py:48` |

### 2.2 Conventional prior represented by GRAPP

GRAPP forms a sample-standardized genotype matrix `Z`. Its conventional
BOLT-LMM-inf kernel is effectively

```text
K_conv = (1 / M) * Z * Z^T,
H      = K_conv + delta * I,
delta  = sigma_e2 / sigma_g2.
```

`BoltLmmOps.apply_k` performs the `Z Z^T` product using GRG traversals and
divides by the active LOCO marker count. Monte Carlo probes draw marker effects
with scale `1 / sqrt(M)`, so their covariance is the same kernel. The current
REML code searches only `log(delta)` because the shape of `K_conv` is fixed.

The standardized operators receive `custom_variance` and apply
`sqrt(custom_variance ** alpha)` with `alpha = -1`. In the BOLT path,
`custom_variance` is the sample genotype variance, yielding unit-variance
columns. This parameter is a genotype scaling input, not an effect-variance
API.

### 2.3 Reusable pieces

Reuse these behaviors, either through a pinned compatibility adapter or a
small extraction into first-party evo-lmm code:

- `GRGCalcInterface`, schedulers, and NumPy/CuPy backend selection.
- Existing raw-dosage operators selected with
  `get_operator("X", standardized=False)` and
  `get_multi_operator("X", standardized=False)`.
- GRG allele-count/frequency traversals.
- `CovariateBasis` and projection semantics.
- Batched conjugate-gradient solves.
- Multi-chromosome operator scheduling and LOCO exclusion.
- Output annotation and the prospective/retrospective calibration algorithm.
- Existing GRAPP fixture GRGs as a conventional-prior regression reference.

Do not modify files inside the submodule. GRAPP's BOLT symbols are internal,
so pin its commit and isolate all imports in one `grapp_backend.py` module.

GRAPP already provides the unnormalized primitive needed by the evolutionary
kernel. `SciPyXOperator` and `CuPyXOperator` apply raw diploid dosage `X` or
`X^T`; their multi-GRG counterparts concatenate or sum chromosome operations.
They support mutation and sample filters, and accept per-mutation imputation
values for missing genotypes. They do not center columns or insert
variant-specific prior weights, which is appropriate: evo-lmm will perform
covariate projection outside the operator and place the weights between the
`X^T` and `X` passes.

The current operator table has no CPU multi-GRG unstandardized `X X^T`, while
the GPU table does. This is not a blocker because the weighted kernel cannot be
expressed by the unweighted `X X^T` operator anyway. Compose raw multi-`X`, or
prefer per-chromosome raw `X` operators and sum their results. The latter also
provides LOCO without relying on `set_exclude`, which the raw multi-`X`
operators currently lack.

### 2.4 Required architectural changes

The evolutionary model is not obtained by substituting `v_j` for GRAPP's
`custom_variance` while leaving the rest of the algorithm unchanged.

First, the notes define `beta_j` on raw diploid dosage. Let `X` be GRAPP's raw
diploid-dosage operator and let `P_C` be the orthogonal projector off the
covariate basis, including the intercept. Then `A = P_C X` is the
covariate-residualized raw-dosage matrix and the evolutionary covariance on the
restricted subspace is

```text
K(theta) = A * diag(w(theta)) * A^T,
V(theta) = sigma_b2 * K(theta) + sigma_e2 * I.
```

There is no default `1 / M` normalization in this expression: `sigma_b2` is
the per-locus focal-effect scale from the notes. Any numerical normalization
must be tracked and undone when reporting `sigma_b2`.

Second, GRAPP currently uses the same standardized operator for two different
roles:

- model columns that construct the random-effect covariance and MC probes;
- test-genotype columns used for calibration, scores, betas, and standard
  errors.

Those roles coincide under the conventional iid standardized-effect prior but
not under the evolutionary raw-effect prior. The implementation must hold two
logical operator families, both built from existing GRAPP primitives:

```text
B_theta = P_C X * diag(sqrt(w(theta)))  # raw prior-weighted model operator
T       = BOLT-normalized P_C X         # tested-genotype operator
```

Use `B_theta` for `K` matvecs, genetic probes, kernel traces, and REML
derivatives. Use `T` for calibration SNP columns, association scores, betas,
and standard errors.

Third, GRAPP's secant search estimates one scalar for a fixed kernel. Here the
kernel shape changes with `tau` and, in the full model, `r`. The new fitter
therefore solves the multi-parameter REML score equations with an
average-information update and fixed random probes across iterations.

## 3. Proposed first-party structure

Create the package rather than growing `main.py`:

```text
src/evo_lmm/
├── __init__.py
├── priors.py          Parameter objects, transforms, weights, derivatives
├── grg_data.py        Frequency extraction, sample masks, variant eligibility
├── operators.py       Weighted model kernel and unweighted test operators
├── reml.py            Matrix-free average-information REML solver
├── trace.py           Shared Hutchinson and optional XTrace estimators
├── bolt.py            End-to-end fit, LOCO solves, calibration, association
├── results.py         Typed fit diagnostics and association result objects
└── grapp_backend.py   All imports/adaptation from the pinned GRAPP commit

tests/
├── test_priors.py
├── test_operators_dense.py
├── test_reml_dense.py
├── test_reml_recovery.py
├── test_loco.py
└── test_grapp_regression.py
```

Add `grapp` to `pyproject.toml` as an editable path source only when this
package scaffold lands:

```toml
dependencies = ["grapp", "pygrgl", ...]

[tool.uv.sources]
grapp = { path = "grapp", editable = true }
pygrgl = { path = "grgl", editable = true }
```

## 4. Public model API

Use explicit model objects rather than boolean flags:

```python
SimplifiedPrior(sigma_b2: float, tau: float)
FullPrior(sigma_b2: float, tau: float, rho: float)
```

Both expose:

```python
weights(frequencies) -> ndarray
effect_variances(frequencies) -> ndarray
weight_derivatives(frequencies) -> mapping[str, ndarray]
validate() -> None
```

Optimization uses unconstrained coordinates but results expose scientific
parameters:

```text
log_delta -> delta = sigma_e2 / sigma_b2 > 0
log_tau   -> tau > 0
logit_r   -> r in (0, 1), reported as both rho2=r and rho=sqrt(r)
```

`sigma_b2` is profiled analytically at every shape-parameter iterate and
`sigma_e2 = delta * sigma_b2` is derived afterward; neither is a separate
optimizer coordinate.

Optimize `r = rho^2` directly because it is the identifiable quantity in the
variance formula. Accept `rho` at the user boundary and document that its sign
is not identifiable.

Analytic shape derivatives needed by REML are

```text
a_j = 2 * tau * q_j

simplified:
    w_j                  = 1 / (1 + a_j)
    d w_j / d log(tau)   = -a_j / (1 + a_j)^2

full:
    w_j                  = 1 - r * a_j / (1 + a_j)
    d w_j / d log(tau)   = -r * a_j / (1 + a_j)^2
    d w_j / d logit(r)   = -r * (1-r) * a_j / (1 + a_j)
```

## 5. Matrix-free REML design

### 5.1 Operator contract

Implement an `EvolutionaryLmmOps` object with the following explicit methods:

```text
apply_model_x(weights, theta, exclude_chrom=None)
model_scores(vector, theta, exclude_chrom=None)
apply_h(vector, phi, exclude_chrom=None)
apply_dh(vector, phi, parameter, exclude_chrom=None)
apply_k(vector, theta, exclude_chrom=None)
apply_dk(vector, theta, parameter, exclude_chrom=None)
solve_ph(rhs_columns, phi, exclude_chrom=None)
test_scores(chrom, vector)
test_column(chrom, local_idx)
kernel_trace(theta, exclude_chrom=None)
```

For each chromosome, construct GRAPP's existing raw operator with
`get_operator("X", standardized=False)` and cache frequencies, `q_j`, centered
raw-column norms, test normalization, model-variant masks, and missing-value
imputation inputs. Recompute only the length-`M` weights when parameters
change; never materialize `N x M` dosage.

The kernel and derivative matvecs use two raw GRG passes with projection on
both sides:

```text
u_p       = P_C u
s         = X^T u_p
K u       = P_C X [w    * s]
dK_k u    = P_C X [dw_k * s].
```

Use a per-chromosome raw `X` operator as the initial implementation. A
whole-genome matvec sums chromosome results; a LOCO matvec skips the excluded
chromosome. A future batched implementation may use raw multi-`X`, but it must
add an exclusion mechanism or maintain one multi-operator per LOCO set.

This direct use of GRAPP's unnormalized matvec is preferable to encoding
inverse effect variances through `custom_variance`: it keeps the raw-effect
units visible, supports analytic derivatives, and avoids confusing genotype
standardization with prior variance. The unweighted raw `X X^T` operator is
not sufficient because the evolutionary kernel needs `diag(w)` between the
two passes.

### 5.2 Nonlinear average-information REML

Use Zhu and Wathen, [*Essential formulae for restricted maximum likelihood and
its derivatives associated with the linear mixed
models*](../notes/1805.05188v1.pdf), Theorems 3 and 6, as the REML algorithmic
specification.

Let `C` denote the fixed-effect design matrix so that `X` remains the raw
genotype operator. Factor out the overall focal-effect scale:

```text
V = sigma_b2 * H(phi),
H(phi) = K(tau, r) + delta * I,
delta = sigma_e2 / sigma_b2,

P_V = V^-1 - V^-1 C (C^T V^-1 C)^-1 C^T V^-1.
```

The simplified model omits `r`. Following the scale-factorization opportunity
noted in TSLMM, optimize only the covariance-shape coordinates

```text
simplified: phi = (log_delta, log_tau)
full:       phi = (log_delta, log_tau, logit_r).
```

At every shape iterate, solve with `H`, let `P_H` be its REML projection, and
profile the overall scale exactly:

```text
d                = N - rank(C)
q                = y^T P_H y
sigma_b2_hat(phi)= q / d
sigma_e2_hat(phi)= delta * sigma_b2_hat(phi).
```

This removes one nonlinear coordinate and one stochastic trace while still
reporting `sigma_b2` and `sigma_e2` in their scientific units. The two/three
parameter counts continue to describe the genetic prior; residual variance is
the additional nuisance component represented by `delta` during fitting.

For a general coordinate `theta_i`, define `V_i = dV / dtheta_i`. The score
from Theorem 3 is

```text
S_i = 0.5 * [y^T P_V V_i P_V y - tr(P_V V_i)].
```

For implementation and validation, regard the full coordinate vector as
`theta = (log_sigma_b2, phi)`. Its first-derivative operators are

```text
V_log_sigma_b2 u = V u
V_log_delta u    = sigma_b2 * delta * u
V_log_tau u      = sigma_b2 * dK/dlog_tau u
V_logit_r u      = sigma_b2 * dK/dlogit_r u       # full model only.
```

Profiling makes the scale score exactly zero because
`tr(P_V V) = d`. Form the full average-information matrix, then eliminate the
scale row/column with its Schur complement before updating `phi`:

```text
AI_profile = AI_phi,phi
             - AI_phi,scale * AI_scale,scale^-1 * AI_scale,phi.
```

This is the AI analogue of optimizing the profiled restricted likelihood.

Our covariance is nonlinear in `tau` and `r`, so `V_ij = d2V /
dtheta_i dtheta_j` is generally nonzero. Theorem 6 shows that the usual
average-information term

```text
AI_ij = 0.5 * y^T P_V V_i P_V V_j P_V y
```

remains the essential information matrix: the difference between it and the
exact average of observed and Fisher information is

```text
IZ_ij = 0.25 * [tr(P_V V_ij) - y^T P_V V_ij P_V y],
E[IZ_ij] = 0.
```

Consequently, the nonlinear evolutionary GRM does not require second
derivatives for AI-REML. Implement only `V_i`/`dK_i` matvecs and solve

```text
AI(theta_k) * step_k = S(theta_k)
theta_(k+1) = theta_k + step_k.
```

### 5.3 Matrix-free AI iteration

Compute one iteration as follows:

1. Build the current per-variant weights and first derivatives of `H`.
2. Draw the trace probes once at fit initialization and keep them fixed. In one
   synchronized multi-right-hand-side projected-CG batch, solve

   ```text
   P_C H P_C [xi, p_1, ..., p_L] = P_C [y, z_1, ..., z_L]
   ```

   subject to every solution column lying in the covariate-orthogonal
   subspace. Thus `xi = P_H y` and `p_l = P_H z_l`. This is one batched GRGL
   `matmat` per CG iteration, not `L+1` independent GRGL traversals.
3. Profile `sigma_b2 = (y^T xi)/d` and derive `sigma_e2`.
4. For every full coordinate, including scale, compute `eta_i = V_i P_V y`.
   Equivalently use scale-free `H_i xi` and apply the required scalar factors
   in the small score/AI algebra.
5. In one second synchronized CG batch, compute all `zeta_i = P_H eta_i`.
6. Form the data quadratics and full AI matrix from small inner products, then
   use the scale Schur complement to obtain `AI_profile`.
7. Estimate every non-scale score trace from the already-solved common probes:

   ```text
   tr(P_H H_i) ~= (1 / L) * sum_l p_l^T (H_i z_l).
   ```

   This shares all inverse applications across parameters; each additional
   evolutionary parameter costs derivative matvecs but no additional CG
   solve. The scale trace is the exact value `d`.
8. Form the profiled score, symmetrize `AI_profile`, add minimal diagonal
   damping if its condition
   number is poor, solve for the step, and cap the step in transformed
   coordinates.
9. Use step halving until the scaled score norm decreases and `H` remains
   positive definite. Recompute `S` after every trial step.

Implement synchronized independent CG in the style used by TSLMM: all columns
share one `H.matmat(search_directions)` call per iteration, while recurrence
coefficients and residual tests remain per column. Remove converged columns
from the active block rather than stopping on one Frobenius norm. Warm-start
the phenotype, probe, and derivative solutions from the preceding accepted AI
iterate and from the preceding step-halving trial.

The complete AI iteration therefore has two sequential inverse stages: one for
`[y, probes]` and one for all derivative right-hand sides. Its expensive GRGL
traversal count depends primarily on CG iterations, not on the number of probes
or fitted prior parameters. The score requires stochastic traces, but the AI
matrix itself uses only first-derivative matvecs, the second CG batch, and small
dense inner products. This is the computational benefit emphasized by Theorem
6 and Algorithm 4 of the REML paper and realized with TSLMM-style batching.

Stop when both the maximum transformed-parameter step and the Fisher-scaled
score norm are below tolerance. Record the random seed, trace-probe count, CG
tolerance and iterations, AI condition number/damping, accepted step length,
score norm, active RHS count, and boundary hits. Increase the fixed probe count
and retry when trace Monte Carlo error prevents score convergence.

Use shared-probe Hutchinson as the default because it amortizes every `P_H`
solve across all derivative traces. Also implement TSLMM's XTrace strategy as
an optional high-accuracy mode. XTrace uses two operator-query blocks to build
and correct a randomized range approximation; enable it only when a pilot
comparison shows that it reaches the requested trace error with fewer total
weighted-GRGL matvec equivalents than shared Hutchinson. Report the estimator
and its empirical standard error.

Implement an exact dense REML oracle for small test data using

```text
l_R = -0.5 * [log|V| + log|C^T V^-1 C| + y^T P_V y + constant].
```

Use exact traces in its score. This oracle validates the matrix-free score,
AI matrix, iteration direction, parameter recovery, and nested-model
equivalence. Stochastic Lanczos log-determinants are optional diagnostics, not
part of the first production fitting path.

### 5.4 Matvec reduction and initialization

Adopt the applicable techniques from
[`tslmm/tslmm.py`](https://github.com/hanbin973/tslmm/blob/main/tslmm/tslmm.py):

- QR-orthonormalize covariates once and standardize the phenotype during
  numerical fitting; transform variance estimates and fixed effects back to
  original units in the result.
- Use synchronized multi-RHS CG so one GRGL `matmat` advances all RHS columns.
- Reuse `P_H y`, fixed trace probes, derivative products, and warm starts
  throughout an accepted AI iteration.
- Profile the global covariance scale instead of estimating a redundant scale
  coordinate stochastically.
- Offer randomized Haseman-Elston initialization.

For each initial `(tau, r)` shape, use a stochastic Haseman-Elston moment fit
to initialize `sigma_b2`, `sigma_e2`, and hence `delta`. With ordinary
covariate projection `P_C`, solve

```text
[tr(P_C K P_C K), tr(P_C K)] [sigma_b2] = [y^T P_C K P_C y]
[tr(P_C K),       d        ] [sigma_e2]   [y^T P_C y        ].
```

Use the same trace-estimator infrastructure as REML, reject negative moment
solutions, and fall back to an equal phenotype-variance split. This is an
initializer only, not the final estimator. The initial implementation uses
unpreconditioned synchronized CG. Changing-kernel Nyström preconditioning is
retained only as a future optimization in Section 9.

### 5.5 Identifiability and boundaries

- At `tau = 0`, both priors reduce to constant raw-effect variance and `r` is
  not identifiable.
- At `r = 0`, the full prior is frequency independent and `tau` is not
  identifiable.
- At `r = 1`, the full and simplified kernels are identical.
- Rare/monomorphic boundaries have `q_j = 0` and `w_j = 1`; monomorphic
  centered columns contribute zero regardless of their positive weight.

Return an explicit weak-identification warning when the optimizer reaches these
regions. Do not report standard errors for an unidentified shape parameter.

## 6. Integration with BOLT calibration and association

After fitting the whole-genome parameters:

1. Freeze `tau`, `r`, `sigma_b2`, and `sigma_e2`.
2. For each chromosome, solve the LOCO system using the evolutionary kernel
   with that chromosome excluded. Do not re-fit shape parameters per
   chromosome in the first implementation.
3. Reuse GRAPP's calibration selection logic, but obtain calibration columns
   from the unweighted test operator `T`.
4. Use evolutionary `V_LOCO^-1 y` residuals with `T^T` scores.
5. Keep GRAPP-compatible association columns while adding fitted-prior
   metadata to the fit result, not to every SNP row.

Audit GRAPP formulas that currently assume `fit.sigma_g2` and a standardized
kernel. Replace that meaning with the fitted `sigma_b2` in raw-effect units and
derive the calibration scale from `V` directly. Do not reuse
`h2_from_log_delta` unchanged; compute

```text
h2 = sigma_b2 * trace(P_C K P_C) /
     [sigma_b2 * trace(P_C K P_C) + sigma_e2 * d].
```

## 7. Implementation phases and acceptance gates

### Phase 0 - scaffold and pin interfaces

- Add the `src/evo_lmm` package, editable GRAPP dependency, and test tooling.
- Implement `grapp_backend.py` and assert the analyzed GRAPP commit/API at
  development time.
- Add a conventional-prior smoke test using the committed GRAPP fixture.

Gate: `uv sync`, `import evo_lmm`, `import grapp`, and the unchanged GRAPP BOLT
regression test pass.

### Phase 1 - priors and dense oracle

- Implement both parameter objects, validation, weights, and derivatives.
- Implement dense centered-dosage kernels and exact restricted likelihood.
- Add finite-difference derivative tests.

Gate: all boundary identities and analytic derivatives agree to tight numeric
tolerance; full `r=1` equals simplified for weights, kernels, and likelihood.

### Phase 2 - GRG weighted operators

- Wrap GRAPP's `get_operator("X", standardized=False)` per chromosome and
  implement projected raw `X`, weighted `B_theta`, derivative-kernel, and test
  operations for NumPy/CPU.
- Implement whole-genome and LOCO products by summing per-chromosome raw
  operator results, without marker-count renormalization.
- Compare every matvec with the dense oracle on small GRGs.

Gate: `A`, `A^T`, `K`, each `dK`, test columns, model scores, test scores, and
every LOCO kernel match dense multiplication.

### Phase 3 - matrix-free REML

- Implement matrix-free `P_H` actions with synchronized projected CG, including
  active-column convergence and warm starts.
- Batch `[y, trace_probes]` in the first inverse stage and every derivative RHS
  in the second stage.
- Profile `sigma_b2`, derive `sigma_e2`, and eliminate the scale coordinate
  from AI with a Schur complement.
- Implement exact data quadratics, Hutchinson score traces with fixed common
  probes, optional XTrace, and the nonlinear average-information matrix from
  Theorem 6.
- Add stochastic Haseman-Elston initialization.
- Add transformed-parameter AI updates, condition-based damping, step capping,
  score-norm step halving, convergence checks, and weak-identification
  diagnostics.
- Cache parameter-independent quantities and reuse common random probes.

Gate: matrix-free `P_H` actions and AI matrices match the dense oracle; scores
match within declared trace-estimation error; increasing the probe count
reduces that error; seeded runs are reproducible; and accepted AI steps reduce
the scaled score norm. Batched solves return the same columns as independent
solves, while requiring one GRGL operator call per synchronized CG iteration.

### Phase 4 - end-to-end evolutionary BOLT-LMM

- Add the driver for phenotype filtering, fitting, LOCO solves, calibration,
  association statistics, and typed results.
- Add seeded simulations under both priors and under frequency-independent
  limiting cases.

Gate: recover genetic covariance and prediction on simulations; null-trait
association statistics are calibrated; full `r=1` reproduces simplified output.

### Phase 5 - performance and GPU

- Profile GRG traversal, projected CG, derivative matvec, trace-probe, and AI
  solve costs before optimizing.
- Batch trace-probe and derivative right-hand sides, reuse score vectors, and
  add CuPy parity only after CPU correctness.
- Measure XTrace error per weighted-GRGL matvec equivalent; retain it only when
  it improves the full fit, not merely a microbenchmark.

Gate: memory remains `O(N * probes + M)` rather than `O(NM)`, CPU/GPU results
agree within tolerance, and whole-genome LOCO does not rebuild dense matrices.

## 8. Required tests

### Prior unit tests

- `tau=0 -> w=1` for both models.
- `r=0 -> w=1` for the full model.
- `r=1 -> w_full=w_simplified` exactly.
- `q=0 -> w=1`; weights remain positive for all valid inputs.
- Weights decrease monotonically with `q` and `tau` when coupling is nonzero.
- Analytic derivatives match central finite differences.

### Operator tests

- GRG matvecs match dense raw-dosage calculations.
- Kernels are symmetric positive semidefinite within tolerance.
- Model and test operators differ when weights vary, and coincide in the
  appropriate frequency-independent normalization test.
- LOCO equals dense removal of exactly the excluded chromosome.
- Missing-phenotype sample masks change frequencies and kernels consistently.

### Fitting tests

- Exact dense REML recovers parameters on moderate seeded simulations.
- Dense analytic scores match finite differences of the restricted
  log-likelihood for both nonlinear priors.
- Dense average-information entries equal their direct quadratic-form
  definitions and remain valid when second derivatives of `V` are nonzero.
- Matrix-free `P_H`, score, and AI results agree with the dense oracle within
  the declared trace Monte Carlo tolerance.
- Repeated simulations confirm that the omitted Theorem 6 second-derivative
  remainder is centered near zero.
- Matrix-free and dense parameter estimates agree within declared Monte Carlo
  tolerance.
- Fixed probes make repeated fits bitwise or tightly numerically reproducible.
- Increasing trace probes reduces score error, and an accepted AI iteration
  reduces the Fisher-scaled score norm.
- Profiled `sigma_b2` and derived `sigma_e2` match a joint dense REML fit.
- Synchronized multi-RHS CG matches independent dense/CG solves for every
  column and uses one covariance matmat per iteration.
- XTrace and Hutchinson both cover exact traces within their reported Monte
  Carlo uncertainty.
- Boundary/non-identification cases return diagnostics rather than unstable
  point estimates.
- Reported `sigma_b2` is invariant to any internal numerical rescaling.

### Regression and integration tests

- The conventional GRAPP path remains unchanged against its truth fixtures.
- Full `r=1` and simplified fits produce the same covariance and association
  outputs when initialized at the same point.
- Dense and GRG-backed association results agree on small data.
- Null simulations have controlled test-statistic inflation; predictive
  simulations compare BLUP accuracy across conventional, simplified, and full
  priors.

## 9. Future potential: changing-kernel preconditioning

Nyström preconditioning is not part of the initial implementation phases,
required tests, or definition of done. Reconsider it only after profiling
shows that unpreconditioned synchronized CG dominates total fit time.

If pursued, approximate a projected reference kernel as
`P_C K_ref P_C ~= U B U^T` and apply

```text
M^-1 b = delta^-1 * (b - U U^T b)
         + U (B + delta I)^-1 U^T b.
```

Unlike TSLMM's constant kernel, `K(tau, r)` changes during optimization. A
future implementation must therefore treat `(U, B)` as a possibly stale SPD
preconditioner, freeze it within every CG solve, and rebuild only between
accepted REML iterations. Updating `delta` is cheap and does not require a new
sketch; changing `tau` or `r` may change the dominant subspace.

Evaluate ranks `32`, `64`, and `128` against the unpreconditioned baseline.
Include sketch construction and every refresh in end-to-end timing. Consider a
refresh only when CG iterations exceed `1.5` times their post-build baseline
or residuals stall, and only when predicted remaining solve savings exceed
the measured rebuild cost. Required experimental checks would establish SPD,
solution preservation, deterministic refresh behavior, and a reduction in
total timed GRGL work, not merely iteration count.

## 10. Definition of done

The feature is complete when both priors can be selected through a typed
library API; all prior and nuisance parameters are fitted and reported in their
scientific units; no dense genotype matrix is used outside test or simulation
oracles; LOCO, calibration, association, and BLUP consume the fitted
evolutionary covariance; nested-model identities and dense-reference tests
pass; and the README documents inputs, parameter meanings, installation, and a
reproducible example.
