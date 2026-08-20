# Variance-component fitting: GRAPP BOLT-LMM-inf versus evo-lmm AI-REML

Last audited: 2026-08-20

This note compares the two variance-component fitters that this repository
depends on:

- **GRAPP BOLT-LMM-inf**, the pinned reference implementation in
  `grapp/grapp/assoc/bolt_inf_core.py`, which reproduces BOLT-LMM v2.5's
  `Bolt::computeStats` / MC-scaling pipeline on GRG data.
- **evo-lmm AI-REML**, the fitter in `src/evo_lmm/reml.py`, which fits the
  evolutionary frequency-weighted kernel by profiled average-information REML.

They are not two implementations of one algorithm. They estimate different
numbers of parameters, they use stochastic randomness for different purposes,
they declare convergence with different criteria, and their per-iteration cost
models differ. This note states each precisely, in the code's own units, so the
comparison in `docs/tutorials/bolt_benchmark.rst` can be read correctly.

Sections that name a tunable end with a parameter table giving the symbol, the
keyword argument, the default, and the code location.

## 1. Common ground

Both fitters work on the same three objects.

Let $N$ be the number of retained individuals, $C$ the orthonormal covariate
basis (always including an intercept) with $\mathrm{rank}(C) = c$, and

$$
d = N - c
$$

the residual dimension (`ops.dim` in both codebases). Let

$$
P_C = I_N - C C^{\top}
$$

be the covariate projection, and let $y$ be the phenotype with $P_C y$ its
projection. Both fitters restrict every operator, norm and solve to the
non-missing individuals, and both recompute allele frequencies after that
restriction.

Both write the model covariance as a **scale times a shape**,

$$
V(\theta) = \sigma^2 \, H(\theta), \qquad H(\theta) = K(\theta) + \delta I_N ,
$$

and both **profile out the scale**: given the shape $\theta$, the scale is not
searched over but computed in closed form from the quadratic form of the data.
Both solve every $H^{-1}$ application by projected conjugate gradients — no
$N \times N$ matrix is formed, and no $N \times M$ dosage matrix is
materialised.

Everything else differs.

### 1.1 The two kernels are not the same object

GRAPP uses the BOLT-standardized kernel

$$
K_{\text{grapp}} = \frac{1}{M_{\text{loco}}}
P_C X_{\text{std}} X_{\text{std}}^{\top} P_C ,
\qquad
(X_{\text{std}})_{ij} = \frac{g_{ij} - 2\hat{x}_j}{\sqrt{\widehat{\mathrm{var}}_j}},
$$

with the explicit $1/M$ marker-count normalization
(`BoltLmmOps.apply_k`). Its scale parameter $\sigma_g^2$ is therefore a
*standardized* genetic variance, and $\delta_{\text{grapp}} = \sigma_e^2 / \sigma_g^2$.

evo-lmm uses the evolutionary frequency-weighted kernel on **raw diploid
dosage**,

$$
K_{\text{evo}}(\tau,\rho^2) = P_C X \, \mathrm{diag}\!\big(w_j(\tau,\rho^2)\big) X^{\top} P_C ,
$$

$$
q_j = \hat{x}_j (1 - \hat{x}_j), \qquad
w_j^{\text{simp}} = \frac{1}{1 + 2\tau q_j}, \qquad
w_j^{\text{full}} = 1 - \rho^2 \frac{2\tau q_j}{1 + 2\tau q_j},
$$

with **no** $1/M$ normalization and no genotype standardization. Its scale
$\sigma_b^2$ is a per-locus raw-dosage effect variance and
$\delta_{\text{evo}} = \sigma_e^2 / \sigma_b^2$.

Consequence: $\delta$, $h^2$, and the profiled scale are **not comparable
numbers across the two fitters**, even on identical data. GRAPP maps

$$
h^2 = \frac{\lVert X_{\text{std}} \rVert_F^2}
{\lVert X_{\text{std}} \rVert_F^2 + M_{\text{proj}} \, d \, e^{\log \delta}}
$$

(`h2_from_log_delta`, `bolt_inf_core.py:755`), while evo-lmm maps

$$
h^2 = \frac{\sigma_b^2 \, \mathrm{tr}(P_C K_{\text{evo}} P_C)}
{\sigma_b^2 \, \mathrm{tr}(P_C K_{\text{evo}} P_C) + \sigma_e^2 d}
$$

(`fit_reml`, `reml.py:616`). Only $h^2$ is on a shared scale; $\delta$ is not.

### 1.2 The search space differs in dimension

| | GRAPP | evo-lmm |
| --- | --- | --- |
| searched coordinates | $\log \delta$ (1) | $(\log \delta, \log \tau)$ (2, simplified) or $(\log \delta, \log \tau, \mathrm{logit}\,\rho^2)$ (3, full) |
| profiled scale | $\sigma_g^2$ | $\sigma_b^2$ |
| kernel shape | fixed | **estimated** ($\tau$, and $\rho^2$ in the full model) |

This is the structural difference from which most of the others follow. GRAPP
solves a one-dimensional root-finding problem in $\log \delta$ with a fixed
kernel. evo-lmm must estimate the kernel's frequency weighting jointly with
$\delta$, so it needs derivatives of $K$ with respect to the shape parameters
and a multi-dimensional Newton-type step.

## 2. GRAPP BOLT-LMM-inf: MC-scaling REML by secant root finding

GRAPP does not differentiate a likelihood. It solves a **moment-matching
equation** built from Monte-Carlo replicates simulated under the current
model — BOLT's "MCscaling" procedure.

### 2.1 The estimating equation

Fix $\log \delta$ and let $\delta = e^{\log \delta}$. Define the two data
statistics from a single solve $z = H^{-1} P_C y$:

$$
B_{\text{data}} = \sum_c \big\lVert X_{\text{std},c}^{\top} z \big\rVert^2 ,
\qquad
E_{\text{data}} = \lVert z \rVert^2 ,
$$

where $B_{\text{data}}$ is the summed squared standardized score
(`_sum_score_squares`, `bolt_inf_core.py:762`) and $E_{\text{data}}$ the squared
residual norm. For $T$ Monte-Carlo replicates $t = 1,\dots,T$, draw

$$
w_{c,t} \sim \mathcal{N}\!\left(0, \tfrac{1}{M_{\text{proj}}} I_{M_c}\right),
\qquad
\varepsilon_t \sim \mathcal{N}(0, I_N),
$$

form the simulated phenotype under the current model,

$$
y_t = P_C\!\left(\sum_c X_{\text{std},c} w_{c,t} + \sqrt{\delta}\,\varepsilon_t\right),
$$

solve $z_t = H^{-1} y_t$, and define $B_t, E_t$ from $z_t$ exactly as above.
The REML estimating function is the log ratio of ratios

$$
f(\log \delta) \;=\;
\log \frac{B_{\text{data}} / E_{\text{data}}}
{\left(\sum_{t} B_t\right) / \left(\sum_{t} E_t\right)} ,
$$

and the estimate is the root $f(\log \hat{\delta}) = 0$
(`_compute_mc_scaling`, `bolt_inf_core.py:846`). The interpretation is direct:
at the correct $\delta$, the data's genetic-to-residual score ratio matches the
ratio that data simulated under that same $\delta$ would produce.

The profiled scale is

$$
\hat{\sigma}_g^2 = \frac{y^{\top} H^{-1} P_C y}{d},
$$

evaluated at the accepted $\log \delta$ (`sigma2_k`), and
$\hat{\sigma}_e^2 = \delta \hat{\sigma}_g^2$.

### 2.2 Common random numbers

The probes $\{w_{c,t}, \varepsilon_t\}$ are drawn **once**, before the search,
from `np.random.default_rng(seed + 1)` in a fixed order (all weight blocks in
chromosome order, then all noise vectors), and reused unchanged at every
$\log \delta$ (`_generate_bolt_mc_components`, `bolt_inf_core.py:789`).
Therefore $f$ is a *deterministic* function of $\log \delta$ given the seed, and
the root find is an ordinary deterministic secant iteration rather than a
stochastic-approximation scheme. Different seeds give different roots; the same
seed always gives the same root.

### 2.3 The secant iteration and its exit rules

1. Evaluate at $\log \delta$ corresponding to $h^2 = 0.25$.
2. Evaluate at $h^2 = 0.125$ if $f < 0$ at the first point, else $h^2 = 0.5$.
   The bracket start uses `log_delta_from_h2` (`bolt_inf_core.py:746`).
3. Order the two points so that `cur` has the smaller $|f|$; then iterate at
   most **5** secant steps

$$
\log \delta_{\text{next}} =
\frac{\log \delta_{\text{prev}} f_{\text{cur}} - \log \delta_{\text{cur}} f_{\text{prev}}}
{f_{\text{cur}} - f_{\text{prev}}},
\qquad
\log \delta_{\text{next}} \leftarrow \mathrm{clip}(\log \delta_{\text{next}}, -10, 10).
$$

4. Stop early when the current point is the best found so far **and**
   $|\log \delta_{\text{next}} - \log \delta_{\text{cur}}| < 0.01$; stop also if
   $|f_{\text{cur}} - f_{\text{prev}}| < 10^{-300}$.
5. If neither early exit fires within 5 steps, log
   `"Secant iteration for h2 estimation may not have converged"` and **accept
   the best iterate anyway**.

So the total number of $f$ evaluations is at most $2 + 5 = 7$, each costing one
batched CG solve with $T + 1$ right-hand sides. There is no likelihood value, no
score vector, no information matrix, and no line search.

### 2.4 Uncertainty: delete-one jackknife over replicates

For each $j = 1,\dots,T$, $f$ is recomputed with replicate $j$ removed from the
denominator sums (`f_jacks`), and each replicate is also scored as if it were
the data (`f_rands_as_data`). These are diagnostics of Monte-Carlo noise in the
estimating function; they are not used to modify the accepted $\log \delta$.
The analogous jackknife *is* consequential later, in calibration, where a
standard error above `0.01` switches the calibration factor from the ratio of
sums to the ratio of medians.

### 2.5 CG in GRAPP

`bolt_conj_grad_solve` (`bolt_inf_core.py:263`) mirrors BOLT-LMM v2.5
exactly, which matters for three reasons:

- **Full batch, no active-column masking.** Every right-hand side is iterated
  until *all* columns satisfy the relative tolerance; converged columns keep
  being multiplied.
- **No denominator guard** on $p^{\top} H p$, and **no exception** when
  `max_iter` is reached — it returns whatever iterate it has, having recorded
  the achieved relative residual in `CgStats`.
- **Relative stopping rule** $\lVert r_k \rVert / \lVert r_0 \rVert \le$ `cg_tol`,
  with $r_0 = P_C b$ from a zero start. There is **no warm starting** anywhere:
  every solve starts from $x = 0$.

The tolerance is loose by design: `cg_tol = 5e-4`.

### 2.6 GRAPP parameters

| Symbol | Keyword | Default | Where |
| --- | --- | --- | --- |
| $T$, MC replicates | `mc_trials` | `0` $\Rightarrow$ auto: $T = \max(\min(\lfloor 4\times10^9 / N^2 \rfloor, 15), 3)$; a positive value is clamped to $\ge 2$ | `DEFAULT_H2_EST_MC_TRIALS`, `fit_bolt_variance_components:942` |
| CG relative tolerance | `cg_tol` | `5e-4` | `DEFAULT_CG_TOL` |
| CG iteration cap | `max_iter` | `10000` (no error on exhaustion) | `DEFAULT_MAX_ITERS` |
| probe seed | `seed` | `12345` in the core (`BOLT_RANDOM_SEED`); the CLI passes `42`. MC probes use `seed + 1`; calibration selection uses `seed + 321` | `bolt_inf_core.py:33`, `cli/bolt_lmm_cli.py` |
| secant step cap | — | 5 steps, $\log \delta$ clipped to $[-10, 10]$ | `fit_bolt_variance_components` |
| secant exit | — | best-point rule with $\lvert \Delta \log \delta \rvert < 0.01$ | same |
| bracket start | — | $h^2 = 0.25$, then $0.125$ or $0.5$ | same |
| calibration variants | `num_calib_snps` | `30` | `DEFAULT_NUM_CALIB_SNPS` |
| threads | `threads` / `-j` | `1` | `bolt_lmm_inf` |

Note that `mc_trials = 0` makes the *statistical* precision of the fit a
function of $N$: with $N = 20{,}000$ the auto rule gives $T = 10$; above
$N \approx 26{,}000$ it saturates at the floor $T = 3$.

## 3. evo-lmm: profiled average-information REML

evo-lmm forms the restricted likelihood's score and an average-information
approximation to its curvature, and takes damped Newton steps in transformed
coordinates.

### 3.1 Coordinates and the profiled likelihood

The searched vector is

$$
\phi = (\log \delta, \log \tau) \quad \text{or} \quad
(\log \delta, \log \tau, \mathrm{logit}\,\rho^2),
$$

chosen so that positivity ($\delta, \tau > 0$) and the box constraint
($\rho^2 \in [0,1]$) hold automatically. With
$P_H = H^{-1} - H^{-1} C (C^{\top} H^{-1} C)^{-1} C^{\top} H^{-1}$ and
$\xi = P_H y$, the profiled scale and objective are

$$
\hat{\sigma}_b^2(\phi) = \frac{y^{\top} P_H y}{d},
\qquad
\ell(\phi) = \tfrac{1}{2}\Big(
\log \lvert H \rvert + \log \lvert C^{\top} H^{-1} C \rvert
+ d \log \frac{y^{\top} P_H y}{d} \Big),
$$

minimised over $\phi$ (`_profile_objective_dense`, `reml.py:223`). Note
$\ell$ is only *evaluated* on the dense/exact path; the matrix-free path never
computes $\log\lvert H \rvert$ and reports `objective = nan`.

### 3.2 Score and average information

For each coordinate $\phi_i$ with $\partial_i H = \partial H / \partial \phi_i$
(analytic, from `priors.weight_derivatives`; for $\log\delta$ it is simply
$\delta I$):

$$
s_i = \frac{1}{2}\left(
\frac{\xi^{\top} (\partial_i H) \xi}{\hat{\sigma}_b^2}
- \mathrm{tr}\!\big(P_H \, \partial_i H\big)
\right),
$$

$$
\mathcal{I}^{\text{AI}}_{ij} =
\frac{1}{2}\,\frac{\xi^{\top} (\partial_i H) P_H (\partial_j H) \xi}
{\hat{\sigma}_b^2}
\;-\;
\frac{\big(\xi^{\top} (\partial_i H) \xi\big)\big(\xi^{\top} (\partial_j H) \xi\big)}
{2\, \hat{\sigma}_b^2 \, y^{\top} P_H y},
$$

then symmetrised. The average information is the average of observed and
expected information; the second term is the Schur complement that removes the
profiled scale coordinate, so $\mathcal{I}^{\text{AI}}$ is the curvature *after*
$\sigma_b^2$ has been eliminated (`_quantities`, `reml.py:237`;
`profiled_average_information`, `reml.py:165`). The step is

$$
\Delta \phi = \big(\mathcal{I}^{\text{AI}} + \lambda I\big)^{-1} s .
$$

Only the first quadratic form is cheap. The trace term is where the stochastic
estimator enters.

### 3.3 The trace estimator

$\mathrm{tr}(P_H \partial_i H)$ is estimated by Hutchinson's identity with
$S$ shared **Rademacher** probes $Z = [z_1,\dots,z_S]$,
$z_{s,k} \in \{-1,+1\}$ i.i.d., drawn once per fit from
`np.random.default_rng(seed)` (`rademacher_probes`, `trace.py:21`):

$$
\widehat{\mathrm{tr}}\big(P_H \partial_i H\big)
= \frac{1}{S} \sum_{s=1}^{S}
\big(H^{-1} P_C z_s\big)^{\top} \big(\partial_i H\, P_C z_s\big),
\qquad
\widehat{\mathrm{se}} = \frac{\mathrm{sd}_{s}(\cdot)}{\sqrt{S}} ,
$$

with the per-coordinate standard error recorded in
`FitDiagnostics.trace_standard_errors`. Two implementation points matter:

- **The probes are fixed for the entire optimization run**, not redrawn per
  iteration. This is the same common-random-numbers discipline GRAPP uses for a
  different purpose: it makes the score a deterministic function of $\phi$, so
  the Newton iteration is not chasing fresh noise and can actually converge.
- **The solve is shared.** One CG call solves
  $H^{-1} P_C [\,y \mid Z\,]$ — one right-hand side for the phenotype and $S$
  for the probes — and the derivative products
  $\partial_i H \, P_C Z$ go through a single batched `apply_dh_matmat`
  traversal per coordinate rather than $S$ separate GRG traversals
  (`operators.py:645`).

The alternative spherical **XTrace** estimator is available via
`trace_method="xtrace"`. It uses Gaussian probes normalised to radius
$\sqrt{N}$ and needs **two** operator query blocks per coordinate ($A\Omega$
and $AQ$ where $Q$ is the QR factor of $A\Omega$), hence
`trace_operator_queries = 2S` instead of $S$. It is retained for experiments;
equal-cost comparisons have not shown a consistent error advantage over
Hutchinson, which is why Hutchinson is the default.

On dense inputs the default is `exact = True`: traces are computed exactly from
$P_H \partial_i H$ and no probes are used at all
(`trace_estimator = "exact"`, `trace_probes = 0`).

### 3.4 Step control

Each iteration:

1. Evaluate score and AI at $\phi$.
2. **Convergence test:** stop when
   $\lVert s \rVert_\infty \le$ `tol` and either the previously accepted step
   was $\le$ `tol` or this is not the first iteration.
3. **Damping:** if the AI is non-finite or its condition number exceeds
   $10^{12}$, raise $\lambda$; on a failed iteration $\lambda$ starts at
   $10^{-6}$ and is multiplied by $10$.
4. **Step capping:** rescale so $\lVert \Delta \phi \rVert_\infty \le$
   `max_step` ($= 2.0$ in transformed coordinates, i.e. at most a factor
   $e^2$ change in $\delta$ or $\tau$ per iteration).
5. **Step halving:** try $\Delta \phi \cdot 2^{-k}$ for $k = 0,\dots,11$ and
   accept the first trial whose scaled score norm does not increase — and, on
   the exact path, whose objective does not increase beyond $10^{-10}$. Each
   trial is a full re-evaluation, so one iteration can cost up to 12
   evaluations.
6. If nothing is accepted, raise $\lambda$; declare convergence anyway if
   $\lVert s \rVert_\infty \le 10\,$`tol`.

On the exact dense path only, a non-converged run is finished by
`scipy.optimize.minimize(method="L-BFGS-B")` on $\ell(\phi)$ with the analytic
gradient, bounds $\pm 30$ on $\log \delta$ and $\log \tau$ and $\pm 20$ on
$\mathrm{logit}\,\rho^2$, `maxiter = 4 * max_iter`, `ftol = 1e-12`,
`gtol = tol`. The matrix-free path has no such fallback.

When the full model is started at $\rho^2 = 1$ exactly, the third coordinate is
frozen and only $(\log \delta, \log \tau)$ are updated, which is what preserves
the exact nested-model identity with the simplified model.

### 3.5 CG in evo-lmm

`EvolutionaryLmmOps.solve_ph` (`operators.py:679`) differs from GRAPP's CG on
every point listed in §2.5:

- **Synchronised multi-RHS CG with active-column masking**: a column that
  reaches its target leaves the active set and stops being multiplied.
- **Per-column relative target**
  $\lVert r \rVert^2 \le \max(\lVert r_0 \rVert^2, 1) \cdot$ `cg_tol`$^2$,
  with `cg_tol = 1e-9` by default — roughly six orders of magnitude tighter
  than GRAPP's `5e-4`.
- **Denominator guard** on $p^{\top} H p$, and a
  `numpy.linalg.LinAlgError` **raised** if any column is still unconverged at
  the iteration cap `max(50, 4N)`. A failed solve is caught by the optimizer,
  which raises damping and retries rather than silently accepting a bad solve.
- **Warm starts, enabled by default.** Solutions are cached under stable keys
  (`"phenotype+trace"`, `"derivative"`, and per-coordinate XTrace keys) and
  reused as the initial iterate at the next parameter point. Two safeguards
  keep this honest: the accepted-iterate cache and the line-search trial cache
  are kept separate, and each cached column is validated against a zero start
  — if its residual is worse, that column alone is reset and counted in
  `cg_warm_start_rejections`. The requested `cg_tol` is unchanged by warm
  starting.

### 3.6 Optional Haseman–Elston initialization

`initialization="he"` replaces the default starting $\delta$ with a projected
moment estimate: solve

$$
\begin{pmatrix} a & b \\ b & d \end{pmatrix}
\begin{pmatrix} \sigma_b^2 \\ \sigma_e^2 \end{pmatrix}
=
\begin{pmatrix} (P_C y)^{\top} K (P_C y) \\ (P_C y)^{\top} (P_C y) \end{pmatrix},
\qquad
a = \mathrm{tr}\big((P_C K P_C)^2\big), \quad b = \mathrm{tr}(P_C K P_C),
$$

and carry only $\delta = \sigma_e^2 / \sigma_b^2$ into the transformed
coordinates; the shape parameters keep their requested values and the scale is
profiled anyway. $a$ is exact for dense inputs and estimated by XTrace for GRG
inputs. A non-finite or non-positive solution falls back to a variance-based
guess and records a warning. The default remains `"default"`.

### 3.7 evo-lmm parameters

| Symbol | Keyword | Default | Where |
| --- | --- | --- | --- |
| $S$, trace probes | `trace_probes` | `64`, clamped to $\ge 2$; ignored when `exact` | `fit_reml:390` |
| trace estimator | `trace_method` | `"hutchinson"` (Rademacher); `"xtrace"` uses spherical Gaussian probes of radius $\sqrt{N}$ and $2S$ queries | `trace.py` |
| exact traces | `exact` | `None` $\Rightarrow$ `True` iff every chromosome is dense | `fit_reml` |
| probe seed | `seed` | `0`; probes drawn once per fit and reused at every $\phi$ | `rademacher_probes:21` |
| CG relative tolerance | `cg_tol` | `1e-9`, per column, error on non-convergence | `solve_ph:679` |
| CG iteration cap | — | $\max(50, 4N)$ | `solve_ph` |
| warm starts | `warm_start` | `True`, with per-column revalidation and accepted/trial cache isolation | `solve_ph`, `_quantities` |
| Newton iterations | `max_iter` | `50` accepted iterations | `fit_reml` |
| score tolerance | `tol` | `1e-6` on $\lVert s \rVert_\infty$ | `fit_reml` |
| step cap | `max_step` | `2.0` in transformed coordinates | `fit_reml` |
| step halvings | — | 12 per iteration ($2^{-k}$, $k \le 11$) | `fit_reml` |
| AI damping | — | starts $10^{-6}$, $\times 10$ per failure; forced when $\mathrm{cond} > 10^{12}$ | `fit_reml` |
| initialization | `initialization` | `"default"`; `"he"` sets the starting $\delta$ only | `haseman_elston_initialization:659` |
| dense finishing | — | L-BFGS-B, `ftol=1e-12`, `gtol=tol`, `maxiter=4*max_iter`, exact path only | `fit_reml` |

## 4. Side-by-side

| Aspect | GRAPP BOLT-LMM-inf | evo-lmm AI-REML |
| --- | --- | --- |
| Estimated shape parameters | $\log \delta$ only | $\log \delta, \log \tau$ (+ $\mathrm{logit}\,\rho^2$) |
| Kernel | fixed, standardized, $1/M$-normalized | estimated frequency weighting, raw dosage, no $1/M$ |
| Criterion | root of the MC-scaling function $f(\log\delta)$ | stationary point of the profiled restricted likelihood |
| Uses derivatives of $K$? | no | yes, analytic $\partial K / \partial \tau$, $\partial K / \partial \rho^2$ |
| Role of randomness | simulate $T$ replicate phenotypes under the current model | estimate $\mathrm{tr}(P_H \partial_i H)$ by Hutchinson |
| Randomness reused across the search? | yes (drawn once, `seed + 1`) | yes (drawn once, `seed`) |
| Search method | 1-D secant, $\le 5$ steps, clipped to $[-10,10]$ | damped Newton with step cap and $\le 12$ halvings |
| Convergence rule | best point and $\lvert \Delta \log \delta \rvert < 0.01$; otherwise warn and accept | $\lVert s\rVert_\infty \le 10^{-6}$; otherwise `converged = False` |
| Non-convergence behaviour | warns, returns best iterate | flags `converged=False` in diagnostics; exact path retries with L-BFGS-B |
| Likelihood value available? | never | yes on the exact path, `nan` matrix-free |
| Uncertainty reported | delete-one jackknife of $f$; `CgStats` | per-coordinate trace standard errors, AI condition, damping, CG residual norms |
| CG tolerance | `5e-4`, relative, batch-wide | `1e-9`, relative, per column |
| CG failure | silent, records residual | raises, optimizer damps and retries |
| Warm starts | none | default, with revalidation |
| Evaluations per fit | $\le 7$ | $\le$ `max_iter` accepted, each up to 12 trials |

## 5. Cost per evaluation

Write one *operator application* as one $X^{\top}v$ plus one $Xw$ traversal over
the retained markers (for GRGs, one up-sweep each), and let $p$ be the number of
searched coordinates ($p = 1$ for GRAPP; $p = 2$ or $3$ for evo-lmm).

**GRAPP, per $f$ evaluation:** one batched CG solve with $T + 1$ right-hand
sides. At `cg_tol = 5e-4` the iteration count $\kappa$ is small; total work is
$\kappa \cdot (T+1)$ applications, plus $T+1$ score reductions. With the whole
fit capped at 7 evaluations and no derivative kernels, the *entire*
variance-component stage costs on the order of $7 \kappa (T+1)$ applications.

**evo-lmm, per `_quantities` call:**

- one CG solve with $1 + S$ right-hand sides (phenotype and probes);
- one CG solve with $p$ right-hand sides (the derivative vectors $\eta$);
- $p$ batched `apply_dh_matmat` calls on the $S$ probe columns;
- $p$ `apply_dh` calls on $\xi$, and $p^2$ `apply_dh` calls on the columns of
  $\zeta$ for the AI matrix.

With defaults $S = 64$, $p = 2$, and `cg_tol = 1e-9` needing a substantially
larger $\kappa'$ than GRAPP's $\kappa$, a single evo-lmm evaluation is far more
expensive than a single GRAPP evaluation, and there can be up to 50 accepted
iterations rather than 7 total. Warm starts recover much of the difference by
cutting $\kappa'$ on later iterations — that is exactly what they were added
for — but the asymmetry in the cost model, not an implementation defect, is the
main reason the benchmark in `docs/tutorials/bolt_benchmark.rst` shows evo-lmm
slower than GRAPP for a fit of comparable quality.

## 6. Practical consequences

**Do not compare $\delta$ or the profiled scale across the fitters.** Compare
$h^2$, predictions, or association output. `FitResult.sigma_g2` exists only as a
compatibility alias and is `sigma_b2`, a raw-effect variance; it is not GRAPP's
standardized $\sigma_g^2$.

**A capped-iteration evo-lmm fit is not a converged fit.** The benchmark
deliberately caps optimization to match GRAPP's evaluation budget, and GRAPP's
own comparison path emits secant non-convergence warnings there. That setup is
valid for timing and invalid for inference. Calibration corrects the *scale* of
an association statistic, not a variance-component estimate that has not
converged: on seeded null data, a GRG fit stopped early with $h^2$ near its
upper boundary produced $\lambda_{\text{GC}} \approx 1.7$, while the same data
with a converged fit gave $\lambda_{\text{GC}}$ in line with plain regression.
Always check `FitDiagnostics.converged`, `score_norm`, and
`trace_standard_errors` before reading an estimate.

**The two fitters fail differently, and that is intentional.** GRAPP mirrors
BOLT-LMM: it never raises, and a silently unconverged CG solve or secant search
returns a plausible-looking number with a log line. evo-lmm raises on CG
failure and reports `converged = False`, on the principle that a failed fit
should return actionable diagnostics rather than a plausible-looking estimate.
When reproducing GRAPP numbers, that difference in failure semantics — not just
the tolerance values — has to be accounted for.

**Tuning order.** For speed, reduce $S$ before loosening `cg_tol`: the trace
probe count multiplies the dominant CG batch width, while `cg_tol` interacts
with warm starting and with the Newton iteration's ability to tell a real score
from solver error. Five probes have been measured as insufficient for final
refinement — they shifted shape estimates materially and enlarged trace
uncertainty — which is why the planned two-stage scheme keeps a small sketch
budget for early steps and a larger, deterministically nested refinement budget
for anything reportable.

## 7. Source map

| Component | GRAPP | evo-lmm |
| --- | --- | --- |
| Kernel application | `BoltLmmOps.apply_k`, `bolt_inf_core.py:694` | `EvolutionaryLmmOps.apply_k`, `operators.py:582` |
| Derivative kernels | — | `apply_dh`, `apply_dh_matmat`, `operators.py:637`, `:645` |
| CG solver | `bolt_conj_grad_solve`, `bolt_inf_core.py:263` | `solve_ph`, `operators.py:679` |
| Probe generation | `_generate_bolt_mc_components`, `bolt_inf_core.py:789` | `rademacher_probes` / `spherical_gaussian_probes`, `trace.py:21`, `:30` |
| Criterion evaluation | `_compute_mc_scaling`, `bolt_inf_core.py:846` | `_quantities`, `reml.py:237` |
| Search driver | `fit_bolt_variance_components`, `bolt_inf_core.py:942` | `fit_reml`, `reml.py:390` |
| LOCO residuals | `solve_loco_hinv_y`, `bolt_inf_core.py:1054` | `loco_solutions`, `bolt.py` |
| Calibration | `calibrate_lmm_inf`, `bolt_inf_core.py:1195` | `calibrate_association`, `calibration.py` |
