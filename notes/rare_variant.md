# Rare-variant heritability under stabilizing selection: reanalysis plan

Last audited: 2026-08-20

## Purpose

Lee & Terhorst (2026) closes with the observation that no real-data analysis was
carried out and that "closing this gap will require an efficient implementation
of the proposed variance-component model for modern genomic datasets."
`evo-lmm` is that implementation. This file is the plan for the first real-data
target: **re-estimating the rare-variant heritability reported by RareEffect**
(Nam et al. 2026, *Nature Genetics*, [10.1038/s41588-026-02705-9](https://doi.org/10.1038/s41588-026-02705-9))
under an evolutionarily derived prior instead of a flat one.

Scope for this plan:

- **Quantitative traits only.** Binary traits, Firth correction, and the
  working-residual construction are deferred (§10).
- **Simplified prior only ($\rho_{ab} = 1$).** The full prior is deferred to
  future work on identifiability grounds the paper itself establishes (§2.2).
  No work item anywhere in this plan — model, code, simulation, or reporting —
  targets the unsimplified model; everything concerning it is collected in §10.

Two outputs are in scope: a companion manuscript, and shipped `evo-lmm`
capability (annotation-partitioned evolutionary kernels, joint multi-component
REML/MoM) with tests and documentation.

---

## 1. The two models in one notation

Let $G$ be the $n \times M$ matrix of **raw minor-allele counts**, $\hat{x}_j$
the sample allele frequency, $q_j = \hat{x}_j(1 - \hat{x}_j)$, and $C$ the
orthonormal covariate basis with $P_C = I - CC^{\top}$, $d = n - \mathrm{rank}(C)$.
Variants are partitioned into annotation categories
$c \in \{\text{LoF}, \text{missense}, \text{synonymous}\}$.

### 1.1 RareEffect as published

Three-step:

1. **Null GLMM.** Fit $y = W\gamma + u + \varepsilon$ without the tested
   genotypes by AI-REML; carry forward residuals $\tilde{y}$.
2. **Variance components, per gene, per category.** With
   $\tilde{y} = \sum_c G_c \beta_c + \varepsilon$,

   $$
   \beta_c \sim \mathrm{MVN}(0, \tau_c \Sigma_c), \qquad \Sigma_c = I_{k_c},
   \qquad \varepsilon \sim \mathcal{N}(0, \psi I_n),
   $$

   fitted marginally per category by FaST-LMM, then adjusted by a
   method-of-moments ratio

   $$
   \hat{\tau}_{c} = \hat{\tau}^{\text{MLE,mar}}_{c} \cdot
   \frac{\hat{\tau}^{\text{MoM,joint}}_{c}}{\hat{\tau}^{\text{MoM,mar}}_{c}},
   $$

   with the joint system $\begin{pmatrix} T & b \\ b^{\top} & n \end{pmatrix}
   \begin{pmatrix} \hat{\tau} \\ \hat{\psi}\end{pmatrix} =
   \begin{pmatrix} c \\ \tilde{y}^{\top}\tilde{y}\end{pmatrix}$ and the rule
   "when MoM yields negative estimates, use the unadjusted marginal estimate."
3. **Variant effects.** BLUP with PEV-based intervals.

Reported heritability, per gene and per category:

$$
h^2_c = \frac{\mathrm{tr}(G_c \Sigma_c G_c^{\top})}
{\mathrm{tr}(G_c \Sigma_c G_c^{\top}) + n\psi}.
$$

Design choices that matter downstream: MAF $\le 1\%$ definition of rare; MAC
$< 10$ ultrarare variants **collapsed into a single column per category**
(SAIGE-GENE+ convention); no MAF weighting anywhere in the prior (MAF enters
only the sign of the gene-level LoF effect); UKB WES, $n = 392{,}748$ white
British, 100 phenotypes.

### 1.2 evo-lmm, single component

$$
y = G\beta + \varepsilon, \qquad
\mathbb{E}[\beta_j^2 \mid G_j] = \frac{\sigma_b^2}{1 + 2\tau q_j},
\qquad \tau = \frac{\sigma_a^2}{W_S}, \qquad W_S = \frac{V_S}{2N},
$$

with $h^2 = \sigma_b^2 \mathrm{tr}(P_C K P_C) / (\sigma_b^2 \mathrm{tr}(P_C K P_C) + \sigma_e^2 d)$.

### 1.3 The central structural fact

**RareEffect's prior is the $\tau = 0$ boundary of the evolutionary prior,
fitted separately per annotation category.** At that boundary
$w_j \equiv 1$ and $v_j \equiv \sigma_b^2$, so $\tau_c$ in RareEffect *is* a
per-category $\sigma_b^2$ with frequency dependence switched off.

This is an exact nesting, which gives the reanalysis a likelihood-ratio test
rather than a beauty contest. It also means `evo-lmm` can reproduce
RareEffect's estimator inside its own code path — as an explicitly named
baseline, per `AGENTS.md`.

Note the direction relative to the paper's simulations: RareEffect sits at
$\alpha = 0$, the *mildest* misspecification. Nonzero $\alpha$ models
"substantially exaggerat[e] the contribution of rare variants," whereas the
$\alpha = 0$ model "absorbs the missing frequency dependence by pushing its
estimate of $\sigma_b^2$ downward." The expected failure mode here is therefore
**understatement of the mutational scale and of the rarest bins**, not
exaggeration. Pre-register that direction.

### 1.4 Contact points

| Aspect | RareEffect | evo-lmm | Reanalysis consequence |
| --- | --- | --- | --- |
| Effect-variance prior | flat within category ($`\tau_c I`$) | $`\sigma_{b,c}^2 / (1 + 2\tau_c q_j)`$ | nested at $`\tau_c = 0`$; LRT available |
| Frequency dependence | none | mechanistic, saturating as $`q \to 0`$ | reallocates variance **within** the rare window |
| Shape parameters per gene | 3 free $`\tau_c`$ | 3 scales, with $`\tau_c`$ pooled across genes | fewer per-gene parameters |
| Ultrarare (MAC<10) | collapsed to one burden column | modeled individually; prior supplies shrinkage | collapsing is where "new mutation" contributions are decided |
| Cross-category LD | marginal ML + MoM ratio, truncation fallback | joint multi-component REML | removes a data-dependent selection rule |
| Covariates | residualized in step 1 | projected inside the fit | two-stage attenuation is testable |
| Estimand denominator | $`n\psi`$, uncentered $`\mathrm{tr}(G\Sigma G^{\top})`$ | $`d\sigma_e^2`$, projected trace | $`O(\hat{x})`$ for rare variants; still must be aligned exactly |
| Reported new quantity | — | $`\sigma_{b,c}^2`$ (mutational scale), $`\tau_c`$ (composite, §2.1) | per-annotation architecture parameters |

---

## 2. The annotation-partitioned prior, derived properly

This section replaces the naive "one shared $\tau$, one $\rho_c^2$ per class"
sketch. That sketch was wrong in both directions.

### 2.1 The selection filter is applied per category

The frequency dependence does not come from a weighting choice. It comes from
conditioning the mutational-effect prior on the observed genotype through the
selection filter, equation (A15):

$$
p(G_j \mid \lVert \alpha_j \rVert) \;\propto\;
\exp\!\left[-\frac{\lVert \alpha_j \rVert^2}{W_S}\,\hat{x}_j(1 - \hat{x}_j)\right].
$$

The posterior mean of $\lVert\alpha_j\rVert^2$ follows from multiplying this
filter by the **prior on $\lVert\alpha_j\rVert^2$** — and that prior is a
property of the variant's mutational class. Under the Gamma DFE generalization
(A16), with shape $k$ and scale $\sigma_a^2$,

$$
\mathbb{E}\big[\lVert\alpha_j\rVert^2 \mid G_j\big]
= \frac{\sigma_a^2}{1 + \dfrac{\sigma_a^2}{k\,W_S}\,q_j},
$$

which reduces to the main-text form at $k = 1/2$.

**Consequence for a partitioned model.** Each annotation category $c$ has its
own mutational-effect distribution $p_c(\lVert\alpha_j\rVert^2)$ — its own
scale $\sigma_{a,c}^2$ and plausibly its own shape $k_c$ — so the filter must
be composed with *that* prior. The conditioning cannot be done once with a
category-averaged prior and then reused. Carrying it through:

$$
\mathbb{E}[\beta_j^2 \mid G_j] = \frac{\sigma_{b,c}^2}{1 + 2\tau_c q_j},
\qquad
\boxed{\;\tau_c = \frac{\rho_{ab,c}^2\,\sigma_{a,c}^2}{2\,k_c\,W_S}\;}
$$

for $j$ in category $c$. Therefore:

- **$\tau$ is category-specific, not shared.** The earlier "shared $\tau$"
  decision was incorrect: $\tau_c$ inherits the category's mutational variance
  and DFE shape.
- **What *is* shared is $W_S = V_S/2N$** — the selection width, a property of
  the selection regime acting on the trait, not of a variant's annotation.
- **$\tau_c$ is a three-way composite** of coupling $\rho_{ab,c}^2$, mutational
  variance $\sigma_{a,c}^2$, and DFE shape $k_c$. It must be reported as such.
  A mechanistic reading of $\hat{\tau}_c$ as "coupling to fitness" is not
  licensed; only the composite is estimated.
- Ratios $\tau_c / \tau_{c'}$ are interpretable only as ratios of these
  composites — still informative (LoF versus synonymous), but not attributable
  to any single mechanism.
- Recall (A17)'s reading of $k$: "the frequency dependence is stronger when
  large effect variants are rarer." So an LoF-versus-missense difference in
  $\hat{\tau}_c$ may reflect DFE shape as much as coupling.

**Shared $W_S$: assumed, not checked.** Treating $W_S$ as a shared constant
independent of the partition rests on the $V_g \ll V_S$ approximation, under
which the paper shows "the shape of $h(u)$ is irrelevant" (A5 $\to$ A17). That
approximation is standard, and it is taken as given here: $W_S$ is shared across
categories and no verification step is planned. Revisiting it is a low-priority
item (§10), relevant only if a partitioned $V_g$ were ever found to be a
non-negligible fraction of $V_S$.

### 2.2 Why the full prior is deferred

The paper states the identifiability problem directly:

> "jointly estimating $\rho_{ab}$ and $\sigma_a^2/W_S$ is difficult because
> $\hat{x}_j(1-\hat{x}_j)$ is typically both small and weakly variable under the
> U-shaped allele-frequency spectrum. As a result, the denominator in equation
> (21) is usually close to one, so the data mainly identify the product
> $\rho_{ab}^2 \sigma_a^2/W_S$ rather than its two components separately. This
> is why fixing $\rho_{ab} = 1$ and interpreting the fitted quantity as
> $\rho_{ab}^2\sigma_a^2/W_S$ works reasonably well in practice."

Two things follow, and they bind harder here than in the paper's own setting:

1. **The full prior is future work, not a later phase of this one.**
   Attempting to estimate $\rho_{ab,c}^2$ separately per category would multiply
   a degeneracy that is already marginal, three times over. Use
   `SimplifiedPrior` per category, fix $\rho_{ab} = 1$, and read $\hat{\tau}_c$
   as the composite of §2.1. This is a scope *reduction* in P0: no
   $\mathrm{logit}\,\rho^2$ coordinates, no $\rho$-boundary handling in the
   multi-component code, and no simulation arm or metric whose purpose is to
   characterize the unsimplified model.
2. **A rare-only window is the worst case for this degeneracy.** The paper's
   argument is that $q$ is small and weakly variable across the *whole*
   spectrum. Restricted to MAF $\le 1\%$ it is smaller and less variable still,
   so the denominator $1 + 2\tau_c q_j$ sits even closer to one. This is now the
   dominant feasibility constraint on the entire reanalysis (§3).

**Corollary that reshapes the deliverable.** The paper also reports the useful
asymmetry: "$\sigma_b^2$ controls the overall scale of the covariance structure,
whereas $\sigma_a^2/W_S$ enters only through a weaker modulation by allele
frequency," and consequently "estimation of $\sigma_b^2$ remains stable" even
when the decomposition is only partially identified. So the primary new estimand
of the reanalysis is **$\sigma_{b,c}^2$, the per-class mutational scale** —
robust, and inaccessible to RareEffect. $\tau_c$ is the secondary, weakly
identified quantity. Write the manuscript around $\sigma_{b,c}^2$, not $\tau_c$.

### 2.3 The model ladder

Fitting proceeds up an explicit nesting ladder rather than straight to the most
general model. Coordinates are counted after profiling one scale.

| Model | Prior | Searched shape coordinates ($`|c| = 3`$) | Interpretation |
| --- | --- | --- | --- |
| **M0** | $`\tau_c = 0`$, per-class scale | 3 | RareEffect's prior, exactly |
| **M1** | shared $`\tau`$, per-class scale | 4 | parsimonious, **mechanistically wrong** (requires $`\rho_c^2\sigma_{a,c}^2/k_c`$ equal across classes); an identifiability crutch, labeled as such |
| **M2** | per-class $`\tau_c`$, per-class scale | 6 | the correct partitioned model of §2.1 |

M0 vs M1 is the RareEffect test. M1 vs M2 is the scientifically interesting
test: does the composite differ by annotation class? Report both, and report M1
honestly as a crutch rather than a model. **The ladder terminates at M2.**
Estimating $\rho_{ab,c}^2$ per category is future work (§10), not a further rung
to be climbed if M2 fits well.

---

## 3. How much frequency dependence is available in the rare window?

Within MAF $\le 1\%$, $q \le q_{\max} = 0.0099$, so the dynamic range of the
weight across the entire window is $1 + 2\tau_c q_{\max}$:

| $`\tau_c`$ | $`2\tau_c q_{\max}`$ | prior-variance range across MAF $`\le 1\%`$ |
| --- | --- | --- |
| 1 | 0.02 | 1.02x — indistinguishable from flat |
| 10 | 0.20 | 1.2x — detectable at $`n \sim 4\times10^5`$, small effect on $`h^2`$ |
| 100 | 1.98 | 3.0x — material misallocation across the window |
| 1000 | 19.8 | 21x — flat prior badly misspecified |

Combined with §2.2, two design consequences:

1. **$\tau_c$ must be anchored outside the rare window.** The design fits
   **common + rare jointly** — imputed array variants supply the leverage on
   $\tau_c$ where $q$ is large and variable; WES supplies the rare window.
   A rare-only fit is a diagnostic, not the primary analysis.
2. **This makes the model predictive, not merely better-fitting.** With
   $\sigma_{b,c}^2$ and $\tau_c$ anchored on the common spectrum, rare-variant
   genic variance follows with **no additional free parameter**. RareEffect
   cannot make that prediction: its $\tau_c$ is free per gene per category.

A robustness point in evo-lmm's favor, from the paper's remark that
"substituting $\hat{x}_j$ for $x_j$ can introduce substantial error for rare
variants": at MAC $= 10$ with $n \approx 3.9\times10^5$, the relative standard
error of $\hat{x}_j$ is $\approx 1/\sqrt{\text{MAC}} \approx 32\%$. Under an
$\alpha > 0$ weight, $[\hat{x}(1-\hat{x})]^{-\alpha}$ amplifies that without
bound. Under the evolutionary weight $w_j \to 1$ as $q \to 0$ with bounded
$\partial w/\partial q$, so plug-in noise is *attenuated* exactly where it is
largest. Quantify explicitly (Phase 0, item 4).

---

## 4. Falsifiable claims

Stated so each can come out negative.

- **H1 (detectability).** $\tau_c > 0$ is detectable for at least some
  quantitative traits. Test: boundary-aware LRT of M1 and M2 against M0,
  exome-wide and per category.
- **H2 (spectrum misallocation).** Under a flat prior $\hat{\tau}_c$ tracks
  $v_j$ near the high-$q$ end of the window, so RareEffect **understates** the
  genic variance carried by the rarest variants and **understates**
  $\sigma_{b,c}^2$, while total-window $h^2$ is comparatively robust. Test:
  per-MAF-bin genic-variance bias in simulation; per-bin decomposition in real
  data.
- **H3 (cross-spectrum extrapolation).** $(\sigma_{b,c}^2, \tau_c)$ anchored on
  common variants predicts rare-variant genic variance within its CI. Given
  §2.2's asymmetry, this is the **strongest** available claim, because
  $\sigma_{b,c}^2$ is the stable parameter. Failure is itself a result.
- **H4 (collapsing).** MAC $< 10$ collapsing materially biases the rarest bins,
  because it imposes both equal prior variance *and* perfectly correlated
  effects on precisely the variants whose prior variance is largest. Test:
  ablation with collapsing on and off, everything else fixed.
- **H5 (architecture ordering, composite).** $\hat{\tau}_c$ is ordered
  $\text{LoF} > \text{missense} > \text{synonymous}$, with
  $\hat{\tau}_{\text{syn}} \approx 0$, and $\hat{\sigma}_{b,c}^2$ is ordered
  likewise. **The ordering is a statement about the composite
  $\rho_{ab,c}^2\sigma_{a,c}^2/k_c W_S$, not about coupling.** Synonymous is
  the negative control: the paper notes that when $\rho_{ab}^2 = 0$ the model
  "collapses to an approximately frequency-independent prior" and "the
  evolutionary fit and the $\alpha = 0$ baseline give nearly identical
  predictions" — so for synonymous variants, **evo-lmm agreeing exactly with
  RareEffect is the expected, passing result.** Disagreement there indicates
  stratification or annotation error, not selection.
- **H6 (estimator audit).** The marginal-ML $\times$ (joint-MoM/marginal-MoM)
  ratio, with "if MoM is negative, use the unadjusted marginal" as fallback, is
  a data-dependent selection rule; audit bias and CI coverage against joint
  multi-component REML.
- **H7 (two-stage attenuation).** Fitting rare-variant components on step-1
  residuals attenuates them relative to a joint fit.

Reference targets: exome-wide $h^2$ of 0.0738 (95% CI 0.0522–0.1155) for HDL-C,
0.0818 (0.0561–0.1494) for LDL-C, 0.0535 (0.0371–0.1127) for triglycerides.
**Verify how those intervals were constructed before comparing** — intervals of
that shape come from a specific resampling or PEV construction, and a
like-for-like comparison requires reproducing it.

---

## 5. Required extensions to evo-lmm

Current state (audited 2026-08-20): the fitter estimates **one** global
$(\sigma_b^2, \tau, \rho^2)$ with chromosomes summed into a single kernel
(`operators.py`, `reml.py`, `bolt.py`). Everything below is new work.

### P0 — Annotation-partitioned multi-component kernel (simplified prior only)

$$
K = \sum_c \sigma_{b,c}^2 \, P_C X_c \,
\mathrm{diag}\!\left(\frac{1}{1 + 2\tau_c q_j}\right) X_c^{\top} P_C .
$$

Model invariants to add to `AGENTS.md`:

- $\tau_c$ is category-specific; $W_S$ is the shared quantity. Do not implement
  a globally shared $\tau$ as the default model — it is available only as M1,
  and must be documented as an identifiability crutch, not a mechanistic claim.
- $\hat{\tau}_c$ estimates the composite
  $\rho_{ab,c}^2 \sigma_{a,c}^2 / (2 k_c W_S)$. Public docstrings must say so;
  do not label it "coupling."
- Each category's selection filter is composed with that category's own
  mutational-effect prior. The partition is part of the model specification,
  not a post-hoc grouping of columns.
- $\rho_{ab} \equiv 1$ on this path. `FullPrior` is not used.

Work items:

- [ ] Multi-component simplified prior objects with analytic derivatives w.r.t.
  $(\log \sigma_{b,c}^2, \log \tau_c)$.
- [ ] Multi-component AI-REML: profile one scale (or $\sigma_e^2$) and search
  the remaining $2|c|$ shape coordinates. The single-scale profiling in
  `reml.py` does not generalize as written; state which scale is profiled and
  keep the objective's ML/REML status declared.
- [ ] Batched derivative-kernel traversals per component, reusing the shared CG
  solve and shared Hutchinson probes (`operators.apply_dh_matmat`).
- [ ] PSD and symmetry tests per component and for the sum.
- [ ] Exact nesting tests up the ladder: all $\tau_c = 0$ reproduces M0
  bit-for-bit; $\tau_c \equiv \tau$ reproduces M1; a single category reproduces
  the existing single-component fit bit-for-bit.

### P1 — Named baselines

- [ ] `flat` prior ($w_j \equiv 1$) as an explicitly named code path (M0).
- [ ] Marginal-per-category ML + MoM-ratio adjustment **with** the negative-MoM
  truncation rule, reproduced faithfully, so H6 is an ablation and not a
  strawman.
- [ ] Optional MAC-threshold collapsing operator (burden column construction)
  so H4 is a toggle.

### P2 — Joint multi-component MoM / Haseman–Elston

`haseman_elston_initialization()` already solves the one-component moment
system. Generalize to the $|c|$-component system — structurally the same object
as RareEffect's $T$ matrix — so that initialization is good in the
multi-component setting and a like-for-like MoM comparison exists. Good
initialization matters more here than in the single-component case: six shape
coordinates in a weakly identified regime.

- [ ] $(|c|+1) \times (|c|+1)$ moment system with XTrace/Hutchinson trace
  entries and evolutionary weights.
- [ ] Report the estimate without truncation, and report separately how often
  truncation would have fired.

### P3 — Estimand adapters and reporting

- [ ] Emit $h^2$ under **both** conventions: RareEffect's ($n$, uncentered
  $\mathrm{tr}(G\Sigma G^{\top})$) and evo-lmm's ($d$, covariate-projected).
  Document the $O(\hat{x})$ gap for rare variants.
- [ ] Per-MAF-bin genic-variance decomposition
  $\sum_{j \in \text{bin}} \sigma_{b,c}^2 w_j \lVert P_C X_j \rVert^2$ — the
  primary quantity for H2.
- [ ] Standard errors for $h^2$, $\sigma_{b,c}^2$, $\tau_c$ by delta method from
  the AI matrix; **profile likelihoods for $\tau_c$**, since a symmetric
  delta-method interval is not credible in a weakly identified coordinate.
- [ ] Boundary-aware LRT for the ladder (mixture null; not a naive $\chi^2$).
- [ ] Gene-level output: pooled $\tau_c$ with per-gene $\sigma_{b,c}^2$
  (empirical-Bayes two-level), matching RareEffect's reporting unit.

### P4 — WES data path

- [ ] pVCF/BGEN $\to$ GRG conversion for exome data, per-chromosome blocks.
- [ ] Annotation masks (LOFTEE high-confidence LoF, missense, synonymous) as
  first-class variant partitions in `grg_data.py`.
- [ ] MAC/MAF filters and sample filters applied **before** frequency
  recomputation (existing invariant).
- [ ] Measure GRG compression on exome rare variants — far less favorable than
  WGS; do not inherit the existing benchmark claim.

### Blocking dependency

`plan/evolutionary-bolt-lmm.md` **Priority 1 (production convergence policy) is
a hard prerequisite for any reported estimate.** Current defaults
(`trace_probes=12`, `cg_tol=5e-4`) are documented as exploratory, and the same
plan records a non-converged fit near the $h^2$ boundary producing
$\lambda_{GC} \approx 1.7$ on null data. M2 has six shape coordinates, several
of them weakly identified by construction (§2.2) — a strictly harder
optimization problem than the two-parameter fits that policy was written for.
Do not begin Phase 3 before the two-stage sketch/refinement policy exists and is
tested at $|c| = 3$.

---

## 6. Phases

### Phase 0 — Estimand alignment and analytic bias (no data, no new code)

Phase 0 does **not** gate P0. The partitioned derivation of §2.1 is settled and
the shared-$W_S$ assumption is taken as standard, so code work in Phase 1 can
start in parallel with everything below.

1. Derive the asymptotic limit of the flat-prior variance-component estimator
   when the truth is evolutionary, as a function of the observed AFS. For a
   moment estimator this is a closed-form trace ratio; write it out.
2. Evaluate that expression on a realistic UKB WES AFS (gnomAD exomes or the
   released RareEffect results) for a **projected** bias per lipid trait before
   running anything.
3. Reconcile the two $h^2$ definitions exactly ($n$ vs $d$, centered vs
   uncentered).
4. Quantify $\hat{x}_j$ plug-in error propagation under the evolutionary weight
   versus $[\hat{x}(1-\hat{x})]^{-\alpha}$ across MAC $= 1 \ldots 4000$.
5. Inspect released artifacts: SAIGE publication release
   ([zenodo.20933696](https://doi.org/10.5281/zenodo.20933696)) and the
   100-phenotype results ([zenodo.20935280](https://doi.org/10.5281/zenodo.20935280)).
   Determine whether per-gene $\hat{\tau}_c$, $\hat{\psi}$, and MAF spectra are
   present — that bounds what summary-level reanalysis can support.

**Gate:** a written projected direction and magnitude for the bias exists.

### Phase 1 — Code extensions (P0–P3)

Order: P0 kernel and multi-component REML $\to$ P1 baselines $\to$ P2 joint MoM
$\to$ P3 reporting. P4 runs in parallel, needed only for Phase 3.

**Gate:** every rung of the ladder reproduces the rung below it exactly at the
boundary; dense-versus-GRG equivalence at `cg_tol=1e-9`; the reproduced
RareEffect estimator matches a from-scratch reimplementation on a small dense
case.

### Phase 2 — Calibrated simulation (primary evidence)

Reuse existing assets: `docs/tutorials/slim_simplified_prior.slim`,
`slim_forward_simplified.py`, and the ten persisted replicates under
`docs/_artifacts/forward_replicates/`.

**Design.** Forward SLiM simulation with stabilizing selection and pleiotropy,
structured as an exome: gene-sized loci, three annotation classes whose
generative $(\rho_{ab,c}^2, \sigma_{a,c}^2, k_c)$ are chosen so the classes
differ in the *composite* $\tau_c$. Only the composite is a target of
estimation; its factors are generative inputs, not quantities the simulation
sets out to recover. Synonymous class: $\rho_{ab}^2 = 0$ exactly. Mutation rates chosen so per-gene variant counts match UKB WES. Sample
sizes $n \in \{20\text{k}, 50\text{k}\}$ plus one scaled run to check
$n$-dependence of the collapsing bias. A faster msprime arm with a
gnomAD-matched AFS serves as a cheap replicate generator.

**Arms.**

| Arm | Residualize | Collapse MAC<10 | Prior | Fit |
| --- | --- | --- | --- | --- |
| A | yes | yes | M0 flat per category | marginal ML + MoM ratio + truncation |
| B | yes | yes | M0 flat per category | joint REML |
| C | no | no | M0 flat per category | joint REML |
| D | no | no | M1 shared $`\tau`$ | joint REML |
| E | no | no | M2 per-class $`\tau_c`$ | joint REML |

Arm A is RareEffect as published. A$\to$B$\to$C$\to$E is a clean ablation path
that **attributes the total bias to individual design choices** — that
attribution, not the aggregate difference, is the contribution. D vs E isolates
the value of category-specific $\tau_c$.

**Metrics.** Bias and RMSE of total $h^2$; bias of per-MAF-bin genic variance
(H2); recovery of $\sigma_{b,c}^2$ and $\tau_c$ (H3, H5); per-variant BLUP accuracy; held-out PRS $R^2$; CI coverage,
including profile-likelihood intervals for $\tau_c$ (H6); rate at which the MoM
truncation rule fires and its effect.

**Gate:** H2 and H4 resolved with stated direction and magnitude across $\ge 10$
replicates, with converged fits (`FitDiagnostics.converged`, `score_norm`,
`trace_standard_errors` reported for every fit, per `notes/model_fitting.md` §6).

### Phase 3 — UKB WES reanalysis (access-gated)

**Prerequisite:** an approved UK Biobank application with WES (fields
23158/23159) and RAP/DNAnexus compute. Until that exists, Phase 3 is a plan.
UKB use is governed by its access agreement; internal clinical cohorts are
**not** a substitute — no patient-level or identifiable data enters this
analysis, and any proposal to use internal sequencing data must go through
compliance review first.

1. Reproduce RareEffect's published numbers for 3–5 anchor traits (HDL-C,
   LDL-C, triglycerides, BMI, height) via the reproduced Arm A path. **Do not
   claim a difference until the baseline is reproduced.**
2. Fit the joint common+rare model (imputed array + WES) to anchor
   $\sigma_{b,c}^2$ and $\tau_c$ outside the rare window (§3).
3. Report: exome-wide and per-category $h^2$ under both conventions; per-MAF-bin
   decomposition; $\hat{\sigma}_{b,c}^2$; $\hat{\tau}_c$ with profile intervals
   and an explicit composite-interpretation caveat; the ladder LRTs.
4. Gene-level output for the genes RareEffect highlights (APOC3 and the lipid
   gene set) with pooled shape parameters.
5. Sensitivity: ancestry subset, relatedness threshold, MAC threshold,
   annotation tool version, collapsing on/off.

**Gate:** the reproduced baseline agrees with published estimates within their
reported intervals, and every reported evolutionary fit is converged at
refinement-stage trace precision.

---

## 7. Pre-registered failure modes

- **$\tau_c$ is small or unidentified in the rare window.** The most likely
  outcome given §2.2. Then RareEffect's $h^2$ is essentially correct within
  MAF $\le 1\%$, and the result becomes: the total is robust, while
  $\sigma_{b,c}^2$ and the spectrum decomposition are new and were previously
  inaccessible. Smaller, honest, and still worth publishing — this is why §2.2
  says to build the manuscript around $\sigma_{b,c}^2$.
- **M1 and M2 are indistinguishable.** If the composite cannot be separated by
  annotation class, report that as a negative result on H5 and fall back to M1
  with the crutch caveat intact.
- **Exome rare-variant LD is weak**, so the MoM cross-category adjustment is
  nearly the identity and H6 has little to bite on.
- **Collapsing turns out near-neutral** at $n \approx 4\times10^5$ because the
  burden column captures most of the aggregate signal.
- **Annotation error dominates.** If $\hat{\tau}_{\text{syn}}$ is materially
  above zero, stop and diagnose stratification/annotation before interpreting
  anything (H5's control).
- **Multi-component identifiability.** Report the AI condition number and
  profile likelihoods for every $\tau_c$; be willing to drop to M1, or to M0
  with per-class scales, and say so.
- **Convergence, not statistics, is the binding constraint** — §5, blocking
  dependency.

---

## 8. Deliverables

| Deliverable | Where |
| --- | --- |
| This plan | `notes/rare_variant.md` |
| Implementation ledger items | `plan/evolutionary-bolt-lmm.md` (new Priority section) |
| Multi-component simplified priors and REML | `src/evo_lmm/` (new module; do not expand `reml.py` in place) |
| Named M0 and MoM-ratio baselines | `src/evo_lmm/baselines.py` |
| Tests | `tests/test_components_*.py`; ladder-nesting and baseline-reproduction tests |
| Simulation study | `docs/tutorials/` + persisted artifacts, following the SLiM tutorial pattern |
| Model invariants: per-class $`\tau_c`$, shared $`W_S`$, composite reading of $`\hat{\tau}_c`$, $`\rho_{ab} \equiv 1`$ | `AGENTS.md` |
| Manuscript | separate repository |

---

## 9. Immediate next actions

1. Phase 0 items 1–3 (analytic, no code): the flat-prior asymptotic limit, its
   evaluation on a realistic WES AFS, and the $h^2$ estimand reconciliation.
2. Pull the two Zenodo deposits; determine what summary-level reanalysis can
   support (Phase 0 item 5).
3. Record the §5 model invariants in `AGENTS.md` and open the `plan/` Priority
   section for the multi-component kernel.
4. Begin P0 — the multi-component simplified prior and its analytic
   derivatives. This is no longer gated on Phase 0 and can run concurrently.
5. Start UKB access in parallel — the long-lead item, gating Phase 3
   independently of all code work.

---

## 10. Deferred

- **The full prior model** ($\rho_{ab,c}^2$ estimated per category). Deferred on
  the paper's own identifiability grounds (§2.2): under a U-shaped spectrum the
  data identify the product $\rho_{ab}^2\sigma_a^2/W_S$, not its factors, and a
  rare-only window is the worst case. Revisiting requires either external
  information on $\sigma_{a,c}^2$ or $k_c$ (mutation-accumulation experiments,
  DFE estimates from independent data), or a design with real leverage across
  the frequency spectrum — most plausibly WGS rather than WES, where common
  variants with large, variable $q$ enter the same partitioned fit. The
  simulation work that would demonstrate the non-identifiability empirically —
  configurations where two classes share a composite $\tau_c$ but differ in its
  factors — belongs to that future effort, not to Phase 2; the paper's
  Discussion is sufficient warrant for the deferral.
- **Separating $k_c$ (DFE shape) from $\rho_{ab,c}^2$ and $\sigma_{a,c}^2$**
  within $\hat{\tau}_c$. Same obstruction; needs an external anchor.
- **Revisiting the shared-$W_S$ assumption.** Low priority. $V_g \ll V_S$ is
  standard, so $W_S$ is treated as partition-independent throughout (§2.1).
  Worth reopening only if a partitioned $V_g$ turned out to be a non-negligible
  fraction of $V_S$.
- **Binary traits**: working residuals from IRLS, variance standardization, and
  RareEffect's fast Firth correction (reported ~240x speedup). The evolutionary
  prior is defined on the trait scale; the liability-scale mapping and Firth's
  interaction with a frequency-dependent prior each need their own derivation.
- Explicit LD modeling inside the evolutionary derivation (named as future work
  in Lee & Terhorst 2026 §Discussion).
- Richer pleiotropy than the single latent selected trait.
- GPU parity (`plan/evolutionary-bolt-lmm.md` Priority 3).

---

## References

- Lee, H., & Terhorst, J. (2026). Parameterizing the genetic architecture under
  stabilizing selection. *Genetics*.
  [10.1093/genetics/iyag180](https://doi.org/10.1093/genetics/iyag180).
  Author version: `notes/stab1_genetics_template.pdf`. Equations (A15)–(A18)
  are the derivation basis for §2.1; the identifiability passage quoted in §2.2
  is in the Discussion.
- Nam, K., Kho, M., Zhou, W., Mukherjee, B., Lee, S., et al. (2026). Rare
  variant effect estimation and polygenic risk prediction. *Nature Genetics*.
  [10.1038/s41588-026-02705-9](https://doi.org/10.1038/s41588-026-02705-9).
  Preprint: [medRxiv 2024.06.23.24309366](https://www.medrxiv.org/content/10.1101/2024.06.23.24309366v2).
- Zhou, W., et al. (2022). SAIGE-GENE+ improves the efficiency and accuracy of
  set-based rare variant association tests. *Nature Genetics*.
  [10.1038/s41588-022-01178-w](https://doi.org/10.1038/s41588-022-01178-w) —
  source of the MAC<10 collapsing convention.
- Weiner, D. J., et al. (2024). A method to estimate the contribution of rare
  coding variants to complex trait heritability. *Nature Communications*.
  [10.1038/s41467-024-45407-8](https://doi.org/10.1038/s41467-024-45407-8) —
  burden heritability regression; the closest existing method modeling
  frequency–effect coupling for rare coding variants, and a required comparison.
- `notes/model_fitting.md` — fitter semantics, diagnostics discipline, and the
  warning that a capped-iteration fit is not a converged fit.
