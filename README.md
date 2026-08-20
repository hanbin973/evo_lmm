[![Documentation Status](https://readthedocs.org/projects/evo-lmm/badge/?version=latest)](https://evo-lmm.readthedocs.io/en/latest/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)

# evo-lmm

`evo-lmm` fits **evolutionary random-effects linear mixed models**: LMMs in which
the prior variance of each SNP effect is derived from a stabilizing-selection
model rather than assumed from an empirical alpha-model. Variance components are
estimated by REML and genetic values by BLUP, with all genotype operations
carried out directly on
[genotype representation graphs (GRG)](https://github.com/aprilweilab/grgl) so
that biobank-scale genotype matrices are never materialized.

The scientific specification lives in
[the accompanying notes](notes/stab1_genetics_template.pdf), which are the source
of truth for the model; the code follows them.

* **Selection-derived priors** — SNP-effect variance follows from Fisher's
  geometric model of stabilizing selection, with two nested model families.
* **GRG-backed** — genotype products run through
  [GRGL](https://github.com/aprilweilab/grgl) and
  [GRAPP](https://github.com/aprilweilab/grapp) operators; no dense `N x M`
  matrix is formed.
* **BOLT-LMM style inference** — conjugate-gradient trace estimation and REML
  variance-component optimization, with LOCO support.
* **Simulation utilities** — [msprime](https://tskit.dev/msprime/) tree
  sequences converted to GRG, with phenotypes drawn from the postulated prior,
  for end-to-end verification.

## Quickstart

```python
from evo_lmm import SimplifiedPrior, fit_evolutionary_bolt_lmm, simulate_grg_lmm

simulation = simulate_grg_lmm(
    SimplifiedPrior(sigma_b2=0.8, tau=0.5),
    n_individuals=24,
    sequence_length=100_000,
    mutation_rate=2e-7,
    residual_variance=0.4,
    seed=7,
)

fit = fit_evolutionary_bolt_lmm(
    [("simulated", simulation.grg)],
    simulation.phenotype,
    frequencies={"simulated": simulation.frequencies},
    model="simplified",
    initial=simulation.prior,
    seed=7,
)
print(fit.prior, fit.sigma_e2, fit.diagnostics.converged)
```

`sigma_b2` is the raw-dosage per-locus effect scale; no implicit `1 / M` kernel
normalization is applied. The boundary cases `tau = 0`, `rho = 0`, monomorphic
frequencies, and the full-model `rho = 1` limit are supported explicitly and
flagged in the fit diagnostics when weakly identified.

## The model

For phenotype vector `y` and raw diploid dosage matrix `G`,

```text
y = G beta + epsilon,
```

with `beta_j` random and, writing `q_j = x_hat_j (1 - x_hat_j)` for the sample
allele frequency heterozygosity at variant `j`:

| Model | SNP-effect variance `E[beta_j^2 given G_j]` | Free parameters |
| --- | --- | --- |
| Simplified evolutionary model | `sigma_b^2 / (1 + 2 tau q_j)` | `sigma_b^2`, `tau` |
| Full evolutionary model | `sigma_b^2 (1 - rho_ab^2 (2 tau q_j) / (1 + 2 tau q_j))` | `sigma_b^2`, `rho_ab`, `tau` |

Here `tau = sigma_a^2 / W_S` is the selection-frequency aggregate, `sigma_b^2`
the focal-trait effect scale, and `rho_ab` the coupling between the focal and
the latent selected trait. The simplified model is the exact `rho_ab^2 = 1`
specialization of the full model. Throughout, the observed focal-trait effect
`beta_j` and the latent fitness-trait effect vector `alpha_j` are kept distinct.

## Installation

Prerequisites: Python 3.12+, [uv](https://docs.astral.sh/uv/), Git with
submodule support, and CMake with a C++17 compiler (to build GRGL's native
Python extension).

```bash
git clone --recurse-submodules https://github.com/hanbin973/evo_lmm.git
cd evo_lmm
uv sync
```

For an existing checkout:

```bash
git submodule update --init --recursive
uv sync
```

`uv` installs the GRGL submodule's Python distribution (`pygrgl`) and the GRAPP
reference implementation as editable local dependencies; `evo-lmm` reaches GRAPP
through a small compatibility adapter and never modifies either submodule.
Verify with:

```bash
uv run python -c "import evo_lmm, pygrgl, grapp, msprime, numpy, scipy, tskit; print('environment ready')"
```

## Documentation

Full documentation is at
[evo-lmm.readthedocs.io](https://evo-lmm.readthedocs.io/en/latest/), organized as
overview, how-to guides, tutorials, and
[API reference](docs/reference/public_api.rst) — the same layout used by GRGL and
GRAPP. It includes a ten-replicate, 1,000-individual variance-component recovery
tutorial. To build locally:

```bash
uv sync
uv run sphinx-build -W -b html docs docs/_build/html
```

## Development

```bash
uv run pytest          # dense, GRG, REML, and LOCO verification
uv run python main.py  # temporary command-line entry point
```

Dependency changes belong in `pyproject.toml`, with the lockfile regenerated via
`uv lock` (or `uv sync`). [AGENTS.md](AGENTS.md) records the implementation
conventions and model invariants that changes must preserve.

Repository layout:

```text
.
├── grgl/         Submodule: native GRGL library and its Python package
├── grapp/        Submodule: reference GRGL-backed BOLT-LMM implementation
├── notes/        Scientific specification for the evolutionary model
├── plan/         Implementation plans
├── src/evo_lmm/  First-party evolutionary LMM package
├── tests/        Dense, GRG, REML, and LOCO verification
├── docs/         Sphinx documentation
└── AGENTS.md     Development conventions and model invariants
```

## Citation

If you use `evo-lmm`, please cite the paper describing the model:

> Lee, H., & Terhorst, J. (2026). Parameterizing the genetic architecture under
> stabilizing selection. *Genetics*.
> https://doi.org/10.1093/genetics/iyag180

Please also cite the GRG papers that the genotype backend rests on:

> DeHaas, D., Pan, Z., & Wei, X. (2025). Enabling efficient analysis of
> biobank-scale data with genotype representation graphs.
> *Nature Computational Science*, 5(2), 112–124.

> DeHaas, D., Adonizio, C., Pan, Z., & Wei, X. (2026). General,
> orders-of-magnitude faster whole-genome analysis with genotype representation
> graphs. *bioRxiv*. https://doi.org/10.64898/2026.04.10.717786
