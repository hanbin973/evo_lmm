# evo-lmm

`evo-lmm` is a Python implementation of the evolutionary random-effects
model developed in [the accompanying notes](notes/stab1_genetics_template.pdf).
It couples a stabilizing-selection model to a linear mixed model (LMM), using
[GRGL](https://github.com/hanbin973/grgl) to store and traverse genotype
representation graphs efficiently.

## Model at a glance

For phenotype vector `y` and genotype matrix `G`, the target model is

```text
y = G beta + epsilon.
```

Instead of assigning SNP-effect variance with an empirical alpha-model, the
notes derive it from Fisher's geometric model of stabilizing selection. For
variant `j`, with sample allele frequency `x_hat_j`, the main-text variance is

```text
E[beta_j^2 | G_j]
  = sigma_b^2 * (1 - rho_ab^2 *
      (2 (sigma_a^2 / W_S) x_hat_j (1 - x_hat_j)) /
      (1 + 2 (sigma_a^2 / W_S) x_hat_j (1 - x_hat_j))).
```

The implementation will form the corresponding frequency-weighted genetic
relatedness matrix, estimate its variance components by REML, and produce
genetic-value predictions with BLUP. It supports two nested model families,
where `q_j = x_hat_j (1 - x_hat_j)`:

| Model | SNP-effect variance | Free parameters |
| --- | --- | --- |
| Simplified evolutionary model | `sigma_b^2 / (1 + 2 * tau * q_j)` | `sigma_b^2`, `tau` |
| Full evolutionary model | `sigma_b^2 * (1 - rho_ab^2 * (2 * tau * q_j) / (1 + 2 * tau * q_j))` | `sigma_b^2`, `rho_ab`, `tau` |

Here `tau = sigma_a^2 / W_S` is the selection-frequency aggregate,
`sigma_b^2` is the focal-trait effect scale, and `rho_ab` controls coupling
between the focal and selected traits. The simplified model is the exact
`rho_ab^2 = 1` specialization of the full model; it has two estimable
parameters because the coupling is fixed rather than separately estimated.

## Repository layout

```text
.
├── grgl/       Git submodule: native GRGL library and its Python package
├── grapp/      Git submodule: reference GRGL-backed BOLT-LMM implementation
├── notes/      Scientific specification for the evolutionary model
├── plan/       Concrete implementation plans
├── main.py     Temporary command-line entry point
├── pyproject.toml
└── AGENTS.md   Development conventions and model invariants
```

## Installation

Prerequisites:

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/)
- CMake and a C++17-capable compiler, required to build GRGL's native Python
  extension
- Git (with submodule support)

Clone with its submodule, then create the environment and build all Python and
native dependencies:

```bash
git clone --recurse-submodules <repository-url>
cd evo_lmm
uv sync
```

For an existing checkout, initialize the submodule before syncing:

```bash
git submodule update --init --recursive
uv sync
```

`uv` installs the GRGL submodule's Python distribution, `pygrgl`, as an
editable local dependency. GRAPP is currently pinned as reference code for the
evolutionary BOLT-LMM implementation plan; it will become an editable project
dependency when that implementation is scaffolded.

Verify the setup with:

```bash
uv run python -c "import pygrgl, msprime, numpy, scipy, tskit; print('environment ready')"
```

## Development

Run the current entry point with `uv run python main.py`. Keep dependency
changes in `pyproject.toml` and regenerate the lockfile with `uv lock` (or
`uv sync`). See [AGENTS.md](AGENTS.md) for implementation conventions.

The notes are the scientific source of truth. In particular, preserve the
distinction between the observed focal-trait effect `beta_j` and the latent
fitness-trait effect vector `alpha_j` when translating the derivation into
code.
