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
genetic-value predictions with BLUP. The parameters have direct evolutionary
interpretations: `sigma_a^2 / W_S` controls selection-induced frequency
dependence, `sigma_b^2` is the focal-trait effect scale, and `rho_ab^2`
controls coupling between the focal and selected traits.

## Repository layout

```text
.
├── grgl/       Git submodule: native GRGL library and its Python package
├── notes/      Scientific specification for the evolutionary model
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

`uv` installs the submodule's Python distribution, `pygrgl`, as an editable
local dependency. This means edits inside `grgl/` take effect on the next
`uv run` invocation (the native extension is rebuilt when its sources change).

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
