# AGENTS.md

## Purpose

This repository implements the evolutionary random-effects model specified in
`notes/stab1_genetics_template.pdf`. The goal is a correct, testable LMM that
uses GRGL-backed genotype data without losing the population-genetic meaning of
the model parameters.

## Repository map

| Path | Responsibility |
| --- | --- |
| `notes/` | Scientific specification; do not edit unless explicitly asked. |
| `grgl/` | Git submodule for GRGL and `pygrgl`; treat as an external dependency. |
| `grapp/` | Git submodule containing the reference GRGL-backed BOLT-LMM implementation. |
| `plan/` | Reviewed implementation plans and design decisions. |
| `main.py` | Command-line entry point while the application surface is small. |
| `pyproject.toml` / `uv.lock` | Python project metadata and reproducible dependency lock. |

Add new model code under a dedicated first-party package (for example,
`src/evo_lmm/`) rather than expanding `main.py`. Put tests in `tests/` with a
parallel module layout.

## Environment and dependencies

- Use `uv` for every Python command: `uv run ...`, `uv sync`, and `uv lock`.
- `pygrgl` is supplied by the checked-out `grgl/` submodule through an editable
  `tool.uv.sources` entry. Do not replace it with a separately pinned PyPI
  dependency without an explicit compatibility decision.
- Initialize submodules before installing: `git submodule update --init --recursive`.
- GRGL compiles native code. Keep build artifacts out of version control and
  document any added system-level build requirement in `README.md`.
- Do not edit files in `grgl/` as part of evo-lmm work. If GRGL needs a change,
  make it in the submodule repository and update this repository's recorded
  submodule commit deliberately.
- Treat `grapp/` as read-only reference code. Implement evo-lmm-specific prior,
  operator, fitting, and testing code in this repository; update the submodule
  commit deliberately when adopting upstream changes.

## Model invariants

- Keep the focal-trait effect `beta_j` distinct from the latent selected-trait
  effect vector `alpha_j`.
- Frequencies are sample allele frequencies `x_hat_j`; define genotype dosage,
  ploidy, centering, and treatment of monomorphic variants explicitly at every
  public data boundary.
- Expose two explicitly named conditional-variance models. Let
  `q_j = x_hat_j * (1 - x_hat_j)` and `tau = sigma_a^2 / W_S`.

  - The **simplified evolutionary model** fixes `rho_ab^2 = 1` and uses the
    two-parameter form

    ```text
    v_j = sigma_b^2 / (1 + 2 * tau * q_j).
    ```

  - The **full evolutionary model** estimates the coupling and uses the
    three-parameter form

    ```text
    v_j = sigma_b^2 * (1 - rho_ab^2 *
          (2 * tau * q_j) / (1 + 2 * tau * q_j)).
    ```

  Do not describe the simplified model as a free reparameterization of the
  full model: it is its exact `rho_ab^2 = 1` specialization. Keep `tau` and
  `sigma_b^2` non-negative, and constrain the full-model `rho_ab^2` to
  `[0, 1]`.
- The weighted GRM must be symmetric positive semidefinite up to numerical
  tolerance. Validate this property in unit tests.
- Use stable parameterizations for optimization (for example log-scales for
  strictly positive variance parameters) and state whether an objective is ML
  or REML.
- Do not silently substitute the alpha-model for the evolutionary weighting.
  Baselines belong in separately named code paths and tests.

## Implementation and tests

- Prefer NumPy/SciPy vectorized operations; avoid materializing a dense
  individual-by-variant matrix when a GRGL traversal can supply the required
  product or summary.
- Keep I/O, GRGL adaptation, model construction, likelihood evaluation, and
  optimization separate so each can be tested independently.
- Give public functions typed signatures and document units, shapes, and
  parameter constraints.
- Add deterministic unit tests for frequency weights, boundary frequencies,
  GRM symmetry/PSD, and likelihood behavior for both model families. Test that
  the full model with `rho_ab^2 = 1` agrees exactly with the simplified model.
  Use a small seeded simulation for each end-to-end test.
- Before handing off changes, run the smallest relevant test suite plus
  `uv run python -c "import pygrgl"` when dependency configuration changes.

## Git hygiene

- Preserve the `grgl` and `grapp` gitlinks and `.gitmodules`; never commit
  submodule build products or vendor a second copy of either dependency.
- Commit source, tests, documentation, and the updated `uv.lock` together when
  dependencies change.
- Keep generated simulation outputs out of the repository unless they are a
  deliberately curated test fixture.
