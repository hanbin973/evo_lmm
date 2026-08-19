"""Fit GRAPP and evo-lmm using already prepared benchmark data.

Run with ``uv run python docs/tutorials/fit_bolt_benchmark.py`` after
``prepare_bolt_benchmark.py``. This script deliberately contains no SLiM call.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

from evo_lmm import EvolutionaryLmmOps, SimplifiedPrior, fit_reml
from evo_lmm.grapp_backend import wrap_grg


_TUTORIAL_DIRECTORY = Path(__file__).resolve().parent
if str(_TUTORIAL_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_TUTORIAL_DIRECTORY))

from bolt_benchmark import (  # noqa: E402
    CG_TOL,
    N_INDIVIDUALS,
    SEED,
    SIGMA_A2,
    TRACE_PROBES,
    TRUE_TAU,
    load_benchmark_data,
)


DATA_DIRECTORY = Path("docs/_artifacts/bolt_seed_812")


def main() -> None:
    data = load_benchmark_data(DATA_DIRECTORY)
    initial = SimplifiedPrior(sigma_b2=SIGMA_A2, tau=TRUE_TAU)
    evo_ops = EvolutionaryLmmOps(
        data.blocks,
        frequencies=data.block_frequencies,
        model="simplified",
    )
    start = time.perf_counter()
    evo_fit = fit_reml(
        evo_ops,
        data.phenotype,
        initial=initial,
        trace_probes=TRACE_PROBES,
        max_iter=8,
        cg_tol=CG_TOL,
        seed=SEED + 2,
    )
    evo_seconds = time.perf_counter() - start

    from grapp.assoc.bolt_inf_core import CovariateBasis
    from grapp.assoc.bolt_lmm import bolt_lmm_inf

    grapp_blocks = [(label, wrap_grg(grg)) for label, grg in data.bolt_blocks]
    covariates = CovariateBasis.intercept_only(N_INDIVIDUALS)
    start = time.perf_counter()
    bolt_fit, _calibration, _residuals, _stats = bolt_lmm_inf(
        grapp_blocks,
        data.phenotype,
        covariates,
        mc_trials=TRACE_PROBES,
        cg_tol=CG_TOL,
        seed=SEED + 2,
        threads=1,
        batched_apply_x=True,
    )
    bolt_seconds = time.perf_counter() - start
    print(
        f"mutations={data.full_grg.num_mutations} "
        f"evo_lmm_seconds={evo_seconds:.6f} "
        f"grapp_bolt_lmm_seconds={bolt_seconds:.6f}"
    )
    print(
        f"evo_lmm_sigma_b2={evo_fit.prior.sigma_b2:.12g} "
        f"evo_lmm_tau={evo_fit.prior.tau:.12g} "
        f"evo_lmm_sigma_e2={evo_fit.sigma_e2:.12g} "
        f"grapp_sigma_g2={bolt_fit.sigma_g2:.12g}"
    )
    print(
        f"evo_trace_estimator={evo_fit.diagnostics.trace_estimator} "
        f"evo_trace_probes={evo_fit.diagnostics.trace_probes} "
        f"evo_trace_queries={evo_fit.diagnostics.trace_operator_queries} "
        f"evo_cg_iterations={sum(evo_fit.diagnostics.cg_iterations)} "
        f"evo_warm_hits={evo_fit.diagnostics.cg_warm_start_hits}"
    )


if __name__ == "__main__":
    main()
