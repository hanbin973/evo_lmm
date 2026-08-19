"""Fit GRAPP and evo-lmm using already prepared benchmark data.

Run with ``uv run python docs/tutorials/fit_bolt_benchmark.py`` after the
forward-replicate artifacts have been prepared. This script deliberately
contains no SLiM call.
"""

from __future__ import annotations

from pathlib import Path
import sys


_TUTORIAL_DIRECTORY = Path(__file__).resolve().parent
if str(_TUTORIAL_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_TUTORIAL_DIRECTORY))

from bolt_benchmark import (  # noqa: E402
    SEED,
    load_simulation_replicates,
    run_benchmark,
)


DATA_DIRECTORY = Path("docs/_artifacts/bolt_seed_812")
FORWARD_ARTIFACT_DIRECTORY = Path("docs/_artifacts/forward_replicates")


def main() -> None:
    if FORWARD_ARTIFACT_DIRECTORY.exists():
        benchmark = run_benchmark(
            forward_results=load_simulation_replicates(FORWARD_ARTIFACT_DIRECTORY),
        )
    else:
        benchmark = run_benchmark(
            data_directory=DATA_DIRECTORY if DATA_DIRECTORY.exists() else None,
        )
    evo_runtime, evo_sd = benchmark.runtime_summary()["evo-lmm"]
    bolt_runtime, bolt_sd = benchmark.runtime_summary()["GRAPP BOLT-LMM"]
    print(
        f"replicates={benchmark.n_replicates} "
        f"mutations_first={benchmark.data.full_grg.num_mutations} "
        f"evo_lmm_seconds_mean={evo_runtime:.6f} "
        f"evo_lmm_seconds_sd={evo_sd:.6f} "
        f"grapp_bolt_lmm_seconds_mean={bolt_runtime:.6f} "
        f"grapp_bolt_lmm_seconds_sd={bolt_sd:.6f}"
    )
    for replicate in benchmark.replicates:
        print(
            f"seed={replicate.seed} "
            f"evo_lmm_seconds={replicate.evo_seconds:.6f} "
            f"grapp_bolt_lmm_seconds={replicate.bolt_seconds:.6f} "
            f"evo_lmm_sigma_b2={replicate.evo_fit.prior.sigma_b2:.12g} "
            f"evo_lmm_tau={replicate.evo_fit.prior.tau:.12g} "
            f"evo_lmm_sigma_e2={replicate.evo_fit.sigma_e2:.12g} "
            f"grapp_sigma_g2={replicate.bolt_fit.sigma_g2:.12g}"
        )


if __name__ == "__main__":
    main()
