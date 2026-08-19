"""Sequential profile of projected-CG warm starts and XTrace.

Run from the repository root with ``uv run python docs/tutorials/trace_profile.py``.
The simulation is shared by all rows; only the REML fitting path is timed.
"""

from __future__ import annotations

import time

from evo_lmm import EvolutionaryLmmOps, SimplifiedPrior, fit_reml

from bolt_benchmark import CG_TOL, SEED, SIGMA_A2, TRUE_TAU, simulate_forward_data


def main() -> None:
    data = simulate_forward_data()
    initial = SimplifiedPrior(sigma_b2=SIGMA_A2, tau=TRUE_TAU)
    configurations = (
        ("cold-hutchinson", "hutchinson", False, 15),
        ("warm-hutchinson", "hutchinson", True, 15),
        ("warm-xtrace-15", "xtrace", True, 15),
        ("warm-xtrace-5", "xtrace", True, 5),
    )
    for label, method, warm_start, probes in configurations:
        ops = EvolutionaryLmmOps(
            data.blocks,
            frequencies=data.block_frequencies,
            model="simplified",
        )
        start = time.perf_counter()
        fit = fit_reml(
            ops,
            data.phenotype,
            initial=initial,
            trace_probes=probes,
            max_iter=8,
            cg_tol=CG_TOL,
            seed=SEED + 2,
            trace_method=method,
            warm_start=warm_start,
        )
        elapsed = time.perf_counter() - start
        print(
            f"{label} seconds={elapsed:.6f} "
            f"sigma_b2={fit.prior.sigma_b2:.12g} "
            f"tau={fit.prior.tau:.12g} "
            f"sigma_e2={fit.sigma_e2:.12g} "
            f"delta={fit.delta:.12g} "
            f"iterations={fit.diagnostics.iterations} "
            f"converged={fit.diagnostics.converged} "
            f"cg_iterations={sum(fit.diagnostics.cg_iterations)} "
            f"warm_hits={fit.diagnostics.cg_warm_start_hits} "
            f"trace_standard_errors={fit.diagnostics.trace_standard_errors}"
        )


if __name__ == "__main__":
    main()
