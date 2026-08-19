"""Prepare persistent data for the GRAPP/evo-lmm benchmark.

Run once with ``uv run python docs/tutorials/prepare_bolt_benchmark.py``.
Subsequent fit-only runs load the resulting directory and do not rerun SLiM.
"""

from __future__ import annotations

from pathlib import Path
import sys


_TUTORIAL_DIRECTORY = Path(__file__).resolve().parent
if str(_TUTORIAL_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_TUTORIAL_DIRECTORY))

from bolt_benchmark import SEED, prepare_benchmark_data  # noqa: E402


OUTPUT_DIRECTORY = Path("docs/_artifacts/bolt_seed_812")


def main() -> None:
    data = prepare_benchmark_data(OUTPUT_DIRECTORY, SEED)
    print(
        f"prepared={OUTPUT_DIRECTORY} seed={SEED} "
        f"individuals={data.full_grg.num_individuals} "
        f"mutations={data.full_grg.num_mutations}"
    )


if __name__ == "__main__":
    main()
