"""Prepare persistent forward replicates for the GRAPP/evo-lmm benchmark.

Run once with ``uv run python docs/tutorials/prepare_bolt_benchmark.py``.
The ten outputs are shared by the two-panel forward figure and the benchmark;
subsequent fit-only runs load them without rerunning SLiM.
"""

from __future__ import annotations

from pathlib import Path
import sys


_TUTORIAL_DIRECTORY = Path(__file__).resolve().parent
if str(_TUTORIAL_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_TUTORIAL_DIRECTORY))

from slim_forward_simplified import N_REPLICATES, prepare_replicates  # noqa: E402


OUTPUT_DIRECTORY = Path("docs/_artifacts/forward_replicates")


def main() -> None:
    data = prepare_replicates(workers=4, artifact_directory=OUTPUT_DIRECTORY)
    print(
        f"prepared={OUTPUT_DIRECTORY} replicates={len(data)} "
        f"expected_replicates={N_REPLICATES} "
        f"seeds={[item['seed'] for item in data]}"
    )


if __name__ == "__main__":
    main()
