"""Generate the checked-in figures used by the documentation.

This script intentionally performs the expensive simulations locally. Hosted
documentation builds only copy the resulting PNG assets and never invoke SLiM,
GRG conversion, or model fitting.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DOCS_DIRECTORY = Path(__file__).resolve().parent
TUTORIAL_DIRECTORY = DOCS_DIRECTORY / "tutorials"
OUTPUT_DIRECTORY = DOCS_DIRECTORY / "_static" / "generated"
sys.path.insert(0, str(TUTORIAL_DIRECTORY))


def generate() -> None:
    """Regenerate every compute-heavy tutorial figure."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    import variance_components

    variance_results = variance_components.run_replicates()
    variance_figure = variance_components.make_box_plot(variance_results)
    variance_figure.savefig(
        OUTPUT_DIRECTORY / "variance_components.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(variance_figure)

    import slim_forward_simplified

    forward_results = slim_forward_simplified.run_replicates(workers=4)
    forward_figure = slim_forward_simplified.make_summary(forward_results)
    forward_figure.savefig(
        OUTPUT_DIRECTORY / "slim_forward_simplified.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(forward_figure)

    import bolt_benchmark

    benchmark_artifacts = DOCS_DIRECTORY / "_artifacts" / "bolt_seed_812"
    benchmark = bolt_benchmark.run_benchmark(
        data_directory=benchmark_artifacts if benchmark_artifacts.exists() else None,
    )
    benchmark_figure = bolt_benchmark.make_summary(benchmark)
    benchmark_figure.savefig(
        OUTPUT_DIRECTORY / "bolt_benchmark.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(benchmark_figure)


if __name__ == "__main__":
    generate()
