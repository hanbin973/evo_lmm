import csv
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


BENCHMARK = (
    Path(__file__).parents[1] / "benchmarks" / "multicomponent" / "plot_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location("multicomponent_benchmark", BENCHMARK)
assert SPEC is not None and SPEC.loader is not None
plot_benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot_benchmark)


def _row(replicate: int) -> dict[str, object]:
    row = {
        "replicate": replicate,
        "seed": 917 + replicate,
        "n_individuals": 20,
        "n_mutations": 30,
        "simulation_seconds": 1.0 + replicate,
        "fit_seconds": 2.0 + replicate,
        "runtime_seconds": 3.0 + 2 * replicate,
        "status": "converged",
        "converged": True,
        "sigma_e2_estimate": 0.4 + 0.01 * replicate,
        "sigma_e2_generating": 0.4,
    }
    sigma_b2 = {"lof": 2.25, "missense": 1.0, "synonymous": 0.25}
    for label in plot_benchmark.CATEGORIES:
        row[f"sigma_b2_{label}_estimate"] = sigma_b2[label]
        row[f"sigma_b2_{label}_generating"] = sigma_b2[label]
        row[f"tau_{label}_estimate"] = sigma_b2[label] / 2.0
        row[f"tau_{label}_generating"] = sigma_b2[label] / 2.0
    return row


def test_results_and_figure_include_runtime_estimates_and_truth_lines(
    tmp_path: Path,
) -> None:
    rows = [_row(index) for index in range(10)]
    results = tmp_path / "multicomponent.csv"
    figure = tmp_path / "multicomponent.png"
    plot_benchmark.write_summary(rows, results)
    rendered = plot_benchmark.make_figure(rows, figure)

    with results.open(newline="", encoding="utf-8") as stream:
        saved = list(csv.DictReader(stream))
    assert len(saved) == 10
    assert saved[0]["tau_lof_generating"] == str(2.25 / 2.0)
    assert figure.exists() and figure.stat().st_size > 0
    assert len(rendered.axes) == 3
    assert rendered.axes[1].get_yscale() == "linear"
    assert rendered.axes[2].get_yscale() == "log"
