from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import numpy as np


_TUTORIAL_DIRECTORY = Path(__file__).resolve().parents[1] / "docs" / "tutorials"
if str(_TUTORIAL_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_TUTORIAL_DIRECTORY))

import bolt_benchmark  # noqa: E402


def _replicate(seed: int, evo_seconds: float, bolt_seconds: float):
    frequencies = np.array([0.01, 0.1, 0.3, 0.5, 1.0])
    data = SimpleNamespace(
        full_frequencies=frequencies,
        alpha=np.ones(frequencies.size),
    )
    evo_fit = SimpleNamespace(
        prior=SimpleNamespace(
            effect_variances=lambda values: np.full_like(values, 1.0),
        )
    )
    bolt_stats = [SimpleNamespace(a1freq=frequencies[:-1], se=np.ones(4))]
    bolt_fit = SimpleNamespace(sigma_g2=4.0)
    return bolt_benchmark.BenchmarkReplicateResult(
        seed=seed,
        data=data,
        evo_fit=evo_fit,
        bolt_fit=bolt_fit,
        bolt_stats=bolt_stats,
        evo_seconds=evo_seconds,
        bolt_seconds=bolt_seconds,
    )


def test_benchmark_runtime_summary_and_curve_aggregation() -> None:
    first = _replicate(812, 2.0, 1.0)
    second = _replicate(813, 4.0, 3.0)
    result = bolt_benchmark.BenchmarkResult((first, second))

    summary = result.runtime_summary()
    np.testing.assert_allclose(summary["evo-lmm"], (3.0, np.sqrt(2.0)))
    np.testing.assert_allclose(summary["GRAPP BOLT-LMM"], (2.0, np.sqrt(2.0)))
    curves = bolt_benchmark._replicate_curves(first)
    assert set(curves) == {
        "SLiM realization",
        "configured evolutionary prior",
        "evo-lmm fitted prior",
        "GRAPP BOLT-LMM",
    }
    for values in curves.values():
        assert values.shape == bolt_benchmark.MAF_THRESHOLDS.shape

    figure = bolt_benchmark.make_summary(result)
    assert len(figure.axes) == 1
