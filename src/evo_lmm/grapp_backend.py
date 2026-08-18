"""Small, isolated compatibility layer for the pinned GRAPP submodule.

No model code imports GRAPP internals directly.  This keeps the evolutionary
implementation independent of GRAPP's private BOLT classes while reusing its
raw GRGL ``X`` operator and frequency traversal.
"""

from __future__ import annotations

from typing import Any, Callable


GRAPP_COMMIT = "be6e2419d6fef51a0a0c6ebe938646ded88c98a9"


def wrap_grg(grg: Any) -> Any:
    """Return GRAPP's calculator adapter for a raw ``pygrgl.GRG`` or adapter."""

    from grapp.grg_calculator import _wrap_grg

    return _wrap_grg(grg)


def raw_operator_class(grg: Any) -> Callable[..., Any]:
    """Return the CPU/GPU raw ``X`` operator selected for ``grg``."""

    return wrap_grg(grg).get_operator("X", standardized=False)


def allele_frequencies(
    grg: Any, *, sample_filter: list[int] | None = None, adjust_missing: bool = True
):
    """Delegate only the GRAPP frequency traversal to its public utility."""

    from grapp.util.simple import allele_frequencies as _allele_frequencies

    return _allele_frequencies(
        grg, sample_filter=sample_filter, adjust_missing=adjust_missing
    )


def covariate_basis(matrix):
    """Create GRAPP's orthonormal covariate basis when integration needs it."""

    from grapp.assoc.bolt_inf_core import CovariateBasis

    return CovariateBasis.from_matrix(
        matrix,
        covar_cols=(),
        q_covar_cols=(),
        covar_max_levels=10,
    )


def assert_compatible() -> None:
    """Fail early if the checked-out GRAPP API no longer matches this adapter."""

    from grapp.grg_calculator import GRGCalcInterface

    required = ("get_operator", "get_multi_operator", "num_mutations", "ploidy")
    missing = [name for name in required if not hasattr(GRGCalcInterface, name)]
    if missing:
        raise RuntimeError(f"incompatible GRAPP API; missing {missing}")

