"""Dense-oracle and matrix-free profiled AI-REML for evolutionary kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from .operators import EvolutionaryLmmOps
from .priors import EvolutionaryPrior, FullPrior, SimplifiedPrior, prior_from_coordinates
from .results import FitDiagnostics, FitResult
from .trace import TraceEstimate, rademacher_probes, spherical_gaussian_probes, xtrace


def _validate_y(y: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(y, dtype=np.float64)
    if values.ndim != 1 or values.size != n:
        raise ValueError(f"phenotype must have shape ({n},)")
    if not np.all(np.isfinite(values)):
        raise ValueError("phenotype must be finite")
    return values


def _initial_coordinates(
    model: str,
    initial: Any,
    *,
    delta: float,
) -> np.ndarray:
    def finish(values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64).copy()
        if model == "full":
            result[2] = float(np.clip(result[2], -20.0, 20.0))
        return result

    if initial is None:
        prior = SimplifiedPrior(1.0, 0.1) if model == "simplified" else FullPrior(1.0, 0.1, 0.5)
        return finish(prior.coordinates(delta))
    if isinstance(initial, EvolutionaryPrior):
        return finish(initial.coordinates(delta))
    if isinstance(initial, Mapping):
        sigma = float(initial.get("sigma_b2", 1.0))
        tau = float(initial.get("tau", 0.1))
        rho = float(initial.get("rho", 0.5 if model == "full" else 1.0))
        prior = FullPrior(sigma, tau, rho) if model == "full" else SimplifiedPrior(sigma, tau)
        return finish(prior.coordinates(float(initial.get("delta", delta))))
    coords = np.asarray(initial, dtype=np.float64)
    expected = 2 if model == "simplified" else 3
    if coords.shape != (expected,):
        raise ValueError(f"initial coordinates must have shape ({expected},)")
    result = coords.copy()
    if model == "full":
        result[2] = float(np.clip(result[2], -20.0, 20.0))
    return result


def _dense_projection(H: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``P_H``, ``H^-1`` and ``log|H|`` for an orthonormal basis."""

    sign, logdet = np.linalg.slogdet(H)
    if sign <= 0.0:
        raise np.linalg.LinAlgError("shape covariance is not positive definite")
    inv_h = np.linalg.inv(H)
    fixed = basis.T @ inv_h @ basis
    sign_fixed, _logdet_fixed = np.linalg.slogdet(fixed)
    if sign_fixed <= 0.0:
        raise np.linalg.LinAlgError("fixed-effect covariance is not positive definite")
    ph = inv_h - inv_h @ basis @ np.linalg.solve(fixed, basis.T @ inv_h)
    return ph, inv_h, float(logdet)


def restricted_log_likelihood(
    y: np.ndarray,
    covariance: np.ndarray,
    covariates: np.ndarray | None = None,
    *,
    scale: float = 1.0,
    include_constant: bool = False,
) -> float:
    """Evaluate the exact restricted log likelihood for a dense covariance.

    ``covariance`` is the full ``V`` matrix.  The profiled form used by the
    fitter is obtained by passing ``scale=1`` and profiling that scale outside
    this function.  A fixed intercept is used when covariates are omitted.
    """

    v = np.asarray(covariance, dtype=np.float64) * float(scale)
    y_arr = np.asarray(y, dtype=np.float64)
    if v.ndim != 2 or v.shape[0] != v.shape[1] or y_arr.shape != (v.shape[0],):
        raise ValueError("covariance and phenotype dimensions do not match")
    n = v.shape[0]
    if covariates is None:
        c = np.ones((n, 1), dtype=np.float64)
    else:
        c = np.asarray(covariates, dtype=np.float64)
        if c.ndim == 1:
            c = c[:, None]
    q_basis, _ = np.linalg.qr(c, mode="reduced")
    ph, inv_v, logdet_v = _dense_projection(v, q_basis)
    fixed = q_basis.T @ inv_v @ q_basis
    _, logdet_fixed = np.linalg.slogdet(fixed)
    q = float(y_arr @ ph @ y_arr)
    d = n - q_basis.shape[1]
    value = -0.5 * (logdet_v + logdet_fixed + q)
    if include_constant:
        value -= 0.5 * d * np.log(2.0 * np.pi)
    return float(value)


def exact_reml_score(
    y: np.ndarray,
    H: np.ndarray,
    derivatives: Sequence[np.ndarray],
    covariates: np.ndarray | None = None,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Return exact shape scores for ``V = scale * H``.

    The scale is normally the profiled ``sigma_b2``.  Omitting it evaluates the
    equivalent unit-scale score.
    """

    h = np.asarray(H, dtype=np.float64)
    n = h.shape[0]
    c = np.ones((n, 1), dtype=np.float64) if covariates is None else np.asarray(covariates, dtype=np.float64)
    if c.ndim == 1:
        c = c[:, None]
    basis, _ = np.linalg.qr(c, mode="reduced")
    ph, _inv, _logdet = _dense_projection(h, basis)
    y_arr = _validate_y(y, n)
    xi = ph @ y_arr
    return np.asarray(
        [0.5 * ((xi @ derivative @ xi) / float(scale) - np.trace(ph @ derivative)) for derivative in derivatives],
        dtype=np.float64,
    )


def exact_average_information(
    y: np.ndarray,
    H: np.ndarray,
    derivatives: Sequence[np.ndarray],
    covariates: np.ndarray | None = None,
) -> np.ndarray:
    """Return the direct quadratic-form average-information matrix."""

    h = np.asarray(H, dtype=np.float64)
    n = h.shape[0]
    c = np.ones((n, 1), dtype=np.float64) if covariates is None else np.asarray(covariates, dtype=np.float64)
    if c.ndim == 1:
        c = c[:, None]
    basis, _ = np.linalg.qr(c, mode="reduced")
    ph, _inv, _logdet = _dense_projection(h, basis)
    xi = ph @ _validate_y(y, n)
    solved = [ph @ derivative @ xi for derivative in derivatives]
    result = np.empty((len(derivatives), len(derivatives)), dtype=np.float64)
    for i, derivative in enumerate(derivatives):
        for j, value in enumerate(solved):
            result[i, j] = 0.5 * float(xi @ derivative @ value)
    return (result + result.T) * 0.5


def profiled_average_information(
    y: np.ndarray,
    H: np.ndarray,
    derivatives: Sequence[np.ndarray],
    covariates: np.ndarray | None = None,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Return the shape AI after eliminating the profiled scale coordinate.

    The Schur complement is formed from the full coordinate vector
    ``(log_sigma_b2, shape...)`` with ``H_scale = H``.  ``scale`` is the
    current profiled ``sigma_b2``.
    """

    h = np.asarray(H, dtype=np.float64)
    n = h.shape[0]
    c = np.ones((n, 1), dtype=np.float64) if covariates is None else np.asarray(covariates, dtype=np.float64)
    if c.ndim == 1:
        c = c[:, None]
    basis, _ = np.linalg.qr(c, mode="reduced")
    ph, _inv, _logdet = _dense_projection(h, basis)
    xi = ph @ _validate_y(y, n)
    q = float(xi @ h @ xi)
    direct = exact_average_information(y, h, derivatives, basis)
    data = np.asarray([xi @ derivative @ xi for derivative in derivatives], dtype=np.float64)
    if q <= 0.0:
        return direct / max(float(scale), np.finfo(float).tiny)
    return (direct - np.outer(data, data) / (2.0 * q)) / max(float(scale), np.finfo(float).tiny)


@dataclass
class _Quantities:
    coordinates: np.ndarray
    prior: EvolutionaryPrior
    delta: float
    sigma_b2: float
    sigma_e2: float
    q: float
    ph_y: np.ndarray
    derivatives: list[str]
    derivative_vectors: list[np.ndarray]
    score: np.ndarray
    ai: np.ndarray
    objective: float
    trace_errors: dict[str, float]
    solve_cache: dict[str, np.ndarray]
    cg_iterations: list[int]
    warm_start_hits: int
    warm_start_rejections: int
    cg_initial_residuals: list[float]
    cg_final_residuals: list[float]


def _coordinate_names(model: str) -> list[str]:
    return ["log_delta", "log_tau"] + (["logit_r"] if model == "full" else [])


def _profile_objective_dense(ops: EvolutionaryLmmOps, y: np.ndarray, coordinates: np.ndarray) -> float:
    prior, delta = prior_from_coordinates(ops.model_name, coordinates)
    H = ops.dense_kernel(prior) + delta * np.eye(ops.n)
    basis = ops.basis
    ph, inv_h, logdet_h = _dense_projection(H, basis)
    fixed = basis.T @ inv_h @ basis
    _, logdet_fixed = np.linalg.slogdet(fixed)
    q = float(y @ ph @ y)
    d = ops.dim
    if q <= 0.0:
        return np.inf
    return float(0.5 * (logdet_h + logdet_fixed + d * np.log(q / d)))


def _quantities(
    ops: EvolutionaryLmmOps,
    y: np.ndarray,
    coordinates: np.ndarray,
    probes: np.ndarray,
    *,
    exact: bool,
    cg_tol: float = 1e-9,
    exclude_chrom: Any = None,
    initial_cache: Mapping[str, np.ndarray] | None = None,
    trace_method: str = "hutchinson",
) -> _Quantities:
    prior, delta = prior_from_coordinates(ops.model_name, coordinates)
    names = _coordinate_names(ops.model_name)
    if exact and all(getattr(chrom, "dense", None) is not None for chrom in ops._chromosomes):
        # Dense exact path uses the same projected kernel as matrix-free code.
        k = ops.dense_kernel(prior, exclude_chrom=exclude_chrom)
        h = k + delta * np.eye(ops.n)
        basis = ops.basis
        ph, inv_h, logdet_h = _dense_projection(h, basis)
        fixed = basis.T @ inv_h @ basis
        _, logdet_fixed = np.linalg.slogdet(fixed)
        ph_y = ph @ y
        derivative_matrices: list[np.ndarray] = []
        for name in names:
            if name == "log_delta":
                derivative_matrices.append(delta * np.eye(ops.n))
            else:
                derivative_matrices.append(_dense_derivative_kernel(ops, prior, name, exclude_chrom))
        q = float(y @ ph_y)
        sigma_b2 = q / max(ops.dim, 1)
        sigma_e2 = delta * sigma_b2
        score = exact_reml_score(y, h, derivative_matrices, basis, scale=sigma_b2)
        ai = profiled_average_information(y, h, derivative_matrices, basis, scale=sigma_b2)
        objective = float(0.5 * (logdet_h + logdet_fixed + ops.dim * np.log(max(q / ops.dim, 1e-300))))
        return _Quantities(
            coordinates, prior, delta, sigma_b2, sigma_e2, q, ph_y, names,
            derivative_matrices, score, ai, objective,
            {name: 0.0 for name in names}, {}, [], 0, 0, [], [],
        )

    cache_in = {} if initial_cache is None else dict(initial_cache)
    cache_out: dict[str, np.ndarray] = {}
    cg_iterations: list[int] = []
    warm_start_hits = 0
    warm_start_rejections = 0
    cg_initial_residuals: list[float] = []
    cg_final_residuals: list[float] = []

    def solve(rhs: np.ndarray, key: str) -> np.ndarray:
        nonlocal warm_start_hits, warm_start_rejections
        stats: dict[str, Any] = {}
        initial = cache_in.get(key)
        result = ops.solve_ph(
            rhs,
            coordinates,
            exclude_chrom,
            tol=cg_tol,
            initial=initial,
            stats=stats,
        )
        result_matrix = np.asarray(result, dtype=np.float64)
        if result_matrix.ndim == 1:
            result_matrix = result_matrix[:, None]
        cache_out[key] = result_matrix.copy()
        cg_iterations.append(int(stats.get("iterations", 0)))
        warm_start_hits += int(stats.get("warm_used", 0))
        warm_start_rejections += int(stats.get("warm_rejected", 0))
        cg_initial_residuals.append(float(stats.get("initial_residual_norm", 0.0)))
        cg_final_residuals.append(float(stats.get("final_residual_norm", 0.0)))
        return result_matrix

    if trace_method not in ("xtrace", "hutchinson"):
        raise ValueError("trace_method must be 'xtrace' or 'hutchinson'")

    # Matrix-free projected inverse stage. Hutchinson retains the historical
    # shared multi-RHS solve; XTrace solves only the phenotype here because its
    # derivative-specific query systems are constructed below.
    if trace_method == "hutchinson":
        solved = solve(
            np.column_stack((np.asarray(y, dtype=np.float64), probes)),
            "phenotype+trace",
        )
        ph_y = solved[:, 0]
        hutch_solved = solved[:, 1:]
    else:
        ph_y = solve(np.asarray(y, dtype=np.float64)[:, None], "phenotype")[:, 0]
        hutch_solved = None
    q = float(y @ ph_y)
    sigma_b2 = q / max(ops.dim, 1)
    sigma_e2 = delta * sigma_b2
    derivative_vectors = []
    score = np.empty(len(names), dtype=np.float64)
    trace_errors: dict[str, float] = {}
    for index, name in enumerate(names):
        derivative_vectors.append(ops.apply_dh(ph_y, coordinates, name, exclude_chrom))
        data_quad = float(ph_y @ derivative_vectors[-1])

        if trace_method == "hutchinson":
            applied = ops.apply_dh_matmat(probes, coordinates, name, exclude_chrom)
            samples = np.sum(hutch_solved * applied, axis=0)
            estimate = TraceEstimate(
                float(np.mean(samples)),
                float(np.std(samples, ddof=1) / np.sqrt(samples.size)) if samples.size > 1 else 0.0,
                "hutchinson",
                int(samples.size),
            )
        else:
            query = 0

            def trace_apply(values: np.ndarray) -> np.ndarray:
                nonlocal query
                rhs = ops.apply_dh_matmat(values, coordinates, name, exclude_chrom)
                key = f"trace:{name}:omega" if query == 0 else f"trace:{name}:q"
                query += 1
                return solve(rhs, key)

            estimate = xtrace(trace_apply, probes)
        trace_errors[name] = float(estimate.standard_error)
        score[index] = 0.5 * (data_quad / max(sigma_b2, np.finfo(float).tiny) - estimate.value)

    # Use the derivative RHS stage for the average-information matrix.
    eta = np.column_stack(derivative_vectors) if derivative_vectors else np.empty((ops.n, 0))
    zeta = solve(eta, "derivative") if eta.shape[1] else eta
    ai = np.empty((len(names), len(names)), dtype=np.float64)
    for i in range(len(names)):
        for j in range(len(names)):
            ai[i, j] = 0.5 * float(ph_y @ ops.apply_dh(zeta[:, j], coordinates, names[i], exclude_chrom)) / max(sigma_b2, np.finfo(float).tiny)
    ai = (ai + ai.T) * 0.5
    data_quadratics = np.asarray([float(ph_y @ value) for value in derivative_vectors], dtype=np.float64)
    if q > 0.0:
        ai -= np.outer(data_quadratics, data_quadratics) / (2.0 * max(sigma_b2 * q, np.finfo(float).tiny))
    return _Quantities(
        coordinates, prior, delta, sigma_b2, sigma_e2, q, ph_y, names,
        derivative_vectors, score, ai, float("nan"), trace_errors,
        cache_out, cg_iterations, warm_start_hits, warm_start_rejections,
        cg_initial_residuals, cg_final_residuals,
    )


def _dense_derivative_kernel(
    ops: EvolutionaryLmmOps,
    prior: EvolutionaryPrior,
    name: str,
    exclude_chrom: Any,
) -> np.ndarray:
    # Applying to each canonical basis is acceptable only for the dense oracle.
    n = ops.n
    return np.column_stack(
        [ops.apply_dh(np.eye(n)[:, i], {"delta": 1.0, "tau": prior.tau, "rho": getattr(prior, "rho", 1.0)}, name, exclude_chrom) for i in range(n)]
    )


def fit_reml(
    ops: EvolutionaryLmmOps,
    y: np.ndarray,
    *,
    model: str | None = None,
    initial: Any = None,
    delta: float = 1.0,
    trace_probes: int = 64,
    seed: int = 0,
    max_iter: int = 50,
    tol: float = 1e-6,
    cg_tol: float = 1e-9,
    max_step: float = 2.0,
    exact: bool | None = None,
    trace_method: str = "hutchinson",
    warm_start: bool = True,
    initialization: str = "default",
) -> FitResult:
    """Fit evolutionary shape parameters by profiled average-information REML.

    ``sigma_b2`` is profiled as ``y'P_H y / (N-rank(C))`` and ``sigma_e2`` is
    derived as ``delta*sigma_b2``. Dense operators use exact traces by default;
    matrix-free operators use fixed spherical XTrace vectors and warm-started
    projected CG solves.
    """

    if model is not None and model != ops.model_name:
        raise ValueError("model does not match EvolutionaryLmmOps.model")
    if initialization not in ("default", "he"):
        raise ValueError("initialization must be 'default' or 'he'")
    model_name = ops.model_name
    y_arr = _validate_y(y, ops.n)
    diagnostics_warnings: set[str] = set()
    fixed_r_boundary = bool(
        model_name == "full"
        and (
            isinstance(initial, FullPrior)
            and initial.rho2 == 1.0
            or isinstance(initial, Mapping)
            and float(initial.get("rho", 0.0)) >= 1.0
            or isinstance(initial, np.ndarray)
            and np.asarray(initial).size == 3
            and float(np.asarray(initial)[2]) >= 20.0
        )
    )
    coords = _initial_coordinates(model_name, initial, delta=delta)
    # Repeated initialization at tau=0 is represented by the lowest finite
    # transformed value while retaining an explicit boundary warning.
    coords[1] = max(coords[1], np.log(np.finfo(np.float64).tiny))
    if model_name == "full":
        coords[2] = float(np.clip(coords[2], -20.0, 20.0))
    if initialization == "he":
        # HE estimates the residual-to-genetic variance ratio while retaining
        # the requested (or default) evolutionary shape parameters.  The
        # scale itself is profiled by REML, so only its delta is carried into
        # transformed coordinates.
        he_prior, _ = prior_from_coordinates(model_name, coords)
        _he_sigma_b2, _he_sigma_e2, he_delta = haseman_elston_initialization(
            ops, y_arr, he_prior, probes=trace_probes, seed=seed
        )
        if np.isfinite(he_delta) and he_delta > 0.0:
            coords[0] = float(np.log(he_delta))
        else:
            diagnostics_warnings = {"HE initialization was invalid; using requested/default delta"}
    if trace_method not in ("xtrace", "hutchinson"):
        raise ValueError("trace_method must be 'xtrace' or 'hutchinson'")
    probe_count = max(int(trace_probes), 2)
    probes = (
        spherical_gaussian_probes(ops.n, probe_count, seed)
        if trace_method == "xtrace"
        else rademacher_probes(ops.n, probe_count, seed)
    )
    is_dense = all(chrom.dense is not None for chrom in ops._chromosomes)
    use_exact = is_dense if exact is None else bool(exact)
    converged = False
    last_step = 0.0
    last_q: _Quantities | None = None
    ai_condition = np.inf
    damping = 0.0
    accepted_iterations = 0
    accepted_cache: dict[str, np.ndarray] = {}

    for iteration in range(1, int(max_iter) + 1):
        try:
            current = _quantities(
                ops, y_arr, coords, probes, exact=use_exact, cg_tol=cg_tol,
                initial_cache=accepted_cache if warm_start else {},
                trace_method=trace_method,
            )
        except (np.linalg.LinAlgError, ValueError):
            damping = max(1e-6, damping * 10.0 if damping else 1e-6)
            coords[0] += 1.0
            continue
        last_q = current
        ai = (current.ai + current.ai.T) * 0.5
        score = current.score
        active_coordinates = np.array([0, 1], dtype=np.int64) if fixed_r_boundary else np.arange(score.size)
        active_score = score[active_coordinates]
        active_ai = ai[np.ix_(active_coordinates, active_coordinates)]
        if np.all(np.isfinite(ai)) and ai.size:
            ai_condition = float(np.linalg.cond(active_ai))
        if np.linalg.norm(active_score, ord=np.inf) <= tol and (last_step <= tol or iteration > 1):
            converged = True
            accepted_iterations = iteration - 1
            break
        if not np.all(np.isfinite(ai)) or ai_condition > 1e12:
            damping = max(damping, 1e-8 * max(float(np.trace(np.abs(active_ai))) / max(active_ai.shape[0], 1), 1.0))
        trial_ai = active_ai + damping * np.eye(active_ai.shape[0])
        try:
            step = np.linalg.solve(trial_ai, active_score)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(trial_ai, active_score, rcond=None)[0]
        step = np.asarray(step, dtype=np.float64)
        step_norm = float(np.max(np.abs(step))) if step.size else 0.0
        if step_norm > max_step:
            step *= max_step / step_norm
        old_norm = float(np.linalg.norm(active_score / np.sqrt(np.maximum(np.diag(trial_ai), 1e-12))))
        accepted = False
        trial_cache = current.solve_cache if warm_start else {}
        for halving in range(12):
            trial_coords = coords.copy()
            trial_coords[active_coordinates] += step * (0.5**halving)
            try:
                trial = _quantities(
                    ops, y_arr, trial_coords, probes, exact=use_exact,
                    cg_tol=cg_tol, initial_cache=trial_cache,
                    trace_method=trace_method,
                )
                trial_cache = trial.solve_cache
                trial_ai_active = trial.ai[np.ix_(active_coordinates, active_coordinates)]
                new_norm = float(np.linalg.norm(trial.score[active_coordinates] / np.sqrt(np.maximum(np.diag(trial_ai_active + damping * np.eye(trial_ai_active.shape[0])), 1e-12))))
                objective_ok = not use_exact or trial.objective <= current.objective + 1e-10
                if objective_ok and (new_norm <= old_norm or np.max(np.abs(step)) * (0.5**halving) <= tol):
                    coords = trial_coords
                    accepted_cache = trial.solve_cache
                    last_step = float(np.max(np.abs(step)) * (0.5**halving))
                    accepted = True
                    accepted_iterations = iteration
                    break
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                continue
        if not accepted:
            last_step = 0.0
            damping = max(1e-6, damping * 10.0 if damping else 1e-6)
            if np.linalg.norm(active_score, ord=np.inf) <= 10.0 * tol:
                converged = True
                break

    if last_q is None:
        raise RuntimeError("REML fitting failed before evaluating a valid covariance")

    # Exact dense minimisation is a robust finishing step for small matrices.
    # It uses the same analytic score and only repairs an AI iteration that was
    # stopped by a conservative damping/step-halving decision.
    if use_exact and not converged:
        def objective(value: np.ndarray) -> float:
            full_value = np.r_[value, 20.0] if fixed_r_boundary else value
            try:
                return _profile_objective_dense(ops, y_arr, full_value)
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                return 1e100

        def jac(value: np.ndarray) -> np.ndarray:
            full_value = np.r_[value, 20.0] if fixed_r_boundary else value
            try:
                return -_quantities(ops, y_arr, full_value, probes, exact=True, cg_tol=cg_tol).score[:2] if fixed_r_boundary else -_quantities(ops, y_arr, full_value, probes, exact=True, cg_tol=cg_tol).score
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                return np.zeros_like(value)

        bounds = [(-30.0, 30.0), (-30.0, 30.0)] + ([] if fixed_r_boundary else ([( -20.0, 20.0)] if model_name == "full" else []))
        optimizer_coords = coords[:2] if fixed_r_boundary else coords
        result = minimize(objective, optimizer_coords, jac=jac, method="L-BFGS-B", bounds=bounds, options={"maxiter": max_iter * 4, "ftol": 1e-12, "gtol": tol})
        if np.isfinite(result.fun):
            coords = np.r_[result.x, 20.0] if fixed_r_boundary else np.asarray(result.x, dtype=np.float64)
            converged = bool(result.success) or np.linalg.norm(jac(coords), ord=np.inf) < 1e-4
            last_q = _quantities(ops, y_arr, coords, probes, exact=True, cg_tol=cg_tol)
            accepted_iterations += int(result.nit)

    # The shape object used internally is unit-scale; report the profiled
    # scientific scale on the public prior object.
    prior = last_q.prior.with_sigma_b2(last_q.sigma_b2)
    delta_value = last_q.delta
    # The transformed boundary values are clipped only for reporting; the prior
    # itself remains scientifically valid and its warning is explicit.
    warnings = set(diagnostics_warnings)
    if prior.tau <= 1e-8:
        warnings.add("tau is at or near zero; frequency-shape parameter is weakly identified")
    if isinstance(prior, FullPrior):
        if prior.rho2 <= 1e-10:
            warnings.add("rho^2 is at or near zero; tau is weakly identified")
        if 1.0 - prior.rho2 <= 1e-10:
            warnings.add("rho^2 is at the simplified-model boundary")
    score_norm = float(np.linalg.norm(last_q.score, ord=np.inf))
    if use_exact:
        objective_value = float(last_q.objective)
    else:
        objective_value = float("nan")
    diagnostics = FitDiagnostics(
        converged=converged,
        iterations=max(accepted_iterations, 1),
        objective=objective_value,
        score_norm=score_norm,
        ai_condition=float(ai_condition),
        ai_damping=float(damping),
        accepted_step=float(last_step),
        trace_estimator="exact" if use_exact else trace_method,
        trace_probes=0 if use_exact else probe_count,
        initialization=initialization,
        trace_operator_queries=(
            0 if use_exact else (2 * probe_count if trace_method == "xtrace" else probe_count)
        ),
        trace_standard_errors=dict(last_q.trace_errors),
        cg_iterations=list(last_q.cg_iterations),
        cg_warm_start_hits=int(last_q.warm_start_hits),
        cg_warm_start_rejections=int(last_q.warm_start_rejections),
        cg_initial_residual_norms=list(last_q.cg_initial_residuals),
        cg_final_residual_norms=list(last_q.cg_final_residuals),
        random_seed=int(seed),
        boundary_hits=tuple(sorted(warnings)),
        warnings=tuple(sorted(warnings)),
    )
    return FitResult(
        prior=prior,
        sigma_b2=float(last_q.sigma_b2),
        sigma_e2=float(last_q.sigma_e2),
        delta=float(delta_value),
        h2=float(last_q.sigma_b2 * ops.kernel_trace(prior) / max(last_q.sigma_b2 * ops.kernel_trace(prior) + last_q.sigma_e2 * ops.dim, np.finfo(float).tiny)),
        log_likelihood=float(-last_q.objective if use_exact else float("nan")),
        fixed_effects=np.linalg.lstsq(ops.basis, y_arr - last_q.ph_y, rcond=None)[0],
        projected_phenotype=ops.project(y_arr),
        ph_y=last_q.ph_y,
        diagnostics=diagnostics,
        model=model_name,
        ops=ops,
    )


class DenseREMLOracle:
    """Convenient exact REML oracle for tests and small dense simulations."""

    def __init__(self, genotypes: np.ndarray, frequencies: np.ndarray, covariates: np.ndarray | None = None, *, model: str = "simplified") -> None:
        self.ops = EvolutionaryLmmOps.from_dense(genotypes, frequencies, covariates, model=model)

    def kernel(self, prior: EvolutionaryPrior) -> np.ndarray:
        return self.ops.dense_kernel(prior)

    def fit(self, y: np.ndarray, **kwargs: Any) -> FitResult:
        return fit_reml(self.ops, y, exact=True, **kwargs)

    def log_likelihood(self, y: np.ndarray, prior: EvolutionaryPrior, sigma_b2: float, sigma_e2: float) -> float:
        v = sigma_b2 * self.kernel(prior) + sigma_e2 * np.eye(self.ops.n)
        return restricted_log_likelihood(y, v, self.ops.basis)


def fit_dense_reml(
    genotypes: np.ndarray,
    frequencies: np.ndarray,
    y: np.ndarray,
    *,
    covariates: np.ndarray | None = None,
    model: str = "simplified",
    **kwargs: Any,
) -> FitResult:
    """Fit the exact dense oracle in one call."""

    oracle = DenseREMLOracle(genotypes, frequencies, covariates, model=model)
    return oracle.fit(y, **kwargs)


def haseman_elston_initialization(
    ops: EvolutionaryLmmOps,
    y: np.ndarray,
    prior: EvolutionaryPrior,
    *,
    probes: int = 64,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Estimate ``(sigma_b2, sigma_e2, delta)`` by projected HE moments.

    The moment solution is intentionally only an initializer.  Exact dense
    traces are used for dense operators; spherical XTrace estimates are used
    for GRG operators.
    """

    y_arr = _validate_y(y, ops.n)
    projected = ops.project(y_arr)
    k_apply = lambda value: ops.apply_k(value, prior)
    if all(chrom.dense is not None for chrom in ops._chromosomes):
        kernel = ops.dense_kernel(prior)
        projector = np.eye(ops.n) - ops.basis @ ops.basis.T
        pkp = projector @ kernel @ projector
        a = float(np.trace(pkp @ pkp))
    else:
        probe_values = spherical_gaussian_probes(ops.n, max(int(probes), 2), seed)

        def squared_kernel_apply(values: np.ndarray) -> np.ndarray:
            return np.column_stack(
                [ops.project(k_apply(ops.project(values[:, i]))) for i in range(values.shape[1])]
            )

        a = float(xtrace(squared_kernel_apply, probe_values).value)
    b = float(ops.kernel_trace(prior))
    d = float(max(ops.dim, 1))
    rhs = np.array([projected @ k_apply(projected), projected @ projected], dtype=np.float64)
    matrix = np.array([[a, b], [b, d]], dtype=np.float64)
    try:
        sigma_b2, sigma_e2 = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        sigma_b2, sigma_e2 = np.nan, np.nan
    if not np.isfinite(sigma_b2) or not np.isfinite(sigma_e2) or sigma_b2 <= 0.0 or sigma_e2 <= 0.0:
        variance = float(projected @ projected / d)
        sigma_b2 = max(variance * 0.5 / max(b / d, 1.0), np.finfo(float).tiny)
        sigma_e2 = max(variance - sigma_b2 * b / d, variance * 0.5, np.finfo(float).tiny)
    return float(sigma_b2), float(sigma_e2), float(sigma_e2 / sigma_b2)


# Descriptive aliases used by callers that want to distinguish this fitter
# from conventional one-component REML implementations.
fit_evolutionary_reml = fit_reml
exact_reml_loglikelihood = restricted_log_likelihood
