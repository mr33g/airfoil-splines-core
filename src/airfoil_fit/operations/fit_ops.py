from __future__ import annotations

import logging
import numpy as np
from scipy import optimize

from airfoil_fit.optimization import (
    OptimizationLayout,
    build_g2_problem,
    control_points_to_initial_vars,
    vars_to_control_points,
)
from airfoil_fit import bspline_helper


def fit_bspline(
    proc,
    upper_data: np.ndarray,
    lower_data: np.ndarray,
    num_control_points: int | tuple[int, int],
    *,
    continuity: int,
    smoothing_weight: float,
    upper_te_tangent_vector: np.ndarray | None = None,
    lower_te_tangent_vector: np.ndarray | None = None,
) -> bool:
    """Fit B-splines with G1 and optional G2 constraints at leading edge."""
    try:
        proc.fitted = False
        proc.upper_curve = proc.lower_curve = None
        proc.upper_control_points = proc.lower_control_points = None
        proc.upper_knot_vector = proc.lower_knot_vector = None
        proc.last_error_message = None
        proc.last_optimizer_info = None
        proc.last_insertion_info = None

        if isinstance(num_control_points, tuple):
            num_cp_upper, num_cp_lower = num_control_points
        else:
            num_cp_upper = num_cp_lower = num_control_points

        if not isinstance(continuity, (int, np.integer)) or isinstance(continuity, bool) or continuity not in (1, 2, 3):
            raise ValueError("Continuity must be 1, 2 or 3.")
        if not np.isfinite(smoothing_weight) or smoothing_weight < 0:
            raise ValueError("Smoothing weight must be finite and nonnegative.")
        for count, degree in ((num_cp_upper, proc.degree_upper), (num_cp_lower, proc.degree_lower)):
            if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < max(4, degree + 1):
                raise ValueError("Each control-point count must be an integer >= max(4, degree + 1).")
            if degree < continuity:
                raise ValueError("Spline degree must be at least the requested continuity order.")
        upper_data = np.asarray(upper_data, dtype=float)
        lower_data = np.asarray(lower_data, dtype=float)
        for data in (upper_data, lower_data):
            if data.ndim != 2 or data.shape[1] != 2 or len(data) < 2 or not np.isfinite(data).all():
                raise ValueError("Provide finite Nx2 surface coordinates.")
        proc.num_cp_upper, proc.num_cp_lower = int(num_cp_upper), int(num_cp_lower)
        proc.enforce_g2, proc.enforce_g3 = continuity >= 2, continuity == 3
        proc.smoothing_weight = float(smoothing_weight)

        proc.is_sharp_te = bool(np.linalg.norm(upper_data[-1] - lower_data[-1]) <= 1e-8)

        le_point = (upper_data[0] + lower_data[0]) / 2
        upper_data_corrected = upper_data.copy()
        lower_data_corrected = lower_data.copy()
        upper_data_corrected[0] = le_point
        lower_data_corrected[0] = le_point

        proc.upper_original_data = upper_data_corrected.copy()
        proc.lower_original_data = lower_data_corrected.copy()

        if proc.is_sharp_te:
            te_point = np.array([1.0, 0.0])
            upper_data_corrected[-1] = te_point
            lower_data_corrected[-1] = te_point

        upper_te_dir = bspline_helper.normalize_vector(upper_te_tangent_vector)
        lower_te_dir = bspline_helper.normalize_vector(lower_te_tangent_vector)
        proc.upper_te_dir = None if upper_te_dir is None else np.asarray(upper_te_dir, dtype=float).copy()
        proc.lower_te_dir = None if lower_te_dir is None else np.asarray(lower_te_dir, dtype=float).copy()

        if proc.enforce_g2:
            success = fit_with_g2_optimization(
                proc,
                upper_data_corrected,
                lower_data_corrected,
                (proc.num_cp_upper, proc.num_cp_lower),
                upper_te_dir,
                lower_te_dir,
            )
            if not success:
                return False

        if not proc.enforce_g2:
            success = proc._fit_g1_independent(
                upper_data_corrected,
                lower_data_corrected,
                (proc.num_cp_upper, proc.num_cp_lower),
                upper_te_dir,
                lower_te_dir,
                enable_soft_te_handle_quality=True,
            )

        if not success:
            return False

        proc._finalize_curves()
        proc.fitted_degree = (proc.degree_upper, proc.degree_lower)
        proc.fitted = True
        proc.num_cp_upper = len(proc.upper_control_points)
        proc.num_cp_lower = len(proc.lower_control_points)
        proc._validate_continuity()
        return True

    except Exception as exc:
        try:
            logger = logging.getLogger(__name__)
            logger.error(f"Error in fit_bspline: {exc}")
        except Exception:
            pass
        proc.last_error_message = f"fit_bspline failed: {exc}"
        proc.fitted = False
        return False


def fit_with_g2_optimization(
    proc,
    upper_data: np.ndarray,
    lower_data: np.ndarray,
    num_control_points: int | tuple[int, int],
    upper_te_dir: np.ndarray | None,
    lower_te_dir: np.ndarray | None,
    use_existing_knot_vectors: bool = False,
    warm_start_from_current: bool = False,
    use_insertion_solver_settings: bool = False,
) -> bool:
    """Fit both surfaces with G2 continuity using constrained optimization."""
    try:
        _ = num_control_points
        enable_soft_te_handle_quality = True
        te_point_upper = upper_data[-1]
        te_point_lower = lower_data[-1]

        if not use_existing_knot_vectors:
            proc.upper_knot_vector = bspline_helper.create_knot_vector(proc.num_cp_upper, proc.degree_upper)
            proc.lower_knot_vector = bspline_helper.create_knot_vector(proc.num_cp_lower, proc.degree_lower)

        if proc.upper_knot_vector is None or proc.lower_knot_vector is None:
            raise ValueError("Knot vectors are unexpectedly None when building basis matrices in G2 optimization.")

        num_cp_upper = len(proc.upper_knot_vector) - proc.degree_upper - 1
        num_cp_lower = len(proc.lower_knot_vector) - proc.degree_lower - 1
        proc.num_cp_upper = num_cp_upper
        proc.num_cp_lower = num_cp_lower

        use_warm_start = bool(warm_start_from_current and use_existing_knot_vectors)
        warm_start_available = (
            use_warm_start
            and proc.upper_control_points is not None
            and proc.lower_control_points is not None
            and len(proc.upper_control_points) == num_cp_upper
            and len(proc.lower_control_points) == num_cp_lower
        )
        if not warm_start_available:
            proc._fit_g1_independent(
                upper_data,
                lower_data,
                (num_cp_upper, num_cp_lower),
                upper_te_dir,
                lower_te_dir,
                enable_soft_te_handle_quality=enable_soft_te_handle_quality,
                use_existing_knot_vectors=use_existing_knot_vectors,
            )
        else:
            proc.upper_control_points = np.asarray(proc.upper_control_points, dtype=float).copy()
            proc.lower_control_points = np.asarray(proc.lower_control_points, dtype=float).copy()

        initial_vars = control_points_to_initial_vars(
            proc.upper_control_points,
            proc.lower_control_points,
            num_cp_upper,
            num_cp_lower,
        )
        layout = OptimizationLayout(num_cp_upper, num_cp_lower)
        u_params_upper = bspline_helper.create_parameter_from_x_coords(upper_data, proc.param_exponent_upper)
        u_params_lower = bspline_helper.create_parameter_from_x_coords(lower_data, proc.param_exponent_lower)
        basis_upper = bspline_helper.build_basis_matrix(u_params_upper, proc.upper_knot_vector, proc.degree_upper)
        basis_lower = bspline_helper.build_basis_matrix(u_params_lower, proc.lower_knot_vector, proc.degree_lower)
        problem = build_g2_problem(
            upper_data=upper_data,
            lower_data=lower_data,
            basis_upper=basis_upper,
            basis_lower=basis_lower,
            upper_knot_vector=proc.upper_knot_vector,
            lower_knot_vector=proc.lower_knot_vector,
            degree_upper=proc.degree_upper,
            degree_lower=proc.degree_lower,
            te_point_upper=te_point_upper,
            te_point_lower=te_point_lower,
            upper_te_dir=upper_te_dir,
            lower_te_dir=lower_te_dir,
            enable_soft_te_handle_quality=enable_soft_te_handle_quality,
            smoothing_weight=float(proc.smoothing_weight),
            initial_vars=initial_vars,
            layout=layout,
            vars_to_control_points_fn=lambda x: vars_to_control_points(x, num_cp_upper, num_cp_lower),
            enforce_g3=proc.enforce_g3,
        )

        num_vars = layout.num_vars
        max_deg = max(proc.degree_upper, proc.degree_lower)
        max_iter = max(200, num_vars * 20)
        ftol = 1e-7

        if use_existing_knot_vectors or use_insertion_solver_settings:
            insertion_target = max(
                proc.insertion_solver_min_maxiter,
                int(np.ceil(num_vars * proc.insertion_solver_maxiter_factor)),
            )
            max_iter = max(max_iter, insertion_target)
            ftol = proc.insertion_solver_ftol

        if max_deg > 10:
            max_iter += (max_deg - 10) * 100
        if enable_soft_te_handle_quality:
            max_iter += 80
        if bool(proc.enforce_g3):
            max_iter += 300

        result = optimize.minimize(
            problem["objective"],
            problem["initial_vars"],
            method="SLSQP",
            jac=problem["objective_jac"],
            constraints=problem["constraints"],
            bounds=problem["bounds"],
            options={"ftol": ftol, "maxiter": max_iter, "disp": False},
        )

        max_constraint_violation = 0.0 if getattr(result, "x", None) is not None else float("inf")
        x_final = np.asarray(result.x, dtype=float) if getattr(result, "x", None) is not None else np.asarray(initial_vars, dtype=float)
        if getattr(result, "x", None) is not None:
            for constraint in problem["constraints"]:
                cval = np.asarray(constraint["fun"](x_final), dtype=float).ravel()
                if cval.size:
                    ctype = str(constraint.get("type", "eq")).strip().lower()
                    if ctype == "ineq":
                        violation = float(np.max(np.maximum(0.0, -cval)))
                    else:
                        violation = float(np.max(np.abs(cval)))
                    max_constraint_violation = max(max_constraint_violation, violation) if np.isfinite(violation) else float("inf")

        for value, (lo, hi) in zip(x_final, problem["bounds"]):
            if lo is not None:
                max_constraint_violation = max(max_constraint_violation, lo - value)
            if hi is not None:
                max_constraint_violation = max(max_constraint_violation, value - hi)
        accepted = bool(result.success and np.isfinite(x_final).all()
                        and np.isfinite(getattr(result, "fun", np.nan))
                        and max_constraint_violation <= 2e-5)
        initial_diag = dict(problem["evaluate_diagnostics"](np.asarray(initial_vars, dtype=float)))
        final_diag = dict(problem["evaluate_diagnostics"](x_final))
        proc.last_optimizer_info = {
            "success": bool(result.success),
            "accepted": accepted,
            "accepted_via_relaxed_criteria": False,
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(getattr(result, "nit", -1)),
            "objective": float(getattr(result, "fun", np.nan)),
            "max_constraint_violation": float(max_constraint_violation),
            "solver_ftol": float(ftol),
            "solver_maxiter": int(max_iter),
            "insertion_mode": bool(use_insertion_solver_settings),
            "solver_profile": "insertion" if use_insertion_solver_settings else "default",
            "smoothing_weight": float(proc.smoothing_weight),
            "fit_error_metric": str(problem.get("fit_error_metric", "vertical")),
            "initial_fit_error_total": float(initial_diag["fit_error_total"]),
            "final_fit_error_total": float(final_diag["fit_error_total"]),
            "initial_smoothing_penalty_total": float(initial_diag["smoothing_penalty_total"]),
            "final_smoothing_penalty_total": float(final_diag["smoothing_penalty_total"]),
            "final_smoothing_penalty_upper_fourth_diff": float(final_diag["smoothing_penalty_upper_fourth_diff"]),
            "final_smoothing_penalty_lower_fourth_diff": float(final_diag["smoothing_penalty_lower_fourth_diff"]),
            "raw_smoothing_baseline_total": float(final_diag.get("raw_smoothing_baseline_total", np.nan)),
            "smoothing_reference_fit": float(final_diag.get("smoothing_reference_fit", np.nan)),
            "smoothing_objective_scale": float(final_diag.get("smoothing_objective_scale", np.nan)),
            "te_handle_weight": float(final_diag.get("te_handle_weight", np.nan)),
            "te_handle_min_length": float(final_diag.get("te_handle_min_length", np.nan)),
            "initial_te_handle_penalty_total": float(initial_diag.get("te_handle_penalty_total", np.nan)),
            "final_te_handle_penalty_total": float(final_diag.get("te_handle_penalty_total", np.nan)),
            "final_te_handle_penalty_upper_angle": float(final_diag.get("te_handle_penalty_upper_angle", np.nan)),
            "final_te_handle_penalty_lower_angle": float(final_diag.get("te_handle_penalty_lower_angle", np.nan)),
            "final_te_handle_penalty_upper_short_length": float(final_diag.get("te_handle_penalty_upper_short_length", np.nan)),
            "final_te_handle_penalty_lower_short_length": float(final_diag.get("te_handle_penalty_lower_short_length", np.nan)),
            "final_te_handle_upper_length": float(final_diag.get("te_handle_upper_length", np.nan)),
            "final_te_handle_lower_length": float(final_diag.get("te_handle_lower_length", np.nan)),
            "raw_te_handle_baseline_total": float(final_diag.get("raw_te_handle_baseline_total", np.nan)),
            "te_handle_objective_scale": float(final_diag.get("te_handle_objective_scale", np.nan)),
        }

        if accepted:
            proc.upper_control_points, proc.lower_control_points = vars_to_control_points(result.x, num_cp_upper, num_cp_lower)
            return True

        continuity = 3 if proc.enforce_g3 else 2
        proc.last_error_message = (f"G{continuity} optimization failed (status={result.status}, "
                                   f"constraint violation={max_constraint_violation:.3g}): {result.message}")
        logger = logging.getLogger(__name__)
        logger.error(f"Error in _fit_with_g2_optimization: Optimization failed with status {result.status}")
        return False

    except Exception as exc:
        try:
            logger = logging.getLogger(__name__)
            logger.error(f"Error in _fit_with_g2_optimization during optimization: {exc}")
        except Exception:
            pass
        proc.last_error_message = f"_fit_with_g2_optimization failed: {exc}"
        return False
