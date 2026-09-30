from __future__ import annotations

import numpy as np
from scipy import interpolate
import logging

from airfoil_fit_core import config
from airfoil_fit_core.operations import (
    apply_knot_insertions,
    cp_spacing_violates_minimum,
    finalize_curves,
    fit_bspline as fit_bspline_op,
    fit_g1_independent,
    fit_single_surface_g1,
    fit_with_g2_optimization as fit_with_g2_optimization_op,
    insert_knot_with_spacing_fallback,
    largest_span_midpoint_knot,
    refine_curve_with_knots as refine_curve_with_knots_op,
    refine_curves_with_surface_knots as refine_curves_with_surface_knots_op,
    refit_after_knot_insertion,
    validate_continuity,
)
from airfoil_fit_core.optimization import (
    vars_to_control_points,
)
from airfoil_fit_core import bspline_helper

class BSplineProcessor:
    """
    B-spline processor with true G2 constraint that maintains G1 continuity.
    Works with arbitrary degree B-splines.
    """

    def __init__(self, degree: int = config.DEFAULT_CP_COUNT-1):
        self.upper_control_points: np.ndarray | None = None
        self.lower_control_points: np.ndarray | None = None
        self.upper_knot_vector: np.ndarray | None = None
        self.lower_knot_vector: np.ndarray | None = None
        self.upper_curve: interpolate.BSpline | None = None
        self.lower_curve: interpolate.BSpline | None = None
        self.degree: int = int(degree)
        self.degree_upper: int = int(degree)
        self.degree_lower: int = int(degree)
        self.fitted_degree: int | None = None  # Stores the actual degree used during fitting (may differ from self.degree if single span mode was used)
        self.num_cp_upper: int = 10  # Current number of control points for upper surface
        self.num_cp_lower: int = 10  # Current number of control points for lower surface
        self.param_exponent_upper: float = 0.5 # Mapping x -> u (0.5 is sqrt)
        self.param_exponent_lower: float = 0.5
        self.fitted: bool = False
        self.is_sharp_te: bool = False
        self.enforce_g2: bool = True
        self.enforce_g3: bool = False
        self.g2_weight: float = 100.0  # Weight for G2 constraint in optimization
        self.smoothing_weight: float = config.DEFAULT_SMOOTHNESS_PENALTY  # Weight for control point smoothing penalty
        self.min_cp_neighbor_distance: float = float(getattr(config, "MIN_CP_NEIGHBOR_DISTANCE", 0.0))
        # Tight insertion-mode SLSQP settings from tuned jacobian implementation.
        self.insertion_solver_ftol: float = 1e-10
        self.insertion_solver_maxiter_factor: float = 50.0
        self.insertion_solver_min_maxiter: int = 480
        self.last_error_message: str | None = None
        self.last_optimizer_info: dict | None = None
        self.last_insertion_info: dict | None = None
        self.upper_te_dir: np.ndarray | None = None
        self.lower_te_dir: np.ndarray | None = None
        
        self.upper_original_data: np.ndarray | None = None
        self.lower_original_data: np.ndarray | None = None


    def fit_bspline(
        self,
        upper_data: np.ndarray,
        lower_data: np.ndarray,
        num_control_points: int | tuple[int, int],
        is_thickened: bool = False,
        upper_te_tangent_vector: np.ndarray | None = None,
        lower_te_tangent_vector: np.ndarray | None = None,
        enforce_g2: bool = False,
        enforce_g3: bool = False,
        single_span: bool = False,
    ) -> bool:
        return fit_bspline_op(
            self,
            upper_data,
            lower_data,
            num_control_points,
            is_thickened=is_thickened,
            upper_te_tangent_vector=upper_te_tangent_vector,
            lower_te_tangent_vector=lower_te_tangent_vector,
            enforce_g2=enforce_g2,
            enforce_g3=enforce_g3,
            single_span=single_span,
        )

    def _fit_with_g2_optimization(
        self,
        upper_data: np.ndarray,
        lower_data: np.ndarray,
        num_control_points: int | tuple[int, int],
        upper_te_dir: np.ndarray | None,
        lower_te_dir: np.ndarray | None,
        use_existing_knot_vectors: bool = False,
        warm_start_from_current: bool = False,
        use_insertion_solver_settings: bool = False,
    ) -> bool:
        return fit_with_g2_optimization_op(
            self,
            upper_data,
            lower_data,
            num_control_points,
            upper_te_dir,
            lower_te_dir,
            use_existing_knot_vectors=use_existing_knot_vectors,
            warm_start_from_current=warm_start_from_current,
            use_insertion_solver_settings=use_insertion_solver_settings,
        )

    def _vars_to_control_points(self, vars: np.ndarray, num_cp_upper: int, num_cp_lower: int) -> tuple[np.ndarray, np.ndarray]:
        """Convert optimization variables to control points maintaining G1 constraints."""
        return vars_to_control_points(vars, num_cp_upper, num_cp_lower)

    def _fit_g1_independent(
        self,
        upper_data: np.ndarray,
        lower_data: np.ndarray,
        num_control_points: int | tuple[int, int],
        upper_te_dir: np.ndarray | None,
        lower_te_dir: np.ndarray | None,
        enable_soft_te_handle_quality: bool = True,
        use_existing_knot_vectors: bool = False,
    ):
        return fit_g1_independent(
            self,
            upper_data,
            lower_data,
            num_control_points,
            upper_te_dir,
            lower_te_dir,
            enable_soft_te_handle_quality=enable_soft_te_handle_quality,
            use_existing_knot_vectors=use_existing_knot_vectors,
        )

    def _fit_single_surface_g1(
        self,
        basis_matrix: np.ndarray,
        surface_data: np.ndarray,
        num_control_points: int,
        is_upper: bool,
        soft_te_tangent_vector: np.ndarray | None = None,
        te_point: np.ndarray | None = None
    ) -> np.ndarray:
        return fit_single_surface_g1(
            self,
            basis_matrix,
            surface_data,
            num_control_points,
            is_upper,
            soft_te_tangent_vector=soft_te_tangent_vector,
            te_point=te_point,
        )

    def _finalize_curves(self):
        return finalize_curves(self)

    def _validate_continuity(self):
        return validate_continuity(self)

    def is_fitted(self) -> bool:
        """Check if B-splines have been successfully fitted."""
        return self.fitted and self.upper_curve is not None and self.lower_curve is not None

    def calculate_curvature_comb_data(self, num_points_per_segment=200, scale_factor=0.050):
        """Calculate curvature comb visualization data."""
        if not self.is_fitted():
            return None
        
        return bspline_helper.calculate_curvature_comb_data(
            self.upper_curve, self.lower_curve, num_points_per_segment, scale_factor
        )

    def insert_knot_at_max_error(self, surface: str, single_span: bool = False) -> bool:
        """
        Insert a knot at the location of maximum deviation on the specified surface.
        
        Args:
            surface: Which surface to insert knot into ('upper' or 'lower')
            single_span: If True, update degree to maintain single span mode (degree = CP - 1)
            
        Returns:
            bool: True if knot was inserted successfully, False otherwise
        """
        if not self.fitted or self.upper_curve is None or self.lower_curve is None:
            return False
        
        if self.upper_original_data is None or self.lower_original_data is None:
            return False
        
        try:
            # Determine target surface
            target_curve = self.upper_curve if surface == 'upper' else self.lower_curve
            target_data = self.upper_original_data if surface == 'upper' else self.lower_original_data
            target_knot_vector = self.upper_knot_vector if surface == 'upper' else self.lower_knot_vector
            current_param_exponent = self.param_exponent_upper if surface == 'upper' else self.param_exponent_lower
            
            # Use max-error-midspan policy for knot location.
            new_knot, adjusted_param_exponent = bspline_helper.prepare_knot_insertion_with_parameter_adjustment(
                target_curve,
                target_data,
                target_knot_vector,
                current_param_exponent,
                adjust_parameter_exponent=False,
            )
            
            # Update parameter exponent
            if surface == 'upper':
                self.param_exponent_upper = adjusted_param_exponent
            else:
                self.param_exponent_lower = adjusted_param_exponent
            
            # Insert the knot using the existing refine method
            # refine_curve_with_knots will automatically detect and maintain single span mode
            success = self.refine_curve_with_knots([new_knot], surface=surface)
            
            return success
            
        except Exception as e:
            try:
                logger = logging.getLogger(__name__)
                logger.error(f"Error in insert_knot_at_max_error: {str(e)}")
            except:
                pass
            return False

    def _apply_knot_insertions(self, new_knots: list[float], surface: str | None = None) -> bool:
        return apply_knot_insertions(self, new_knots, surface=surface)

    def _refit_after_knot_insertion(self, *, use_existing_knot_vectors: bool) -> bool:
        return refit_after_knot_insertion(self, use_existing_knot_vectors=use_existing_knot_vectors)

    def _insert_knot_with_spacing_fallback(
        self,
        control_points: np.ndarray,
        knot_vector: np.ndarray,
        degree: int,
        requested_knot: float,
        surface_name: str,
    ) -> tuple[np.ndarray, np.ndarray, float, bool] | None:
        return insert_knot_with_spacing_fallback(
            self,
            control_points,
            knot_vector,
            degree,
            requested_knot,
            surface_name,
        )

    def _largest_span_midpoint_knot(
        self,
        knot_vector: np.ndarray,
        degree: int,
        *,
        exclude: float | None = None,
    ) -> float | None:
        return largest_span_midpoint_knot(self, knot_vector, degree, exclude=exclude)

    def _cp_spacing_violates_minimum(self, control_points: np.ndarray) -> bool:
        return cp_spacing_violates_minimum(self, control_points)

    def refine_curve_with_knots(self, new_knots: list[float], surface: str | None = None) -> bool:
        """
        Refine the B-spline curves by inserting new knots.
        This operation preserves the curve's shape while increasing the number of control points.

        Args:
            new_knots: A list of parameter values (knots) to insert.
            surface: Which surface(s) to insert knots into. 
                     None (default) = both surfaces, 'upper' = upper only, 'lower' = lower only.

        Returns:
            bool: True if knots were inserted successfully, False otherwise.
        """
        return refine_curve_with_knots_op(self, new_knots, surface=surface)

    def refine_curves_with_surface_knots(
        self,
        *,
        upper_knots: list[float] | None = None,
        lower_knots: list[float] | None = None,
    ) -> bool:
        return refine_curves_with_surface_knots_op(self, upper_knots=upper_knots, lower_knots=lower_knots)
