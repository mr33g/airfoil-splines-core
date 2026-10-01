from pathlib import Path
import numpy as np

from airfoil_fit import config
from airfoil_fit.data_loader import parse_airfoil_data


class AirfoilProcessor:
    """
    Load normalized airfoil surfaces, reference data, and endpoint tangents.
    This model has no application or UI dependencies.
    """

    def __init__(self, logger_func=print):
        # Core airfoil data
        self.upper_data = None
        self.lower_data = None
        self.upper_display_reference_data = None
        self.lower_display_reference_data = None
        self.upper_te_tangent_vector = None
        self.lower_te_tangent_vector = None
        self._is_blunt_TE = False  # True if original airfoil has thickened TE
        self._te_thickness = 0.0   # TE thickness normalized to chord length
        self.logger_func = logger_func
        self.airfoil_name = ""


    def load_airfoil_data_and_initialize_model(self, file_path):
        """File convenience wrapper around the in-memory loader."""
        return self.load_text(Path(file_path).read_text())

    def load_text(self, text):
        """
        Loads airfoil data and initializes the model.
        Resets internal flags and state.
        """
        self.upper_data = None
        self.lower_data = None
        self.upper_display_reference_data = None
        self.lower_display_reference_data = None
        self.upper_te_tangent_vector = None
        self.lower_te_tangent_vector = None
        self._is_blunt_TE = False
        self._te_thickness = 0.0

        try:
            upper, lower, airfoil_name, blunt_te, te_thickness = parse_airfoil_data(text, logger_func=self.logger_func)
            self.upper_data = upper
            self.lower_data = lower
            self._load_display_reference_data(text, upper, lower)
            self.airfoil_name = airfoil_name
            self._is_blunt_TE = blunt_te
            self._te_thickness = te_thickness
            # Recalculate TE tangent vectors using configured default
            te_vector_points = config.DEFAULT_TE_VECTOR_POINTS
            self.upper_te_tangent_vector, self.lower_te_tangent_vector = self._calculate_te_tangent(
                self.upper_data, self.lower_data, te_vector_points)
            self.logger_func("Airfoil data loaded.")
            return True
        except Exception as e:
            self.logger_func(f"Failed to load or initialize airfoil data: {e}")
            return False

    def _load_display_reference_data(self, text, fallback_upper, fallback_lower) -> None:
        """Keep normalized non-repaneled input for display and final error reporting."""
        try:
            upper_ref, lower_ref, _name, _blunt_te, _te_thickness = parse_airfoil_data(
                text,
                logger_func=lambda _msg: None,
                repanel_input=False,
            )
            self.upper_display_reference_data = upper_ref
            self.lower_display_reference_data = lower_ref
        except Exception as exc:
            self.logger_func(f"Warning: Could not load non-repaneled display reference: {exc}")
            self.upper_display_reference_data = fallback_upper.copy()
            self.lower_display_reference_data = fallback_lower.copy()

    def error_reference_data(self):
        """Return the display/error reference, falling back to fit data if needed."""
        if (
            self.upper_display_reference_data is not None
            and self.lower_display_reference_data is not None
            and len(self.upper_display_reference_data) > 0
            and len(self.lower_display_reference_data) > 0
        ):
            return self.upper_display_reference_data, self.lower_display_reference_data
        return self.upper_data, self.lower_data

    def is_trailing_edge_thickened(self):
        """Returns True if the loaded airfoil has a thickened trailing edge."""
        return self._is_blunt_TE

    def get_te_thickness(self):
        """Returns the trailing edge thickness normalized to chord length."""
        return self._te_thickness

    def _calculate_te_tangent(self, upper_data, lower_data, te_vector_points):
        """
        Calculate trailing edge tangent vectors for upper and lower surfaces using the last N points.
        Returns (upper_te_tangent_vector, lower_te_tangent_vector)
        """
        def tangent(data, n):
            # Use the last n points to estimate the tangent at the trailing edge
            if n < 2 or len(data) < n:
                n = min(3, len(data))
            pts = data[-n:]
            dx = pts[-1, 0] - pts[0, 0]
            dy = pts[-1, 1] - pts[0, 1]
            norm = np.hypot(dx, dy)
            if norm == 0:
                return np.array([1.0, 0.0])
            return np.array([dx, dy]) / norm
            
        upper_te_tangent = tangent(upper_data, te_vector_points)
        lower_te_tangent = tangent(lower_data, te_vector_points)
        return upper_te_tangent, lower_te_tangent

