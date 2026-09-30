"""Central project configuration constants.

This module gathers default numeric parameters and tunable hyper-parameters
used across the airfoil processing library so they live in one place.
Import these values instead of hard-coding magic numbers inside
algorithms or UI widgets.
"""
from __future__ import annotations

# B-spline settings
DEFAULT_SMOOTHNESS_PENALTY: float = 0.001  # Weight for control point smoothing penalty (higher = smoother, lower = more accurate)
DEFAULT_CP_COUNT: int = 10
MIN_CP_NEIGHBOR_DISTANCE: float = 1.0e-3
FIT_ERROR_OBJECTIVE: str = "vertical"

# ---- Sampling & Debugging -----------------------------------------------
NUM_POINTS_CURVE_ERROR: int = 35000

# Input repaneling samples normalized fit data from the temporary cubic spline
# used to locate the true leading edge. The non-repaneled normalized input is
# kept for display and final error reporting.
ENABLE_INPUT_REPANELING: bool = True
INPUT_REPANEL_POINTS_PER_SURFACE: int = 51
INPUT_REPANEL_LE_BUNCH: float = 0.85
INPUT_REPANEL_TE_BUNCH: float = 0.30

# Number of points used for trailing edge vector calculations
# Higher numbers provide more robust tangent estimates but may be less sensitive to local geometry
DEFAULT_TE_VECTOR_POINTS: int = 2

# Soft TE handle quality penalty. It gently prefers the last free control point
# to form a reasonable handle direction and length with the trailing edge.
ENABLE_SOFT_TE_HANDLE_QUALITY: bool = True
DEFAULT_TE_HANDLE_QUALITY_WEIGHT: float = 0.05
TE_HANDLE_MIN_LENGTH: float = 0.040
TE_HANDLE_SHORT_LENGTH_WEIGHT: float = 0.25

# Debug and logging settings
DEBUG_WORKER_LOGGING: bool = False  # Enable detailed debug logging for worker/processing operations
