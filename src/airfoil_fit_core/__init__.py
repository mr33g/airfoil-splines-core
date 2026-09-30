"""Headless airfoil fitting, independent of Fusion and desktop UI frameworks."""
from .bspline_processor import BSplineProcessor
from .data_loader import load_airfoil_data

__version__ = "0.1.0"
__all__ = ["BSplineProcessor", "load_airfoil_data"]
