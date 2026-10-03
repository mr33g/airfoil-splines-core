"""Headless airfoil fitting, independent of Fusion and desktop UI frameworks."""
from .bspline_processor import BSplineProcessor
from .data_loader import AirfoilData, load_airfoil_data, parse_airfoil_data

__version__ = "0.2.0"
__all__ = ["BSplineProcessor", "AirfoilProcessor", "AirfoilData", "load_airfoil_data", "export_bspline_to_bsp", "parse_airfoil_data", "FitResult"]

from airfoil_splines_core.airfoil_processor import AirfoilProcessor
from airfoil_splines_core.bsp_exporter import export_bspline_to_bsp

from .fit_result import FitResult
