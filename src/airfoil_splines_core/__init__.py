"""Headless airfoil fitting, independent of Fusion and desktop UI frameworks."""
from .bspline_processor import BSplineProcessor
from .bsp_importer import BSPModelData, load_bspline_from_bsp
from .data_loader import AirfoilData, load_airfoil_data, parse_airfoil_data

__version__ = "0.3.0"
__all__ = [
    "AirfoilData",
    "AirfoilProcessor",
    "BSPModelData",
    "BSplineProcessor",
    "FitResult",
    "export_bspline_to_bsp",
    "load_airfoil_data",
    "load_bspline_from_bsp",
    "parse_airfoil_data",
]

from airfoil_splines_core.airfoil_processor import AirfoilProcessor
from airfoil_splines_core.bsp_exporter import export_bspline_to_bsp

from .fit_result import FitResult
