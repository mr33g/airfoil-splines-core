"""Read and validate B-spline control points and knots from BSP files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BSPModelData:
    """Validated, normalized spline representation stored in a BSP file."""

    airfoil_name: str
    upper_control_points: np.ndarray
    lower_control_points: np.ndarray
    upper_knots: np.ndarray
    lower_knots: np.ndarray
    upper_degree: int
    lower_degree: int


def _parse_surface_json(payload: object, *, surface_name: str) -> tuple[np.ndarray, np.ndarray, int]:
    if not isinstance(payload, dict):
        raise ValueError(f"Surface '{surface_name}' must be a JSON object.")

    px = payload.get("px")
    py = payload.get("py")
    knots = payload.get("knots")
    degree = payload.get("degree")
    if not isinstance(px, list) or not isinstance(py, list) or not isinstance(knots, list):
        raise ValueError(f"Surface '{surface_name}' must contain list fields 'px', 'py', and 'knots'.")
    if len(px) != len(py):
        raise ValueError(f"Surface '{surface_name}' has mismatched 'px'/'py' lengths.")
    if not px:
        raise ValueError(f"Surface '{surface_name}' must contain at least one control point.")
    if isinstance(degree, bool) or not isinstance(degree, int):
        raise ValueError(f"Surface '{surface_name}' degree must be an integer.")

    try:
        control_points = np.column_stack((np.asarray(px, dtype=float), np.asarray(py, dtype=float)))
        knot_vector = np.asarray(knots, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Surface '{surface_name}' contains invalid numeric values.") from exc

    degree_value = int(degree)
    if degree_value < 1:
        raise ValueError(f"Surface '{surface_name}' has invalid degree {degree_value}.")
    if not np.isfinite(control_points).all() or not np.isfinite(knot_vector).all():
        raise ValueError(f"Surface '{surface_name}' contains nonfinite values.")
    if knot_vector.ndim != 1 or np.any(np.diff(knot_vector) < 0):
        raise ValueError(f"Surface '{surface_name}' contains unsorted knots.")
    if len(control_points) < degree_value + 1:
        raise ValueError(f"Surface '{surface_name}' has too few control points.")

    expected_knot_count = len(control_points) + degree_value + 1
    if len(knot_vector) != expected_knot_count:
        raise ValueError(
            f"Surface '{surface_name}' has {len(knot_vector)} knots, expected {expected_knot_count} "
            f"for {len(control_points)} control points and degree {degree_value}."
        )
    if (
        not np.all(knot_vector[: degree_value + 1] == 0.0)
        or not np.all(knot_vector[-degree_value - 1 :] == 1.0)
    ):
        raise ValueError(f"Surface '{surface_name}' must have clamped unit-domain knots.")
    interior_knots = knot_vector[degree_value + 1 : -degree_value - 1]
    if interior_knots.size:
        _, multiplicities = np.unique(interior_knots, return_counts=True)
        if np.any(multiplicities > degree_value):
            raise ValueError(f"Surface '{surface_name}' has excessive interior-knot multiplicity.")

    return control_points, knot_vector, degree_value


def _load_bspline_from_json(path: Path, payload: object) -> BSPModelData:
    if not isinstance(payload, dict):
        raise ValueError("BSP JSON root must be an object.")

    upper_cp, upper_knots, upper_degree = _parse_surface_json(payload.get("upper"), surface_name="upper")
    lower_cp, lower_knots, lower_degree = _parse_surface_json(payload.get("lower"), surface_name="lower")
    return BSPModelData(
        airfoil_name=str(payload.get("name") or path.stem),
        upper_control_points=upper_cp,
        lower_control_points=lower_cp,
        upper_knots=upper_knots,
        lower_knots=lower_knots,
        upper_degree=upper_degree,
        lower_degree=lower_degree,
    )


def load_bspline_from_bsp(file_path: str | Path) -> BSPModelData:
    """Read a JSON BSP file and return a validated spline representation."""

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"BSP file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid BSP JSON in '{path}': {exc.msg}.") from exc
    return _load_bspline_from_json(path, payload)
