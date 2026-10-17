# airfoil-splines-core

`airfoil-splines-core` is a Python library for loading two-dimensional airfoil
coordinates and fitting upper and lower non-rational B-spline curves. It uses
NumPy and SciPy.

The library provides coordinate preparation, constrained fitting, knot
refinement, error measurement, and export. Applications supply fitting parameters,
manage accepted models, and handle user interaction or CAD integration. It does
not analyze aerodynamic performance.

## Installation

Python 3.10 or newer is required. Runtime dependencies are `numpy` and `scipy`;
they are installed by pip.

From a source checkout:

```sh
git clone https://github.com/mr33g/airfoil-splines-core.git
cd airfoil-splines-core
python -m pip install .
```

The distribution name is `airfoil-splines-core`; the import name is
`airfoil_splines_core`.

## Load, fit, and evaluate

This example expects a Selig or Lednicer DAT file at `profile.dat`:

```python
import numpy as np
from airfoil_splines_core import (
    BSplineProcessor,
    load_airfoil_data,
    export_bspline_to_bsp,
)

data = load_airfoil_data("profile.dat", repanel_input=False)
model = BSplineProcessor(degree=4)
result = model.fit_bspline(
    data.upper,
    data.lower,
    num_control_points=6,
    continuity=2,
    smoothing_weight=0.0,
)
if not result:
    raise RuntimeError(result.message)

u = np.linspace(0.0, 1.0, 201)
upper_points = model.upper_curve(u)  # shape (201, 2)
lower_points = model.lower_curve(u)
upper_derivatives = model.upper_curve.derivative(1)(u)

if not export_bspline_to_bsp(model, data.name, "profile.bsp"):
    raise RuntimeError("BSP export failed")
```

`upper_curve` and `lower_curve` are `scipy.interpolate.BSpline` objects. Their
parameter domain is `[0, 1]`, from leading edge to trailing edge. The parameter
`u` is not the chord coordinate `x/c`. Evaluate the curves to obtain coordinates;
do not interpret parameter values as chord positions.

## Coordinate loading and conventions

`load_airfoil_data(path, ...)` reads a file. `parse_airfoil_data(text, ...)`
performs the same preparation on an in-memory string. Both return an `AirfoilData`
named tuple:

| Field | Meaning |
| --- | --- |
| `upper`, `lower` | Floating-point coordinate arrays, shape `(N, 2)`, ordered leading edge to trailing edge |
| `name` | Airfoil name from the input |
| `blunt_te` | Whether the prepared input has a finite trailing-edge gap |
| `te_thickness` | Trailing-edge gap divided by chord length |

The parser accepts Selig contour ordering and Lednicer surface blocks. It locates
the leading edge using a temporary cubic contour spline, translates and rotates
the coordinates, and scales them to unit chord. Both surfaces start at `(0, 0)`
and end at `x = 1`. A sharp trailing edge ends at `(1, 0)`; a blunt trailing edge
has upper and lower endpoint ordinates symmetric about zero.

Returned coordinates are dimensionless. For a physical chord length `c`, multiply
both coordinate components by `c`, then apply the rotation and translation needed
by the calling application. The loading result does not return an inverse
transform to the original file coordinate system.

Input repaneling is enabled by default, with 51 points per surface. Specify
`repanel_input=False` to avoid resampling the surfaces. Leading-edge detection and
normalization still occur. With `repanel_input=True`, `points_per_surface`,
`le_bunch`, and `te_bunch` control resampling; their defaults are 51, 0.85, and 0.30.

```python
from airfoil_splines_core import parse_airfoil_data

text = open("profile.dat", encoding="utf-8").read()
data = parse_airfoil_data(
    text,
    repanel_input=True,
    points_per_surface=101,
    logger_func=lambda message: None,
)
```

Loading functions can raise file I/O or parsing exceptions. `logger_func` accepts
a callable taking a message string; its default is `print`.

### Prepared source model

`AirfoilProcessor` is an optional stateful wrapper. It stores fitting data,
non-repaneled normalized reference data, and estimated trailing-edge tangents.

```python
from airfoil_splines_core import AirfoilProcessor

source = AirfoilProcessor(logger_func=lambda message: None)
if not source.load_text(text):
    raise ValueError("Could not prepare airfoil data")
reference_upper, reference_lower = source.error_reference_data()
```

Use `source.upper_data` and `source.lower_data` for fitting. The reference arrays
are useful for measuring error against the input rather than the repaneled
samples. Pass `source.upper_te_tangent_vector` and
`source.lower_te_tangent_vector` to `fit_bspline` when using those tangent estimates.
`load_text` returns a Boolean and reports caught errors through the logger.
`load_airfoil_data_and_initialize_model(path)` is its file wrapper; file-reading
errors occur before `load_text` and can propagate.

## Fitting parameters and constraints

`BSplineProcessor(degree)` accepts a positive integer for both surfaces or an
`(upper_degree, lower_degree)` tuple. `fit_bspline` requires explicit control-point
counts, continuity order, and smoothing weight.

| Argument | Contract |
| --- | --- |
| `upper_data`, `lower_data` | Finite `(N, 2)` arrays with at least two points per surface |
| `num_control_points` | Integer for both surfaces or `(upper_count, lower_count)`; each count must be at least `max(4, degree + 1)` |
| `continuity` | `1`, `2`, or `3` for G1, G2, or G3 continuity at the leading edge; each degree must be at least this order |
| `smoothing_weight` | Finite nonnegative weight for the control-point smoothing penalty; zero disables this penalty |
| `upper_te_tangent_vector`, `lower_te_tangent_vector` | Optional trailing-edge direction vectors |
| `knot_vectors` | Optional `(upper_knots, lower_knots)` tuple for a fresh fit with specified knots |

Direct array input bypasses the loader. Prepare it in the same normalized,
leading-edge-to-trailing-edge convention before fitting. The solver assumes this
coordinate frame; passing arrays does not normalize arbitrary geometry.

G1 constrains tangent continuity, G2 additionally constrains curvature, and G3
additionally constrains the derivative of curvature with respect to arc length.
These are geometric continuity conditions, not equality of the two curves'
parameter derivatives. Both stored curves run away from the leading edge toward
the trailing edge. The solver preserves trailing-edge endpoint geometry inferred
from the supplied surfaces and does not substitute a weaker continuity request.

The fitting residual is vertical separation at matching chord coordinates.
It is not shortest Euclidean distance to the curve. Smoothing and internal handle
quality penalties can trade residual error for curve quality. Parameters and
point counts need to be selected for the input data; convergence is not guaranteed.

### Results and failure handling

`fit_bspline` returns a `FitResult`. Boolean conversion returns `success`.

| Field | Meaning |
| --- | --- |
| `success` | Whether the requested fit succeeded |
| `requested_continuity` | Requested order |
| `achieved_continuity` | Requested order on success; `None` on failure |
| `message` | Outcome or error description |
| `diagnostics` | Snapshot of available optimizer information |

Diagnostics depend on the solver path; callers should not assume every key is
present. `model.is_fitted()` checks whether fitted curves are available.

A failed fresh fit clears fitted curves, control points, and knots on that model.
To preserve a previously accepted fit, solve on a new `BSplineProcessor` and
replace the accepted model only after success. Invalid constructor degrees raise
`ValueError`; errors caught inside fitting are represented by an unsuccessful
`FitResult`.

### Explicit knot vectors

For degree `p` and `n` control points, the knot vector must have `n + p + 1`
entries. It must be finite, nondecreasing, and clamped to `[0, 1]`: the first
`p + 1` entries are zero and the last `p + 1` entries are one. Interior knots must
lie strictly inside this interval, with multiplicity no greater than `p`.
Counts and knot vectors must agree. Supplied vectors are copied.

This supports refitting an already refined layout with different smoothing or
source data. Pass the accepted model's knot vectors and counts to a new processor
with the same degrees, then check the new `FitResult` before replacing the model.

## Refinement

After fitting, the processor provides these operations:

| Method | Operation |
| --- | --- |
| `insert_knot_at_max_error("upper" or "lower")` | Select an insertion location from the error distribution, insert a knot, and refit |
| `refine_curve_with_knots([u, ...], surface=...)` | Insert specified knots on `"upper"`, `"lower"`, or both (`None`), then refit |
| `refine_curves_with_surface_knots(upper_knots=..., lower_knots=...)` | Insert separate knot lists and refit |

Refinement retains the requested degrees and continuity constraints. Adding a
knot to one surface increases that surface's control-point count; the joint refit
may also change the other surface's geometry. These methods return Booleans.
Failed refinement restores the previous valid model. Inspect `last_error_message`
and `last_optimizer_info` for failure details. An invalid surface selector raises
`ValueError`.

Knot insertion alone preserves curve shape, but these refinement methods include
a solve that can change it. For representation-only insertion, use the lower-level
`bspline_helper.insert_knot` function and its returned control points and knots;
it does not update a processor model.

## Error measurement and trailing-edge thickness

```python
from airfoil_splines_core.optimization.fit_metrics import vertical_error_metrics

metrics = vertical_error_metrics(model.upper_curve, data.upper)
print(metrics["rms"], metrics["max_error"])
```

The returned dictionary includes `signed_residuals`, `abs_residuals`, `sum_sq`,
`rms`, `max_error`, `max_error_idx`, and `u_at_max_error`. Measure against the
non-repaneled reference arrays when reporting error relative to the input.
RMS and maximum error are chord-normalized distances; multiply by physical chord
length to express them in the same physical units.

`bspline_helper.apply_te_thickness_to_reference(upper, lower, thickness)` returns
copies adjusted to a target normalized trailing-edge gap. It uses a smooth
chordwise displacement and leaves the leading edge fixed. Validate that the target
is finite and nonnegative in the calling application, then fit the adjusted arrays.
For a gap of 0.2 mm on a 200 mm chord, pass
`target_te_thickness=0.2 / 200`.

## Export and integration

`export_bspline_to_bsp(model, name, path, logger_func=print)` writes a UTF-8 JSON
file and returns a Boolean. `load_bspline_from_bsp(path)` validates that format
and returns a `BSPModelData` instance. The top-level keys are `name`, `upper`, and `lower`.
Each surface contains `px`, `py`, `knots`, and `degree`; `px` and `py` are the
control-point coordinate lists. Coordinates remain normalized. This preserves
the spline representation rather than exporting sampled points.

`data_loader.export_airfoil_to_selig_format(upper, lower, name, path)` writes DAT
coordinates in Selig contour order. To export fitted geometry, pass sampled curve
points. This writer propagates file I/O errors and has no success Boolean.

For CAD adapters, use `upper_control_points`, `lower_control_points`,
`upper_knot_vector`, `lower_knot_vector`, `degree_upper`, and `degree_lower`.
The host application is responsible for coordinate transforms and conversion to
its spline API. DXF export and application-specific geometry creation are outside
this package.

`calculate_curvature_comb_data(...)` returns numerical visualization data without
rendering it. Lower-level helpers and the `operations` and `optimization` modules
contain implementation details; underscore-prefixed methods are internal.

## Development and tests

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
python -m pip wheel --no-deps . --wheel-dir dist
```

The test suite covers parsing, normalized geometry, continuity, fitting failures,
explicit knots, refinement, and numerical gradients. See [tests/README.md](tests/README.md)
for numerical baseline diagnostics. Numerical defaults live in
`src/airfoil_splines_core/config.py`; changing module-level settings affects other
users of the package in the same process.

## License

MIT. See [LICENSE](LICENSE).
