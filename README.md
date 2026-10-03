# airfoil-splines-core

Headless B-spline fitting for airfoil coordinate data, using NumPy and SciPy.
The shared solver uses vertical fitting error with G1, G2 or G3 leading-edge
continuity. Callers specify degrees and control-point counts explicitly for each
surface; knot insertion always preserves degree. The fitter infers trailing-edge
geometry from its input surfaces. `apply_te_thickness_to_reference` in
`airfoil_splines_core.bspline_helper` adjusts those surfaces to a requested thickness.

```sh
python -m pip install .
python -m unittest discover -s tests
```

```python
from airfoil_splines_core import BSplineProcessor, load_airfoil_data, export_bspline_to_bsp

data = load_airfoil_data("profile.dat")
fit = BSplineProcessor(degree=7)
if fit.fit_bspline(data.upper, data.lower, num_control_points=8,
                   continuity=2, smoothing_weight=0.001):
    export_bspline_to_bsp(fit, data.name, "profile.bsp")
```

`load_airfoil_data` returns named fields `upper`, `lower`, `name`, `blunt_te`,
and `te_thickness`. Repaneling can be controlled per call. `AirfoilProcessor`
also supplies original normalized reference points and trailing-edge tangents.
Pass continuity and smoothing explicitly to each fit; degree is required when
constructing the fitter and may be an `(upper, lower)` pair. Application defaults
and single-span restrictions belong to consumers. `fit_bspline` returns a
`FitResult` with success, requested/achieved continuity, message and diagnostics.
A failed fit does not silently relax continuity or expose an initial estimate as
fitted geometry. Failed refinement leaves the previous valid geometry intact.

`parse_airfoil_data(text)` parses embedded DAT contents without file I/O;
`AirfoilProcessor.load_text(text)` also prepares reference points and tangents.
Already prepared coordinate arrays can be passed directly to `fit_bspline`.
Vertical residual metrics are in `airfoil_splines_core.optimization.fit_metrics`.
