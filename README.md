# AirfoilFitCore

A Python library for fitting B-spline curves to airfoil coordinate data.
Only NumPy and SciPy are required. It imports no Fusion, Qt, or plotting UI modules.

## Status

Version 0.1.0 extracts the current Fusion fitting implementation without changing
its numerical algorithms. Fusion logging is replaced by standard Python logging.
The API is provisional until the standalone and mass-fitting interfaces are reconciled.

Intended consumers:

- AirfoilFitter: legacy Fusion add-in, pinned to a validated library release.
- AirfoilSpline: parametric Fusion add-in.
- AirfoilFitter desktop application: UI and exports stay in the application.
- AF-tests: mass-fitting and analytics depend directly on this package, not an app or UI.

## Installation

```sh
python -m pip install .
python -m unittest discover -s tests
```

```python
from airfoil_fit_core import BSplineProcessor, load_airfoil_data

upper, lower, name, blunt_te, te_thickness = load_airfoil_data("profile.dat")
fit = BSplineProcessor(degree=7)
ok = fit.fit_bspline(upper, lower, num_control_points=8,
                     enforce_g2=True, single_span=True)
```

The loader currently returns the Fusion five-value tuple. The standalone loader
uses four values; this must be adapted explicitly during migration. Applications
must pin a tested release and bundle its wheel for offline installation.

See MIGRATION.md for the remaining consumer integration work.
