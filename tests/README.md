# Core tests

From the repository root, select either source code or an installed/bundled package.
Tests intentionally do not prepend `src` themselves, so wheel validation is real.

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m unittest discover -s tests
```

To exercise AirfoilSpline's generated bundle instead:

```powershell
$env:PYTHONPATH = (Resolve-Path ..\AirfoilSpline\_vendor).Path
python -c "import airfoil_fit; print(airfoil_fit.__file__)"
python -m unittest discover -s tests
```

With coverage installed in the test environment:

```powershell
python -m coverage run --branch --source=airfoil_fit -m unittest discover -s tests
python -m coverage report
```

The 2026-10-01 review measured 73% combined statement/branch coverage across the
whole core, with 22 passing tests. The original symmetric baseline is supplemented
by cambered geometry checks, analytic vertical-error checks, numerical derivative
checks, and failure/refinement tests. Remaining priorities are real-profile data,
spacing/repeated-knot edge cases, export failures and broader invalid-input coverage.
A passing synthetic suite is not evidence that every real profile converges.
