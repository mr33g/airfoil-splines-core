# Extraction and consumer migration

Source: mr33g/AirfoilFitterFusionAddin, CustomCommand at fc5da871879b1b7357babda85d5e51d7de912c20,
with the current AirfoilSpline branding changes. Original copyright and MIT license retained.

## Completed

- Extracted Fusion numerical core, data loader, and spline helpers into the
  unique `airfoil_fit_core` namespace.
- Replaced Fusion logging dependencies with Python logging.
- Preserved fitting defaults and algorithms from that baseline.

## Before switching consumers

1. Compare standalone MSR/vertical error objectives, arbitrary-degree fitting,
   knot preservation, TE thickening, and additional metrics with this baseline.
2. Retain existing application-specific defaults through explicit settings.
3. Adapt four-value versus five-value loader contracts deliberately.
4. Validate representative datasets and insertion/continuity results against
   each application's own baseline; do not treat a basic smoke test as parity.
5. Move Qt `AirfoilModel` responsibilities to the desktop application. Keep
   drawing, timeline, preview, and geometry creation in the Fusion add-ins.
6. Replace AF-tests' `AirfoilFitter` submodule imports with this library and a
   headless export module. Its optional results viewer remains outside the library.
7. Pin and bundle a tested wheel in each application, then remove its copied core.

The existing consumer repositories have not yet been switched. Keeping their
current numerical implementation until parity is verified is intentional.
