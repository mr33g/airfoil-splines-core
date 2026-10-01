import ast
import json
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from airfoil_fit import BSplineProcessor

class CoreTests(unittest.TestCase):
    def test_no_application_imports(self):
        forbidden = {"adsk", "PySide6", "PyQt5", "PyQt6", "gui", "ui", "matplotlib", "pyqtgraph", "core", "utils"}
        for path in (ROOT / "src").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                for name in names:
                    self.assertNotIn(name.split(".")[0], forbidden, str(path))

    def test_fusion_baseline_continuity_modes(self):
        expected = json.loads((ROOT / "tests" / "fusion_baseline.json").read_text())
        x = (1 - np.cos(np.linspace(0, np.pi, 61))) / 2
        y = 5 * .12 * (.2969*np.sqrt(x)-.126*x-.3516*x*x+.2843*x**3-.1036*x**4)
        for mode in expected:
            with self.subTest(mode=mode):
                proc = BSplineProcessor(degree=7)
                self.assertTrue(proc.fit_bspline(np.column_stack([x,y]), np.column_stack([x,-y]), 8,
                    continuity=int(mode[-1]), smoothing_weight=.001), proc.last_error_message)
                np.testing.assert_allclose(proc.upper_control_points, expected[mode]["upper"], atol=1e-8, rtol=1e-7)
                np.testing.assert_allclose(proc.lower_control_points, expected[mode]["lower"], atol=1e-8, rtol=1e-7)

    def surfaces(self, blunt=False):
        x = (1 - np.cos(np.linspace(0, np.pi, 61))) / 2
        y = .6 * (.2969*np.sqrt(x)-.126*x-.3516*x*x+.2843*x**3-(.1015 if blunt else .1036)*x**4)
        return np.column_stack([x, y]), np.column_stack([x, -y])

    def test_input_trailing_edge_is_preserved(self):
        for blunt in (False, True):
            upper, lower = self.surfaces(blunt)
            proc = BSplineProcessor(degree=7)
            self.assertTrue(proc.fit_bspline(upper, lower, 8, continuity=1, smoothing_weight=.001))
            self.assertEqual(proc.is_sharp_te, not blunt)
            np.testing.assert_allclose(proc.upper_control_points[-1], upper[-1], atol=1e-8)
            np.testing.assert_allclose(proc.lower_control_points[-1], lower[-1], atol=1e-8)

    def test_requested_thickness_adjusts_surfaces_without_mutating_source(self):
        from airfoil_fit.bspline_helper import apply_te_thickness_to_reference
        for blunt in (False, True):
            upper, lower = self.surfaces(blunt)
            original_upper, original_lower = upper.copy(), lower.copy()
            for thickness in (0.02, 0.0):
                adjusted_upper, adjusted_lower = apply_te_thickness_to_reference(
                    upper, lower, thickness)
                np.testing.assert_array_equal(upper, original_upper)
                np.testing.assert_array_equal(lower, original_lower)
                np.testing.assert_array_equal(adjusted_upper[0], upper[0])
                np.testing.assert_array_equal(adjusted_lower[0], lower[0])
                proc = BSplineProcessor(degree=7)
                self.assertTrue(proc.fit_bspline(adjusted_upper, adjusted_lower, 8, continuity=1, smoothing_weight=.001))
                gap = proc.upper_control_points[-1, 1] - proc.lower_control_points[-1, 1]
                self.assertAlmostEqual(gap, thickness, places=8)
                self.assertEqual(proc.is_sharp_te, thickness == 0)

    def test_fixed_degree_insertion_retains_knots_and_shape_error(self):
        from airfoil_fit.optimization.fit_metrics import pure_surface_fit_error
        upper, lower = self.surfaces()
        proc = BSplineProcessor(degree=3)
        self.assertTrue(proc.fit_bspline(upper, lower, 4, continuity=1, smoothing_weight=.001))
        proc.min_cp_neighbor_distance = 0
        before = sum(pure_surface_fit_error(d, c, k, 3) for d, c, k in (
            (upper, proc.upper_control_points, proc.upper_knot_vector),
            (lower, proc.lower_control_points, proc.lower_knot_vector)))
        self.assertTrue(proc.refine_curve_with_knots([.4], surface='upper'), proc.last_error_message)
        self.assertEqual(proc.degree_upper, 3)
        self.assertEqual(proc.num_cp_upper, 5)
        self.assertIn(.4, proc.upper_knot_vector)
        self.assertTrue(proc.refine_curve_with_knots([.7], surface='upper'), proc.last_error_message)
        self.assertIn(.4, proc.upper_knot_vector)
        self.assertIn(.7, proc.upper_knot_vector)
        after = sum(pure_surface_fit_error(d, c, k, 3) for d, c, k in (
            (upper, proc.upper_control_points, proc.upper_knot_vector),
            (lower, proc.lower_control_points, proc.lower_knot_vector)))
        self.assertLessEqual(after, before + 1e-10)

    def test_knot_insertion_does_not_raise_degree_of_initial_single_span(self):
        upper, lower = self.surfaces()
        proc = BSplineProcessor(degree=7)
        self.assertTrue(proc.fit_bspline(upper, lower, 8, continuity=1, smoothing_weight=.001))
        proc.min_cp_neighbor_distance = 0
        self.assertTrue(proc.refine_curve_with_knots([.4], surface='upper'), proc.last_error_message)
        self.assertEqual(proc.degree_upper, 7)
        self.assertEqual(proc.num_cp_upper, 9)
        np.testing.assert_array_equal(np.unique(proc.upper_knot_vector), [0, .4, 1])
        self.assertTrue(np.isfinite(proc.upper_curve(np.linspace(0, 1, 50))).all())

    def test_loader_and_export_are_headless(self):
        from airfoil_fit import load_airfoil_data, AirfoilData, export_bspline_to_bsp
        upper, lower = self.surfaces(True)
        contour = np.vstack([upper[::-1], lower[1:]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'foil.dat'
            path.write_text('NACA0012\n' + '\n'.join(f'{x:.12f} {y:.12f}' for x, y in contour))
            data = load_airfoil_data(path, logger_func=lambda _: None, points_per_surface=65)
            self.assertIsInstance(data, AirfoilData)
            self.assertEqual(len(data.upper), 65)
            self.assertTrue(data.blunt_te)
            proc = BSplineProcessor(degree=7)
            self.assertTrue(proc.fit_bspline(data.upper, data.lower, 8, continuity=1, smoothing_weight=.001))
            output = Path(directory) / 'foil.bsp'
            self.assertTrue(export_bspline_to_bsp(proc, data.name, str(output)))
            payload = json.loads(output.read_text())
            np.testing.assert_allclose(payload['upper']['px'], proc.upper_control_points[:, 0])
            self.assertEqual(payload['upper']['degree'], 7)

    def test_vertical_gradient_matches_finite_difference(self):
        from airfoil_fit.optimization.fit_metrics import vertical_distance_and_grad
        upper, lower = self.surfaces()
        proc = BSplineProcessor(degree=7)
        self.assertTrue(proc.fit_bspline(upper, lower, 8, continuity=1, smoothing_weight=.001))
        cp = proc.upper_control_points.copy()
        knots = proc.upper_knot_vector
        _, gradient, _, _ = vertical_distance_and_grad(upper, cp, knots, 7)
        for axis in (0, 1):
            plus, minus = cp.copy(), cp.copy()
            plus[4, axis] += 1e-6
            minus[4, axis] -= 1e-6
            a = vertical_distance_and_grad(upper, plus, knots, 7)[0]
            b = vertical_distance_and_grad(upper, minus, knots, 7)[0]
            self.assertAlmostEqual(gradient[4, axis], (a-b)/2e-6, delta=1e-6)

    def test_explicit_asymmetric_degrees_and_invalid_request(self):
        upper, lower = self.surfaces()
        proc = BSplineProcessor(degree=(5, 7))
        result = proc.fit_bspline(upper, lower, (6, 8), continuity=2, smoothing_weight=.001)
        self.assertTrue(result, result.message)
        self.assertEqual(result.achieved_continuity, 2)
        self.assertEqual(proc.fitted_degree, (5, 7))
        result = proc.fit_bspline(upper, lower, (5, 8), continuity=2, smoothing_weight=.001)
        self.assertFalse(result)
        self.assertIsNone(result.achieved_continuity)
        self.assertFalse(proc.is_fitted())
        self.assertIsNone(proc.upper_control_points)

    def test_g1_optimizer_failure_is_not_reported_as_success(self):
        from scipy.optimize import OptimizeResult
        upper, lower = self.surfaces()
        def failed(_objective, x, **kwargs):
            return OptimizeResult(x=x, success=False, status=9, message='Iteration limit', fun=1., nit=1)
        proc = BSplineProcessor(degree=7)
        with patch('airfoil_fit.operations.g1_ops.optimize.minimize', side_effect=failed):
            result = proc.fit_bspline(upper, lower, 8, continuity=1, smoothing_weight=.001)
        self.assertFalse(result)
        self.assertFalse(proc.is_fitted())
        self.assertIn('Iteration limit', result.message)
        self.assertTrue(all(not d['success'] for d in result.diagnostics['surfaces']))
        self.assertIsNone(proc.upper_control_points)

    def test_failed_joint_fit_does_not_fall_back(self):
        upper, lower = self.surfaces()
        for continuity in (2, 3):
            proc = BSplineProcessor(degree=7)
            with patch('airfoil_fit.operations.fit_ops.fit_with_g2_optimization', return_value=False), \
                 patch.object(proc, '_fit_g1_independent') as fallback:
                result = proc.fit_bspline(upper, lower, 8, continuity=continuity, smoothing_weight=.001)
            self.assertFalse(result)
            self.assertEqual(result.requested_continuity, continuity)
            self.assertIsNone(result.achieved_continuity)
            fallback.assert_not_called()
            self.assertFalse(proc.is_fitted())

    def test_iteration_limit_is_not_silently_accepted(self):
        from scipy import optimize
        minimize = optimize.minimize
        upper, lower = self.surfaces()
        def limited(*args, **kwargs):
            result = minimize(*args, **kwargs)
            result.success = False
            result.status = 9
            result.message = 'Iteration limit'
            return result
        proc = BSplineProcessor(degree=7)
        with patch('airfoil_fit.operations.fit_ops.optimize.minimize', side_effect=limited):
            result = proc.fit_bspline(upper, lower, 8, continuity=2, smoothing_weight=.001)
        self.assertFalse(result)
        self.assertEqual(result.diagnostics['status'], 9)
        self.assertFalse(result.diagnostics['accepted'])
        self.assertIsNone(proc.upper_curve)

    def test_failed_refinement_preserves_existing_geometry(self):
        upper, lower = self.surfaces()
        proc = BSplineProcessor(degree=7)
        self.assertTrue(proc.fit_bspline(upper, lower, 8, continuity=2, smoothing_weight=.001))
        before = proc.upper_control_points.copy(), proc.upper_knot_vector.copy()
        proc.min_cp_neighbor_distance = 0
        with patch.object(proc, '_fit_with_g2_optimization', return_value=False), \
             patch.object(proc, '_fit_g1_independent') as fallback:
            self.assertFalse(proc.refine_curve_with_knots([.4], surface='upper'))
            fallback.assert_not_called()
        np.testing.assert_array_equal(proc.upper_control_points, before[0])
        np.testing.assert_array_equal(proc.upper_knot_vector, before[1])
        self.assertEqual(proc.num_cp_upper, 8)
        self.assertTrue(proc.is_fitted())

    def test_text_and_file_loading_match_for_both_formats(self):
        from airfoil_fit import parse_airfoil_data, load_airfoil_data, AirfoilProcessor
        upper, lower = self.surfaces(True)
        for lednicer in (False, True):
            points = np.vstack([upper, lower]) if lednicer else np.vstack([upper[::-1], lower[1:]])
            text = 'Test foil\n' + (f'{len(upper)} {len(lower)}\n' if lednicer else '')
            text += '\n'.join(f'{x:.12f} {y:.12f}' for x, y in points)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'foil.dat'
                path.write_text(text)
                file_data = load_airfoil_data(path, logger_func=lambda _: None)
            with patch('builtins.open', side_effect=AssertionError('No file I/O during parsing')):
                text_data = parse_airfoil_data(text, logger_func=lambda _: None)
                processor = AirfoilProcessor(logger_func=lambda _: None)
                self.assertTrue(processor.load_text(text))
            np.testing.assert_array_equal(file_data.upper, text_data.upper)
            np.testing.assert_array_equal(file_data.lower, text_data.lower)
            np.testing.assert_array_equal(processor.upper_data, text_data.upper)

if __name__ == "__main__":
    unittest.main()
