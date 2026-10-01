"""Independent numerical checks, beyond matching a saved optimizer result."""
import unittest
import numpy as np
from scipy.interpolate import BSpline
from airfoil_fit import BSplineProcessor, parse_airfoil_data
from airfoil_fit.bspline_helper import create_knot_vector, insert_knot
from airfoil_fit.optimization.fit_metrics import solve_u_for_x_targets, vertical_distance_and_grad
from airfoil_fit.optimization.te_handle_quality import te_handle_quality_penalty_and_grad
from airfoil_fit.optimization.continuity_metrics import (
    start_derivative_weights, curvature_value_and_cp_grad, curvature_derivative_value_and_cp_grad)


def numerical_gradient(function, points, step=1e-6):
    gradient = np.zeros_like(points)
    for index in np.ndindex(points.shape):
        plus, minus = points.copy(), points.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (function(plus) - function(minus)) / (2*step)
    return gradient


class NumericalContracts(unittest.TestCase):
    def test_vertical_distance_matches_closed_form_and_bracket_solver(self):
        # Quadratic Bezier x=u^2, y=0.4*u*(1-u); this has zero dx/du at the LE.
        cp = np.array([[0., 0.], [0., .2], [1., 0.]])
        knots = create_knot_vector(3, 2)
        x = np.array([0., .0001, .04, .25, .64, 1.])
        u = solve_u_for_x_targets(x, cp, knots, 2, initial_u=np.zeros(len(x)), newton_iters=1)
        np.testing.assert_allclose(u, np.sqrt(x), atol=1e-8)
        data = np.column_stack([x, .4*np.sqrt(x)*(1-np.sqrt(x)) + .003])
        error, _, _, _ = vertical_distance_and_grad(data, cp, knots, 2)
        self.assertAlmostEqual(error, len(x)*.003**2, delta=1e-10)

    def test_vertical_gradient_away_from_optimum(self):
        cp = np.array([[0., 0.], [.12, .12], [.65, .07], [1., .01]])
        knots = create_knot_vector(4, 3)
        curve = BSpline(knots, cp, 3)
        data = curve(np.linspace(.08, .92, 19))
        data[:, 1] += .01*np.cos(np.linspace(0, 2, len(data)))
        error, gradient, _, _ = vertical_distance_and_grad(data, cp, knots, 3)
        self.assertGreater(error, 1e-4)
        expected = numerical_gradient(lambda points: vertical_distance_and_grad(data, points, knots, 3)[0], cp)
        np.testing.assert_allclose(gradient, expected, rtol=2e-4, atol=2e-7)

    def test_te_penalty_gradient_for_short_and_long_handles(self):
        for handle in (.01, .1):
            cp = np.array([[0., 0.], [.4, .12], [1-handle, .003], [1., 0.]])
            def evaluate(points):
                return te_handle_quality_penalty_and_grad(points, np.array([1., -.2]),
                    min_length=.04, short_length_weight=.25)
            value, gradient, parts = evaluate(cp)
            self.assertGreater(value, 0)
            self.assertEqual(parts['short_length'] > 0, handle < .04)
            np.testing.assert_allclose(gradient, numerical_gradient(lambda points: evaluate(points)[0], cp),
                                       rtol=2e-5, atol=1e-6)
            np.testing.assert_allclose(gradient.sum(axis=0), 0, atol=1e-12)

    def test_te_penalty_degenerate_inputs_remain_finite(self):
        cp = np.array([[0., 0.], [1., 0.], [1., 0.]])
        for target in (None, np.zeros(2), np.array([1., 0.])):
            value, gradient, _ = te_handle_quality_penalty_and_grad(cp, target, min_length=.04, short_length_weight=.25)
            self.assertTrue(np.isfinite(value))
            self.assertTrue(np.isfinite(gradient).all())
            self.assertGreaterEqual(value, 0)

    def test_le_derivatives_and_gradients_against_scipy(self):
        cp = np.array([[0., 0.], [0., .08], [.12, .16], [.35, .07], [.75, .025], [1., 0.]])
        for degree in (3, 5):
            knots = create_knot_vector(len(cp), degree)
            curve = BSpline(knots, cp, degree)
            weights = start_derivative_weights(len(cp), knots, degree, 3)
            for order, weight in enumerate(weights, 1):
                np.testing.assert_allclose(weight @ cp, curve.derivative(order)(0), atol=1e-10)
            for evaluator, count in ((curvature_value_and_cp_grad, 2), (curvature_derivative_value_and_cp_grad, 3)):
                value, gradient = evaluator(cp, knots, degree, weights[:count])
                expected = numerical_gradient(lambda points: evaluator(points, knots, degree, weights[:count])[0], cp)
                self.assertTrue(np.isfinite(value))
                np.testing.assert_allclose(gradient, expected, rtol=2e-5, atol=1e-4)

    def test_knot_insertion_preserves_geometry_and_derivatives(self):
        cp = np.array([[0., 0.], [0., .08], [.12, .16], [.35, .07], [.75, .025], [1., 0.]])
        knots = create_knot_vector(len(cp), 3)
        curve = BSpline(knots, cp, 3)
        original = cp.copy()
        for knot in (.17, .51, .82):
            cp, knots = insert_knot(cp, knots, 3, knot)
        refined = BSpline(knots, cp, 3)
        for order in (0, 1, 2):
            np.testing.assert_allclose(refined(np.linspace(0, 1, 151), nu=order),
                                       curve(np.linspace(0, 1, 151), nu=order), atol=1e-10)
        np.testing.assert_array_equal(curve.c, original)

    def test_cambered_fit_satisfies_geometry_not_just_success_flag(self):
        x = (1-np.cos(np.linspace(0, np.pi, 81)))/2
        thickness = .6*(.2969*np.sqrt(x)-.126*x-.3516*x*x+.2843*x**3-.1015*x**4)
        camber = .06*x*(1-x)
        upper, lower = np.column_stack([x, camber+thickness]), np.column_stack([x, camber-thickness])
        for continuity in (1, 2, 3):
            proc = BSplineProcessor(degree=(7, 8))
            result = proc.fit_bspline(upper, lower, (8, 9), continuity=continuity, smoothing_weight=.001,
                upper_te_tangent_vector=upper[-1]-upper[-2], lower_te_tangent_vector=lower[-1]-lower[-2])
            self.assertTrue(result, result.message)
            for curve, data in ((proc.upper_curve, upper), (proc.lower_curve, lower)):
                np.testing.assert_allclose(curve([0, 1]), data[[0, -1]], atol=1e-7)
                self.assertAlmostEqual(curve.derivative()(0)[0], 0, delta=1e-9)
                self.assertGreaterEqual(np.min(curve.derivative()(np.linspace(0, 1, 101))[:, 0]), -1e-9)
            if continuity >= 2:
                def k(curve):
                    a,b=curve.derivative(1)(0),curve.derivative(2)(0)
                    return abs(a[0]*b[1]-a[1]*b[0])/np.linalg.norm(a)**3
                self.assertAlmostEqual(k(proc.upper_curve), k(proc.lower_curve), delta=2e-5)
            if continuity == 3:
                def dk(curve):
                    a,b,c=[curve.derivative(n)(0) for n in (1,2,3)]
                    cross=lambda x,y:x[0]*y[1]-x[1]*y[0]
                    return (cross(a,c)*np.dot(a,a)-3*cross(a,b)*np.dot(a,b))/np.linalg.norm(a)**5
                self.assertAlmostEqual(dk(proc.upper_curve), dk(proc.lower_curve), delta=2e-5)

    def test_empty_and_malformed_dat_are_rejected(self):
        for text in ('', 'name only', 'foil\nnot coordinates', 'foil\n0 0\n1 0'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_airfoil_data(text, logger_func=lambda _: None)
