import unittest
import numpy as np
from airfoil_fit import BSplineProcessor


class ExplicitKnotTests(unittest.TestCase):
    def setUp(self):
        x = np.linspace(0, 1, 81)
        y = .12*np.sqrt(x)*(1-x)
        self.upper, self.lower = np.column_stack((x,y)), np.column_stack((x,-y))
        self.upper_knots = np.array([0]*5 + [.3] + [1]*5, dtype=float)
        self.lower_knots = np.array([0]*5 + [1]*5, dtype=float)

    def test_explicit_asymmetric_knots_at_all_continuities(self):
        for continuity in (1,2,3):
            with self.subTest(continuity=continuity):
                model = BSplineProcessor(4)
                result = model.fit_bspline(self.upper,self.lower,(6,5),continuity=continuity,
                    smoothing_weight=0, knot_vectors=(self.upper_knots,self.lower_knots))
                self.assertTrue(result, result.message)
                self.assertEqual(result.achieved_continuity, continuity)
                np.testing.assert_array_equal(model.upper_knot_vector,self.upper_knots)
                np.testing.assert_array_equal(model.lower_knot_vector,self.lower_knots)
                self.assertFalse(np.shares_memory(model.upper_knot_vector,self.upper_knots))
                np.testing.assert_allclose(model.upper_curve(1), self.upper[-1], atol=1e-12)

    def test_invalid_knots_fail_without_exposing_partial_model(self):
        for invalid in (self.upper_knots[:-1], self.upper_knots[::-1],
                        np.array([0]*5+[float('nan')]+[1]*5),
                        np.array([0]*6+[1]*5), np.array([0]*5+[1]*6)):
            with self.subTest(knots=invalid):
                model = BSplineProcessor(4)
                result = model.fit_bspline(self.upper,self.lower,(6,5),continuity=2,
                    smoothing_weight=0,knot_vectors=(invalid,self.lower_knots))
                self.assertFalse(result)
                self.assertFalse(model.is_fitted())
                self.assertIsNone(model.upper_curve)
                self.assertIsNone(model.upper_control_points)
