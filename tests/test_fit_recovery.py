import unittest
from unittest.mock import patch

import numpy as np

from airfoil_splines_core import BSplineProcessor, FitResult, fit_with_recovery


class FitRecoveryTests(unittest.TestCase):
    def setUp(self):
        x = (1 - np.cos(np.linspace(0, np.pi, 61))) / 2
        y = .6 * (.2969*np.sqrt(x)-.126*x-.3516*x*x+.2843*x**3-.1036*x**4)
        self.upper = np.column_stack((x, y))
        self.lower = np.column_stack((x, -y))

    def recover(self, smoothing, continuity, should_fail):
        original = BSplineProcessor.fit_bspline
        calls = []

        def controlled(processor, upper, lower, counts, **kwargs):
            setting = (kwargs['smoothing_weight'], kwargs['continuity'])
            calls.append(setting)
            if should_fail(setting, len(calls)):
                return FitResult(False, setting[1], None, 'forced failure')
            return original(processor, upper, lower, counts, **kwargs)

        with patch.object(BSplineProcessor, 'fit_bspline', autospec=True,
                          side_effect=controlled):
            recovery = fit_with_recovery(
                4, self.upper, self.lower, 8,
                smoothing_weight=smoothing, continuity=continuity)
        return recovery, calls

    def test_retries_zero_smoothing_before_reducing_continuity(self):
        recovery, calls = self.recover(.01, 2, lambda setting, number: number == 1)
        self.assertTrue(recovery)
        self.assertEqual(calls, [(.01, 2), (0.0, 2)])
        self.assertEqual((recovery.effective_smoothing, recovery.effective_continuity), (0.0, 2))
        self.assertTrue(recovery.fallback_used)

    def test_falls_back_to_g1_after_zero_smoothing_failure(self):
        recovery, calls = self.recover(.01, 2, lambda setting, number: setting[1] != 1)
        self.assertTrue(recovery)
        self.assertEqual(calls, [(.01, 2), (0.0, 2), (0.0, 1)])
        self.assertEqual((recovery.effective_smoothing, recovery.effective_continuity), (0.0, 1))

    def test_zero_smoothing_skips_duplicate_attempt(self):
        recovery, calls = self.recover(0.0, 3, lambda setting, number: setting[1] != 1)
        self.assertTrue(recovery)
        self.assertEqual(calls, [(0.0, 3), (0.0, 1)])

    def test_reports_every_failed_attempt(self):
        recovery, calls = self.recover(.01, 2, lambda setting, number: True)
        self.assertFalse(recovery)
        self.assertEqual(calls, [(.01, 2), (0.0, 2), (0.0, 1)])
        self.assertIn('G2 smoothing=0.01', recovery.failure_message)
        self.assertIn('G1 smoothing=0', recovery.failure_message)


if __name__ == '__main__':
    unittest.main()
