import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from airfoil_splines_core import export_bspline_to_bsp, load_bspline_from_bsp


class BSPIOTests(unittest.TestCase):
    def model(self):
        return SimpleNamespace(
            upper_control_points=np.array([[0, 0], [0, .1], [.5, .08], [1, 0]], dtype=float),
            lower_control_points=np.array([[0, 0], [0, -.1], [.5, -.08], [1, 0]], dtype=float),
            upper_knot_vector=np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float),
            lower_knot_vector=np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float),
            degree_upper=3,
            degree_lower=3,
        )

    def test_export_import_round_trip(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roundtrip.bsp"
            self.assertTrue(export_bspline_to_bsp(model, "Round trip", path))
            loaded = load_bspline_from_bsp(path)
        self.assertEqual(loaded.airfoil_name, "Round trip")
        self.assertEqual((loaded.upper_degree, loaded.lower_degree), (3, 3))
        np.testing.assert_array_equal(loaded.upper_control_points, model.upper_control_points)
        np.testing.assert_array_equal(loaded.lower_knots, model.lower_knot_vector)

    def test_rejects_invalid_surfaces(self):
        valid_surface = dict(px=[0, 0, .5, 1], py=[0, .1, .1, 0], knots=[0, 0, 0, 0, 1, 1, 1, 1], degree=3)
        invalid_values = (
            dict(px=[0, 0, float("nan"), 1]),
            dict(degree=3.5),
            dict(knots=[0, 0, 0, 0, 1, .5, 1, 1]),
            dict(knots=[0, 0, 0, 0, .5, 1, 1, 1]),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.bsp"
            for invalid in invalid_values:
                with self.subTest(invalid=invalid):
                    payload = {"upper": valid_surface | invalid, "lower": valid_surface}
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_bspline_from_bsp(path)

    def test_missing_and_malformed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.bsp"
            with self.assertRaises(FileNotFoundError):
                load_bspline_from_bsp(path)
            path.write_text("not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_bspline_from_bsp(path)


if __name__ == "__main__":
    unittest.main()
