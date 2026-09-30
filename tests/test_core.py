import ast
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from airfoil_fit_core import BSplineProcessor

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
                    enforce_g2=mode != "g1", enforce_g3=mode == "g3", single_span=True), proc.last_error_message)
                np.testing.assert_allclose(proc.upper_control_points, expected[mode]["upper"], atol=1e-8, rtol=1e-7)
                np.testing.assert_allclose(proc.lower_control_points, expected[mode]["lower"], atol=1e-8, rtol=1e-7)

if __name__ == "__main__":
    unittest.main()
