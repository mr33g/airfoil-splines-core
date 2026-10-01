"""Report cross-platform differences against the original Fusion fit."""
import json
from pathlib import Path
import platform
import numpy as np
import scipy
from scipy.interpolate import BSpline
from unittest.mock import patch
from airfoil_fit import BSplineProcessor
from airfoil_fit.operations import fit_ops
from airfoil_fit.optimization import control_points_to_initial_vars
from airfoil_fit.optimization.fit_metrics import solve_u_for_x_targets


def report():
    expected = json.loads((Path(__file__).parent / 'fusion_baseline.json').read_text())
    x = (1 - np.cos(np.linspace(0, np.pi, 61))) / 2
    y = .6 * (.2969*np.sqrt(x)-.126*x-.3516*x*x+.2843*x**3-.1036*x**4)
    print(json.dumps(dict(platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__)))
    for mode in (2, 3):
        problems = []
        original = fit_ops.build_g2_problem
        def capture(**kwargs):
            problem = original(**kwargs)
            problems.append(problem)
            return problem
        proc = BSplineProcessor(degree=7)
        with patch.object(fit_ops, 'build_g2_problem', side_effect=capture):
            result = proc.fit_bspline(np.column_stack([x,y]), np.column_stack([x,-y]), 8,
                                      continuity=mode, smoothing_weight=.001)
        if not result:
            print(result)
            continue
        baseline = expected[f'g{mode}']
        variables = control_points_to_initial_vars(np.array(baseline['upper']), np.array(baseline['lower']), 8, 8)
        info = dict(proc.last_optimizer_info)
        info['baseline_objective'] = float(problems[-1]['objective'](variables))
        info['mode'] = mode
        for side in ('upper', 'lower'):
            old = BSpline(getattr(proc, side+'_knot_vector'), np.array(baseline[side]), 7)
            curve = getattr(proc, side+'_curve')
            u = np.linspace(0, 1, 1001)
            info[side+'_max_curve_distance'] = float(np.max(np.linalg.norm(curve(u)-old(u),axis=1)))
            targets = (1 - np.cos(np.linspace(0, np.pi, 1001))) / 2
            new_u = solve_u_for_x_targets(targets, curve.c, curve.t, curve.k)
            old_u = solve_u_for_x_targets(targets, old.c, old.t, old.k)
            info[side+'_max_vertical_shape_difference'] = float(np.max(np.abs(curve(new_u)[:,1]-old(old_u)[:,1])))
            info[side+'_max_control_point_difference'] = float(np.max(np.abs(curve.c-old.c)))
        print(json.dumps(info, sort_keys=True))

if __name__ == '__main__':
    report()
