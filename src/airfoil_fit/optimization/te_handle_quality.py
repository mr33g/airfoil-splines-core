from __future__ import annotations

import numpy as np


def te_handle_quality_penalty_and_grad(
    control_points: np.ndarray,
    target_direction: np.ndarray | None,
    *,
    min_length: float,
    short_length_weight: float,
    eps: float = 1e-12,
) -> tuple[float, np.ndarray, dict[str, float]]:
    """
    Softly prefer the TE handle vector to align with the measured TE tangent.

    The handle is the vector from the last free control point to the fixed TE
    point. The angle term is scale-invariant; the short-length term prevents
    the last free point from collapsing onto the TE.
    """
    cp = np.asarray(control_points, dtype=float)
    grad = np.zeros_like(cp, dtype=float)
    if target_direction is None or cp.shape[0] < 2:
        return 0.0, grad, {"angle": 0.0, "short_length": 0.0, "total": 0.0, "length": 0.0}

    t = np.asarray(target_direction, dtype=float)
    t_norm = float(np.linalg.norm(t))
    if t_norm <= eps:
        return 0.0, grad, {"angle": 0.0, "short_length": 0.0, "total": 0.0, "length": 0.0}
    t = t / t_norm

    v = cp[-1] - cp[-2]
    vv = float(np.dot(v, v))
    length = float(np.sqrt(max(vv, 0.0)))
    if vv <= eps:
        short = float(min_length * min_length * short_length_weight)
        return short, grad, {"angle": 0.0, "short_length": short, "total": short, "length": length}

    cross = float(v[0] * t[1] - v[1] * t[0])
    angle = float((cross * cross) / vv)
    dc_dv = np.asarray([t[1], -t[0]], dtype=float)
    grad_angle_v = (2.0 * cross / vv) * dc_dv - (2.0 * cross * cross / (vv * vv)) * v

    short = 0.0
    grad_short_v = np.zeros(2, dtype=float)
    if length < min_length:
        gap = float(min_length - length)
        short = float(short_length_weight * gap * gap)
        grad_short_v = -2.0 * short_length_weight * gap * v / max(length, eps)

    grad_v = grad_angle_v + grad_short_v
    grad[-1] += grad_v
    grad[-2] -= grad_v

    total = float(angle + short)
    return total, grad, {"angle": angle, "short_length": short, "total": total, "length": length}
