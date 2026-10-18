"""Shared ordered recovery policy for fresh B-spline fits."""
from __future__ import annotations

from dataclasses import dataclass

from .bspline_processor import BSplineProcessor
from .fit_result import FitResult


@dataclass(frozen=True)
class FitAttempt:
    smoothing_weight: float
    continuity: int
    result: FitResult


@dataclass(frozen=True)
class FitRecovery:
    processor: BSplineProcessor | None
    attempts: tuple[FitAttempt, ...]

    @property
    def success(self) -> bool:
        return self.processor is not None

    def __bool__(self) -> bool:
        return self.success

    @property
    def result(self) -> FitResult:
        return self.attempts[-1].result

    @property
    def effective_smoothing(self) -> float:
        return self.attempts[-1].smoothing_weight

    @property
    def effective_continuity(self) -> int:
        return self.attempts[-1].continuity

    @property
    def fallback_used(self) -> bool:
        if len(self.attempts) > 1:
            return True
        attempt = self.attempts[0]
        return not attempt.result

    @property
    def failure_message(self) -> str:
        details = '; '.join(
            f'G{attempt.continuity} smoothing={attempt.smoothing_weight:g}: '
            f'{attempt.result.message}'
            for attempt in self.attempts)
        return 'Cannot fit the airfoil. Attempts: ' + details


def fit_with_recovery(
    degree,
    upper_data,
    lower_data,
    counts,
    *,
    continuity: int,
    smoothing_weight: float,
    upper_te_tangent_vector=None,
    lower_te_tangent_vector=None,
    knot_vectors=None,
) -> FitRecovery:
    """Fit requested settings, then zero smoothing, then G1 at zero smoothing.

    Invalid inputs and unexpected solver exceptions propagate immediately. Only a
    normal unsuccessful ``FitResult`` advances to the next policy attempt.
    """
    requested_smoothing = float(smoothing_weight)
    requested_continuity = int(continuity)
    settings = [(requested_smoothing, requested_continuity)]
    if requested_smoothing != 0.0:
        settings.append((0.0, requested_continuity))
    if requested_continuity != 1:
        settings.append((0.0, 1))

    attempts = []
    for attempt_smoothing, attempt_continuity in settings:
        candidate = BSplineProcessor(degree)
        result = candidate.fit_bspline(
            upper_data, lower_data, counts,
            continuity=attempt_continuity,
            smoothing_weight=attempt_smoothing,
            upper_te_tangent_vector=upper_te_tangent_vector,
            lower_te_tangent_vector=lower_te_tangent_vector,
            knot_vectors=knot_vectors)
        attempts.append(FitAttempt(attempt_smoothing, attempt_continuity, result))
        if result:
            return FitRecovery(candidate, tuple(attempts))

    return FitRecovery(None, tuple(attempts))
