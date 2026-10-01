"""Outcome of a requested fit, independent of application fallback policy."""
from dataclasses import dataclass, field

@dataclass(frozen=True)
class FitResult:
    success: bool
    requested_continuity: int
    achieved_continuity: int | None
    message: str
    diagnostics: dict = field(default_factory=dict)

    def __bool__(self):
        return self.success
