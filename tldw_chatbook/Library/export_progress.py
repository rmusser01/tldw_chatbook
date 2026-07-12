"""Pure helpers for rendering chatbook-export progress in the Library canvas.

Kept dependency-free (no Textual) so it is trivially unit-testable; the screen
supplies ``time.monotonic()`` as ``now`` so the throttle has no hidden clock.
"""
from __future__ import annotations

from typing import Optional

EXPORT_PHASE_LABELS: dict[str, str] = {
    "conversations": "Collecting conversations",
    "notes": "Collecting notes",
    "characters": "Collecting characters",
    "media": "Collecting media",
    "prompts": "Collecting prompts",
    "relationships": "Resolving links",
    "packaging": "Packaging archive",
}


def format_export_progress_line(phase: str, current: int, total: int) -> str:
    label = EXPORT_PHASE_LABELS.get(phase, "Exporting")
    unit = " files" if phase == "packaging" else ""
    return f"{label}…  {current}/{total}{unit}"


class ExportProgressThrottle:
    """Decides whether a progress tick should be pushed to the UI thread.

    Emits when the phase changes, when the phase's final item is reached
    (``current >= total``), or when ``min_interval`` seconds have elapsed since
    the last emit — so a fast inner loop never floods the UI, yet the line
    never freezes mid-count.
    """

    def __init__(self, min_interval: float = 0.1) -> None:
        self._min_interval = min_interval
        self._last_phase: Optional[str] = None
        self._last_emit: Optional[float] = None

    def should_emit(self, phase: str, current: int, total: int, now: float) -> bool:
        if (
            phase != self._last_phase
            or current >= total
            or self._last_emit is None
            or (now - self._last_emit) >= self._min_interval
        ):
            self._last_phase = phase
            self._last_emit = now
            return True
        return False
