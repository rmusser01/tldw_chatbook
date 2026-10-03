"""The first-run wizard's busy line: what a slow Next is doing (TASK-34100.1).

Review finding cross-cutting-14 (2026-10-02) measured Nexts of 2.5-4 s,
3-6 s when choosing the Full track, and up to 30 s while Voice waits for its
save. The only feedback was nav buttons going disabled. The line shows only
once a Next has run for about 400 ms, so the common sub-second Next does not
flicker. It names the work ("Saving voice settings…"), and after 2 s it adds
the elapsed seconds. It lives in the wizard chrome above the pinned error
strip, and the container starts and stops it from ``_set_advancing``.
"""

from __future__ import annotations

import time
from typing import Any

from textual.timer import Timer
from textual.widgets import Static

#: A Next that settles before this many seconds shows no line at all.
BUSY_REVEAL_SECONDS = 0.4
#: From this many seconds on, the line also counts the elapsed seconds.
BUSY_ELAPSED_AFTER_SECONDS = 2.0
#: How often the shown line is refreshed (it only repaints when it changes).
_TICK_SECONDS = 0.1


def busy_label_for(step: Any) -> str:
    """Name the work a Next on ``step`` waits for.

    A step can say what its own commit does by defining ``busy_label()``.
    Otherwise, and whenever that raises or returns nothing, the step's title
    is used.

    Args:
        step: The setup step being committed.

    Returns:
        A short present-tense line ending in an ellipsis.
    """
    own = getattr(step, "busy_label", None)
    if callable(own):
        try:
            label = own()
        except Exception:  # noqa: BLE001 - a label must never break a Next.
            label = ""
        if isinstance(label, str) and label.strip():
            return label
    config = getattr(step, "config", None)
    title = getattr(config, "title", "") or "this step's"
    return f"Saving {title} settings…"


class SetupBusyStatus(Static):
    """One muted line that names a slow Next's work, with elapsed seconds."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("", markup=False, **kwargs)
        self._label = ""
        self._started_at: float | None = None
        self._timer: Timer | None = None
        self._shown = ""
        self.display = False

    def start(self, label: str) -> None:
        """Begin timing a Next; the line appears only if it runs long.

        Args:
            label: The work the Next waits for, from ``busy_label_for``.
        """
        self.stop()
        self._label = label
        self._started_at = time.monotonic()
        self._timer = self.set_interval(_TICK_SECONDS, self._tick)

    def stop(self) -> None:
        """Hide the line; the Next settled (or never started)."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._started_at = None
        self._show("")

    def _tick(self) -> None:
        if self._started_at is None:
            return
        elapsed = time.monotonic() - self._started_at
        if elapsed < BUSY_REVEAL_SECONDS:
            return
        if elapsed < BUSY_ELAPSED_AFTER_SECONDS:
            self._show(self._label)
        else:
            self._show(f"{self._label} {int(elapsed)} s")

    def _show(self, text: str) -> None:
        if text == self._shown:
            return
        self._shown = text
        self.update(text)
        # Styles-level too: hosts without the app stylesheet have no
        # ``.hidden`` rule, and a line that only pretends to hide would
        # shift the nav bar.
        self.set_class(not text, "hidden")
        self.display = bool(text)
