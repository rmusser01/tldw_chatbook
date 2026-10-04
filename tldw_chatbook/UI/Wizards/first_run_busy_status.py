"""The first-run wizard's busy line: what a slow Next is doing (TASK-34100.1).

Review finding cross-cutting-14 (2026-10-02): Nexts of 2.5-4 s, up to 30 s on
Voice, had no cue but disabled buttons. The line shows once a Next has run for
about 400 ms (no flicker on a quick one), names the work and adds the elapsed
seconds after 2 s. The container runs it from ``_set_advancing``.
"""

from __future__ import annotations

import asyncio
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
    """Name the work a Next on ``step`` waits for: its own ``busy_label()``
    when that gives one, else the step's title.

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
    if getattr(config, "id", "") == "summary":  # its exits finish setup
        return "Finishing setup…"
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
        """Start timing a Next; the line shows only if it runs past 400 ms.

        Any earlier timing is stopped first, so a new Next restarts the clock.

        Args:
            label: The work the Next waits for, as ``busy_label_for`` names
                it (a short present-tense line ending in an ellipsis). It is
                shown as given, with the elapsed seconds appended from 2 s.
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

    async def reveal_before_step_change(self) -> None:
        """Show a nearly due line and let it paint before the step change,
        whose synchronous mount would starve the tick (review round 2)."""
        if self._started_at is not None and not self._shown:
            self._tick(early=_TICK_SECONDS)
            if self._shown:  # the screen repaints on idle, a frame later
                await asyncio.sleep(_TICK_SECONDS / 2)

    def _tick(self, early: float = 0.0) -> None:
        if self._started_at is None:
            return
        elapsed = time.monotonic() - self._started_at
        if elapsed < BUSY_REVEAL_SECONDS - early:
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
        # Styles-level too: a host without the app stylesheet has no
        # ``.hidden`` rule, and a line only pretending to hide shifts the nav.
        self.set_class(not text, "hidden")
        self.display = bool(text)
