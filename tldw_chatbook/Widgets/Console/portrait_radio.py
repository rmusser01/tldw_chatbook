"""Fixed-row radio telemetry; the existing portrait/Console clocks drive updates."""

from __future__ import annotations

import time

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Static

from ...Audio.speech_presentation import SpeechPresentationSnapshot
from ...Chat.portrait_performance_state import (
    portrait_radio_level,
    portrait_radio_status,
)


class PortraitRadio(Vertical):
    """Present actual owned voice state without a timer or an audio controller."""

    BUNDLED_CSS = """
    PortraitRadio { height: $ds-height-auto; width: $ds-width-full; min-width: $ds-size-0; }
    PortraitRadio Static { height: $ds-control-height-compact; width: $ds-width-full; text-wrap: nowrap; text-overflow: ellipsis; }
    #console-character-voice-status { color: $ds-text-muted; }
    #console-character-voice-status.voice-failed { color: $ds-status-error-readable; }
    #console-character-voice-meter { color: $ds-status-running; }
    #console-character-voice-meter.voice-suppressed { visibility: hidden; }
    PortraitRadio.voice-compact #console-character-voice-meter { display: none; }
    """

    def __init__(self) -> None:
        super().__init__(id="console-character-radio")
        self._projection: tuple | None = None

    def compose(self) -> ComposeResult:
        yield Static("Ready", id="console-character-voice-status", markup=False)
        yield Static(
            "",
            id="console-character-voice-meter",
            markup=False,
            classes="voice-suppressed",
        )

    def set_compact(self, compact: bool) -> None:
        """Restore a hidden meter until the existing clock supplies current state."""
        self.set_class(compact, "voice-compact")
        self._projection = None
        self.query_one("#console-character-voice-meter").add_class("voice-suppressed")

    def update_presentation(
        self, snapshot: SpeechPresentationSnapshot | None, *, motion: bool
    ) -> None:
        """Change text and visibility only; every ordinary row keeps its geometry."""
        if not self.is_mounted:
            return
        label = portrait_radio_status(snapshot)
        level = portrait_radio_level(snapshot, now=time.monotonic()) if motion else None
        status = self.query_one("#console-character-voice-status", Static)
        meter = self.query_one("#console-character-voice-meter", Static)
        cells = max(1, self.content_size.width)
        projection = (
            label,
            level is not None,
            round(cells * (level or 0)),
            cells,
            motion,
        )
        if projection == self._projection:
            return
        self._projection = projection
        status.update(label, layout=False)
        status.set_class(label == "Voice unavailable", "voice-failed")
        status.tooltip = (
            "Speech failed. Open Settings to review speech configuration."
            if label == "Voice unavailable"
            else "Output timing unavailable; showing voice status only."
            if label == "Transmitting" and level is None and motion
            else None
        )
        meter.set_class(level is None, "voice-suppressed")
        meter.update("▰" * projection[2], layout=False)


def update_portrait_radio(
    screen: Screen,
    snapshot: SpeechPresentationSnapshot | None,
    *,
    show: bool,
    motion: bool,
) -> None:
    """Apply the controller's scalar projection at the existing Textual edge."""
    from textual.css.query import NoMatches

    try:
        radio = screen.query_one(PortraitRadio)
    except NoMatches:
        return
    radio.display = show
    if (
        show
        and radio.screen.is_current
        and radio.is_on_screen
        and all(getattr(node, "display", True) for node in radio.ancestors_with_self)
    ):
        radio.update_presentation(snapshot, motion=motion)
