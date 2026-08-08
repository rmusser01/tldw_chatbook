"""In-card silent video preview widget (task-3401.9, ADR-044 §5A).

Architecture mirrors textual-video (MIT): decode off the UI thread, render
frames through the existing ``rich_pixels`` inline-image path. Hard product
rules (all AC-driven):

- **Paused by default.** The poster (a ▶ glyph + hint) costs zero CPU/GPU;
  the first decode happens only after an explicit play (click), so a
  transcript full of cards never burns cycles unprompted.
- **One active preview.** Starting one pauses whichever preview was playing
  (class-level registry -- the Console transcript has a single viewport's
  worth of attention).
- **Off-screen pauses.** A slow interval timer (running only while playing)
  pauses the preview once its region leaves the viewport.
- **Capped.** Clip eligibility is enforced upstream
  (``Media_Playback.preview_policy``); frames render at the preview's
  target fps through the same fit-to-cell sizing as image rows.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from loguru import logger
from rich_pixels import Pixels
from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Static

from tldw_chatbook.Chat.console_image_view import (
    PIXELS_MAX_COLS,
    PIXELS_MAX_LINES,
)
from tldw_chatbook.Media_Playback.availability import (
    VIDEO_PLAYBACK_INSTALL_GUIDANCE,
    av_available,
)
from tldw_chatbook.Media_Playback.frame_source import AvFrameSource

PreviewState = str  # "poster" | "playing" | "paused"


def _format_clock(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    total = max(0, int(seconds))
    return f"{total // 60}:{total % 60:02d}"


def progress_line(
    position: float | None,
    duration: float | None,
    *,
    width: int = 20,
) -> str:
    """Return a text progress bar like ``▓▓▓░░░░░ 0:04 / 0:06`` (no deps).

    Pure helper so the preview never needs textual-canvas to show progress
    (the canvas scrubber is task-3401.10's player-screen surface).
    """
    if duration is None or duration <= 0 or position is None:
        return f"{_format_clock(position)} / {_format_clock(duration)}"
    fraction = min(1.0, max(0.0, position / duration))
    filled = round(fraction * width)
    bar = "▓" * filled + "░" * (width - filled)
    return f"{bar} {_format_clock(position)} / {_format_clock(duration)}"


class ConsoleVideoPreview(Vertical):
    """One in-card silent preview for a live video file.

    Mount it only when the card's status is ``ready`` (file resolved). An
    ineligible clip (caps) or a missing ``av`` renders guidance text instead
    of a player -- never an exception, never an empty frame.
    """

    #: The one-active-preview registry: the currently playing instance, if any.
    _active: "ConsoleVideoPreview | None" = None

    state: reactive[PreviewState] = reactive("poster")

    def __init__(
        self,
        file_path: str,
        *,
        duration_seconds: float | None = None,
        eligible: bool = True,
        ineligible_reason: str = "",
    ) -> None:
        super().__init__(
            id=f"console-video-preview-{uuid4().hex[:12]}",
            classes="console-video-preview",
        )
        self._file_path = file_path
        self._duration = duration_seconds
        if not av_available():
            self._eligible = False
            self._ineligible_reason = VIDEO_PLAYBACK_INSTALL_GUIDANCE
        elif not eligible:
            self._eligible = False
            self._ineligible_reason = ineligible_reason
        else:
            self._eligible = True
            self._ineligible_reason = ""
        self._source: AvFrameSource | None = None
        self._position: float | None = None
        self._pixels: Pixels | None = None
        self._offscreen_timer: Any | None = None

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static(
            self._poster_text(),
            id=f"{self.id}-frame",
            classes="console-video-preview-frame",
        )
        yield Static(
            progress_line(None, self._duration),
            id=f"{self.id}-progress",
            classes="console-video-preview-progress",
        )

    def _poster_text(self) -> str:
        if not self._eligible:
            return f"({self._ineligible_reason})"
        if self._pixels is not None or self.state == "paused":
            hint = "paused — click to resume"
        else:
            hint = "click to play (silent preview)"
        return f"▶ Preview — {hint}"

    # -- playback control -----------------------------------------------------

    def play(self) -> None:
        """Start decoding (explicit user action only -- never called on mount)."""
        if not self._eligible or self.state == "playing":
            return
        self._pause_active_peer()
        ConsoleVideoPreview._active = self
        self.state = "playing"
        if self._source is None:
            self._source = AvFrameSource(self._file_path)
            eligible, reason = self._source.check_eligible()
            if not eligible:
                # Probed shape violates the caps even though metadata passed.
                self._eligible = False
                self._ineligible_reason = reason
                self.state = "poster"
                ConsoleVideoPreview._active = None
                self._source.close()
                self._source = None
                self._refresh_frame()
                return
        self.run_worker(
            self._decode_loop, thread=True, exclusive=True, group="video-preview-decode"
        )
        self._offscreen_timer = self.set_interval(0.5, self._pause_if_offscreen)

    def pause(self) -> None:
        """Pause (keeps the last rendered frame; resumes on next play)."""
        if self.state != "playing":
            return
        self.state = "paused"
        self._stop_decode()
        self._refresh_frame()

    def _stop_decode(self) -> None:
        if ConsoleVideoPreview._active is self:
            ConsoleVideoPreview._active = None
        if self._offscreen_timer is not None:
            self._offscreen_timer.stop()
            self._offscreen_timer = None
        if self._source is not None:
            self._source.close()
            self._source = None

    @classmethod
    def _pause_active_peer(cls) -> None:
        """Pause whichever preview is playing (one-active rule)."""
        active = cls._active
        if active is not None:
            try:
                active.pause()
            except Exception:
                logger.opt(exception=True).debug("video preview peer pause failed")
            finally:
                cls._active = None

    # -- decode loop (worker thread) -------------------------------------------

    def _decode_loop(self) -> None:
        """Drive the frame source, pushing frames onto the UI thread (worker)."""
        source = self._source
        if source is None:
            return
        try:
            for timestamp, image in source.iter_frames():
                if self.state != "playing":
                    break
                self.call_from_thread(self._show_frame, timestamp, image)
        except Exception as exc:
            logger.warning("video preview decode loop ended early: {}", exc)
        finally:
            # Natural EOF lands paused-at-end (the last frame stays up);
            # an explicit pause already transitioned state and stopped us.
            if self.state == "playing":
                self.call_from_thread(self.pause)

    def _show_frame(self, timestamp: float, image: Any) -> None:
        """Render one decoded frame (UI thread)."""
        if self.state != "playing":
            return
        try:
            scaled = image.copy()
            scaled.thumbnail((PIXELS_MAX_COLS, PIXELS_MAX_LINES * 2))
            self._pixels = Pixels.from_image(scaled)
        except Exception as exc:
            logger.debug("video preview frame render skipped: {}", exc)
            return
        self._position = timestamp
        self._refresh_frame()
        self._refresh_progress()

    def _refresh_frame(self) -> None:
        try:
            frame = self.query_one(f"#{self.id}-frame", Static)
        except Exception:
            return  # children not mounted yet (or already gone)
        if self._pixels is not None and self.state in {"playing", "paused"}:
            frame.update(self._pixels)
        else:
            frame.update(self._poster_text())

    def _refresh_progress(self) -> None:
        try:
            progress = self.query_one(f"#{self.id}-progress", Static)
        except Exception:
            return
        progress.update(progress_line(self._position, self._duration))

    # -- off-screen / lifecycle guards ------------------------------------------

    def _pause_if_offscreen(self) -> None:
        """Pause when the preview's region has left the viewport (AC2)."""
        if self.state != "playing":
            return
        try:
            region = self.region
            viewport_height = self.screen.size.height
        except Exception:
            return  # unmounted/transitioning -- the unmount hook covers it
        if region.bottom < 0 or region.top > viewport_height:
            self.pause()

    def on_click(self, event: events.Click) -> None:
        """Toggle playback on click (the explicit user action AC6 requires)."""
        event.stop()
        if self.state == "playing":
            self.pause()
        else:
            self.play()

    def on_unmount(self) -> None:
        """Release the decoder when the row reconciles away."""
        self._stop_decode()
