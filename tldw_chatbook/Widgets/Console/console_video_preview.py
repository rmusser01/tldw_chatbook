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

from dataclasses import dataclass
from functools import partial
from threading import Event
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


@dataclass(frozen=True)
class _PreviewRun:
    generation: int
    cancelled: Event


def _log_preview_failure(phase: str, error: BaseException) -> None:
    """Log bounded diagnostics without media paths or exception payloads."""
    logger.warning(
        "Console video playback failed: component=inline_preview "
        "phase={} error_type={}",
        phase,
        type(error).__name__,
    )


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
        self._generation = 0
        self._run: _PreviewRun | None = None
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
        self._generation += 1
        run = _PreviewRun(self._generation, Event())
        self._run = run
        ConsoleVideoPreview._active = self
        self.state = "playing"
        self.run_worker(
            partial(self._decode_loop, run),
            thread=True,
            exclusive=True,
            group="video-preview-decode",
        )

    def pause(self) -> None:
        """Pause (keeps the last rendered frame; resumes on next play)."""
        if self.state != "playing":
            return
        self.state = "paused"
        self._stop_decode()
        self._refresh_frame()

    def _stop_decode(self) -> None:
        run, self._run = self._run, None
        self._source = None
        if run is not None:
            run.cancelled.set()
        if ConsoleVideoPreview._active is self:
            ConsoleVideoPreview._active = None
        timer, self._offscreen_timer = self._offscreen_timer, None
        if timer is not None:
            try:
                timer.stop()
            except Exception as exc:
                _log_preview_failure("cleanup", exc)

    @classmethod
    def _pause_active_peer(cls) -> None:
        """Pause whichever preview is playing (one-active rule)."""
        active = cls._active
        if active is not None:
            try:
                active.pause()
            except Exception as exc:
                _log_preview_failure("cleanup", exc)
            finally:
                cls._active = None

    # -- decode loop (worker thread) -------------------------------------------

    def _bridge(self, phase: str, callback: Any, *args: Any) -> Any:
        """Attempt one worker-to-app bridge with no direct-thread fallback."""
        try:
            return self.app.call_from_thread(callback, *args)
        except Exception as exc:
            _log_preview_failure(phase, exc)
            return False

    def _decode_loop(self, run: _PreviewRun) -> None:
        """Create, probe, decode, and close one private source in its worker."""
        source: AvFrameSource | None = None
        try:
            try:
                source = AvFrameSource(self._file_path)
                eligible, reason = source.check_eligible()
            except Exception as exc:
                _log_preview_failure("open", exc)
                self._bridge("frame_dispatch", self._activation_failed, run, source)
                return

            if not eligible:
                self._bridge(
                    "frame_dispatch",
                    self._activation_ineligible,
                    run,
                    source,
                    reason,
                )
                return
            if run.cancelled.is_set():
                return
            if (
                self._bridge("frame_dispatch", self._accept_source, run, source)
                is not True
            ):
                return

            try:
                iterator = iter(source.iter_frames())
                while not run.cancelled.is_set():
                    try:
                        timestamp, image = next(iterator)
                    except StopIteration:
                        self._bridge("eof", self._finish_run, run, source)
                        return
                    except Exception as exc:
                        _log_preview_failure("decode", exc)
                        self._bridge("frame_dispatch", self._degrade_run, run, source)
                        return
                    if (
                        self._bridge(
                            "frame_dispatch",
                            self._show_frame,
                            run,
                            source,
                            timestamp,
                            image,
                        )
                        is not True
                    ):
                        return
            except Exception as exc:
                _log_preview_failure("decode", exc)
                self._bridge("frame_dispatch", self._degrade_run, run, source)
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception as exc:
                    _log_preview_failure("cleanup", exc)

    def _accept_source(self, run: _PreviewRun, source: AvFrameSource) -> bool:
        """Attach a current worker source and arm policy before decode starts."""
        if (
            not self.is_attached
            or self._run is not run
            or self._source is not None
            or run.cancelled.is_set()
            or self.state != "playing"
        ):
            return False
        self._source = source
        try:
            self._offscreen_timer = self.set_interval(
                0.5, partial(self._pause_if_offscreen, run, source)
            )
        except Exception as exc:
            _log_preview_failure("cleanup", exc)
            self._degrade_run(run, source)
            return False
        return True

    def _activation_failed(
        self,
        run: _PreviewRun,
        source: AvFrameSource | None,
    ) -> bool:
        """Degrade only a still-current activation that published no source."""
        if (
            not self.is_attached
            or self._run is not run
            or self._source is not None
            or run.cancelled.is_set()
            or self.state != "playing"
        ):
            return False
        self._make_unavailable()
        return True

    def _activation_ineligible(
        self,
        run: _PreviewRun,
        source: AvFrameSource,
        reason: str,
    ) -> bool:
        """Publish a probed cap refusal only for its current activation."""
        if (
            not self.is_attached
            or self._run is not run
            or self._source is not None
            or run.cancelled.is_set()
            or self.state != "playing"
        ):
            return False
        self._eligible = False
        self._ineligible_reason = reason
        self.state = "poster"
        self._stop_decode()
        self._refresh_frame()
        return True

    def _matches(self, run: _PreviewRun, source: AvFrameSource) -> bool:
        return (
            self.is_attached
            and self._run is run
            and self._source is source
            and self.state == "playing"
        )

    def _make_unavailable(self) -> None:
        self._eligible = False
        self._ineligible_reason = (
            "Inline preview stopped — use Play for the full player "
            "or open the clip in your system player."
        )
        self.state = "poster"
        self._stop_decode()
        self._refresh_frame()

    def _degrade_run(self, run: _PreviewRun, source: AvFrameSource) -> bool:
        if not self._matches(run, source):
            return False
        self._make_unavailable()
        return True

    def _finish_run(self, run: _PreviewRun, source: AvFrameSource) -> bool:
        if not self._matches(run, source):
            return False
        self.state = "paused"
        self._stop_decode()
        self._refresh_frame()
        return True

    def _show_frame(
        self,
        run: _PreviewRun,
        source: AvFrameSource,
        timestamp: float,
        image: Any,
    ) -> bool:
        """Render one decoded frame (UI thread)."""
        if not self._matches(run, source):
            return False
        try:
            scaled = image.copy()
            scaled.thumbnail((PIXELS_MAX_COLS, PIXELS_MAX_LINES * 2))
            self._pixels = Pixels.from_image(scaled)
            self._position = timestamp
            self._update_frame()
            self._update_progress()
        except Exception as exc:
            _log_preview_failure("render", exc)
            self._degrade_run(run, source)
            return False
        return True

    def _update_frame(self) -> None:
        frame = self.query_one(f"#{self.id}-frame", Static)
        if self._pixels is not None and self.state in {"playing", "paused"}:
            frame.update(self._pixels)
        else:
            frame.update(self._poster_text())

    def _refresh_frame(self) -> None:
        try:
            self._update_frame()
        except Exception:
            return  # children not mounted yet (or already gone)

    def _update_progress(self) -> None:
        progress = self.query_one(f"#{self.id}-progress", Static)
        progress.update(progress_line(self._position, self._duration))

    # -- off-screen / lifecycle guards ------------------------------------------

    def _pause_if_offscreen(
        self,
        run: _PreviewRun,
        source: AvFrameSource,
    ) -> None:
        """Pause when the preview's region has left the viewport (AC2)."""
        if not self._matches(run, source):
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
