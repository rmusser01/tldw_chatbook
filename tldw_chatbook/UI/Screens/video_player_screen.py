"""Modal full-player screen (task-3401.10, ADR-044 decision 4).

The real watching surface, unlike the in-card preview: audio (ffplay,
master clock), clock-driven A/V sync with drift/dropped stats, seek, and
terminal-capability rendering (kitty -> sixel -> halfcell -> ascii). One
ffmpeg process demuxes the source ONCE into the video frame pipe and the
audio PCM pipe (AC6). Controls follow the repo's keybinding conventions
(decision 031): single-letter htop-style, no terminal-reserved chords, and
the hints line names only implemented actions.

Opening the player pauses any playing transcript preview (one-active rule,
shared with task-3401.9's preview registry).
"""

from __future__ import annotations

import time
from functools import partial
from typing import Any, Callable

from loguru import logger
from rich_pixels import Pixels
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Static

from tldw_chatbook.Chat.console_image_view import (
    PIXELS_MAX_COLS,
    PIXELS_MAX_LINES,
)
from tldw_chatbook.Media_Playback.player_pipeline import (
    PlayerPipeline,
    PlayerProbe,
    PlayerRun,
    playback_tools_available,
    probe_file,
)
from tldw_chatbook.Media_Playback.render_mode import (
    RenderMode,
    detect_render_mode,
    frame_to_ascii,
)
from tldw_chatbook.Widgets.Console.console_video_preview import ConsoleVideoPreview
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

SEEK_STEP_SECONDS = 5.0
_STATUS_INTERVAL_SECONDS = 0.25
_STALL_THRESHOLD_SECONDS = 3.0
_HINTS = "space pause · s stop · ←/→ seek ±5s · q close"
_HINTS_NO_SEEK = "space pause · s stop · seek unavailable for this stream · q close"
_FAILURE_GUIDANCE = (
    "Video playback stopped. Try Play again or open the video in your system player."
)
_BRIDGE_REFUSED = object()


def _log_failure(phase: str, exc: BaseException) -> None:
    """Log bounded modal diagnostics without exception text or media identity."""
    logger.warning(
        "Console video playback failed: component=modal_player phase={} error_type={}",
        phase,
        type(exc).__name__,
    )


def _format_clock(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    total = max(0, int(seconds))
    return f"{total // 60}:{total % 60:02d}"


class VideoPlayerScreen(SafeModalDismissMixin, ModalScreen[None]):
    """Play one video file with audio + sync inside the Console."""

    BINDINGS = [
        Binding("space", "toggle_pause", "Pause", show=True),
        Binding("s", "stop_playback", "Stop", show=True),
        Binding("left", "seek_back", "-5s", show=True),
        Binding("right", "seek_fwd", "+5s", show=True),
        Binding("q", "close_player", "Close", show=True),
        Binding("escape", "request_safe_cancel", "Close", show=False),
    ]

    # The player occupies the whole screen, so it has no synthetic backdrop.
    SAFE_MODAL_CONTENT = None

    def __init__(
        self,
        file_path: str,
        *,
        title: str = "",
        render_mode: RenderMode | None = None,
        seekable: bool = True,
        max_seconds: float | None = None,
    ) -> None:
        super().__init__()
        self._file_path = file_path
        self._title = title
        self._mode: RenderMode = render_mode or detect_render_mode()
        #: AC4 (streams): a source without range support plays without seek.
        self._seekable = seekable
        #: AC5 (streams): session time-box; ``None`` for local files.
        self._max_seconds = max_seconds
        self._activation_token = 0
        self._pipeline: PlayerPipeline | None = None
        self._run: PlayerRun | None = None
        self._probe: PlayerProbe | None = None
        self._seek_in_flight = False
        self._status_timer: Any | None = None
        self._mounted = False
        self._paused = False
        self._finished = False
        self._frame_static_id = "video-player-frame"
        self._started_wall: float | None = None
        self._last_frame_wall: float | None = None

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        if self._mode == "sixel":
            try:
                from textual_image.widget.sixel import Image as SixelImage

                yield SixelImage(
                    None, id=self._frame_static_id, classes="video-player-frame"
                )
            except Exception:
                # sixel import failed after detection: degrade one rung.
                self._mode = "halfcell"
                yield Static("", id=self._frame_static_id, classes="video-player-frame")
        else:
            yield Static("", id=self._frame_static_id, classes="video-player-frame")
        yield Static("", id="video-player-status", classes="video-player-status")
        yield Static(
            _HINTS if self._seekable else _HINTS_NO_SEEK,
            id="video-player-hints",
            classes="video-player-hints",
        )

    # -- lifecycle ----------------------------------------------------------

    def on_mount(self) -> None:
        self._mounted = True
        ok, guidance = playback_tools_available()
        if not ok:
            self._notify_and_dismiss(guidance, severity="warning")
            return
        # One-active rule: the modal player supersedes any playing preview.
        ConsoleVideoPreview._pause_active_peer()
        self._activation_token += 1
        token = self._activation_token
        self.run_worker(
            partial(self._activate, token),
            thread=True,
            group="video-player-activation",
        )

    def _activate(self, token: int) -> None:
        """Probe and start a privately owned pipeline off the UI thread."""
        pipeline: PlayerPipeline | None = None
        try:
            probe = probe_file(self._file_path)
            pipeline = PlayerPipeline(self._file_path, probe)
            run = pipeline.start()
        except Exception as exc:
            _log_failure("activation", exc)
            if pipeline is not None:
                self._cleanup_pipeline(pipeline)
            self._bridge(self._activation_failed, token)
            return

        accepted = self._bridge(
            self._accept_activation,
            token,
            pipeline,
            run,
            probe,
        )
        if accepted is not True:
            self._cleanup_pipeline(pipeline)

    def _accept_activation(
        self,
        token: int,
        pipeline: PlayerPipeline,
        run: PlayerRun,
        probe: PlayerProbe,
    ) -> bool:
        if not self._mounted or token != self._activation_token:
            return False
        self._pipeline = pipeline
        self._run = run
        self._probe = probe
        self._finished = False
        self._started_wall = time.monotonic()
        try:
            self._status_timer = self.set_interval(
                _STATUS_INTERVAL_SECONDS, self._refresh_status
            )
            self._start_pump(token, pipeline, run)
        except Exception as exc:
            _log_failure("activation", exc)
            self._invalidate_current()
            self._notify_and_dismiss(_FAILURE_GUIDANCE, severity="error")
            return False
        return True

    def _activation_failed(self, token: int) -> bool:
        if not self._mounted or token != self._activation_token:
            return False
        self._invalidate_current()
        self._notify_and_dismiss(_FAILURE_GUIDANCE, severity="error")
        return True

    def on_unmount(self) -> None:
        app = self.app
        pipeline = self._invalidate_current()
        if pipeline is not None:
            self._schedule_cleanup(app, pipeline)

    def _invalidate_current(self) -> PlayerPipeline | None:
        self._mounted = False
        self._activation_token += 1
        pipeline = self._pipeline
        self._pipeline = None
        self._run = None
        self._probe = None
        self._seek_in_flight = False
        timer, self._status_timer = self._status_timer, None
        if timer is not None:
            try:
                timer.stop()
            except Exception as exc:
                _log_failure("cleanup", exc)
        return pipeline

    @staticmethod
    def _schedule_cleanup(app: Any, pipeline: PlayerPipeline) -> None:
        try:
            app.run_worker(
                partial(VideoPlayerScreen._cleanup_pipeline, pipeline),
                thread=True,
                exit_on_error=False,
                group="video-player-cleanup",
            )
        except Exception as exc:
            _log_failure("cleanup", exc)

    @staticmethod
    def _cleanup_pipeline(pipeline: PlayerPipeline) -> None:
        try:
            pipeline.stop()
        except Exception as exc:
            _log_failure("cleanup", exc)

    def _bridge(self, callback: Callable[..., Any], *args: Any) -> Any:
        """Attempt one worker-to-UI dispatch, with no unsafe fallback."""
        try:
            return self.app.call_from_thread(callback, *args)
        except Exception as exc:
            _log_failure("frame_dispatch", exc)
            return _BRIDGE_REFUSED

    # -- frame pump (worker thread) -------------------------------------------

    def _start_pump(self, token: int, pipeline: PlayerPipeline, run: PlayerRun) -> None:
        self.run_worker(
            partial(self._pump_loop, token, pipeline, run),
            thread=True,
            group="video-player-pump",
        )

    def _pump_loop(self, token: int, pipeline: PlayerPipeline, run: PlayerRun) -> None:
        """Drive exactly one immutable playback generation."""
        try:
            for pts, data in pipeline.iter_frames(run):
                if pipeline.frames_behind(run, pts):
                    pipeline.note_dropped(run, pts)
                    continue
                if not pipeline.frame_due(run, pts):
                    # Frame arrived early (CFR pipe ahead of the clock):
                    # wait out the gap once, bounded, then render regardless.
                    wait = min(0.1, max(0.0, pts - pipeline.sync_clock(run)))
                    if wait:
                        time.sleep(wait)
                pipeline.note_rendered(run, pts)
                dispatched = self._bridge(
                    self._render_frame, token, pipeline, run, data
                )
                if dispatched is _BRIDGE_REFUSED:
                    return
                if dispatched is False:
                    return
            if run.eof:
                self._bridge(self._finish_run, token, pipeline, run)
        except Exception as exc:
            _log_failure("pump", exc)
            self._bridge(self._fail_run, token, pipeline, run)

    def _matches(self, token: int, pipeline: PlayerPipeline, run: PlayerRun) -> bool:
        return (
            self._mounted
            and token == self._activation_token
            and pipeline is self._pipeline
            and run is self._run
        )

    def _render_frame(
        self,
        token: int,
        pipeline: PlayerPipeline,
        run: PlayerRun,
        data: bytes,
    ) -> bool:
        """Render one rgb24 frame on the UI thread, per detected mode."""
        if not self._matches(token, pipeline, run):
            return False
        probe = self._probe
        if probe is None or self._finished:
            return False
        self._last_frame_wall = time.monotonic()
        try:
            from PIL import Image as PILImage

            image = PILImage.frombytes("RGB", (probe.width, probe.height), data)
            if self._mode == "kitty":
                from textual_image.renderable.tgp import Image as TGPImage

                self.query_one(f"#{self._frame_static_id}", Static).update(
                    TGPImage(image, width=PIXELS_MAX_COLS, height=PIXELS_MAX_LINES)
                )
            elif self._mode == "sixel":
                from textual_image.widget.sixel import Image as SixelImage

                widget = self.query_one(f"#{self._frame_static_id}", SixelImage)
                widget.image = image
            elif self._mode == "halfcell":
                scaled = image.copy()
                scaled.thumbnail((PIXELS_MAX_COLS, PIXELS_MAX_LINES * 2))
                self.query_one(f"#{self._frame_static_id}", Static).update(
                    Pixels.from_image(scaled)
                )
            else:  # ascii
                self.query_one(f"#{self._frame_static_id}", Static).update(
                    frame_to_ascii(image, cols=PIXELS_MAX_COLS)
                )
        except Exception as exc:
            _log_failure("render", exc)
            self._fail_run(token, pipeline, run)
            return False
        return True

    def _finish_run(self, token: int, pipeline: PlayerPipeline, run: PlayerRun) -> bool:
        if not self._matches(token, pipeline, run):
            return False
        self._finished = True
        self._refresh_status()
        return True

    def _fail_run(self, token: int, pipeline: PlayerPipeline, run: PlayerRun) -> bool:
        if not self._matches(token, pipeline, run):
            return False
        app = self.app
        detached = self._invalidate_current()
        if detached is not None:
            self._schedule_cleanup(app, detached)
        self._notify_and_dismiss(_FAILURE_GUIDANCE, severity="error")
        return True

    def _notify_and_dismiss(self, message: str, *, severity: str) -> None:
        try:
            self.app.notify(message, severity=severity)
        except Exception as exc:
            _log_failure("cleanup", exc)
        try:
            self.dismiss(None)
        except Exception as exc:
            _log_failure("cleanup", exc)

    # -- status ---------------------------------------------------------------

    def _refresh_status(self) -> None:
        run = self._run
        if run is None:
            return
        # AC5: time-boxed stream sessions auto-stop at the cap.
        if self._max_seconds is not None and self._started_wall is not None:
            elapsed = time.monotonic() - self._started_wall
            if elapsed > self._max_seconds:
                self.app.notify(
                    f"Stream session hit the {int(self._max_seconds // 60)}-minute time box.",
                    severity="information",
                )
                self.dismiss(None)
                return
        state = (
            "⏸ paused"
            if self._paused
            else ("■ finished" if self._finished else "▶ playing")
        )
        # AC3: surface stalls (no frames for a while mid-play) -- ffmpeg's
        # reconnect flags are what actually resumes them.
        if (
            state == "▶ playing"
            and self._last_frame_wall is not None
            and time.monotonic() - self._last_frame_wall > _STALL_THRESHOLD_SECONDS
        ):
            state = "…stalled"
        probe = self._probe
        duration = probe.duration_seconds if probe else None
        stats = run.stats
        title = f"{self._title} — " if self._title else ""
        text = (
            f"{title}{state} · {_format_clock(stats.position_seconds)} / "
            f"{_format_clock(duration)} · drift {stats.last_drift_ms:+.0f}ms "
            f"(max {stats.max_drift_ms:.0f}ms) · dropped {stats.dropped_frames} · {self._mode}"
        )
        try:
            self.query_one("#video-player-status", Static).update(text)
        except Exception:
            pass

    # -- actions ----------------------------------------------------------------

    def action_toggle_pause(self) -> None:
        pipeline = self._pipeline
        run = self._run
        if pipeline is None or run is None or self._finished:
            return
        if self._paused:
            pipeline.resume()
            self._paused = False
        else:
            pipeline.pause()
            self._paused = True
        self._refresh_status()

    def action_stop_playback(self) -> None:
        self.dismiss_safe_once(None)

    def action_close_player(self) -> None:
        self.dismiss_safe_once(None)

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        self.action_close_player()

    def action_seek_back(self) -> None:
        self._seek_relative(-SEEK_STEP_SECONDS)

    def action_seek_fwd(self) -> None:
        self._seek_relative(SEEK_STEP_SECONDS)

    def _seek_relative(self, delta: float) -> None:
        pipeline = self._pipeline
        run = self._run
        if pipeline is None or run is None or self._finished or self._seek_in_flight:
            return
        if not self._seekable:
            # AC4: non-seekable streams disable seek (the hints line says so).
            self.app.notify("Seek is unavailable for this stream.", severity="warning")
            return
        target = max(0.0, run.stats.position_seconds + delta)
        self._seek_in_flight = True
        self._activation_token += 1
        token = self._activation_token
        self._run = None
        self.run_worker(
            partial(self._seek, token, pipeline, target),
            thread=True,
            group="video-player-seek",
        )

    def _seek(self, token: int, pipeline: PlayerPipeline, target: float) -> None:
        try:
            run = pipeline.seek(target)
        except Exception as exc:
            _log_failure("seek", exc)
            self._bridge(self._seek_failed, token, pipeline)
            return
        accepted = self._bridge(self._accept_seek, token, pipeline, run)
        if accepted is not True:
            self._cleanup_pipeline(pipeline)

    def _accept_seek(
        self, token: int, pipeline: PlayerPipeline, run: PlayerRun
    ) -> bool:
        if (
            not self._mounted
            or token != self._activation_token
            or pipeline is not self._pipeline
            or not self._seek_in_flight
        ):
            return False
        self._run = run
        self._finished = False
        try:
            self._start_pump(token, pipeline, run)
        except Exception as exc:
            _log_failure("pump", exc)
            self._fail_run(token, pipeline, run)
            return False
        self._seek_in_flight = False
        return True

    def _seek_failed(self, token: int, pipeline: PlayerPipeline) -> bool:
        if (
            not self._mounted
            or token != self._activation_token
            or pipeline is not self._pipeline
            or not self._seek_in_flight
        ):
            return False
        app = self.app
        detached = self._invalidate_current()
        if detached is not None:
            self._schedule_cleanup(app, detached)
        self._notify_and_dismiss(_FAILURE_GUIDANCE, severity="error")
        return True
