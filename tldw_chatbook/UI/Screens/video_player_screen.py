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
    playback_tools_available,
    probe_file,
)
from tldw_chatbook.Media_Playback.render_mode import (
    RenderMode,
    detect_render_mode,
    frame_to_ascii,
)
from tldw_chatbook.Widgets.Console.console_video_preview import ConsoleVideoPreview

SEEK_STEP_SECONDS = 5.0
_STATUS_INTERVAL_SECONDS = 0.25
_HINTS = "space pause · s stop · ←/→ seek ±5s · q close"


def _format_clock(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    total = max(0, int(seconds))
    return f"{total // 60}:{total % 60:02d}"


class VideoPlayerScreen(ModalScreen[None]):
    """Play one video file with audio + sync inside the Console."""

    BINDINGS = [
        Binding("space", "toggle_pause", "Pause", show=True),
        Binding("s", "stop_playback", "Stop", show=True),
        Binding("left", "seek_back", "-5s", show=True),
        Binding("right", "seek_fwd", "+5s", show=True),
        Binding("q", "close_player", "Close", show=True),
    ]

    def __init__(
        self,
        file_path: str,
        *,
        title: str = "",
        render_mode: RenderMode | None = None,
    ) -> None:
        super().__init__()
        self._file_path = file_path
        self._title = title
        self._mode: RenderMode = render_mode or detect_render_mode()
        self._pipeline: PlayerPipeline | None = None
        self._probe: PlayerProbe | None = None
        self._paused = False
        self._finished = False
        self._stop_requested = False
        self._frame_static_id = "video-player-frame"

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
        yield Static(_HINTS, id="video-player-hints", classes="video-player-hints")

    # -- lifecycle ----------------------------------------------------------

    def on_mount(self) -> None:
        ok, guidance = playback_tools_available()
        if not ok:
            self.app.notify(guidance, severity="warning")
            self.dismiss(None)
            return
        # One-active rule: the modal player supersedes any playing preview.
        ConsoleVideoPreview._pause_active_peer()
        try:
            self._probe = probe_file(self._file_path)
        except Exception as exc:
            self.app.notify(f"Could not read the video: {exc}", severity="error")
            self.dismiss(None)
            return
        self._pipeline = PlayerPipeline(self._file_path, self._probe)
        self._pipeline.start()
        self.run_worker(self._pump_loop, thread=True, exclusive=True, group="video-player-pump")
        self.set_interval(_STATUS_INTERVAL_SECONDS, self._refresh_status)

    def on_unmount(self) -> None:
        self._stop_requested = True
        if self._pipeline is not None:
            self._pipeline.stop()

    # -- frame pump (worker thread) -------------------------------------------

    def _pump_loop(self) -> None:
        """Drive the pipeline; re-enter the frame iterator across seeks.

        The pipeline's generation counter distinguishes a seek restart
        (generation bumped -- re-enter the pump) from natural EOF
        (unchanged -- finish).
        """
        pipeline = self._pipeline
        if pipeline is None:
            return
        while not self._stop_requested:
            generation = pipeline._generation
            for pts, data in pipeline.iter_frames():
                if self._stop_requested or pipeline._generation != generation:
                    break
                if pipeline.frames_behind(pts):
                    pipeline.note_dropped(pts)
                    continue
                if not pipeline.frame_due(pts):
                    # Frame arrived early (CFR pipe ahead of the clock):
                    # wait out the gap once, bounded, then render regardless.
                    wait = min(0.1, max(0.0, pts - pipeline.sync_clock))
                    if wait:
                        time.sleep(wait)
                pipeline.note_rendered(pts)
                try:
                    self.call_from_thread(self._render_frame, data)
                except Exception:
                    return  # screen already gone
            if self._stop_requested:
                return
            if pipeline._generation != generation:
                continue  # seek restarted the pipeline mid-pump
            if pipeline.at_eof and not self._paused:
                self._finish()
                return

    def _render_frame(self, data: bytes) -> None:
        """Render one rgb24 frame on the UI thread, per detected mode."""
        probe = self._probe
        if probe is None or self._finished:
            return
        from PIL import Image as PILImage

        image = PILImage.frombytes("RGB", (probe.width, probe.height), data)
        try:
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
            logger.debug("video player frame render skipped: {}", exc)

    # -- status ---------------------------------------------------------------

    def _refresh_status(self) -> None:
        pipeline = self._pipeline
        if pipeline is None:
            return
        state = "⏸ paused" if self._paused else ("■ finished" if self._finished else "▶ playing")
        probe = self._probe
        duration = probe.duration_seconds if probe else None
        stats = pipeline.stats
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

    def _finish(self) -> None:
        self._finished = True
        try:
            self.call_from_thread(self._refresh_status)
        except Exception:
            self._refresh_status()

    # -- actions ----------------------------------------------------------------

    def action_toggle_pause(self) -> None:
        if self._pipeline is None or self._finished:
            return
        if self._paused:
            self._pipeline.resume()
            self._paused = False
        else:
            self._pipeline.pause()
            self._paused = True
        self._refresh_status()

    def action_stop_playback(self) -> None:
        self.dismiss(None)

    def action_close_player(self) -> None:
        self.dismiss(None)

    def action_seek_back(self) -> None:
        self._seek_relative(-SEEK_STEP_SECONDS)

    def action_seek_fwd(self) -> None:
        self._seek_relative(SEEK_STEP_SECONDS)

    def _seek_relative(self, delta: float) -> None:
        pipeline = self._pipeline
        if pipeline is None or self._finished:
            return
        target = max(0.0, pipeline.stats.position_seconds + delta)
        pipeline.seek(target)
