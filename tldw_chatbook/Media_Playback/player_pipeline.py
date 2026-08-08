"""Full-player pipeline: single-demux ffmpeg/ffplay playback with A/V sync
(task-3401.10, ADR-044 decision 4).

One ``ffmpeg`` process demuxes the source ONCE into two pipes -- raw video
frames on stdout and raw PCM audio on an inherited fd -- while ``ffplay``
(-nodisp) plays the PCM from that pipe end-to-end (the parent brokers the
fds; audio never round-trips through us, so the source is fetched exactly
once -- AC6). Video is forced constant-frame-rate at the target fps, so
every frame's presentation time is exactly ``index / target_fps``.

Sync clock: wall time (pause-adjusted), with ffplay's small internal
buffer subtracted when audio is present. Local playback consumes at
real-time rate, so decode position tracks wall time; ``-progress`` fd
telemetry is the documented upgrade path for streaming (task-3401.11),
where stalls make wall time run ahead. Render when ``pts <= clock``,
drop-forward when video falls more than ``DRIFT_DROP_SECONDS`` behind,
and surface drift/dropped stats on the status line.

Seek restarts the pair with ``-ss`` (AC3); pause/resume uses
SIGSTOP/SIGCONT on POSIX (degrading to a no-op elsewhere, where pause
simply isn't offered -- the footer only names implemented actions).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess  # nosec B404 # pipelines invoke probed system binaries with fixed argv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


PLAYER_TARGET_FPS = 24.0
AUDIO_RATE = 44100
AUDIO_CHANNELS = 2
#: Video more than this far behind the sync clock is dropped forward.
DRIFT_DROP_SECONDS = 0.08
#: ffplay buffers a little internally; subtract to approximate true position.
AUDIO_BUFFER_LAG_SECONDS = 0.15

PLAYBACK_TOOLS_GUIDANCE = (
    "The in-app player needs ffmpeg and ffplay (e.g. 'brew install ffmpeg' "
    "or your distro's ffmpeg package). The video can still be opened with "
    "the system player."
)


def playback_tools_available() -> tuple[bool, str]:
    """Whether ffmpeg+ffplay are on PATH; returns (ok, guidance-if-missing)."""
    if shutil.which("ffmpeg") and shutil.which("ffplay"):
        return True, ""
    missing = [
        tool
        for tool in ("ffmpeg", "ffplay")
        if shutil.which(tool) is None
    ]
    return False, f"Missing {', '.join(missing)}. {PLAYBACK_TOOLS_GUIDANCE}"


@dataclass(frozen=True)
class PlayerProbe:
    """A source's playback shape from ffprobe (no frames decoded)."""

    width: int
    height: int
    duration_seconds: float | None
    has_audio: bool


def probe_file(source: str | Path) -> PlayerProbe:
    """Probe width/height/duration/audio-presence via ffprobe JSON."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(source),
    ]
    try:
        result = subprocess.run(  # nosec B603 # fixed argv, probed binary
            cmd, capture_output=True, text=True, timeout=30
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise RuntimeError(f"ffprobe failed for {source!r}") from exc
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe could not read {source!r}")
    data = json.loads(result.stdout or "{}")
    return parse_probe_json(data, source)


def parse_probe_json(data: dict[str, Any], source: str | Path) -> PlayerProbe:
    """Build a :class:`PlayerProbe` from ffprobe's JSON shape (pure)."""
    video_stream: dict[str, Any] | None = None
    has_audio = False
    for stream in data.get("streams", []):
        codec_type = stream.get("codec_type")
        if codec_type == "video" and video_stream is None:
            video_stream = stream
        elif codec_type == "audio":
            has_audio = True
    if video_stream is None:
        raise RuntimeError(f"no video stream found in {source!r}")
    duration: float | None = None
    raw_duration = (data.get("format") or {}).get("duration")
    if raw_duration is not None:
        try:
            duration = float(raw_duration)
        except (TypeError, ValueError):
            duration = None
    if duration is None and video_stream.get("duration") is not None:
        try:
            duration = float(video_stream["duration"])
        except (TypeError, ValueError):
            duration = None
    return PlayerProbe(
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        duration_seconds=duration,
        has_audio=has_audio,
    )


@dataclass
class SyncStats:
    """Live A/V sync counters for the player's status line."""

    rendered_frames: int = 0
    dropped_frames: int = 0
    last_drift_ms: float = 0.0
    max_drift_ms: float = 0.0
    position_seconds: float = 0.0


def _read_exact(stream: Any, size: int) -> bytes:
    """Read exactly ``size`` bytes (looping over short pipe reads) or b""."""
    parts: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return b""
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


class PlayerPipeline:
    """Owns one single-demux ffmpeg/ffplay pair and its sync bookkeeping.

    Driven from a worker thread: ``iter_frames`` blocks on the frame pipe
    and yields ``(pts, rgb24_bytes)`` at the CFR cadence; the screen renders
    each frame whose pts has come due per :meth:`frame_due`.
    ``seek``/``pause``/``resume``/``stop`` manage the process pair; every
    method is safe to call from any state.
    """

    def __init__(
        self,
        source: str | Path,
        probe: PlayerProbe,
        *,
        target_fps: float = PLAYER_TARGET_FPS,
        volume: int = 100,
        spawn=subprocess.Popen,
    ) -> None:
        self._source = str(source)
        self._probe = probe
        self._target_fps = float(target_fps)
        self._volume = volume
        self._spawn = spawn
        self._ffmpeg: subprocess.Popen | None = None
        self._ffplay: subprocess.Popen | None = None
        self._frame_index = 0
        self._offset_seconds = 0.0
        self._eof = False
        self._started_wall: float | None = None
        self._pause_started: float | None = None
        self._paused_total = 0.0
        #: Bumped by every start(); the frame pump treats a generation change
        #: as "seek happened -- re-enter", never as natural EOF.
        self._generation = 0
        self.stats = SyncStats()

    # -- lifecycle ---------------------------------------------------------

    def start(self, *, offset_seconds: float = 0.0) -> None:
        """Start (or restart, for seek) the demux pair at ``offset_seconds``."""
        self.stop()
        self._generation += 1
        self._frame_index = 0
        self._eof = False
        self._offset_seconds = max(0.0, offset_seconds)
        self._started_wall = None
        self._pause_started = None
        self._paused_total = 0.0
        audio_r, audio_w = os.pipe()
        seek_args = ["-ss", f"{self._offset_seconds:.3f}"] if self._offset_seconds else []
        ffmpeg_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            *seek_args,
            "-re",  # pace input at native frame rate (silent sources especially)
            "-i", self._source,
            # Video: CFR at the target fps so pts == index / fps exactly.
            "-map", "0:v:0",
            "-vf", f"fps={self._target_fps:g}",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "pipe:1",
        ]
        pass_fds: tuple[int, ...] = ()
        if self._probe.has_audio:
            ffmpeg_cmd += [
                "-map", "0:a:0",
                "-f", "s16le", "-ac", str(AUDIO_CHANNELS), "-ar", str(AUDIO_RATE),
                f"pipe:{audio_w}",
            ]
            pass_fds = (audio_w,)
        else:
            os.close(audio_w)
        self._ffmpeg = self._spawn(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            pass_fds=pass_fds,
        )
        if self._probe.has_audio:
            os.close(audio_w)  # the ffmpeg child holds its own copy now
            ffplay_cmd = [
                "ffplay", "-autoexit", "-nodisp", "-loglevel", "error",
                "-volume", str(self._volume),
                "-f", "s16le", "-ar", str(AUDIO_RATE), "-ac", str(AUDIO_CHANNELS),
                "-i", "pipe:0",
            ]
            self._ffplay = self._spawn(
                ffplay_cmd,
                stdin=audio_r,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        os.close(audio_r)

    # -- sync clock ---------------------------------------------------------

    @property
    def sync_clock(self) -> float:
        """Estimated playback position in seconds (pause-adjusted wall time)."""
        elapsed = 0.0 if self._started_wall is None else time.monotonic() - self._started_wall
        lag = AUDIO_BUFFER_LAG_SECONDS if self._probe.has_audio else 0.0
        return self._offset_seconds + elapsed - self._paused_total - lag

    def frame_due(self, pts: float) -> bool:
        """Whether a frame at ``pts`` should render now (pts <= clock)."""
        return pts <= self.sync_clock

    def frames_behind(self, pts: float) -> bool:
        """Whether a frame at ``pts`` is too far behind to bother rendering."""
        return pts < self.sync_clock - DRIFT_DROP_SECONDS

    def note_rendered(self, pts: float) -> None:
        """Record one rendered frame for the drift/dropped stats."""
        drift = pts - self.sync_clock
        self.stats.rendered_frames += 1
        self.stats.last_drift_ms = drift * 1000.0
        self.stats.max_drift_ms = max(self.stats.max_drift_ms, abs(drift) * 1000.0)
        self.stats.position_seconds = max(self.stats.position_seconds, pts)

    def note_dropped(self, pts: float) -> None:
        """Record one dropped (too-late-to-render) frame."""
        self.stats.dropped_frames += 1

    # -- frame pump ----------------------------------------------------------

    def iter_frames(self) -> Iterator[tuple[float, bytes]]:
        """Yield ``(pts, rgb24_frame_bytes)`` from the video pipe (blocking).

        Runs on the caller's worker thread; ends at EOF or when the pipeline
        is stopped.
        """
        if self._started_wall is None:
            self._started_wall = time.monotonic()
        frame_bytes = self._probe.width * self._probe.height * 3
        stdout = self._ffmpeg.stdout if self._ffmpeg else None
        if stdout is None:
            return
        while not self._eof:
            data = _read_exact(stdout, frame_bytes)
            if not data:
                self._eof = True
                break
            pts = self._offset_seconds + self._frame_index / self._target_fps
            self._frame_index += 1
            yield pts, data

    # -- control -------------------------------------------------------------

    def seek(self, offset_seconds: float) -> None:
        """Restart the demux pair at ``offset_seconds`` (AC3)."""
        duration = self._probe.duration_seconds
        if duration is not None:
            offset_seconds = min(offset_seconds, max(0.0, duration))
        self.start(offset_seconds=offset_seconds)

    def pause(self) -> None:
        """Freeze both processes and stop the sync clock (POSIX only)."""
        if self._pause_started is not None:
            return
        self._pause_started = time.monotonic()
        if hasattr(signal, "SIGSTOP"):
            for proc in (self._ffmpeg, self._ffplay):
                if proc is not None and proc.poll() is None:
                    try:
                        os.kill(proc.pid, signal.SIGSTOP)
                    except (OSError, ProcessLookupError):
                        pass

    def resume(self) -> None:
        """Unfreeze both processes; paused wall time is folded out of the clock."""
        if self._pause_started is None:
            return
        self._paused_total += time.monotonic() - self._pause_started
        self._pause_started = None
        if hasattr(signal, "SIGCONT"):
            for proc in (self._ffplay, self._ffmpeg):
                if proc is not None and proc.poll() is None:
                    try:
                        os.kill(proc.pid, signal.SIGCONT)
                    except (OSError, ProcessLookupError):
                        pass

    def stop(self) -> None:
        """Terminate the pair (idempotent)."""
        for attr in ("_ffplay", "_ffmpeg"):
            proc = getattr(self, attr)
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                setattr(self, attr, None)
        self._eof = True

    @property
    def at_eof(self) -> bool:
        return self._eof
