"""Frame sources for video playback (task-3401.9).

``VideoFrameSource`` is the seam the preview widget drives: probe a file's
shape without decoding, then iterate frames as PIL images throttled to a
target fps. The only production implementation today is ``AvFrameSource``
(PyAV); tests inject fakes through the same protocol, and task-3401.10's
ffmpeg pipe source lands behind it too.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

from loguru import logger

from tldw_chatbook.Media_Playback.preview_policy import (
    PREVIEW_TARGET_FPS,
    check_preview_eligibility,
)


@dataclass(frozen=True)
class VideoProbe:
    """A video file's shape, probed WITHOUT decoding frames."""

    duration_seconds: float | None
    width: int | None
    height: int | None
    fps: float | None


class VideoFrameSource(Protocol):
    """Protocol for playable video frame sources."""

    def probe(self) -> VideoProbe:
        """Return the source's shape without decoding."""
        ...

    def iter_frames(
        self, *, target_fps: float = PREVIEW_TARGET_FPS
    ) -> Iterator[tuple[float, Any]]:
        """Yield ``(timestamp_seconds, PIL.Image)`` throttled to ``target_fps``.

        Runs on a worker thread; implementations must tolerate being
        abandoned mid-iteration (pause/stop closes the generator).
        """
        ...

    def close(self) -> None:
        """Release the source (idempotent)."""
        ...


class AvFrameSource:
    """PyAV-backed frame source for local files (av-gated).

    Decodes on the caller's thread -- the preview widget runs the iterator
    on a Textual worker, never on the UI loop. Frame drops: when decode
    falls behind the wall clock, upcoming frames are skipped (not queued),
    so a slow machine plays choppy-but-current instead of lagging further
    behind with every frame (the preview is a glance, not a reference
    player).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._container: Any | None = None
        self._stream: Any | None = None
        self._opened = False

    def _open(self) -> None:
        if self._opened:
            return
        import av  # optional dependency -- probed by the caller first

        self._container = av.open(str(self._path))
        self._stream = self._container.streams.video[0]
        self._opened = True

    def probe(self) -> VideoProbe:
        self._open()
        container = self._container
        stream = self._stream
        duration: float | None = None
        if container is not None and container.duration:
            duration = float(container.duration) / 1_000_000.0
        elif stream is not None and stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        fps: float | None = None
        if stream is not None:
            try:
                rate = stream.average_rate
                if rate:
                    fps = float(rate)
            except Exception:
                fps = None
        return VideoProbe(
            duration_seconds=duration,
            width=getattr(stream, "width", None),
            height=getattr(stream, "height", None),
            fps=fps,
        )

    def check_eligible(self) -> tuple[bool, str]:
        """Re-check the preview caps against the PROBED shape (never metadata).

        Returns:
            ``(eligible, reason)`` -- the reason names the failed cap.
        """
        probe = self.probe()
        eligibility = check_preview_eligibility(
            duration_seconds=probe.duration_seconds,
            width=probe.width,
            height=probe.height,
        )
        return eligibility.eligible, eligibility.reason

    def iter_frames(
        self, *, target_fps: float = PREVIEW_TARGET_FPS
    ) -> Iterator[tuple[float, Any]]:
        self._open()
        container = self._container
        if container is None:
            return
        interval = 1.0 / max(1.0, float(target_fps))
        started = time.monotonic()
        next_emit = 0.0
        try:
            for frame in container.decode(video=0):
                timestamp = (
                    float(frame.pts * frame.time_base) if frame.pts is not None else 0.0
                )
                now = time.monotonic() - started
                if now < next_emit:
                    # Decode outruns the wall clock: pace by sleeping until the
                    # next emission slot (real-time playback, not fast-forward).
                    time.sleep(next_emit - now)
                elif now > next_emit + interval:
                    # Decode fell behind: drop forward to the current slot
                    # rather than queuing stale frames (choppy-but-current).
                    next_emit = now
                image = frame.to_image()  # PIL.Image
                yield timestamp, image
                next_emit += interval
        except GeneratorExit:
            # Pause/stop abandoned the iterator mid-stream -- normal path.
            logger.debug("AvFrameSource: iteration abandoned (pause/stop)")
            raise

    def close(self) -> None:
        container = self._container
        self._container = None
        self._stream = None
        self._opened = False
        if container is not None:
            container.close()
