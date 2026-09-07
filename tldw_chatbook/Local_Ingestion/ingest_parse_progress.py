"""Spawn-safe, best-effort progress telemetry for local ingest parsing."""

from __future__ import annotations

from dataclasses import dataclass
import math
import queue
from typing import Any


INGEST_PARSE_PROGRESS_MESSAGE_MAX_CHARS = 160
"""Maximum message length retained in an emitted progress event."""

INGEST_PARSE_PROGRESS_JOB_ID_MAX_CHARS = 64
"""Maximum accepted job identity length; oversized IDs are rejected, never truncated."""

INGEST_PARSE_PROGRESS_GENERATION_MIN = 0
"""Smallest accepted ingest generation identifier."""

INGEST_PARSE_PROGRESS_GENERATION_MAX = 2**63 - 1
"""Largest accepted ingest generation identifier, safe for signed 64-bit IPC peers."""

INGEST_PARSE_PROGRESS_FLUSH_SECONDS = 0.25
INGEST_PARSE_PROGRESS_QUEUE_MAXSIZE = 64
INGEST_PARSE_PROGRESS_PHASES = frozenset(
    {
        "inspecting",
        "extracting",
        "processing",
        "transcribing",
        "chunking",
        "analyzing",
        "preparing",
        "loading",
        "post-processing",
        "writing",
    }
)


@dataclass(frozen=True, slots=True)
class ParseProgressEvent:
    """A small, picklable snapshot of local ingest parse progress."""

    generation: int
    job_id: str
    phase: str
    message: str
    percent: float | None = None


_progress_queue: Any | None = None


def _normalize_text(value: object, *, max_chars: int | None = None) -> str:
    """Return printable, single-line text suitable for an IPC event."""
    text = "" if value is None else str(value)
    normalized = "".join(
        character for character in text if character.isprintable() or character.isspace()
    ).strip()
    normalized = " ".join(normalized.split())
    if max_chars is not None:
        return normalized[:max_chars]
    return normalized


def _normalize_percent(value: object) -> float | None:
    """Return a truthful percentage, or omit malformed progress values."""
    if isinstance(value, bool):
        return None
    try:
        percent = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(percent) or not 0.0 <= percent <= 100.0:
        return None
    return percent


def make_parse_progress_event(
    generation: int,
    job_id: str,
    phase: str,
    message: str,
    percent: float | None = None,
) -> ParseProgressEvent | None:
    """Create an IPC-safe event within the public progress contract.

    Args:
        generation: Owning parse-pool generation.
        job_id: Ingest job identifier.
        phase: Controlled progress phase.
        message: Human-readable progress detail.
        percent: Optional measured percentage from zero through one hundred.

    Returns:
        A normalized progress event, or ``None`` when identity or phase data
        falls outside the public contract.
    """
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or not INGEST_PARSE_PROGRESS_GENERATION_MIN
        <= generation
        <= INGEST_PARSE_PROGRESS_GENERATION_MAX
    ):
        return None

    normalized_job_id = _normalize_text(job_id)
    normalized_phase = _normalize_text(phase)
    if (
        not normalized_job_id
        or len(normalized_job_id) > INGEST_PARSE_PROGRESS_JOB_ID_MAX_CHARS
        or normalized_phase not in INGEST_PARSE_PROGRESS_PHASES
    ):
        return None

    return ParseProgressEvent(
        generation=generation,
        job_id=normalized_job_id,
        phase=normalized_phase,
        message=_normalize_text(message, max_chars=INGEST_PARSE_PROGRESS_MESSAGE_MAX_CHARS),
        percent=_normalize_percent(percent),
    )


def install_parse_progress_sink(progress_queue: Any | None) -> None:
    """Install the worker-local non-blocking progress sink.

    Args:
        progress_queue: Queue-like sink supporting ``put_nowait``, or ``None``
            to disable worker progress emission.

    Returns:
        None.
    """
    global _progress_queue
    _progress_queue = progress_queue


def emit_parse_progress(
    generation: int,
    job_id: str,
    phase: str,
    message: str,
    percent: float | None = None,
) -> None:
    """Best-effort emit one validated progress snapshot without blocking.

    Args:
        generation: Owning parse-pool generation.
        job_id: Ingest job identifier.
        phase: Controlled progress phase.
        message: Human-readable progress detail.
        percent: Optional measured percentage from zero through one hundred.

    Returns:
        None. Invalid events and unavailable or saturated sinks are ignored.
    """
    event = make_parse_progress_event(generation, job_id, phase, message, percent)
    if event is None or _progress_queue is None:
        return
    try:
        _progress_queue.put_nowait(event)
    except (queue.Full, BrokenPipeError, EOFError, OSError, ValueError):
        return


class ParseProgressCoalescer:
    """Retain the newest event per job and flush on a fixed minimum interval."""

    def __init__(self, *, interval: float, started_at: float) -> None:
        self._interval = interval
        self._next_due = started_at + interval
        self._pending: dict[str, ParseProgressEvent] = {}

    def accept(self, event: ParseProgressEvent) -> None:
        """Retain the newest event for an ingest job."""
        self._pending[event.job_id] = event

    def take_due(
        self, now: float, *, force: bool = False
    ) -> tuple[ParseProgressEvent, ...]:
        """Return due events in job-id order and schedule the next deadline."""
        if not force and now < self._next_due:
            return ()

        events = tuple(self._pending[job_id] for job_id in sorted(self._pending))
        self._pending.clear()
        self._next_due = now + self._interval
        return events
