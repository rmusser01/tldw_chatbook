# scheduler_heartbeat.py
"""TASK-26025: durable scheduler liveness heartbeat.

A small atomic JSON file the scheduler loop rewrites each tick, holding the
last tick/success timestamps, the last error, the poll interval, and the
tick count. It is readable by any surface (or the next process) so a dead
loop is distinguishable from an idle one -- and, crucially, from one that
never started (no file at all). Staleness is judged against the poll
interval so a deliberately long interval is not mistaken for a stall.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger

#: Liveness verdict is stale once the last tick is older than the poll
#: interval times this factor -- a couple of missed polls, not one late one.
_STALENESS_FACTOR = 3.0

#: A floor so a very short poll interval still allows normal scheduling jitter
#: before the loop is called stalled.
_MIN_STALENESS_WINDOW_SECONDS = 90.0

SchedulerLiveness = str  # "never_started" | "live" | "stale"


@dataclass(frozen=True, slots=True)
class SchedulerHeartbeat:
    """One durable snapshot of scheduler liveness."""

    last_tick_at: datetime | None
    last_success_at: datetime | None = None
    last_error: str | None = None
    poll_interval: float = 30.0
    tick_count: int = 0


def classify_scheduler_liveness(
    heartbeat: SchedulerHeartbeat | None,
    *,
    now: datetime,
    poll_interval: float,
) -> SchedulerLiveness:
    """Classify liveness from a heartbeat and the current time.

    ``None`` (or a heartbeat with no tick recorded) is ``never_started``
    (AC#6). Otherwise the last tick's age is compared against a window
    scaled by ``poll_interval`` (AC#4): within the window is ``live``,
    beyond it ``stale``.
    """
    if heartbeat is None or heartbeat.last_tick_at is None:
        return "never_started"
    window = max(
        _MIN_STALENESS_WINDOW_SECONDS,
        max(0.0, float(poll_interval)) * _STALENESS_FACTOR,
    )
    age = (now - heartbeat.last_tick_at).total_seconds()
    return "stale" if age > window else "live"


def read_heartbeat(path: Path) -> SchedulerHeartbeat | None:
    """Load a heartbeat, or None when absent/corrupt."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None

    def _dt(value: object) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    return SchedulerHeartbeat(
        last_tick_at=_dt(raw.get("last_tick_at")),
        last_success_at=_dt(raw.get("last_success_at")),
        last_error=raw.get("last_error") if isinstance(raw.get("last_error"), str) else None,
        poll_interval=float(raw.get("poll_interval", 30.0) or 30.0),
        tick_count=int(raw.get("tick_count", 0) or 0),
    )


def write_heartbeat(path: Path, heartbeat: SchedulerHeartbeat) -> None:
    """Atomically persist a heartbeat. Never raises -- a diagnostics write
    must never break the loop it observes."""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "last_tick_at": heartbeat.last_tick_at.isoformat()
                if heartbeat.last_tick_at
                else None,
                "last_success_at": heartbeat.last_success_at.isoformat()
                if heartbeat.last_success_at
                else None,
                "last_error": heartbeat.last_error,
                "poll_interval": heartbeat.poll_interval,
                "tick_count": heartbeat.tick_count,
            }
        )
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".heartbeat-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp, path)
        except Exception:
            with open(os.devnull, "w"):
                pass
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception as exc:  # noqa: BLE001 -- observation never breaks the loop
        logger.debug(f"scheduler heartbeat write failed: {exc!r}")


def _humanize_age(seconds: float) -> str:
    if seconds < 90:
        return f"{int(seconds)}s ago"
    if seconds < 5400:
        return f"{int(seconds // 60)}m ago"
    return f"{seconds / 3600:.1f}h ago"


def scheduler_liveness_line(
    heartbeat: SchedulerHeartbeat | None,
    *,
    now: datetime,
    poll_interval: float,
) -> str:
    """A one-line human liveness summary for the Schedules surface (AC#2).

    The three states read distinctly (AC#6): not-started, live, and
    stalled -- and a stall carries the retained last error (AC#3).
    """
    state = classify_scheduler_liveness(
        heartbeat, now=now, poll_interval=poll_interval
    )
    if state == "never_started":
        return "Scheduler: not started"
    assert heartbeat is not None and heartbeat.last_tick_at is not None
    age = _humanize_age(max(0.0, (now - heartbeat.last_tick_at).total_seconds()))
    if state == "live":
        return f"Scheduler: live · last tick {age}"
    error = f" · last error: {heartbeat.last_error}" if heartbeat.last_error else ""
    return f"Scheduler: STALLED — last tick {age}{error}"


def default_heartbeat_path() -> Path:
    from tldw_chatbook.config import get_user_data_dir

    return get_user_data_dir() / "scheduler_heartbeat.json"
