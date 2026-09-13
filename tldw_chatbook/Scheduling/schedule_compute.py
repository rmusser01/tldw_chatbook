"""Schedule computation for the five reference schedule kinds.

Pure module (no I/O, no imports of other Scheduling submodules): given a
schedule dict and the current time, compute the next UTC run time. Never
raises -- a malformed schedule row must not kill the queue; invalid/junk
input yields ``None`` (logged at debug).

Advance-from-now semantics (spec Sec 4.3 / TASK-18937 discipline): elapsed
slots are skipped, never replayed. For example an ``interval`` schedule's
next run is always ``now + every_seconds``, never a replay of a slot that
has already passed.

Supported kinds (matches the server's ``_SUPPORTED_SCHEDULE_KINDS``):

- ``one_time``: ``run_at`` (ISO datetime string).
- ``interval``: ``every_seconds`` (int, floor 60).
- ``daily``: ``time_of_day`` ("HH:MM"), optional IANA ``timezone``
  (defaults to the machine's local zone).
- ``weekly``: ``daily`` fields plus ``weekday`` (0-6, 0=Monday).
- ``cron``: ``cron`` (5-field cron expression, via ``croniter``), optional
  IANA ``timezone`` (defaults to UTC).
"""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter
from loguru import logger

_MIN_INTERVAL_SECONDS = 60


def _naive_as_utc(dt: datetime) -> datetime:
    """Naive datetimes are assumed UTC, matching ``_to_utc_iso``'s discipline."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _resolve_timezone(name: Any, *, default: Any) -> Any:
    if not name:
        return default
    try:
        return ZoneInfo(str(name))
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        logger.debug(f"schedule_compute: unknown timezone {name!r}, falling back to UTC")
        return timezone.utc


def _parse_time_of_day(value: Any) -> time | None:
    if not isinstance(value, str):
        return None
    parts = value.split(":")
    if len(parts) != 2:
        return None
    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return time(hour=hour, minute=minute)


def _compute_one_time(schedule: dict[str, Any], now: datetime) -> datetime | None:
    run_at = schedule.get("run_at")
    if not isinstance(run_at, str):
        return None
    try:
        parsed = datetime.fromisoformat(run_at)
    except ValueError:
        return None
    parsed = _naive_as_utc(parsed).astimezone(timezone.utc)
    return parsed if parsed > now else None


def _compute_interval(schedule: dict[str, Any], now: datetime) -> datetime | None:
    every_seconds = schedule.get("every_seconds")
    if isinstance(every_seconds, bool) or not isinstance(every_seconds, (int, float)):
        return None
    if every_seconds < _MIN_INTERVAL_SECONDS:
        return None
    return now + timedelta(seconds=every_seconds)


def _local_now_for_schedule(schedule: dict[str, Any], now: datetime) -> datetime:
    """Local "now" to build daily/weekly candidates from.

    An explicit IANA ``timezone`` resolves through ``ZoneInfo`` as before
    (fold-aware, DST-correct for that zone). With no explicit timezone
    (Finding E), this returns a NAIVE local wall-clock value instead of
    localizing through a single fixed-offset snapshot (the old
    ``_machine_timezone()``, which memoized ``datetime.now().astimezone()
    .tzinfo`` -- wrong across a DST boundary, since every computed
    candidate got that one snapshot's offset regardless of the
    candidate's own date). A naive datetime's own ``.astimezone()`` (see
    ``_localize_after`` below) instead resolves the platform's local rule
    per the date it actually carries.
    """
    explicit_tz_name = schedule.get("timezone")
    if explicit_tz_name:
        return now.astimezone(_resolve_timezone(explicit_tz_name, default=timezone.utc))
    return now.astimezone().replace(tzinfo=None)


def _localize_after(candidate: datetime, now: datetime, period: timedelta) -> datetime:
    """Localize ``candidate`` (aware or naive-local) to UTC.

    Guards against DST fall-back computing a past instant (Finding F):
    Python's aware-datetime arithmetic always resets ``fold`` to 0 on its
    result (documented behavior), which can silently flip a
    correctly-resolved second-occurrence (``fold=1``) candidate to the
    first occurrence's interpretation after a ``candidate += timedelta``
    -- up to an hour earlier than intended. If the localized result is
    not strictly after ``now``, advance one more ``period`` (a day or a
    week) and recompute rather than ever handing back a stale slot.
    """
    result = candidate.astimezone(timezone.utc)
    if result <= now:
        result = (candidate + period).astimezone(timezone.utc)
    return result


def _compute_daily(schedule: dict[str, Any], now: datetime) -> datetime | None:
    tod = _parse_time_of_day(schedule.get("time_of_day"))
    if tod is None:
        return None
    local_now = _local_now_for_schedule(schedule, now)
    candidate = local_now.replace(hour=tod.hour, minute=tod.minute, second=0, microsecond=0)
    if candidate <= local_now:
        candidate += timedelta(days=1)
    return _localize_after(candidate, now, timedelta(days=1))


def _compute_weekly(schedule: dict[str, Any], now: datetime) -> datetime | None:
    tod = _parse_time_of_day(schedule.get("time_of_day"))
    weekday = schedule.get("weekday")
    if tod is None or isinstance(weekday, bool) or not isinstance(weekday, int) or not (0 <= weekday <= 6):
        return None
    local_now = _local_now_for_schedule(schedule, now)
    candidate = local_now.replace(hour=tod.hour, minute=tod.minute, second=0, microsecond=0)
    days_ahead = (weekday - candidate.weekday()) % 7
    candidate += timedelta(days=days_ahead)
    if candidate <= local_now:
        candidate += timedelta(days=7)
    return _localize_after(candidate, now, timedelta(days=7))


def _compute_cron(schedule: dict[str, Any], now: datetime) -> datetime | None:
    cron_expr = schedule.get("cron")
    if not isinstance(cron_expr, str) or not cron_expr.strip():
        return None
    tz = _resolve_timezone(schedule.get("timezone"), default=timezone.utc)
    local_now = now.astimezone(tz)
    try:
        next_run = croniter(cron_expr, local_now).get_next(datetime)
    except (ValueError, KeyError):
        logger.debug(f"schedule_compute: invalid cron expression {cron_expr!r}")
        return None
    return next_run.astimezone(timezone.utc)


_KIND_COMPUTERS = {
    "one_time": _compute_one_time,
    "interval": _compute_interval,
    "daily": _compute_daily,
    "weekly": _compute_weekly,
    "cron": _compute_cron,
}


def compute_next_run_at(schedule: dict[str, Any], *, now: datetime) -> datetime | None:
    """Compute the next UTC run time for a schedule dict.

    Advance-from-now semantics (spec Sec 4.3): elapsed slots are skipped,
    never replayed. Returns ``None`` for a spent ``one_time`` schedule or
    any invalid/junk schedule -- this never raises, since a bad row must
    not take down the queue.

    Nonexistent local times (the spring-forward gap -- e.g. a ``daily``/
    ``weekly`` ``time_of_day`` that falls in the hour skipped when clocks
    jump forward) resolve FORWARD per ``zoneinfo``/PEP 495 fold semantics:
    02:30 becomes 03:30 that same day, matching common cron behavior. This
    is deliberate and unchanged (Finding G) -- not a case this function
    treats as invalid or shifts to a different day.

    Args:
        schedule: A schedule dict with a ``kind`` key (one of
            ``_KIND_COMPUTERS``: ``one_time``, ``interval``, ``daily``,
            ``weekly``, ``cron``) plus that kind's own fields.
        now: The current time, used as the basis to advance from.

    Returns:
        The next UTC run time, or ``None`` if the schedule is spent,
        malformed, or of an unrecognized ``kind``.
    """
    if not isinstance(schedule, dict):
        return None
    computer = _KIND_COMPUTERS.get(schedule.get("kind"))
    if computer is None:
        return None
    try:
        return computer(schedule, now)
    except Exception:
        logger.debug(f"schedule_compute: failed to compute next run for {schedule!r}", exc_info=True)
        return None


def schedule_slot_for(next_run: datetime) -> str:
    """Canonical UTC ISO string used as a run's ``schedule_slot``.

    Args:
        next_run: The next scheduled run time. Every caller in this
            codebase passes an aware datetime (this module's own
            computers always return one); a naive value here would be
            interpreted as system-local time by ``astimezone()``, not UTC.

    Returns:
        ``next_run`` converted to UTC and rendered as an ISO 8601 string --
        the value stored as a run's ``schedule_slot`` and used in its
        ``(definition_id, definition_version, schedule_slot)`` UNIQUE.
    """
    return next_run.astimezone(timezone.utc).isoformat()
