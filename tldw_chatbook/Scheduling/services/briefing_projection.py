"""Read-only projection of per-watchlist scheduled briefing jobs.

Mirrors `watchlist_projection.py`'s shape: one `ScheduledTask` per row a
DB-layer lister returns, fed into `PriorityQueue` by `app.py`. The source
here is `SubscriptionsDB.list_briefing_schedules` (briefings phase 4, task
2) rather than `get_all_subscriptions` -- one row per watchlist with a
non-NULL `briefing_cadence_seconds` (Locked Decision 4: scheduled briefings
are opt-in, off by default), each already carrying `last_completed_at`
computed with the same `status IN ('complete', 'empty')` allowlist as
`latest_completed_watermark` -- a failed or still-`generating` briefing
never advances the schedule (the completion watermark, unchanged).

`next_run_at`, however, is attempt-aware (whole-branch review FIX 1): it
also reads the row's status-blind `last_attempt_at` and uses whichever of
`last_completed_at`/`last_attempt_at` is later. A schedule with a
completion history but a MORE RECENT failure must retry one cadence period
after that failure, not stay pinned to the stale completion -- the prior
behavior left `next_run_at` in the past for any watchlist whose most
recent run failed, so every ~60-tick queue reload (~30 min) re-emitted the
job, uncapped, forever. Never-attempted (`last_completed_at` and
`last_attempt_at` both `None`) is still due now.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Scheduling.models import ScheduledTask, TaskStatus

#: The one place this project's briefing scheduled-task ids are built AND
#: parsed. `watchlist_projection.py`/`watchlist_check_handler.py` hold two
#: independent copies of the "watchlist:" prefix -- one hardcoded literal in
#: each module -- that happen to agree today but could silently drift apart
#: (the 2b lesson this module exists not to repeat). `briefing_handler.py`
#: imports `BRIEFING_TASK_PREFIX` and `parse_briefing_task_id` from here
#: rather than holding its own copy, so there is exactly one definition of
#: the id shape for both the write side (below) and the read side.
BRIEFING_TASK_PREFIX = "briefing"


def parse_briefing_task_id(task_id: Any) -> int | None:
    """Extract the numeric watchlist id from a `briefing:<id>` task id.

    The one parser for this id shape -- see `BRIEFING_TASK_PREFIX`'s own
    docstring for why it lives here rather than being reimplemented in
    `briefing_handler.py`.

    Args:
        task_id: The scheduled task's `id` field, expected to look like
            `"briefing:<watchlist_id>"`.

    Returns:
        The parsed watchlist id, or `None` if `task_id` is not a string,
        has no `:`, has the wrong prefix, or the suffix is not an integer.
    """
    if not isinstance(task_id, str) or ":" not in task_id:
        return None
    prefix, raw_id = task_id.split(":", 1)
    if prefix != BRIEFING_TASK_PREFIX:
        return None
    try:
        return int(raw_id)
    except ValueError:
        return None


def _parse_iso_timestamp(value: str | datetime | None) -> datetime | None:
    """Normalize a `last_completed_at` value to a timezone-aware datetime.

    `SubscriptionsDB` timestamps come back as naive `CURRENT_TIMESTAMP`
    strings (`"2020-01-01 00:00:00"`, no `T`, no offset), which
    `datetime.fromisoformat` accepts as of Python 3.11. This mirrors
    `watchlist_projection._parse_iso_timestamp` byte for byte -- kept as
    its own small copy rather than importing a private helper across an
    otherwise-unrelated module, since the two-copies-drift lesson this
    module exists to avoid is specifically about the id prefix/parser (an
    explicit, named, tested contract), not this generic timestamp shim.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    try:
        parsed = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class BriefingProjection:
    """Project `SubscriptionsDB.list_briefing_schedules` rows into tasks."""

    def __init__(self, subscriptions_db: SubscriptionsDB) -> None:
        """Initialize the projection.

        Args:
            subscriptions_db: The `SubscriptionsDB` `list_briefing_schedules`
                is read from.
        """
        self.subscriptions_db = subscriptions_db

    def list_jobs(
        self, owner_id: str = "local", *, now: datetime | None = None
    ) -> list[ScheduledTask]:
        """Read scheduled watchlists and project them as scheduled tasks.

        Args:
            owner_id: Owner to stamp on every emitted task.
            now: Injected clock for a never-briefed watchlist's "due now"
                run time. Defaults to the current UTC time; tests pass a
                fixed value so "due now" is assertable by equality.

        Returns:
            One `ScheduledTask` per watchlist `list_briefing_schedules`
            returns (non-NULL cadence only -- an un-cadenced watchlist
            never appears here at all).
        """
        current = now if now is not None else datetime.now(timezone.utc)
        rows = self.subscriptions_db.list_briefing_schedules()
        return [self._to_scheduled_task(row, owner_id, current) for row in rows]

    def _to_scheduled_task(
        self, row: dict[str, Any], owner_id: str, now: datetime
    ) -> ScheduledTask:
        """Map a single `list_briefing_schedules` row to a `ScheduledTask`.

        `next_run_at` is `max(last_completed_at, last_attempt_at) +
        cadence` -- attempt-aware, not completion-only (whole-branch review
        FIX 1). `last_attempt_at` is status-blind (failed/generating
        included), so a failure that is more recent than the last
        completion pushes the next run one cadence period past the
        FAILURE, rather than leaving `next_run_at` frozen at the stale
        completion (which the queue's ~30-minute reload cycle would then
        re-emit every single time, uncapped). A watchlist that has never
        had any attempt at all (`None` for both) is due right now, rather
        than at some indefinitely deferred time -- an opted-in schedule
        with no history should fire on the next tick, not wait a full
        cadence period.

        Args:
            row: One `list_briefing_schedules` row: `watchlist_id`, `name`,
                `briefing_cadence_seconds`, `last_completed_at`, and
                `last_attempt_at`.
            owner_id: Owner to stamp on the emitted task.
            now: The "due now" fallback used when the row has neither a
                completion nor an attempt on record yet.

        Returns:
            The row projected into a `ScheduledTask`.
        """
        watchlist_id = row["watchlist_id"]
        cadence_seconds = int(row["briefing_cadence_seconds"])
        last_completed = _parse_iso_timestamp(row.get("last_completed_at"))
        last_attempt = _parse_iso_timestamp(row.get("last_attempt_at"))
        last_activity = max(
            (dt for dt in (last_completed, last_attempt) if dt is not None),
            default=None,
        )
        next_run_at = (
            last_activity + timedelta(seconds=cadence_seconds)
            if last_activity is not None
            else now
        )
        return ScheduledTask(
            id=f"{BRIEFING_TASK_PREFIX}:{watchlist_id}",
            title=row.get("name") or f"Watchlist {watchlist_id}",
            type="briefing_job",
            status=TaskStatus.WAITING,
            next_run_at=next_run_at,
            owner_id=owner_id,
        )
