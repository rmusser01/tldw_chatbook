"""Read-only projection of Watchlist/Subscription jobs into scheduled tasks."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from tldw_chatbook.Scheduling.models import ScheduledTask, TaskStatus


def _parse_iso_timestamp(value: str | datetime | None) -> datetime | None:
    """Normalize a subscription timestamp value to a timezone-aware datetime."""
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


def _compute_next_run(
    last_checked: str | datetime | None,
    check_frequency: int | None,
    created_at: str | datetime | None,
) -> datetime | None:
    """Compute the next expected run time from the subscription cadence."""
    base = _parse_iso_timestamp(last_checked)
    if base is None:
        base = _parse_iso_timestamp(created_at)
    if base is None or not check_frequency:
        return None
    return base + timedelta(seconds=int(check_frequency))


def _build_schedule_summary(check_frequency: int | None) -> str | None:
    """Return a human-readable summary of the subscription check cadence."""
    if check_frequency is None:
        return None
    seconds = int(check_frequency)
    if seconds < 60:
        return f"Every {seconds}s"
    if seconds < 3600:
        return f"Every {seconds // 60}m"
    if seconds < 86400:
        return f"Every {seconds // 3600}h"
    return f"Every {seconds // 86400}d"


def _resolve_status(row: dict[str, Any]) -> TaskStatus:
    """Map a subscription row to the unified task status enum."""
    if not row.get("is_active"):
        return TaskStatus.DISABLED
    if row.get("is_paused"):
        return TaskStatus.PAUSED
    if row.get("last_error") and not row.get("last_successful_check"):
        return TaskStatus.NEEDS_ATTENTION
    return TaskStatus.WAITING


class WatchlistProjection:
    """Project Subscriptions_DB rows into lightweight ScheduledTask objects."""

    def __init__(self, subscriptions_db):
        self.subscriptions_db = subscriptions_db

    def list_jobs(self, owner_id: str = "local") -> list[ScheduledTask]:
        """Read subscriptions from Subscriptions_DB and project them as scheduled tasks."""
        rows = self.subscriptions_db.get_all_subscriptions(include_inactive=True)
        return [self._to_scheduled_task(row, owner_id) for row in rows]

    def _to_scheduled_task(self, row: dict[str, Any], owner_id: str) -> ScheduledTask:
        """Map a single Subscriptions_DB row to a ScheduledTask model."""
        subscription_id = row.get("id")
        return ScheduledTask(
            id=f"watchlist:{subscription_id}",
            title=row.get("name") or f"Watchlist {subscription_id}",
            type="watchlist_job",
            status=_resolve_status(row),
            schedule_summary=_build_schedule_summary(row.get("check_frequency")),
            next_run_at=_compute_next_run(
                row.get("last_checked"),
                row.get("check_frequency"),
                row.get("created_at"),
            ),
            owner_id=owner_id,
            source=row.get("source") or row.get("type"),
        )
