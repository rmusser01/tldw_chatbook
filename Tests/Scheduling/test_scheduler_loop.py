"""Tests for the SchedulerLoop."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop


@pytest.mark.asyncio
async def test_scheduler_triggers_due_reminder(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    db.create_reminder_task(
        owner_id="local",
        title="Test",
        schedule_kind="one_time",
        next_run_at="2026-01-01T00:00:00+00:00",
    )
    handler = AsyncMock()
    loop = SchedulerLoop(
        db,
        handlers={"reminder": handler},
        clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    await loop.tick()
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_scheduler_does_not_trigger_future_reminder(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    db.create_reminder_task(
        owner_id="local",
        title="Future",
        schedule_kind="one_time",
        next_run_at="2026-01-02T00:00:00+00:00",
    )
    handler = AsyncMock()
    loop = SchedulerLoop(
        db,
        handlers={"reminder": handler},
        clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    await loop.tick()
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduler_ignores_disabled_reminder(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    db.create_reminder_task(
        owner_id="local",
        title="Disabled",
        schedule_kind="one_time",
        next_run_at="2026-01-01T00:00:00+00:00",
        enabled=False,
    )
    handler = AsyncMock()
    loop = SchedulerLoop(
        db,
        handlers={"reminder": handler},
        clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    await loop.tick()
    handler.assert_not_awaited()
