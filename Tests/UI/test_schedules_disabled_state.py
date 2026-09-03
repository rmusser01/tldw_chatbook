"""task-23101: a disabled task must read as Disabled, not Waiting.

``_task_status`` used to return ``last_status``, which disabling never
touches, so a disabled task kept showing "Waiting" with a concrete
future Next Run in both the queue row and the detail badge.
"""

from datetime import datetime, timezone

import pytest
from textual.widgets import DataTable, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Scheduling.models import (
    ReminderTask,
    ScheduledTask,
    ScheduleKind,
    TaskStatus,
)
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import (
    SchedulesWorkbench,
)
from tldw_chatbook.UI.Screens.scheduling.task_detail import (
    _format_next_run,
    _task_status,
)


_RUN_AT = datetime(2099, 7, 20, 14, 0, tzinfo=timezone.utc)


def _reminder(*, enabled: bool, last_status=TaskStatus.WAITING) -> ReminderTask:
    return ReminderTask(
        id="task-d",
        title="Backup",
        schedule_kind=ScheduleKind.ONE_TIME,
        run_at=_RUN_AT,
        next_run_at=_RUN_AT,
        enabled=enabled,
        last_status=last_status,
    )


# --- derivation unit tests -------------------------------------------------


def test_disabled_reminder_status_is_disabled():
    assert _task_status(_reminder(enabled=False)) is TaskStatus.DISABLED


def test_enabled_reminder_keeps_last_status():
    assert (
        _task_status(_reminder(enabled=True, last_status=TaskStatus.MISSED))
        is TaskStatus.MISSED
    )


def test_projection_status_unaffected():
    projection = ScheduledTask(
        id="watchlist:1",
        title="Watch",
        type="watchlist_job",
        status=TaskStatus.WAITING,
        next_run_at=_RUN_AT,
    )
    assert _task_status(projection) is TaskStatus.WAITING
    assert "2099" in _format_next_run(projection)


def test_disabled_reminder_next_run_shows_no_concrete_time():
    rendered = _format_next_run(_reminder(enabled=False))
    assert "2099" not in rendered
    assert "disabled" in rendered.lower()


def test_enabled_reminder_next_run_shows_time():
    assert "2099-07-20" in _format_next_run(_reminder(enabled=True))


# --- review F6: suppression covers DISABLED/PAUSED projections too --------


def _projection(status: TaskStatus) -> ScheduledTask:
    return ScheduledTask(
        id="watchlist:9",
        title="Watch",
        type="watchlist_job",
        status=status,
        next_run_at=_RUN_AT,
    )


def test_disabled_projection_next_run_is_suppressed():
    rendered = _format_next_run(_projection(TaskStatus.DISABLED))
    assert "2099" not in rendered
    assert rendered == "— (disabled)"


def test_paused_projection_next_run_is_suppressed():
    rendered = _format_next_run(_projection(TaskStatus.PAUSED))
    assert "2099" not in rendered
    assert rendered == "— (paused)"


def test_waiting_projection_next_run_still_concrete():
    assert "2099-07-20" in _format_next_run(_projection(TaskStatus.WAITING))


# --- Qodo review: suppression must not depend on a surviving timestamp ----
#
# ``_format_next_run`` used to return "-" for a null ``next_run_at``
# BEFORE consulting the status. Dispatching a one-time reminder sets
# enabled=False AND clears next_run_at (``mark_reminder_dispatched``), so
# a completed task rendered "-" in the Next Run column while its status
# badge read "Disabled" -- two surfaces disagreeing about the same row.


def _completed_one_time() -> ReminderTask:
    """A one-time reminder in its post-dispatch state: off, no next run."""
    return ReminderTask(
        id="task-done",
        title="Backup",
        schedule_kind=ScheduleKind.ONE_TIME,
        run_at=_RUN_AT,
        next_run_at=None,
        enabled=False,
        last_status=TaskStatus.COMPLETED,
    )


def test_completed_one_time_reminder_still_reads_disabled():
    assert _format_next_run(_completed_one_time()) == "— (disabled)"


def test_completed_one_time_reminder_badge_and_next_run_agree():
    task = _completed_one_time()
    assert _task_status(task) is TaskStatus.DISABLED
    assert "disabled" in _format_next_run(task).lower()


def test_enabled_reminder_without_next_run_still_reads_dash():
    """Negative control: only DISABLED/PAUSED earn the em-dash form."""
    task = ReminderTask(
        id="task-e",
        title="Backup",
        schedule_kind=ScheduleKind.ONE_TIME,
        run_at=_RUN_AT,
        next_run_at=None,
        enabled=True,
    )
    assert _format_next_run(task) == "-"


def test_paused_projection_without_next_run_still_reads_paused():
    projection = ScheduledTask(
        id="watchlist:paused",
        title="Watch",
        type="watchlist_job",
        status=TaskStatus.PAUSED,
        next_run_at=None,
    )
    assert _format_next_run(projection) == "— (paused)"


def test_waiting_projection_without_next_run_reads_dash():
    projection = ScheduledTask(
        id="watchlist:idle",
        title="Watch",
        type="watchlist_job",
        status=TaskStatus.WAITING,
        next_run_at=None,
    )
    assert _format_next_run(projection) == "-"


def test_real_dispatch_lifecycle_leaves_next_run_labelled_disabled(tmp_path):
    """End-to-end over the REAL DB, not a hand-built post-dispatch model."""
    from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
    from tldw_chatbook.Scheduling.services.scheduling_service import (
        SchedulingService,
    )

    due_at = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)
    db = ScheduledTasksDB(tmp_path / "dispatched.db")
    try:
        task_id = db.create_reminder_task(
            owner_id="local",
            title="Backup",
            schedule_kind="one_time",
            run_at=due_at.isoformat(),
            next_run_at=due_at.isoformat(),
            enabled=True,
        )
        db.mark_reminder_dispatched(task_id, due_at, success=True)
        row = db.get_reminder_task(task_id)
    finally:
        db.close()

    assert row is not None
    task = SchedulingService._row_to_reminder(row)
    # The state the bug depended on, asserted rather than assumed.
    assert task.enabled is False
    assert task.next_run_at is None

    assert _task_status(task) is TaskStatus.DISABLED
    assert _format_next_run(task) == "— (disabled)"


# --- review F5: behavior consumers use the underlying status --------------


def test_underlying_status_ignores_the_disabled_overlay():
    from tldw_chatbook.UI.Screens.scheduling.task_detail import (
        _underlying_status,
    )

    task = _reminder(enabled=False, last_status=TaskStatus.MISSED)
    assert _task_status(task) is TaskStatus.DISABLED
    assert _underlying_status(task) is TaskStatus.MISSED


# --- mounted behavior ------------------------------------------------------


from Tests.UI.schedules_test_helpers import MockSchedulingServiceMixin


class _DisabledTaskService(MockSchedulingServiceMixin):
    def __init__(self) -> None:
        self.enabled = False

    async def list_tasks(self, owner_id=None, include_projections=True):
        return [_reminder(enabled=self.enabled)]

    async def update_reminder(self, task_id, fields, *, owner_id=None):
        self.enabled = fields.get("enabled", self.enabled)


class _WorkbenchApp(ConsolidatedCSSApp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scheduling_service = _DisabledTaskService()


@pytest.mark.asyncio
async def test_disabled_row_and_badge_read_disabled_and_survive_refresh():
    app = _WorkbenchApp()
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(workbench)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        # redesign PR-2, Task 2: glyph/title/subtitle (spec S4) replaces
        # the old Title/Type/Status/Next-Run columns -- "Disabled" is now
        # conveyed by the paused glyph, not a separate Status cell. The
        # subtitle's schedule-summary half legitimately still names the
        # configured 2099 date (descriptive, unaffected by enabled state
        # -- same as the create/edit form would show); the "no concrete
        # future time" promise-avoidance rule (task-23101) applies only
        # to the next-run half, after the "·" separator.
        table = workbench.query_one("#scheduling-task-table", DataTable)
        row = table.get_row_at(0)
        assert str(row[0]) == "⏸"
        next_run_text = str(row[2]).rsplit("·", 1)[-1]
        assert "2099" not in next_run_text
        assert "disabled" in next_run_text.lower()

        badge = workbench.query_one("#scheduling-task-status-badge", Static)
        assert "Disabled" in str(badge.render())

        next_run = workbench.query_one(
            "#scheduling-task-detail-next-run", Static
        )
        assert "2099" not in str(next_run.render())

        # The derived state survives a queue refresh.
        await workbench.load_tasks()
        await pilot.pause()
        row = table.get_row_at(0)
        assert str(row[0]) == "⏸"


# --- review F5: mounted consumers of the underlying status ----------------


class _DisabledMissedService(MockSchedulingServiceMixin):
    """One disabled task whose last dispatch failed, one with a conflict."""

    async def list_tasks(self, owner_id=None, include_projections=True):
        return [
            _reminder_with(
                "task-m", "Failed backup", TaskStatus.MISSED, enabled=False
            ),
            _reminder_with(
                "task-c", "Conflicted", TaskStatus.CONFLICT, enabled=False
            ),
        ]


def _reminder_with(
    task_id: str, title: str, last_status: TaskStatus, *, enabled: bool
) -> ReminderTask:
    return ReminderTask(
        id=task_id,
        title=title,
        schedule_kind=ScheduleKind.ONE_TIME,
        run_at=_RUN_AT,
        next_run_at=_RUN_AT,
        enabled=enabled,
        last_status=last_status,
    )


class _MissedApp(ConsolidatedCSSApp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scheduling_service = _DisabledMissedService()


async def _mounted_missed_workbench(pilot):
    workbench = SchedulesWorkbench(app_instance=pilot.app)
    await pilot.app.push_screen(workbench)
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()
    return workbench


@pytest.mark.asyncio
async def test_disabled_missed_task_keeps_the_retry_affordance():
    """Run now explicitly works on disabled tasks; a disabled task whose
    last dispatch failed must keep 'Run now (retry)' (review F5)."""
    from textual.widgets import Button

    app = _MissedApp()
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = await _mounted_missed_workbench(pilot)
        table = workbench.query_one("#scheduling-task-table", DataTable)
        table.move_cursor(row=0)
        await pilot.pause()

        run_now = workbench.query_one("#scheduling-run-now", Button)
        assert str(run_now.label) == "Run now (retry)"
        # Display status stays Disabled.
        badge = workbench.query_one("#scheduling-task-status-badge", Static)
        assert "Disabled" in str(badge.render())


@pytest.mark.asyncio
async def test_missed_filter_matches_a_disabled_missed_task():
    """redesign PR-2, Task 2: the old ad hoc queue filter substring-
    matched status/type vocabulary too, so typing "missed" found a row
    whose `_was_missed_while_away` flag was set regardless of its title
    -- that was the old single-primitive table's own filter shape. Spec
    S4's unified search is explicitly title + question/body only (ruling
    5, `unified_rows.filter_rows`), so a bare status keyword no longer
    narrows the list; a title search still does.
    """
    app = _MissedApp()
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = await _mounted_missed_workbench(pilot)

        workbench._filter_text = "missed"
        workbench._render_table()
        await pilot.pause()
        titles = [task.title for task in workbench._visible_tasks]
        assert "Failed backup" not in titles, titles

        workbench._filter_text = "Failed backup"
        workbench._render_table()
        await pilot.pause()
        titles = [task.title for task in workbench._visible_tasks]
        assert "Failed backup" in titles, titles


@pytest.mark.asyncio
async def test_conflict_card_shows_for_a_disabled_conflicted_task():
    app = _MissedApp()
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = await _mounted_missed_workbench(pilot)
        table = workbench.query_one("#scheduling-task-table", DataTable)
        table.move_cursor(row=1)
        await pilot.pause()

        conflict_text = workbench.query_one("#scheduling-conflict-text", Static)
        assert "Conflict detected" in str(conflict_text.render())
