"""task-23111: Next Run pairs the absolute time with a relative form.

"2026-08-28 09:00 UTC" alone forces the reader to do timezone
arithmetic; the detail pane now renders "2026-08-28 09:00 UTC (in 14h)"
and the queue a shorter "2026-08-28 09:00 (in 14h)". All tests inject
``now`` for determinism.
"""

from datetime import datetime, timedelta, timezone

import pytest
from textual.widgets import DataTable, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Scheduling.models import ReminderTask, ScheduleKind
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import (
    SchedulesWorkbench,
)
from tldw_chatbook.UI.Screens.scheduling.task_detail import _format_next_run


_NOW = datetime(2026, 8, 27, 19, 0, tzinfo=timezone.utc)


def _task(next_run_at: datetime, *, enabled: bool = True) -> ReminderTask:
    return ReminderTask(
        id="task-r",
        title="Relative",
        schedule_kind=ScheduleKind.ONE_TIME,
        run_at=next_run_at,
        next_run_at=next_run_at,
        enabled=enabled,
    )


def test_detail_form_pairs_absolute_with_relative_hours():
    run_at = datetime(2026, 8, 28, 9, 0, tzinfo=timezone.utc)  # 14h ahead
    assert _format_next_run(_task(run_at), now=_NOW) == (
        "2026-08-28 09:00 UTC (in 14h)"
    )


def test_queue_form_is_shorter_but_same_relative():
    run_at = datetime(2026, 8, 28, 9, 0, tzinfo=timezone.utc)
    assert _format_next_run(_task(run_at), now=_NOW, compact=True) == (
        "2026-08-28 09:00 (in 14h)"
    )


def test_minutes_bucket():
    run_at = _NOW + timedelta(minutes=25)
    assert "(in 25m)" in _format_next_run(_task(run_at), now=_NOW)


def test_days_bucket():
    run_at = _NOW + timedelta(days=3, hours=2)
    assert "(in 3d)" in _format_next_run(_task(run_at), now=_NOW)


def test_under_a_minute_is_due_now():
    run_at = _NOW + timedelta(seconds=30)
    assert "(due now)" in _format_next_run(_task(run_at), now=_NOW)


def test_past_time_reads_overdue():
    run_at = _NOW - timedelta(hours=2)
    assert "(overdue 2h)" in _format_next_run(_task(run_at), now=_NOW)


def test_disabled_still_shows_no_time_and_no_relative():
    run_at = _NOW + timedelta(hours=5)
    rendered = _format_next_run(_task(run_at, enabled=False), now=_NOW)
    assert rendered == "— (disabled)"


def test_naive_next_run_treated_as_utc():
    run_at = datetime(2026, 8, 28, 9, 0)  # naive
    rendered = _format_next_run(_task(run_at), now=_NOW)
    assert "(in 14h)" in rendered


# --- consistency between the queue column and the detail pane --------------


from Tests.UI.schedules_test_helpers import MockSchedulingServiceMixin


class _Service(MockSchedulingServiceMixin):
    async def list_tasks(self):
        return [
            ReminderTask(
                id="task-1",
                title="Soon",
                schedule_kind=ScheduleKind.ONE_TIME,
                run_at=datetime.now(timezone.utc) + timedelta(hours=14),
                next_run_at=datetime.now(timezone.utc) + timedelta(hours=14),
            )
        ]


class _App(ConsolidatedCSSApp):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scheduling_service = _Service()


@pytest.mark.asyncio
async def test_queue_cell_and_detail_pane_agree_on_the_relative_form():
    app = _App()
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(workbench)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        table = workbench.query_one("#scheduling-task-table", DataTable)
        cell = str(table.get_row_at(0)[3])
        detail = str(
            workbench.query_one(
                "#scheduling-task-detail-next-run", Static
            ).render()
        )
        assert "(in 13h)" in cell or "(in 14h)" in cell, cell
        # Same relative rendering in both surfaces.
        relative = cell[cell.index("(") :]
        assert relative in detail, (cell, detail)
