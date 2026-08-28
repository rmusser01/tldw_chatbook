"""task-23107: the bulk-mark mechanism must be visible.

``x`` marks rows (●) and ◇ flags missed-while-away, but nothing on
screen said how many rows were marked, which keys act on the marks, or
what the glyphs mean.
"""

from datetime import datetime, timezone

import pytest
from textual.widgets import DataTable, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Scheduling.models import ReminderTask, ScheduleKind
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import (
    SchedulesWorkbench,
)


from Tests.UI.schedules_test_helpers import MockSchedulingServiceMixin


class _Service(MockSchedulingServiceMixin):
    def __init__(self, *, with_missed: bool = False) -> None:
        self._with_missed = with_missed

    async def list_tasks(self):
        tasks = [
            ReminderTask(
                id="task-1",
                title="First",
                schedule_kind=ScheduleKind.ONE_TIME,
                run_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
            ),
            ReminderTask(
                id="task-2",
                title="Second",
                schedule_kind=ScheduleKind.ONE_TIME,
                run_at=datetime(2099, 1, 2, tzinfo=timezone.utc),
            ),
        ]
        if self._with_missed:
            tasks.append(
                ReminderTask(
                    id="task-3",
                    title="Late one",
                    schedule_kind=ScheduleKind.ONE_TIME,
                    run_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
                    missed_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
                )
            )
        return tasks


class _App(ConsolidatedCSSApp):
    def __init__(self, service, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scheduling_service = service


def _notice_text(workbench) -> str:
    return str(workbench.query_one("#scheduling-pane-notice", Static).render())


async def _mounted(pilot):
    workbench = SchedulesWorkbench(app_instance=pilot.app)
    await pilot.app.push_screen(workbench)
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()
    return workbench


@pytest.mark.asyncio
async def test_marking_rows_shows_count_and_keys():
    app = _App(_Service())
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        assert "marked" not in _notice_text(workbench)

        workbench.action_mark_task()
        await pilot.pause()
        text = _notice_text(workbench)
        assert "1 marked" in text
        assert "space" in text and "d" in text and "esc" in text

        table = workbench.query_one("#scheduling-task-table", DataTable)
        table.move_cursor(row=1)
        await pilot.pause()
        workbench.action_mark_task()
        await pilot.pause()
        assert "2 marked" in _notice_text(workbench)


@pytest.mark.asyncio
async def test_clearing_marks_clears_the_legend():
    app = _App(_Service())
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        workbench.action_mark_task()
        await pilot.pause()
        assert "1 marked" in _notice_text(workbench)

        workbench.action_clear_marks()
        await pilot.pause()
        assert "marked" not in _notice_text(workbench)


@pytest.mark.asyncio
async def test_missed_glyph_has_a_visible_explanation():
    app = _App(_Service(with_missed=True))
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        text = _notice_text(workbench)
        assert "◇" in text
        assert "ran late" in text.lower()


@pytest.mark.asyncio
async def test_no_missed_rows_means_no_glyph_legend():
    app = _App(_Service(with_missed=False))
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        assert "◇" not in _notice_text(workbench)


@pytest.mark.asyncio
async def test_resize_notice_and_mark_legend_coexist():
    app = _App(_Service())
    async with app.run_test(size=(100, 48)) as pilot:
        workbench = await _mounted(pilot)
        workbench.action_mark_task()
        await pilot.pause()
        text = _notice_text(workbench)
        assert "Inspector hidden" in text
        assert "1 marked" in text
