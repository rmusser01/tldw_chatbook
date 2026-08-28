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
        # Full phrases (review F13: 'assert "d" in text' was vacuous --
        # "marked" itself contains a d).
        assert "space toggles all" in text
        assert "d deletes all" in text
        assert "esc clears" in text

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


# --- review F1: marks never fall through, projections not markable --------


class _MixedService(MockSchedulingServiceMixin):
    """One reminder plus one read-only watchlist projection."""

    def __init__(self) -> None:
        self.updated: list = []

    async def list_tasks(self):
        from tldw_chatbook.Scheduling.models import ScheduledTask, TaskStatus

        return [
            ReminderTask(
                id="task-1",
                title="First",
                schedule_kind=ScheduleKind.ONE_TIME,
                run_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
            ),
            ScheduledTask(
                id="watchlist:1",
                title="Watchlist Title",
                type="watchlist_job",
                status=TaskStatus.WAITING,
                next_run_at=datetime(2099, 1, 2, tzinfo=timezone.utc),
            ),
        ]

    async def update_reminder(self, task_id, fields):
        self.updated.append((task_id, fields))


@pytest.mark.asyncio
async def test_projection_rows_are_not_markable():
    app = _App(_MixedService())
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        table = workbench.query_one("#scheduling-task-table", DataTable)
        table.move_cursor(row=1)  # the projection
        await pilot.pause()

        workbench.action_mark_task()
        await pilot.pause()

        assert not workbench._marked_ids
        assert "marked" not in _notice_text(workbench)
        messages = [n.message for n in pilot.app._notifications]
        assert any("Managed by Watchlists" in m for m in messages), messages


@pytest.mark.asyncio
async def test_bulk_delete_never_falls_through_to_the_highlighted_row():
    """d with marks that no longer resolve refuses instead of opening the
    single-row delete for a task the user never marked."""
    from tldw_chatbook.Widgets.delete_confirmation_dialog import (
        DeleteConfirmationDialog,
    )

    app = _App(_MixedService())
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        # Simulate marks that vanished between renders (deleted elsewhere).
        workbench._marked_ids = {"ghost-id"}

        workbench.action_delete()
        await pilot.pause()

        assert not isinstance(pilot.app.screen, DeleteConfirmationDialog), (
            "d fell through to the single-row delete while marks existed"
        )
        assert not workbench._marked_ids  # stale marks cleared
        messages = [n.message for n in pilot.app._notifications]
        assert any("nothing was deleted" in m.lower() for m in messages), messages


@pytest.mark.asyncio
async def test_bulk_toggle_never_falls_through_to_the_highlighted_row():
    app = _App(_MixedService())
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        workbench._marked_ids = {"ghost-id"}

        workbench.action_toggle_enabled()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()

        assert pilot.app.scheduling_service.updated == [], (
            "space fell through and toggled the highlighted, unmarked row"
        )
        assert not workbench._marked_ids


@pytest.mark.asyncio
async def test_stale_marks_are_pruned_on_reload():
    app = _App(_MixedService())
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        workbench._marked_ids = {"task-1", "ghost-id"}

        await workbench.load_tasks()
        await pilot.pause()

        assert workbench._marked_ids == {"task-1"}


@pytest.mark.asyncio
async def test_legend_states_marks_hidden_by_the_filter():
    app = _App(_Service())
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        table = workbench.query_one("#scheduling-task-table", DataTable)
        workbench.action_mark_task()
        await pilot.pause()
        table.move_cursor(row=1)
        await pilot.pause()
        workbench.action_mark_task()
        await pilot.pause()
        assert "2 marked" in _notice_text(workbench)

        # Filter so only "First" is visible; the second mark goes hidden.
        workbench._filter_text = "First"
        workbench._render_table()
        await pilot.pause()

        text = _notice_text(workbench)
        assert "2 marked (1 hidden by the filter)" in text, text
