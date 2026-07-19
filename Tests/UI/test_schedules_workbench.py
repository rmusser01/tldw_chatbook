"""Tests for the SchedulesWorkbench shell."""

from datetime import datetime, timezone

import pytest
from textual.app import App
from textual.widgets import Button, DataTable, Static

from tldw_chatbook.Scheduling.models import ReminderTask, ScheduleKind
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import SchedulesWorkbench
from tldw_chatbook.UI.Screens.scheduling.task_detail import TaskDetail, TaskInspector


class WorkbenchTestApp(App):
    """Minimal test app that may not expose a real SchedulingService."""

    scheduling_service = None


class MockSchedulingService:
    """Stub service returning a single reminder task."""

    async def list_reminders(self):
        return [
            ReminderTask(
                id="task-1",
                title="Test",
                schedule_kind=ScheduleKind.ONE_TIME,
                run_at=datetime.now(timezone.utc),
            )
        ]


class WorkbenchTestAppWithService(App):
    """Test app with a mock scheduling service."""

    scheduling_service = MockSchedulingService()


@pytest.mark.asyncio
async def test_schedules_workbench_renders_panes():
    app = WorkbenchTestApp()
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        assert isinstance(pilot.app.screen, SchedulesWorkbench)
        assert pilot.app.screen.query_one("#scheduling-workbench") is not None
        assert pilot.app.screen.query_one("#scheduling-list-pane") is not None
        assert pilot.app.screen.query_one("#scheduling-detail-pane") is not None
        assert pilot.app.screen.query_one("#scheduling-inspector-pane") is not None


@pytest.mark.asyncio
async def test_select_task_updates_detail():
    """Selecting a task row updates the detail pane with task information."""
    async with WorkbenchTestAppWithService().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        assert table.row_count == 1
        table.cursor_coordinate = (0, 0)
        await pilot.pause()
        detail = pilot.app.screen.query_one("#scheduling-task-detail", TaskDetail)
        title = detail.query_one("#scheduling-task-detail-title", Static)
        assert "Test" in title.visual.plain


@pytest.mark.asyncio
async def test_console_follow_selector_exists():
    app = WorkbenchTestApp()
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        assert pilot.app.screen.query_one("#schedules-follow-in-console") is not None


@pytest.mark.asyncio
async def test_task_detail_renders_selected_task():
    """The TaskDetail widget renders the selected reminder's metadata."""
    async with WorkbenchTestAppWithService().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        assert table.row_count == 1
        table.cursor_coordinate = (0, 0)
        await pilot.pause()

        detail = pilot.app.screen.query_one("#scheduling-task-detail", TaskDetail)
        title = detail.query_one("#scheduling-task-detail-title", Static)
        kind = detail.query_one("#scheduling-task-detail-type", Static)
        status = detail.query_one("#scheduling-task-detail-status", Static)
        enable_button = detail.query_one("#scheduling-enable-task", Button)
        disable_button = detail.query_one("#scheduling-disable-task", Button)
        delete_button = detail.query_one("#scheduling-delete-task", Button)
        follow_button = detail.query_one("#schedules-follow-in-console", Button)

        assert "Test" in title.visual.plain
        assert "one_time" in kind.visual.plain
        assert "waiting" in status.visual.plain
        assert enable_button.label.plain == "Enable"
        assert disable_button.label.plain == "Disable"
        assert delete_button.label.plain == "Delete"
        assert follow_button.label.plain == "Follow in Console"


@pytest.mark.asyncio
async def test_task_inspector_renders_status():
    """The TaskInspector widget shows status, sync, and conflict text."""
    async with WorkbenchTestAppWithService().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        assert table.row_count == 1
        table.cursor_coordinate = (0, 0)
        await pilot.pause()

        inspector = pilot.app.screen.query_one("#scheduling-task-inspector", TaskInspector)
        status = inspector.query_one("#scheduling-inspector-status", Static)
        sync = inspector.query_one("#scheduling-inspector-sync", Static)
        conflict_card = inspector.query_one("#scheduling-conflict-card")
        conflict_text = inspector.query_one("#scheduling-conflict-text", Static)

        assert "waiting" in status.visual.plain
        assert "version" in sync.visual.plain
        assert "No conflict" in conflict_text.visual.plain
