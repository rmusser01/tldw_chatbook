"""Tests for the SchedulesWorkbench shell."""

from datetime import datetime, timezone

import pytest
from textual.app import App
from textual.widgets import DataTable, Static

from tldw_chatbook.Scheduling.models import ReminderTask, ScheduleKind
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import SchedulesWorkbench


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
        detail = pilot.app.screen.query_one("#scheduling-detail-content", Static)
        assert "Test" in detail.visual.plain
