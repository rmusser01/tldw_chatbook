"""Tests for the SchedulesWorkbench shell."""

import pytest
from textual.app import App

from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import SchedulesWorkbench


class WorkbenchTestApp(App):
    """Minimal test app that may not expose a real SchedulingService."""

    scheduling_service = None


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
