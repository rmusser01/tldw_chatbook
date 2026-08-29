"""Automations tab behavior on the Schedules workbench (task-18940 slice 2).

The tab surfaces the server's automation definitions (ADR-077) and its
``r`` Run-now dispatches through the server control plane -- never the
local scheduler loop.
"""

from unittest.mock import AsyncMock

import pytest
from textual.widgets import DataTable, TabbedContent

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.schedules_test_helpers import (
    MockSchedulingDB,
    MockSchedulingServiceMixin,
)
from tldw_chatbook.Scheduling.services.server_client import (
    ServerClientValidationError,
)
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import SchedulesWorkbench


class AutomationsServerClient:
    """Stub scheduling server client with the automation control plane."""

    def __init__(self, notifications_service=None) -> None:
        self.notifications_service = notifications_service or object()
        self.list_automation_definitions = AsyncMock(
            return_value={
                "items": [
                    {
                        "id": "def-1",
                        "name": "Morning brief",
                        "family": "recurring_question",
                        "lifecycle": "configured",
                        "health": "ready",
                    },
                    {
                        "id": "def-2",
                        "name": "Paused one",
                        "family": "recurring_question",
                        "lifecycle": "paused",
                        "health": "ready",
                    },
                ],
                "total": 2,
            }
        )
        self.run_automation_definition_now = AsyncMock(
            return_value={
                "definition_id": "def-1",
                "run_slot_utc": "slot-1",
                "job_id": 42,
                "deduped": False,
            }
        )


class AutomationsMockService(MockSchedulingServiceMixin):
    """Scheduling service whose server client knows automations."""

    def __init__(self, server_client) -> None:
        self.owner_id = "local"
        self.server_client = server_client
        self.db = MockSchedulingDB()
        self.sync_engine = None

    async def list_tasks(self):
        return []


class _ServerRuntimeState:
    active_server_id = "server-1"


class _ServerRuntimePolicy:
    state = _ServerRuntimeState()


class AutomationsTestApp(ConsolidatedCSSApp):
    def __init__(self, service, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scheduling_service = service
        self.runtime_policy = _ServerRuntimePolicy()


class LocalOnlyTestApp(ConsolidatedCSSApp):
    scheduling_service = None


async def _mounted_workbench(app):
    """Run the app and return (pilot context, screen) with the workbench up."""
    return app


@pytest.mark.asyncio
async def test_automations_tab_loads_server_definitions():
    server_client = AutomationsServerClient()
    app = AutomationsTestApp(AutomationsMockService(server_client))
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        table = workbench.query_one("#scheduling-automations-table", DataTable)
        notice = workbench.query_one("#scheduling-automations-notice")

        server_client.list_automation_definitions.assert_awaited_once()
        assert table.row_count == 2
        assert "2 automations on the server" in str(notice.content)
        # Highlighting a row records the selection Run-now will act on.
        table.cursor_coordinate = (0, 0)
        await pilot.pause()
        assert workbench._selected_automation_id == "def-1"


@pytest.mark.asyncio
async def test_automations_tab_shows_notice_without_server():
    app = LocalOnlyTestApp()
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        notice = workbench.query_one("#scheduling-automations-notice")
        table = workbench.query_one("#scheduling-automations-table", DataTable)
        assert table.row_count == 0
        assert "need a connected server" in str(notice.content)


@pytest.mark.asyncio
async def test_run_now_on_automations_tab_dispatches_server_side():
    server_client = AutomationsServerClient()
    app = AutomationsTestApp(AutomationsMockService(server_client))
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        table = workbench.query_one("#scheduling-automations-table", DataTable)
        tabs = workbench.query_one("#scheduling-tabs", TabbedContent)
        tabs.active = "scheduling-automations-tab"
        table.cursor_coordinate = (0, 0)
        await pilot.pause()

        workbench.action_run_task_now()
        await pilot.pause()

        server_client.run_automation_definition_now.assert_awaited_once_with("def-1")


@pytest.mark.asyncio
async def test_run_now_refusal_surfaces_without_raising():
    server_client = AutomationsServerClient()
    server_client.run_automation_definition_now = AsyncMock(
        side_effect=ServerClientValidationError("definition_paused")
    )
    app = AutomationsTestApp(AutomationsMockService(server_client))
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        table = workbench.query_one("#scheduling-automations-table", DataTable)
        tabs = workbench.query_one("#scheduling-tabs", TabbedContent)
        tabs.active = "scheduling-automations-tab"
        table.cursor_coordinate = (0, 0)
        await pilot.pause()

        workbench.action_run_task_now()
        await pilot.pause()

        server_client.run_automation_definition_now.assert_awaited_once()
        # The refusal path must not raise out of the action handler.
        assert workbench._selected_automation_id == "def-1"


@pytest.mark.asyncio
async def test_run_now_on_queue_tab_never_reaches_server_client():
    server_client = AutomationsServerClient()
    app = AutomationsTestApp(AutomationsMockService(server_client))
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen

        # Queue tab is active by default; r must not reach the server client.
        workbench.action_run_task_now()
        await pilot.pause()
        server_client.run_automation_definition_now.assert_not_awaited()
