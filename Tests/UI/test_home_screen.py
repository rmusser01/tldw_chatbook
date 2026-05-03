from unittest.mock import Mock

import pytest
from textual.app import App

from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Home.dashboard_state import HomeDashboardInput
from tldw_chatbook.UI.Screens.home_screen import HomeScreen
from Tests.UI.test_screen_navigation import _build_test_app


class HomeHarness(App):
    def __init__(self, app_instance, seen_routes=None):
        super().__init__()
        self.app_instance = app_instance
        self.seen_routes = seen_routes if seen_routes is not None else []

    async def on_mount(self) -> None:
        await self.push_screen(HomeScreen(self.app_instance))

    def on_navigate_to_screen(self, message) -> None:
        self.seen_routes.append(message.screen_name)


def _active_home_screen(host: HomeHarness):
    return host.screen_stack[-1]


@pytest.mark.asyncio
async def test_home_screen_shows_dashboard_sections():
    app = _build_test_app()
    host = HomeHarness(app)

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        home = _active_home_screen(host)

        assert home.query_one("#home-title").has_class("ds-destination-header")
        assert home.query_one("#home-next-best-action").has_class("ds-panel")
        for selector in [
            "#home-status",
            "#home-attention",
            "#home-active-work",
            "#home-next-best-action",
            "#home-recent-work",
        ]:
            assert home.query_one(selector)


@pytest.mark.asyncio
async def test_home_primary_action_opens_target_route():
    app = _build_test_app()
    seen = []
    host = HomeHarness(app, seen)

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        await pilot.click("#home-primary-action")
        await pilot.pause(0.1)

    assert seen[-1] in {"chat", "llm", "library", "schedules"}


@pytest.mark.asyncio
async def test_home_screen_shows_lightweight_agent_and_schedule_controls():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        pending_approval_count=1,
        running_run_count=1,
        paused_run_count=1,
        failed_run_count=1,
        failed_schedule_count=1,
        active_run_count=3,
        has_library_content=True,
    )
    host = HomeHarness(app)

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        home = _active_home_screen(host)

        for selector in [
            "#home-approve",
            "#home-reject",
            "#home-pause",
            "#home-resume",
            "#home-retry",
            "#home-open-details",
            "#home-open-in-console",
        ]:
            assert home.query_one(selector).has_class("ds-toolbar")


@pytest.mark.asyncio
async def test_home_control_clicks_call_available_runtime_hooks():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        pending_approval_count=1,
        running_run_count=1,
        paused_run_count=1,
        failed_run_count=1,
        failed_schedule_count=1,
        active_run_count=3,
        has_library_content=True,
    )
    app.approve_active_home_item = Mock()
    app.reject_active_home_item = Mock()
    app.pause_active_home_item = Mock()
    app.resume_active_home_item = Mock()
    app.retry_active_home_item = Mock()
    host = HomeHarness(app)

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        for selector in [
            "#home-approve",
            "#home-reject",
            "#home-pause",
            "#home-resume",
            "#home-retry",
        ]:
            await pilot.click(selector)
            await pilot.pause(0.1)

    app.approve_active_home_item.assert_called_once()
    app.reject_active_home_item.assert_called_once()
    app.pause_active_home_item.assert_called_once()
    app.resume_active_home_item.assert_called_once()
    app.retry_active_home_item.assert_called_once()


@pytest.mark.asyncio
async def test_home_detail_controls_route_to_owner_and_console():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        pending_approval_count=1,
        active_run_count=1,
        has_library_content=True,
        active_detail_route="workflows",
    )
    seen = []
    host = HomeHarness(app, seen)

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        await pilot.click("#home-open-details")
        await pilot.pause(0.1)
        await pilot.click("#home-open-in-console")
        await pilot.pause(0.1)

    assert "workflows" in seen
    assert "chat" in seen


@pytest.mark.asyncio
async def test_pending_chat_handoff_does_not_create_live_work_controls():
    app = _build_test_app()
    app.pending_chat_handoff = ChatHandoffPayload(
        source="library",
        item_type="note",
        title="Research note",
        body="Context to stage in Console.",
    )
    host = HomeHarness(app)

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        home = _active_home_screen(host)

        assert len(home.query("#home-pause")) == 0
        assert len(home.query("#home-resume")) == 0
        assert len(home.query("#home-retry")) == 0
        assert len(home.query("#home-open-in-console")) == 0
