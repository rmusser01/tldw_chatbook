from unittest.mock import Mock

import pytest
from textual.app import App

from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Home.active_work_adapter import (
    HomeConsoleLaunch,
    HomeControlAction,
    HomeControlResult,
    HomeControlResultStatus,
)
from tldw_chatbook.Home.dashboard_state import HomeActiveWorkItem, HomeDashboardInput
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


class RecordingHomeActiveWorkAdapter:
    def __init__(self, dashboard_input=None, responses=None):
        self.dashboard_calls = 0
        self.control_actions = []
        self.control_target_routes = []
        self.control_target_ids = []
        self.dashboard_input = dashboard_input
        self.responses = responses or {}

    def build_dashboard_input(self, *, providers_models, has_recent_work):
        self.dashboard_calls += 1
        if self.dashboard_input is not None:
            return self.dashboard_input
        return HomeDashboardInput(
            model_ready=True,
            pending_approval_count=1,
            running_run_count=1,
            active_run_count=1,
            has_library_content=True,
            has_recent_work=has_recent_work,
        )

    def handle_control(self, action, *, target_id=None, target_route=None):
        self.control_actions.append(action)
        self.control_target_routes.append(target_route)
        self.control_target_ids.append(target_id)
        if action in self.responses:
            return self.responses[action]
        return HomeControlResult(
            action=action,
            status=HomeControlResultStatus.HANDLED,
            message=f"{action.value} handled by adapter",
            severity="information",
            recovery_route="chat",
        )


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

    assert seen[-1] in {"chat", "llm", "library", "schedules", "subscriptions"}


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
async def test_home_screen_renders_unread_notification_snapshot_without_controls():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        notification_count=2,
        has_library_content=True,
    )
    host = HomeHarness(app)

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        home = _active_home_screen(host)

        assert "Unread notifications: 2" in str(
            home.query_one("#home-attention-body").renderable
        )
        assert len(home.query("#home-approve")) == 0
        assert len(home.query("#home-pause")) == 0
        assert len(home.query("#home-open-in-console")) == 0


@pytest.mark.asyncio
async def test_home_notification_primary_action_opens_notifications_inbox_context():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        notification_count=2,
        has_library_content=True,
    )
    seen = []
    host = HomeHarness(app, seen)

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        await pilot.click("#home-primary-action")
        await pilot.pause(0.1)

    assert seen[-1] == "subscriptions"
    assert app.pending_subscription_initial_tab == "notifications"


@pytest.mark.asyncio
async def test_home_failed_watchlist_primary_action_opens_watchlist_runs_context():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        active_work_items=(
            HomeActiveWorkItem(
                item_id="local:watchlist_run:5",
                title="Daily security feed",
                source="W+C",
                status="failed",
                detail_route="subscriptions",
            ),
        ),
    )
    seen = []
    host = HomeHarness(app, seen)

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        await pilot.click("#home-primary-action")
        await pilot.pause(0.1)

    assert seen[-1] == "subscriptions"
    assert app.pending_subscription_initial_tab == "watchlist-runs"


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


def test_app_exposes_home_runtime_control_hooks():
    app = _build_test_app()

    for method_name in [
        "approve_active_home_item",
        "reject_active_home_item",
        "pause_active_home_item",
        "resume_active_home_item",
        "retry_active_home_item",
    ]:
        assert callable(getattr(app, method_name, None))


@pytest.mark.asyncio
async def test_home_screen_uses_active_work_adapter_for_dashboard_and_controls():
    app = _build_test_app()
    adapter = RecordingHomeActiveWorkAdapter()
    app.home_active_work_adapter = adapter
    app.notify = Mock()
    host = HomeHarness(app)

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        await pilot.click("#home-approve")
        await pilot.pause(0.1)

    assert adapter.dashboard_calls == 1
    assert adapter.control_actions == [HomeControlAction.APPROVE]
    app.notify.assert_called_once_with(
        "approve handled by adapter",
        severity="information",
    )


@pytest.mark.asyncio
async def test_home_detail_controls_do_not_directly_navigate_without_adapter_payload():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        pending_approval_count=1,
        active_run_count=1,
        has_library_content=True,
        active_detail_route="workflows",
    )
    app.notify = Mock()
    seen = []
    host = HomeHarness(app, seen)

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        await pilot.click("#home-open-details")
        await pilot.pause(0.1)
        await pilot.click("#home-open-in-console")
        await pilot.pause(0.1)

    assert seen == []
    assert app.notify.call_count == 2


@pytest.mark.asyncio
async def test_home_detail_and_console_buttons_call_runtime_hooks_with_target_route():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        pending_approval_count=1,
        active_run_count=1,
        has_library_content=True,
        active_detail_route="workflows",
    )
    app.open_active_home_item_details = Mock()
    app.open_active_home_item_in_console = Mock()
    seen = []
    host = HomeHarness(app, seen)

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        await pilot.click("#home-open-details")
        await pilot.pause(0.1)
        await pilot.click("#home-open-in-console")
        await pilot.pause(0.1)

    app.open_active_home_item_details.assert_called_once_with(target_route="workflows")
    app.open_active_home_item_in_console.assert_called_once_with(target_route="chat")
    assert seen == []


@pytest.mark.asyncio
async def test_home_active_work_item_controls_pass_target_id_to_runtime_hooks():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        active_work_items=(
            HomeActiveWorkItem(
                item_id="run-1",
                title="Daily digest",
                source="workflows",
                status="running",
                detail_route="workflows",
                console_available=True,
            ),
        ),
    )
    app.pause_active_home_item = Mock()
    app.open_active_home_item_details = Mock()
    app.open_active_home_item_in_console = Mock()
    host = HomeHarness(app, [])

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        await pilot.click("#home-pause")
        await pilot.pause(0.1)
        await pilot.click("#home-open-details")
        await pilot.pause(0.1)
        await pilot.click("#home-open-in-console")
        await pilot.pause(0.1)

    app.pause_active_home_item.assert_called_once_with(target_id="run-1")
    app.open_active_home_item_details.assert_called_once_with(
        target_id="run-1",
        target_route="workflows",
    )
    app.open_active_home_item_in_console.assert_called_once_with(
        target_id="run-1",
        target_route="chat",
    )


@pytest.mark.asyncio
async def test_home_saved_chatbook_artifact_resume_controls_pass_artifact_target():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        active_work_items=(
            HomeActiveWorkItem(
                item_id="local:chatbook:77",
                title="Grounded Answer",
                source="Artifacts",
                status="ready",
                detail_route="artifacts",
                console_available=True,
            ),
        ),
    )
    app.open_active_home_item_details = Mock()
    app.open_active_home_item_in_console = Mock()
    host = HomeHarness(app, [])

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        home = _active_home_screen(host)

        active_work_text = str(home.query_one("#home-active-work-body").renderable)
        assert "Grounded Answer" in active_work_text
        assert "Artifacts" in active_work_text
        assert "ready" in active_work_text

        await pilot.click("#home-open-details")
        await pilot.pause(0.1)
        await pilot.click("#home-open-in-console")
        await pilot.pause(0.1)

    app.open_active_home_item_details.assert_called_once_with(
        target_id="local:chatbook:77",
        target_route="artifacts",
    )
    app.open_active_home_item_in_console.assert_called_once_with(
        target_id="local:chatbook:77",
        target_route="chat",
    )


@pytest.mark.asyncio
async def test_home_mixed_active_work_exposes_chatbook_artifact_resume_controls():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        active_work_items=(
            HomeActiveWorkItem(
                item_id="local:watchlist_run:5",
                title="Daily Feed",
                source="W+C",
                status="running",
                detail_route="watchlists",
                console_available=True,
            ),
            HomeActiveWorkItem(
                item_id="local:chatbook:77",
                title="Grounded Answer",
                source="Artifacts",
                status="ready",
                detail_route="artifacts",
                console_available=True,
            ),
        ),
    )
    app.open_active_home_item_details = Mock()
    app.open_active_home_item_in_console = Mock()
    host = HomeHarness(app, [])

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        home = _active_home_screen(host)

        assert len(home.query("#home-open-chatbook-details")) == 1
        assert len(home.query("#home-open-chatbook-in-console")) == 1
        await pilot.click("#home-open-chatbook-details")
        await pilot.pause(0.1)
        await pilot.click("#home-open-chatbook-in-console")
        await pilot.pause(0.1)

    app.open_active_home_item_details.assert_called_once_with(
        target_id="local:chatbook:77",
        target_route="artifacts",
    )
    app.open_active_home_item_in_console.assert_called_once_with(
        target_id="local:chatbook:77",
        target_route="chat",
    )


def test_app_detail_hook_delegates_to_adapter_and_navigates_handled_route():
    app = _build_test_app()
    adapter = RecordingHomeActiveWorkAdapter(
        responses={
            HomeControlAction.OPEN_DETAILS: HomeControlResult(
                action=HomeControlAction.OPEN_DETAILS,
                status=HomeControlResultStatus.HANDLED,
                message="Opening workflow details.",
                target_route="workflows",
            ),
        }
    )
    app.home_active_work_adapter = adapter
    app.notify = Mock()
    app.post_message = Mock()

    result = app.open_active_home_item_details(target_id="run-1", target_route="schedules")

    assert result.status is HomeControlResultStatus.HANDLED
    assert adapter.control_actions == [HomeControlAction.OPEN_DETAILS]
    assert adapter.control_target_ids == ["run-1"]
    assert adapter.control_target_routes == ["schedules"]
    app.notify.assert_called_once_with("Opening workflow details.", severity="information")
    app.post_message.assert_called_once()
    assert app.post_message.call_args.args[0].screen_name == "workflows"


def test_app_detail_hook_stages_watchlist_runs_context_for_handled_watchlist_detail():
    app = _build_test_app()
    adapter = RecordingHomeActiveWorkAdapter(
        responses={
            HomeControlAction.OPEN_DETAILS: HomeControlResult(
                action=HomeControlAction.OPEN_DETAILS,
                status=HomeControlResultStatus.HANDLED,
                message="Opening W+C run details for Daily security feed.",
                target_id="local:watchlist_run:5",
                target_route="subscriptions",
            ),
        }
    )
    app.home_active_work_adapter = adapter
    app.notify = Mock()
    app.post_message = Mock()

    result = app.open_active_home_item_details(
        target_id="local:watchlist_run:5",
        target_route="subscriptions",
    )

    assert result.status is HomeControlResultStatus.HANDLED
    assert app.pending_subscription_initial_tab == "watchlist-runs"
    assert app.pending_subscription_watchlist_run_id == "local:watchlist_run:5"
    app.post_message.assert_called_once()
    assert app.post_message.call_args.args[0].screen_name == "subscriptions"


def test_app_console_hook_requires_adapter_launch_payload():
    app = _build_test_app()
    adapter = RecordingHomeActiveWorkAdapter(
        responses={
            HomeControlAction.OPEN_IN_CONSOLE: HomeControlResult(
                action=HomeControlAction.OPEN_IN_CONSOLE,
                status=HomeControlResultStatus.UNAVAILABLE,
                message="Open in Console is not connected.",
                severity="warning",
                recovery_route="chat",
            ),
        }
    )
    app.home_active_work_adapter = adapter
    app.notify = Mock()
    app.open_console_for_live_work = Mock()

    result = app.open_active_home_item_in_console(target_id="run-1", target_route="chat")

    assert result.status is HomeControlResultStatus.UNAVAILABLE
    assert adapter.control_actions == [HomeControlAction.OPEN_IN_CONSOLE]
    assert adapter.control_target_ids == ["run-1"]
    assert adapter.control_target_routes == ["chat"]
    app.notify.assert_called_once_with("Open in Console is not connected.", severity="warning")
    app.open_console_for_live_work.assert_not_called()


def test_app_console_hook_opens_console_with_adapter_launch_payload():
    app = _build_test_app()
    launch = HomeConsoleLaunch(
        source="workflows",
        title="Daily digest",
        payload={"run_id": "run-1"},
    )
    adapter = RecordingHomeActiveWorkAdapter(
        responses={
            HomeControlAction.OPEN_IN_CONSOLE: HomeControlResult(
                action=HomeControlAction.OPEN_IN_CONSOLE,
                status=HomeControlResultStatus.HANDLED,
                message="Opening Console for Daily digest.",
                console_launch=launch,
            ),
        }
    )
    app.home_active_work_adapter = adapter
    app.notify = Mock()
    app.open_console_for_live_work = Mock()

    result = app.open_active_home_item_in_console(target_id="run-1", target_route="chat")

    assert result.status is HomeControlResultStatus.HANDLED
    assert adapter.control_target_ids == ["run-1"]
    app.notify.assert_called_once_with("Opening Console for Daily digest.", severity="information")
    app.open_console_for_live_work.assert_called_once_with(
        source="workflows",
        title="Daily digest",
        payload={"run_id": "run-1"},
    )


def test_app_console_hook_preserves_status_recovery_and_action_label():
    app = _build_test_app()
    launch = HomeConsoleLaunch(
        source="W+C",
        title="Daily security feed",
        payload={"run_id": 5, "target_id": "local:watchlist_run:5"},
        status="failed",
        recovery="Review the W+C run details or retry from W+C.",
        action_label="Open W+C run",
    )
    adapter = RecordingHomeActiveWorkAdapter(
        responses={
            HomeControlAction.OPEN_IN_CONSOLE: HomeControlResult(
                action=HomeControlAction.OPEN_IN_CONSOLE,
                status=HomeControlResultStatus.HANDLED,
                message="Opening Console for Daily security feed.",
                console_launch=launch,
            ),
        }
    )
    app.home_active_work_adapter = adapter
    app.notify = Mock()
    app.open_console_for_live_work = Mock()

    result = app.open_active_home_item_in_console(
        target_id="local:watchlist_run:5",
        target_route="chat",
    )

    assert result.status is HomeControlResultStatus.HANDLED
    app.open_console_for_live_work.assert_called_once_with(
        source="W+C",
        title="Daily security feed",
        payload={"run_id": 5, "target_id": "local:watchlist_run:5"},
        status="failed",
        recovery="Review the W+C run details or retry from W+C.",
        action_label="Open W+C run",
    )


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


@pytest.mark.asyncio
async def test_pending_console_launch_does_not_create_home_live_work_controls():
    app = _build_test_app()
    app.pending_console_launch = {
        "source": "workflows",
        "title": "Daily digest",
        "payload": {},
    }
    host = HomeHarness(app)

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        home = _active_home_screen(host)

        assert len(home.query("#home-pause")) == 0
        assert len(home.query("#home-open-in-console")) == 0
