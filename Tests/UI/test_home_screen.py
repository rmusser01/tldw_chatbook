from pathlib import Path
from unittest.mock import Mock

import pytest
from textual.app import App

from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Constants import LIBRARY_NAV_CONTEXT_INGEST, TAB_LIBRARY
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Home.active_work_adapter import (
    HomeConsoleLaunch,
    HomeControlAction,
    HomeControlResult,
    HomeControlResultStatus,
)
from tldw_chatbook.Home.dashboard_state import (
    HOME_FLASHCARDS_DUE_ROW_ID,
    HomeActiveWorkItem,
    HomeDashboardInput,
)
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.UI.Screens import home_screen as home_screen_module
from tldw_chatbook.UI.Screens.home_screen import HomeScreen
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from Tests.UI.test_screen_navigation import _build_test_app


HOME_TEST_SIZE = (160, 40)
HOME_MOUNT_PAUSE = 0.1
HOME_FOLLOWUP_ROW_MAX_HEIGHT = 6


@pytest.fixture(autouse=True)
def _stub_home_rail_preferences_cli_fallback(monkeypatch):
    """Isolate ``HomeScreen`` construction from the real on-disk CLI config.

    ``_home_rail_preferences`` (C4) falls back to ``get_cli_setting`` when
    ``app_config`` has no in-memory ``home.rail_state`` yet -- the same
    restart-persistence fix already applied to Library's rail preferences
    and search history. Tests share one real ``HOME``/``config.toml``
    across the whole pytest session, so without this stub a freshly
    constructed ``HomeScreen`` could non-deterministically inherit
    whatever ``[home.rail_state]`` a prior test (or prior session) happened
    to leave on disk (mirrors ``test_library_shell.py``'s
    ``_stub_library_search_history_cli_fallback`` fixture and its
    documented autouse-stub hazard). Tests that want to exercise the
    CLI-config fallback itself re-patch ``home_screen_module.get_cli_setting``
    after this fixture runs, which takes precedence for the rest of the
    test.
    """
    monkeypatch.setattr(
        home_screen_module, "get_cli_setting", lambda *args, **kwargs: None
    )


class HomeHarness(App):
    CSS_PATH = str(
        Path(__file__).resolve().parents[2] / "tldw_chatbook" / "css" / "tldw_cli_modular.tcss"
    )

    def __init__(self, app_instance, seen_routes=None):
        super().__init__()
        self.app_instance = app_instance
        self.seen_routes = seen_routes if seen_routes is not None else []
        self.seen_contexts = []

    async def on_mount(self) -> None:
        await self.push_screen(HomeScreen(self.app_instance))

    def on_navigate_to_screen(self, message) -> None:
        self.seen_routes.append(message.screen_name)
        self.seen_contexts.append(dict(message.screen_context or {}))


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

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        assert home.query_one("#home-header-line").has_class("destination-status-row")
        for selector in [
            "#home-rail",
            "#home-canvas",
            "#home-rail-section-header-attention",
            "#home-rail-section-header-running",
            "#home-rail-section-header-recent",
            "#home-rail-section-header-details",
            "#home-details-body",
        ]:
            assert home.query_one(selector)


@pytest.mark.asyncio
async def test_home_rail_preferences_loads_from_cli_config_fallback(monkeypatch):
    """(C4) Same restart-persistence gap as Library's rail preferences /
    search history: ``app_config`` (from ``load_settings()``) can come
    back without a ``home`` section at all even when ``config.toml`` has
    persisted ``[home.rail_state]`` sections on disk -- so a freshly
    started app would otherwise always reopen every Home rail section at
    its hardcoded default instead of the user's last-chosen open/collapsed
    state. Mirrors Library's ``_library_rail_preferences`` fallback
    template exactly (1-arg dotted ``get_cli_setting`` call, ``sections``
    sub-key extracted from the returned ``rail_state`` dict).
    """
    app = _build_test_app()
    assert "home" not in app.app_config

    calls: list[tuple] = []

    def fake_get_cli_setting(section, key=None, default=None):
        calls.append((section, key, default))
        if section == "home.rail_state" and key is None:
            return {"sections": {"details_open": True, "attention_open": False}}
        return default

    monkeypatch.setattr(home_screen_module, "get_cli_setting", fake_get_cli_setting)

    host = HomeHarness(app)
    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        preferences = home._home_rail_preferences()
        assert preferences.details_open is True
        assert preferences.attention_open is False
        assert calls, "get_cli_setting fallback was never consulted"


@pytest.mark.asyncio
async def test_home_rail_preferences_prefers_app_config_over_cli_config(monkeypatch):
    """Precedence: when ``app_config`` already carries rail-state sections,
    the ``get_cli_setting`` fallback must never be consulted.
    """
    app = _build_test_app()
    app.app_config["home"] = {"rail_state": {"sections": {"details_open": True}}}

    def raising_get_cli_setting(*args, **kwargs):
        raise AssertionError(
            "get_cli_setting should not be called when app_config already has rail state"
        )

    monkeypatch.setattr(home_screen_module, "get_cli_setting", raising_get_cli_setting)

    host = HomeHarness(app)
    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        assert home._home_rail_preferences().details_open is True


@pytest.mark.asyncio
async def test_home_screen_compacts_multi_module_readiness_summary():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        rag_ready=False,
        mcp_ready=True,
        acp_ready=False,
        pending_approval_count=1,
        active_run_count=2,
        has_library_content=True,
    )
    host = HomeHarness(app)

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        status_text = str(home.query_one("#home-details-body").renderable)
        assert "Model: Ready" in status_text
        assert "RAG: Missing sources" in status_text
        assert "MCP: Ready" in status_text
        assert "ACP: Blocked" in status_text
        assert "Active: 2" in status_text
        assert "Approvals: 1" in status_text


@pytest.mark.asyncio
async def test_home_screen_acp_readiness_uses_runtime_process_state():
    app = _build_test_app()
    app.home_active_work_adapter = RecordingHomeActiveWorkAdapter(
        HomeDashboardInput(
            model_ready=True,
            rag_ready=True,
            mcp_ready=True,
            acp_ready=True,
        )
    )
    app.acp_runtime_process_manager = Mock()
    app.acp_runtime_process_manager.snapshot.return_value = {"status": "not_configured"}
    host = HomeHarness(app)

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        status_text = str(home.query_one("#home-details-body").renderable)
        system_text = str(home.query_one("#home-details-body").renderable)

        assert "ACP: Blocked" in status_text
        assert "ACP blocked" in system_text


@pytest.mark.asyncio
async def test_home_system_status_groups_runtime_readiness_and_work_state():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        rag_ready=False,
        mcp_ready=True,
        acp_ready=False,
        runtime_source="local",
        server_configured=False,
        active_run_count=2,
        pending_approval_count=1,
        has_library_content=True,
    )
    host = HomeHarness(app)

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        system_text = str(home.query_one("#home-details-body").renderable)

        assert "Runtime: Local" in system_text
        assert "Server sync: Not configured (local mode)" in system_text
        assert "Local mode is active. Server sync is optional." in system_text
        assert "Agent readiness: Model ready, RAG needs sources, MCP ready, ACP blocked" in system_text
        assert "Work: 2 active, 1 approvals" in system_text


@pytest.mark.asyncio
async def test_home_screen_status_row_surfaces_server_auth_state():
    app = _build_test_app()
    app.runtime_policy.state = RuntimeSourceState(
        active_source="server",
        active_server_id="primary",
        server_configured=True,
        server_reachability="reachable",
        server_auth_state="session_invalid",
        last_known_server_label="Primary Server",
    )
    host = HomeHarness(app)

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        status_text = str(home.query_one("#home-details-body").renderable)
        status_row_text = str(home.query_one("#home-header-line").renderable)
        assert "Mode: Server" in status_text
        assert "Server: Auth expired" in status_text
        assert "Primary Server" in status_row_text


@pytest.mark.asyncio
async def test_home_empty_state_inspector_explains_selected_primary_action():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        has_library_content=False,
    )
    host = HomeHarness(app)

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        selected_title = str(home.query_one("#home-canvas-title").renderable)
        selected_text = str(home.query_one("#home-canvas-lines").renderable)
        assert "Import Library sources" in selected_title
        assert "Library content makes Console and RAG more useful." in selected_text
        assert "Ctrl+P" not in selected_text
        primary = home.query_one("#home-primary-action")
        assert "Import Library sources" in str(primary.label)


@pytest.mark.asyncio
async def test_home_next_actions_offer_distinct_followup_choices():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        has_library_content=False,
    )
    host = HomeHarness(app)

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        canvas_title = str(home.query_one("#home-canvas-title").renderable)
        assert "Import Library sources" in canvas_title
        assert canvas_title.count("Import Library sources") == 1
        primary = home.query_one("#home-primary-action")
        assert "Import Library sources" in str(primary.label)


@pytest.mark.asyncio
async def test_home_next_actions_prioritize_recent_work_without_console_duplicate():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        has_recent_work=True,
    )
    host = HomeHarness(app)

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        canvas_title = str(home.query_one("#home-canvas-title").renderable)
        assert "Start in Console" in canvas_title
        assert canvas_title.count("Start in Console") == 1


@pytest.mark.asyncio
async def test_home_selected_action_uses_user_facing_route_labels():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(model_ready=False)
    host = HomeHarness(app)

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        selected_title = str(home.query_one("#home-canvas-title").renderable)
        selected_text = str(home.query_one("#home-canvas-lines").renderable)
        assert "Set up Console model" in selected_title
        assert "tab_llm" not in selected_text
        assert "Destination: Llm" not in selected_text


@pytest.mark.asyncio
async def test_home_recent_work_empty_state_sets_expectation():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        has_library_content=False,
        has_recent_work=False,
    )
    host = HomeHarness(app)

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        recent_text = str(home.query_one("#home-rail-empty-recent").renderable)
        assert "Runs, chatbooks, imports, and schedules will appear here." in recent_text


@pytest.mark.asyncio
async def test_home_recent_work_available_state_points_to_resume_paths():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        has_recent_work=True,
        recent_work_items=(
            HomeActiveWorkItem(
                item_id="recent:digest",
                title="Nightly digest run",
                source="Watchlists",
                status="completed",
                detail_route="subscriptions",
            ),
        ),
    )
    host = HomeHarness(app)

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        recent_row = next(
            btn for btn in home.query("Button")
            if str(getattr(btn, "row_id", "")) == "recent:digest"
        )
        assert "Nightly digest run" in str(recent_row.label)


@pytest.mark.asyncio
async def test_home_dashboard_uses_bordered_terminal_panes():
    app = _build_test_app()
    host = HomeHarness(app)

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        assert home.query_one("#home-triage-grid").has_class("destination-workbench")
        assert home.query_one("#home-triage-grid").has_class("ds-panel")
        for selector in ["#home-rail", "#home-canvas"]:
            assert home.query_one(selector).has_class("destination-workbench-pane")


@pytest.mark.asyncio
async def test_home_followup_row_stays_compact_below_dashboard_grid():
    app = _build_test_app()
    host = HomeHarness(app)

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        header = home.query_one("#home-header-line")
        dashboard_grid = home.query_one("#home-triage-grid")

        assert dashboard_grid.region.y <= header.region.y + header.region.height + 1
        assert dashboard_grid.region.height >= 12


@pytest.mark.asyncio
async def test_home_primary_action_opens_target_route():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(model_ready=False)
    seen = []
    host = HomeHarness(app, seen)

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        await pilot.click("#home-primary-action")
        await pilot.pause(HOME_MOUNT_PAUSE)

    assert seen[-1] == "settings"
    assert host.seen_contexts[-1] == {
        "category": SettingsCategoryId.PROVIDERS_MODELS.value,
    }


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

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        toolbar = home.query_one("#home-canvas-actions .ds-toolbar")
        for selector in [
            "#home-approve",
            "#home-reject",
            "#home-pause",
            "#home-resume",
            "#home-retry",
            "#home-open-details",
            "#home-open-in-console",
        ]:
            button = home.query_one(selector)
            assert button.has_class("home-canvas-action")
            assert toolbar in button.ancestors


@pytest.mark.asyncio
async def test_home_screen_renders_unread_notification_snapshot_without_controls():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        notification_count=2,
        has_library_content=True,
    )
    host = HomeHarness(app)

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        assert "Notifications: 2 unread" in str(
            home.query_one("#home-details-body").renderable
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

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        await pilot.click("#home-primary-action")
        await pilot.pause(HOME_MOUNT_PAUSE)

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

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        await pilot.click("#home-primary-action")
        await pilot.pause(HOME_MOUNT_PAUSE)

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

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        for selector in [
            "#home-approve",
            "#home-reject",
            "#home-pause",
            "#home-resume",
            "#home-retry",
        ]:
            await pilot.click(selector)
            await pilot.pause(HOME_MOUNT_PAUSE)

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

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        await pilot.click("#home-approve")
        await pilot.pause(HOME_MOUNT_PAUSE)

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

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        await pilot.click("#home-open-details")
        await pilot.pause(HOME_MOUNT_PAUSE)
        await pilot.click("#home-open-in-console")
        await pilot.pause(HOME_MOUNT_PAUSE)

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

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        await pilot.click("#home-open-details")
        await pilot.pause(HOME_MOUNT_PAUSE)
        await pilot.click("#home-open-in-console")
        await pilot.pause(HOME_MOUNT_PAUSE)

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

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        await pilot.click("#home-pause")
        await pilot.pause(HOME_MOUNT_PAUSE)
        await pilot.click("#home-open-details")
        await pilot.pause(HOME_MOUNT_PAUSE)
        await pilot.click("#home-open-in-console")
        await pilot.pause(HOME_MOUNT_PAUSE)

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

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        canvas_title = str(home.query_one("#home-canvas-title").renderable)
        active_work_text = str(home.query_one("#home-canvas-lines").renderable)
        assert "Grounded Answer" in canvas_title
        assert "Artifacts" in active_work_text
        assert "ready" in active_work_text

        await pilot.click("#home-open-details")
        await pilot.pause(HOME_MOUNT_PAUSE)
        await pilot.click("#home-open-in-console")
        await pilot.pause(HOME_MOUNT_PAUSE)

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

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        assert len(home.query("#home-open-chatbook-details")) == 1
        assert len(home.query("#home-open-chatbook-in-console")) == 1
        await pilot.click("#home-open-chatbook-details")
        await pilot.pause(HOME_MOUNT_PAUSE)
        await pilot.click("#home-open-chatbook-in-console")
        await pilot.pause(HOME_MOUNT_PAUSE)

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

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        assert len(home.query("#home-pause")) == 0
        assert len(home.query("#home-resume")) == 0
        assert len(home.query("#home-retry")) == 0
        assert len(home.query("#home-open-in-console")) == 0


@pytest.mark.asyncio
async def test_home_flashcards_due_row_and_control_route_one_hop_to_study():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        flashcards_due_count=12,
    )
    host = HomeHarness(app)

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        row_button = next(
            btn for btn in home.query("Button")
            if str(getattr(btn, "row_id", "")) == HOME_FLASHCARDS_DUE_ROW_ID
        )
        assert "Flashcards due: 12" in str(row_button.label)

        await pilot.click(f"#{row_button.id}")
        await pilot.pause(HOME_MOUNT_PAUSE)

        canvas_title = str(home.query_one("#home-canvas-title").renderable)
        assert canvas_title == "Flashcards due: 12"

        await pilot.click("#home-review-flashcards")
        await pilot.pause(HOME_MOUNT_PAUSE)

    # open_home_flashcards_review() calls app.open_study_screen(initial_section=...),
    # which is verified directly (app_instance is not the running harness App, so
    # its own post_message() does not bubble into host.on_navigate_to_screen here).
    assert app.pending_study_initial_section == "flashcards"


@pytest.mark.asyncio
async def test_home_flashcards_due_row_absent_when_count_zero():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        flashcards_due_count=0,
    )
    host = HomeHarness(app)

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        assert not any(
            str(getattr(btn, "row_id", "")) == HOME_FLASHCARDS_DUE_ROW_ID
            for btn in home.query("Button")
        )
        assert len(home.query("#home-review-flashcards")) == 0


@pytest.mark.asyncio
async def test_home_canvas_primary_control_follows_selection_between_failed_item_and_flashcards():
    """C1: primary emphasis follows the selected row rather than sticking
    to one permanently-accented button. Failed ingest item selected (the
    default selection here, since it is the only attention-worthy item) ->
    Retry is primary; selecting the flashcards-due row flips primary
    emphasis to Review flashcards.

    H2 (fix batch F1b): the global "Review flashcards" shortcut is scoped
    to "no real item selected" -- while the failed ingest item is selected,
    the control is not merely non-primary, it is absent entirely (it isn't
    about this item). It (re)appears once the flashcards row itself is
    selected.
    """
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        flashcards_due_count=12,
        active_work_items=(
            HomeActiveWorkItem(
                item_id="local:ingest:1",
                title="report.xyz",
                source="Library",
                status="failed",
                detail_route="library",
            ),
        ),
    )
    host = HomeHarness(app)

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        retry_button = home.query_one("#home-retry")
        assert retry_button.has_class("console-action-primary")
        assert len(home.query("#home-review-flashcards")) == 0

        row_button = next(
            btn for btn in home.query("Button")
            if str(getattr(btn, "row_id", "")) == HOME_FLASHCARDS_DUE_ROW_ID
        )
        await pilot.click(f"#{row_button.id}")
        await pilot.pause(HOME_MOUNT_PAUSE)

        review_flashcards_button = home.query_one("#home-review-flashcards")
        assert review_flashcards_button.has_class("console-action-primary")
        # The Retry control itself stays available (the failed item is
        # still active work), it just no longer carries primary emphasis.
        assert not home.query_one("#home-retry").has_class("console-action-primary")


@pytest.mark.asyncio
async def test_home_flashcards_due_snapshot_reads_in_memory_db_via_real_worker():
    """F4b (PR #590 review, Qodo): ``_refresh_home_chatbook_artifact_snapshot``
    (``@work(thread=True)``) must not call the flashcards-due provider
    directly on its own worker thread when ChaChaNotes is an in-memory
    SQLite DB -- the connection is thread-local (``threading.local``), so
    the worker thread would open a brand-new, unmigrated ``:memory:``
    connection and ``count_due_flashcards`` would fail, degrading the
    dashboard count to 0 even though a due card exists. Exercises the real
    mount-time worker (not the ``_home_dashboard_test_input`` override) with
    a real in-memory ``CharactersRAGDB`` seeded via the real
    ``TldwCli._local_flashcards_due_count`` provider wiring.
    """
    app = _build_test_app()
    db = CharactersRAGDB(":memory:", client_id="test-client")
    try:
        deck_id = db.create_deck("Biology")
        db.create_flashcard(
            {"deck_id": deck_id, "front": "ATP", "back": "Energy", "tags": "", "type": "basic"}
        )
        app.chachanotes_db = db
        host = HomeHarness(app)

        async with host.run_test(size=HOME_TEST_SIZE) as pilot:
            home = _active_home_screen(host)

            row_button = None
            for _ in range(150):
                row_button = next(
                    (
                        btn
                        for btn in home.query("Button")
                        if str(getattr(btn, "row_id", "")) == HOME_FLASHCARDS_DUE_ROW_ID
                    ),
                    None,
                )
                if row_button is not None:
                    break
                await pilot.pause(0.02)

            assert row_button is not None, "Flashcards-due row never appeared."
            assert "Flashcards due: 1" in str(row_button.label)
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_pending_console_launch_does_not_create_home_live_work_controls():
    app = _build_test_app()
    app.pending_console_launch = {
        "source": "workflows",
        "title": "Daily digest",
        "payload": {},
    }
    host = HomeHarness(app)

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        assert len(home.query("#home-pause")) == 0
        assert len(home.query("#home-open-in-console")) == 0


# --- Library ingest jobs -> Home Running / Needs Attention (L3b Task 6) ---


@pytest.mark.asyncio
async def test_home_running_section_shows_running_library_ingest_job():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        active_work_items=(
            HomeActiveWorkItem(
                item_id="local:ingest:ingest-job-1",
                title="quarterly.txt",
                source="Library",
                status="running",
                detail_route="library",
                console_available=False,
                updated_at="",
            ),
        ),
    )
    host = HomeHarness(app)

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        running_body = home.query_one("#home-rail-section-body-running")
        row_button = next(
            btn for btn in home.query("Button")
            if str(getattr(btn, "row_id", "")) == "local:ingest:ingest-job-1"
        )
        assert row_button in running_body.children
        assert "quarterly.txt" in str(row_button.label)
        assert "Library" in str(row_button.label)
        assert not any(
            str(getattr(btn, "row_id", "")) == "local:ingest:ingest-job-1"
            for btn in home.query_one("#home-rail-section-body-attention").query("Button")
        )


@pytest.mark.asyncio
async def test_home_needs_attention_section_shows_failed_library_ingest_job():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        active_work_items=(
            HomeActiveWorkItem(
                item_id="local:ingest:ingest-job-3",
                title="broken.pdf",
                source="Library",
                status="failed",
                detail_route="library",
                console_available=False,
                updated_at="",
            ),
        ),
    )
    host = HomeHarness(app)

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        attention_body = home.query_one("#home-rail-section-body-attention")
        row_button = next(
            btn for btn in home.query("Button")
            if str(getattr(btn, "row_id", "")) == "local:ingest:ingest-job-3"
        )
        assert row_button in attention_body.children
        assert "broken.pdf" in str(row_button.label)


@pytest.mark.asyncio
async def test_home_running_section_survives_markup_hostile_ingest_job_title():
    """(Critical, L3b Task 6 fix wave) A running ingest job whose source
    filename contains Rich-markup-like bracket syntax (e.g. dropped into
    the Library ingest form) must not crash Home's mount. The raw
    basename flows from ``LibraryIngestJob.source_path`` through the
    real ``LocalNotificationHomeActiveWorkAdapter`` into a Textual
    ``Button`` label in ``HomeRail.compose()`` -- Button labels parse
    Rich markup, so an unescaped hostile title previously raised
    ``MarkupError`` during compose, breaking Home's mount entirely for
    as long as the job stayed queued/running/failed.

    Uses ``a [b="c].txt`` rather than the reviewer's illustrative
    ``weird [/bracket].txt``: title is derived via
    ``Path(source_path).name``, and a literal ``/`` can never survive
    inside a real basename (POSIX reserves it as the separator, so
    ``Path.name`` strips everything before it) -- confirmed empirically
    that ``[/bracket].txt`` alone does not reproduce a crash through this
    exact code path, while an unterminated-quoted-value bracket sequence
    does (same hazard class, same ``MarkupError`` failure mode, verified
    against ``textual.markup.to_content`` directly). Kept short so the
    escaped form survives HomeRail's 20-char row-title truncation intact.

    Deliberately drives the REAL adapter (submitting to
    ``app.library_ingest_jobs``, exactly like the Library ingest canvas
    does) rather than ``_home_dashboard_test_input``, so this exercises
    the actual fix in ``_local_ingest_job_items`` -- not just HomeRail's
    own row-title truncation.
    """
    app = _build_test_app()
    job = app.library_ingest_jobs.submit(source_path='/tmp/a [b="c].txt')
    app.library_ingest_jobs.mark_running(job.job_id)
    host = HomeHarness(app)

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        # Reaching this line at all is the core assertion: a MarkupError
        # during compose/mount would have raised before ``run_test``'s
        # context manager ever returned control here.
        home = _active_home_screen(host)

        running_body = home.query_one("#home-rail-section-body-running")
        row_button = next(
            btn for btn in home.query("Button")
            if str(getattr(btn, "row_id", "")).startswith("local:ingest:")
        )
        assert row_button in running_body.children
        # The rendered (parsed) label shows the literal filename text --
        # rich.markup.escape's backslash is consumed by the markup parser
        # itself, leaving the bracket as a plain character rather than the
        # start of a (would-be-crashing) tag.
        assert 'a [b="c].txt' in str(row_button.label)


def test_app_detail_hook_navigates_library_with_ingest_context_for_handled_ingest_detail():
    """Home's ``Open details`` control on a Library ingest job routes to the
    Library screen carrying the ingest nav-context flag -- the one-hop route
    an ingest job takes from Home's Running/Needs Attention feed back to the
    in-canvas ingest queue (mirrors the subscriptions staging special-case
    right above this test).
    """
    app = _build_test_app()
    adapter = RecordingHomeActiveWorkAdapter(
        responses={
            HomeControlAction.OPEN_DETAILS: HomeControlResult(
                action=HomeControlAction.OPEN_DETAILS,
                status=HomeControlResultStatus.HANDLED,
                message="Opening Library ingest job details.",
                target_id="local:ingest:ingest-job-1",
                target_route="library",
            ),
        }
    )
    app.home_active_work_adapter = adapter
    app.notify = Mock()
    app.post_message = Mock()

    result = app.open_active_home_item_details(
        target_id="local:ingest:ingest-job-1",
        target_route="library",
    )

    assert result.status is HomeControlResultStatus.HANDLED
    app.post_message.assert_called_once()
    posted = app.post_message.call_args.args[0]
    assert posted.screen_name == "library"
    assert posted.screen_context == {LIBRARY_NAV_CONTEXT_INGEST: True}


def test_app_detail_hook_invalidates_cached_library_screen_for_ingest_detail():
    """Home's ``Open details`` deep link must drop any cached Library screen
    before navigating (mirrors ``open_notes_workspace``).

    Library is a CACHEABLE route: without invalidation the deep link switches
    to an already-mounted-then-unmounted cached instance that advances the
    screen stack but never repaints (the live symptom -- the terminal keeps
    rendering Home even though the app is "on" Library). The flashcards deep
    link never hit this because Study is not a cacheable route and always
    builds a fresh screen. This is the RED regression guard for that fix.
    """
    app = _build_test_app()
    adapter = RecordingHomeActiveWorkAdapter(
        responses={
            HomeControlAction.OPEN_DETAILS: HomeControlResult(
                action=HomeControlAction.OPEN_DETAILS,
                status=HomeControlResultStatus.HANDLED,
                message="Opening Library ingest job details.",
                target_id="local:ingest:ingest-job-1",
                target_route="library",
            ),
        }
    )
    app.home_active_work_adapter = adapter
    app.notify = Mock()
    app.post_message = Mock()
    library_sentinel = object()
    app._screen_cache = {TAB_LIBRARY: library_sentinel}

    app.open_active_home_item_details(
        target_id="local:ingest:ingest-job-1",
        target_route="library",
    )

    assert TAB_LIBRARY not in app._screen_cache
    posted = app.post_message.call_args.args[0]
    assert posted.screen_name == "library"
    assert posted.screen_context == {LIBRARY_NAV_CONTEXT_INGEST: True}


@pytest.mark.asyncio
async def test_home_open_details_button_click_navigates_library_for_failed_ingest_job():
    """(L3b live-QA repro) Clicking the REAL ``Open details`` canvas button on
    a failed Library ingest job in Needs Attention must drive the full UI hop
    -- button press -> _activate_home_control -> app.open_active_home_item_details
    -> NavigateToScreen("library", {ingest}) -- not just the direct app-method
    call the sibling test above exercises. Uses the real registry + real
    adapter (submit + mark_failed, exactly like the Library ingest canvas) so
    the failed job flows through ``_local_ingest_job_items`` into active work.
    """
    app = _build_test_app()
    job = app.library_ingest_jobs.submit(source_path="/tmp/report.xyz")
    app.library_ingest_jobs.mark_failed(job.job_id, error="Unsupported extension")
    app.notify = Mock()
    app.post_message = Mock()
    # A Library screen the user already visited is cached under TAB_LIBRARY.
    # The live bug: switching back to this cached-then-unmounted instance from
    # the ingest deep link advanced the screen stack to Library but never
    # repainted (the terminal stayed on Home). The fix drops the cached
    # instance so a fresh Library screen composes + mounts + repaints.
    library_sentinel = object()
    app._screen_cache = {TAB_LIBRARY: library_sentinel}
    host = HomeHarness(app)

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        # Select the failed ingest row in Needs Attention via a real press.
        row_button = next(
            btn for btn in home.query("Button")
            if str(getattr(btn, "row_id", "")) == f"local:ingest:{job.job_id}"
        )
        await pilot.click(row_button)
        await pilot.pause(HOME_MOUNT_PAUSE)

        await pilot.click("#home-open-details")
        await pilot.pause(HOME_MOUNT_PAUSE)

    navigations = [
        call.args[0]
        for call in app.post_message.call_args_list
        if getattr(call.args[0], "screen_name", None) == "library"
    ]
    assert navigations, (
        "Open details did not post a Library navigation; "
        f"post_message calls: {app.post_message.call_args_list}"
    )
    assert navigations[-1].screen_context == {LIBRARY_NAV_CONTEXT_INGEST: True}
    # The stale cached Library screen must be dropped so the deep link lands
    # on a freshly composed, repainted ingest canvas (regression guard for the
    # live "advances to Library but keeps rendering Home" symptom).
    assert app._screen_cache.get(TAB_LIBRARY) is not library_sentinel
    assert TAB_LIBRARY not in app._screen_cache


def test_retry_active_home_item_requeues_ingest_job_via_real_seam():
    """(L3b live-QA repro) Retry on a failed Library ingest job in Needs
    Attention must requeue through the real ``retry_library_ingest_job``
    seam (L3b Task 2) instead of degrading to the adapter's honest
    "not connected to an active run service yet" fallback -- the adapter has
    no visibility into the in-memory ingest job registry, so it can never
    handle this target shape on its own.
    """
    app = _build_test_app()
    job = app.library_ingest_jobs.submit(source_path="/tmp/report.xyz")
    app.library_ingest_jobs.mark_failed(job.job_id, error="Unsupported extension")
    app.notify = Mock()
    jobs_before = len(app.library_ingest_jobs.jobs())

    result = app.retry_active_home_item(target_id=f"local:ingest:{job.job_id}")

    assert result.status is HomeControlResultStatus.HANDLED
    app.notify.assert_called_once_with(
        "Retry queued for report.xyz.",
        severity="information",
    )
    jobs_after = app.library_ingest_jobs.jobs()
    # (L3b AB wave, B1) Retry supersedes the original failed job instead of
    # leaving both visible -- net job count is unchanged.
    assert len(jobs_after) == jobs_before
    newest = jobs_after[0]  # jobs() is newest-first.
    assert newest.job_id != job.job_id
    assert newest.source_path == "/tmp/report.xyz"


def test_retry_active_home_item_unknown_ingest_id_warns_without_requeue():
    """An ingest target id that no longer maps to a ``FAILED`` job (already
    retried, unknown, or finished by the time Retry is pressed) must warn
    honestly rather than silently no-op or crash --
    ``LibraryIngestJobRegistry.requeue`` is a documented no-op (returns
    ``None``) for exactly this case.
    """
    app = _build_test_app()
    app.notify = Mock()
    jobs_before = len(app.library_ingest_jobs.jobs())

    result = app.retry_active_home_item(target_id="local:ingest:does-not-exist")

    assert result.status is HomeControlResultStatus.UNAVAILABLE
    app.notify.assert_called_once_with(
        "This ingest job can no longer be retried.",
        severity="warning",
    )
    assert len(app.library_ingest_jobs.jobs()) == jobs_before


def test_retry_active_home_item_non_ingest_target_still_routes_through_adapter():
    """Non-ingest Retry targets (approvals/watchlist runs/schedules) must be
    unaffected by the ingest special-case and keep degrading through the
    adapter's honest "not connected" fallback -- regression guard for
    existing behavior the ingest fix must not disturb.
    """
    app = _build_test_app()
    app.notify = Mock()

    result = app.retry_active_home_item(target_id="local:watchlist_run:5")

    assert result.status is HomeControlResultStatus.UNAVAILABLE
    app.notify.assert_called_once_with(
        "Retry is not connected to an active run service yet. "
        "Open details or Console to inspect the work.",
        severity="warning",
    )


@pytest.mark.asyncio
async def test_home_retry_button_click_requeues_failed_ingest_job():
    """(L3b live-QA repro) Clicking the REAL ``Retry`` canvas button on a
    failed Library ingest job in Needs Attention must drive the full UI hop
    -- button press -> _activate_home_control -> app.retry_active_home_item
    -> the real ``retry_library_ingest_job`` requeue seam -- not just the
    direct app-method call the sibling test above exercises, and not the
    adapter's generic "not connected to an active run service yet" toast
    (the live-QA finding this test guards against). Uses the real registry
    (submit + mark_failed, exactly like the Library ingest canvas), mirroring
    ``test_home_open_details_button_click_navigates_library_for_failed_ingest_job``.
    """
    app = _build_test_app()
    job = app.library_ingest_jobs.submit(source_path="/tmp/report.xyz")
    app.library_ingest_jobs.mark_failed(job.job_id, error="Unsupported extension")
    app.notify = Mock()
    jobs_before = len(app.library_ingest_jobs.jobs())
    host = HomeHarness(app)

    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)

        # Select the failed ingest row in Needs Attention via a real press.
        row_button = next(
            btn for btn in home.query("Button")
            if str(getattr(btn, "row_id", "")) == f"local:ingest:{job.job_id}"
        )
        await pilot.click(row_button)
        await pilot.pause(HOME_MOUNT_PAUSE)

        await pilot.click("#home-retry")
        await pilot.pause(HOME_MOUNT_PAUSE)

    notify_messages = [call.args[0] for call in app.notify.call_args_list]
    assert "Retry queued for report.xyz." in notify_messages, (
        f"Retry did not requeue the failed ingest job; notify calls: {notify_messages}"
    )
    assert not any("not connected" in str(message) for message in notify_messages)
    # (L3b AB wave, B1) Retry supersedes the original failed job -- net job
    # count is unchanged (one hidden, one added).
    assert len(app.library_ingest_jobs.jobs()) == jobs_before
