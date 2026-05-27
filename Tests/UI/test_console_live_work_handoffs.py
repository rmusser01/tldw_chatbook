"""Console live-work launch and staged-context handoff boundary tests."""

import threading
import time
from pathlib import Path
from unittest.mock import Mock

import pytest
from textual.app import App

from Tests.UI.test_destination_shells import DestinationHarness, _wait_for_selector
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Home.dashboard_state import HomeActiveWorkItem, HomeDashboardInput
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.schedules_screen import SchedulesScreen
from tldw_chatbook.UI.Screens.workflows_screen import WorkflowsScreen


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE3_STATUS_CARD_EVIDENCE = (
    REPO_ROOT / "Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-console-live-work-status-card-seam.md"
)
PHASE3_HOME_WC_CONSOLE_EVIDENCE = (
    REPO_ROOT / "Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-home-wc-active-work-console-launch.md"
)
PHASE3_CONSOLE_WC_ACTION_EVIDENCE = (
    REPO_ROOT / "Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-console-wc-action-routing.md"
)
PHASE3_CONSOLE_SOURCE_READINESS_EVIDENCE = (
    REPO_ROOT / "Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-console-live-work-source-readiness.md"
)
PHASE3_WC_DESTINATION_CONSOLE_EVIDENCE = (
    REPO_ROOT / "Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-wc-destination-console-launch.md"
)
PHASE3_SCHEDULES_CONSOLE_EVIDENCE = (
    REPO_ROOT / "Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-schedules-console-launch.md"
)
PHASE3_SCHEDULES_DIGEST_CONSOLE_EVIDENCE = (
    REPO_ROOT / "Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-schedules-digest-console-launch.md"
)
PHASE3_RAG_CONSOLE_EVIDENCE = (
    REPO_ROOT / "Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-rag-search-console-launch.md"
)
PHASE3_ARTIFACTS_CHATBOOK_CONSOLE_EVIDENCE = (
    REPO_ROOT / "Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-artifacts-chatbook-console-launch.md"
)
PHASE3_WORKFLOWS_CONSOLE_EVIDENCE = (
    REPO_ROOT / "Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-workflows-console-launch.md"
)


def _load_console_live_work_contract():
    try:
        from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
    except ModuleNotFoundError:
        pytest.fail("Console live-work launch contract module is missing")
    return ConsoleLiveWorkLaunch


def _load_console_live_work_status_card_state():
    try:
        from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkStatusCardState
    except ImportError:
        pytest.fail("Console live-work status card state is missing")
    return ConsoleLiveWorkStatusCardState


def _load_console_live_work_source_readiness_state():
    try:
        from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkSourceReadinessState
    except ImportError:
        pytest.fail("Console live-work source readiness state is missing")
    return ConsoleLiveWorkSourceReadinessState


class ConsoleHarness(App):
    def __init__(self, app_instance):
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(ChatScreen(self.app_instance))


def _active_console_screen(host: ConsoleHarness):
    return host.screen_stack[-1]


def _screen_static_text(screen) -> str:
    return " ".join(str(widget.renderable) for widget in screen.query("Static"))


class StaticHomeActiveWorkAdapter:
    def __init__(self, items):
        self.items = tuple(items)
        self.build_calls = []

    def build_dashboard_input(self, *, providers_models, has_recent_work):
        self.build_calls.append(
            {
                "providers_models": providers_models,
                "has_recent_work": has_recent_work,
            }
        )
        return HomeDashboardInput(active_work_items=self.items)

    def handle_control(self, action, *, target_id=None, target_route=None):
        raise AssertionError(f"Unexpected direct control call: {action} {target_id} {target_route}")


class ThreadRecordingHomeActiveWorkAdapter(StaticHomeActiveWorkAdapter):
    def __init__(self, items):
        super().__init__(items)
        self.call_threads = []

    def build_dashboard_input(self, *, providers_models, has_recent_work):
        self.call_threads.append(threading.get_ident())
        return super().build_dashboard_input(
            providers_models=providers_models,
            has_recent_work=has_recent_work,
        )


class RaisingHomeActiveWorkAdapter:
    def build_dashboard_input(self, *, providers_models, has_recent_work):
        raise RuntimeError("adapter unavailable")


class RotatingHomeActiveWorkAdapter:
    def __init__(self, *snapshots):
        self.snapshots = tuple(tuple(snapshot) for snapshot in snapshots)
        self.build_calls = 0

    def build_dashboard_input(self, *, providers_models, has_recent_work):
        snapshot_index = min(self.build_calls, len(self.snapshots) - 1)
        self.build_calls += 1
        return HomeDashboardInput(active_work_items=self.snapshots[snapshot_index])


class FailOnceHomeActiveWorkAdapter:
    def __init__(self, items):
        self.items = tuple(items)
        self.build_calls = 0

    def build_dashboard_input(self, *, providers_models, has_recent_work):
        self.build_calls += 1
        if self.build_calls == 1:
            raise RuntimeError("temporary adapter failure")
        return HomeDashboardInput(active_work_items=self.items)


class StaticReadingDigestService:
    def __init__(self, outputs):
        self.outputs = tuple(outputs)
        self.calls = []

    def list_reading_digest_outputs(self, *, schedule_id=None, limit=50, offset=0):
        self.calls.append(
            {
                "schedule_id": schedule_id,
                "limit": limit,
                "offset": offset,
            }
        )
        return {
            "items": list(self.outputs),
            "total": len(self.outputs),
            "limit": limit,
            "offset": offset,
        }


class ThreadRecordingReadingDigestService(StaticReadingDigestService):
    def __init__(self, outputs):
        super().__init__(outputs)
        self.call_threads = []

    def list_reading_digest_outputs(self, *, schedule_id=None, limit=50, offset=0):
        self.call_threads.append(threading.get_ident())
        return super().list_reading_digest_outputs(
            schedule_id=schedule_id,
            limit=limit,
            offset=offset,
        )


class StaticLocalChatbookService:
    def __init__(self, chatbooks):
        self.chatbooks = tuple(chatbooks)
        self.calls = []

    async def list_chatbooks(self, *, q=None, limit=100, offset=0, **kwargs):
        self.calls.append(
            {
                "q": q,
                "limit": limit,
                "offset": offset,
                "kwargs": dict(kwargs),
            }
        )
        return list(self.chatbooks)[int(offset) : int(offset) + int(limit)]


async def _wait_for_destination_recovery_state(
    screen,
    pilot,
    *,
    button_selector: str,
    static_selector: str,
    expected_tooltip: str,
    expected_fragments: tuple[str, ...],
    timeout: float = 2.0,
) -> tuple[object, str]:
    deadline = time.monotonic() + timeout
    last_recovery_text = ""
    while time.monotonic() < deadline:
        if screen.query(button_selector) and screen.query(static_selector):
            button = screen.query_one(button_selector)
            last_recovery_text = str(screen.query_one(static_selector).renderable)
            if (
                button.disabled is True
                and str(button.tooltip) == expected_tooltip
                and all(fragment in last_recovery_text for fragment in expected_fragments)
            ):
                return button, last_recovery_text
        await pilot.pause(0.02)
    raise AssertionError(
        "Destination recovery state did not become ready. "
        f"selector={static_selector!r} last_text={last_recovery_text!r}"
    )


@pytest.mark.parametrize(
    (
        "route",
        "button_selector",
        "static_selector",
        "service_setup",
        "expected_fragments",
        "expected_tooltip",
    ),
    [
        (
            "acp",
            "#acp-launch-agent",
            "#acp-empty-state",
            "default",
            (
                "Runtime not configured",
                "Unavailable: ACP agent launch.",
                "Why: no ACP-compatible runtime is configured.",
                "Next: Configure ACP runtime setup in ACP before launch.",
                "Recovery: ACP.",
                "Owner: ACP runtime.",
            ),
            "Configure an ACP-compatible runtime in ACP before launching an ACP agent.",
        ),
        (
            "schedules",
            "#schedules-follow-in-console",
            "#schedules-console-unavailable",
            "empty-schedules",
            (
                "Select an active run",
                "Unavailable: Console follow for Schedules.",
                "Why: no active schedule run or reading digest output is available.",
                "Next: Start or select a schedule run before opening it in Console.",
                "Recovery: Schedules.",
                "Owner: local schedule data.",
            ),
            "Start or select a schedule run before opening it in Console.",
        ),
        (
            "workflows",
            "#workflows-launch-in-console",
            "#workflows-console-unavailable",
            "empty-workflows",
            (
                "Select an active run",
                "Unavailable: Console launch for Workflows.",
                "Why: no active workflow run is available.",
                "Next: Start or select a workflow run before opening it in Console.",
                "Recovery: Workflows.",
                "Owner: local workflow data.",
            ),
            "Start or select a workflow run before opening it in Console.",
        ),
        (
            "artifacts",
            "#artifacts-use-in-console",
            "#artifacts-console-unavailable",
            "empty-chatbooks",
            (
                "Select an artifact",
                "Unavailable: Console launch for Chatbook artifacts.",
                "Why: no local Chatbook artifact exists.",
                "Next: Create or import a Chatbook artifact before opening it in Console.",
                "Recovery: Artifacts.",
                "Owner: local Chatbook service.",
            ),
            "Create or import a Chatbook artifact before opening it in Console.",
        ),
    ],
)
@pytest.mark.asyncio
async def test_phase_five_destination_blockers_expose_taxonomy_recovery_fields(
    route,
    button_selector,
    static_selector,
    service_setup,
    expected_fragments,
    expected_tooltip,
):
    app = _build_test_app()
    if service_setup == "empty-schedules":
        app.home_active_work_adapter = StaticHomeActiveWorkAdapter(())
        app.local_media_reading_service = StaticReadingDigestService(())
    elif service_setup == "empty-workflows":
        app.home_active_work_adapter = StaticHomeActiveWorkAdapter(())
    elif service_setup == "empty-chatbooks":
        app.local_chatbook_service = StaticLocalChatbookService(())

    host = DestinationHarness(app, route)

    async with host.run_test(size=(180, 40)) as pilot:
        screen = _active_console_screen(host)
        button, recovery_text = await _wait_for_destination_recovery_state(
            screen,
            pilot,
            button_selector=button_selector,
            static_selector=static_selector,
            expected_tooltip=expected_tooltip,
            expected_fragments=expected_fragments,
        )

        assert button.disabled is True
        assert str(button.tooltip) == expected_tooltip
        for fragment in expected_fragments:
            assert fragment in recovery_text


class RaisingLocalChatbookService:
    async def list_chatbooks(self, *, q=None, limit=100, offset=0, **kwargs):
        raise RuntimeError("registry read failed")


class StaticWatchlistSnapshotService:
    async def list_watch_items(self, **kwargs):
        return []


class StaticReadItLaterSnapshotService:
    async def list_read_it_later(self, **kwargs):
        return {"items": [], "total": 0}


def test_app_exposes_open_console_for_live_work_helper():
    app = _build_test_app()

    assert hasattr(app, "open_console_for_live_work")


def test_console_live_work_launch_contract_normalizes_defaults_and_metadata():
    ConsoleLiveWorkLaunch = _load_console_live_work_contract()

    launch = ConsoleLiveWorkLaunch.from_values(
        source=" workflows ",
        title=" ",
        payload={"run_id": "run-1", "attempt": 2},
        status=" running ",
        recovery=" Workflow is starting. ",
        action_label=" Open workflow run ",
    )

    assert launch.source == "workflows"
    assert launch.title == "Untitled"
    assert launch.payload == {"run_id": "run-1", "attempt": 2}
    assert launch.status == "running"
    assert launch.recovery == "Workflow is starting."
    assert launch.action_label == "Open workflow run"
    assert launch.to_pending_payload() == {
        "source": "workflows",
        "title": "Untitled",
        "payload": {"run_id": "run-1", "attempt": 2},
        "status": "running",
        "recovery": "Workflow is starting.",
        "action_label": "Open workflow run",
    }


def test_open_console_for_live_work_routes_to_chat_route():
    ConsoleLiveWorkLaunch = _load_console_live_work_contract()
    app = _build_test_app()
    seen = []
    app.post_message = lambda message: seen.append(getattr(message, "screen_name", None))

    app.open_console_for_live_work(
        source="workflows",
        title="Daily digest",
        payload={"run_id": "run-1"},
        status="running",
        recovery="Workflow is starting.",
        action_label="Open workflow run",
    )

    assert seen == ["chat"]
    assert isinstance(app.pending_console_launch, ConsoleLiveWorkLaunch)
    assert app.pending_console_launch.to_pending_payload() == {
        "source": "workflows",
        "title": "Daily digest",
        "payload": {"run_id": "run-1"},
        "status": "running",
        "recovery": "Workflow is starting.",
        "action_label": "Open workflow run",
    }


def test_open_console_for_live_work_preserves_minimal_call_defaults():
    ConsoleLiveWorkLaunch = _load_console_live_work_contract()
    app = _build_test_app()
    app.post_message = lambda message: None

    app.open_console_for_live_work(source="workflows", title="Daily digest")

    assert isinstance(app.pending_console_launch, ConsoleLiveWorkLaunch)
    assert app.pending_console_launch.to_pending_payload() == {
        "source": "workflows",
        "title": "Daily digest",
        "payload": {},
        "status": "pending",
        "recovery": "Console has staged this live-work request.",
        "action_label": "Open in Console",
    }


def test_console_live_work_status_card_state_derives_stable_rows_from_launch():
    ConsoleLiveWorkLaunch = _load_console_live_work_contract()
    ConsoleLiveWorkStatusCardState = _load_console_live_work_status_card_state()
    launch = ConsoleLiveWorkLaunch.from_values(
        source="workflows",
        title="Daily digest",
        payload={"attempt": 2, "run_id": "run-1"},
        status="running",
        recovery="Workflow is starting.",
        action_label="Open workflow run",
    )

    card_state = ConsoleLiveWorkStatusCardState.from_launch(launch)

    assert card_state.container_id == "console-pending-launch-card"
    assert "console-live-work-status-card" in card_state.container_classes
    assert card_state.badge_text == "Pending Console launch"
    rows_by_id = {row.widget_id: row.text for row in card_state.rows}
    assert rows_by_id == {
        "console-live-work-source": "Source: workflows",
        "console-live-work-title": "Title: Daily digest",
        "console-live-work-status": "Status: running",
        "console-live-work-recovery": "Recovery: Workflow is starting.",
        "console-live-work-action": "Action: Open workflow run",
        "console-live-work-payload-attempt": "attempt: 2",
        "console-live-work-payload-run-id": "run_id: run-1",
    }
    payload_row = next(row for row in card_state.rows if row.widget_id == "console-live-work-payload-run-id")
    assert "console-live-work-payload-row" in payload_row.classes


def test_console_live_work_status_card_state_exposes_wc_primary_action():
    ConsoleLiveWorkLaunch = _load_console_live_work_contract()
    ConsoleLiveWorkStatusCardState = _load_console_live_work_status_card_state()
    launch = ConsoleLiveWorkLaunch.from_values(
        source="Watchlists",
        title="Daily security feed",
        payload={"target_id": "local:watchlist_run:91", "run_id": 91},
        status="failed",
        recovery="Review the Watchlists run details or retry from Watchlists.",
        action_label="Open Watchlists run",
    )

    card_state = ConsoleLiveWorkStatusCardState.from_launch(launch)

    assert card_state.primary_action is not None
    assert card_state.primary_action.widget_id == "console-live-work-primary-action"
    assert card_state.primary_action.label == "Open Watchlists run"
    assert card_state.primary_action.target_route == "subscriptions"
    assert card_state.primary_action.target_id == "local:watchlist_run:91"


def test_console_live_work_status_card_state_keeps_unsupported_payloads_non_actionable():
    ConsoleLiveWorkLaunch = _load_console_live_work_contract()
    ConsoleLiveWorkStatusCardState = _load_console_live_work_status_card_state()
    launch = ConsoleLiveWorkLaunch.from_values(
        source="workflows",
        title="Daily digest",
        payload={"run_id": "run-1"},
        status="running",
        recovery="Workflow detail routing is not wired yet.",
        action_label="Open workflow run",
    )

    card_state = ConsoleLiveWorkStatusCardState.from_launch(launch)

    assert card_state.primary_action is None


def test_console_live_work_source_readiness_marks_connected_sources_and_future_sources_unavailable():
    ConsoleLiveWorkSourceReadinessState = _load_console_live_work_source_readiness_state()

    state = ConsoleLiveWorkSourceReadinessState.default()

    assert state.container_id == "console-live-work-source-readiness"
    assert "console-live-work-source-readiness" in state.container_classes
    rows_by_id = {row.widget_id: row for row in state.rows}
    assert rows_by_id["console-live-work-source-wc"].text == (
        "Watchlists: Connected - Home run details."
    )
    assert "console-live-work-source-connected" in rows_by_id["console-live-work-source-wc"].classes
    assert rows_by_id["console-live-work-source-schedules"].text == (
        "Schedules: Connected - Open job context."
    )
    assert "console-live-work-source-connected" in rows_by_id["console-live-work-source-schedules"].classes
    assert rows_by_id["console-live-work-source-rag"].text == (
        "RAG: Connected - Stage search evidence."
    )
    assert "console-live-work-source-connected" in rows_by_id["console-live-work-source-rag"].classes
    assert rows_by_id["console-live-work-source-workflows"].text == (
        "Workflows: Connected - Stage run context."
    )
    assert "console-live-work-source-connected" in rows_by_id["console-live-work-source-workflows"].classes
    assert rows_by_id["console-live-work-source-artifacts"].text == (
        "Artifacts: Connected - Launch Chatbooks."
    )
    assert "console-live-work-source-connected" in rows_by_id["console-live-work-source-artifacts"].classes
    assert rows_by_id["console-live-work-source-acp"].text == (
        "ACP: Blocked - Configure ACP runtime."
    )
    assert "console-live-work-source-unavailable" in rows_by_id["console-live-work-source-acp"].classes
    for source_id in ("console-live-work-source-mcp",):
        assert "Not wired" in rows_by_id[source_id].text
        assert "console-live-work-source-unavailable" in rows_by_id[source_id].classes


def test_console_live_work_source_readiness_reflects_acp_runtime_state():
    ConsoleLiveWorkSourceReadinessState = _load_console_live_work_source_readiness_state()

    blocked = ConsoleLiveWorkSourceReadinessState.from_acp_runtime_status("not_configured")
    starting = ConsoleLiveWorkSourceReadinessState.from_acp_runtime_status("starting")
    running = ConsoleLiveWorkSourceReadinessState.from_acp_runtime_status("running")
    failed = ConsoleLiveWorkSourceReadinessState.from_acp_runtime_status("failed")

    blocked_rows = {row.widget_id: row for row in blocked.rows}
    starting_rows = {row.widget_id: row for row in starting.rows}
    running_rows = {row.widget_id: row for row in running.rows}
    failed_rows = {row.widget_id: row for row in failed.rows}

    assert blocked_rows["console-live-work-source-acp"].text == (
        "ACP: Blocked - Configure ACP runtime."
    )
    assert "console-live-work-source-unavailable" in blocked_rows["console-live-work-source-acp"].classes
    assert starting_rows["console-live-work-source-acp"].text == (
        "ACP: Starting - Waiting for runtime."
    )
    assert running_rows["console-live-work-source-acp"].text == (
        "ACP: Connected - Follow ACP session."
    )
    assert "console-live-work-source-connected" in running_rows["console-live-work-source-acp"].classes
    assert failed_rows["console-live-work-source-acp"].text == (
        "ACP: Failed - Review ACP runtime."
    )


def test_console_live_work_primary_action_routes_acp_session_details():
    ConsoleLiveWorkLaunch = _load_console_live_work_contract()
    ConsoleLiveWorkStatusCardState = _load_console_live_work_status_card_state()
    launch = ConsoleLiveWorkLaunch.from_values(
        source="ACP",
        title="Research agent",
        payload={"target_id": "local:acp_session:session-1", "session_id": "session-1"},
        status="running",
        recovery="Console can follow this ACP session payload.",
        action_label="Open ACP session",
    )

    card_state = ConsoleLiveWorkStatusCardState.from_launch(launch)

    assert card_state.primary_action is not None
    assert card_state.primary_action.label == "Open ACP session"
    assert card_state.primary_action.target_route == "acp"
    assert card_state.primary_action.target_id == "local:acp_session:session-1"


def test_app_console_live_work_primary_action_routes_wc_run_details():
    ConsoleLiveWorkLaunch = _load_console_live_work_contract()
    app = _build_test_app()
    app.post_message = Mock()
    app.notify = Mock()
    launch = ConsoleLiveWorkLaunch.from_values(
        source="Watchlists",
        title="Daily security feed",
        payload={"target_id": "local:watchlist_run:91", "run_id": 91},
        status="failed",
        recovery="Review the Watchlists run details or retry from Watchlists.",
        action_label="Open Watchlists run",
    )

    handled = app.open_console_live_work_primary_action(launch)

    assert handled is True
    assert app.pending_subscription_initial_tab == "watchlist-runs"
    assert app.pending_subscription_watchlist_run_id == "local:watchlist_run:91"
    app.post_message.assert_called_once()
    assert app.post_message.call_args.args[0].screen_name == "subscriptions"
    app.notify.assert_not_called()


def test_phase3_status_card_tracking_evidence_links_task_and_roadmap():
    evidence = PHASE3_STATUS_CARD_EVIDENCE.read_text()
    readme = (REPO_ROOT / "Docs/superpowers/qa/unified-shell/phase-3/README.md").read_text()
    roadmap = (REPO_ROOT / "Docs/superpowers/trackers/unified-shell-maturity-roadmap.md").read_text()
    task = (
        REPO_ROOT
        / "backlog/tasks/task-3.2 - Phase-3.2-Add-Console-live-work-status-card-seam.md"
    ).read_text()

    assert "TASK-3.2" in evidence
    assert "ConsoleLiveWorkStatusCardState" in evidence
    assert "2026-05-03-console-live-work-status-card-seam.md" in readme
    assert "Phase 3.2: Add Console live-work status card seam - `TASK-3.2`" in roadmap
    assert "`TASK-3`, `TASK-3.1`, `TASK-3.2`" in roadmap
    assert "ConsoleLiveWorkStatusCardState" in task


def test_phase3_home_wc_console_launch_tracking_evidence_links_task_and_roadmap():
    evidence = PHASE3_HOME_WC_CONSOLE_EVIDENCE.read_text()
    readme = (REPO_ROOT / "Docs/superpowers/qa/unified-shell/phase-3/README.md").read_text()
    roadmap = (REPO_ROOT / "Docs/superpowers/trackers/unified-shell-maturity-roadmap.md").read_text()
    task = (
        REPO_ROOT
        / "backlog/tasks/task-3.3 - Phase-3.3-Open-Home-WC-active-work-in-Console.md"
    ).read_text()

    assert "TASK-3.3" in evidence
    assert "Home W+C active-work" in evidence
    assert "ConsoleLiveWorkLaunch" in evidence
    assert "2026-05-03-home-wc-active-work-console-launch.md" in readme
    assert "Phase 3.3: Open Home W+C active work in Console - `TASK-3.3`" in roadmap
    assert "`TASK-3`, `TASK-3.1`, `TASK-3.2`, `TASK-3.3`" in roadmap
    assert "ConsoleLiveWorkLaunch" in task


def test_phase3_console_wc_action_tracking_evidence_links_task_and_roadmap():
    evidence = PHASE3_CONSOLE_WC_ACTION_EVIDENCE.read_text()
    readme = (REPO_ROOT / "Docs/superpowers/qa/unified-shell/phase-3/README.md").read_text()
    roadmap = (REPO_ROOT / "Docs/superpowers/trackers/unified-shell-maturity-roadmap.md").read_text()
    task = (
        REPO_ROOT
        / "backlog/tasks/task-3.4 - Phase-3.4-Route-Console-WC-live-work-actions.md"
    ).read_text()

    assert "TASK-3.4" in evidence
    assert "Console live-work action" in evidence
    assert "console-live-work-primary-action" in evidence
    assert "2026-05-03-console-wc-action-routing.md" in readme
    assert "Phase 3.4: Route Console W+C live-work actions - `TASK-3.4`" in roadmap
    assert "`TASK-3`, `TASK-3.1`, `TASK-3.2`, `TASK-3.3`, `TASK-3.4`" in roadmap
    assert "console-live-work-primary-action" in task


def test_phase3_console_source_readiness_tracking_evidence_links_task_and_roadmap():
    evidence = PHASE3_CONSOLE_SOURCE_READINESS_EVIDENCE.read_text()

    readme = (REPO_ROOT / "Docs/superpowers/qa/unified-shell/phase-3/README.md").read_text()
    roadmap = (REPO_ROOT / "Docs/superpowers/trackers/unified-shell-maturity-roadmap.md").read_text()
    task = (
        REPO_ROOT
        / "backlog/tasks/task-3.6 - Phase-3.6-Show-Console-live-work-source-readiness.md"
    ).read_text()

    assert "TASK-3.6" in evidence
    assert "ConsoleLiveWorkSourceReadinessState" in evidence
    assert "2026-05-03-console-live-work-source-readiness.md" in readme
    assert "Phase 3.6: Show Console live-work source readiness - `TASK-3.6`" in roadmap
    assert "`TASK-3`, `TASK-3.1`, `TASK-3.2`, `TASK-3.3`, `TASK-3.4`, `TASK-3.5`, `TASK-3.6`" in roadmap
    assert "ConsoleLiveWorkSourceReadinessState" in task


def test_phase3_wc_destination_console_tracking_evidence_links_task_and_roadmap():
    evidence = PHASE3_WC_DESTINATION_CONSOLE_EVIDENCE.read_text()
    readme = (REPO_ROOT / "Docs/superpowers/qa/unified-shell/phase-3/README.md").read_text()
    roadmap = (REPO_ROOT / "Docs/superpowers/trackers/unified-shell-maturity-roadmap.md").read_text()
    task = (
        REPO_ROOT
        / "backlog/tasks/task-3.5 - Phase-3.5-Launch-latest-WC-run-from-WC-into-Console.md"
    ).read_text()

    assert "TASK-3.5" in evidence
    assert "W+C destination Console follow" in evidence
    assert "watchlists-follow-in-console" in evidence
    assert "2026-05-03-wc-destination-console-launch.md" in readme
    assert "Phase 3.5: Launch latest W+C run from W+C into Console - `TASK-3.5`" in roadmap
    assert "`TASK-3`, `TASK-3.1`, `TASK-3.2`, `TASK-3.3`, `TASK-3.4`, `TASK-3.5`" in roadmap
    assert "watchlists-follow-in-console" in task


def test_phase3_schedules_console_tracking_evidence_links_task_and_roadmap():
    evidence = PHASE3_SCHEDULES_CONSOLE_EVIDENCE.read_text()
    readme = (REPO_ROOT / "Docs/superpowers/qa/unified-shell/phase-3/README.md").read_text()
    roadmap = (REPO_ROOT / "Docs/superpowers/trackers/unified-shell-maturity-roadmap.md").read_text()
    task = (
        REPO_ROOT
        / "backlog/tasks/task-3.7 - Phase-3.7-Launch-active-Schedules-run-from-Schedules-into-Console.md"
    ).read_text()

    assert "TASK-3.7" in evidence
    assert "Schedules destination Console follow" in evidence
    assert "schedules-follow-in-console" in evidence
    assert "2026-05-03-schedules-console-launch.md" in readme
    assert "Phase 3.7: Launch active Schedules run from Schedules into Console - `TASK-3.7`" in roadmap
    assert (
        "`TASK-3`, `TASK-3.1`, `TASK-3.2`, `TASK-3.3`, `TASK-3.4`, "
        "`TASK-3.5`, `TASK-3.6`, `TASK-3.7`"
    ) in roadmap
    assert "schedules-follow-in-console" in task


def test_phase3_schedules_digest_console_tracking_evidence_links_task_and_roadmap():
    evidence = PHASE3_SCHEDULES_DIGEST_CONSOLE_EVIDENCE.read_text()
    readme = (REPO_ROOT / "Docs/superpowers/qa/unified-shell/phase-3/README.md").read_text()
    roadmap = (REPO_ROOT / "Docs/superpowers/trackers/unified-shell-maturity-roadmap.md").read_text()
    task = (
        REPO_ROOT
        / "backlog/tasks/task-3.7 - Phase-3.7-Launch-active-Schedules-run-from-Schedules-into-Console.md"
    ).read_text()

    assert "TASK-3.7" in evidence
    assert "Schedules Reading Digest Console Launch" in evidence
    assert "schedules-follow-in-console" in evidence
    assert "2026-05-03-schedules-digest-console-launch.md" in readme
    assert "Schedules Reading Digest Console Launch Evidence" in roadmap
    assert "reading-digest" in task


def test_phase3_rag_console_launch_tracking_evidence_links_task_and_roadmap():
    evidence = PHASE3_RAG_CONSOLE_EVIDENCE.read_text()
    readme = (REPO_ROOT / "Docs/superpowers/qa/unified-shell/phase-3/README.md").read_text()
    roadmap = (REPO_ROOT / "Docs/superpowers/trackers/unified-shell-maturity-roadmap.md").read_text()
    task = (
        REPO_ROOT
        / "backlog/tasks/task-3.8 - Phase-3.8-Launch-RAG-search-result-from-Search-RAG-into-Console.md"
    ).read_text()

    assert "TASK-3.8" in evidence
    assert "Search/RAG" in evidence
    assert "open_console_for_live_work" in evidence
    assert "2026-05-03-rag-search-console-launch.md" in readme
    assert "Phase 3.8: Launch RAG search result from Search/RAG into Console - `TASK-3.8`" in roadmap
    assert (
        "`TASK-3`, `TASK-3.1`, `TASK-3.2`, `TASK-3.3`, `TASK-3.4`, "
        "`TASK-3.5`, `TASK-3.6`, `TASK-3.7`, `TASK-3.8`"
    ) in roadmap
    assert "RAG result" in task


def test_phase3_artifacts_chatbook_console_launch_tracking_evidence_links_task_and_roadmap():
    evidence = PHASE3_ARTIFACTS_CHATBOOK_CONSOLE_EVIDENCE.read_text()
    readme = (REPO_ROOT / "Docs/superpowers/qa/unified-shell/phase-3/README.md").read_text()
    roadmap = (REPO_ROOT / "Docs/superpowers/trackers/unified-shell-maturity-roadmap.md").read_text()
    task = (
        REPO_ROOT
        / "backlog/tasks/task-3.9 - Phase-3.9-Launch-latest-Chatbook-artifact-from-Artifacts-into-Console.md"
    ).read_text()

    assert "TASK-3.9" in evidence
    assert "Artifacts Chatbook Console Launch" in evidence
    assert "artifacts-use-in-console" in evidence
    assert "2026-05-03-artifacts-chatbook-console-launch.md" in readme
    assert "Phase 3.9: Launch latest Chatbook artifact from Artifacts into Console - `TASK-3.9`" in roadmap
    assert (
        "`TASK-3`, `TASK-3.1`, `TASK-3.2`, `TASK-3.3`, `TASK-3.4`, "
        "`TASK-3.5`, `TASK-3.6`, `TASK-3.7`, `TASK-3.8`, `TASK-3.9`"
    ) in roadmap
    assert "Chatbook payload" in task


def test_phase3_workflows_console_tracking_evidence_links_task_and_roadmap():
    evidence = PHASE3_WORKFLOWS_CONSOLE_EVIDENCE.read_text()
    readme = (REPO_ROOT / "Docs/superpowers/qa/unified-shell/phase-3/README.md").read_text()
    roadmap = (REPO_ROOT / "Docs/superpowers/trackers/unified-shell-maturity-roadmap.md").read_text()
    task = (
        REPO_ROOT
        / "backlog/tasks/task-3.10 - Phase-3.10-Launch-active-Workflows-run-from-Workflows-into-Console.md"
    ).read_text()

    assert "TASK-3.10" in evidence
    assert "Workflows destination Console launch" in evidence
    assert "workflows-launch-in-console" in evidence
    assert "2026-05-03-workflows-console-launch.md" in readme
    assert "Phase 3.10: Launch active Workflows run from Workflows into Console - `TASK-3.10`" in roadmap
    assert (
        "`TASK-3`, `TASK-3.1`, `TASK-3.2`, `TASK-3.3`, `TASK-3.4`, "
        "`TASK-3.5`, `TASK-3.6`, `TASK-3.7`, `TASK-3.8`, `TASK-3.9`, `TASK-3.10`"
    ) in roadmap
    assert "workflows-launch-in-console" in task


def test_schedules_console_follow_uses_home_dashboard_app_inputs():
    app = _build_test_app()
    app.providers_models = {"OpenAI": ["gpt-4.1"]}
    app._screen_states = {"chat": {"conversation_id": "c1"}}
    app.home_active_work_adapter = StaticHomeActiveWorkAdapter(
        (
            HomeActiveWorkItem(
                item_id="schedule:run:11",
                title="Daily digest schedule",
                source="Schedules",
                status="running",
                detail_route="schedules",
                console_available=True,
            ),
        )
    )
    screen = SchedulesScreen(app)

    item = screen._latest_console_follow_item()

    assert getattr(item, "item_id", None) == "schedule:run:11"
    assert app.home_active_work_adapter.build_calls == [
        {
            "providers_models": {"OpenAI": ["gpt-4.1"]},
            "has_recent_work": True,
        }
    ]


@pytest.mark.asyncio
async def test_schedules_destination_loads_console_follow_item_off_main_thread():
    main_thread_id = threading.get_ident()
    app = _build_test_app()
    app.home_active_work_adapter = ThreadRecordingHomeActiveWorkAdapter(
        (
            HomeActiveWorkItem(
                item_id="schedule:run:11",
                title="Daily digest schedule",
                source="Schedules",
                status="running",
                detail_route="schedules",
                console_available=True,
            ),
        )
    )
    host = DestinationHarness(app, "schedules")

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)

    assert app.home_active_work_adapter.call_threads
    assert main_thread_id not in app.home_active_work_adapter.call_threads


@pytest.mark.parametrize(
    ("route", "button_id", "expected_copy"),
    [
        (
            "schedules",
            "schedules-follow-in-console",
            "Unavailable: Console follow for Schedules.",
        ),
        (
            "workflows",
            "workflows-launch-in-console",
            "Unavailable: Console launch for Workflows.",
        ),
        (
            "acp",
            "acp-follow-in-console",
            "Unavailable: Console follow for ACP sessions.",
        ),
    ],
)
@pytest.mark.asyncio
async def test_skeletal_destination_console_actions_are_disabled_with_recovery_copy(
    route,
    button_id,
    expected_copy,
):
    app = _build_test_app()
    app.open_console_for_live_work = Mock()
    host = DestinationHarness(app, route)

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        button = host.screen.query_one(f"#{button_id}")
        assert button.disabled is True
        assert "unavailable" in str(button.label).lower()
        assert expected_copy in " ".join(str(widget.renderable) for widget in host.screen.query("Static"))
        await pilot.click(f"#{button_id}")
        await pilot.pause(0.1)

    app.open_console_for_live_work.assert_not_called()


@pytest.mark.asyncio
async def test_schedules_destination_keeps_console_follow_disabled_without_active_run():
    app = _build_test_app()
    app.home_active_work_adapter = StaticHomeActiveWorkAdapter(())
    app.open_active_home_item_in_console = Mock()
    host = DestinationHarness(app, "schedules")

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_console_screen(host)
        button = screen.query_one("#schedules-follow-in-console")

        assert button.disabled is True
        assert str(button.label) == "Console recovery unavailable"
        assert "Unavailable: Console follow for Schedules." in _screen_static_text(screen)
        assert "Next: Start or select a schedule run before opening it in Console." in _screen_static_text(screen)

    app.open_active_home_item_in_console.assert_not_called()


@pytest.mark.asyncio
async def test_schedules_destination_routes_latest_active_run_to_console():
    app = _build_test_app()
    app.home_active_work_adapter = StaticHomeActiveWorkAdapter(
        (
            HomeActiveWorkItem(
                item_id="schedule:run:7",
                title="Daily digest schedule",
                source="Schedules",
                status="failed",
                detail_route="schedules",
                console_available=True,
            ),
            HomeActiveWorkItem(
                item_id="local:watchlist_run:9",
                title="Watchlist run",
                source="Watchlists",
                status="failed",
                detail_route="subscriptions",
                console_available=True,
            ),
        )
    )
    app.open_active_home_item_in_console = Mock()
    host = DestinationHarness(app, "schedules")

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_console_screen(host)
        button = screen.query_one("#schedules-follow-in-console")

        assert button.disabled is False
        assert "Daily digest schedule" in str(button.label)
        assert "failed" in _screen_static_text(screen)

        await pilot.click("#schedules-follow-in-console")
        await pilot.pause(0.1)

    app.open_active_home_item_in_console.assert_called_once_with(
        target_id="schedule:run:7",
        target_route="chat",
    )


@pytest.mark.asyncio
async def test_schedules_inspector_state_matches_active_run_status():
    app = _build_test_app()
    app.home_active_work_adapter = StaticHomeActiveWorkAdapter(
        (
            HomeActiveWorkItem(
                item_id="schedule:run:7",
                title="Daily digest schedule",
                source="Schedules",
                status="failed",
                detail_route="schedules",
                console_available=True,
            ),
        )
    )
    host = DestinationHarness(app, "schedules")

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_console_screen(host)
        screen_text = _screen_static_text(screen)

        assert "Console can follow active schedule run: Daily digest schedule (failed)." in screen_text
        assert "State: failed" in screen_text
        assert "State: ready" not in screen_text
        assert "Retry/backoff: retry available from Schedules" in screen_text


def test_workflows_console_launch_uses_home_dashboard_app_inputs():
    app = _build_test_app()
    app.providers_models = {"OpenAI": ["gpt-4.1"]}
    app._screen_states = {"chat": {"conversation_id": "c1"}}
    app.home_active_work_adapter = StaticHomeActiveWorkAdapter(
        (
            HomeActiveWorkItem(
                item_id="workflow:run:11",
                title="Digest workflow",
                source="Workflows",
                status="running",
                detail_route="workflows",
                console_available=True,
            ),
        )
    )
    screen = WorkflowsScreen(app)

    item = screen._latest_console_follow_item()

    assert getattr(item, "item_id", None) == "workflow:run:11"
    assert app.home_active_work_adapter.build_calls == [
        {
            "providers_models": {"OpenAI": ["gpt-4.1"]},
            "has_recent_work": True,
        }
    ]


def test_workflows_console_launch_accepts_route_style_source():
    app = _build_test_app()
    app.home_active_work_adapter = StaticHomeActiveWorkAdapter(
        (
            HomeActiveWorkItem(
                item_id="workflow:run:12",
                title="Route style workflow",
                source="workflows",
                status="running",
                detail_route="workflows",
                console_available=True,
            ),
        )
    )
    screen = WorkflowsScreen(app)

    item = screen._latest_console_follow_item()

    assert getattr(item, "item_id", None) == "workflow:run:12"


@pytest.mark.asyncio
async def test_workflows_destination_loads_console_follow_item_off_main_thread():
    main_thread_id = threading.get_ident()
    app = _build_test_app()
    app.home_active_work_adapter = ThreadRecordingHomeActiveWorkAdapter(
        (
            HomeActiveWorkItem(
                item_id="workflow:run:11",
                title="Digest workflow",
                source="Workflows",
                status="running",
                detail_route="workflows",
                console_available=True,
            ),
        )
    )
    host = DestinationHarness(app, "workflows")

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)

    assert app.home_active_work_adapter.call_threads
    assert main_thread_id not in app.home_active_work_adapter.call_threads


@pytest.mark.asyncio
async def test_workflows_destination_keeps_console_launch_disabled_without_active_run():
    app = _build_test_app()
    app.home_active_work_adapter = StaticHomeActiveWorkAdapter(())
    app.open_active_home_item_in_console = Mock()
    host = DestinationHarness(app, "workflows")

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_console_screen(host)
        button = screen.query_one("#workflows-launch-in-console")

        assert button.disabled is True
        assert str(button.label) == "Console launch unavailable"
        screen_text = _screen_static_text(screen)
        assert "Unavailable: Console launch for Workflows." in screen_text
        assert "Next: Start or select a workflow run before opening it in Console." in screen_text
        assert "State: blocked" in screen_text
        assert "Console: blocked" in screen_text

    app.open_active_home_item_in_console.assert_not_called()


@pytest.mark.asyncio
async def test_workflows_destination_routes_latest_active_run_to_console():
    app = _build_test_app()
    app.home_active_work_adapter = StaticHomeActiveWorkAdapter(
        (
            HomeActiveWorkItem(
                item_id="workflow:run:7",
                title="Daily digest workflow",
                source="Workflows",
                status="failed",
                detail_route="workflows",
                console_available=True,
            ),
            HomeActiveWorkItem(
                item_id="schedule:run:7",
                title="Schedule run",
                source="Schedules",
                status="failed",
                detail_route="schedules",
                console_available=True,
            ),
        )
    )
    app.open_active_home_item_in_console = Mock()
    host = DestinationHarness(app, "workflows")

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_console_screen(host)
        button = screen.query_one("#workflows-launch-in-console")

        assert button.disabled is False
        assert "Daily digest workflow" in str(button.label)
        screen_text = _screen_static_text(screen)
        assert "failed" in screen_text
        assert "State: failed" in screen_text
        assert "State: ready" not in screen_text

        await pilot.click("#workflows-launch-in-console")
        await pilot.pause(0.1)

    app.open_active_home_item_in_console.assert_called_once_with(
        target_id="workflow:run:7",
        target_route="chat",
    )


@pytest.mark.asyncio
async def test_workflows_destination_treats_pending_status_as_pending_approval():
    app = _build_test_app()
    app.home_active_work_adapter = StaticHomeActiveWorkAdapter(
        (
            HomeActiveWorkItem(
                item_id="workflow:run:8",
                title="Approval workflow",
                source="Workflows",
                status="pending",
                detail_route="workflows",
                console_available=True,
            ),
        )
    )
    host = DestinationHarness(app, "workflows")

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_console_screen(host)
        screen_text = _screen_static_text(screen)

        assert "State: pending" in screen_text
        assert "Approvals: pending" in screen_text
        assert "Approvals: none pending" not in screen_text


@pytest.mark.asyncio
async def test_watchlists_destination_keeps_console_follow_disabled_without_active_run():
    app = _build_test_app()
    app.home_active_work_adapter = StaticHomeActiveWorkAdapter(())
    app.open_active_home_item_in_console = Mock()
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_console_screen(host)
        button = screen.query_one("#watchlists-follow-in-console")

        assert button.disabled is True
        assert str(button.label) == "Console follow unavailable"
        text = _screen_static_text(screen)
        assert "No active Watchlists run is available for Console follow." in text

    app.open_active_home_item_in_console.assert_not_called()


@pytest.mark.asyncio
async def test_watchlists_destination_routes_latest_active_run_to_console():
    app = _build_test_app()
    app.home_active_work_adapter = StaticHomeActiveWorkAdapter(
        (
            HomeActiveWorkItem(
                item_id="local:watchlist_run:5",
                title="Daily security feed",
                source="Watchlists",
                status="failed",
                detail_route="subscriptions",
                console_available=True,
            ),
            HomeActiveWorkItem(
                item_id="local:watchlist_run:9",
                title="Other source",
                source="Watchlists",
                status="queued",
                detail_route="subscriptions",
                console_available=False,
            ),
        )
    )
    app.open_active_home_item_in_console = Mock()
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_console_screen(host)
        button = screen.query_one("#watchlists-follow-in-console")

        assert button.disabled is False
        assert "Daily security feed" in str(button.label)
        assert "failed" in _screen_static_text(screen)

        await pilot.click("#watchlists-follow-in-console")
        await pilot.pause(0.1)

    app.open_active_home_item_in_console.assert_called_once_with(
        target_id="local:watchlist_run:5",
        target_route="chat",
    )


@pytest.mark.asyncio
async def test_watchlists_destination_logs_adapter_failure_and_disables_follow(monkeypatch):
    from tldw_chatbook.UI.Screens import watchlists_collections_screen

    app = _build_test_app()
    app.home_active_work_adapter = RaisingHomeActiveWorkAdapter()
    app.open_active_home_item_in_console = Mock()
    logger = Mock()
    monkeypatch.setattr(watchlists_collections_screen, "logger", logger, raising=False)
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_console_screen(host)
        button = screen.query_one("#watchlists-follow-in-console")

        assert button.disabled is True
        assert "No active Watchlists run is available for Console follow." in _screen_static_text(screen)

    logger.warning.assert_called_once()
    assert "Watchlists Console follow" in logger.warning.call_args.args[0]
    assert logger.warning.call_args.kwargs["exc_info"] is True
    app.open_active_home_item_in_console.assert_not_called()


@pytest.mark.asyncio
async def test_watchlists_destination_retries_console_follow_after_initial_adapter_failure():
    app = _build_test_app()
    app.home_active_work_adapter = FailOnceHomeActiveWorkAdapter(
        (
            HomeActiveWorkItem(
                item_id="local:watchlist_run:11",
                title="Recovered run",
                source="Watchlists",
                status="running",
                detail_route="subscriptions",
                console_available=True,
            ),
        )
    )
    app.watchlist_scope_service = StaticWatchlistSnapshotService()
    app.media_reading_scope_service = StaticReadItLaterSnapshotService()
    app.open_active_home_item_in_console = Mock()
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(180, 40)) as pilot:
        screen = _active_console_screen(host)
        for _ in range(100):
            button = screen.query_one("#watchlists-follow-in-console")
            if "Recovered run" in str(button.label):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(f"Console follow did not recover. Text: {_screen_static_text(screen)}")

        assert button.disabled is False
        await pilot.click("#watchlists-follow-in-console")
        await pilot.pause(0.1)

    assert app.home_active_work_adapter.build_calls >= 2
    app.open_active_home_item_in_console.assert_called_once_with(
        target_id="local:watchlist_run:11",
        target_route="chat",
    )


@pytest.mark.asyncio
async def test_watchlists_destination_click_uses_item_promised_by_button_label():
    app = _build_test_app()
    app.home_active_work_adapter = RotatingHomeActiveWorkAdapter(
        (
            HomeActiveWorkItem(
                item_id="local:watchlist_run:5",
                title="First visible run",
                source="Watchlists",
                status="failed",
                detail_route="subscriptions",
                console_available=True,
            ),
        ),
        (
            HomeActiveWorkItem(
                item_id="local:watchlist_run:7",
                title="Newer unseen run",
                source="Watchlists",
                status="running",
                detail_route="subscriptions",
                console_available=True,
            ),
        ),
    )
    app.open_active_home_item_in_console = Mock()
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_console_screen(host)
        button = screen.query_one("#watchlists-follow-in-console")

        assert "First visible run" in str(button.label)
        assert "Newer unseen run" not in str(button.label)

        await pilot.click("#watchlists-follow-in-console")
        await pilot.pause(0.1)

    app.open_active_home_item_in_console.assert_called_once_with(
        target_id="local:watchlist_run:5",
        target_route="chat",
    )


@pytest.mark.asyncio
async def test_watchlists_destination_escapes_console_follow_markup_labels():
    app = _build_test_app()
    app.home_active_work_adapter = StaticHomeActiveWorkAdapter(
        (
            HomeActiveWorkItem(
                item_id="local:watchlist_run:5",
                title="[red]Daily[/red] feed",
                source="Watchlists",
                status="[bold]failed[/bold]",
                detail_route="subscriptions",
                console_available=True,
            ),
        )
    )
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_console_screen(host)
        button = screen.query_one("#watchlists-follow-in-console")
        static_text = _screen_static_text(screen)

        assert "[red]Daily[/red] feed" in str(button.label)
        assert getattr(button.label, "spans", []) == []
        assert "[red]Daily[/red] feed" in static_text
        assert "[bold]failed[/bold]" in static_text


@pytest.mark.asyncio
async def test_schedules_destination_keeps_console_launch_disabled_without_digest_output():
    app = _build_test_app()
    app.home_active_work_adapter = StaticHomeActiveWorkAdapter(())
    app.local_media_reading_service = StaticReadingDigestService(())
    app.open_console_for_live_work = Mock()
    host = DestinationHarness(app, "schedules")

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_console_screen(host)
        button = screen.query_one("#schedules-follow-in-console")

        assert button.disabled is True
        assert str(button.label) == "Console recovery unavailable"
        assert "Unavailable: Console follow for Schedules." in _screen_static_text(screen)
        assert "Next: Start or select a schedule run before opening it in Console." in _screen_static_text(screen)
        await pilot.click("#schedules-follow-in-console")
        await pilot.pause(0.1)

    assert app.local_media_reading_service.calls == [
        {
            "schedule_id": None,
            "limit": 1,
            "offset": 0,
        }
    ]
    app.open_console_for_live_work.assert_not_called()


@pytest.mark.asyncio
async def test_schedules_destination_loads_digest_output_off_main_thread():
    main_thread_id = threading.get_ident()
    app = _build_test_app()
    app.home_active_work_adapter = StaticHomeActiveWorkAdapter(())
    app.local_media_reading_service = ThreadRecordingReadingDigestService(
        (
            {
                "output_id": 91,
                "schedule_id": "local-digest-12",
                "title": "Morning Digest Output",
                "metadata": {"item_count": 2, "schedule_name": "Morning Digest"},
            },
        )
    )
    host = DestinationHarness(app, "schedules")

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)

    assert app.local_media_reading_service.call_threads
    assert main_thread_id not in app.local_media_reading_service.call_threads


@pytest.mark.asyncio
async def test_schedules_destination_routes_latest_digest_output_to_console():
    app = _build_test_app()
    app.home_active_work_adapter = StaticHomeActiveWorkAdapter(())
    app.local_media_reading_service = StaticReadingDigestService(
        (
            {
                "output_id": 91,
                "schedule_id": "local-digest-12",
                "title": "Morning Digest Output",
                "format": "md",
                "download_url": "local://reading_digest/12/91",
                "created_at": "2026-05-03T08:00:00Z",
                "metadata": {"item_count": 2, "schedule_name": "Morning Digest"},
            },
        )
    )
    app.open_console_for_live_work = Mock()
    host = DestinationHarness(app, "schedules")

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_console_screen(host)
        button = screen.query_one("#schedules-follow-in-console")

        assert button.disabled is False
        assert "Morning Digest Output" in str(button.label)
        text = _screen_static_text(screen)
        assert "Console launch available" in text
        assert "Console recovery unavailable" not in text
        assert "Console can launch latest reading digest output: Morning Digest Output." in text
        assert "State: digest output available" in text
        assert "State: ready" not in text

        await pilot.click("#schedules-follow-in-console")
        await pilot.pause(0.1)

    app.open_console_for_live_work.assert_called_once_with(
        source="schedules",
        title="Morning Digest Output",
        payload={
            "target_id": "local:reading_digest_output:91",
            "output_id": 91,
            "schedule_id": "local-digest-12",
            "schedule_name": "Morning Digest",
            "download_url": "local://reading_digest/12/91",
            "created_at": "2026-05-03T08:00:00Z",
            "item_count": 2,
        },
        status="ready",
        recovery="Review this reading digest output from Schedules or return to Library.",
        action_label="Open schedule output",
    )


@pytest.mark.asyncio
async def test_artifacts_destination_keeps_console_launch_disabled_without_chatbooks():
    app = _build_test_app()
    app.local_chatbook_service = StaticLocalChatbookService(())
    app.open_console_for_live_work = Mock()
    app.open_chat_with_handoff = Mock()
    host = DestinationHarness(app, "artifacts")

    async with host.run_test(size=(180, 40)) as pilot:
        screen = _active_console_screen(host)
        await _wait_for_selector(screen, pilot, "#artifacts-console-unavailable")
        button = screen.query_one("#artifacts-use-in-console")

        assert button.disabled is True
        assert str(button.label) == "Open selected in Console"
        assert "Unavailable: Console launch for Chatbook artifacts." in _screen_static_text(screen)
        assert "Next: Create or import a Chatbook artifact before opening it in Console." in _screen_static_text(
            screen
        )

        await pilot.click("#artifacts-use-in-console")

    app.open_console_for_live_work.assert_not_called()
    app.open_chat_with_handoff.assert_not_called()
    assert app.local_chatbook_service.calls == [{"q": None, "limit": 25, "offset": 0, "kwargs": {}}]


@pytest.mark.asyncio
async def test_artifacts_destination_launches_latest_local_chatbook_in_console():
    app = _build_test_app()
    app.local_chatbook_service = StaticLocalChatbookService(
        (
            {
                "chatbook_id": 41,
                "id": "41",
                "name": "Older Pack",
                "description": "Previous bundle",
                "file_path": "/tmp/older-pack.chatbook",
                "tags": ["archive"],
                "categories": ["Library"],
                "updated_at": "2026-05-02T20:00:00Z",
            },
            {
                "chatbook_id": 42,
                "id": "42",
                "name": "Research Pack",
                "description": "A portable research bundle",
                "file_path": "/tmp/research-pack.chatbook",
                "tags": ["research", "portable"],
                "categories": ["Library"],
                "updated_at": "2026-05-03T20:00:00Z",
            },
        )
    )
    app.open_console_for_live_work = Mock()
    app.open_chat_with_handoff = Mock()
    host = DestinationHarness(app, "artifacts")

    async with host.run_test(size=(180, 40)) as pilot:
        screen = _active_console_screen(host)
        await _wait_for_selector(screen, pilot, "#artifacts-console-available")
        button = screen.query_one("#artifacts-use-in-console")

        assert button.disabled is False
        assert "Research Pack" in str(button.label)
        text = _screen_static_text(screen)
        assert "Open Console for latest Chatbook artifact: Research Pack." in text
        assert "A portable research bundle" in text

        await pilot.click("#artifacts-use-in-console")

    app.open_console_for_live_work.assert_called_once_with(
        source="artifacts",
        title="Research Pack",
        payload={
            "target_id": "local:chatbook:42",
            "chatbook_id": 42,
            "record_id": "42",
            "file_path": "/tmp/research-pack.chatbook",
            "description": "A portable research bundle",
            "tags": "research, portable",
            "categories": "Library",
            "updated_at": "2026-05-03T20:00:00Z",
        },
        status="ready",
        recovery="Review this Chatbook artifact in Console or return to Artifacts.",
        action_label="Open Chatbook artifact",
    )
    app.open_chat_with_handoff.assert_not_called()


@pytest.mark.asyncio
async def test_artifacts_destination_reopens_console_saved_chatbook_with_provenance():
    app = _build_test_app()
    app.local_chatbook_service = StaticLocalChatbookService(
        (
            {
                "chatbook_id": 77,
                "id": "77",
                "name": "Grounded Answer",
                "description": "Saved from Console assistant response. Preview: Grounded answer body.",
                "file_path": "/tmp/grounded-answer.chatbook",
                "tags": ["console", "artifact"],
                "categories": ["Console", "Artifacts"],
                "metadata": {
                    "artifact_source": "console",
                    "artifact_kind": "assistant-response",
                    "conversation_id": "conv-123",
                    "message_id": "msg-456",
                    "message_role": "Assistant",
                    "provider": "OpenAI",
                    "model": "gpt-4.1",
                    "content": "Grounded answer body from saved artifact.",
                    "content_truncated": False,
                },
                "updated_at": "2026-05-05T20:00:00Z",
            },
        )
    )
    app.open_console_for_live_work = Mock()
    host = DestinationHarness(app, "artifacts")

    async with host.run_test(size=(180, 40)) as pilot:
        screen = _active_console_screen(host)
        await _wait_for_selector(screen, pilot, "#artifacts-console-available")
        text = _screen_static_text(screen)

        assert "Open Console for latest Chatbook artifact: Grounded Answer." in text
        assert "Saved from Console assistant response." in text
        assert "OpenAI / gpt-4.1" in text
        assert "Grounded answer body from saved artifact." in text

        await pilot.click("#artifacts-use-in-console")

    app.open_console_for_live_work.assert_called_once()
    launch_kwargs = app.open_console_for_live_work.call_args.kwargs
    assert launch_kwargs["source"] == "artifacts"
    assert launch_kwargs["title"] == "Grounded Answer"
    assert launch_kwargs["payload"] == {
        "target_id": "local:chatbook:77",
        "chatbook_id": 77,
        "record_id": "77",
        "file_path": "/tmp/grounded-answer.chatbook",
        "description": "Saved from Console assistant response. Preview: Grounded answer body.",
        "tags": "console, artifact",
        "categories": "Console, Artifacts",
        "updated_at": "2026-05-05T20:00:00Z",
        "artifact_source": "console",
        "artifact_kind": "assistant-response",
        "conversation_id": "conv-123",
        "message_id": "msg-456",
        "message_role": "Assistant",
        "provider": "OpenAI",
        "model": "gpt-4.1",
        "content_preview": "Grounded answer body from saved artifact.",
        "content_truncated": False,
    }


@pytest.mark.asyncio
async def test_artifacts_destination_reopens_console_saved_chatbook_with_citation_metadata():
    app = _build_test_app()
    app.local_chatbook_service = StaticLocalChatbookService(
        (
            {
                "chatbook_id": 77,
                "id": "77",
                "name": "Grounded Answer",
                "description": "Saved from Console assistant response.",
                "file_path": "/tmp/grounded-answer.chatbook",
                "tags": ["console", "artifact"],
                "categories": ["Console", "Artifacts"],
                "metadata": {
                    "artifact_source": "console",
                    "artifact_kind": "assistant-response",
                    "content": "The credential expired [S1].",
                    "citation_validation": {
                        "status": "validated",
                        "citations": [
                            {
                                "evidence_id": "S1",
                                "source_id": "note-1",
                                "status": "validated",
                                "quote": "The credential expired [S1].",
                            }
                        ],
                        "cited_evidence_ids": ["S1"],
                        "unknown_citation_ids": [],
                        "uncited_evidence_ids": [],
                        "recovery": "",
                    },
                    "evidence_bundle": {
                        "bundle_id": "library-rag:incident",
                        "query": "Why did the incident happen?",
                        "status": "available",
                        "references": [
                            {
                                "evidence_id": "S1",
                                "source_id": "note-1",
                                "source_type": "note",
                                "title": "Incident Review",
                                "snippet": "Expired credential caused the incident.",
                                "authority_label": "Source authority: local",
                                "status": "available",
                            }
                        ],
                    },
                },
                "updated_at": "2026-05-05T20:00:00Z",
            },
        )
    )
    app.open_console_for_live_work = Mock()
    host = DestinationHarness(app, "artifacts")

    async with host.run_test(size=(180, 40)) as pilot:
        screen = _active_console_screen(host)
        await _wait_for_selector(screen, pilot, "#artifacts-console-available")
        await pilot.click("#artifacts-use-in-console")

    launch_payload = app.open_console_for_live_work.call_args.kwargs["payload"]
    assert launch_payload["citation_status"] == "validated"
    assert launch_payload["citation_cited_evidence_ids"] == "S1"
    assert launch_payload["citation_count"] == 1
    assert launch_payload["evidence_bundle_id"] == "library-rag:incident"
    assert launch_payload["evidence_source_count"] == 1
    assert launch_payload["evidence_snippet_count"] == 1


@pytest.mark.asyncio
async def test_artifacts_destination_sanitizes_chatbook_metadata_before_console_launch():
    app = _build_test_app()
    app.local_chatbook_service = StaticLocalChatbookService(
        (
            {
                "chatbook_id": "7",
                "id": "7",
                "name": "Research <script>alert(1)</script> Pack\x00",
                "description": "Open javascript:alert(1) and onerror=bad",
                "file_path": "/tmp/<script>bad</script>.chatbook\x00",
                "tags": ["safe", "<script>tag</script>"],
                "categories": ["onclick=bad", "Library"],
                "metadata": {
                    "artifact_source": "console",
                    "artifact_kind": "assistant-response",
                    "conversation_id": "conv-<script>bad</script>",
                    "message_id": "msg-onclick=bad",
                    "message_role": "Assistant",
                    "provider": "javascript:alert(1)",
                    "model": "onerror=bad",
                    "content": "<script>bad</script> onerror=bad",
                    "content_truncated": False,
                    "citation_validation": {
                        "status": "<script>validated</script>",
                        "cited_evidence_ids": ["S1", "javascript:bad"],
                        "citations": [{"evidence_id": "S1"}],
                    },
                    "evidence_bundle": {
                        "bundle_id": "bundle-onclick=bad",
                        "query": "<script>query</script>",
                        "references": [{"snippet": "safe"}],
                    },
                },
                "updated_at": "2026-05-03T20:00:00Z",
            },
        )
    )
    app.open_console_for_live_work = Mock()
    host = DestinationHarness(app, "artifacts")

    async with host.run_test(size=(180, 40)) as pilot:
        screen = _active_console_screen(host)
        await _wait_for_selector(screen, pilot, "#artifacts-console-available")
        await pilot.click("#artifacts-use-in-console")

    launch_kwargs = app.open_console_for_live_work.call_args.kwargs
    payload_strings = [launch_kwargs["title"]]
    payload_strings.extend(
        str(value)
        for value in launch_kwargs["payload"].values()
        if value is not None
    )
    combined = " ".join(payload_strings).lower()
    assert "\x00" not in combined
    assert "<script" not in combined
    assert "javascript:" not in combined
    assert "onclick=" not in combined
    assert "onerror=" not in combined
    assert "&lt;script&gt;" in combined


@pytest.mark.asyncio
async def test_artifacts_destination_uses_numeric_id_tie_break_for_latest_chatbook():
    app = _build_test_app()
    app.local_chatbook_service = StaticLocalChatbookService(
        (
            {
                "chatbook_id": 9,
                "id": "9",
                "name": "Nine Pack",
                "updated_at": "2026-05-03T20:00:00Z",
            },
            {
                "chatbook_id": 10,
                "id": "10",
                "name": "Ten Pack",
                "updated_at": "2026-05-03T20:00:00Z",
            },
        )
    )
    app.open_console_for_live_work = Mock()
    host = DestinationHarness(app, "artifacts")

    async with host.run_test(size=(180, 40)) as pilot:
        screen = _active_console_screen(host)
        await _wait_for_selector(screen, pilot, "#artifacts-console-available")
        button = screen.query_one("#artifacts-use-in-console")

        assert "Ten Pack" in str(button.label)

        await pilot.click("#artifacts-use-in-console")

    app.open_console_for_live_work.assert_called_once()
    assert app.open_console_for_live_work.call_args.kwargs["payload"]["chatbook_id"] == 10


@pytest.mark.asyncio
async def test_artifacts_destination_consumes_pending_chatbook_target_before_latest_fallback():
    app = _build_test_app()
    app.pending_artifacts_chatbook_target_id = "local:chatbook:77"
    app.local_chatbook_service = StaticLocalChatbookService(
        (
            {
                "chatbook_id": 77,
                "id": "77",
                "name": "Requested Pack",
                "updated_at": "2026-05-01T20:00:00Z",
            },
            {
                "chatbook_id": 99,
                "id": "99",
                "name": "Latest Pack",
                "updated_at": "2026-05-05T20:00:00Z",
            },
        )
    )
    app.open_console_for_live_work = Mock()
    host = DestinationHarness(app, "artifacts")

    async with host.run_test(size=(180, 40)) as pilot:
        screen = _active_console_screen(host)
        await _wait_for_selector(screen, pilot, "#artifacts-console-available")

        text = _screen_static_text(screen)
        assert "Open Console for requested Chatbook artifact: Requested Pack." in text
        assert "Latest Pack" not in text
        assert getattr(app, "pending_artifacts_chatbook_target_id", None) is None

        await pilot.click("#artifacts-use-in-console")

    app.open_console_for_live_work.assert_called_once()
    assert app.open_console_for_live_work.call_args.kwargs["payload"]["target_id"] == "local:chatbook:77"


@pytest.mark.asyncio
async def test_artifacts_destination_distinguishes_chatbook_service_failure_from_empty_state():
    app = _build_test_app()
    app.local_chatbook_service = RaisingLocalChatbookService()
    app.open_console_for_live_work = Mock()
    host = DestinationHarness(app, "artifacts")

    async with host.run_test(size=(180, 40)) as pilot:
        screen = _active_console_screen(host)
        await _wait_for_selector(screen, pilot, "#artifacts-console-unavailable")
        button = screen.query_one("#artifacts-use-in-console")
        text = _screen_static_text(screen)

        assert button.disabled is True
        assert "Unavailable: Console launch for Chatbook artifacts." in text
        assert "Why: the local Chatbook service is unavailable." in text
        assert "Next: Retry Artifacts after the local Chatbook service is available." in text
        assert "Why: no local Chatbook artifact exists." not in text

        await pilot.click("#artifacts-use-in-console")

    app.open_console_for_live_work.assert_not_called()


@pytest.mark.asyncio
async def test_console_renders_pending_launch_context():
    ConsoleLiveWorkLaunch = _load_console_live_work_contract()
    app = _build_test_app()
    app.pending_console_launch = {
        "source": "workflows",
        "title": "Daily digest",
        "payload": {"attempt": 2, "run_id": "run-1"},
        "status": "running",
        "recovery": "Workflow is starting.",
        "action_label": "Open workflow run",
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_console_screen(host)

        assert screen.query_one("#console-pending-launch-card")
        assert len(screen.query("#console-live-work-source-readiness")) == 0
        assert screen.query_one("#console-live-work-source").renderable == "Source: workflows"
        assert screen.query_one("#console-live-work-title").renderable == "Title: Daily digest"
        assert screen.query_one("#console-live-work-status").renderable == "Status: running"
        assert screen.query_one("#console-live-work-recovery").renderable == "Recovery: Workflow is starting."
        assert screen.query_one("#console-live-work-action").renderable == "Action: Open workflow run"
        assert screen.query_one("#console-live-work-payload-attempt").renderable == "attempt: 2"
        assert screen.query_one("#console-live-work-payload-run-id").renderable == "run_id: run-1"
        text = _screen_static_text(screen)
        assert "Source: workflows" in text
        assert "Title: Daily digest" in text
        assert "Status: running" in text
        assert "Recovery: Workflow is starting." in text
        assert "Action: Open workflow run" in text
        assert "attempt: 2" in text
        assert "run_id: run-1" in text
        assert isinstance(screen._pending_console_launch_context, ConsoleLiveWorkLaunch)
        assert app.pending_console_launch is None


@pytest.mark.asyncio
async def test_console_renders_source_readiness_summary_without_pending_launch():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_console_screen(host)

        assert len(screen.query("#console-pending-launch-card")) == 0
        assert screen.query_one("#console-live-work-source-readiness")
        assert screen.query_one("#console-live-work-source-readiness-title").renderable == "Live work sources"
        assert screen.query_one("#console-live-work-source-wc").renderable == (
            "Watchlists: Connected - Home run details."
        )
        assert "Workflows: Connected" in str(screen.query_one("#console-live-work-source-workflows").renderable)
        assert "Schedules: Connected" in str(screen.query_one("#console-live-work-source-schedules").renderable)
        assert screen.query_one("#console-live-work-source-acp").renderable == (
            "ACP: Blocked - Configure ACP runtime."
        )
        assert "MCP: Not wired" in str(screen.query_one("#console-live-work-source-mcp").renderable)
        assert "RAG: Connected" in str(screen.query_one("#console-live-work-source-rag").renderable)
        assert "Artifacts: Connected" in str(screen.query_one("#console-live-work-source-artifacts").renderable)


@pytest.mark.asyncio
async def test_console_wc_live_work_action_button_routes_run_details():
    app = _build_test_app()
    app.pending_console_launch = {
        "source": "Watchlists",
        "title": "Daily security feed",
        "payload": {"target_id": "local:watchlist_run:91", "run_id": 91},
        "status": "failed",
        "recovery": "Review the Watchlists run details or retry from Watchlists.",
        "action_label": "Open Watchlists run",
    }
    app.post_message = Mock()
    app.notify = Mock()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_console_screen(host)
        button = screen.query_one("#console-live-work-primary-action")
        assert str(button.label) == "Open Watchlists run"

        await pilot.click("#console-live-work-primary-action")
        await pilot.pause(0.1)

    assert app.pending_subscription_initial_tab == "watchlist-runs"
    assert app.pending_subscription_watchlist_run_id == "local:watchlist_run:91"
    app.post_message.assert_called_once()
    assert app.post_message.call_args.args[0].screen_name == "subscriptions"
    app.notify.assert_not_called()
