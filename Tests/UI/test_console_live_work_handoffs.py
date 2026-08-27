"""Console live-work launch and staged-context handoff boundary tests."""

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.css.query import NoMatches

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Static

from Tests.UI.test_destination_shells import DestinationHarness, _wait_for_selector
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Chat.citation_evidence_models import (
    EvidenceBundle,
    EvidenceReference,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
    LocalRagContextResult,
    capture_console_staged_evidence_for_chat,
)
from tldw_chatbook.Home.dashboard_state import HomeActiveWorkItem, HomeDashboardInput
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module
from tldw_chatbook.UI.Screens.artifacts_screen import ArtifactsScreen
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import (
    SchedulesWorkbench,
)
from tldw_chatbook.UI.Screens.workflows_screen import WorkflowsScreen

#: One poll of a UI condition. Small enough that a satisfied condition exits
#: promptly, large enough not to spin the event loop.
_POLL_INTERVAL_SECONDS = 0.02

#: Polls to wait for the Console-follow RECOVERY (~12s). Deliberately generous:
#: the loop exits the instant the condition holds, so the budget only bounds
#: the failure case -- and this recovery is the load-sensitive one documented
#: on `test_watchlists_destination_retries_console_follow_after_initial_adapter_failure`.
_RECOVERY_POLL_LIMIT = 600

#: Polls to wait for an already-triggered action's observable EFFECT (~2s).
#: Shorter than the recovery budget on purpose: by this point the press has
#: landed, so a long wait here would only slow a genuine failure down.
_EFFECT_POLL_LIMIT = 100


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE3_STATUS_CARD_EVIDENCE = (
    REPO_ROOT
    / "Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-console-live-work-status-card-seam.md"
)
PHASE3_HOME_WC_CONSOLE_EVIDENCE = (
    REPO_ROOT
    / "Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-home-wc-active-work-console-launch.md"
)
PHASE3_CONSOLE_WC_ACTION_EVIDENCE = (
    REPO_ROOT
    / "Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-console-wc-action-routing.md"
)
PHASE3_CONSOLE_SOURCE_READINESS_EVIDENCE = (
    REPO_ROOT
    / "Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-console-live-work-source-readiness.md"
)
PHASE3_WC_DESTINATION_CONSOLE_EVIDENCE = (
    REPO_ROOT
    / "Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-wc-destination-console-launch.md"
)
PHASE3_SCHEDULES_CONSOLE_EVIDENCE = (
    REPO_ROOT
    / "Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-schedules-console-launch.md"
)
PHASE3_SCHEDULES_DIGEST_CONSOLE_EVIDENCE = (
    REPO_ROOT
    / "Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-schedules-digest-console-launch.md"
)
PHASE3_RAG_CONSOLE_EVIDENCE = (
    REPO_ROOT
    / "Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-rag-search-console-launch.md"
)
PHASE3_ARTIFACTS_CHATBOOK_CONSOLE_EVIDENCE = (
    REPO_ROOT
    / "Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-artifacts-chatbook-console-launch.md"
)
PHASE3_WORKFLOWS_CONSOLE_EVIDENCE = (
    REPO_ROOT
    / "Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-workflows-console-launch.md"
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
        from tldw_chatbook.Chat.console_live_work import (
            ConsoleLiveWorkSourceReadinessState,
        )
    except ImportError:
        pytest.fail("Console live-work source readiness state is missing")
    return ConsoleLiveWorkSourceReadinessState


class ConsoleHarness(ConsolidatedCSSApp):
    def __init__(self, app_instance):
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(ChatScreen(self.app_instance))


def _active_console_screen(host: ConsoleHarness):
    return host.screen_stack[-1]


async def _wait_for_production_chat_screen(
    app, pilot, *, timeout: float = 6.0
) -> ChatScreen:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        screen = app.screen
        if isinstance(screen, ChatScreen) and screen.region.width > 0:
            await pilot.pause()
            return screen
        await pilot.pause(0.01)
    raise AssertionError(
        f"Timed out waiting for production ChatScreen; active={type(app.screen).__name__}"
    )


async def _wait_for_production_artifacts_screen(
    app, pilot, *, timeout: float = 6.0
) -> ArtifactsScreen:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        screen = app.screen
        if isinstance(screen, ArtifactsScreen) and screen.region.width > 0:
            await pilot.pause()
            return screen
        await pilot.pause(0.01)
    raise AssertionError(
        f"Timed out waiting for production ArtifactsScreen; "
        f"active={type(app.screen).__name__}"
    )


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
        raise AssertionError(
            f"Unexpected direct control call: {action} {target_id} {target_route}"
        )


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


class StaticLocalChatbookService:
    def __init__(self, chatbooks):
        self.chatbooks = tuple(chatbooks)
        self.calls = []
        self.get_calls = []

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

    async def get_chatbook(self, chatbook_id):
        self.get_calls.append(chatbook_id)
        for chatbook in self.chatbooks:
            record_id = chatbook.get("chatbook_id") or chatbook.get("id")
            if str(record_id) == str(chatbook_id):
                return dict(chatbook)
        raise KeyError(chatbook_id)


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
                and all(
                    fragment in last_recovery_text for fragment in expected_fragments
                )
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

    def create_form_source_types(self, *, runtime_backend=None):
        return ("rss", "atom", "url")


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


def test_console_setup_staged_receipt_is_empty_with_no_pending_launch():
    """Task-2852: nothing staged means nothing to receipt."""
    from tldw_chatbook.Chat.console_live_work import console_setup_staged_receipt

    assert console_setup_staged_receipt(None) == ""


def test_console_setup_staged_receipt_names_the_launch_source():
    """Task-2852: the locked-Console receipt names WHAT is staged (the
    launch's own source), so a Library handoff and a Watchlists handoff read
    differently -- not a single generic "something is staged" line."""
    ConsoleLiveWorkLaunch = _load_console_live_work_contract()
    from tldw_chatbook.Chat.console_live_work import console_setup_staged_receipt

    launch = ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="Incident Review",
        payload={"result_id": "note-42:chunk-7"},
        status="staged",
    )

    receipt = console_setup_staged_receipt(launch)

    assert "Library Search/RAG" in receipt
    assert "staged" in receipt.lower()
    assert "finish provider setup" in receipt.lower()


def test_open_console_for_live_work_routes_to_chat_route():
    ConsoleLiveWorkLaunch = _load_console_live_work_contract()
    app = _build_test_app()
    seen = []
    app.post_message = lambda message: seen.append(
        getattr(message, "screen_name", None)
    )

    app.open_console_for_live_work(
        source="workflows",
        title="Daily digest",
        payload={"run_id": "run-1"},
        status="running",
        recovery="Workflow is starting.",
        action_label="Open workflow run",
    )

    assert seen == ["chat"]
    claim = app.pending_handoffs.claim(HandoffChannel.CONSOLE_LIVE_WORK)
    assert claim is not None
    assert isinstance(claim.value, ConsoleLiveWorkLaunch)
    assert claim.value.to_pending_payload() == {
        "source": "workflows",
        "title": "Daily digest",
        "payload": {"run_id": "run-1"},
        "status": "running",
        "recovery": "Workflow is starting.",
        "action_label": "Open workflow run",
    }
    assert app.pending_handoffs.acknowledge(claim) is True
    assert not app.pending_handoffs.has_pending(HandoffChannel.CONSOLE_LIVE_WORK)


def test_open_console_for_live_work_preserves_minimal_call_defaults():
    ConsoleLiveWorkLaunch = _load_console_live_work_contract()
    app = _build_test_app()
    app.post_message = lambda message: None

    app.open_console_for_live_work(source="workflows", title="Daily digest")

    claim = app.pending_handoffs.claim(HandoffChannel.CONSOLE_LIVE_WORK)
    assert claim is not None
    assert isinstance(claim.value, ConsoleLiveWorkLaunch)
    assert claim.value.to_pending_payload() == {
        "source": "workflows",
        "title": "Daily digest",
        "payload": {},
        "status": "pending",
        "recovery": "Console has staged this live-work request.",
        "action_label": "Open in Console",
    }
    assert app.pending_handoffs.acknowledge(claim) is True
    assert not app.pending_handoffs.has_pending(HandoffChannel.CONSOLE_LIVE_WORK)


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
    payload_row = next(
        row
        for row in card_state.rows
        if row.widget_id == "console-live-work-payload-run-id"
    )
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
    assert card_state.primary_action.target_route == "watchlists_collections"
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
    ConsoleLiveWorkSourceReadinessState = (
        _load_console_live_work_source_readiness_state()
    )

    state = ConsoleLiveWorkSourceReadinessState.default()

    assert state.container_id == "console-live-work-source-readiness"
    assert "console-live-work-source-readiness" in state.container_classes
    rows_by_id = {row.widget_id: row for row in state.rows}
    assert rows_by_id["console-live-work-source-wc"].text == (
        "Watchlists: Connected - Home run details."
    )
    assert (
        "console-live-work-source-connected"
        in rows_by_id["console-live-work-source-wc"].classes
    )
    assert rows_by_id["console-live-work-source-schedules"].text == (
        "Schedules: Connected - Open job context."
    )
    assert (
        "console-live-work-source-connected"
        in rows_by_id["console-live-work-source-schedules"].classes
    )
    assert rows_by_id["console-live-work-source-rag"].text == (
        "RAG: Connected - Stage search evidence."
    )
    assert (
        "console-live-work-source-connected"
        in rows_by_id["console-live-work-source-rag"].classes
    )
    assert rows_by_id["console-live-work-source-workflows"].text == (
        "Workflows: Connected - Stage run context."
    )
    assert (
        "console-live-work-source-connected"
        in rows_by_id["console-live-work-source-workflows"].classes
    )
    assert rows_by_id["console-live-work-source-artifacts"].text == (
        "Artifacts: Connected - Launch Chatbooks."
    )
    assert (
        "console-live-work-source-connected"
        in rows_by_id["console-live-work-source-artifacts"].classes
    )
    assert rows_by_id["console-live-work-source-acp"].text == (
        "ACP: Blocked - Configure ACP runtime."
    )
    assert (
        "console-live-work-source-unavailable"
        in rows_by_id["console-live-work-source-acp"].classes
    )
    for source_id in ("console-live-work-source-mcp",):
        assert "Not wired" in rows_by_id[source_id].text
        assert "console-live-work-source-unavailable" in rows_by_id[source_id].classes


def test_console_live_work_source_readiness_reflects_acp_runtime_state():
    ConsoleLiveWorkSourceReadinessState = (
        _load_console_live_work_source_readiness_state()
    )

    blocked = ConsoleLiveWorkSourceReadinessState.from_acp_runtime_status(
        "not_configured"
    )
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
    assert (
        "console-live-work-source-unavailable"
        in blocked_rows["console-live-work-source-acp"].classes
    )
    assert starting_rows["console-live-work-source-acp"].text == (
        "ACP: Starting - Waiting for runtime."
    )
    assert running_rows["console-live-work-source-acp"].text == (
        "ACP: Connected - Follow ACP session."
    )
    assert (
        "console-live-work-source-connected"
        in running_rows["console-live-work-source-acp"].classes
    )
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
    app.post_message.assert_called_once()
    navigation = app.post_message.call_args.args[0]
    assert navigation.screen_name == "watchlists_collections"
    assert navigation.screen_context == {
        "section": "runs",
        "backend": "local",
        "run_id": "local:watchlist_run:91",
    }
    assert not hasattr(app, "pending_watchlists_section")
    assert not hasattr(app, "pending_watchlists_run_id")
    app.notify.assert_not_called()


@pytest.mark.asyncio
async def test_schedules_console_follow_uses_home_dashboard_app_inputs():
    app = _build_test_app()
    app.providers_models = {"OpenAI": ["gpt-4.1"]}
    app.screen_state_store.save(
        "chat",
        {"conversation_id": "c1"},
        app._current_runtime_identity(),
    )
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
    screen = SchedulesWorkbench(app)

    item = await screen._latest_console_follow_item_from_adapter()

    assert getattr(item, "item_id", None) == "schedule:run:11"
    assert app.home_active_work_adapter.build_calls == [
        {
            "providers_models": {"OpenAI": ["gpt-4.1"]},
            "has_recent_work": True,
        }
    ]


@pytest.mark.parametrize(
    ("route", "button_id", "expected_copy"),
    [
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
        deadline = time.monotonic() + 2.0
        while expected_copy not in _screen_static_text(host.screen):
            if time.monotonic() >= deadline:
                raise AssertionError(
                    f"Timed out waiting for recovery copy: {expected_copy}"
                )
            await pilot.pause(0.01)
        button = host.screen.query_one(f"#{button_id}")
        assert button.disabled is True
        assert "unavailable" in str(button.label).lower()
        assert expected_copy in _screen_static_text(host.screen)
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
        assert str(button.label) == "Follow in Console"
        assert str(button.tooltip) == (
            "Start or select a schedule run to enable Console follow."
        )

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
        assert str(button.label) == "Follow in Console"

        button.press()
        await pilot.pause(0.1)

    app.open_active_home_item_in_console.assert_called_once_with(
        target_id="schedule:run:7",
        target_route="chat",
    )


def test_workflows_console_launch_uses_home_dashboard_app_inputs():
    app = _build_test_app()
    app.providers_models = {"OpenAI": ["gpt-4.1"]}
    app.screen_state_store.save(
        "chat",
        {"conversation_id": "c1"},
        app._current_runtime_identity(),
    )
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
        assert (
            "Next: Start or select a workflow run before opening it in Console."
            in screen_text
        )
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
        # TASK-2313, AC#6: "Console follow" appeared twice in two
        # adjacent lines; the disabled button's label now matches the
        # enabled state's own "Follow ... in Console" phrasing rather
        # than restating the noun phrase a second time. `disabled=True`
        # (asserted above) already conveys unavailability visually.
        assert str(button.label) == "Follow in Console"
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
async def test_watchlists_destination_logs_adapter_failure_and_disables_follow(
    monkeypatch,
):
    from tldw_chatbook.UI.Watchlists_Modules import watchlists_console_handoff

    app = _build_test_app()
    app.home_active_work_adapter = RaisingHomeActiveWorkAdapter()
    app.open_active_home_item_in_console = Mock()
    logger = Mock()
    monkeypatch.setattr(watchlists_console_handoff, "logger", logger, raising=False)
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_console_screen(host)
        button = screen.query_one("#watchlists-follow-in-console")

        assert button.disabled is True
        assert (
            "No active Watchlists run is available for Console follow."
            in _screen_static_text(screen)
        )

    # The call site is `logger.opt(exception=True).warning(...)` -- loguru's
    # traceback capture (the stdlib `exc_info=True` kwarg is a silent no-op
    # under loguru) -- so the warning lands on the Mock returned by `.opt()`.
    # It now lives on the extracted WatchlistsConsoleHandoff object (moved off
    # the screen), so assert on the chained warning mock rather than pinning
    # an exact `.opt()` call count.
    assert logger.opt.call_args_list
    assert all(c.kwargs.get("exception") is True for c in logger.opt.call_args_list)
    opt_warning = logger.opt.return_value.warning
    opt_warning.assert_called_once()
    assert "Watchlists Console follow" in opt_warning.call_args.args[0]
    app.open_active_home_item_in_console.assert_not_called()


@pytest.mark.asyncio
async def test_watchlists_destination_retries_console_follow_after_initial_adapter_failure():
    """Console follow recovers after the adapter's first build fails.

    **task-2769 -- this test is load-sensitive, and the residue is real.**
    Three genuine races were fixed here: `query_one` was called inside the
    retry loop and RAISED on a not-yet-mounted node (making the first
    iteration fatal rather than a retry); the gate waited on the label alone,
    so a press could land while the control was still disabled; and a fixed
    `pause(0.1)` after the press assumed the async handler had finished.

    What remains is not a test artifact. `pilot.click()` computes an offset
    and then dispatches mouse events, and the recovery re-renders this rail
    between those steps: measured on a quiet machine, `pilot.click()` passed
    6 of 10 isolated runs where `press()` passed 10 of 10, while
    `get_widget_at` resolved to this button every time. So reachability is
    asserted directly and the button is pressed, rather than the click
    standing in for both. Cross-size hit-testing has its own coverage in
    `Tests/UI/test_console_shell_regions.py`.

    Under CPU load the recovery itself can still exceed the wait budget --
    measured 7 of 12 isolated runs at load average ~8, against 10 of 10 on an
    idle machine, with a 12-second budget that raising further did not help
    (it is stuck, not slow). That points at nondeterminism in the recovery
    retry rather than at this test, and is recorded on task-2769 rather than
    hidden behind a longer sleep.
    """
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
        # task-2769: `query_one` RAISES when the node is not mounted yet, so
        # calling it inside the retry loop made the first iteration fatal
        # rather than a retry -- the loop only ever tolerated the label being
        # stale, never the button being absent. 4 of 5 isolated runs failed,
        # about half of them on `NoMatches`.
        button = None
        # 600 x 0.02s = 12s. The original budget was 2s and the recovery
        # genuinely exceeds it on a loaded machine -- raising it is not
        # papering over a race, because the loop exits the instant the
        # condition holds; the budget only bounds the failure case.
        for _ in range(_RECOVERY_POLL_LIMIT):
            try:
                button = screen.query_one("#watchlists-follow-in-console")
            except NoMatches:
                button = None
            else:
                # Gate on the button being both recovered AND enabled: the
                # label can be repainted while the control is still disabled,
                # and a press in that window is dropped.
                if "Recovered run" in str(button.label) and not button.disabled:
                    break
            await pilot.pause(_POLL_INTERVAL_SECONDS)
        else:
            raise AssertionError(
                f"Console follow did not recover. Text: {_screen_static_text(screen)}"
            )

        assert button.disabled is False
        # Reachability is asserted directly rather than implied by a
        # coordinate click: `pilot.click()` computes an offset and then
        # dispatches mouse events, and the recovery re-renders this rail
        # between those two steps, so the press can land on a stale position.
        # Measured: `pilot.click()` passed 6 of 10 isolated runs, `press()`
        # 10 of 10, while `get_widget_at` resolved to this button every time
        # (task-2769). The user-facing property -- the control is hit-testable
        # where it is drawn -- is what the click was really standing in for,
        # so assert THAT, then press. Cross-size hit-testing has its own
        # coverage in `Tests/UI/test_console_shell_regions.py`.
        # The Inspector is scrollable and this action may start below the
        # viewport at shorter terminal heights. Bring the live control into
        # view before asserting hit-testability; testing its pre-scroll
        # document coordinate against screen coordinates is invalid.
        button.scroll_visible()
        await pilot.pause()
        assert button.region.area, "follow button has no drawn region"
        button.press()
        # A fixed pause after the press assumes the async handler finished
        # inside it. Wait for the observable effect instead.
        for _ in range(_EFFECT_POLL_LIMIT):
            if app.open_active_home_item_in_console.call_count:
                break
            await pilot.pause(_POLL_INTERVAL_SECONDS)
        else:
            raise AssertionError(
                "Console follow click produced no open_active_home_item_in_console "
                f"call. Text: {_screen_static_text(screen)}"
            )

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

        button.scroll_visible()
        await pilot.pause()
        button.press()
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
        assert str(button.label) == "Follow in Console"
        assert str(button.tooltip) == (
            "Start or select a schedule run to enable Console follow."
        )

    assert app.local_media_reading_service.calls == [
        {
            "schedule_id": None,
            "limit": 1,
            "offset": 0,
        }
    ]
    app.open_console_for_live_work.assert_not_called()


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
        assert str(button.label) == "Follow in Console"

        button.press()
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
        assert (
            "Unavailable: Console launch for Chatbook artifacts."
            in _screen_static_text(screen)
        )
        assert (
            "Next: Create or import a Chatbook artifact before opening it in Console."
            in _screen_static_text(screen)
        )

        await pilot.click("#artifacts-use-in-console")

    app.open_console_for_live_work.assert_not_called()
    app.open_chat_with_handoff.assert_not_called()
    assert app.local_chatbook_service.calls == [
        {"q": None, "limit": 25, "offset": 0, "kwargs": {}}
    ]


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
        str(value) for value in launch_kwargs["payload"].values() if value is not None
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
    assert (
        app.open_console_for_live_work.call_args.kwargs["payload"]["chatbook_id"] == 10
    )


@pytest.mark.asyncio
async def test_artifacts_destination_consumes_pending_chatbook_target_before_latest_fallback():
    app = _build_test_app()
    chatbook_service = StaticLocalChatbookService(
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

    async with app.run_test(size=(180, 40)) as pilot:
        await _wait_for_production_chat_screen(app, pilot)
        app.local_chatbook_service = chatbook_service
        app.pending_handoffs.stage(
            HandoffChannel.ARTIFACT_CHATBOOK_TARGET,
            "local:chatbook:77",
        )
        app.post_message(NavigateToScreen("artifacts"))
        screen = await _wait_for_production_artifacts_screen(app, pilot)
        await _wait_for_selector(screen, pilot, "#artifacts-console-available")

        text = _screen_static_text(screen)
        assert "Open Console for requested Chatbook artifact: Requested Pack." in text
        assert "Latest Pack" not in text
        assert not app.pending_handoffs.has_pending(
            HandoffChannel.ARTIFACT_CHATBOOK_TARGET
        )
        assert (
            app.pending_handoffs.claim(HandoffChannel.ARTIFACT_CHATBOOK_TARGET) is None
        )
        assert chatbook_service.get_calls == ["77"]

        await pilot.click("#artifacts-use-in-console")

    app.open_console_for_live_work.assert_called_once()
    assert (
        app.open_console_for_live_work.call_args.kwargs["payload"]["target_id"]
        == "local:chatbook:77"
    )


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
        assert (
            "Next: Retry Artifacts after the local Chatbook service is available."
            in text
        )
        assert "Why: no local Chatbook artifact exists." not in text

        await pilot.click("#artifacts-use-in-console")

    app.open_console_for_live_work.assert_not_called()


@pytest.mark.asyncio
async def test_console_renders_pending_launch_context():
    ConsoleLiveWorkLaunch = _load_console_live_work_contract()
    app = _build_test_app()
    app.pending_handoffs.stage(
        HandoffChannel.CONSOLE_LIVE_WORK,
        {
            "source": "workflows",
            "title": "Daily digest",
            "payload": {"attempt": 2, "run_id": "run-1"},
            "status": "running",
            "recovery": "Workflow is starting.",
            "action_label": "Open workflow run",
        },
    )

    async with app.run_test(size=(180, 40)) as pilot:
        screen = await _wait_for_production_chat_screen(app, pilot)
        await _wait_for_selector(screen, pilot, "#console-pending-launch-card")

        assert screen.query_one("#console-pending-launch-card")
        assert len(screen.query("#console-live-work-source-readiness")) == 0
        assert (
            screen.query_one("#console-live-work-source").renderable
            == "Source: workflows"
        )
        assert (
            screen.query_one("#console-live-work-title").renderable
            == "Title: Daily digest"
        )
        assert (
            screen.query_one("#console-live-work-status").renderable
            == "Status: running"
        )
        assert (
            screen.query_one("#console-live-work-recovery").renderable
            == "Recovery: Workflow is starting."
        )
        assert (
            screen.query_one("#console-live-work-action").renderable
            == "Action: Open workflow run"
        )
        assert (
            screen.query_one("#console-live-work-payload-attempt").renderable
            == "attempt: 2"
        )
        assert (
            screen.query_one("#console-live-work-payload-run-id").renderable
            == "run_id: run-1"
        )
        text = _screen_static_text(screen)
        assert "Source: workflows" in text
        assert "Title: Daily digest" in text
        assert "Status: running" in text
        assert "Recovery: Workflow is starting." in text
        assert "Action: Open workflow run" in text
        assert "attempt: 2" in text
        assert "run_id: run-1" in text
        assert isinstance(screen._pending_console_launch_context, ConsoleLiveWorkLaunch)
        assert not app.pending_handoffs.has_pending(HandoffChannel.CONSOLE_LIVE_WORK)
        assert app.pending_handoffs.claim(HandoffChannel.CONSOLE_LIVE_WORK) is None


def _bare_console_screen_for_restore(app_instance=None) -> ChatScreen:
    """Build a native-console screen shell for direct restore-path calls.

    Mirrors ``test_console_native_chat_flow.py``'s own ``_bare_console_
    screen`` helper: bypasses ``ChatScreen.__init__`` (heavy, requires a
    mounted Textual app) while resolving the class's inherited restore/
    consume helpers normally, so the D3 (staged-evidence-survives-
    navigation) tests below can drive
    ``_restore_native_console_state``/``_consume_pending_console_launch``
    as plain, fast calls instead of a full pilot-driven screen.

    Unlike that helper, this one also accepts ``app_instance`` -- the whole
    point of the regression tests below is proving a *restored* launch does
    NOT reach back into the real ``PendingHandoffStore`` on ``app_instance``,
    which ``_consume_pending_console_launch`` only touches when
    ``_pending_console_launch_context`` is still ``None``.

    Args:
        app_instance: The (real or fake) app instance to attach, or ``None``
            for callers that never exercise the handoff-store seam.

    Returns:
        ChatScreen: A bare ChatScreen instance suitable for unit-level
            restore-path testing.
    """
    from Tests.UI.console_controller_stubs import (
        NO_APP,
        stub_fleet_controller,
        stub_image_controller,
        stub_message_controller,
    )

    screen = ChatScreen.__new__(ChatScreen)
    screen.app_instance = app_instance
    screen._retrieval = SimpleNamespace(_capture_console_staged_rag=Mock())
    # Precedes the `_console_chat_store` assignment: that setter reaches
    # `_console_runtime().set_chat_store`, which reads
    # `self._fleet._console_wake_user_priority` (TASK-21381).
    stub_fleet_controller(screen, context="live work handoffs screen")
    screen._console_chat_store = ConsoleChatStore()
    screen._session = ConsoleSessionController.__new__(ConsoleSessionController)
    screen._console_visible_draft_session_id = None
    screen._console_composer_or_none = lambda: None
    screen._task_resume_state = TaskResumeState()
    # `_restore_native_console_state` calls the three
    # `_rehydrate_console_message_*` helpers, which moved to
    # `ConsoleMessageController` (wave-3 console decomposition, task 1) and
    # are reached through `ChatScreen`'s delegations. `ChatScreen.__new__`
    # skips the construction `__init__` would do. Those three read only
    # `app_instance`, so nothing else is wired.
    resolved_app = app_instance if app_instance is not None else NO_APP
    stub_message_controller(
        screen,
        context="test_console_live_work_handoffs._bare_console_screen",
        # Three of the six call sites pass no app at all -- they exercise
        # restore paths that read `app_instance` only through
        # `getattr(..., None)`. `stub_message_controller` refuses to INFER a
        # missing app (an inferred `None` snapshot is a silent-default hole),
        # so the absence is declared here instead. task-3024/2769.
        app_instance=resolved_app,
    )
    stub_image_controller(
        screen,
        context="test_console_live_work_handoffs._bare_console_screen",
        app_instance=resolved_app,
    )
    return screen


@pytest.mark.asyncio
async def test_console_staged_launch_with_evidence_bundle_survives_screen_recreation_and_a_fresh_handoff_supersedes_it():
    """D3: a staged live-work launch (with its real evidence bundle) must
    survive screen re-creation; PR-T1 C1: a launch staged AFTER it must
    supersede it on the next consume.

    ``ChatScreen`` instances are never reused across navigation --
    ``TldwCli._create_navigation_screen`` builds a fresh one on every
    ``NavigateToScreen`` -- so continuity depends entirely on
    ``save_state``/``restore_state`` carrying
    ``_pending_console_launch_context`` (and its sibling
    ``_console_evidence_sent_notice``) across. Before D3, neither
    ``_serialize_native_console_state`` nor ``_restore_native_console_state``
    touched either field at all: ANY navigation away from Console silently
    dropped staged evidence with no error and no user-visible warning.

    UPDATED BY PR-T1's final review (C1). This test previously asserted the
    opposite of the second half: that a decoy staged into the store while a
    restored launch was resident stayed there UNTOUCHED, as proof that a
    restored launch never re-claims the channel. That "resident always
    wins" rule was safe only while a launch could not survive navigation.
    Once D3 made it survive, the leftover store entry was not a decoy at
    all -- it was the user's newest "Use in Console" click, left invisible
    and later spent on an unrelated message (see
    ``_supersede_resident_console_launch_from_store``). The store entry is
    now claimed and staged, and the assertions below are inverted to match:
    the newer launch becomes resident and the channel is drained.

    What the original test protected is still protected, by construction:
    the restore itself is asserted in full (evidence bundle, sent notice)
    BEFORE the newer launch is staged, and the no-newer-entry case -- the
    plain tab-switch survivor, which must still not reach into the store --
    is covered by
    ``test_console_restored_launch_does_not_touch_an_empty_handoff_channel``
    below.
    """
    ConsoleLiveWorkLaunch = _load_console_live_work_contract()
    from tldw_chatbook.UI.Views.RAGSearch.search_handoff import (
        build_library_rag_console_live_work_payload,
    )

    app = _build_test_app()
    result = {
        "result_id": "note-42:chunk-7",
        "title": "Incident Review",
        "snippet": "Expired credential caused the incident.",
        "source_id": "note-42",
        "chunk_id": "chunk-7",
        "score": 0.93,
        "runtime_backend": "local-fts",
    }
    # Built the exact same way `library_screen.py::_stage_library_rag_
    # result_in_console` builds it, so `evidence_bundle` is real
    # `to_payload()`-shaped data, not a hand-rolled stand-in.
    launch_payload = build_library_rag_console_live_work_payload(
        result, query="Why did the incident happen?"
    )
    original_launch = ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title=result["title"],
        payload=launch_payload,
        status="staged",
        recovery="Review citations before sending.",
        action_label="Review evidence in Console",
    )
    app.pending_handoffs.stage(HandoffChannel.CONSOLE_LIVE_WORK, original_launch)

    async with app.run_test(size=(180, 40)) as pilot:
        screen1 = await _wait_for_production_chat_screen(app, pilot)
        await _wait_for_selector(screen1, pilot, "#console-pending-launch-card")

        launch1 = screen1._pending_console_launch_context
        assert isinstance(launch1, ConsoleLiveWorkLaunch)
        assert launch1.title == "Incident Review"
        assert launch1.payload.get("evidence_bundle", {}).get("references")
        assert not app.pending_handoffs.has_pending(HandoffChannel.CONSOLE_LIVE_WORK)

        # Simulate a leftover "evidence sent" memory from an earlier send in
        # this same screen instance (PR-4/task-1) coexisting with the launch
        # just consumed via the handoff store --
        # `_consume_pending_console_launch` never touches this field, unlike
        # `_stage_console_library_rag_launch`, which clears it on new staging.
        screen1._console_evidence_sent_notice = 2

        state = screen1.save_state()
        saved = state["native_console_state"]
        assert saved["pending_console_launch"]["title"] == "Incident Review"
        assert (
            saved["pending_console_launch"]["payload"]["evidence_bundle"]
            == launch1.payload["evidence_bundle"]
        )
        assert saved["console_evidence_sent_notice"] == 2

        # Stage a NEWER launch after the original was consumed and
        # acknowledged -- the store's slot is empty at this point, so this
        # is a legitimate fresh stage (the user's next "Use in Console"
        # click), not an overwrite of anything still owned by screen1.
        newer_launch = ConsoleLiveWorkLaunch.from_values(
            source="Library Search/RAG",
            title="Rotation Runbook",
            payload={},
            status="staged",
        )
        app.pending_handoffs.stage(HandoffChannel.CONSOLE_LIVE_WORK, newer_launch)

        screen2 = _bare_console_screen_for_restore(app)
        screen2._restore_native_console_state(saved)

        # The restore itself is complete and faithful, newer store entry or
        # not: this is the D3 half of the test, asserted before any consume.
        launch2 = screen2._pending_console_launch_context
        assert isinstance(launch2, ConsoleLiveWorkLaunch)
        assert launch2.title == "Incident Review"
        assert launch2.source == "Library Search/RAG"
        assert launch2.payload["evidence_bundle"] == launch1.payload["evidence_bundle"]
        assert screen2._console_evidence_sent_notice == 2

        strip_state = screen2._build_console_staged_evidence_strip_state(launch2)
        assert strip_state.visible is True
        assert strip_state.rows
        assert strip_state.rows[0].title == "Incident Review"

        # C1: consume supersedes -- the newer explicit user action wins over
        # the stale survivor, and the channel is DRAINED so no later send can
        # be ambushed by it.
        consumed = screen2._consume_pending_console_launch()
        assert consumed is not launch2
        assert consumed is not None
        assert consumed.title == "Rotation Runbook"
        assert screen2._pending_console_launch_context is consumed
        assert not app.pending_handoffs.has_pending(HandoffChannel.CONSOLE_LIVE_WORK)
        assert app.pending_handoffs.claim(HandoffChannel.CONSOLE_LIVE_WORK) is None
        # Superseding is a fresh staging: the previous send's "evidence
        # sent" line must not survive onto unrelated new evidence.
        assert screen2._console_evidence_sent_notice is None
        assert screen2._pending_console_launch_auto_open_inspector is True


@pytest.mark.asyncio
async def test_locked_console_shows_staged_evidence_receipt_task_2852():
    """Task-2852 AC#2, end to end: a Library "Use in Console" handoff that
    lands on a locked (setup-incomplete) Console must show a visible
    receipt naming what's staged -- the original UAT repro found zero trace
    of the selection on the blocking "Get started" overlay.

    `_build_test_app()`'s synthetic config carries no `api_settings`, so a
    fresh app here reproduces the "fresh profile with no provider" repro
    state without any extra config wiring.
    """
    from tldw_chatbook.Widgets.Console.console_setup_modal import ConsoleSetupModal

    ConsoleLiveWorkLaunch = _load_console_live_work_contract()
    app = _build_test_app()
    launch = ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="Incident Review",
        payload={"result_id": "note-42:chunk-7"},
        status="staged",
        recovery="Review citations before sending.",
        action_label="Review evidence in Console",
    )
    app.pending_handoffs.stage(HandoffChannel.CONSOLE_LIVE_WORK, launch)

    async with app.run_test(size=(180, 40)) as pilot:
        screen = await _wait_for_production_chat_screen(app, pilot)
        await _wait_for_selector(screen, pilot, "#console-pending-launch-card")

        modal = screen.query_one("#console-setup-modal", ConsoleSetupModal)
        # The repro's precondition: Console really is locked here, not
        # already past setup for some other reason (e.g. a previous test's
        # config leaking through).
        assert modal.is_blocking is True

        notice = screen.query_one("#console-setup-modal-staged-notice", Static)
        assert notice.display is True
        rendered = str(notice.renderable).strip()
        assert "Library Search/RAG" in rendered
        assert "staged" in rendered.lower()
        assert "finish provider setup" in rendered.lower()


@pytest.mark.asyncio
async def test_configured_console_staged_strip_unaffected_by_receipt_task_2852():
    """Task-2852 AC#3 regression guard: once Console is configured, PR
    #1320's staged-evidence strip must render exactly as before, and the
    new locked-Console receipt (which only exists inside the setup modal's
    blocking overlay) must stay hidden -- the fix must not touch the
    already-shipped configured path.
    """
    from tldw_chatbook.Chat.console_display_state import (
        ConsoleStagedEvidenceStripState,
    )
    from tldw_chatbook.Widgets.Console.console_setup_modal import ConsoleSetupModal

    ConsoleLiveWorkLaunch = _load_console_live_work_contract()
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1-2025-04-14",
        },
        "api_settings": {"openai": {"api_key": "configured-test-key"}},
    }
    launch = ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="Incident Review",
        payload={"result_id": "note-42:chunk-7"},
        status="staged",
        recovery="Review citations before sending.",
        action_label="Review evidence in Console",
    )
    app.pending_handoffs.stage(HandoffChannel.CONSOLE_LIVE_WORK, launch)

    async with app.run_test(size=(180, 40)) as pilot:
        screen = await _wait_for_production_chat_screen(app, pilot)
        await _wait_for_selector(screen, pilot, "#console-staged-evidence-strip")

        modal = screen.query_one("#console-setup-modal", ConsoleSetupModal)
        assert modal.is_blocking is False

        notice = screen.query_one("#console-setup-modal-staged-notice", Static)
        assert notice.display is False

        strip_state = screen._build_console_staged_evidence_strip_state(
            screen._pending_console_launch_context
        )
        assert isinstance(strip_state, ConsoleStagedEvidenceStripState)
        assert strip_state.visible is True
        assert strip_state.rows
        assert strip_state.rows[0].title == "Incident Review"


@pytest.mark.asyncio
async def test_console_restored_launch_does_not_touch_an_empty_handoff_channel():
    """PR-T1 C1: superseding is scoped to an actually-pending store entry.

    The C1 fix loosened ``_consume_pending_console_launch``'s resident-wins
    early return. This pins the other side of that loosening: with nothing
    staged in the channel, a restored (tab-switch survivor) launch is still
    returned as-is, is still not re-claimed, and the store is left exactly
    as it was -- no spurious claim left in flight to poison the next real
    handoff.
    """
    ConsoleLiveWorkLaunch = _load_console_live_work_contract()

    app = _build_test_app()
    async with app.run_test(size=(180, 40)):
        survivor = ConsoleLiveWorkLaunch.from_values(
            source="Library Search/RAG",
            title="Tab-switch survivor",
            payload={"source_id": "note-9"},
            status="staged",
        )
        screen = _bare_console_screen_for_restore(app)
        screen._pending_console_launch_context = survivor
        screen._console_evidence_sent_notice = 3

        assert not app.pending_handoffs.has_pending(HandoffChannel.CONSOLE_LIVE_WORK)

        consumed = screen._consume_pending_console_launch()

        assert consumed is survivor
        # Untouched: no clear of the sent notice, no claim left in flight.
        assert screen._console_evidence_sent_notice == 3
        assert not app.pending_handoffs.has_pending(HandoffChannel.CONSOLE_LIVE_WORK)

        # A real handoff arriving later is still claimable, which it would
        # not be if the consume above had left an unsettled in-flight claim.
        later = ConsoleLiveWorkLaunch.from_values(
            source="Library Search/RAG",
            title="Later handoff",
            payload={},
            status="staged",
        )
        app.pending_handoffs.stage(HandoffChannel.CONSOLE_LIVE_WORK, later)
        assert screen._consume_pending_console_launch().title == "Later handoff"


@pytest.mark.asyncio
async def test_console_stage_then_navigate_then_stage_again_displays_the_newest_launch():
    """PR-T1 C1 pilot regression: the real stage A -> leave -> stage B flow.

    Drives the exact user path the final review reconstructed, through the
    production navigation machinery rather than direct restore calls:

    1. A "Use in Console" handoff (A) lands and Console displays it.
    2. The user navigates away; ``save_state`` persists A (D3).
    3. A second handoff (B) is staged while Console is not mounted.
    4. The user returns to Console.

    Before the fix, ``restore_state`` (which runs BEFORE compose) made A
    resident, compose's consume returned A, and B was never displayed --
    the second click looked dead. B then sat in the store until some later
    send claimed it from inside a send gate and silently prepended it to an
    unrelated message.

    Asserted here: B is what the rebuilt Console displays AND stages, and
    the channel is empty afterwards so nothing is left to ambush a send.
    """
    ConsoleLiveWorkLaunch = _load_console_live_work_contract()

    app = _build_test_app()
    launch_a = ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="Launch A (stale survivor)",
        payload={"source_id": "note-a"},
        status="staged",
        recovery="Review citations before sending.",
        action_label="Review evidence in Console",
    )
    app.pending_handoffs.stage(HandoffChannel.CONSOLE_LIVE_WORK, launch_a)

    async with app.run_test(size=(180, 40)) as pilot:
        screen1 = await _wait_for_production_chat_screen(app, pilot)
        await _wait_for_selector(screen1, pilot, "#console-pending-launch-card")
        assert screen1._pending_console_launch_context.title == (
            "Launch A (stale survivor)"
        )

        # Leave Console. `save_state` persists A onto the app's per-screen
        # state, which the next Console instance restores before compose.
        app.post_message(NavigateToScreen("library"))
        await pilot.pause()
        await pilot.pause()

        # The user stages a SECOND result while standing in Library.
        launch_b = ConsoleLiveWorkLaunch.from_values(
            source="Library Search/RAG",
            title="Launch B (the click that must not die)",
            payload={"source_id": "note-b"},
            status="staged",
            recovery="Review citations before sending.",
            action_label="Review evidence in Console",
        )
        app.pending_handoffs.stage(HandoffChannel.CONSOLE_LIVE_WORK, launch_b)

        app.post_message(NavigateToScreen("chat"))
        await pilot.pause()
        screen2 = await _wait_for_production_chat_screen(app, pilot)
        await _wait_for_selector(screen2, pilot, "#console-pending-launch-card")

        staged = screen2._pending_console_launch_context
        assert staged is not None
        assert staged.title == "Launch B (the click that must not die)"

        # Displayed, not merely staged.
        strip_state = screen2._build_console_staged_evidence_strip_state(staged)
        assert strip_state.visible is True
        tray_state = screen2._build_console_staged_context_state(staged)
        assert "Launch B (the click that must not die)" in tray_state.summary

        # And A is not lurking in the store to be spent on a later send.
        assert not app.pending_handoffs.has_pending(HandoffChannel.CONSOLE_LIVE_WORK)


@pytest.mark.asyncio
async def test_console_armed_sent_notice_round_trips_to_a_fresh_screen():
    """D3: the one-send "Evidence sent with this message" memory
    (``_console_evidence_sent_notice``) must also survive screen
    re-creation, independent of any staged launch.

    Unlike the staged-launch case above, a sent notice has nothing to do
    with ``PendingHandoffStore`` -- it is purely local memory of what the
    LAST send consumed (PR-4/task-1) -- so this is a plain serialize/
    restore round trip, mirroring the existing unit-level round-trip tests
    in ``test_console_native_chat_flow.py``.
    """
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession

    store = ConsoleChatStore()
    session = ConsoleChatSession(id="session-a", title="Chat 1")
    store.restore_state(
        sessions=[session],
        messages_by_session={session.id: []},
        active_session_id=session.id,
    )
    screen = _bare_console_screen_for_restore()
    screen._console_chat_store = store
    screen._pending_console_launch_context = None
    screen._console_evidence_sent_notice = 5

    payload = screen._serialize_native_console_state()
    assert payload is not None
    assert payload["pending_console_launch"] is None
    assert payload["console_evidence_sent_notice"] == 5

    restored_store = ConsoleChatStore()
    restored_screen = _bare_console_screen_for_restore()
    restored_screen._console_chat_store = restored_store
    restored_screen._restore_native_console_state(payload)

    assert restored_screen._pending_console_launch_context is None
    assert restored_screen._console_evidence_sent_notice == 5

    strip_state = restored_screen._build_console_staged_evidence_strip_state(
        restored_screen._pending_console_launch_context
    )
    assert strip_state.visible is True
    assert strip_state.notice == "Evidence sent with this message · 5 sources"


def test_console_native_state_restore_tolerates_legacy_payload_without_launch_or_notice_keys():
    """D3 legacy tolerance: a payload saved before this fix (no
    ``pending_console_launch``/``console_evidence_sent_notice`` keys at all)
    must restore cleanly to "nothing staged, nothing sent" instead of
    raising.
    """
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession

    store = ConsoleChatStore()
    session = ConsoleChatSession(id="session-a", title="Chat 1")
    store.restore_state(
        sessions=[session],
        messages_by_session={session.id: []},
        active_session_id=session.id,
    )
    legacy_payload = {
        "version": "1.0",
        "active_session_id": session.id,
        "task_resume_state": {},
        "sessions": [
            {
                "id": session.id,
                "title": "Chat 1",
                "workspace_id": "default",
                "persisted_conversation_id": None,
                "draft": "",
                "settings": None,
                "updated_at": None,
                "character_id": None,
                "character_name": None,
            }
        ],
        "messages_by_session": {session.id: []},
        "image_view_modes": {},
        # Deliberately no "pending_console_launch"/"console_evidence_sent_
        # notice" keys -- this is what every payload saved before PR-T1
        # task-3 looks like.
    }

    screen = _bare_console_screen_for_restore()
    screen._console_chat_store = store

    screen._restore_native_console_state(legacy_payload)

    assert screen._pending_console_launch_context is None
    assert screen._console_evidence_sent_notice is None


@pytest.mark.asyncio
async def test_console_renders_source_readiness_summary_without_pending_launch():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_console_screen(host)

        assert len(screen.query("#console-pending-launch-card")) == 0
        assert screen.query_one("#console-live-work-source-readiness")
        assert (
            screen.query_one("#console-live-work-source-readiness-title").renderable
            == "Live work sources"
        )
        assert screen.query_one("#console-live-work-source-wc").renderable == (
            "Watchlists: Connected - Home run details."
        )
        assert "Workflows: Connected" in str(
            screen.query_one("#console-live-work-source-workflows").renderable
        )
        assert "Schedules: Connected" in str(
            screen.query_one("#console-live-work-source-schedules").renderable
        )
        assert screen.query_one("#console-live-work-source-acp").renderable == (
            "ACP: Blocked - Configure ACP runtime."
        )
        assert "MCP: Not wired" in str(
            screen.query_one("#console-live-work-source-mcp").renderable
        )
        assert "RAG: Connected" in str(
            screen.query_one("#console-live-work-source-rag").renderable
        )
        assert "Artifacts: Connected" in str(
            screen.query_one("#console-live-work-source-artifacts").renderable
        )


@pytest.mark.asyncio
async def test_console_wc_live_work_action_button_routes_run_details():
    app = _build_test_app()
    app.pending_handoffs.stage(
        HandoffChannel.CONSOLE_LIVE_WORK,
        {
            "source": "Watchlists",
            "title": "Daily security feed",
            "payload": {"target_id": "local:watchlist_run:91", "run_id": 91},
            "status": "failed",
            "recovery": ("Review the Watchlists run details or retry from Watchlists."),
            "action_label": "Open Watchlists run",
        },
    )

    async with app.run_test(size=(180, 40)) as pilot:
        screen = await _wait_for_production_chat_screen(app, pilot)
        await _wait_for_selector(screen, pilot, "#console-live-work-primary-action")
        navigation_messages = []
        original_post_message = app.post_message

        def record_and_post(message):
            if getattr(message, "screen_name", None) is not None:
                navigation_messages.append(message)
            return original_post_message(message)

        app.post_message = record_and_post
        button = screen.query_one("#console-live-work-primary-action")
        assert str(button.label) == "Open Watchlists run"

        button.press()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not navigation_messages:
            await pilot.pause(0.01)

        assert len(navigation_messages) == 1
        navigation = navigation_messages[0]
        assert navigation.screen_name == "watchlists_collections"
        assert navigation.screen_context == {
            "section": "runs",
            "backend": "local",
            "run_id": "local:watchlist_run:91",
        }
        assert not hasattr(app, "pending_watchlists_section")
        assert not hasattr(app, "pending_watchlists_run_id")
        app.post_message = original_post_message


# ---------------------------------------------------------------------------
# TASK-259: staging a Library Search/RAG launch must NOT recompose the whole
# ChatScreen -- the pending-launch card and every launch-context reader are
# refreshed with targeted widget updates instead.
# ---------------------------------------------------------------------------


def _library_rag_launch(status: str = "searching"):
    ConsoleLiveWorkLaunch = _load_console_live_work_contract()
    return ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="Library Search/RAG retrieval",
        payload={
            "query": "vector fusion",
            "source_scope": "media, notes, conversations",
        },
        status=status,
        recovery="Retrieving Library Search/RAG evidence.",
        action_label="Review evidence in Console",
    )


def _spy_screen_recompose(screen):
    """Wrap ``screen.refresh`` recording any recompose=True calls."""
    recompose_calls = []
    original_refresh = screen.refresh

    def spy_refresh(*args, **kwargs):
        if kwargs.get("recompose"):
            recompose_calls.append(kwargs)
        return original_refresh(*args, **kwargs)

    screen.refresh = spy_refresh
    return recompose_calls


@pytest.mark.asyncio
async def test_stage_console_library_rag_launch_swaps_card_without_screen_recompose():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_console_screen(host)
        await _wait_for_selector(screen, pilot, "#console-live-work-source-readiness")
        transcript_before = screen.query_one("#console-native-transcript")
        composer_before = screen.query_one("#console-native-composer")
        recompose_calls = _spy_screen_recompose(screen)

        screen._retrieval._stage_console_library_rag_launch(_library_rag_launch())
        await pilot.pause()
        await _wait_for_selector(screen, pilot, "#console-pending-launch-card")

        assert recompose_calls == []
        # A screen recompose would have replaced these widget instances.
        assert screen.query_one("#console-native-transcript") is transcript_before
        assert screen.query_one("#console-native-composer") is composer_before
        assert len(screen.query("#console-live-work-source-readiness")) == 0
        assert (
            screen.query_one("#console-live-work-source").renderable
            == "Source: Library Search/RAG"
        )
        assert (
            screen.query_one("#console-live-work-status").renderable
            == "Status: searching"
        )
        # Launch-context readers refreshed without recompose:
        staged_tray = screen.query_one("#console-staged-context-tray")
        assert not staged_tray.state.is_empty
        inspector = screen.query_one("#console-run-inspector-state")
        assert any(row.label == "Live work" for row in inspector.state.rows)


@pytest.mark.asyncio
async def test_stage_console_library_rag_launch_restage_replaces_single_card():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_console_screen(host)
        await _wait_for_selector(screen, pilot, "#console-live-work-source-readiness")
        recompose_calls = _spy_screen_recompose(screen)

        screen._retrieval._stage_console_library_rag_launch(
            _library_rag_launch("searching")
        )
        await pilot.pause()
        await _wait_for_selector(screen, pilot, "#console-pending-launch-card")

        screen._retrieval._stage_console_library_rag_launch(
            _library_rag_launch("staged")
        )
        await pilot.pause()
        await pilot.pause()

        assert recompose_calls == []
        assert len(screen.query("#console-pending-launch-card")) == 1
        assert (
            screen.query_one("#console-live-work-status").renderable == "Status: staged"
        )


@pytest.mark.asyncio
async def test_console_live_work_card_swap_keeps_tray_on_top_and_cards_at_bottom():
    """Task-400: swaps keep the pre-move card slot; the tray stays on top.

    ``_frame_console_region`` styles the tray IN PLACE (adds a class and an
    inline border, returns the same widget -- no wrapper container), so the
    tray is a direct child of the inspector rail body, pinned as its FIRST
    child. Live-work cards keep anchoring after the run-inspector block at
    the bottom. This drives the real swap seam both directions.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_console_screen(host)
        await _wait_for_selector(screen, pilot, "#console-live-work-source-readiness")

        rail_body = screen.query_one("#console-inspector-rail-body")
        tray = screen.query_one("#console-staged-context-tray")
        run_inspector = screen.query_one("#console-run-inspector")
        # Ancestry evidence: the framed tray is a DIRECT child of the rail
        # body (no frame wrapper), composed at the very top.
        assert tray.parent is rail_body
        assert tray.has_class("console-frame-quiet")
        assert list(rail_body.children).index(tray) == 0

        # Readiness -> pending-launch swap mounts after the run inspector.
        screen._retrieval._stage_console_library_rag_launch(
            _library_rag_launch("searching")
        )
        await pilot.pause()
        await _wait_for_selector(screen, pilot, "#console-pending-launch-card")
        card = screen.query_one("#console-pending-launch-card")
        children = list(rail_body.children)
        assert card.parent is rail_body
        assert (
            children.index(tray) < children.index(run_inspector) < children.index(card)
        )

        # Pending-launch -> readiness swap (launch resolved) re-anchors too.
        screen._pending_console_launch_context = None
        assert screen._sync_console_pending_launch_surfaces() is True
        await pilot.pause()
        await _wait_for_selector(screen, pilot, "#console-live-work-source-readiness")
        assert len(screen.query("#console-pending-launch-card")) == 0
        readiness = screen.query_one("#console-live-work-source-readiness")
        children = list(rail_body.children)
        assert readiness.parent is rail_body
        assert (
            children.index(tray)
            < children.index(run_inspector)
            < children.index(readiness)
        )


@pytest.mark.asyncio
async def test_stage_console_library_rag_launch_still_auto_opens_inspector():
    """The blocked-outcome auto-open must survive the recompose removal."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_console_screen(host)
        await _wait_for_selector(screen, pilot, "#console-right-rail")

        screen._set_console_rail_preference(right_open=False)
        await pilot.pause()
        assert screen.query_one("#console-right-rail").styles.display == "none"

        # Mirrors `_apply_console_library_rag_search_outcome`'s blocked
        # branch: flag set BEFORE staging (TASK-259 ordering).
        screen._pending_console_launch_auto_open_inspector = True
        screen._retrieval._stage_console_library_rag_launch(
            _library_rag_launch("blocked")
        )
        await pilot.pause()

        assert screen.query_one("#console-right-rail").styles.display != "none"


# --------------------------------------------------------------------------
# Task 9 (Task-2 review bonus find): non-RAG "Use in Console" handoffs
# silently sent zero staged content to the model. Only a handoff whose
# `payload.source` contained the substring "rag" reached the
# `evidence_bundle`-building branch of `_stage_handoff_as_console_live_work`
# -- so Library media/conversation handoffs (`source="library"`) and Library
# note handoffs (`source="notes"`) staged visibly in the strip/tray but
# `capture_console_staged_evidence_for_chat` short-circuited to
# `LocalRagContextResult(None, None)` on send, because
# `payload.get("evidence_bundle")` was never a mapping for them.
# --------------------------------------------------------------------------


def _media_handoff_payload() -> ChatHandoffPayload:
    return ChatHandoffPayload(
        source="library",
        item_type="media",
        title="Transformer notes",
        body="Attention is all you need. " * 5,
        source_id="media-77",
        content_ref="media-77#c1",
        display_summary="Media staged: Transformer notes",
        suggested_prompt="Use this media as source context for my next question.",
        runtime_backend="local",
        source_owner="local",
        source_selector_state="local",
    )


def _notes_handoff_payload() -> ChatHandoffPayload:
    return ChatHandoffPayload(
        source="notes",
        item_type="note",
        title="Meeting notes",
        body="Discussed the Q3 roadmap and follow-up owners.",
        source_id="42",
        suggested_prompt="Use this note as context and help me work with it.",
        runtime_backend="local",
        source_owner="local",
        source_selector_state="local",
    )


def _conversation_handoff_payload() -> ChatHandoffPayload:
    return ChatHandoffPayload(
        source="library",
        item_type="conversation",
        title="Prior planning chat",
        body=(
            "Conversation: Prior planning chat\n"
            "Conversation ID: conv-9\n"
            "Messages: 12\n"
            "Workspace: unassigned\n"
            "Updated: unknown\n"
            "Source authority: local"
        ),
        source_id="conv-9",
        display_summary="Conversation staged: Prior planning chat",
        suggested_prompt="Use this conversation as source context for my next question.",
        runtime_backend="local",
        source_owner="local",
        source_selector_state="local",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "build_payload",
    (_media_handoff_payload, _notes_handoff_payload, _conversation_handoff_payload),
    ids=("media", "notes", "conversation"),
)
async def test_non_rag_handoff_stages_a_non_empty_evidence_bundle(build_payload):
    """A Library/Notes handoff (no "rag" token in its source) must still
    carry a real, non-empty `evidence_bundle` after staging -- the strip and
    the model must agree on what content is staged."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        screen = _active_console_screen(host)
        await _wait_for_selector(screen, pilot, "#console-native-composer")

        payload = build_payload()
        screen._stage_handoff_as_console_live_work(payload)
        await pilot.pause()

        launch = screen._pending_console_launch_context
        assert launch is not None
        bundle_payload = launch.payload.get("evidence_bundle")
        assert isinstance(bundle_payload, dict), (
            "evidence_bundle must be a real mapping, not absent -- "
            "capture_console_staged_evidence_for_chat treats anything else "
            "as if nothing were staged at all"
        )
        references = bundle_payload.get("references")
        assert references, "bundle must carry at least one reference"
        reference = references[0]
        assert reference["source_id"] == payload.source_id
        assert reference["snippet"].strip()
        assert reference["title"].strip()


@pytest.mark.asyncio
async def test_rag_labeled_handoff_evidence_bundle_shape_is_byte_unchanged():
    """Pin: dropping the `"rag" in source` gate so every handoff builds a
    bundle must not change the bundle a RAG-labeled handoff already built.
    The expected shape below is hand-derived from the branch's own formula
    (see `_stage_handoff_as_console_live_work`), independent of the
    production code path, so a regression in field derivation trips this."""
    payload = ChatHandoffPayload(
        source="Library Search/RAG",
        item_type="rag-result",
        title="Transformer notes",
        body="Attention is all you need.",
        source_id="media-77",
        content_ref="media-77#c1",
        suggested_prompt="Use this retrieved result as context.",
        runtime_backend="local",
    )
    expected_bundle = EvidenceBundle(
        bundle_id="media-77#c1",
        query="Use this retrieved result as context.",
        source="Library Search/RAG",
        references=(
            EvidenceReference(
                evidence_id="S1",
                source_id="media-77",
                source_type="rag-result",
                title="Transformer notes",
                snippet="Attention is all you need.",
                authority_label="local",
                content_ref="media-77#c1",
            ),
        ),
    ).to_payload()

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        screen = _active_console_screen(host)
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        screen._stage_handoff_as_console_live_work(payload)
        await pilot.pause()
        launch = screen._pending_console_launch_context
        assert launch is not None
        actual_bundle = launch.payload["evidence_bundle"]

    assert actual_bundle == expected_bundle


class _RowsDouble:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _ExistingMediaDBDouble:
    """Minimal double for `app.media_db`'s existence-check seam.

    Module name starts with "Tests." so `_sensitive_fetchall`'s test-double
    allowance (`chat_rag_events._sensitive_fetchall`) accepts `execute_query`
    in place of the production `get_connection` path.
    """

    is_memory_db = True

    def __init__(self, *existing_ids: str):
        self.existing_ids = set(existing_ids)

    def execute_query(self, _query, params):
        requested = set(json.loads(params[0]))
        return _RowsDouble(
            [(source_id,) for source_id in sorted(requested & self.existing_ids)]
        )


class _ExistingChaChaDBDouble:
    """Minimal double for `app.chachanotes_db`'s existence-check seam."""

    is_memory_db = True

    def __init__(self, *, note_ids=(), conversation_ids=()):
        self.note_ids = set(note_ids)
        self.conversation_ids = set(conversation_ids)

    def execute_query(self, _query, params):
        requested_notes = set(json.loads(params[0]))
        requested_conversations = set(json.loads(params[1]))
        rows = [
            ("notes", note_id) for note_id in sorted(requested_notes & self.note_ids)
        ]
        rows += [
            ("chat_history", conversation_id)
            for conversation_id in sorted(
                requested_conversations & self.conversation_ids
            )
        ]
        return _RowsDouble(rows)


@pytest.mark.asyncio
async def test_media_handoff_evidence_bundle_reaches_capture_as_real_context():
    """The exact launch a media handoff stages must let
    `capture_console_staged_evidence_for_chat` return REAL context, not the
    `LocalRagContextResult(None, None)` it always returned before this fix
    (the handoff carried no `evidence_bundle` key at all).

    The pre-existing (byte-unchanged) snippet formula in
    `_stage_handoff_as_console_live_work` is `payload.display_summary or
    payload.body` -- and `library_screen.py`'s real media/conversation
    handoffs set `display_summary` to a short "Media staged: <title>" label
    rather than the body excerpt, so that label -- not the raw body -- is
    what reaches the model here. This test asserts the actual production
    formula's output, not the excerpt; see the report for this as a
    separate, pre-existing content-fidelity note out of this task's scope.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        screen = _active_console_screen(host)
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        screen._stage_handoff_as_console_live_work(_media_handoff_payload())
        await pilot.pause()
        launch = screen._pending_console_launch_context
        assert launch is not None

    capture_app = SimpleNamespace(media_db=_ExistingMediaDBDouble("media-77"))
    result = await capture_console_staged_evidence_for_chat(
        capture_app, launch, user_message="What does this media say?"
    )

    assert isinstance(result, LocalRagContextResult)
    assert result.context is not None
    assert "Media staged: Transformer notes" in result.context


@pytest.mark.asyncio
async def test_notes_handoff_evidence_bundle_reaches_capture_as_real_context():
    """Same round trip as the media case, for a Library note handoff."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        screen = _active_console_screen(host)
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        screen._stage_handoff_as_console_live_work(_notes_handoff_payload())
        await pilot.pause()
        launch = screen._pending_console_launch_context
        assert launch is not None

    capture_app = SimpleNamespace(
        chachanotes_db=_ExistingChaChaDBDouble(note_ids={"42"})
    )
    result = await capture_console_staged_evidence_for_chat(
        capture_app, launch, user_message="What should I follow up on?"
    )

    assert isinstance(result, LocalRagContextResult)
    assert result.context is not None
    assert "Discussed the Q3 roadmap" in result.context


@pytest.mark.asyncio
async def test_conversation_handoff_evidence_bundle_reaches_capture_as_real_context():
    """Same round trip as the media/notes cases, for a Library conversation
    handoff -- the third of the three brief-named kinds. Rounds out the
    suite so all three kinds this task actually fixes (media, notes,
    conversations) get the same deep, unmocked capture-round-trip proof,
    not just two of the three."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        screen = _active_console_screen(host)
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        screen._stage_handoff_as_console_live_work(_conversation_handoff_payload())
        await pilot.pause()
        launch = screen._pending_console_launch_context
        assert launch is not None

    capture_app = SimpleNamespace(
        chachanotes_db=_ExistingChaChaDBDouble(conversation_ids={"conv-9"})
    )
    result = await capture_console_staged_evidence_for_chat(
        capture_app, launch, user_message="What did we plan?"
    )

    assert isinstance(result, LocalRagContextResult)
    assert result.context is not None
    # `item_type="conversation"` maps to `CanonicalSourceKind.CHAT_HISTORY`,
    # labeled "CHAT HISTORY" in the formatted context (`_SOURCE_LABELS` in
    # `RAG_Search/local_citation_capture.py`).
    assert "CHAT HISTORY" in result.context
    # Same pre-existing `display_summary`-over-`body` snippet formula as the
    # media case (see the report): the conversation handoff also sets a
    # generic `display_summary`, so that -- not the multi-line body -- is
    # what reaches the model here.
    assert "Conversation staged: Prior planning chat" in result.context


@pytest.mark.asyncio
async def test_console_send_blocked_reason_sendable_for_media_handoff_with_new_bundle():
    """Send-gating blast radius: `_console_send_blocked_reason` only checks
    available evidence for a RAG-labeled source (`_source_mentions_rag`).
    `"library"` never matches that token, so a media handoff gaining a real
    `evidence_bundle` here must not newly block the send."""
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1-2025-04-14",
        },
        "api_settings": {"openai": {"api_key": "configured-test-key"}},
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = "gpt-4.1-2025-04-14"
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        screen = _active_console_screen(host)
        await _wait_for_selector(screen, pilot, "#console-native-composer")

        screen._stage_handoff_as_console_live_work(_media_handoff_payload())
        await pilot.pause()

        launch = screen._pending_console_launch_context
        assert launch is not None
        assert isinstance(launch.payload.get("evidence_bundle"), dict)
        assert chat_screen_module._source_mentions_rag(launch.source) is False
        assert screen._console_send_blocked_reason() == ""
