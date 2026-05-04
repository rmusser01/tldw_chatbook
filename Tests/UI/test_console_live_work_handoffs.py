"""Console live-work launch and staged-context handoff boundary tests."""

import threading
from pathlib import Path
from unittest.mock import Mock

import pytest
from textual.app import App

from Tests.UI.test_destination_shells import DestinationHarness
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Home.dashboard_state import HomeActiveWorkItem, HomeDashboardInput
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.schedules_screen import SchedulesScreen


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
        source="W+C",
        title="Daily security feed",
        payload={"target_id": "local:watchlist_run:91", "run_id": 91},
        status="failed",
        recovery="Review the W+C run details or retry from W+C.",
        action_label="Open W+C run",
    )

    card_state = ConsoleLiveWorkStatusCardState.from_launch(launch)

    assert card_state.primary_action is not None
    assert card_state.primary_action.widget_id == "console-live-work-primary-action"
    assert card_state.primary_action.label == "Open W+C run"
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
        "W+C: Connected - Home W+C active work can open and route run details in Console."
    )
    assert "console-live-work-source-connected" in rows_by_id["console-live-work-source-wc"].classes
    assert rows_by_id["console-live-work-source-schedules"].text == (
        "Schedules: Connected - Schedules active work can open Console when adapter context exists."
    )
    assert "console-live-work-source-connected" in rows_by_id["console-live-work-source-schedules"].classes
    assert rows_by_id["console-live-work-source-rag"].text == (
        "RAG: Connected - Search/RAG results can stage retrieved evidence in Console."
    )
    assert "console-live-work-source-connected" in rows_by_id["console-live-work-source-rag"].classes
    for source_id in (
        "console-live-work-source-workflows",
        "console-live-work-source-acp",
        "console-live-work-source-mcp",
    ):
        assert "Not wired" in rows_by_id[source_id].text
        assert "console-live-work-source-unavailable" in rows_by_id[source_id].classes
    assert rows_by_id["console-live-work-source-artifacts"].text == (
        "Artifacts: Connected - Latest local Chatbook artifacts can launch into Console."
    )
    assert "console-live-work-source-connected" in rows_by_id["console-live-work-source-artifacts"].classes


def test_app_console_live_work_primary_action_routes_wc_run_details():
    ConsoleLiveWorkLaunch = _load_console_live_work_contract()
    app = _build_test_app()
    app.post_message = Mock()
    app.notify = Mock()
    launch = ConsoleLiveWorkLaunch.from_values(
        source="W+C",
        title="Daily security feed",
        payload={"target_id": "local:watchlist_run:91", "run_id": 91},
        status="failed",
        recovery="Review the W+C run details or retry from W+C.",
        action_label="Open W+C run",
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
            "No active schedule run is available for Console follow.",
        ),
        (
            "workflows",
            "workflows-launch-in-console",
            "Console launch is unavailable until workflow execution payloads are wired.",
        ),
        (
            "acp",
            "acp-follow-in-console",
            "Console follow is unavailable until ACP session payloads are wired.",
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
        assert "No active schedule run is available for Console follow." in _screen_static_text(screen)

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
                source="W+C",
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
        assert "No active W+C run is available for Console follow." in text

    app.open_active_home_item_in_console.assert_not_called()


@pytest.mark.asyncio
async def test_watchlists_destination_routes_latest_active_run_to_console():
    app = _build_test_app()
    app.home_active_work_adapter = StaticHomeActiveWorkAdapter(
        (
            HomeActiveWorkItem(
                item_id="local:watchlist_run:5",
                title="Daily security feed",
                source="W+C",
                status="failed",
                detail_route="subscriptions",
                console_available=True,
            ),
            HomeActiveWorkItem(
                item_id="local:watchlist_run:9",
                title="Other source",
                source="W+C",
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
        assert "No active W+C run is available for Console follow." in _screen_static_text(screen)

    logger.warning.assert_called_once()
    assert "W+C Console follow" in logger.warning.call_args.args[0]
    assert logger.warning.call_args.kwargs["exc_info"] is True
    app.open_active_home_item_in_console.assert_not_called()


@pytest.mark.asyncio
async def test_watchlists_destination_click_uses_item_promised_by_button_label():
    app = _build_test_app()
    app.home_active_work_adapter = RotatingHomeActiveWorkAdapter(
        (
            HomeActiveWorkItem(
                item_id="local:watchlist_run:5",
                title="First visible run",
                source="W+C",
                status="failed",
                detail_route="subscriptions",
                console_available=True,
            ),
        ),
        (
            HomeActiveWorkItem(
                item_id="local:watchlist_run:7",
                title="Newer unseen run",
                source="W+C",
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
                source="W+C",
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
        assert "No active schedule run is available for Console follow." in _screen_static_text(screen)
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
        await pilot.pause(0.1)
        screen = _active_console_screen(host)
        button = screen.query_one("#artifacts-use-in-console")

        assert button.disabled is True
        assert str(button.label) == "Console launch unavailable"
        assert "No local Chatbook artifact is available for Console launch." in _screen_static_text(screen)

        await pilot.click("#artifacts-use-in-console")
        await pilot.pause(0.1)

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
        await pilot.pause(0.1)
        screen = _active_console_screen(host)
        button = screen.query_one("#artifacts-use-in-console")

        assert button.disabled is False
        assert "Research Pack" in str(button.label)
        text = _screen_static_text(screen)
        assert "Console can launch latest Chatbook artifact: Research Pack." in text
        assert "A portable research bundle" in text

        await pilot.click("#artifacts-use-in-console")
        await pilot.pause(0.1)

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


@pytest.mark.parametrize(
    ("route", "button_id"),
    [
        ("library", "library-use-in-console"),
        ("personas", "personas-attach-to-console"),
        ("skills", "skills-attach-to-console"),
    ],
)
@pytest.mark.asyncio
async def test_staged_context_actions_use_chat_handoff_not_live_launch(route, button_id):
    app = _build_test_app()
    app.open_chat_with_handoff = Mock()
    app.open_console_for_live_work = Mock()
    host = DestinationHarness(app, route)

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        await pilot.click(f"#{button_id}")
        await pilot.pause(0.1)

    app.open_chat_with_handoff.assert_called_once()
    payload = app.open_chat_with_handoff.call_args.args[0]
    assert isinstance(payload, ChatHandoffPayload)
    assert payload.source == route
    app.open_console_for_live_work.assert_not_called()
    assert getattr(app, "pending_console_launch", None) in (None, {})


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
            "W+C: Connected - Home W+C active work can open and route run details in Console."
        )
        assert "Workflows: Not wired" in str(screen.query_one("#console-live-work-source-workflows").renderable)
        assert "Schedules: Connected" in str(screen.query_one("#console-live-work-source-schedules").renderable)
        assert "ACP: Not wired" in str(screen.query_one("#console-live-work-source-acp").renderable)
        assert "MCP: Not wired" in str(screen.query_one("#console-live-work-source-mcp").renderable)
        assert "RAG: Connected" in str(screen.query_one("#console-live-work-source-rag").renderable)
        assert "Artifacts: Connected" in str(screen.query_one("#console-live-work-source-artifacts").renderable)


@pytest.mark.asyncio
async def test_console_wc_live_work_action_button_routes_run_details():
    app = _build_test_app()
    app.pending_console_launch = {
        "source": "W+C",
        "title": "Daily security feed",
        "payload": {"target_id": "local:watchlist_run:91", "run_id": 91},
        "status": "failed",
        "recovery": "Review the W+C run details or retry from W+C.",
        "action_label": "Open W+C run",
    }
    app.post_message = Mock()
    app.notify = Mock()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_console_screen(host)
        button = screen.query_one("#console-live-work-primary-action")
        assert str(button.label) == "Open W+C run"

        await pilot.click("#console-live-work-primary-action")
        await pilot.pause(0.1)

    assert app.pending_subscription_initial_tab == "watchlist-runs"
    assert app.pending_subscription_watchlist_run_id == "local:watchlist_run:91"
    app.post_message.assert_called_once()
    assert app.post_message.call_args.args[0].screen_name == "subscriptions"
    app.notify.assert_not_called()
