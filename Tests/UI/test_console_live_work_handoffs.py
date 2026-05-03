"""Console live-work launch and staged-context handoff boundary tests."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from textual.app import App

from Tests.UI.test_destination_shells import DestinationHarness
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE3_STATUS_CARD_EVIDENCE = (
    REPO_ROOT / "Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-console-live-work-status-card-seam.md"
)
PHASE3_HOME_WC_CONSOLE_EVIDENCE = (
    REPO_ROOT / "Docs/superpowers/qa/unified-shell/phase-3/2026-05-03-home-wc-active-work-console-launch.md"
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


@pytest.mark.parametrize(
    ("route", "button_id", "expected_copy"),
    [
        (
            "watchlists_collections",
            "watchlists-follow-in-console",
            "Console follow is unavailable until watchlist and collection live-work payloads are wired.",
        ),
        (
            "schedules",
            "schedules-follow-in-console",
            "Console recovery is unavailable until schedule run payloads are wired.",
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


@pytest.mark.parametrize(
    ("route", "button_id"),
    [
        ("library", "library-use-in-console"),
        ("artifacts", "artifacts-use-in-console"),
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
