from pathlib import Path

import pytest
from textual.widgets import Button, Input, Select, Static

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)
from tldw_chatbook.Chat.chat_models import ChatSessionData
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.compact_model_bar import CompactModelBar


REPO_ROOT = Path(__file__).resolve().parents[2]
GATE15_EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-3/"
    "2026-05-07-gate-1-5-console-internals-decomposition.md"
)
ROADMAP = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_3_README = Path("Docs/superpowers/qa/product-maturity/phase-3/README.md")
TASK_10_6 = Path(
    "backlog/tasks/task-10.6 - "
    "Product-Maturity-Phase-3.6-Gate-1.5-Console-Internals-Decomposition.md"
)


def _repo_text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_gate15_console_internals_evidence_is_tracked():
    evidence = _repo_text(GATE15_EVIDENCE)
    roadmap = _repo_text(ROADMAP)
    readme = _repo_text(PHASE_3_README)
    task = _repo_text(TASK_10_6)

    for heading in (
        "## Scope",
        "## Walkthrough",
        "## Functional Result",
        "## Verification",
        "## Defects",
        "## Residual Risk",
        "## Exit Decision",
    ):
        assert heading in evidence
    for selector in (
        "#console-control-bar",
        "#console-session-surface",
        "#console-native-composer",
        "#console-run-inspector",
    ):
        assert selector in evidence
    assert "/Users/macbook-dev/" not in evidence
    assert GATE15_EVIDENCE.name in readme
    assert GATE15_EVIDENCE.name in roadmap
    assert "Gate 1.5" in roadmap
    assert "TASK-10.6" in roadmap
    assert "status: Done" in task
    assert "## Implementation Notes" in task


@pytest.mark.asyncio
async def test_console_gate15_does_not_mount_full_legacy_chat_window_chrome():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        assert console.query_one("#console-control-bar")
        assert console.query_one("#console-session-surface")
        assert console.query_one("#console-native-composer")

        assert len(console.query("#chat-enhanced-sidebar")) == 0
        assert len(console.query("#toggle-chat-left-sidebar")) == 0
        assert len(console.query("#chat-main-content")) == 0


@pytest.mark.asyncio
async def test_console_gate15_keeps_existing_chat_send_control_reachable():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        text = _visible_text(console)
        assert "Send" in text
        assert "Stop" in text
        assert "Attach" in text
        assert "Save Chatbook" in text
        send_controls = [
            button
            for button in console.query(Button)
            if (button.id or "").startswith("send-stop-chat")
            or button.has_class("console-send-button")
        ]
        assert send_controls
        assert console.query_one("#console-stop-generation", Button)
        assert console.query_one("#console-attach-context", Button)
        assert console.query_one("#console-save-chatbook", Button)


@pytest.mark.asyncio
async def test_console_composer_status_renders_session_metadata_as_plain_text():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.sync_session_data(
            ChatSessionData(
                tab_id="metadata",
                title="[red]Injected[/red]",
                runtime_backend="[blue]server[/blue]",
                assistant_id="[green]persona[/green]",
                scope_type="workspace",
                workspace_id="[yellow]workspace[/yellow]",
            )
        )
        await pilot.pause()

        status = console.query_one("#console-composer-status", Static)
        rendered = status.render()
        plain = getattr(rendered, "plain", str(rendered))

        assert "[red]Injected[/red]" in plain
        assert "[blue]server[/blue]" in plain
        assert "[green]persona[/green]" in plain
        assert "[yellow]workspace[/yellow]" in plain


@pytest.mark.asyncio
async def test_console_native_control_bar_and_staged_context_reflect_pending_handoff():
    app = _build_test_app()
    app.pending_console_launch = {
        "source": "Library Search/RAG",
        "title": "Transformer notes",
        "status": "ready",
        "recovery": "Review citations before sending.",
        "payload": {"source_id": "note-1", "citation_count": 2},
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-control-bar")

        text = _visible_text(console)
        assert "Provider:" in text
        assert "Model:" in text
        assert "Assistant: General" in text
        assert "RAG:" in text
        assert "Sources: 1 staged" in text
        assert "Transformer notes" in text
        assert "citation_count: 2" in text
        assert "Review citations before sending." in text


@pytest.mark.asyncio
async def test_console_native_control_bar_uses_existing_compact_model_sync_seam():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-compact-model-bar")

        compact_bar = console.query_one("#console-compact-model-bar")
        console._sync_compact_shell_controls(
            model="console-test-model",
            temperature="0.2",
        )

        assert (
            compact_bar.query_one("#compact-api-model", Select).value
            == "console-test-model"
        )
        assert compact_bar.query_one("#compact-temperature", Input).value == "0.2"


@pytest.mark.asyncio
async def test_console_mounts_only_one_compact_model_bar():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-compact-model-bar")

        assert len(console.query(CompactModelBar)) == 1
        assert len(console.query("#compact-model-bar")) == 0


@pytest.mark.asyncio
async def test_console_control_labels_refresh_after_compact_control_sync():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-compact-model-bar")

        compact_bar = console.query_one("#console-compact-model-bar")
        provider_select = compact_bar.query_one("#compact-api-provider", Select)
        provider = next(
            value
            for _, value in provider_select._options
            if isinstance(value, str)
        )

        console._sync_compact_shell_controls(
            provider=provider,
            model="console-test-model",
        )
        await pilot.pause()

        assert provider in str(console.query_one("#console-provider-label").renderable)
        assert "console-test-model" in str(
            console.query_one("#console-model-label").renderable
        )


def test_console_control_state_tolerates_missing_config_and_precise_rag_source():
    app = _build_test_app()
    app.app_config = None
    screen = ChatScreen(app)

    assert screen._chat_default_value("provider") is None

    non_rag_state = screen._build_console_control_state(
        ConsoleLiveWorkLaunch(source="storage", title="Storage queue"),
    )
    rag_state = screen._build_console_control_state(
        ConsoleLiveWorkLaunch(source="Library Search/RAG", title="RAG result"),
    )

    assert non_rag_state.rag_label == "RAG: off"
    assert rag_state.rag_label == "RAG: on"


def test_console_control_state_tolerates_missing_launch_source():
    app = _build_test_app()
    screen = ChatScreen(app)

    state = screen._build_console_control_state(
        ConsoleLiveWorkLaunch(source=None, title="Unknown source"),
    )

    assert state.rag_label == "RAG: off"


def test_console_control_and_inspector_share_effective_provider_model_sources():
    app = _build_test_app()
    app.app_config = {"chat_defaults": {"model": "reactive-model"}}
    app.chat_api_provider_value = "ReactiveOpenAI"
    screen = ChatScreen(app)

    control_state = screen._build_console_control_state(None)
    inspector_state = screen._build_console_inspector_state(None)
    rows_by_label = {row.label: row for row in inspector_state.rows}

    assert control_state.provider_label == "Provider: ReactiveOpenAI"
    assert control_state.model_label == "Model: reactive-model"
    assert rows_by_label["Provider"].text == "Provider: ready"


@pytest.mark.asyncio
async def test_console_run_inspector_shows_blocked_provider_and_missing_rag_source():
    app = _build_test_app()
    app.app_config = {"chat_defaults": {}}
    app.console_provider_ready = False
    app.pending_console_launch = {
        "source": "Library Search/RAG",
        "title": "Grounded answer",
        "status": "ready",
        "recovery": "Attach a source before asking the model.",
        "payload": {},
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-provider")

        assert "Provider: blocked" in str(
            console.query_one("#console-inspector-provider", Static).renderable
        )
        assert "Select a provider and model before sending." in str(
            console.query_one("#console-inspector-provider", Static).renderable
        )
        assert "RAG/source: missing source" in str(
            console.query_one("#console-inspector-rag-source", Static).renderable
        )
        assert console.query_one("#console-inspector-review-tool-call", Button).disabled is True
        assert "No tool calls are ready for review." in str(
            console.query_one("#console-inspector-review-tool-call-reason", Static).renderable
        )


@pytest.mark.asyncio
async def test_console_run_inspector_exposes_pending_approval_and_chatbook_artifact_actions():
    app = _build_test_app()
    app.console_pending_approval_count = 1
    app.console_tool_count = 1
    app.pending_console_launch = {
        "source": "artifacts",
        "title": "Grounded Answer Chatbook",
        "status": "ready",
        "recovery": "Review this Chatbook artifact in Console or return to Artifacts.",
        "payload": {"target_id": "local:chatbook:77", "chatbook_id": 77},
        "action_label": "Open Chatbook artifact",
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-review-approval")

        assert "Approvals: 1 pending" in str(
            console.query_one("#console-inspector-approvals", Static).renderable
        )
        assert "Artifacts: Chatbook artifact available" in str(
            console.query_one("#console-inspector-artifacts", Static).renderable
        )
        assert console.query_one("#console-inspector-review-approval", Button).disabled is False
        assert console.query_one("#console-inspector-review-tool-call", Button).disabled is False
        assert console.query_one("#console-inspector-save-chatbook", Button).disabled is False
        assert console.query_one("#console-live-work-primary-action", Button).disabled is False
