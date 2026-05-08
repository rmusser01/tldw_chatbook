import time
from pathlib import Path

import pytest
from textual.events import Paste
from textual.widgets import Button, Footer, Input, Select, Static

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


class StaticConsoleLibraryRagSearchService:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def search(self, query, scope, mode, **kwargs):
        self.calls.append(
            {
                "query": query,
                "scope": scope,
                "mode": mode,
                **kwargs,
            }
        )
        return self.result


async def _wait_for_console_library_rag_button_state(
    console,
    pilot,
    *,
    disabled: bool,
    tooltip_contains: str = "",
    timeout: float = 2.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        buttons = list(console.query("#console-run-library-rag"))
        if buttons:
            button = buttons[0]
            tooltip = str(button.tooltip or "")
            if button.disabled is disabled and tooltip_contains in tooltip:
                await pilot.pause()
                return
        await pilot.pause(0.01)
    raise AssertionError(
        "Timed out waiting for Console Library RAG run button "
        f"disabled={disabled!r} tooltip={tooltip_contains!r}"
    )


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

    async with host.run_test(size=(212, 64)) as pilot:
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

    async with host.run_test(size=(212, 64)) as pilot:
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
async def test_console_native_composer_spans_below_workbench_with_single_input_surface():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        workbench = console.query_one("#console-workspace-grid")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        command_input = composer.query_one("#console-command-input", Input)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        send_button = composer.query_one("#console-send-message", Button)
        stop_button = composer.query_one("#console-stop-generation", Button)
        attach_button = composer.query_one("#console-attach-context", Button)
        save_button = composer.query_one("#console-save-chatbook", Button)
        legacy_inputs = [
            widget
            for widget in console.query(".chat-input-area")
            if widget.region.height > 0 and widget.region.width > 0
        ]

        assert composer.region.x == workbench.region.x
        assert composer.region.width == workbench.region.width
        assert composer.region.y > workbench.region.y
        assert composer.region.y + composer.region.height <= console.size.height
        assert command_input.display is False
        assert visible_draft.region.width > 20
        assert 4 <= composer.region.height <= 6
        assert visible_draft.region.height == 1
        composer.load_draft("visible composer text")
        await pilot.pause(0.1)
        assert composer.draft_text() == "visible composer text"
        assert "visible composer text" in visible_draft.renderable.plain
        assert "visible composer text" in _visible_text(composer)
        for action_button in (send_button, stop_button, attach_button, save_button):
            assert action_button.compact is True
            assert action_button.region.height == 1
            assert action_button.region.y + action_button.region.height <= composer.region.y + composer.region.height
        assert str(send_button.label) == "Send"
        assert str(stop_button.label) == "Stop"
        assert str(attach_button.label) == "Attach"
        assert str(save_button.label) == "Save Chatbook"
        assert save_button.region.width >= 20
        assert legacy_inputs == []


@pytest.mark.asyncio
async def test_console_native_composer_receives_typing_on_open():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        command_input = composer.query_one("#console-command-input", Input)
        visible_draft = composer.query_one("#console-command-visible-text", Static)

        assert command_input.display is False
        await pilot.press("v", "i", "s", "i", "b", "l", "e")
        await pilot.pause(0.1)

        assert composer.draft_text() == "visible"
        assert "visible" in visible_draft.renderable.plain
        assert "visible" in _visible_text(composer)


@pytest.mark.asyncio
async def test_console_native_composer_auto_expands_for_long_drafts():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        long_draft = "long composer qa " * 20

        composer.load_draft(long_draft)
        await pilot.pause(0.2)

        visible_plain = visible_draft.renderable.plain
        assert composer.draft_text() == long_draft
        assert composer.region.height > 5
        assert composer.region.height <= 10
        assert visible_draft.region.height > 1
        assert visible_draft.region.height <= 4
        assert "\n" in visible_plain
        assert "long composer qa" in visible_plain


@pytest.mark.asyncio
async def test_console_native_composer_paste_keeps_large_draft_bounded_and_visible():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "pasted composer qa " * 80

        console.on_paste(Paste(pasted_text))
        await pilot.pause(0.2)

        visible_plain = visible_draft.renderable.plain
        assert composer.draft_text() == pasted_text
        assert composer.region.height <= 10
        assert visible_draft.region.height <= 4
        assert "pasted composer qa" in visible_plain
        assert len(visible_plain) < len(pasted_text)


@pytest.mark.asyncio
async def test_console_native_composer_captures_printable_typing_from_non_text_focus():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        command_input = composer.query_one("#console-command-input", Input)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        save_button = composer.query_one("#console-save-chatbook", Button)

        save_button.focus()
        await pilot.pause(0.1)
        await pilot.press("k")
        await pilot.pause(0.1)

        assert command_input.display is False
        assert composer.draft_text() == "k"
        assert "k" in visible_draft.renderable.plain
        assert "k" in _visible_text(composer)


@pytest.mark.asyncio
async def test_console_send_without_ready_runtime_shows_native_blocked_event(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _build_test_app()
    app.app_config.setdefault("api_settings", {}).setdefault("openai", {})["api_key"] = ""
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello console")
        send_button = console.query_one("#console-send-message", Button)
        await console.handle_console_send_message(Button.Pressed(send_button))
        await pilot.pause(0.2)

        text = _visible_text(console)
        assert "Console send blocked" in text
        assert "Missing API key" in text
        assert "Internal Error" not in text
        assert "Missing UI elements" not in text
        assert composer.draft_text() == "hello console"


@pytest.mark.asyncio
async def test_console_enter_sends_native_composer_draft(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _build_test_app()
    app.app_config.setdefault("api_settings", {}).setdefault("openai", {})["api_key"] = ""
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        await pilot.press("h", "e", "l", "l", "o")
        await pilot.press("enter")
        await pilot.pause(0.2)

        text = _visible_text(console)
        assert "Console send blocked" in text
        assert "Missing API key" in text
        assert "Internal Error" not in text
        assert "Missing UI elements" not in text
        assert composer.draft_text() == "hello"


@pytest.mark.asyncio
async def test_console_transcript_new_tab_control_is_fully_visible():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#new-chat-tab-button")

        transcript = console.query_one("#console-transcript-region")
        new_tab_button = console.query_one("#new-chat-tab-button", Button)
        label = str(new_tab_button.label)

        assert label == "New tab"
        assert new_tab_button.region.width >= len(label) + 2
        assert new_tab_button.region.x >= transcript.region.x
        assert new_tab_button.region.x + new_tab_button.region.width <= (
            transcript.region.x + transcript.region.width
        )
        assert "New tab" in _visible_text(console)


@pytest.mark.asyncio
async def test_console_app_footer_status_bar_remains_visible_below_console():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        footer = console.query_one(Footer)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        assert footer.region.height == 1
        assert footer.region.y == console.size.height - 1
        assert composer.region.y + composer.region.height <= footer.region.y


@pytest.mark.asyncio
async def test_console_inspector_live_work_sources_stay_near_top():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(220, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-live-work-source-readiness")

        inspector = console.query_one("#console-run-inspector")
        inspector_state = console.query_one("#console-run-inspector-state")
        source_readiness = console.query_one("#console-live-work-source-readiness")

        assert inspector_state.region.height <= 14
        assert source_readiness.region.y <= inspector.region.y + inspector_state.region.height + 2
        assert source_readiness.region.height <= 18


@pytest.mark.asyncio
async def test_console_inspector_source_readiness_rows_fit_without_tooltip_overlay():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(196, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-live-work-source-readiness")

        run_rag = console.query_one("#console-run-library-rag", Button)
        scope = console.query_one("#console-library-rag-scope", Static)
        rows = list(console.query(".console-live-work-source-row"))

        assert run_rag.disabled is True
        assert str(run_rag.tooltip or "") == ""
        scope_plain = getattr(scope.render(), "plain", str(scope.render()))
        assert len(scope_plain) <= scope.region.width
        assert rows
        for row in rows:
            rendered = row.render()
            plain = getattr(rendered, "plain", str(rendered))
            assert len(plain) <= row.region.width


@pytest.mark.asyncio
async def test_console_empty_inspector_hides_disabled_actions_until_actionable():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(196, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-run-inspector-state")

        for selector in (
            "#console-inspector-review-approval",
            "#console-inspector-review-tool-call",
            "#console-inspector-save-chatbook",
        ):
            button = console.query_one(selector, Button)
            assert button.disabled is True
            assert button.region.height == 0
            assert str(button.tooltip or "") == ""


@pytest.mark.asyncio
async def test_console_workbench_weights_transcript_as_primary_region():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-grid")

        staged = console.query_one("#console-staged-context-tray")
        main = console.query_one("#console-main-column")
        inspector = console.query_one("#console-run-inspector")
        transcript = console.query_one("#console-transcript-region")

        assert main.region.width > staged.region.width
        assert main.region.width > inspector.region.width
        assert staged.region.width >= 40
        assert inspector.region.width >= 40
        assert transcript.region.width == main.region.width


@pytest.mark.asyncio
async def test_console_workbench_panes_have_visible_terminal_frames():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-grid")

        for selector in (
            "#console-workspace-grid",
            "#console-staged-context-tray",
            "#console-transcript-region",
            "#console-run-inspector",
            "#console-native-composer",
        ):
            border = console.query_one(selector).styles.border
            assert border.top[0] == "solid", f"{selector} missing top frame"
            assert border.right[0] == "solid", f"{selector} missing right frame"
            assert border.bottom[0] == "solid", f"{selector} missing bottom frame"
            assert border.left[0] == "solid", f"{selector} missing left frame"


@pytest.mark.asyncio
async def test_console_empty_staged_context_recovery_fits_tray():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-staged-context-recovery")

        recovery = console.query_one("#console-staged-context-recovery")
        rendered = recovery.render()
        plain = getattr(rendered, "plain", str(rendered))

        assert len(plain) <= recovery.region.width - 4


@pytest.mark.asyncio
async def test_console_control_bar_renders_readable_summary_line():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-control-bar")

        summary = console.query_one("#console-control-status-line", Static)
        plain = getattr(summary.render(), "plain", str(summary.render()))

        assert "Provider:" in plain
        assert " | Model:" in plain
        assert " | Assistant:" in plain
        assert " | Sources:" in plain


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


@pytest.mark.asyncio
async def test_console_rag_action_requests_library_retrieval_and_stages_result():
    app = _build_test_app()
    service = StaticConsoleLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "Incident Review",
                    "snippet": "Expired credential caused the incident.",
                    "score": 0.93,
                    "source_id": "note-42",
                    "chunk_id": "chunk-7",
                    "runtime_backend": "local-fts",
                    "citations": [{"label": "Incident Review p.2"}],
                }
            ],
            "runtime_backend": "local-fts",
        }
    )
    app.library_rag_search_service = service
    host = ConsoleHarness(app)
    query = "Why did the incident happen?"

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-run-library-rag")

        assert "Scope: notes, media, conversations" in _visible_text(console)
        query_input = console.query_one("#console-library-rag-query-input", Input)
        query_input.value = query
        await pilot.pause(0.1)

        run_button = console.query_one("#console-run-library-rag", Button)
        assert run_button.disabled is False
        run_button.press()
        await _wait_for_selector(console, pilot, "#console-live-work-payload-source-id")

        assert service.calls == [
            {
                "query": query,
                "scope": ("notes", "media", "conversations"),
                "mode": "rag",
                "top_k": 5,
                "include_citations": True,
            }
        ]
        text = _visible_text(console)
        assert "RAG/source: staged from Library Search/RAG" in text
        assert "Title: Incident Review" in text
        assert "source_id: note-42" in text
        assert "chunk_id: chunk-7" in text
        assert "Review citations before sending." in text


@pytest.mark.asyncio
async def test_console_rag_query_validation_blocks_unsafe_markup():
    app = _build_test_app()
    service = StaticConsoleLibraryRagSearchService({"results": []})
    app.library_rag_search_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-run-library-rag")

        console.query_one("#console-library-rag-query-input", Input).value = (
            "<script>alert('bad')</script>"
        )
        await _wait_for_console_library_rag_button_state(
            console,
            pilot,
            disabled=True,
        )
        assert str(console.query_one("#console-run-library-rag", Button).tooltip or "") == ""

        assert console._console_library_rag_query == ""
        assert service.calls == []


@pytest.mark.asyncio
async def test_console_rag_action_without_service_stages_recoverable_blocker():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-run-library-rag")

        console.query_one("#console-library-rag-query-input", Input).value = "What changed?"
        await pilot.pause(0.1)
        console.query_one("#console-run-library-rag", Button).press()
        await _wait_for_selector(console, pilot, "#console-live-work-status")

        text = _visible_text(console)
        assert "Status: blocked" in text
        assert "RAG/source: unavailable" in text
        assert "Unavailable: Library Search/RAG retrieval." in text
        assert "Owner: Library retrieval service." in text
