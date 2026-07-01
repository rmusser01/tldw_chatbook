import time
from pathlib import Path

import pytest
from rich.console import Console
from rich.text import Text
from textual.app import App, ComposeResult
from textual.events import Paste
from textual.widgets import Button, Footer, Input, Select, Static

from Tests.UI.test_destination_shells import (
    _build_test_app,
    _wait_for_selector,
    _wait_for_visible_text,
)
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)
from tldw_chatbook.Chat.chat_models import ChatSessionData
from tldw_chatbook.Chat.console_display_state import (
    ConsoleInspectorState,
    ConsoleStagedContextState,
    build_console_evidence_display_state,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.chat_screen import (
    CONSOLE_PROVIDER_CONFIGURE_API_KEY_LABEL,
    ChatScreen,
)
from tldw_chatbook.config import resolve_provider_name
from tldw_chatbook.Widgets.Console import ConsoleComposerBar, ConsoleStagedContextTray
from tldw_chatbook.Widgets.compact_model_bar import CompactModelBar


REPO_ROOT = Path(__file__).resolve().parents[2]


class StagedContextHarness(App):
    def __init__(self, state: ConsoleStagedContextState) -> None:
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        yield ConsoleStagedContextTray(self.state, id="console-staged-context-tray")


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


def _assert_visible_literal_projection(visible_plain: str, expected: str) -> None:
    """Assert the bounded composer is showing literal text, allowing wrap/clipping."""
    flattened = visible_plain.replace("\n", "")
    if flattened.startswith("..."):
        visible_suffix = flattened[3:]
        assert visible_suffix
        assert expected.endswith(visible_suffix)
    else:
        assert flattened == expected


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


class _PressedEvent:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def _configure_native_ready_console(app, model: str = "local-model") -> None:
    app.app_config = {
        "chat_defaults": {"provider": "llama_cpp", "model": model},
        "api_settings": {
            "llama_cpp": {
                "api_url": "http://127.0.0.1:9099",
                "model": model,
            },
        },
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = model


def _configure_openai_missing_key_console(app, model: str = "gpt-4.1-2025-04-14") -> None:
    app.app_config = {
        "chat_defaults": {"provider": "OpenAI", "model": model},
        "api_settings": {"openai": {"api_key": ""}},
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = model


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


async def _open_console_inspector(console, pilot) -> None:
    """Open the persistent Inspector rail and wait for measurable layout."""
    right_rail = console.query_one("#console-right-rail")
    if getattr(right_rail, "display", False) and right_rail.region.width > 0:
        return

    await _wait_for_selector(console, pilot, "#console-inspector-rail-open")
    await pilot.click("#console-inspector-rail-open")

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        right_rail = console.query_one("#console-right-rail")
        inspector_state = console.query_one("#console-run-inspector-state")
        if (
            getattr(right_rail, "display", False)
            and right_rail.region.width > 0
            and inspector_state.region.width > 0
        ):
            await pilot.pause()
            return
        await pilot.pause(0.01)
    raise AssertionError("Timed out waiting for Console Inspector rail to open")


def _wrapped_plain_lines(rendered: object, width: int) -> list[str]:
    rich_text = rendered if isinstance(rendered, Text) else Text(str(rendered))
    console = Console(width=max(1, width), color_system=None)
    return [line.plain for line in rich_text.wrap(console, max(1, width))]


def _assert_single_style_span(renderable: Text, *, style: str, expected_text: str) -> None:
    matching_spans = [span for span in renderable.spans if span.style == style]
    assert len(matching_spans) == 1
    span = matching_spans[0]
    assert renderable.plain[span.start : span.end] == expected_text


def test_console_session_surface_uses_flex_height_not_full_percent_height():
    for stylesheet in (
        Path("tldw_chatbook/css/tldw_cli_modular.tcss"),
        Path("tldw_chatbook/css/components/_agentic_terminal.tcss"),
    ):
        css = _repo_text(stylesheet)

        assert "#console-session-surface,\n#console-chat-tabs" not in css
        assert "#console-session-surface {\n    height: 1fr;" in css
        assert (
            "#console-session-surface {\n"
            "    padding: 0;\n"
            "    margin: 0;\n"
            "    border: none;"
        ) in css
        assert (
            "#console-transcript-title,\n"
            ".console-transcript-title {\n"
            "    height: 1;\n"
            "    min-height: 1;\n"
            "    max-height: 1;\n"
            "    margin: 0;\n"
            "    background: $ds-surface-panel;\n"
            "    color: $ds-text-muted;"
        ) in css
        assert (
            "#console-native-tab-strip,\n"
            ".console-session-tab-strip {\n"
            "    height: 1;\n"
            "    min-height: 1;\n"
            "    max-height: 1;\n"
            "    margin: 0;"
        ) in css
        assert (
            "#console-start-here,\n"
            "#console-action-hints {\n"
            "    display: none;\n"
            "    height: 0;\n"
            "    min-height: 0;"
        ) in css
        assert (
            ".console-left-rail-section.console-settings-summary {\n"
            "    margin: 0 0 1 0;\n"
            "    padding: 0 1;\n"
            "    border: none;"
        ) in css
        assert (
            "#console-staged-context-tray {\n"
            "    height: auto;\n"
            "    min-height: 3;\n"
            "    max-height: 6;"
        ) in css
        assert (
            "#console-workspace-context {\n"
            "    height: 1fr;\n"
            "    min-height: 8;"
        ) in css
        assert (
            ".console-transcript-action-guide {\n"
            "    height: auto;\n"
            "    min-height: 1;"
        ) in css
        assert (
            "#console-left-rail-body {\n"
            "    width: 100%;\n"
            "    min-width: 0;\n"
            "    height: 1fr;\n"
            "    min-height: 0;"
        ) in css
        assert "    overflow-y: auto;" in css
        assert (
            "#console-inspector-rail-body {\n"
            "    width: 100%;\n"
            "    min-width: 0;\n"
            "    height: 1fr;\n"
            "    min-height: 0;"
        ) in css
        assert (
            "#console-composer-actions {\n"
            "    width: 37;"
        ) in css


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
async def test_console_hidden_workbench_strips_do_not_reserve_rows():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-grid")

        control_bar = console.query_one("#console-control-bar")
        command_strip = console.query_one("#console-workbench-command-strip")
        workbench = console.query_one("#console-workspace-grid")

        assert control_bar.styles.display != "none"
        assert control_bar.region.height > 0
        assert command_strip.styles.display == "none"
        assert command_strip.region.height == 0
        assert workbench.region.y >= control_bar.region.y + control_bar.region.height
        assert workbench.region.y > command_strip.region.y + command_strip.region.height


@pytest.mark.asyncio
async def test_console_mode_bar_groups_location_mode_and_readiness():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-mode-bar")

        title = console.query_one("#console-title", Static)
        mode_bar = console.query_one("#console-mode-bar", Static)

        title_plain = getattr(title.render(), "plain", str(title.render()))
        mode_plain = getattr(mode_bar.render(), "plain", str(mode_bar.render()))

        assert title_plain == "Console"
        assert (
            mode_plain
            == "Chat/RAG/Follow | General | Sources 0 | Tools 0 | Approvals 0"
        )


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
        assert "Save" in text
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
        assert str(save_button.label) == "Save"
        assert save_button.region.width >= len("Save")
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
async def test_console_composer_marks_focus_state():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        transcript = console.query_one("#console-native-transcript")

        visible_draft.focus()
        await pilot.pause(0.1)

        assert composer.has_focus_within
        assert composer.has_class("console-composer-focused")

        transcript.can_focus = True
        transcript.focus()
        await pilot.pause(0.1)

        assert not composer.has_focus_within
        assert not composer.has_class("console-composer-focused")


@pytest.mark.asyncio
async def test_console_composer_marks_has_draft_state():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        assert not composer.has_class("console-composer-has-draft")
        composer.load_draft("draft state marker")
        await pilot.pause(0.1)
        assert composer.has_class("console-composer-has-draft")

        composer.clear_draft()
        await pilot.pause(0.1)
        assert not composer.has_class("console-composer-has-draft")

        composer.load_draft(" ")
        await pilot.pause(0.1)
        assert composer.has_class("console-composer-has-draft")


@pytest.mark.asyncio
async def test_console_composer_send_is_primary_only_with_draft():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        send_button = composer.query_one("#console-send-message", Button)

        assert send_button.disabled is False
        assert not send_button.has_class("console-action-primary")
        assert send_button.has_class("console-action-subdued")
        assert send_button.has_class("console-send-inactive")

        composer.load_draft("   ")
        await pilot.pause(0.1)

        assert composer.has_class("console-composer-has-draft")
        assert send_button.disabled is False
        assert not send_button.has_class("console-action-primary")
        assert send_button.has_class("console-action-subdued")
        assert send_button.has_class("console-send-inactive")

        composer.load_draft("ready to send")
        await pilot.pause(0.1)

        assert send_button.disabled is False
        assert send_button.has_class("console-action-primary")
        assert send_button.has_class("console-send-ready")
        assert not send_button.has_class("console-action-subdued")
        assert not send_button.has_class("console-send-inactive")

        composer.clear_draft()
        await pilot.pause(0.1)

        assert send_button.disabled is False
        assert not send_button.has_class("console-action-primary")
        assert send_button.has_class("console-action-subdued")
        assert send_button.has_class("console-send-inactive")


@pytest.mark.asyncio
async def test_console_composer_ranks_actions_by_current_availability():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        send_button = composer.query_one("#console-send-message", Button)
        stop_button = composer.query_one("#console-stop-generation", Button)
        attach_button = composer.query_one("#console-attach-context", Button)
        save_button = composer.query_one("#console-save-chatbook", Button)

        assert send_button.disabled is False
        assert stop_button.disabled is False
        assert save_button.disabled is False
        assert attach_button.disabled is False
        assert attach_button.has_class("console-action-secondary")
        assert save_button.has_class("console-action-secondary")
        assert save_button.has_class("console-save-chatbook-secondary")
        assert save_button.has_class("console-action-subdued")
        assert save_button.has_class("console-action-disabled")
        assert not save_button.has_class("console-action-primary")

        composer.sync_action_state(
            has_draft=True,
            run_active=False,
            can_save_chatbook=True,
        )
        await pilot.pause(0.1)

        assert send_button.has_class("console-action-primary")
        assert save_button.disabled is False
        assert save_button.has_class("console-action-secondary")
        assert save_button.has_class("console-save-chatbook-secondary")
        assert save_button.has_class("console-save-chatbook-ready")
        assert not save_button.has_class("console-action-disabled")
        assert not save_button.has_class("console-action-subdued")
        assert not save_button.has_class("console-action-primary")


@pytest.mark.asyncio
async def test_console_composer_empty_setup_blocked_state_shows_reason():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        recovery = composer.query_one("#console-composer-recovery", Static)
        disabled_reason = composer.query_one("#console-send-disabled-reason", Static)
        send_button = composer.query_one("#console-send-message", Button)

        composer.sync_action_state(
            has_draft=False,
            run_active=False,
            can_save_chatbook=False,
            send_blocked=True,
            setup_blocked_reason="Choose a model in Console Settings before sending.",
        )
        await pilot.pause(0.1)

        assert visible_draft.renderable.plain == ConsoleComposerBar.DRAFT_PLACEHOLDER
        assert recovery.styles.display == "none"
        assert disabled_reason.styles.display == "none"
        assert send_button.disabled is False
        assert send_button.tooltip == "Choose a model in Console Settings before sending."

        composer.load_draft("draft despite missing setup")
        await pilot.pause(0.1)

        assert visible_draft.renderable.plain == "draft despite missing setup"
        assert send_button.disabled is False
        assert send_button.tooltip == "Choose a model in Console Settings before sending."


@pytest.mark.asyncio
async def test_console_composer_active_run_blocker_keeps_generic_placeholder():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        send_button = composer.query_one("#console-send-message", Button)

        composer.sync_action_state(
            has_draft=False,
            run_active=True,
            can_save_chatbook=False,
            send_blocked=True,
            setup_blocked_reason="",
        )
        await pilot.pause(0.1)

        assert "Setup required" not in visible_draft.renderable.plain
        assert visible_draft.renderable.plain == ConsoleComposerBar.DRAFT_PLACEHOLDER
        assert send_button.tooltip == "Wait for the active Console run to finish before sending."


@pytest.mark.asyncio
async def test_console_composer_actions_remain_visible_inside_composer_bounds():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.pause(0.1)

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        actions = composer.query_one("#console-composer-actions")
        send_button = composer.query_one("#console-send-message", Button)
        stop_button = composer.query_one("#console-stop-generation", Button)
        attach_button = composer.query_one("#console-attach-context", Button)
        save_button = composer.query_one("#console-save-chatbook", Button)

        composer_right = composer.region.x + composer.region.width
        assert actions.region.x > visible_draft.region.x
        assert actions.region.x + actions.region.width <= composer_right
        for button in (send_button, stop_button, attach_button, save_button):
            assert button.display is True
            assert button.region.x + button.region.width <= composer_right

        assert str(save_button.label) == "Save"
        assert save_button.tooltip == "No Chatbook artifact is available to save yet."


@pytest.mark.asyncio
async def test_console_composer_save_chatbook_routes_available_artifact_action():
    app = _build_test_app()
    handled_launches = []
    app.pending_console_launch = {
        "source": "artifacts",
        "title": "Grounded Answer Chatbook",
        "status": "ready",
        "payload": {"target_id": "local:chatbook:77", "chatbook_id": 77},
        "action_label": "Open Chatbook artifact",
    }
    app.open_console_live_work_primary_action = lambda launch: handled_launches.append(launch) or True
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        save_button = console.query_one("#console-save-chatbook", Button)

        assert save_button.disabled is False
        save_button.press()
        await pilot.pause(0.1)

    assert len(handled_launches) == 1
    assert handled_launches[0].payload["target_id"] == "local:chatbook:77"


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
        assert "Pasted Text:" not in visible_plain
        assert "long composer qa" in visible_plain


@pytest.mark.asyncio
async def test_console_large_paste_collapses_visible_token_but_preserves_payload():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        command_input = composer.query_one("#console-command-input", Input)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "pasted composer qa " * 80
        expected_token = f"Pasted Text: {len(pasted_text)} Characters"

        console.on_paste(Paste(pasted_text))
        await pilot.pause(0.2)

        visible_plain = visible_draft.renderable.plain
        assert composer.draft_text() == pasted_text
        assert command_input.value == pasted_text
        assert composer.region.height <= 10
        assert visible_draft.region.height <= 4
        assert expected_token in visible_plain
        assert pasted_text not in visible_plain
        assert len(visible_plain) < len(pasted_text)
        assert isinstance(visible_draft.renderable, Text)
        _assert_single_style_span(
            visible_draft.renderable,
            style=ConsoleComposerBar.PASTE_TOKEN_STYLE,
            expected_text=expected_token,
        )


def test_console_paste_token_style_span_survives_literal_ellipsis_prefix():
    pasted_text = "x" * 51
    expected_token = f"Pasted Text: {len(pasted_text)} Characters"
    display_text = f"... {expected_token}"

    renderable = ConsoleComposerBar._draft_renderable(
        display_text,
        width=200,
        style_ranges=[(4, len(display_text), ConsoleComposerBar.PASTE_TOKEN_STYLE)],
    )

    assert renderable.plain == display_text
    _assert_single_style_span(
        renderable,
        style=ConsoleComposerBar.PASTE_TOKEN_STYLE,
        expected_text=expected_token,
    )


def test_console_paste_token_style_span_survives_crlf_before_token():
    pasted_text = "x" * 51
    expected_token = f"Pasted Text: {len(pasted_text)} Characters"
    display_text = f"before\r\n{expected_token}"
    token_start = display_text.index(expected_token)

    renderable = ConsoleComposerBar._draft_renderable(
        display_text,
        width=200,
        style_ranges=[(token_start, len(display_text), ConsoleComposerBar.PASTE_TOKEN_STYLE)],
    )

    assert renderable.plain == f"before\n{expected_token}"
    _assert_single_style_span(
        renderable,
        style=ConsoleComposerBar.PASTE_TOKEN_STYLE,
        expected_text=expected_token,
    )


def test_console_literal_segments_merge_during_typing_and_small_pastes():
    composer = ConsoleComposerBar()

    composer.insert_text("a")
    composer.insert_text("b")
    composer.insert_pasted_text("small paste")

    assert composer.draft_text() == "absmall paste"
    assert len(composer._segments) == 1
    assert composer._segments[0].collapse_state == "literal"


def test_console_composer_empty_placeholder_is_task_oriented():
    renderable = ConsoleComposerBar._draft_renderable("")

    assert renderable.plain == "Ask, command, or paste task..."


@pytest.mark.asyncio
async def test_console_paste_under_threshold_remains_literal():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "x" * ConsoleComposerBar.PASTE_COLLAPSE_THRESHOLD

        composer.insert_pasted_text(pasted_text)
        await pilot.pause(0.1)

        visible_plain = visible_draft.renderable.plain
        assert composer.draft_text() == pasted_text
        assert composer._display_draft_text() == pasted_text
        _assert_visible_literal_projection(visible_plain, pasted_text)
        assert "Pasted Text:" not in visible_plain


@pytest.mark.asyncio
async def test_console_paste_threshold_can_be_configured_from_app_config():
    app = _build_test_app()
    app.app_config["console"] = {
        "collapse_large_pastes": True,
        "paste_collapse_threshold": 80,
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)

        under_custom_threshold = "x" * 60
        composer.insert_pasted_text(under_custom_threshold)
        await pilot.pause(0.1)

        assert composer.draft_text() == under_custom_threshold
        assert composer._display_draft_text() == under_custom_threshold
        _assert_visible_literal_projection(
            visible_draft.renderable.plain,
            under_custom_threshold,
        )

        composer.clear_draft()
        over_custom_threshold = "y" * 81
        composer.insert_pasted_text(over_custom_threshold)
        await pilot.pause(0.1)

        assert composer.draft_text() == over_custom_threshold
        assert visible_draft.renderable.plain == "Pasted Text: 81 Characters"


@pytest.mark.parametrize("collapse_setting", [False, "false"])
@pytest.mark.asyncio
async def test_console_large_paste_collapse_can_be_disabled_from_config(collapse_setting):
    app = _build_test_app()
    app.app_config["console"] = {"collapse_large_pastes": collapse_setting}
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "literal paste chunk " * 10
        assert len(pasted_text) > ConsoleComposerBar.PASTE_COLLAPSE_THRESHOLD

        composer.insert_pasted_text(pasted_text)
        await pilot.pause(0.1)

        visible_plain = visible_draft.renderable.plain
        assert composer.draft_text() == pasted_text
        assert composer._display_draft_text() == pasted_text
        assert "Pasted Text:" not in visible_plain
        _assert_visible_literal_projection(visible_plain, pasted_text)


@pytest.mark.asyncio
async def test_console_clear_draft_keeps_canonical_payload_empty():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "clear me " * 20

        composer.insert_pasted_text(pasted_text)
        composer.clear_draft()
        await pilot.pause(0.1)

        assert composer.draft_text() == ""
        assert visible_draft.renderable.plain == ConsoleComposerBar.DRAFT_PLACEHOLDER
        assert pasted_text not in visible_draft.renderable.plain


@pytest.mark.asyncio
async def test_console_collapsed_paste_backspace_deletes_whole_chunk():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        prefix = "literal prefix "
        pasted_text = "delete after paste " * 10

        composer.insert_text(prefix)
        composer.insert_pasted_text(pasted_text)
        composer.delete_left()
        await pilot.pause(0.1)

        visible_plain = visible_draft.renderable.plain
        assert composer.draft_text() == prefix
        assert visible_plain == prefix
        assert pasted_text not in composer.draft_text()
        assert "Pasted Text:" not in visible_plain


@pytest.mark.asyncio
async def test_console_collapsed_paste_delete_key_deletes_whole_chunk():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        prefix = "literal prefix "
        pasted_text = "delete key paste " * 10

        composer.insert_text(prefix)
        composer.insert_pasted_text(pasted_text)
        composer.focus()
        await pilot.press("delete")
        await pilot.pause(0.1)

        visible_plain = visible_draft.renderable.plain
        assert composer.draft_text() == prefix
        assert visible_plain == prefix
        assert pasted_text not in composer.draft_text()
        assert "Pasted Text:" not in visible_plain


@pytest.mark.asyncio
async def test_console_collapsed_paste_real_click_enters_unfurl_prompt():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "click to confirm paste " * 10

        composer.insert_pasted_text(pasted_text)
        await pilot.click("#console-command-visible-text")
        await pilot.pause(0.1)

        assert visible_draft.renderable.plain == "Unfurl?"
        assert composer.draft_text() == pasted_text
        assert isinstance(visible_draft.renderable, Text)
        _assert_single_style_span(
            visible_draft.renderable,
            style=ConsoleComposerBar.PASTE_CONFIRM_STYLE,
            expected_text="Unfurl?",
        )


@pytest.mark.asyncio
async def test_console_collapsed_paste_composer_row_click_enters_unfurl_prompt():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "container row paste " * 8

        composer.insert_pasted_text(pasted_text)
        await pilot.click("#console-native-composer", offset=(13, 1))
        await pilot.pause(0.1)

        assert visible_draft.renderable.plain == "Unfurl?"
        assert composer.draft_text() == pasted_text


@pytest.mark.asyncio
async def test_console_collapsed_paste_textual_web_row_click_enters_unfurl_prompt():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(205, 62)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "textual web paste coordinate " * 6

        composer.insert_pasted_text(pasted_text)
        await pilot.pause(0.1)

        visible_region = composer._screen_region(visible_draft)
        assert composer.activate_visible_draft_screen_position(
            visible_region.x + 4,
            visible_region.y - 1,
        )

        assert visible_draft.renderable.plain == "Unfurl?"
        assert composer.draft_text() == pasted_text


@pytest.mark.asyncio
async def test_console_collapsed_paste_textual_web_bottom_boundary_click_enters_unfurl_prompt():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(205, 62)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "textual web bottom boundary paste " * 5

        composer.insert_pasted_text(pasted_text)
        await pilot.pause(0.1)

        visible_region = composer._screen_region(visible_draft)
        assert composer.activate_visible_draft_screen_position(
            visible_region.x + 4,
            visible_region.y + visible_draft.size.height,
        )

        assert visible_draft.renderable.plain == "Unfurl?"
        assert composer.draft_text() == pasted_text


@pytest.mark.asyncio
async def test_console_collapsed_paste_row_click_keeps_focus_on_composer():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(205, 62)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "focused composer visible paste " * 6

        composer.insert_pasted_text(pasted_text)
        await pilot.pause(0.1)

        visible_region = composer._screen_region(visible_draft)
        assert composer.activate_visible_draft_screen_position(
            visible_region.x + 4,
            visible_region.y + visible_draft.size.height,
        )

        assert visible_draft.renderable.plain == "Unfurl?"
        assert console.app.focused is composer
        assert composer.draft_text() == pasted_text


@pytest.mark.asyncio
async def test_console_collapsed_paste_second_click_unfurls_literal_text():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "literal unfurled paste " * 10

        composer.insert_pasted_text(pasted_text)
        await pilot.click("#console-command-visible-text")
        await pilot.click("#console-command-visible-text")
        await pilot.pause(0.1)

        visible_plain = visible_draft.renderable.plain
        assert "literal unfurled paste" in visible_plain
        assert "Pasted Text:" not in visible_plain
        assert "Unfurl?" not in visible_plain
        assert composer.draft_text() == pasted_text


@pytest.mark.asyncio
async def test_console_collapsed_paste_confirm_click_outside_token_resets_to_collapsed():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "literal forgiving paste " * 10

        composer.load_draft("prefix ")
        composer.insert_pasted_text(pasted_text)
        await pilot.click("#console-command-visible-text", offset=(8, 0))
        await pilot.pause(0.1)
        assert "Unfurl?" in visible_draft.renderable.plain

        await pilot.click("#console-command-visible-text", offset=(0, 0))
        await pilot.pause(0.1)

        visible_plain = visible_draft.renderable.plain
        assert "prefix " in visible_plain
        assert "Pasted Text:" in visible_plain
        assert "Unfurl?" not in visible_plain
        assert composer.draft_text() == f"prefix {pasted_text}"


@pytest.mark.asyncio
async def test_console_collapsed_paste_enter_on_focused_composer_matches_click_flow():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "keyboard unfurl paste " * 10

        composer.insert_pasted_text(pasted_text)
        composer.focus()
        await pilot.press("enter")
        await pilot.pause(0.1)

        assert console.app.focused is composer
        assert visible_draft.renderable.plain == "Unfurl?"
        assert composer.draft_text() == pasted_text

        await pilot.press("enter")
        await pilot.pause(0.1)

        visible_plain = visible_draft.renderable.plain
        assert "keyboard unfurl paste" in visible_plain
        assert "Pasted Text:" not in visible_plain
        assert "Unfurl?" not in visible_plain
        assert composer.draft_text() == pasted_text


@pytest.mark.asyncio
async def test_console_collapsed_paste_click_targets_token_after_literal_newline():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "newline preceded paste " * 10

        composer.load_draft("prefix\n")
        composer.insert_pasted_text(pasted_text)
        await pilot.pause(0.1)

        await pilot.click("#console-command-visible-text", offset=(0, 1))
        await pilot.pause(0.1)

        assert visible_draft.renderable.plain == "prefix\nUnfurl?"
        assert composer.draft_text() == f"prefix\n{pasted_text}"


@pytest.mark.asyncio
async def test_console_collapsed_paste_click_targets_second_chunk_independently():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        first_paste = "first large paste " * 10
        second_paste = "second large paste " * 10
        first_token = f"Pasted Text: {len(first_paste)} Characters"
        second_token = f"Pasted Text: {len(second_paste)} Characters"

        composer.insert_pasted_text(first_paste)
        composer.insert_pasted_text(second_paste)
        await pilot.pause(0.1)
        assert visible_draft.renderable.plain == f"{first_token}{second_token}"

        await pilot.click(
            "#console-command-visible-text",
            offset=(len(first_token) + 2, 0),
        )
        await pilot.pause(0.1)
        assert visible_draft.renderable.plain == f"{first_token}Unfurl?"

        await pilot.click(
            "#console-command-visible-text",
            offset=(len(first_token) + 2, 0),
        )
        await pilot.pause(0.1)

        visible_plain = visible_draft.renderable.plain
        assert first_token in visible_plain
        assert "second large paste" in visible_plain
        assert "first large paste" not in visible_plain
        assert "Unfurl?" not in visible_plain
        assert composer.draft_text() == f"{first_paste}{second_paste}"


@pytest.mark.asyncio
async def test_console_collapsed_paste_typing_resets_pending_unfurl_prompt():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "typing resets pending paste " * 10
        expected_token = f"Pasted Text: {len(pasted_text)} Characters"

        composer.insert_pasted_text(pasted_text)
        await pilot.click("#console-command-visible-text", offset=(1, 0))
        await pilot.pause(0.1)
        assert visible_draft.renderable.plain == "Unfurl?"

        await pilot.press("x")
        await pilot.pause(0.1)
        assert visible_draft.renderable.plain == f"{expected_token}x"
        assert composer.draft_text() == f"{pasted_text}x"

        await pilot.click("#console-command-visible-text", offset=(1, 0))
        await pilot.pause(0.1)
        assert visible_draft.renderable.plain == "Unfurl?x"
        assert composer.draft_text() == f"{pasted_text}x"


@pytest.mark.asyncio
async def test_console_collapsed_paste_click_targets_token_after_visible_clipping():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        prefix = "preceding wrapped composer text " * 40
        pasted_text = "visible clipped paste " * 10
        expected_token = f"Pasted Text: {len(pasted_text)} Characters"

        composer.insert_text(prefix)
        composer.insert_pasted_text(pasted_text)
        await pilot.pause(0.1)

        visible_plain = visible_draft.renderable.plain
        visible_lines = visible_plain.splitlines()
        display_text = composer._display_draft_text()
        token_start = display_text.index(expected_token)
        token_end = token_start + len(expected_token)
        visible_slices = composer._visible_draft_line_slices(
            display_text,
            composer._draft_render_width(),
        )
        assert len(composer._wrap_draft_lines(composer._display_draft_text(), composer._draft_render_width())) > (
            ConsoleComposerBar.MAX_DRAFT_ROWS
        )
        assert visible_lines[0].startswith("...")
        assert expected_token in visible_plain.replace("\n", "")
        token_row, token_slice = next(
            (index, line_slice)
            for index, line_slice in enumerate(visible_slices)
            if max(line_slice.start, token_start) < min(line_slice.end, token_end)
        )
        click_source_index = max(token_start, token_slice.start)
        if click_source_index + 1 < min(token_end, token_slice.end):
            click_source_index += 1
        token_column = (
            click_source_index
            - token_slice.start
            + token_slice.synthetic_prefix_columns
        )

        await pilot.click(
            "#console-command-visible-text",
            offset=(token_column, token_row),
        )
        await pilot.pause(0.1)

        assert "Unfurl?" in visible_draft.renderable.plain
        assert expected_token not in visible_draft.renderable.plain
        assert composer.draft_text() == f"{prefix}{pasted_text}"


@pytest.mark.asyncio
async def test_console_collapsed_paste_click_elsewhere_resets_unfurl_prompt():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "reset pending unfurl " * 10
        expected_token = f"Pasted Text: {len(pasted_text)} Characters"

        composer.insert_pasted_text(pasted_text)
        await pilot.click("#console-command-visible-text")
        await pilot.pause(0.1)
        assert visible_draft.renderable.plain == "Unfurl?"

        await pilot.click("#console-workspace-grid")
        await pilot.pause(0.1)

        visible_plain = visible_draft.renderable.plain
        assert expected_token in visible_plain
        assert "Unfurl?" not in visible_plain
        assert composer.draft_text() == pasted_text


@pytest.mark.asyncio
async def test_console_stale_suppressed_click_does_not_swallow_unrelated_click():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "stale suppressed click reset paste " * 6
        expected_token = f"Pasted Text: {len(pasted_text)} Characters"

        composer.insert_pasted_text(pasted_text)
        await pilot.click("#console-command-visible-text")
        await pilot.pause(0.1)
        assert visible_draft.renderable.plain == "Unfurl?"

        composer.suppress_next_draft_click()
        await pilot.click("#console-workspace-grid")
        await pilot.pause(0.1)

        visible_plain = visible_draft.renderable.plain
        assert expected_token in visible_plain
        assert "Unfurl?" not in visible_plain
        assert composer.draft_text() == pasted_text


@pytest.mark.asyncio
async def test_console_normal_typing_remains_literal_over_paste_threshold():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        typed_text = "normaltypedcomposertext" * 4

        await pilot.press(*typed_text)
        await pilot.pause(0.1)

        visible_plain = visible_draft.renderable.plain
        assert len(typed_text) > ConsoleComposerBar.PASTE_COLLAPSE_THRESHOLD
        assert composer.draft_text() == typed_text
        assert "Pasted Text:" not in visible_plain
        assert "normaltypedcomposertext" in visible_plain
        assert isinstance(visible_draft.renderable, Text)
        assert not visible_draft.renderable.spans


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
async def test_console_native_composer_select_all_shortcut_preserves_draft_and_copies():
    app = _build_test_app()
    host = ConsoleHarness(app)
    copied: list[str] = []
    app.copy_to_clipboard = copied.append

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)

        composer.load_draft("replace this text")
        composer.focus()
        await pilot.press("ctrl+a")
        await pilot.pause(0.1)

        assert composer.draft_text() == "replace this text"
        assert composer.has_full_draft_selection()
        assert visible_draft.renderable.plain == "replace this text"

        await pilot.press("ctrl+c")
        await pilot.pause(0.1)

        assert copied == ["replace this text"]
        assert composer.draft_text() == "replace this text"

        await pilot.press("n", "e", "w")
        await pilot.pause(0.1)

        assert composer.draft_text() == "new"
        assert visible_draft.renderable.plain == "new"
        assert not composer.has_full_draft_selection()


@pytest.mark.asyncio
async def test_console_native_composer_clicking_visible_draft_captures_typing():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)

        await pilot.click("#console-command-visible-text")
        await pilot.press("a", "b", "c")
        await pilot.pause(0.1)

        assert composer.draft_text() == "abc"
        assert "abc" in visible_draft.renderable.plain


@pytest.mark.asyncio
async def test_console_native_composer_click_focuses_composer_not_visible_static():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)

        await pilot.click("#console-command-visible-text")
        await pilot.pause(0.1)

        assert console.app.focused is composer
        assert console.app.focused is not visible_draft


@pytest.mark.asyncio
async def test_console_native_composer_does_not_capture_typing_from_select_focus(monkeypatch):
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-compact-model-bar")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        compact_bar = console.query_one("#console-compact-model-bar")
        provider_select = compact_bar.query_one("#compact-api-provider", Select)

        monkeypatch.setattr(type(console.app), "focused", property(lambda _app: provider_select))
        await pilot.press("x")
        await pilot.pause(0.1)

        assert composer.draft_text() == ""


@pytest.mark.asyncio
async def test_console_native_composer_does_not_capture_paste_from_select_focus(monkeypatch):
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-compact-model-bar")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        compact_bar = console.query_one("#console-compact-model-bar")
        provider_select = compact_bar.query_one("#compact-api-provider", Select)

        monkeypatch.setattr(type(console.app), "focused", property(lambda _app: provider_select))
        console.on_paste(Paste("paste should stay with focused control"))
        await pilot.pause(0.1)

        assert composer.draft_text() == ""


@pytest.mark.asyncio
async def test_console_send_without_ready_runtime_keeps_setup_block_tooltip_only(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _build_test_app()
    _configure_openai_missing_key_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(provider="openai", model="gpt-4.1-2025-04-14"),
        )
        await console._sync_native_console_chat_ui()

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello console")
        send_button = console.query_one("#console-send-message", Button)
        await console.handle_console_send_message(Button.Pressed(send_button))
        await pilot.pause(0.2)

        text = _visible_text(console)
        assert "Console send blocked" not in text
        assert "missing API key" in text
        assert "Internal Error" not in text
        assert "Missing UI elements" not in text
        assert composer.draft_text() == "hello console"


@pytest.mark.asyncio
async def test_console_enter_sends_native_composer_draft(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _build_test_app()
    _configure_openai_missing_key_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(provider="openai", model="gpt-4.1-2025-04-14"),
        )
        await console._sync_native_console_chat_ui()

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        await pilot.press("h", "e", "l", "l", "o")
        await pilot.press("enter")
        await pilot.pause(0.2)

        text = _visible_text(console)
        send_button = console.query_one("#console-send-message", Button)
        assert "Console send blocked" not in text
        assert send_button.disabled is False
        assert send_button.tooltip == (
            "Add API key in Settings > Providers & Models before sending."
        )
        assert "missing API key" in text
        assert "Internal Error" not in text
        assert "Missing UI elements" not in text
        assert composer.draft_text() == "hello"


@pytest.mark.asyncio
async def test_console_empty_transcript_promotes_start_here_and_provider_recovery():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1-2025-04-14",
        },
        "api_settings": {"openai": {}},
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = "gpt-4.1-2025-04-14"
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-provider-blocker")
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        start_here = console.query_one("#console-start-here", Static)
        action_hints = console.query_one("#console-action-hints", Static)
        provider_strip = console.query_one("#console-provider-recovery-strip")
        recovery = console.query_one("#workbench-recovery-callout")
        recovery_action = console.query_one("#workbench-recovery-action", Button)
        transcript = console.query_one("#console-native-transcript")
        assert provider_strip.styles.display == "none"
        assert recovery.display is True
        assert recovery.region.y < transcript.region.y
        assert recovery_action.display is True
        assert str(recovery_action.label) == CONSOLE_PROVIDER_CONFIGURE_API_KEY_LABEL
        assert start_here.styles.display == "none"
        assert action_hints.styles.display == "none"

        text = _visible_text(console)
        for expected in (
            "Provider setup needed",
            "OpenAI missing API key",
            "Impact: Send is blocked until setup is finished.",
            CONSOLE_PROVIDER_CONFIGURE_API_KEY_LABEL,
            "Start Console",
            "Add an API key to enable Send. Then type in Composer or attach context.",
            "Attach context",
            "Run Library RAG",
        ):
            assert expected in text
        for redundant_copy in (
            "Run command",
            "Provider setup required before sending.",
            "Finish provider setup to start chatting.",
            "Enter send",
            "Ctrl+P commands",
            "No messages yet.",
            "Setup required: finish provider setup.",
            "No messages yet. Send a prompt or attach context.",
            "1. Finish provider setup",
            "2. Attach Library, runs, Artifacts, or RAG",
            "3. Type a message or command in Composer",
        ):
            assert redundant_copy not in text
        assert "Provider: OpenAI is not ready" not in text
        assert "Provider setup is shown in the recovery strip above." not in text
        blocker = console.query_one("#console-provider-blocker", Static)
        assert blocker.styles.display == "none"
        recovery_text = getattr(recovery.renderable, "plain", str(recovery.renderable))
        assert recovery_text.startswith("Console setup blocked")
        assert "Provider setup needed: OpenAI missing API key" in recovery_text
        assert "Impact: Send is blocked until setup is finished." in recovery_text
        assert console.query_one("#console-inspector-rail-handle").display is True
        assert console.query_one("#console-right-rail").display is False
        assert text.lower().count("missing api key") == 1


@pytest.mark.asyncio
async def test_console_empty_ready_transcript_shows_first_run_guidance():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        text = _visible_text(console)
        assert "Start Console" in text
        assert "Type in Composer, attach sources, or run Library RAG before sending." in text
        assert "Choose model" in text
        assert "Attach context" in text
        assert "Run Library RAG" in text

        await pilot.press("h")
        await pilot.pause(0.1)

        text = _visible_text(console)
        assert "No messages yet. Send a prompt or attach context." not in text


@pytest.mark.asyncio
async def test_console_provider_blocker_exposes_open_settings_action(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1-2025-04-14",
        },
        "api_settings": {"openai": {"api_key": ""}},
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = "gpt-4.1-2025-04-14"
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#workbench-recovery-callout")
        await _wait_for_selector(console, pilot, "#workbench-recovery-action")

        strip = console.query_one("#console-provider-recovery-strip")
        blocker = console.query_one("#console-provider-blocker", Static)
        legacy_button = console.query_one("#console-open-provider-settings", Button)
        recovery = console.query_one("#workbench-recovery-callout")
        button = console.query_one("#workbench-recovery-action", Button)
        assert strip.styles.display == "none"
        assert blocker.styles.display == "none"
        assert legacy_button.styles.display == "none"
        assert button.display is True
        assert button.disabled is False
        assert button.region.height == 1
        assert button.region.width >= len(CONSOLE_PROVIDER_CONFIGURE_API_KEY_LABEL)
        assert str(button.label) == CONSOLE_PROVIDER_CONFIGURE_API_KEY_LABEL
        assert button.has_class("is-primary")
        assert recovery.region.y < button.region.y
        recovery_text = getattr(recovery.renderable, "plain", str(recovery.renderable))
        assert recovery_text.startswith("Console setup blocked")
        assert "Provider setup needed: OpenAI missing API key" in recovery_text
        assert "Impact: Send is blocked until setup is finished." in recovery_text
        text = _visible_text(console)
        assert CONSOLE_PROVIDER_CONFIGURE_API_KEY_LABEL in text
        assert console.query_one("#console-inspector-rail-handle").display is True
        assert console.query_one("#console-right-rail").display is False
        assert text.lower().count("missing api key") == 1


@pytest.mark.asyncio
async def test_console_provider_settings_action_posts_navigation_message(monkeypatch):
    app = _build_test_app()
    console = ChatScreen(app)
    event = _PressedEvent()
    posted_messages: list[object] = []
    monkeypatch.setattr(console, "post_message", posted_messages.append)

    await console.handle_console_open_provider_settings(event)  # type: ignore[arg-type]

    assert event.stopped is True
    assert [
        message.screen_name
        for message in posted_messages
        if isinstance(message, NavigateToScreen)
    ] == ["settings"]


@pytest.mark.asyncio
async def test_console_provider_settings_action_hidden_when_provider_ready(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.pause(0.1)

        buttons = [
            button
            for button in console.query("#console-open-provider-settings")
            if button.display
        ]
        assert buttons == []


@pytest.mark.asyncio
async def test_console_choose_model_state_hides_redundant_recovery_strip(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.app_config["chat_defaults"]["model"] = ""
    app.app_config["api_settings"]["llama_cpp"].pop("model", None)
    app.chat_api_model_value = ""
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#workbench-recovery-callout")
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        strip = console.query_one("#console-provider-recovery-strip")
        recovery = console.query_one("#workbench-recovery-callout")
        recovery_action = console.query_one("#workbench-recovery-action", Button)
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        await pilot.pause()
        blocker = console.query_one("#console-provider-blocker", Static)

        assert strip.styles.display == "none"
        assert blocker.styles.display == "none"
        assert recovery.display is True
        assert str(recovery_action.label) == "Choose model"
        assert "Provider setup needed: choose a model" in getattr(
            recovery.renderable, "plain", str(recovery.renderable)
        )
        assert "Provider setup needed: choose a model" not in _visible_text(
            console.query_one("#console-native-transcript")
        )
        assert "Choose model" in _visible_text(console)
        assert "Choose a model in Console Settings to start chatting." not in _visible_text(console)
        assert "No messages yet." not in _visible_text(console)
        assert "Impact: Send is blocked" in _visible_text(console)
        assert "Setup required: Choose model before sending." not in _visible_text(console)

        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(provider="llama_cpp", model="local-model"),
        )
        console._sync_console_control_bar()
        await pilot.pause()

        assert blocker.styles.display == "none"
        assert str(blocker.renderable) == ""
        assert strip.styles.display == "none"
        assert "Choose a model in Console Settings to start chatting." not in _visible_text(console)
        assert "Setup required: Choose model before sending." not in _visible_text(console)

        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(provider="llama_cpp", model=None),
        )
        console._sync_console_control_bar()
        await pilot.pause()

        assert blocker.styles.display == "none"
        assert strip.styles.display == "none"
        assert recovery.display is True
        assert str(recovery_action.label) == "Choose model"
        assert "Provider setup needed: choose a model" in getattr(
            recovery.renderable, "plain", str(recovery.renderable)
        )
        assert "Provider setup needed: choose a model" not in _visible_text(
            console.query_one("#console-native-transcript")
        )
        assert "Choose a model in Console Settings to start chatting." not in _visible_text(console)
        assert "No messages yet." not in _visible_text(console)


@pytest.mark.asyncio
async def test_console_empty_transcript_stays_neutral_when_setup_blocked(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.app_config["chat_defaults"]["model"] = ""
    app.app_config["api_settings"]["llama_cpp"].pop("model", None)
    app.chat_api_model_value = ""
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        transcript = console.query_one("#console-native-transcript")
        text = _visible_text(transcript)

        assert "Start Console" in text
        assert "Choose a model to enable Send. Then type in Composer or attach context." in text
        assert "Choose model" in text
        assert "Attach context" in text
        assert "Run Library RAG" in text
        assert "Choose a model in Console Settings to start chatting." not in text
        assert "Ready. Ask a question" not in text


@pytest.mark.asyncio
async def test_console_inline_guidance_does_not_reserve_transcript_space():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-start-here")
        await _wait_for_selector(console, pilot, "#console-action-hints")
        await _wait_for_selector(console, pilot, "#console-transcript-title")
        start_here = console.query_one("#console-start-here", Static)
        action_hints = console.query_one("#console-action-hints", Static)
        transcript_title = console.query_one("#console-transcript-title", Static)

        assert start_here.styles.display == "none"
        assert action_hints.styles.display == "none"
        assert str(start_here.styles.height) == "0"
        start_copy = getattr(start_here.render(), "plain", str(start_here.render()))
        title_copy = getattr(transcript_title.render(), "plain", str(transcript_title.render()))
        assert start_copy == ""
        assert title_copy == "Transcript / Event Stream"

        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.append_message(session.id, role=ConsoleMessageRole.USER, content="Hello Console")
        await console._sync_native_console_chat_ui()
        console._sync_console_control_bar()
        await pilot.pause()

        assert start_here.styles.display == "none"
        assert action_hints.styles.display == "none"
        title_copy = getattr(transcript_title.render(), "plain", str(transcript_title.render()))
        assert title_copy == "Transcript / Event Stream"


@pytest.mark.asyncio
async def test_console_inline_guidance_disappears_after_user_starts_typing():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_selector(console, pilot, "#console-transcript-title")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        transcript_title = console.query_one("#console-transcript-title", Static)

        title_copy = getattr(transcript_title.render(), "plain", str(transcript_title.render()))
        assert title_copy == "Transcript / Event Stream"

        await pilot.press("h")
        await pilot.pause(0.1)

        title_copy = getattr(transcript_title.render(), "plain", str(transcript_title.render()))
        assert composer.draft_text() == "h"
        assert title_copy == "Transcript / Event Stream"

        composer.clear_draft()
        console._sync_console_transcript_guidance()
        await pilot.pause(0.1)

        title_copy = getattr(transcript_title.render(), "plain", str(transcript_title.render()))
        assert composer.draft_text() == ""
        assert title_copy == "Transcript / Event Stream"


@pytest.mark.asyncio
async def test_console_transcript_header_sits_at_top_of_center_panel():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-start-here")
        await _wait_for_selector(console, pilot, "#console-transcript-title")

        start_here = console.query_one("#console-start-here", Static)
        transcript_region = console.query_one("#console-transcript-region")
        transcript_title = console.query_one("#console-transcript-title", Static)
        tab_strip = console.query_one("#console-native-tab-strip")
        transcript = console.query_one("#console-native-transcript")

        assert start_here.styles.display == "none"
        assert start_here.region.height == 0
        assert transcript_region.styles.border.top[0] in {"", "none"}
        assert transcript_title.region.y == transcript_region.region.y
        assert tab_strip.region.y == transcript_title.region.y + transcript_title.region.height
        assert transcript.region.y == tab_strip.region.y + tab_strip.region.height


@pytest.mark.asyncio
async def test_console_transcript_header_and_tabs_have_distinct_visual_roles():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1-2025-04-14",
        },
        "api_settings": {"openai": {"api_key": "DUMMY_TEST_KEY"}},
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = "gpt-4.1-2025-04-14"
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-transcript-title")
        await _wait_for_selector(console, pilot, "#console-native-tab-strip")
        await _wait_for_selector(console, pilot, ".console-session-tab-active")

        transcript_title = console.query_one("#console-transcript-title", Static)
        tab_strip = console.query_one("#console-native-tab-strip")
        active_tab = console.query_one(".console-session-tab-active", Button)
        transcript = console.query_one("#console-native-transcript")

        assert transcript_title.has_class("console-transcript-title")
        assert tab_strip.has_class("console-session-tab-strip")
        assert transcript_title.region.height == 1
        assert tab_strip.region.height == 1
        assert transcript_title.styles.height.value == 1
        assert tab_strip.styles.height.value == 1
        assert tab_strip.region.y == transcript_title.region.y + 1
        assert transcript.region.y == tab_strip.region.y + 1
        assert transcript_title.styles.color != active_tab.styles.color
        assert active_tab.styles.background != tab_strip.styles.background
        assert active_tab.has_class("console-session-tab-active")


@pytest.mark.asyncio
async def test_console_native_transcript_is_visible_transcript_surface():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        transcript = console.query_one("#console-native-transcript")

        assert transcript.region.width > 0
        assert transcript.region.height > 0
        assert transcript.styles.display != "none"
        text = _visible_text(console)
        assert "Start Console" in text
        assert "Type in Composer, attach sources, or run Library RAG before sending." in text
        assert "Choose model" in text
        assert "Attach context" in text
        assert "Run Library RAG" in text
        assert "No messages yet. Send a prompt or attach context." not in text


@pytest.mark.asyncio
async def test_console_empty_transcript_uses_compact_ready_state():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        tab_strip = console.query_one("#console-native-tab-strip")
        transcript = console.query_one("#console-native-transcript")
        empty_rows = list(transcript.query("#console-transcript-empty-state"))

        assert len(empty_rows) == 1
        empty_panel = empty_rows[0]
        assert _visible_text(empty_panel) == (
            "Start Console Type in Composer, attach sources, or run Library RAG before "
            "sending. Choose model Attach context Run Library RAG"
        )
        assert "No messages yet. Send a prompt or attach context." not in _visible_text(
            empty_panel
        )
        assert empty_panel.region.y == tab_strip.region.y + tab_strip.region.height


@pytest.mark.asyncio
async def test_console_session_tab_strip_uses_readable_new_tab_control_with_tooltips():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-new-chat-tab")
        await _wait_for_selector(console, pilot, ".console-session-close-button")

        new_tab = console.query_one("#console-new-chat-tab", Button)
        close_buttons = list(console.query(".console-session-close-button"))

        assert str(new_tab.label) == "New tab"
        assert new_tab.tooltip == "New Console tab"
        assert new_tab.region.width >= 12
        assert close_buttons
        for close_button in close_buttons:
            assert str(close_button.label) == "x"
            assert close_button.tooltip == "Close Console tab"
            assert close_button.region.width >= 3


@pytest.mark.asyncio
async def test_console_gate15_does_not_mount_full_legacy_chat_window_chrome():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        assert len(console.query("#console-chat-tabs")) == 0
        assert [
            button.id
            for button in console.query(Button)
            if button.id and button.id.startswith(("chat-tab-", "close-tab-"))
        ] == []


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
        await _open_console_inspector(console, pilot)
        await _wait_for_selector(console, pilot, "#console-live-work-source-readiness")

        inspector = console.query_one("#console-run-inspector")
        inspector_state = console.query_one("#console-run-inspector-state")
        body = console.query_one("#console-inspector-rail-body")
        source_readiness = console.query_one("#console-live-work-source-readiness")

        assert inspector.parent is body
        assert source_readiness.parent is body
        assert source_readiness.region.y >= inspector.region.y
        assert source_readiness.region.y <= body.region.y + body.region.height
        assert source_readiness.region.height <= 18


@pytest.mark.asyncio
async def test_console_inspector_source_readiness_rows_fit_without_tooltip_overlay():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(196, 64)) as pilot:
        console = host.screen_stack[-1]
        await _open_console_inspector(console, pilot)
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
            lines = _wrapped_plain_lines(row.render(), row.region.width)
            assert lines
            assert len(lines) <= row.region.height
            assert all(len(line) <= row.region.width for line in lines)


@pytest.mark.asyncio
async def test_console_empty_inspector_hides_disabled_actions_until_actionable():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(196, 64)) as pilot:
        console = host.screen_stack[-1]
        await _open_console_inspector(console, pilot)
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
async def test_console_run_inspector_orders_state_source_tools_and_approvals():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(196, 64)) as pilot:
        console = host.screen_stack[-1]
        await _open_console_inspector(console, pilot)
        await _wait_for_selector(console, pilot, "#console-inspector-run-heading")
        await _wait_for_selector(console, pilot, "#console-inspector-source-readiness-heading")
        await _wait_for_selector(console, pilot, "#console-inspector-tools-heading")
        await _wait_for_selector(console, pilot, "#console-inspector-approvals-heading")
        await _wait_for_selector(console, pilot, "#console-inspector-artifacts-heading")

        assert (
            getattr(
                console.query_one("#console-inspector-run-status-summary", Static).render(),
                "plain",
                "",
            )
            == "Status: Ready"
        )
        assert not list(console.query("#console-run-inspector-title"))
        assert (
            getattr(
                console.query_one("#console-inspector-run-heading", Static).render(),
                "plain",
                "",
            )
            == "Run"
        )
        assert (
            getattr(
                console.query_one("#console-inspector-approvals-heading", Static).render(),
                "plain",
                "",
            )
            == "Approvals"
        )
        assert (
            getattr(
                console.query_one("#console-inspector-source-readiness-heading", Static).render(),
                "plain",
                "",
            )
            == "Source Readiness"
        )
        assert (
            getattr(
                console.query_one("#console-inspector-tools-heading", Static).render(),
                "plain",
                "",
            )
            == "Tools"
        )
        assert (
            console.query_one("#console-inspector-run-heading").region.y
            < console.query_one("#console-inspector-source-readiness-heading").region.y
            < console.query_one("#console-inspector-tools-heading").region.y
            < console.query_one("#console-inspector-approvals-heading").region.y
            < console.query_one("#console-inspector-artifacts-heading").region.y
        )


@pytest.mark.asyncio
async def test_console_workbench_weights_transcript_as_primary_region():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-grid")

        staged = console.query_one("#console-left-rail")
        main = console.query_one("#console-main-column")
        right_handle = console.query_one("#console-inspector-rail-handle")
        right_rail = console.query_one("#console-right-rail")
        transcript = console.query_one("#console-transcript-region")

        assert main.region.width > staged.region.width
        assert main.region.width > right_handle.region.width
        assert right_handle.display is True
        assert right_handle.region.width > 0
        assert right_rail.display is False
        assert right_rail.region.width == 0
        assert staged.region.width >= 36
        assert transcript.region.width == main.region.width

        await _open_console_inspector(console, pilot)

        assert right_rail.display is True
        assert right_rail.region.width >= 40
        assert right_handle.display is False
        assert console.query_one("#console-run-inspector-state").region.width > 0


@pytest.mark.asyncio
async def test_console_left_rail_sections_use_available_space():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")

        left_rail = console.query_one("#console-left-rail")
        body = console.query_one("#console-left-rail-body")
        header = console.query_one(".console-rail-header")
        staged_context = console.query_one("#console-staged-context-tray")
        workspace_context = console.query_one("#console-workspace-context")

        assert body.parent is left_rail
        assert header.region.height == 1
        assert body.region.y >= header.region.y + header.region.height
        assert body.region.height <= left_rail.region.height - header.region.height
        assert staged_context.parent is body
        assert workspace_context.parent is body
        assert staged_context.region.y < workspace_context.region.y
        assert workspace_context.region.height > staged_context.region.height
        body_content_width = body.scrollable_content_region.width
        assert 0 <= body.region.width - body_content_width <= 2
        assert staged_context.region.width == body_content_width
        assert workspace_context.region.width == body_content_width


@pytest.mark.asyncio
async def test_console_empty_regions_do_not_stack_nested_terminal_frames():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-grid")

        workbench_border = console.query_one("#console-workspace-grid").styles.border
        assert workbench_border.top[0] == "solid"
        assert workbench_border.right[0] == "solid"
        assert workbench_border.bottom[0] == "solid"
        assert workbench_border.left[0] == "solid"

        transcript_border = console.query_one("#console-transcript-region").styles.border
        assert transcript_border.top[0] in {"", "none"}
        assert transcript_border.right[0] == "solid"
        assert transcript_border.bottom[0] == "solid"
        assert transcript_border.left[0] == "solid"

        staged_context_border = console.query_one("#console-staged-context-tray").styles.border
        assert staged_context_border.top[0] in {"", "none"}
        assert staged_context_border.right[0] in {"", "none"}
        assert staged_context_border.bottom[0] in {"", "none"}
        assert staged_context_border.left[0] in {"", "none"}

        workspace_context_border = console.query_one("#console-workspace-context").styles.border
        assert workspace_context_border.top[0] in {"", "none"}
        assert workspace_context_border.right[0] in {"", "none"}
        assert workspace_context_border.bottom[0] in {"", "none"}
        assert workspace_context_border.left[0] in {"", "none"}

        composer_border = console.query_one("#console-native-composer").styles.border
        assert composer_border.top[0] == "solid"
        assert composer_border.right[0] == "solid"
        assert composer_border.bottom[0] == "solid"
        assert composer_border.left[0] == "solid"


@pytest.mark.asyncio
async def test_console_workbench_panes_have_visible_terminal_frames():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-grid")

        for selector in (
            "#console-workspace-grid",
            "#console-left-rail",
            "#console-native-composer",
        ):
            border = console.query_one(selector).styles.border
            assert border.top[0] == "solid", f"{selector} missing top frame"
            assert border.right[0] == "solid", f"{selector} missing right frame"
            assert border.bottom[0] == "solid", f"{selector} missing bottom frame"
            assert border.left[0] == "solid", f"{selector} missing left frame"

        right_handle = console.query_one("#console-inspector-rail-handle")
        assert right_handle.has_class("console-frame-quiet")
        handle_border = right_handle.styles.border
        assert handle_border.top[0] in {"", "none"}
        assert handle_border.right[0] in {"", "none"}
        assert handle_border.bottom[0] in {"", "none"}
        assert handle_border.left[0] in {"", "none"}

        transcript_border = console.query_one("#console-transcript-region").styles.border
        assert transcript_border.top[0] in {"", "none"}
        assert transcript_border.right[0] == "solid"
        assert transcript_border.bottom[0] == "solid"
        assert transcript_border.left[0] == "solid"

        for selector in (
            "#console-staged-context-tray",
            "#console-workspace-context",
        ):
            border = console.query_one(selector).styles.border
            assert border.top[0] in {"", "none"}, f"{selector} has a heavy top frame"
            assert border.right[0] in {"", "none"}, f"{selector} has a heavy right frame"
            assert border.bottom[0] in {"", "none"}, f"{selector} has a heavy bottom frame"
            assert border.left[0] in {"", "none"}, f"{selector} has a heavy left frame"

        await _open_console_inspector(console, pilot)

        right_rail = console.query_one("#console-right-rail")
        inspector_state = console.query_one("#console-run-inspector-state")
        border = right_rail.styles.border
        assert border.top[0] == "solid", "#console-right-rail missing top frame"
        assert border.right[0] == "solid", "#console-right-rail missing right frame"
        assert border.bottom[0] == "solid", "#console-right-rail missing bottom frame"
        assert border.left[0] == "solid", "#console-right-rail missing left frame"
        assert inspector_state.region.width > 0
        assert right_rail.region.x <= inspector_state.region.x
        assert (
            inspector_state.region.x + inspector_state.region.width
            <= right_rail.region.x + right_rail.region.width
        )


@pytest.mark.asyncio
async def test_console_empty_staged_context_action_fits_tray():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-staged-context-attach")

        summary = console.query_one("#console-staged-context-summary")
        attach_button = console.query_one("#console-staged-context-attach", Button)
        summary_plain = getattr(summary.render(), "plain", str(summary.render()))

        assert summary_plain == "No sources attached."
        assert str(attach_button.label) == "Attach"
        visible_text_width = max(0, summary.region.width - 2)
        assert all(len(line) <= visible_text_width for line in summary_plain.splitlines())
        assert len(str(attach_button.label)) <= max(0, attach_button.region.width - 2)


@pytest.mark.asyncio
async def test_console_empty_staged_context_exposes_attach_action():
    app = _build_test_app()
    host = ConsoleHarness(app)

    class FakeSession:
        def __init__(self) -> None:
            self.attach_events = []

        def handle_attach_button(self, event) -> None:
            self.attach_events.append(event)

    fake_session = FakeSession()

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        console._get_active_chat_session = lambda: fake_session
        await _wait_for_selector(console, pilot, "#console-staged-context-attach")

        summary = console.query_one("#console-staged-context-summary", Static)
        attach_button = console.query_one("#console-staged-context-attach", Button)
        tray_text = _visible_text(console.query_one("#console-staged-context-tray"))

        assert getattr(summary.render(), "plain", str(summary.render())) == "No sources attached."
        assert str(attach_button.label) == "Attach"
        assert attach_button.compact is True
        assert "Attach sources." not in tray_text

        await pilot.click("#console-staged-context-attach")
        await pilot.pause()

        assert len(fake_session.attach_events) == 1


@pytest.mark.asyncio
async def test_console_staged_context_attach_uses_semantic_empty_state():
    state = ConsoleStagedContextState(
        heading="Staged Context",
        summary="Nothing attached yet.",
        is_empty=True,
    )
    app = StagedContextHarness(state)

    async with app.run_test(size=(60, 10)) as pilot:
        await _wait_for_selector(app.screen, pilot, "#console-staged-context-attach")

        assert app.query_one("#console-staged-context-summary", Static).render().plain == (
            "Nothing attached yet."
        )
        assert str(app.query_one("#console-staged-context-attach", Button).label) == "Attach"


@pytest.mark.asyncio
async def test_console_non_empty_staged_context_keeps_room_for_source_details():
    app = _build_test_app()
    app.pending_console_launch = {
        "source": "Library Search/RAG",
        "title": "Incident Review",
        "status": "ready",
        "recovery": "Review citations before sending.",
        "payload": {
            "source_id": "note-42",
            "chunk_id": "chunk-7",
            "runtime_backend": "local-fts",
        },
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-staged-context-row-0")

        staged_context = console.query_one("#console-staged-context-tray")
        max_height = getattr(
            staged_context.styles.max_height,
            "value",
            staged_context.styles.max_height,
        )
        assert max_height >= 10
        assert "source_id: note-42" in _visible_text(staged_context)
        assert "Review citations before sending." in _visible_text(staged_context)


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
    _configure_native_ready_console(app, model="reactive-model")
    screen = ChatScreen(app)

    control_state = screen._build_console_control_state(None)
    inspector_state = screen._build_console_inspector_state(None)
    rows_by_label = {row.label: row for row in inspector_state.rows}

    assert control_state.provider_label == "Provider: llama_cpp"
    assert control_state.model_label == "Model: reactive-model"
    assert rows_by_label["Provider"].text == "Provider: ready"


def test_console_prefers_configured_provider_when_app_reactive_is_stale_default():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {
            "provider": "llama_cpp",
            "model": "local-model",
        },
        "api_settings": {
            "llama_cpp": {
                "api_url": "http://127.0.0.1:9099",
            },
        },
    }
    app.chat_api_provider_value = "OpenAI"
    screen = ChatScreen(app)

    selection = screen._build_console_provider_selection()

    assert selection.provider == "llama_cpp"
    assert selection.explicit_model == "local-model"
    assert selection.base_url == "http://127.0.0.1:9099"


def test_provider_name_resolution_matches_config_key_case_insensitively():
    providers_models = {
        "OpenAI": ["gpt-4o"],
        "Llama_cpp": ["local-model"],
    }

    assert resolve_provider_name("llama_cpp", providers_models) == "Llama_cpp"
    assert resolve_provider_name("local-llamacpp", providers_models) == "local-llamacpp"


def test_console_provider_selection_normalizes_display_provider_key():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {
            "provider": "llama_cpp",
            "model": "local-model",
        },
        "api_settings": {
            "llama_cpp": {
                "api_url": "http://127.0.0.1:9099",
            },
        },
    }
    screen = ChatScreen(app)
    screen._console_control_provider = "Llama_cpp"
    screen._console_control_model = "local-model"

    selection = screen._build_console_provider_selection()

    assert selection.provider == "llama_cpp"
    assert selection.explicit_model == "local-model"
    assert selection.base_url == "http://127.0.0.1:9099"


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

    async with host.run_test(size=(196, 48)) as pilot:
        console = host.screen_stack[-1]
        await _open_console_inspector(console, pilot)
        await _wait_for_selector(console, pilot, "#console-inspector-provider")

        assert "Provider: blocked" in str(
            console.query_one("#console-inspector-provider", Static).renderable
        )
        assert "Select a provider and model before sending." in str(
            console.query_one("#console-inspector-provider", Static).renderable
        )
        assert "Sources: missing source" in str(
            console.query_one("#console-inspector-sources", Static).renderable
        )
        assert not list(console.query("#console-inspector-rag-source"))
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

    async with host.run_test(size=(196, 48)) as pilot:
        console = host.screen_stack[-1]
        await _open_console_inspector(console, pilot)
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
        assert (
            console.query_one("#console-inspector-source-readiness-heading").region.y
            < console.query_one("#console-inspector-tools-heading").region.y
        )
        assert (
            console.query_one("#console-inspector-tools-heading").region.y
            < console.query_one("#console-inspector-tools").region.y
            < console.query_one("#console-inspector-review-tool-call").region.y
            < console.query_one("#console-inspector-approvals-heading").region.y
        )
        assert (
            console.query_one("#console-inspector-approvals-heading").region.y
            < console.query_one("#console-inspector-approvals").region.y
            < console.query_one("#console-inspector-review-approval").region.y
            < console.query_one("#console-inspector-artifacts-heading").region.y
            < console.query_one("#console-inspector-artifacts").region.y
            < console.query_one("#console-inspector-save-chatbook").region.y
        )
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

    async with host.run_test(size=(196, 48)) as pilot:
        console = host.screen_stack[-1]
        await _open_console_inspector(console, pilot)
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
        assert "Sources: staged from Library Search/RAG" in text
        assert "RAG/source:" not in text
        assert "Title: Incident Review" in text
        assert "source_id: note-42" in text
        assert "chunk_id: chunk-7" in text
        assert "Review citations before sending." in text


def test_console_evidence_display_state_sanitizes_markup_fields():
    launch = ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="Grounded answer",
        status="ready",
        payload={
            "evidence_bundle": {
                "bundle_id": "bundle-1",
                "query": "Why did the incident happen?",
                "status": "available",
                "references": [
                    {
                        "evidence_id": "S1",
                        "source_id": "note-42",
                        "source_type": "note",
                        "title": "Incident [red]Review[/red] <script>",
                        "snippet": (
                            "Expired <b>credential</b> caused "
                            "[bold]the incident[/bold]."
                        ),
                        "authority_label": "Workspace <b>A</b>",
                        "status": "available",
                    }
                ],
            }
        },
    )

    state = build_console_evidence_display_state(launch)

    assert state is not None
    rendered = "\n".join(
        [
            state.authority,
            *(row.text for row in state.reference_rows),
        ]
    )
    assert "<script>" not in rendered
    assert "<b>" not in rendered
    assert "[bold]" in rendered
    assert "[red]" in rendered
    assert "&lt;script&gt;" in rendered
    assert "&lt;b&gt;credential&lt;/b&gt;" in rendered


def test_console_evidence_authority_rows_preserve_blocked_status():
    launch = ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="Grounded answer",
        status="blocked",
        payload={
            "evidence_bundle": {
                "bundle_id": "bundle-1",
                "query": "Summarize this source",
                "status": "blocked",
                "references": [
                    {
                        "evidence_id": "S1",
                        "source_id": "note-other",
                        "source_type": "note",
                        "title": "Other Workspace Note",
                        "snippet": "This source belongs to another workspace.",
                        "authority_label": "Workspace B only",
                        "status": "blocked",
                    }
                ],
            }
        },
    )

    evidence_state = build_console_evidence_display_state(launch)
    staged_state = ConsoleStagedContextState.from_live_work(launch)
    inspector_state = ConsoleInspectorState.from_values(
        evidence_summary=evidence_state.summary if evidence_state else "",
        evidence_status=evidence_state.status if evidence_state else "",
        evidence_authority=evidence_state.authority if evidence_state else "",
    )

    staged_rows = {row.label: row for row in staged_state.rows}
    inspector_rows = {row.label: row for row in inspector_state.rows}
    assert staged_rows["Authority"].status == "blocked"
    assert inspector_rows["Authority"].status == "blocked"


@pytest.mark.asyncio
async def test_console_rag_staging_shows_evidence_summary_authority_and_snippet():
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

    async with host.run_test(size=(196, 48)) as pilot:
        console = host.screen_stack[-1]
        await _open_console_inspector(console, pilot)
        await _wait_for_selector(console, pilot, "#console-run-library-rag")

        console.query_one("#console-library-rag-query-input", Input).value = (
            "Why did the incident happen?"
        )
        await _wait_for_console_library_rag_button_state(
            console,
            pilot,
            disabled=False,
        )
        console.query_one("#console-run-library-rag", Button).press()
        await _wait_for_selector(console, pilot, "#console-live-work-payload-source-id")
        await _open_console_inspector(console, pilot)
        await _wait_for_selector(console, pilot, "#console-inspector-evidence")

        text = _visible_text(console)
        assert "Evidence: 1/1 available (available)" in text
        assert "Authority: Source authority: local" in text
        assert "Evidence source: [S1] Incident Review" in text
        assert "Evidence authority: Source authority: local" in text
        assert "Evidence status: available" in text
        assert "Expired credential caused the incident." in text
        assert "evidence_bundle:" not in text


@pytest.mark.asyncio
async def test_console_rag_send_blocks_when_staged_evidence_is_not_context_eligible():
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
    service = StaticConsoleLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "Other Workspace Note",
                    "snippet": "This source belongs to another workspace.",
                    "source_id": "note-other",
                    "chunk_id": "chunk-other",
                    "workspace_ids": ["workspace-b"],
                    "active_workspace_id": "workspace-a",
                    "runtime_backend": "local-fts",
                }
            ],
            "runtime_backend": "local-fts",
        }
    )
    app.library_rag_search_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(196, 48)) as pilot:
        console = host.screen_stack[-1]
        await _open_console_inspector(console, pilot)
        await _wait_for_selector(console, pilot, "#console-run-library-rag")

        console.query_one("#console-library-rag-query-input", Input).value = (
            "Summarize this source"
        )
        await _wait_for_console_library_rag_button_state(
            console,
            pilot,
            disabled=False,
        )
        console.query_one("#console-run-library-rag", Button).press()
        await _wait_for_selector(console, pilot, "#console-live-work-payload-source-id")
        await _open_console_inspector(console, pilot)
        await _wait_for_selector(console, pilot, "#console-inspector-evidence")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("Answer using the staged RAG evidence")
        send_button = console.query_one("#console-send-message", Button)
        await console.handle_console_send_message(Button.Pressed(send_button))
        await _wait_for_visible_text(
            console,
            pilot,
            "Console send blocked: Library Search/RAG has no available evidence",
        )

        text = _visible_text(console)
        assert "Evidence: 0/1 available (blocked)" in text
        assert "Console send blocked: Library Search/RAG has no available evidence" in text
        assert "Review source authority before sending." in text
        assert composer.draft_text() == "Answer using the staged RAG evidence"


@pytest.mark.asyncio
async def test_console_rag_query_validation_blocks_unsafe_markup():
    app = _build_test_app()
    service = StaticConsoleLibraryRagSearchService({"results": []})
    app.library_rag_search_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(196, 48)) as pilot:
        console = host.screen_stack[-1]
        await _open_console_inspector(console, pilot)
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

    async with host.run_test(size=(196, 48)) as pilot:
        console = host.screen_stack[-1]
        await _open_console_inspector(console, pilot)
        await _wait_for_selector(console, pilot, "#console-run-library-rag")

        console.query_one("#console-library-rag-query-input", Input).value = "What changed?"
        await pilot.pause(0.1)
        console.query_one("#console-run-library-rag", Button).press()
        await _wait_for_selector(console, pilot, "#console-live-work-status")

        text = _visible_text(console)
        assert "Status: blocked" in text
        assert "Sources: unavailable" in text
        assert "RAG/source:" not in text
        assert "Unavailable: Library Search/RAG retrieval." in text
        assert "Owner: Library retrieval service." in text
