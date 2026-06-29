import pytest
from textual.app import App, ComposeResult

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus
from tldw_chatbook.Chat.console_display_state import ConsoleControlState
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.Console.console_workbench_state import (
    build_console_workbench_state,
)


class ConsoleHarness(App):
    def __init__(self, app_instance):
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(ChatScreen(self.app_instance))


class ConsoleFooterHarness(App):
    def __init__(self, app_instance):
        super().__init__()
        self.app_instance = app_instance

    def compose(self) -> ComposeResult:
        yield AppFooterStatus(id="app-footer-status")

    async def on_mount(self) -> None:
        await self.push_screen(ChatScreen(self.app_instance))


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


def _is_displayed(widget) -> bool:
    current = widget
    while current is not None:
        if current.display is False or current.styles.display == "none":
            return False
        current = getattr(current, "parent", None)
    return True


def _visible_workbench_focus_ids(console: ChatScreen) -> set[str]:
    candidates = {
        "console-left-rail",
        "console-transcript-surface",
        "console-right-rail",
        "console-native-composer",
    }
    visible: set[str] = set()
    for widget_id in candidates:
        widget = console.query_one(f"#{widget_id}")
        if _is_displayed(widget):
            visible.add(widget_id)
    return visible


def _control_state() -> ConsoleControlState:
    return ConsoleControlState(
        provider_label="Provider: llama.cpp",
        model_label="Model: local-model",
        persona_label="Assistant: General",
        rag_label="RAG: off",
        sources_label="Sources: 0",
        tools_label="Tools: 0",
        approvals_label="Approvals: 0",
    )


def test_console_workbench_state_exposes_core_actions_visibly():
    state = build_console_workbench_state(
        control_state=_control_state(),
        provider_blocker_copy="",
        can_send=True,
        can_stop=False,
        can_save_chatbook=True,
    )

    actions = {action.id: action for action in state.actions}
    action_labels = {action.label for action in state.actions}

    assert {
        "Settings",
        "Attach context",
        "Run Library RAG",
        "Save Chatbook",
        "Help",
    } <= action_labels
    assert tuple(actions) == (
        "new-tab",
        "settings",
        "attach-context",
        "run-library-rag",
        "save-chatbook",
        "send",
        "stop",
        "help",
    )
    assert actions["save-chatbook"].disabled is False
    assert actions["send"].disabled is False
    assert actions["send"].primary is True
    assert actions["stop"].disabled is True
    assert state.route_id == "chat"
    assert state.density == "normal"
    assert state.header.title == "Console"
    assert tuple(pane.id for pane in state.panes) == (
        "context",
        "transcript",
        "inspector",
        "composer",
    )
    assert state.recovery is None


def test_console_workbench_state_surfaces_provider_recovery():
    state = build_console_workbench_state(
        control_state=ConsoleControlState(
            provider_label="Provider: OpenAI",
            model_label="Model: --",
            persona_label="Assistant: General",
            rag_label="RAG: off",
            sources_label="Sources: 0",
            tools_label="Tools: 0",
            approvals_label="Approvals: 0",
        ),
        provider_blocker_copy="Provider setup needed: choose a model",
        provider_action_label="Choose model",
        can_send=False,
        can_stop=False,
        can_save_chatbook=False,
    )

    assert state.recovery is not None
    assert "choose a model" in state.recovery.body.lower()
    assert state.recovery.action is not None
    assert state.recovery.action.label == "Choose model"
    assert state.recovery.action.id == "provider-recovery"
    assert state.recovery.action.primary is True

    modes = {mode.id: mode for mode in state.modes}
    assert modes["provider"].status == "blocked"
    assert modes["model"].status == "blocked"


def test_console_workbench_state_disables_send_when_provider_is_blocked():
    state = build_console_workbench_state(
        control_state=_control_state(),
        provider_blocker_copy="Provider setup needed: choose a model",
        can_send=True,
        can_stop=True,
        can_save_chatbook=True,
        density="compact",
    )

    actions = {action.id: action for action in state.actions}

    assert state.density == "compact"
    assert state.header.status == "blocked"
    assert actions["send"].disabled is True
    assert actions["send"].primary is False
    assert actions["stop"].disabled is False


@pytest.mark.asyncio
async def test_console_core_controls_are_visible_without_command_palette():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        for selector in (
            "#workbench-action-settings",
            "#workbench-action-attach-context",
            "#workbench-action-run-library-rag",
            "#workbench-action-help",
            "#console-native-composer",
        ):
            widget = console.query_one(selector)
            assert _is_displayed(widget), selector


@pytest.mark.asyncio
async def test_console_recovery_action_is_visible_when_provider_setup_blocks_send():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "OpenAI", "model": ""},
        "api_settings": {"openai": {"api_key": ""}},
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = ""
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        recovery = console.query_one("#workbench-recovery-callout")
        assert _is_displayed(recovery)
        recovery_text = " ".join(
            getattr(child.renderable, "plain", str(getattr(child, "renderable", "")))
            for child in recovery.query("Static")
        )
        assert "Send is blocked" in recovery_text


@pytest.mark.asyncio
async def test_console_recovery_action_button_is_visible_and_actionable():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "OpenAI", "model": ""},
        "api_settings": {"openai": {"api_key": ""}},
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = ""
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")
        await _wait_for_selector(console, pilot, "#workbench-recovery-action")

        action = console.query_one("#workbench-recovery-action")
        assert _is_displayed(action)
        assert not action.disabled
        await pilot.click("#workbench-recovery-action")
        await pilot.pause()

        assert host.screen.query("#console-settings-modal") or host.screen.query(
            "#settings-screen"
        )


@pytest.mark.asyncio
async def test_console_workbench_send_action_enables_after_typing_draft():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#workbench-action-send")

        send_action = console.query_one("#workbench-action-send")
        assert _is_displayed(send_action)
        assert send_action.disabled is True

        await pilot.press("h")
        await pilot.pause()

        send_action = console.query_one("#workbench-action-send")
        assert _is_displayed(send_action)
        assert send_action.disabled is False


@pytest.mark.asyncio
async def test_console_workbench_send_action_disables_during_active_run():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#workbench-action-send")
        await _wait_for_selector(console, pilot, "#workbench-action-stop")

        await pilot.press("h")
        await pilot.pause()

        controller = console._ensure_console_chat_controller()
        controller.run_state = ConsoleRunState(
            ConsoleRunStatus.STREAMING,
            "Streaming response.",
        )
        console._sync_console_control_bar()
        await pilot.pause()

        send_action = console.query_one("#workbench-action-send")
        stop_action = console.query_one("#workbench-action-stop")
        assert _is_displayed(send_action)
        assert _is_displayed(stop_action)
        assert send_action.disabled is True
        assert send_action.has_class("is-primary") is False
        assert stop_action.disabled is False


@pytest.mark.asyncio
async def test_console_f6_cycles_visible_workbench_panes():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        visible_focus_ids = _visible_workbench_focus_ids(console)
        assert "console-transcript-surface" in visible_focus_ids
        assert "console-native-composer" in visible_focus_ids

        await pilot.press("f6")
        await pilot.pause()
        first_focus = pilot.app.focused
        assert first_focus is not None
        assert first_focus.id in visible_focus_ids

        await pilot.press("f6")
        await pilot.pause()
        second_focus = pilot.app.focused
        assert second_focus is not None
        assert second_focus.id in visible_focus_ids
        assert second_focus.id != first_focus.id


@pytest.mark.asyncio
async def test_console_route_switch_restores_workbench_focus():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        console.query_one("#console-native-composer").focus()
        await pilot.pause()
        assert pilot.app.focused is not None
        assert pilot.app.focused.id == "console-native-composer"

        await console.remove()
        await pilot.pause()
        await host.push_screen(ChatScreen(app))
        await pilot.pause()

        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")
        visible_focus_ids = _visible_workbench_focus_ids(console)
        assert pilot.app.focused is not None
        assert pilot.app.focused.id in visible_focus_ids


@pytest.mark.asyncio
async def test_console_f1_help_lists_visible_actions():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        await pilot.press("f1")
        await pilot.pause()

        panel = host.screen.query_one("#workbench-help-panel")
        assert _is_displayed(panel)
        body = str(host.screen.query_one("#workbench-help-body").renderable)
        assert "Settings" in body
        assert "Attach context" in body
        assert "Run Library RAG" in body
        assert "F6" in body
        assert "next pane" in body
        assert "Ctrl+P" in body


@pytest.mark.asyncio
async def test_console_registers_footer_workbench_shortcuts():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleFooterHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")
        footer = host.query_one(AppFooterStatus)

        assert footer.shortcut_text == (
            "F6 next pane | F1 help | Enter send | Ctrl+P palette"
        )

        await console.remove()
        await pilot.pause()

        assert footer.shortcut_text == AppFooterStatus.DEFAULT_SHORTCUT_TEXT


@pytest.mark.asyncio
async def test_console_shell_uses_configured_workbench_density():
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.app_config.setdefault("appearance", {})["ui_density"] = "compact"
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        shell = console.query_one("#console-shell")
        assert shell.has_class("density-compact")
        assert not shell.has_class("density-normal")


@pytest.mark.asyncio
async def test_console_shell_uses_normal_workbench_density_by_default():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        shell = console.query_one("#console-shell")
        assert shell.has_class("density-normal")
        assert not shell.has_class("density-compact")


@pytest.mark.asyncio
async def test_console_shell_uses_legacy_density_setting_fallback():
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.app_config.setdefault("appearance", {})["density"] = "compact"
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        shell = console.query_one("#console-shell")
        assert shell.has_class("density-compact")
        assert not shell.has_class("density-normal")


@pytest.mark.asyncio
async def test_console_shell_invalid_workbench_density_falls_back_to_normal():
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.app_config.setdefault("appearance", {})["ui_density"] = "spacious"
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        shell = console.query_one("#console-shell")
        assert shell.has_class("density-normal")
        assert not shell.has_class("density-compact")
