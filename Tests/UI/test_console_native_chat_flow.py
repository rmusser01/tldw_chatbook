import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from textual import on
from textual.widgets import Button, Input, Static, TextArea

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole, ConsoleRunStatus
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.Widgets.Console import ConsoleComposerBar, ConsoleTranscript
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.Workspaces import DEFAULT_WORKSPACE_ID


DUMMY_OPENAI_API_KEY = "DUMMY_OPENAI_API_KEY"


class _ReadyResolutionGateway:
    async def resolve_for_send(self, selection):
        return SimpleNamespace(
            provider=selection.provider,
            base_url=selection.base_url or "",
            model=selection.explicit_model or selection.configured_model or "test-model",
            ready=True,
            visible_copy="",
        )


class SelectionCapturingGateway(_ReadyResolutionGateway):
    def __init__(self) -> None:
        self.selections = []
        self.sent_messages = []

    async def resolve_for_send(self, selection):
        self.selections.append(selection)
        return await super().resolve_for_send(selection)

    async def stream_chat(self, resolution, messages):
        self.sent_messages.append(list(messages))
        yield "accepted"


class WaitingGateway(_ReadyResolutionGateway):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream_chat(self, resolution, messages):
        yield "partial"
        self.started.set()
        await self.release.wait()
        yield " done"


class DelayedWaitingGateway(WaitingGateway):
    def __init__(self) -> None:
        super().__init__()
        self.validation_started = asyncio.Event()
        self.validation_release = asyncio.Event()

    async def resolve_for_send(self, selection):
        self.validation_started.set()
        await self.validation_release.wait()
        return await super().resolve_for_send(selection)


class ConsoleNavigationHarness(ConsoleHarness):
    def __init__(self, app_instance: object) -> None:
        super().__init__(app_instance)
        self.navigation_messages = []

    @on(NavigateToScreen)
    def capture_navigation(self, message: NavigateToScreen) -> None:
        self.navigation_messages.append(message)
        message.stop()


class BlockedGateway:
    async def resolve_for_send(self, selection):
        return SimpleNamespace(
            provider="llama_cpp",
            base_url=selection.base_url or "",
            model="test-model",
            ready=False,
            visible_copy="Provider blocked: llama.cpp unavailable.",
        )

    async def stream_chat(self, resolution, messages):
        raise AssertionError("Blocked gateway should not stream")


class CapturingGateway(_ReadyResolutionGateway):
    def __init__(self, chunks=("accepted",)) -> None:
        self.chunks = chunks
        self.sent_messages = []

    async def stream_chat(self, resolution, messages):
        self.sent_messages.append(list(messages))
        for chunk in self.chunks:
            yield chunk


class WorkspaceLinkingPersistence:
    def __init__(self, registry_service) -> None:
        self.registry_service = registry_service
        self.conversation_count = 0
        self.message_count = 0

    def create_conversation(self, **kwargs):
        self.conversation_count += 1
        conversation_id = f"persisted-conversation-{self.conversation_count}"
        workspace_id = kwargs.get("workspace_id")
        if kwargs.get("scope_type") == "workspace" and workspace_id:
            self.registry_service.link_membership(
                workspace_id,
                item_type="conversation",
                item_id=conversation_id,
                role="workspace-thread",
                title=kwargs.get("conversation_title") or "Chat 1",
            )
        return conversation_id

    def create_message(self, **kwargs):
        self.message_count += 1
        return f"persisted-message-{self.message_count}"

    def update_message_content(self, **kwargs):
        return True


class FailThenRecoverGateway(_ReadyResolutionGateway):
    def __init__(self) -> None:
        self.calls = 0

    async def stream_chat(self, resolution, messages):
        self.calls += 1
        if self.calls == 1:
            yield "partial"
            raise RuntimeError("llama.cpp stream failed")
        yield "recovered"


async def _wait_for_text(screen, pilot, expected: str, *, attempts: int = 80) -> None:
    for _ in range(attempts):
        if expected in _visible_text(screen):
            return
        await pilot.pause(0.05)
    raise AssertionError(f"Text not found: {expected!r}. Visible text: {_visible_text(screen)!r}")


async def _wait_for_focus(app, pilot, widget, *, attempts: int = 40) -> None:
    for _ in range(attempts):
        if getattr(app, "focused", None) is widget:
            return
        await pilot.pause(0.05)
    focused = getattr(app, "focused", None)
    raise AssertionError(
        f"Focus did not reach {getattr(widget, 'id', widget)!r}; "
        f"focused={getattr(focused, 'id', focused)!r}"
    )


def _static_plain_text(widget: Static) -> str:
    renderable = widget.renderable
    return getattr(renderable, "plain", str(renderable))


def _widget_text(widget) -> str:
    if hasattr(widget, "renderable"):
        renderable = widget.renderable
        return getattr(renderable, "plain", str(renderable))
    label = getattr(widget, "label", "")
    return getattr(label, "plain", str(label))


def _console_workspace_conversation_texts(console) -> list[str]:
    rows = console.query(".console-workspace-conversation-row")
    return [_widget_text(row) for row in rows]


def _console_workspace_conversation_row_id_for_session(console, session_id: str) -> str:
    target_conversation_id = f"native:{session_id}"
    for row in console.query(".console-workspace-conversation-row"):
        if getattr(row, "conversation_id", None) == target_conversation_id:
            return str(row.id)
    rows = [
        (getattr(row, "id", ""), getattr(row, "conversation_id", None), _widget_text(row))
        for row in console.query(".console-workspace-conversation-row")
    ]
    raise AssertionError(
        f"Workspace conversation row for {target_conversation_id!r} not found. "
        f"Rows: {rows!r}"
    )


async def _click_console_workspace_conversation_for_session(
    console,
    pilot,
    store,
    session_id: str,
    *,
    attempts: int = 20,
) -> None:
    """Click a workspace conversation row once Textual hit-testing is ready."""
    row_id = _console_workspace_conversation_row_id_for_session(console, session_id)
    for _ in range(attempts):
        if await pilot.click(f"#{row_id}"):
            for _ in range(10):
                if store.active_session_id == session_id:
                    return
                await pilot.pause(0.05)
        await pilot.pause(0.05)
    rows = [
        (
            getattr(row, "id", ""),
            getattr(row, "conversation_id", None),
            getattr(row, "region", None),
            _widget_text(row),
        )
        for row in console.query(".console-workspace-conversation-row")
    ]
    raise AssertionError(
        f"Workspace conversation click did not activate {session_id!r}. "
        f"active={store.active_session_id!r}; rows={rows!r}"
    )


async def _wait_for_workspace_conversation_text(
    console,
    pilot,
    expected: str,
    *,
    selected: bool | None = None,
    attempts: int = 40,
) -> list[str]:
    for _ in range(attempts):
        row_texts = _console_workspace_conversation_texts(console)
        for text in row_texts:
            if expected not in text:
                continue
            if selected is None or text.startswith("> ") == selected:
                return row_texts
        await pilot.pause(0.05)
    raise AssertionError(
        f"Workspace conversation {expected!r} not found. "
        f"Rows: {_console_workspace_conversation_texts(console)!r}"
    )


async def _wait_for_console_rename_modal(host: ConsoleHarness, pilot):
    for _ in range(40):
        if (
            host.screen_stack
            and host.screen_stack[-1].query("#console-rename-session-modal")
            and host.screen_stack[-1].query("#console-rename-session-title")
        ):
            await pilot.pause()
            return host.screen_stack[-1]
        await pilot.pause(0.05)
    raise AssertionError("Console rename modal did not open")


async def _wait_for_console_screen(host: ConsoleHarness, console, pilot) -> None:
    for _ in range(40):
        if host.screen_stack and host.screen_stack[-1] is console:
            await pilot.pause()
            return
        await pilot.pause(0.05)
    raise AssertionError("Console modal did not dismiss")


async def _wait_for_workspace_switcher_modal(host: ConsoleHarness, pilot):
    for _ in range(40):
        if (
            host.screen_stack
            and host.screen_stack[-1].query("#console-workspace-switcher-modal")
        ):
            await pilot.pause()
            return host.screen_stack[-1]
        await pilot.pause(0.05)
    raise AssertionError("Console workspace switcher modal did not open")


def _select_llamacpp_console(console: ChatScreen) -> None:
    """Select the native llama.cpp path after mounted controls initialize."""
    app_config = console.app_instance.app_config
    api_settings = app_config.setdefault("api_settings", {})
    llama_settings = api_settings.setdefault("llama_cpp", {})
    llama_settings.setdefault("api_url", "http://127.0.0.1:9099/v1")
    llama_settings.setdefault("model", "test-model")
    console._console_control_provider = "llama_cpp"
    console._console_control_model = "test-model"
    console._sync_console_control_bar()


@pytest.mark.asyncio
async def test_console_native_generic_provider_send_renders_completed_message(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {}
    captured_kwargs = []

    def fake_chat_api_call(**_kwargs):
        captured_kwargs.append(_kwargs)
        return "generic provider response"

    monkeypatch.setattr(
        "tldw_chatbook.Chat.Chat_Functions.chat_api_call",
        fake_chat_api_call,
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        gateway = console._ensure_console_provider_gateway()
        app.app_config["api_settings"] = {"openai": {"api_key": DUMMY_OPENAI_API_KEY}}
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "generic provider response")

        assert isinstance(gateway, ConsoleProviderGateway)
        assert captured_kwargs
        assert captured_kwargs[-1]["api_endpoint"] == "openai"
        assert captured_kwargs[-1]["api_key"] == DUMMY_OPENAI_API_KEY
        assert console._ensure_console_chat_controller().run_state.status is ConsoleRunStatus.COMPLETED
        store = console._ensure_console_chat_store()
        messages = store.messages_for_session(store.active_session_id)
        assistant_messages = [
            message for message in messages if message.role is ConsoleMessageRole.ASSISTANT
        ]
        assert assistant_messages[-1].status == "complete"


@pytest.mark.asyncio
async def test_console_native_send_button_click_dispatches_message(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {}
    captured_kwargs = []

    def fake_chat_api_call(**_kwargs):
        captured_kwargs.append(_kwargs)
        return "click provider response"

    monkeypatch.setattr(
        "tldw_chatbook.Chat.Chat_Functions.chat_api_call",
        fake_chat_api_call,
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        app.app_config["api_settings"] = {"openai": {"api_key": DUMMY_OPENAI_API_KEY}}
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("click send")

        await pilot.click("#console-send-message")
        await _wait_for_text(console, pilot, "click provider response")

        assert captured_kwargs
        assert composer.draft_text() == ""


@pytest.mark.asyncio
async def test_console_successful_send_does_not_leave_empty_send_tooltip(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": {"api_key": DUMMY_OPENAI_API_KEY}}

    def fake_chat_api_call(**_kwargs):
        return "sent response"

    monkeypatch.setattr(
        "tldw_chatbook.Chat.Chat_Functions.chat_api_call",
        fake_chat_api_call,
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("send once")

        await pilot.click("#console-send-message")
        await _wait_for_text(console, pilot, "sent response")

        send_button = console.query_one("#console-send-message", Button)
        assert composer.draft_text() == ""
        assert send_button.tooltip != "Type a message before sending."


@pytest.mark.asyncio
async def test_console_native_missing_key_blocks_before_clearing_generic_draft():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {"api_key_env_var": "MISSING_OPENAI_KEY"}
    }
    app.console_provider_gateway_factory = lambda: ConsoleProviderGateway(
        config_provider=lambda: app.app_config,
        environ={},
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("preserve this")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "missing API key")

        assert composer.draft_text() == "preserve this"


@pytest.mark.asyncio
async def test_console_native_enter_on_setup_blocked_send_shows_recovery_feedback():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {"api_key_env_var": "MISSING_OPENAI_KEY"}
    }
    app.console_provider_gateway_factory = lambda: ConsoleProviderGateway(
        config_provider=lambda: app.app_config,
        environ={},
    )
    notifications: list[tuple[str, dict]] = []
    app.notify = lambda message, **kwargs: notifications.append((message, kwargs))
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("preserve this from keyboard")

        await pilot.press("enter")
        await pilot.pause(0.05)

        assert composer.draft_text() == "preserve this from keyboard"
        assert notifications == [
            (
                "Add API Key in Settings before sending.",
                {"severity": "warning"},
            )
        ]


@pytest.mark.asyncio
async def test_console_native_blocked_send_preserves_composer_text_and_shows_recovery():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = BlockedGateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("blocked draft")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "Provider blocked")

        assert composer.draft_text() == "blocked draft"


@pytest.mark.parametrize(
    ("raw_url", "expected"),
    (
        ("http://127.0.0.1:9099/v1/chat/completions", "http://127.0.0.1:9099"),
        ("http://127.0.0.1:9099/v1/models", "http://127.0.0.1:9099"),
        ("http://127.0.0.1:9099/v1", "http://127.0.0.1:9099"),
        ("127.0.0.1:9099", "http://127.0.0.1:9099"),
        ("127.0.0.1:9099/v1", "http://127.0.0.1:9099"),
        ("http://127.0.0.1:9099/completion", "http://127.0.0.1:9099"),
        ("http://127.0.0.1:9099/", "http://127.0.0.1:9099"),
        (None, "http://127.0.0.1:9099"),
    ),
)
def test_console_llamacpp_base_url_normalizes_openai_compatible_endpoints(raw_url, expected):
    screen = ChatScreen(_build_test_app())

    assert screen._normalize_llamacpp_base_url(raw_url) == expected


def test_console_transcript_sync_timer_polls_at_coarse_interval(monkeypatch):
    screen = ChatScreen(_build_test_app())
    captured = {}

    def fake_set_interval(interval, callback):
        captured["interval"] = interval
        captured["callback"] = callback
        return SimpleNamespace(stop=lambda: None)

    monkeypatch.setattr(screen, "set_interval", fake_set_interval)

    screen._start_console_transcript_sync_timer()

    assert captured["interval"] >= 0.15


def test_console_transcript_fingerprint_tolerates_empty_variant_container():
    screen = ChatScreen(_build_test_app())
    message = SimpleNamespace(
        id="m1",
        role=ConsoleMessageRole.ASSISTANT,
        content="answer",
        status="complete",
        turn_id="turn-1",
        persisted_message_id=None,
        variants=SimpleNamespace(selected_index=0, variants=None),
    )

    fingerprint = screen._native_console_transcript_fingerprint([message])

    assert fingerprint[1][0][-1] == (0, ())


def test_console_provider_selection_reads_local_llamacpp_configured_model():
    app = _build_test_app()
    app.chat_api_provider_value = "local_llamacpp"
    app.chat_api_model_value = "runtime-model"
    app.app_config["api_settings"] = {
        "local_llamacpp": {
            "api_url": "http://127.0.0.1:9099/v1/chat/completions",
            "model": "configured-model",
        }
    }
    screen = ChatScreen(app)

    selection = screen._build_console_provider_selection()

    assert selection.provider == "local_llamacpp"
    assert selection.base_url == "http://127.0.0.1:9099"
    assert selection.explicit_model == "runtime-model"
    assert selection.configured_model == "configured-model"
    assert selection.workspace_context.active_workspace_id == DEFAULT_WORKSPACE_ID


def test_console_provider_selection_restores_default_workspace_when_none_active():
    app = _build_test_app()
    service = app.workspace_registry_service
    with service.db.transaction() as conn:
        conn.execute("UPDATE workspace_records SET active = 0")
    assert service.get_active_workspace() is None
    screen = ChatScreen(app)

    selection = screen._build_console_provider_selection()

    assert selection.workspace_context.active_workspace_id == DEFAULT_WORKSPACE_ID
    assert service.get_active_workspace().workspace_id == DEFAULT_WORKSPACE_ID


def test_console_configured_llamacpp_override_wins_over_provider_api_url():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "configured-model"
    app.app_config["console"] = {
        "llama_cpp_base_url_override": "http://127.0.0.1:9099/v1",
    }
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://localhost:8080/v1",
            "model": "fallback-model",
        }
    }
    screen = ChatScreen(app)

    selection = screen._build_console_provider_selection()

    assert selection.base_url == "http://127.0.0.1:9099"


def test_console_llamacpp_api_base_url_wins_over_merged_provider_api_url(monkeypatch):
    monkeypatch.delenv("TLDW_CONSOLE_LLAMA_CPP_BASE_URL", raising=False)
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "configured-model"
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://localhost:8080/v1",
            "api_base_url": "http://127.0.0.1:9099/v1",
            "model": "fallback-model",
        }
    }
    screen = ChatScreen(app)

    selection = screen._build_console_provider_selection()

    assert selection.base_url == "http://127.0.0.1:9099"


def test_console_llamacpp_env_url_wins_over_provider_api_url(monkeypatch):
    monkeypatch.setenv("TLDW_CONSOLE_LLAMA_CPP_BASE_URL", "http://127.0.0.1:9099/v1")
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "configured-model"
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://localhost:8080/v1",
            "model": "fallback-model",
        }
    }
    screen = ChatScreen(app)

    selection = screen._build_console_provider_selection()

    assert selection.base_url == "http://127.0.0.1:9099"


def test_console_session_settings_blank_base_url_keeps_llamacpp_fallback(monkeypatch):
    monkeypatch.setenv("TLDW_CONSOLE_LLAMA_CPP_BASE_URL", "http://127.0.0.1:9099/v1")
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "runtime-model"
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://localhost:8080/v1",
            "model": "fallback-model",
        }
    }
    screen = ChatScreen(app)
    store = ConsoleChatStore()
    session = store.create_session(
        settings=ConsoleSessionSettings(
            provider="llama_cpp",
            model="settings-model",
            base_url=None,
        )
    )
    store.switch_session(session.id)
    screen._console_chat_store = store

    selection = screen._build_console_provider_selection()

    assert selection.base_url == "http://127.0.0.1:9099"
    assert selection.explicit_model == "settings-model"


def test_console_session_settings_base_url_wins_over_llamacpp_fallback(monkeypatch):
    monkeypatch.setenv("TLDW_CONSOLE_LLAMA_CPP_BASE_URL", "http://127.0.0.1:9099/v1")
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "runtime-model"
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://localhost:8080/v1",
            "model": "fallback-model",
        }
    }
    screen = ChatScreen(app)
    store = ConsoleChatStore()
    session = store.create_session(
        settings=ConsoleSessionSettings(
            provider="llama_cpp",
            model="settings-model",
            base_url="http://127.0.0.1:9999/v1",
        )
    )
    store.switch_session(session.id)
    screen._console_chat_store = store

    selection = screen._build_console_provider_selection()

    assert selection.base_url == "http://127.0.0.1:9999"
    assert selection.explicit_model == "settings-model"


@pytest.mark.asyncio
async def test_console_stop_interrupts_stream_and_keeps_partial_message_visible():
    gateway = WaitingGateway()
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await asyncio.wait_for(gateway.started.wait(), timeout=1)
        await _wait_for_text(console, pilot, "partial")
        assert "streaming" in _visible_text(console).lower()

        console.query_one("#console-stop-generation", Button).press()
        await _wait_for_text(console, pilot, "stopped")

        store = console._ensure_console_chat_store()
        messages = store.messages_for_session(store.active_session_id)
        assert messages[-1].content == "partial"
        assert messages[-1].status == "stopped"


@pytest.mark.asyncio
async def test_console_composer_stop_is_subdued_when_idle():
    gateway = WaitingGateway()
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        send_button = composer.query_one("#console-send-message", Button)
        stop_button = composer.query_one("#console-stop-generation", Button)

        assert stop_button.disabled is False
        assert stop_button.has_class("console-action-disabled")
        assert stop_button.has_class("console-stop-idle")
        assert not stop_button.has_class("console-stop-active")

        composer.load_draft("hello")
        console.query_one("#console-send-message", Button).press()
        await asyncio.wait_for(gateway.started.wait(), timeout=1)
        await _wait_for_text(console, pilot, "partial")

        assert send_button.disabled is False
        assert send_button.has_class("console-action-disabled")
        assert send_button.has_class("console-send-blocked")
        assert not send_button.has_class("console-action-primary")
        assert stop_button.disabled is False
        assert stop_button.has_class("console-stop-active")
        assert not stop_button.has_class("console-action-disabled")
        assert not stop_button.has_class("console-stop-idle")

        stop_button.press()
        await _wait_for_text(console, pilot, "stopped")


@pytest.mark.asyncio
async def test_console_duplicate_send_during_stream_does_not_break_stop_control():
    gateway = WaitingGateway()
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await asyncio.wait_for(gateway.started.wait(), timeout=1)
        await _wait_for_text(console, pilot, "partial")

        composer.load_draft("second send")
        send_button = console.query_one("#console-send-message", Button)
        assert send_button.disabled is False
        assert send_button.has_class("console-send-blocked")
        await console.handle_console_send_message(Button.Pressed(send_button))
        await pilot.pause(0.1)
        assert console._ensure_console_chat_controller().run_state.status.value == "streaming"

        console.query_one("#console-stop-generation", Button).press()
        await _wait_for_text(console, pilot, "stopped")


@pytest.mark.asyncio
async def test_console_streaming_chunks_render_after_slow_provider_validation():
    gateway = DelayedWaitingGateway()
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await asyncio.wait_for(gateway.validation_started.wait(), timeout=1)
        assert console._ensure_console_chat_controller().run_state.status is ConsoleRunStatus.VALIDATING
        console._sync_console_control_bar()
        send_button = console.query_one("#console-send-message", Button)
        stop_button = console.query_one("#console-stop-generation", Button)

        assert send_button.disabled is False
        assert send_button.has_class("console-action-disabled")
        assert send_button.has_class("console-send-blocked")
        assert not send_button.has_class("console-action-primary")
        assert stop_button.disabled is False
        assert stop_button.has_class("console-stop-idle")

        gateway.validation_release.set()
        await asyncio.wait_for(gateway.started.wait(), timeout=1)
        await _wait_for_text(console, pilot, "partial")
        gateway.release.set()
        await _wait_for_text(console, pilot, "partial done")


@pytest.mark.asyncio
async def test_console_collapsed_paste_sends_full_payload_not_visible_token():
    long_text = "x" * 80
    gateway = CapturingGateway()
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_pasted_text(long_text)

        assert "Pasted Text: 80 Characters" in _visible_text(console)
        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "accepted")

    assert gateway.sent_messages[-1][-1]["content"] == long_text
    assert "Pasted Text: 80 Characters" not in gateway.sent_messages[-1][-1]["content"]


@pytest.mark.asyncio
async def test_console_native_send_preserves_expanded_payload_whitespace():
    gateway = CapturingGateway()
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("  padded payload  ")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "accepted")

    assert gateway.sent_messages[-1][-1]["content"] == "  padded payload  "


@pytest.mark.asyncio
async def test_console_configured_model_reaches_gateway_when_ui_model_is_unset():
    gateway = SelectionCapturingGateway()
    app = _build_test_app()
    app.chat_api_provider_value = "local_llamacpp"
    app.chat_api_model_value = None
    app.console_provider_gateway_factory = lambda: gateway
    app.app_config["api_settings"] = {
        "local_llamacpp": {
            "api_url": "http://127.0.0.1:9099/v1/chat/completions",
            "model": "configured-model",
        }
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        console._console_control_provider = "local_llamacpp"
        console._console_control_model = None
        console._sync_console_control_bar()
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "accepted")

    assert gateway.selections[-1].explicit_model is None
    assert gateway.selections[-1].configured_model == "configured-model"


@pytest.mark.asyncio
async def test_console_native_send_clears_composer_after_acceptance_and_updates_store():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: CapturingGateway(chunks=("hel", "lo"))
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "Assistant\nhello")

        assert composer.draft_text() == ""
        store = console._ensure_console_chat_store()
        messages = store.messages_for_session(store.active_session_id)
        assert messages[-2].content == "hello"
        assert messages[-1].content == "hello"


@pytest.mark.asyncio
async def test_console_send_refreshes_workspace_conversation_rail_after_persistence():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: CapturingGateway(chunks=("accepted",))
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        row_texts = _console_workspace_conversation_texts(console)
        assert any("Chat 1" in text for text in row_texts)
        assert len(console.query("#console-workspace-empty-conversations")) == 0
        store = console._ensure_console_chat_store()
        store.persistence = WorkspaceLinkingPersistence(app.workspace_registry_service)
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "accepted")
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-0")

        row = console.query_one("#console-workspace-conversation-0")
        row_text = _widget_text(row)
        assert row_text.startswith("> ")
        assert "Chat 1" in row_text
        assert "workspace-thread" in row_text
        assert len(console.query("#console-workspace-empty-conversations")) == 0


@pytest.mark.asyncio
async def test_console_send_after_workspace_switch_persists_to_selected_workspace():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: CapturingGateway(chunks=("accepted",))
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")
    service.set_active_workspace("ws-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_selector(console, pilot, "#console-change-workspace")
        store = console._ensure_console_chat_store()
        store.persistence = WorkspaceLinkingPersistence(service)
        _select_llamacpp_console(console)
        first_session = store.ensure_session()
        store.replace_session_settings(
            first_session.id,
            ConsoleSessionSettings(provider="llama_cpp", model="test-model"),
        )
        assert first_session.workspace_id == "ws-a"

        console.query_one("#console-change-workspace", Button).press()
        modal_screen = await _wait_for_workspace_switcher_modal(host, pilot)
        switch_button = next(
            button
            for button in modal_screen.query(Button)
            if str(button.label) == "Workspace B"
        )
        switch_button.press()
        await _wait_for_console_screen(host, console, pilot)
        assert service.get_active_workspace().workspace_id == "ws-b"

        active_session = store.switch_session(store.active_session_id)
        assert active_session.workspace_id == "ws-b"
        assert active_session.title == "Workspace B Chat"
        assert active_session.settings.provider == "llama_cpp"
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello from b")
        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "accepted")

        workspace_a_conversations = service.list_workspace_conversations("ws-a")
        workspace_b_conversations = service.list_workspace_conversations("ws-b")
        assert workspace_a_conversations == ()
        assert [row.title for row in workspace_b_conversations] == [active_session.title]


@pytest.mark.asyncio
async def test_console_workspace_switch_refreshes_visible_session_tabs():
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")
    service.set_active_workspace("ws-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_selector(console, pilot, "#console-change-workspace")
        store = console._ensure_console_chat_store()
        first_session = store.ensure_session()
        assert first_session.workspace_id == "ws-a"

        console.query_one("#console-change-workspace", Button).press()
        modal_screen = await _wait_for_workspace_switcher_modal(host, pilot)
        switch_button = next(
            button
            for button in modal_screen.query(Button)
            if str(button.label) == "Workspace B"
        )
        switch_button.press()
        await _wait_for_console_screen(host, console, pilot)

        active_session = store.switch_session(store.active_session_id)
        assert active_session.workspace_id == "ws-b"
        await _wait_for_selector(console, pilot, f"#console-session-tab-{active_session.id}")
        assert "Workspace B Chat" in _visible_text(console)


@pytest.mark.asyncio
async def test_console_workspace_switch_refresh_is_not_dropped_during_inflight_sync():
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")
    service.set_active_workspace("ws-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_selector(console, pilot, "#console-change-workspace")
        store = console._ensure_console_chat_store()
        first_session = store.ensure_session()
        assert first_session.workspace_id == "ws-a"

        first_sync_blocked = asyncio.Event()
        release_first_sync = asyncio.Event()
        original_sync_tabs = console._sync_console_native_session_tabs
        blocked_once = False

        async def blocking_sync_tabs():
            nonlocal blocked_once
            await original_sync_tabs()
            if blocked_once:
                return
            blocked_once = True
            first_sync_blocked.set()
            await release_first_sync.wait()

        console._sync_console_native_session_tabs = blocking_sync_tabs
        first_sync_task = asyncio.create_task(console._sync_native_console_chat_ui())
        await first_sync_blocked.wait()

        console.query_one("#console-change-workspace", Button).press()
        modal_screen = await _wait_for_workspace_switcher_modal(host, pilot)
        switch_button = next(
            button
            for button in modal_screen.query(Button)
            if str(button.label) == "Workspace B"
        )
        switch_button.press()
        await _wait_for_console_screen(host, console, pilot)

        active_session = store.switch_session(store.active_session_id)
        assert active_session.workspace_id == "ws-b"
        release_first_sync.set()
        await first_sync_task

        await _wait_for_selector(console, pilot, f"#console-session-tab-{active_session.id}")
        assert "Workspace B Chat" in _visible_text(console)


@pytest.mark.asyncio
async def test_console_mount_uses_active_workspace_title_for_initial_session():
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.set_active_workspace("ws-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_text(console, pilot, "Workspace A Chat")
        assert "Workspace A" in _visible_text(console)


@pytest.mark.asyncio
async def test_console_tab_switch_aligns_active_workspace_context():
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")
    service.set_active_workspace("ws-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Workspace A Chat", workspace_id="ws-a")
        second = store.create_session(title="Workspace B Chat", workspace_id="ws-b")
        service.set_active_workspace("ws-b")
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-session-tab-{first.id}")
        assert store.active_session_id == second.id
        assert service.get_active_workspace().workspace_id == "ws-b"

        await pilot.click(f"#console-session-tab-{first.id}")

        assert store.active_session_id == first.id
        assert service.get_active_workspace().workspace_id == "ws-a"
        await _wait_for_text(console, pilot, "Workspace A")


@pytest.mark.asyncio
async def test_console_unsupported_provider_block_renders_one_normalized_system_message():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(provider="wip_provider", model="test-model"),
        )
        await console._sync_native_console_chat_ui()

        await console._submit_console_native_draft("hello")
        await _wait_for_text(console, pilot, "Provider blocked")

        messages = store.messages_for_session(store.active_session_id)
        system_messages = [message.content for message in messages if message.role is ConsoleMessageRole.SYSTEM]
        assert system_messages == [
            "Provider blocked: 'wip_provider' is not available in Console yet. "
            "Choose a supported provider."
        ]
        assert console._ensure_console_chat_controller().run_state.visible_copy == system_messages[0]


@pytest.mark.asyncio
async def test_console_add_api_key_recovery_targets_provider_settings_category():
    app = _build_test_app()
    app.app_config["api_settings"] = {"huggingface": {}}
    host = ConsoleNavigationHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(provider="huggingface", model="meta-llama/test-model"),
        )
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, "#console-open-provider-settings")

        await pilot.click("#console-open-provider-settings")

        assert len(host.navigation_messages) == 1
        message = host.navigation_messages[0]
        assert message.screen_name == "settings"
        assert message.screen_context == {
            "category": SettingsCategoryId.PROVIDERS_MODELS.value,
            "provider": "huggingface",
            "model": "meta-llama/test-model",
        }


@pytest.mark.asyncio
async def test_console_add_api_key_recovery_tolerates_missing_session_settings():
    app = _build_test_app()
    host = ConsoleNavigationHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(provider="huggingface", model="meta-llama/test-model"),
        )
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, "#console-open-provider-settings")
        console._active_console_provider_model_display = lambda: (
            "huggingface",
            "meta-llama/test-model",
            None,
        )

        await pilot.click("#console-open-provider-settings")

        assert len(host.navigation_messages) == 1
        message = host.navigation_messages[0]
        assert message.screen_context == {
            "category": SettingsCategoryId.PROVIDERS_MODELS.value,
            "provider": "huggingface",
            "model": "meta-llama/test-model",
        }


@pytest.mark.asyncio
async def test_console_assistant_message_click_exposes_selected_actions():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-{message.id}")

        await pilot.click(f"#console-message-{message.id}")
        await _wait_for_selector(console, pilot, f"#console-message-action-regenerate-{message.id}")

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        assert transcript.selected_message_id == message.id


@pytest.mark.asyncio
async def test_console_transcript_wraps_long_message_content_without_horizontal_overflow():
    app = _build_test_app()
    host = ConsoleHarness(app)

    long_answer = " ".join(["wrapped assistant response segment"] * 180)

    async with host.run_test(size=(92, 32)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content=long_answer,
        )
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-{message.id}")

        row = console.query_one(f"#console-message-{message.id}", Static)

        assert row.region.width <= transcript.region.width
        assert transcript.virtual_size.width <= transcript.region.width
        assert row.region.height > 2


@pytest.mark.asyncio
async def test_console_selected_message_copy_action_uses_app_clipboard():
    app = _build_test_app()
    app.copy_to_clipboard = Mock()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-copy-{message.id}")

        await pilot.click(f"#console-message-action-copy-{message.id}")
        await pilot.pause()

    app.copy_to_clipboard.assert_called_once_with("answer")
    assert console._last_console_action.action_id == "copy"


@pytest.mark.asyncio
async def test_console_clicking_rendered_message_shows_action_row():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-{message.id}")

        await pilot.click(f"#console-message-{message.id}")
        await _wait_for_selector(console, pilot, f"#console-message-action-copy-{message.id}")


@pytest.mark.asyncio
async def test_console_selected_message_copy_action_works_from_keyboard():
    app = _build_test_app()
    app.copy_to_clipboard = Mock()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.focus()
        await _wait_for_focus(console.app, pilot, transcript)
        await pilot.press("down")
        await pilot.press("enter")
        await _wait_for_selector(console, pilot, f"#console-message-action-copy-{message.id}")

        copy_selector = f"console-message-action-copy-{message.id}"
        for _ in range(16):
            focused = getattr(console.app, "focused", None)
            if getattr(focused, "id", None) == copy_selector:
                break
            await pilot.press("tab")
        else:
            raise AssertionError("Keyboard focus did not reach the selected-message Copy action")

        await pilot.press("enter")
        await pilot.pause()

    app.copy_to_clipboard.assert_called_once_with("answer")
    assert console._last_console_action.action_id == "copy"


@pytest.mark.asyncio
async def test_console_selected_message_feedback_action_records_rating():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-feedback-up-{message.id}")

        await pilot.click(f"#console-message-action-feedback-up-{message.id}")
        await pilot.pause()

    updated = store.get_message(message.id)
    assert updated.feedback == "up"
    assert console._last_console_action.action_id == "feedback-up"
    assert console._last_console_action.visible_copy == "Marked message feedback: up."


@pytest.mark.asyncio
async def test_console_selected_message_delete_action_removes_message_from_transcript():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-delete-{message.id}")

        await pilot.click(f"#console-message-action-delete-{message.id}")
        await pilot.pause()

    assert store.messages_for_session(session.id) == []
    assert console._last_console_action.action_id == "delete"
    assert console._last_console_action.visible_copy == "Deleted message from transcript."


@pytest.mark.asyncio
async def test_console_selected_message_edit_action_opens_modal_and_saves_content():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-edit-{message.id}")

        await pilot.click(f"#console-message-action-edit-{message.id}")
        await _wait_for_selector(host.screen_stack[-1], pilot, "#console-edit-message-modal")
        edit_modal = host.screen_stack[-1]
        assert "Editing existing transcript message" in _static_plain_text(
            edit_modal.query_one("#console-edit-message-context", Static)
        )

        editor = edit_modal.query_one("#console-edit-message-body", TextArea)
        assert editor.text == "answer"
        editor.text = "edited answer"
        await pilot.click("#console-edit-message-save")
        await pilot.pause()

    assert store.get_message(message.id).content == "edited answer"
    assert console._last_console_action.action_id == "edit"
    assert console._last_console_action.visible_copy == "Edited message."


@pytest.mark.asyncio
async def test_console_selected_message_edit_action_cancel_preserves_content():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-edit-{message.id}")

        await pilot.click(f"#console-message-action-edit-{message.id}")
        await _wait_for_selector(host.screen_stack[-1], pilot, "#console-edit-message-modal")
        edit_modal = host.screen_stack[-1]
        editor = edit_modal.query_one("#console-edit-message-body", TextArea)
        editor.text = "discard this"
        await pilot.click("#console-edit-message-cancel")
        await pilot.pause()

    assert store.get_message(message.id).content == "answer"


@pytest.mark.asyncio
async def test_console_selected_message_edit_action_blank_save_stays_open_with_error():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-edit-{message.id}")

        await pilot.click(f"#console-message-action-edit-{message.id}")
        await _wait_for_selector(host.screen_stack[-1], pilot, "#console-edit-message-modal")
        edit_modal = host.screen_stack[-1]
        editor = edit_modal.query_one("#console-edit-message-body", TextArea)
        editor.text = "   "
        await pilot.click("#console-edit-message-save")
        await _wait_for_selector(edit_modal, pilot, "#console-edit-message-error")

        assert "cannot be blank" in _static_plain_text(
            edit_modal.query_one("#console-edit-message-error", Static)
        ).lower()
        assert store.get_message(message.id).content == "answer"


@pytest.mark.asyncio
async def test_console_sync_skips_transcript_refresh_when_messages_unchanged(monkeypatch):
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        original_refresh = transcript.refresh_messages
        refresh_calls = 0

        async def counted_refresh():
            nonlocal refresh_calls
            refresh_calls += 1
            await original_refresh()

        monkeypatch.setattr(transcript, "refresh_messages", counted_refresh)

        await console._sync_native_console_chat_ui()
        assert refresh_calls == 1

        await console._sync_native_console_chat_ui()
        assert refresh_calls == 1

        store.add_variant(message.id, "updated answer")
        await console._sync_native_console_chat_ui()
        assert refresh_calls == 2


@pytest.mark.asyncio
async def test_console_selected_message_save_as_action_opens_modal():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-save-as-{message.id}")

        await pilot.click(f"#console-message-action-save-as-{message.id}")
        await _wait_for_selector(host.screen_stack[-1], pilot, "#console-save-as-modal")

    assert console._last_console_action.action_id == "save-as"


@pytest.mark.asyncio
async def test_console_selected_message_save_as_note_creates_note_from_message():
    app = _build_test_app()
    app.notes_scope_service = SimpleNamespace(
        save_note=AsyncMock(return_value={"id": "note-1", "title": "Console message", "content": "answer"})
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-save-as-{message.id}")

        await pilot.click(f"#console-message-action-save-as-{message.id}")
        await _wait_for_selector(host.screen_stack[-1], pilot, "#console-save-as-destination-note")
        await pilot.click("#console-save-as-destination-note")
        await pilot.pause()

    app.notes_scope_service.save_note.assert_awaited_once_with(
        scope="local_note",
        title="Console message",
        content="answer",
        note_id=None,
        version=None,
        user_id="default_user",
        workspace_id=None,
        keywords=["console"],
    )
    assert console._last_console_action.action_id == "save-as-note"
    assert console._last_console_action.visible_copy == "Saved message as Note."


@pytest.mark.asyncio
async def test_console_failed_stream_renders_inline_retry_and_recovers():
    gateway = FailThenRecoverGateway()
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "llama.cpp stream failed")

        store = console._ensure_console_chat_store()
        failed = store.messages_for_session(store.active_session_id)[-1]
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(failed.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-retry-{failed.id}")
        retry_button = console.query_one(f"#console-message-action-retry-{failed.id}", Button)
        assert str(retry_button.label) == "Try"
        assert retry_button.tooltip == "Retry the failed response."

        await pilot.click(f"#console-message-action-retry-{failed.id}")
        await _wait_for_text(console, pilot, "recovered")

    assert store.get_message(failed.id).status == "complete"


@pytest.mark.asyncio
async def test_console_continue_action_streams_new_message_from_selected_turn():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: CapturingGateway(chunks=("hel", "lo"))
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        _select_llamacpp_console(console)
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        source = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="seed",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(source.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-continue-{source.id}")

        await pilot.click(f"#console-message-action-continue-{source.id}")
        await _wait_for_text(console, pilot, "hello")

        messages = store.messages_for_session(session.id)
        assert messages[-1].role is ConsoleMessageRole.ASSISTANT
        assert messages[-1].content == "hello"
        assert messages[-1].id != source.id
        assert transcript.selected_message_id is None
        assert not list(console.query(f"#console-message-actions-{source.id}"))


@pytest.mark.asyncio
async def test_console_regenerate_action_streams_selected_variant():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: CapturingGateway(chunks=("hel", "lo"))
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        _select_llamacpp_console(console)
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        source = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="seed",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(source.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-regenerate-{source.id}")

        await pilot.click(f"#console-message-action-regenerate-{source.id}")
        await _wait_for_text(console, pilot, "hello")

        updated = store.get_message(source.id)
        assert updated.variants.current.content == "hello"
        assert updated.variants.can_go_previous is True


@pytest.mark.asyncio
async def test_console_native_tab_strip_creates_and_switches_sessions():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second}")
        assert "Chat 2" in _visible_text(console)

        await pilot.click(f"#console-session-tab-{first.id}")

        assert store.active_session_id == first.id
        assert "Chat 1" in _visible_text(console)


@pytest.mark.asyncio
async def test_console_native_tab_switch_restores_transcript_messages():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        store.append_message(
            first.id,
            role=ConsoleMessageRole.USER,
            content="first tab user prompt",
        )
        store.append_message(
            first.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="first tab assistant reply",
        )
        await console._sync_native_console_chat_ui()
        await _wait_for_text(console, pilot, "first tab assistant reply")

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second}")
        await _wait_for_text(console, pilot, "No messages yet.")
        assert "first tab assistant reply" not in _visible_text(console)

        await pilot.click(f"#console-session-tab-{first.id}")

        assert store.active_session_id == first.id
        await _wait_for_text(console, pilot, "first tab user prompt")
        await _wait_for_text(console, pilot, "first tab assistant reply")


@pytest.mark.asyncio
async def test_console_workspace_conversation_switch_restores_transcript_messages():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        store.append_message(
            first.id,
            role=ConsoleMessageRole.USER,
            content="workspace row user prompt",
        )
        store.append_message(
            first.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="workspace row assistant reply",
        )
        await console._sync_native_console_chat_ui()
        await _wait_for_text(console, pilot, "workspace row assistant reply")

        await _wait_for_selector(console, pilot, "#console-new-workspace-conversation")
        console.query_one("#console-new-workspace-conversation", Button).press()
        for _ in range(20):
            if store.active_session_id != first.id:
                break
            await pilot.pause(0.05)
        second = store.active_session_id
        assert second != first.id
        await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Chat 2",
            selected=True,
        )
        await _wait_for_text(console, pilot, "No messages yet.")
        assert "workspace row assistant reply" not in _visible_text(console)

        await _click_console_workspace_conversation_for_session(
            console,
            pilot,
            store,
            first.id,
        )

        assert store.active_session_id == first.id
        await _wait_for_text(console, pilot, "workspace row user prompt")
        await _wait_for_text(console, pilot, "workspace row assistant reply")


@pytest.mark.asyncio
async def test_console_new_chat_tab_appears_in_workspace_conversation_rail():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    service.link_membership(
        active_workspace.workspace_id,
        item_type="conversation",
        item_id="persisted-chat-1",
        role="workspace-thread",
        title="Chat 1",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        first.persisted_conversation_id = "persisted-chat-1"
        await console._sync_native_console_chat_ui()

        assert any("Chat 1" in text for text in _console_workspace_conversation_texts(console))

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second}")

        row_texts = _console_workspace_conversation_texts(console)
        assert any("Chat 1" in text for text in row_texts)
        assert any("Chat 2" in text for text in row_texts)
        assert any(text.startswith("> ") and "Chat 2" in text for text in row_texts)


@pytest.mark.asyncio
async def test_console_workspace_rail_new_conversation_creates_default_workspace_session():
    app = _build_test_app()
    service = app.workspace_registry_service
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        await _wait_for_selector(console, pilot, "#console-new-workspace-conversation")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        console.query_one("#console-new-workspace-conversation", Button).press()
        await pilot.pause()

        active_session = store.switch_session(store.active_session_id)
        assert active_session.id != first.id
        assert active_session.workspace_id == service.get_active_workspace().workspace_id
        row_texts = await _wait_for_workspace_conversation_text(
            console,
            pilot,
            active_session.title,
            selected=True,
        )
        assert any("Chat 2" in text for text in row_texts)
        assert all(
            "Workspace conversation creation lands in a later slice" not in text
            for text in row_texts
        )


@pytest.mark.asyncio
async def test_console_new_chat_tab_promotes_active_native_session_in_workspace_rail():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    for index in range(5):
        service.link_membership(
            active_workspace.workspace_id,
            item_type="conversation",
            item_id=f"persisted-chat-{index}",
            role="workspace-thread",
            title=f"Older chat {index + 1}",
        )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second}")

        row_texts = await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Chat 2",
            selected=True,
        )
        assert "Chat 2" in row_texts[0]
        assert row_texts[0].startswith("> ")


@pytest.mark.asyncio
async def test_console_workspace_rail_new_conversation_creates_default_workspace_session():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    assert active_workspace is not None
    assert active_workspace.workspace_id == DEFAULT_WORKSPACE_ID
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-new-workspace-conversation")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        await pilot.click("#console-new-workspace-conversation")
        second = store.active_session_id
        assert second != first.id

        active_session = next(
            session for session in store.sessions() if session.id == second
        )
        assert active_session.workspace_id == DEFAULT_WORKSPACE_ID
        row_texts = await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Chat 2",
            selected=True,
        )
        assert any(text.startswith("> ") and "Chat 2" in text for text in row_texts)
        assert "file tools disabled" in _visible_text(console)


@pytest.mark.asyncio
async def test_console_workspace_rail_new_conversation_stays_scoped_to_active_workspace():
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")
    service.set_active_workspace("ws-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-new-workspace-conversation")
        store = console._ensure_console_chat_store()
        first_session_id = store.active_session_id

        await pilot.click("#console-new-workspace-conversation")
        session_id = store.active_session_id
        assert session_id is not None
        assert session_id != first_session_id
        active_session = next(
            session for session in store.sessions() if session.id == session_id
        )
        assert active_session.workspace_id == "ws-a"
        active_title = active_session.title

        row_texts = await _wait_for_workspace_conversation_text(
            console,
            pilot,
            active_title,
            selected=True,
        )
        assert any(
            text.startswith("> ") and active_title in text
            for text in row_texts
        )

        console.query_one("#console-change-workspace", Button).press()
        modal_screen = await _wait_for_workspace_switcher_modal(host, pilot)
        switch_button = next(
            button
            for button in modal_screen.query(Button)
            if str(button.label) == "Workspace B"
        )
        switch_button.press()
        await _wait_for_console_screen(host, console, pilot)
        await _wait_for_selector(console, pilot, "#console-workspace-context")

        assert service.get_active_workspace().workspace_id == "ws-b"
        assert all(
            active_title not in row_text
            for row_text in _console_workspace_conversation_texts(console)
        )


@pytest.mark.asyncio
async def test_console_workspace_conversation_row_switches_native_session():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id
        await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Chat 1",
            selected=False,
        )
        await _click_console_workspace_conversation_for_session(console, pilot, store, first.id)

        assert store.active_session_id == first.id
        row_texts = await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Chat 1",
            selected=True,
        )
        assert any(text.startswith("> ") and "Chat 1" in text for text in row_texts)


@pytest.mark.asyncio
async def test_console_workspace_rail_keeps_active_native_session_visible_when_scope_is_global():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    service.link_membership(
        active_workspace.workspace_id,
        item_type="conversation",
        item_id="persisted-chat-1",
        role="workspace-thread",
        title="Chat 1",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1", workspace_id="global")
        first.persisted_conversation_id = "persisted-chat-1"
        second = store.create_session(title="Chat 2", workspace_id="global")
        await console._sync_native_console_chat_ui()

        assert store.active_session_id == second.id
        row_texts = await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Chat 2",
            selected=True,
        )
        assert any("Chat 1" in text for text in row_texts), row_texts


@pytest.mark.asyncio
async def test_console_new_chat_focuses_composer_for_immediate_typing():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second}")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        await pilot.press("n")
        await pilot.pause(0.1)

        assert console.app.focused is composer
        assert composer.draft_text() == "n"


@pytest.mark.asyncio
async def test_console_tab_switch_focuses_composer_for_immediate_typing():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()
        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second}")

        await pilot.click(f"#console-session-tab-{first.id}")
        assert store.active_session_id == first.id

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        await pilot.press("s")
        await pilot.pause(0.1)

        assert console.app.focused is composer
        assert composer.draft_text() == "s"


@pytest.mark.asyncio
async def test_console_native_tab_strip_isolates_composer_drafts():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("first tab draft")

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second}")

        assert composer.draft_text() == ""

        composer.load_draft("second tab draft")
        await pilot.click(f"#console-session-tab-{first.id}")
        assert store.active_session_id == first.id
        assert composer.draft_text() == "first tab draft"

        await pilot.click(f"#console-session-tab-{second}")
        assert store.active_session_id == second
        assert composer.draft_text() == "second tab draft"


@pytest.mark.asyncio
async def test_console_native_tab_strip_keeps_compact_close_x():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        close_selector = f"#console-close-session-tab-{first.id}"
        await _wait_for_selector(console, pilot, close_selector)
        close_button = console.query_one(close_selector, Button)

        assert close_button.label.plain == "x"
        assert 2 <= close_button.region.width <= 4

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        await _wait_for_selector(console, pilot, f"#console-close-session-tab-{second}")
        await pilot.click(f"#console-close-session-tab-{second}")

        assert store.active_session_id == first.id
        assert second not in {session.id for session in store.sessions()}


@pytest.mark.asyncio
async def test_console_native_tab_title_has_stable_visible_label_region():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.rename_session(session.id, "Planning session with a long descriptive name")
        await console._sync_native_console_chat_ui()

        tab_selector = f"#console-session-tab-{session.id}"
        await _wait_for_selector(console, pilot, tab_selector)
        tab = console.query_one(tab_selector, Button)

        assert tab.tooltip == (
            "Rename Console tab: Planning session with a long descriptive name"
        )
        assert str(tab.label) == "Planning session..."
        assert tab.region.width >= 18
        assert "Planning session" in _visible_text(console)


@pytest.mark.asyncio
async def test_console_native_active_tab_title_opens_rename_modal():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        assert not list(console.query(f"#console-rename-session-tab-{session.id}"))

        await pilot.click(f"#console-session-tab-{session.id}")
        modal_screen = await _wait_for_console_rename_modal(host, pilot)

        rename_input = modal_screen.query_one("#console-rename-session-title", Input)
        assert rename_input.value == "Chat 1"
        assert getattr(console.app.focused, "id", None) == rename_input.id

        await pilot.press(*"Planning")
        modal_screen.query_one("#console-rename-session-save", Button).press()
        await _wait_for_console_screen(host, console, pilot)
        await _wait_for_selector(console, pilot, f"#console-session-tab-{session.id}")

        assert store.sessions()[0].title == "Planning"
        assert "Planning" in _visible_text(console)
        assert not list(console.query(f"#console-session-rename-input-{session.id}"))


@pytest.mark.asyncio
async def test_console_native_rename_modal_buttons_are_not_clipped():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        await pilot.click(f"#console-session-tab-{session.id}")
        modal_screen = await _wait_for_console_rename_modal(host, pilot)

        action_row = modal_screen.query_one("#console-rename-session-actions")
        cancel_button = modal_screen.query_one("#console-rename-session-cancel", Button)
        save_button = modal_screen.query_one("#console-rename-session-save", Button)

        assert action_row.region.height >= 3
        assert cancel_button.region.height >= 3
        assert save_button.region.height >= 3
        assert str(cancel_button.label) == "Cancel"
        assert str(save_button.label) == "Save"


@pytest.mark.asyncio
async def test_console_native_tab_rename_escape_restores_existing_title():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        await pilot.click(f"#console-session-tab-{session.id}")

        modal_screen = await _wait_for_console_rename_modal(host, pilot)
        rename_input = modal_screen.query_one("#console-rename-session-title", Input)
        assert rename_input.value == "Chat 1"
        await pilot.press(*"Discarded")
        await pilot.press("escape")
        await _wait_for_console_screen(host, console, pilot)
        await _wait_for_selector(console, pilot, f"#console-session-tab-{session.id}")

        assert store.sessions()[0].title == "Chat 1"
        assert "Chat 1" in _visible_text(console)
        assert not list(console.query(f"#console-session-rename-input-{session.id}"))


@pytest.mark.asyncio
async def test_console_close_tab_with_messages_shows_confirmation():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        store.append_message(session.id, role=ConsoleMessageRole.USER, content="hello")
        store.create_session(title="Chat 2")
        await console._sync_native_console_chat_ui()

        close_selector = f"#console-close-session-tab-{session.id}"
        await _wait_for_selector(console, pilot, close_selector)
        await pilot.click(close_selector)

        from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

        for _ in range(20):
            await pilot.pause()
            if any(isinstance(s, ConfirmationDialog) for s in host.screen_stack):
                break

        dialog_screens = [s for s in host.screen_stack if isinstance(s, ConfirmationDialog)]
        assert len(dialog_screens) == 1, "confirmation dialog should appear for tab with messages"
        assert session.id in {s.id for s in store.sessions()}, "session not closed yet"

        await pilot.click("#confirm-button")
        for _ in range(10):
            await pilot.pause()

        assert session.id not in {s.id for s in store.sessions()}, "session closed after confirm"
