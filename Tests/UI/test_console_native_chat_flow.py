import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)
from textual.widgets import Button

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole, ConsoleRunStatus
from tldw_chatbook.Widgets.Console import ConsoleComposerBar, ConsoleTranscript
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


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
    async def resolve_for_send(self, selection):
        await asyncio.sleep(0.15)
        return await super().resolve_for_send(selection)


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


class FailThenRecoverGateway(_ReadyResolutionGateway):
    def __init__(self) -> None:
        self.calls = 0

    async def stream_chat(self, resolution, messages):
        self.calls += 1
        if self.calls == 1:
            yield "partial"
            raise RuntimeError("llama.cpp stream failed")
        yield "recovered"


async def _wait_for_text(screen, pilot, expected: str, *, attempts: int = 20) -> None:
    for _ in range(attempts):
        if expected in _visible_text(screen):
            return
        await pilot.pause(0.05)
    raise AssertionError(f"Text not found: {expected!r}. Visible text: {_visible_text(screen)!r}")


def _select_llamacpp_console(console: ChatScreen) -> None:
    """Select the native llama.cpp path after mounted controls initialize."""
    console._console_control_provider = "llama_cpp"
    console._console_control_model = "test-model"
    console._sync_console_control_bar()


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
    assert selection.workspace_context.active_workspace_id == "global"


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
async def test_console_composer_stop_is_disabled_when_idle():
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

        assert stop_button.disabled is True
        assert stop_button.has_class("console-action-disabled")
        assert stop_button.has_class("console-stop-idle")
        assert not stop_button.has_class("console-stop-active")

        composer.load_draft("hello")
        console.query_one("#console-send-message", Button).press()
        await asyncio.wait_for(gateway.started.wait(), timeout=1)
        await _wait_for_text(console, pilot, "partial")

        assert send_button.disabled is True
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
        assert send_button.disabled is True
        assert send_button.has_class("console-send-blocked")
        send_button.press()
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
        for _ in range(20):
            if (
                console._ensure_console_chat_controller().run_state.status
                is ConsoleRunStatus.VALIDATING
            ):
                break
            await pilot.pause(0.01)
        console._sync_console_control_bar()
        send_button = console.query_one("#console-send-message", Button)
        stop_button = console.query_one("#console-stop-generation", Button)

        assert send_button.disabled is True
        assert send_button.has_class("console-action-disabled")
        assert send_button.has_class("console-send-blocked")
        assert not send_button.has_class("console-action-primary")
        assert stop_button.disabled is True
        assert stop_button.has_class("console-stop-idle")

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
async def test_console_unsupported_provider_block_renders_one_normalized_system_message():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        controller = console._ensure_console_chat_controller()
        controller.provider = "openai"
        controller.model = "test-model"

        await console._submit_console_native_draft("hello")
        await _wait_for_text(console, pilot, "Provider blocked: WIP")

        store = console._ensure_console_chat_store()
        messages = store.messages_for_session(store.active_session_id)
        system_messages = [message.content for message in messages if message.role is ConsoleMessageRole.SYSTEM]
        assert system_messages == [
            "Provider blocked: WIP: Console native provider 'openai' is not wired yet. "
            "Select llama.cpp for this slice."
        ]
        assert controller.run_state.visible_copy == system_messages[0]


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
        await _wait_for_selector(app, pilot, "#console-save-as-modal")

    assert console._last_console_action.action_id == "save-as"


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
        assert "Retry" in _visible_text(console)

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
