import asyncio
from types import SimpleNamespace

import pytest

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)
from textual.widgets import Button

from tldw_chatbook.Widgets.Console import ConsoleComposerBar
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
        ("http://127.0.0.1:9099/completion", "http://127.0.0.1:9099"),
        ("http://127.0.0.1:9099/", "http://127.0.0.1:9099"),
        (None, "http://127.0.0.1:9099"),
    ),
)
def test_console_llamacpp_base_url_normalizes_openai_compatible_endpoints(raw_url, expected):
    screen = ChatScreen(_build_test_app())

    assert screen._normalize_llamacpp_base_url(raw_url) == expected


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
        console.query_one("#console-send-message", Button).press()
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
        await _wait_for_text(console, pilot, "Assistant: hello")

        assert composer.draft_text() == ""
        store = console._ensure_console_chat_store()
        messages = store.messages_for_session(store.active_session_id)
        assert messages[-2].content == "hello"
        assert messages[-1].content == "hello"
