import pytest
from textual.widgets import Button, Input, Select

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="Gate 1.5 Task 2/3 will replace the full legacy ChatWindowEnhanced chrome.",
    strict=True,
)
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
@pytest.mark.xfail(
    reason="Gate 1.5 Task 3 will expose a native Console composer.",
    strict=True,
)
async def test_console_gate15_keeps_existing_chat_send_control_reachable():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        text = _visible_text(console)
        assert "Send" in text
        send_controls = [
            button
            for button in console.query(Button)
            if (button.id or "").startswith("send-stop-chat")
            or button.has_class("console-send-button")
        ]
        assert send_controls


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
