"""Composer command interception + unknown-command Enter-again (Task 10)."""

from unittest.mock import AsyncMock

import pytest
from textual.widgets import Button

from Tests.UI.test_console_native_chat_flow import (
    CapturingGateway,
    _configure_native_ready_console,
    _wait_for_text,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import ConsoleHarness
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


UNKNOWN_NOPE_HINT = (
    "Unknown command /nope — available: /prompt, /system. "
    "Press Enter again to send as text."
)
UNKNOWN_NADA_HINT = (
    "Unknown command /nada — available: /prompt, /system. "
    "Press Enter again to send as text."
)


def _system_message_contents(console) -> list[str]:
    store = console._ensure_console_chat_store()
    if store.active_session_id is None:
        return []
    messages = store.messages_for_session(store.active_session_id)
    return [message.content for message in messages if message.role is ConsoleMessageRole.SYSTEM]


async def _spy_submit_draft(console) -> AsyncMock:
    """Wrap the active controller's ``submit_draft`` so real sends still work."""
    controller = console._ensure_console_chat_controller()
    spy = AsyncMock(wraps=controller.submit_draft)
    controller.submit_draft = spy
    return spy


@pytest.mark.asyncio
async def test_console_unknown_command_first_enter_renders_hint_and_does_not_send():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/nope x")
        submit_spy = await _spy_submit_draft(console)

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, UNKNOWN_NOPE_HINT)

        assert composer.draft_text() == "/nope x"
        submit_spy.assert_not_called()
        assert console._console_unknown_send_armed == "/nope x"


@pytest.mark.asyncio
async def test_console_unknown_command_second_unmodified_enter_sends_as_text():
    gateway = CapturingGateway()
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/nope x")
        submit_spy = await _spy_submit_draft(console)
        send_button = console.query_one("#console-send-message", Button)

        send_button.press()
        await _wait_for_text(console, pilot, UNKNOWN_NOPE_HINT)
        submit_spy.assert_not_called()

        send_button.press()
        await _wait_for_text(console, pilot, "accepted")

        submit_spy.assert_called_once_with("/nope x")
        assert gateway.sent_messages[-1][-1]["content"] == "/nope x"
        assert console._console_unknown_send_armed is None


@pytest.mark.asyncio
async def test_console_unknown_command_edit_between_enters_re_hints_and_does_not_send():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/nope x")
        submit_spy = await _spy_submit_draft(console)
        send_button = console.query_one("#console-send-message", Button)

        send_button.press()
        await _wait_for_text(console, pilot, UNKNOWN_NOPE_HINT)
        assert console._console_unknown_send_armed == "/nope x"

        # Edit the draft to a different unknown command between Enters.
        composer.load_draft("/nada y")
        await pilot.pause()
        assert console._console_unknown_send_armed is None

        send_button.press()
        await _wait_for_text(console, pilot, UNKNOWN_NADA_HINT)

        submit_spy.assert_not_called()
        assert composer.draft_text() == "/nada y"
        assert console._console_unknown_send_armed == "/nada y"
        contents = _system_message_contents(console)
        assert contents.count(UNKNOWN_NOPE_HINT) == 1
        assert contents.count(UNKNOWN_NADA_HINT) == 1


@pytest.mark.asyncio
async def test_console_unknown_command_roundtrip_edit_back_to_armed_text_requires_fresh_arm():
    """Editing away and back to the armed text still disarms (Task 10 hardening).

    Comparing the armed snapshot to the current draft text alone would let a
    user edit away from an armed unknown draft and back to the exact same
    text, then have an unrelated second Enter silently send it. The composer
    change subscription must disarm on *any* edit, not just a text mismatch.
    """
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/nope x")
        submit_spy = await _spy_submit_draft(console)
        send_button = console.query_one("#console-send-message", Button)

        send_button.press()
        await _wait_for_text(console, pilot, UNKNOWN_NOPE_HINT)
        assert console._console_unknown_send_armed == "/nope x"

        composer.load_draft("/nope xy")
        composer.load_draft("/nope x")
        await pilot.pause()
        assert console._console_unknown_send_armed is None

        send_button.press()
        await pilot.pause(0.1)

        submit_spy.assert_not_called()
        assert composer.draft_text() == "/nope x"
        assert console._console_unknown_send_armed == "/nope x"


@pytest.mark.asyncio
async def test_console_collapsed_paste_starting_with_slash_sends_normally():
    gateway = CapturingGateway()
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)
    pasted_text = "/nope " + ("x" * 80)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_pasted_text(pasted_text)
        assert composer.has_paste_segments()
        submit_spy = await _spy_submit_draft(console)

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "accepted")

        submit_spy.assert_called_once_with(pasted_text)
        assert gateway.sent_messages[-1][-1]["content"] == pasted_text
        assert console._console_unknown_send_armed is None
        assert composer.draft_text() == ""


@pytest.mark.asyncio
async def test_console_prompt_command_dispatches_insert_prompt_stub():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/prompt")
        submit_spy = await _spy_submit_draft(console)
        insert_prompt_spy = AsyncMock()
        console._console_command_insert_prompt = insert_prompt_spy

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.1)

        insert_prompt_spy.assert_called_once()
        called_parse = insert_prompt_spy.call_args.args[0]
        assert called_parse.name == "prompt"
        submit_spy.assert_not_called()
        assert composer.draft_text() == "/prompt"


@pytest.mark.asyncio
async def test_console_system_command_dispatches_apply_system_stub():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/system helpful")
        submit_spy = await _spy_submit_draft(console)
        apply_system_spy = AsyncMock()
        console._console_command_apply_system = apply_system_spy

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.1)

        apply_system_spy.assert_called_once()
        called_parse = apply_system_spy.call_args.args[0]
        assert called_parse.name == "system"
        assert called_parse.args == "helpful"
        submit_spy.assert_not_called()
        assert composer.draft_text() == "/system helpful"
