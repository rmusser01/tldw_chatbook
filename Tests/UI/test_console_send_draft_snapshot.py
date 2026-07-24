"""TASK-340: Enter must snapshot the draft synchronously at the keypress.

The Console Enter branch posts a ``Button.Pressed`` message and the send
handler used to read ``composer.draft_text()`` only when that message was
finally processed — printable keys handled in between mutated the draft and
were folded into the sent message (UX review finding
j6-send-captures-late-keystrokes). These tests deliver Enter synchronously
via ``ChatScreen.on_key`` and interleave typing before the message pump runs,
which is exactly the interleave a fast typist produces.
"""

import pytest
from textual.events import Key

from Tests.UI.test_console_native_chat_flow import (
    BlockedGateway,
    _select_llamacpp_console,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Widgets.Console import ConsoleComposerBar

DUMMY_OPENAI_API_KEY = "DUMMY_OPENAI_API_KEY"


async def _wait_for_text(console, pilot, needle: str, tries: int = 40) -> None:
    for _ in range(tries):
        if needle in _visible_text(console):
            return
        await pilot.pause(0.05)
    raise AssertionError(f"timed out waiting for {needle!r}")


def _ready_openai_app(monkeypatch, reply: str):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": {"api_key": DUMMY_OPENAI_API_KEY}}

    def fake_chat_api_call(**_kwargs):
        return reply

    monkeypatch.setattr(
        "tldw_chatbook.Chat.Chat_Functions.chat_api_call",
        fake_chat_api_call,
    )
    return app


def _press_enter_synchronously(console) -> None:
    """Deliver Enter to the screen key handler without pumping messages."""
    console.on_key(Key(key="enter", character="\r"))


@pytest.mark.asyncio
async def test_console_enter_snapshots_draft_before_late_keystrokes(monkeypatch):
    app = _ready_openai_app(monkeypatch, "snapshot reply")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        composer.load_draft("line one")

        _press_enter_synchronously(console)
        # Keystrokes arriving before the Button.Pressed message is processed:
        composer.insert_text("line two")

        await _wait_for_text(console, pilot, "snapshot reply")

        store = console._ensure_console_chat_store()
        messages = store.messages_for_session(store.active_session_id)
        user_messages = [
            m for m in messages if m.role is ConsoleMessageRole.USER
        ]
        assert user_messages[-1].content == "line one"
        # The late keystrokes belong to the NEXT draft — and the
        # accepted-submit clear must not eat them either.
        assert composer.draft_text() == "line two"


@pytest.mark.asyncio
async def test_console_blocked_send_restores_snapshot_before_late_keystrokes():
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
        composer.focus()
        await pilot.pause()
        composer.load_draft("keep me")

        _press_enter_synchronously(console)
        composer.insert_text("!")
        await _wait_for_text(console, pilot, "Provider blocked")
        await pilot.pause()

        # Blocked send restores the snapshot ahead of the late typing —
        # original text first, later keystrokes appended after it.
        assert composer.draft_text() == "keep me!"


@pytest.mark.asyncio
async def test_console_unknown_command_hint_restores_draft(monkeypatch):
    app = _ready_openai_app(monkeypatch, "never sent")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        composer.load_draft("/nosuchcommand")

        _press_enter_synchronously(console)
        await pilot.pause()
        await pilot.pause()

        # The unknown-command hint path must put the draft back so the
        # armed second-Enter flow still compares against the same text.
        assert composer.draft_text() == "/nosuchcommand"


@pytest.mark.asyncio
async def test_console_blocked_send_restore_preserves_paste_segments():
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
        composer.focus()
        await pilot.pause()
        composer.insert_text_as_paste("pasted payload " * 20)
        composer.insert_text(" tail")
        # Unfurl the token fully (collapsed -> confirm -> expanded); Enter
        # only sends once no token is awaiting the unfurl flow, and expanded
        # segments retain their paste provenance.
        assert composer.activate_focused_paste_token()
        assert composer.activate_focused_paste_token()
        assert not composer.activate_focused_paste_token()
        assert composer.has_paste_segments()
        expected = composer.draft_text()

        _press_enter_synchronously(console)
        await _wait_for_text(console, pilot, "Provider blocked")
        await pilot.pause()

        assert composer.draft_text() == expected
        assert composer.has_paste_segments()


@pytest.mark.asyncio
async def test_console_double_enter_sends_once_and_loses_nothing(monkeypatch):
    """A second Enter before the first Pressed handler runs must not
    overwrite the pending stash with None (that ate the message)."""
    app = _ready_openai_app(monkeypatch, "double reply")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        composer.load_draft("line one")

        _press_enter_synchronously(console)
        _press_enter_synchronously(console)

        await _wait_for_text(console, pilot, "double reply")

        store = console._ensure_console_chat_store()
        messages = store.messages_for_session(store.active_session_id)
        user_messages = [m for m in messages if m.role is ConsoleMessageRole.USER]
        assert [m.content for m in user_messages] == ["line one"]
        assert composer.draft_text() == ""


@pytest.mark.asyncio
async def test_console_submit_exception_restores_draft_and_keeps_app_alive(
    monkeypatch,
):
    """If submit_draft raises, the keypress-cleared draft must come back."""
    app = _ready_openai_app(monkeypatch, "never used")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        controller = console._ensure_console_chat_controller()

        async def exploding_submit(draft):
            raise RuntimeError("provider imploded")

        monkeypatch.setattr(controller, "submit_draft", exploding_submit)
        composer.load_draft("precious draft")

        _press_enter_synchronously(console)
        for _ in range(10):
            await pilot.pause(0.05)

        assert composer.draft_text() == "precious draft"
        # App survived the worker exception (queries still work).
        assert console.query_one("#console-native-composer", ConsoleComposerBar)
