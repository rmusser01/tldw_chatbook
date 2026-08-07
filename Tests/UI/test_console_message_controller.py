"""Characterisation tests for the Console message cluster (wave-3 task 1).

Run GREEN against unmodified `ChatScreen` before `ConsoleMessageController`
exists (`tldw_chatbook/UI/Console_Modules/message.py`) -- see
`.superpowers/sdd/2026-08-06-console-decomposition-wave3/progress.md` for the
mandatory characterise-before-extract sequence this file satisfies. Every
assertion below reads the ACTUAL persisted result (`ConsoleChatStore` rows),
never widget/DOM state, and drives the real screen methods end-to-end --
nothing here is monkeypatched.

Covers, at minimum, the two surfaces the wave-3 brief calls out explicitly:
the send/receive path (`test_console_message_send_persists_user_and_
assistant_rows`) and at least one `handle_console_message_action` branch
(delete + feedback-up below). Also pins the pure serialize/restore round
trip and the sibling-variant navigation helper, since both move as part of
the same cluster and have no other direct coverage of their post-move
public surface (`ChatScreen._serialize_console_message` /
`ChatScreen._restore_console_message` / `ChatScreen._select_console_message_
variant`, all still reachable under their pre-move names after the move --
see the extraction report's delegation table).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.widgets import Button

from Tests.UI.test_destination_shells import _build_test_app
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from Tests.UI.test_console_native_chat_flow import (
    CapturingGateway,
    _configure_native_ready_console,
    _select_llamacpp_console,
    _wait_for_selector,
    _wait_for_text,
)

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console import ConsoleComposerBar, ConsoleTranscript


@pytest.mark.asyncio
async def test_console_message_send_persists_user_and_assistant_rows():
    """Send/receive path: a real send queues a user turn and persists the
    streamed assistant reply as store rows, not just visible text."""
    gateway = CapturingGateway(chunks=("hello ", "there"))
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("characterisation probe")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "hello there")

        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        assert session_id is not None
        messages = store.messages_for_session(session_id)

    user_rows = [m for m in messages if m.role == ConsoleMessageRole.USER]
    assistant_rows = [m for m in messages if m.role == ConsoleMessageRole.ASSISTANT]
    assert any(row.content == "characterisation probe" for row in user_rows)
    assert any(
        row.content == "hello there" and row.status == "complete"
        for row in assistant_rows
    )


@pytest.mark.asyncio
async def test_console_message_action_delete_removes_persisted_row():
    """`handle_console_message_action`'s delete branch: two presses actually
    remove the row from the store (persisted result), not just the widget."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="answer"
        )
        await console._sync_native_console_chat_ui()
        # `_sync_console_pending_delete_confirmation` resets the armed id the
        # moment it disagrees with the transcript's own selection -- an
        # unselected row would silently disarm between the two presses below.
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()

        event = SimpleNamespace(
            button=SimpleNamespace(
                id=f"console-message-action-delete-{message.id}"
            ),
            stop=Mock(),
        )
        # First press only arms the confirmation -- nothing removed yet.
        handled = await console.handle_console_message_action(event)
        assert handled is True
        assert message in store.messages_for_session(session.id)

        # Second press on the same id actually deletes.
        handled_again = await console.handle_console_message_action(event)
        assert handled_again is True

    assert message not in store.messages_for_session(session.id)


@pytest.mark.asyncio
async def test_console_message_action_feedback_persists_to_store():
    """`handle_console_message_action`'s feedback branch writes through to
    the store's own `feedback` field, not just `_last_console_action`."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="answer"
        )
        await console._sync_native_console_chat_ui()

        event = SimpleNamespace(
            button=SimpleNamespace(
                id=f"console-message-action-feedback-up-{message.id}"
            ),
            stop=Mock(),
        )
        handled = await console.handle_console_message_action(event)
        assert handled is True

    assert store.get_message(message.id).feedback == "up"


def test_console_message_serialize_restore_round_trip():
    """Pure (de)serialization: a round trip through the screen-state payload
    shape preserves role/content/status/persisted id."""
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="round trip me",
    )

    payload = ChatScreen._serialize_console_message(message)
    assert payload["role"] == "user"
    assert payload["content"] == "round trip me"

    restored = ChatScreen._restore_console_message(payload)
    assert restored is not None
    assert restored.role is ConsoleMessageRole.USER
    assert restored.content == "round trip me"
    assert restored.status == "complete"


def test_console_message_select_variant_moves_active_leaf():
    """`_select_console_message_variant` moves the store's active leaf to
    the target sibling -- persisted store state, not a transcript selection.

    Mirrors `Tests/Chat/test_console_sibling_nav.py`'s established
    unmounted-`ChatScreen(app)` pattern: the method under test only touches
    `self._ensure_console_chat_store()`, so no Textual mount is required.
    """
    app = _build_test_app()
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session(title="t")
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    first = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="first"
    )
    second = store.create_sibling(
        first.id, role=ConsoleMessageRole.ASSISTANT, content="second"
    )
    assert store.active_leaf(session.id) == second.id

    target = screen._select_console_message_variant(
        second.id, direction="variant-previous"
    )

    assert target == first.id
    assert store.active_leaf(session.id) == first.id
