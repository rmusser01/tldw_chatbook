"""Console message Delete states its real scope on the row and can be undone.

TASK-33628.2 (Console UX review 2026-09-29, finding G1-02). Delete removes
the selected message AND every later message beneath it. Before this task
the first activation only recorded a blocked action that the collapsed
Inspector showed, the receipt said "Deleted message from transcript." in
the singular whatever the count, nothing could bring the rows back, and a
media-free delete raised a spurious recovered-media cleanup warning.

Every case drives the real ``ChatScreen`` with the real Console store bound
to a real ``ChatPersistenceService`` over an in-memory ChaChaNotes database,
and reads what was PAINTED (``Screen._compositor.render_strips()``), not
only ``_last_console_action``.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import attach_chachanotes_db
from Tests.UI.test_console_native_chat_flow import _wait_for_selector
from Tests.UI.test_destination_shells import _build_test_app
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_conversation_hydration import (
    console_messages_from_conversation_tree,
)
from tldw_chatbook.Widgets.Console import ConsoleTranscript

_TURNS = 5
_CLEANUP_WARNING = "recovered-media reference cleanup is pending"


def _painted(host: Any) -> str:
    """Return the text actually painted on the active screen."""
    return "\n".join(strip.text for strip in host.screen._compositor.render_strips())


async def _wait_until(pilot: Any, predicate: Any, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            await pilot.pause()
            return
        await pilot.pause(0.05)
    assert predicate()


def _deleted_flags(db: Any, conversation_id: str) -> dict[str, int]:
    with db.transaction() as conn:
        rows = conn.execute(
            "SELECT id, deleted FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchall()
    return {row["id"]: int(row["deleted"]) for row in rows}


def _tree_nodes(db: Any, conversation_id: str) -> list[Any]:
    tree = ChatConversationService(db).get_conversation_tree(
        conversation_id, depth_cap=10_000, root_limit=10_000
    )
    return console_messages_from_conversation_tree(tree, db=db)


async def _open_persisted_conversation(console: Any, db: Any) -> dict[str, Any]:
    """Persist five turns and open them in the mounted Console."""
    conversation_id = ChatConversationService(db).create_conversation(
        id="delete-undo-conversation",
        title="Delete undo",
        scope_type="global",
        state="in-progress",
    )
    persisted: list[str] = []
    parent = None
    for index in range(_TURNS * 2):
        role = "user" if index % 2 == 0 else "assistant"
        parent = db.add_message(
            {
                "id": f"delete-undo-{index}",
                "conversation_id": conversation_id,
                "parent_message_id": parent,
                "sender": role,
                "role": role,
                "content": f"Turn {index // 2 + 1} {role} text",
                "timestamp": f"2026-09-30T00:00:{index:02d}.000000+00:00",
            }
        )
        persisted.append(parent)
    assert db.set_conversation_active_cursor(
        conversation_id, active_leaf_message_id=persisted[-1], before_message_id=None
    )
    store = console._ensure_console_chat_store()
    assert isinstance(store.persistence, ChatPersistenceService)
    assert store.persistence.db is db
    session = store.restore_persisted_session(
        title="Delete undo",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=_tree_nodes(db, conversation_id),
        active_leaf_persisted_id=persisted[-1],
    )
    await console._sync_native_console_chat_ui()
    native = {
        message.persisted_message_id: message.id
        for message in store.messages_for_session(session.id)
    }
    return {
        "conversation_id": conversation_id,
        "persisted": persisted,
        "native": [native[message_id] for message_id in persisted],
        "session_id": session.id,
        "store": store,
    }


async def _arm_delete_from_more(console: Any, pilot: Any, message_id: str) -> None:
    """Select one row, then choose More… › Delete through the real menu."""
    transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
    transcript.select_message(message_id)
    await console._sync_native_console_chat_ui()
    opener = f"#console-message-action-more-{message_id}"
    await _wait_for_selector(console, pilot, opener)
    console.query_one(opener, Button).press()
    await _wait_for_selector(console, pilot, "#console-message-more-delete")
    console.query_one("#console-message-more-delete", Button).press()
    await pilot.pause()
    await console._sync_native_console_chat_ui()
    await pilot.pause()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (235, 52)])
async def test_delete_confirms_scope_on_row_then_undo_restores_exact_subtree(size):
    """AC#1/#3/#4/#6: scoped in-row confirmation, counted receipt, exact Undo."""
    app = _build_test_app()
    db = attach_chachanotes_db(app)
    notices: list[tuple[str, dict[str, Any]]] = []
    app.notify = lambda message, **kwargs: notices.append((str(message), kwargs))
    host = ConsoleHarness(app)

    async with host.run_test(size=size) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        seeded = await _open_persisted_conversation(console, db)
        store = seeded["store"]
        session_id = seeded["session_id"]
        conversation_id = seeded["conversation_id"]
        original_ids = [m.id for m in store.messages_for_session(session_id)]
        assert original_ids == seeded["native"]
        target = seeded["native"][2]  # Turn 2's user prompt: 8 messages from here.

        await _arm_delete_from_more(console, pilot, target)

        confirm = f"#console-message-action-delete-confirm-{target}"
        cancel = f"#console-message-action-delete-cancel-{target}"
        await _wait_for_selector(console, pilot, confirm)
        await _wait_until(
            pilot,
            lambda: "Delete this message and 7 later messages?" in _painted(host),
        )
        painted = _painted(host)
        # The scope and both controls are painted on the selected row itself;
        # the Inspector is collapsed and was never opened.
        assert "Delete 8 messages" in painted
        assert "Cancel" in painted
        for selector in (confirm, cancel):
            button = console.query_one(selector, Button)
            assert button.region.area > 0
            assert host.screen.region.contains_region(button.region)
        assert [m.id for m in store.messages_for_session(session_id)] == original_ids
        assert set(_deleted_flags(db, conversation_id).values()) == {0}

        console.query_one(confirm, Button).press()
        await _wait_until(pilot, lambda: bool(host.screen.query("#console-delete-receipt")))
        receipt_text = _painted(host)
        assert "Deleted 8 messages" in receipt_text
        assert "Deleted message from transcript." not in receipt_text
        flags = _deleted_flags(db, conversation_id)
        assert [flags[pid] for pid in seeded["persisted"]] == [0, 0] + [1] * 8
        assert [m.id for m in store.messages_for_session(session_id)] == (
            original_ids[:2]
        )

        host.screen.query_one("#console-delete-receipt-undo", Button).press()
        await _wait_until(
            pilot,
            lambda: set(_deleted_flags(db, conversation_id).values()) == {0},
        )
        await _wait_until(
            pilot,
            lambda: [m.id for m in store.messages_for_session(session_id)]
            == original_ids,
        )
        assert db.get_conversation_active_cursor(conversation_id) == (
            seeded["persisted"][-1],
            None,
        )
        await console._sync_native_console_chat_ui()
        await pilot.pause()
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        assert [m.id for m in transcript._messages if m.id in original_ids] == (
            original_ids
        )

    assert not any(_CLEANUP_WARNING in message for message, _ in notices), notices

    # Reopen the conversation from storage in a fresh store: the restored
    # rows and the previous active branch are durable, not just in memory.
    reopened = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = reopened.restore_persisted_session(
        title="Delete undo",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=_tree_nodes(db, conversation_id),
        active_leaf_persisted_id=db.get_conversation_active_cursor(conversation_id)[0],
    )
    assert [
        m.persisted_message_id for m in reopened.messages_for_session(session.id)
    ] == seeded["persisted"]


@pytest.mark.asyncio
async def test_cancel_or_moving_selection_clears_pending_delete():
    """AC#2: Cancel, or selecting another message, removes nothing."""
    app = _build_test_app()
    db = attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        seeded = await _open_persisted_conversation(console, db)
        store = seeded["store"]
        session_id = seeded["session_id"]
        original_ids = [m.id for m in store.messages_for_session(session_id)]
        target, other = seeded["native"][4], seeded["native"][1]
        confirm = f"#console-message-action-delete-confirm-{target}"

        await _arm_delete_from_more(console, pilot, target)
        await _wait_for_selector(console, pilot, confirm)
        assert "Delete this message and 5 later messages?" in _painted(host)

        console.query_one(f"#console-message-action-delete-cancel-{target}", Button).press()
        await _wait_until(pilot, lambda: not console.query(confirm))
        await _wait_for_selector(console, pilot, f"#console-message-action-more-{target}")
        assert "later messages?" not in _painted(host)
        assert console._pending_console_delete_message_id is None

        # Moving the selection away -- through the transcript's own selection
        # path, with no extra screen sync -- clears it; coming back shows the
        # ordinary row, not a stale confirmation (caught live at 80x24).
        await _arm_delete_from_more(console, pilot, target)
        await _wait_for_selector(console, pilot, confirm)
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(other)
        await _wait_until(pilot, lambda: not console.query(confirm))
        assert console._pending_console_delete_message_id is None
        transcript.select_message(target)
        await _wait_for_selector(console, pilot, f"#console-message-action-more-{target}")
        assert not console.query(confirm)
        assert "later messages?" not in _painted(host)

        # Esc on the focused confirmation clears the selection, and with it
        # the pending delete.
        await _arm_delete_from_more(console, pilot, target)
        await _wait_for_selector(console, pilot, confirm)
        cancel = f"#console-message-action-delete-cancel-{target}"
        await _wait_until(pilot, lambda: console.query_one(cancel, Button).has_focus)
        await pilot.press("escape")
        await _wait_until(pilot, lambda: not console.query(confirm))
        transcript.select_message(target)
        await _wait_for_selector(console, pilot, f"#console-message-action-more-{target}")
        assert not console.query(confirm)
        assert console._pending_console_delete_message_id is None

        assert [m.id for m in store.messages_for_session(session_id)] == original_ids
        assert set(_deleted_flags(db, seeded["conversation_id"]).values()) == {0}
