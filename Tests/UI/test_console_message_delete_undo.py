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

# The real ChatScreen/store goes through config-participant admission, which the
# per-test sandbox refuses (RecoveryRequired); keep the collection-time profile.
pytestmark = pytest.mark.bootstrap_profile

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
        await _wait_until(
            pilot, lambda: bool(host.screen.query("#console-delete-receipt"))
        )
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
            lambda: (
                [m.id for m in store.messages_for_session(session_id)] == original_ids
            ),
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

        console.query_one(
            f"#console-message-action-delete-cancel-{target}", Button
        ).press()
        await _wait_until(pilot, lambda: not console.query(confirm))
        await _wait_for_selector(
            console, pilot, f"#console-message-action-more-{target}"
        )
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
        await _wait_for_selector(
            console, pilot, f"#console-message-action-more-{target}"
        )
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
        await _wait_for_selector(
            console, pilot, f"#console-message-action-more-{target}"
        )
        assert not console.query(confirm)
        assert console._pending_console_delete_message_id is None

        assert [m.id for m in store.messages_for_session(session_id)] == original_ids
        assert set(_deleted_flags(db, seeded["conversation_id"]).values()) == {0}


async def _open_rows(
    console: Any, db: Any, rows: list[tuple[str, str, str | None]], leaf: str
) -> dict[str, Any]:
    """Persist ``(id, role, parent)`` rows as one tree and open it."""
    conversation_id = ChatConversationService(db).create_conversation(
        id=f"delete-undo-{leaf}",
        title="Delete undo",
        scope_type="global",
        state="in-progress",
    )
    for index, (message_id, role, parent) in enumerate(rows):
        db.add_message(
            {
                "id": message_id,
                "conversation_id": conversation_id,
                "parent_message_id": parent,
                "sender": role,
                "role": role,
                "content": f"{message_id} text",
                "timestamp": f"2026-09-30T00:00:{index:02d}.000000+00:00",
            }
        )
    assert db.set_conversation_active_cursor(
        conversation_id, active_leaf_message_id=leaf, before_message_id=None
    )
    store = console._ensure_console_chat_store()
    session = store.restore_persisted_session(
        title="Delete undo",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=_tree_nodes(db, conversation_id),
        active_leaf_persisted_id=leaf,
    )
    await console._sync_native_console_chat_ui()
    native = {
        node.persisted_message_id: node.id
        for node in store._nodes_by_session[session.id].values()
    }
    return {
        "conversation_id": conversation_id,
        "native": native,
        "session_id": session.id,
        "store": store,
    }


def _tree_shape(store: Any, session_id: str) -> dict[str, Any]:
    """Return the store's tree registration for one session."""
    return {
        "children": {
            parent: list(children)
            for parent, children in store._children_by_parent[session_id].items()
        },
        "parents": {
            node_id: store._native_parent_by_message.get(node_id)
            for node_id in store._nodes_by_session[session_id]
        },
        "leaf": store._active_leaf_by_session.get(session_id),
        "path": list(store.active_path_message_ids(session_id)),
    }


async def _confirm_delete(console: Any, pilot: Any, host: Any, message_id: str) -> None:
    await _arm_delete_from_more(console, pilot, message_id)
    confirm = f"#console-message-action-delete-confirm-{message_id}"
    await _wait_for_selector(console, pilot, confirm)
    console.query_one(confirm, Button).press()
    await _wait_until(pilot, lambda: bool(host.screen.query("#console-delete-receipt")))


@pytest.mark.asyncio
async def test_confirm_rearms_when_the_subtree_changed_while_pending():
    """Confirm deletes only the scope the user saw; a changed subtree re-asks."""
    app = _build_test_app()
    db = attach_chachanotes_db(app)
    notices: list[str] = []
    app.notify = lambda message, **kwargs: notices.append(str(message))
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        rows = [
            (f"m{i}", "user" if i % 2 == 0 else "assistant", f"m{i - 1}" if i else None)
            for i in range(6)
        ]
        seeded = await _open_rows(console, db, rows, "m5")
        native, store = seeded["native"], seeded["store"]
        confirm = f"#console-message-action-delete-confirm-{native['m2']}"

        await _arm_delete_from_more(console, pilot, native["m2"])
        await _wait_for_selector(console, pilot, confirm)
        assert "Delete 4 messages" in str(console.query_one(confirm, Button).label)

        # Something else removes the last turn while the confirmation shows.
        store.delete_message(native["m4"])
        console.query_one(confirm, Button).press()
        await pilot.pause(0.3)

        flags = _deleted_flags(db, seeded["conversation_id"])
        assert (flags["m2"], flags["m3"]) == (0, 0), flags
        assert not host.screen.query("#console-delete-receipt")
        await _wait_until(
            pilot,
            lambda: (
                bool(console.query(confirm))
                and "Delete 2 messages" in str(console.query_one(confirm, Button).label)
            ),
        )
        assert any("changed" in notice for notice in notices), notices


@pytest.mark.asyncio
async def test_branched_undo_restores_sibling_order_and_selection():
    """AC#3 on a branched tree: off-branch copy, exact position, reselection."""
    app = _build_test_app()
    db = attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        rows = [
            ("u1", "user", None),
            ("a1", "assistant", "u1"),
            # The active branch is the FIRST sibling: a naive re-append on
            # Undo would move it to the end.
            ("u2a", "user", "a1"),
            ("a2a", "assistant", "u2a"),
            ("u2b", "user", "a1"),
            ("a2b", "assistant", "u2b"),
        ]
        seeded = await _open_rows(console, db, rows, "a2a")
        native, store, session_id = (
            seeded["native"],
            seeded["store"],
            seeded["session_id"],
        )
        before = _tree_shape(store, session_id)
        assert before["children"][native["a1"]] == [native["u2a"], native["u2b"]]

        # Deleting a1 would also take the off-path branch u2b/a2b.
        await _arm_delete_from_more(console, pilot, native["a1"])
        await _wait_for_selector(
            console, pilot, f"#console-message-action-delete-confirm-{native['a1']}"
        )
        await _wait_until(
            pilot,
            lambda: (
                "Delete this message and 4 later messages (2 on other branches)?"
                in _painted(host)
            ),
        )
        console.query_one(
            f"#console-message-action-delete-cancel-{native['a1']}", Button
        ).press()
        await pilot.pause()

        await _confirm_delete(console, pilot, host, native["u2a"])
        assert _tree_shape(store, session_id) != before
        host.screen.query_one("#console-delete-receipt-undo", Button).press()
        await _wait_until(
            pilot,
            lambda: set(_deleted_flags(db, seeded["conversation_id"]).values()) == {0},
        )
        await pilot.pause()

        assert _tree_shape(store, session_id) == before
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        await _wait_until(
            pilot, lambda: transcript.selected_message_id == native["u2a"]
        )


@pytest.mark.asyncio
async def test_done_keeps_the_delete_without_a_cleanup_warning():
    """AC#4 on the path that releases references: Undo focused, Esc is Done."""
    app = _build_test_app()
    db = attach_chachanotes_db(app)
    notices: list[str] = []
    app.notify = lambda message, **kwargs: notices.append(str(message))
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        rows = [
            ("u1", "user", None),
            ("a1", "assistant", "u1"),
            ("u2", "user", "a1"),
            ("a2", "assistant", "u2"),
        ]
        seeded = await _open_rows(console, db, rows, "a2")

        await _confirm_delete(console, pilot, host, seeded["native"]["u2"])
        undo = host.screen.query_one("#console-delete-receipt-undo", Button)
        await _wait_until(pilot, lambda: undo.has_focus)
        await pilot.press("escape")
        await _wait_until(
            pilot, lambda: not host.screen.query("#console-delete-receipt")
        )
        await pilot.pause(0.3)

        assert _deleted_flags(db, seeded["conversation_id"]) == {
            "u1": 0,
            "a1": 0,
            "u2": 1,
            "a2": 1,
        }
    assert not any(_CLEANUP_WARNING in notice for notice in notices), notices


@pytest.mark.asyncio
async def test_a_transient_undo_failure_keeps_undo_on_offer(monkeypatch):
    """A busy database is not "these messages changed": Undo can be retried."""
    import sqlite3

    app = _build_test_app()
    db = attach_chachanotes_db(app)
    notices: list[str] = []
    app.notify = lambda message, **kwargs: notices.append(str(message))
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        rows = [
            ("u1", "user", None),
            ("a1", "assistant", "u1"),
            ("u2", "user", "a1"),
            ("a2", "assistant", "u2"),
        ]
        seeded = await _open_rows(console, db, rows, "a2")
        conversation_id = seeded["conversation_id"]
        await _confirm_delete(console, pilot, host, seeded["native"]["u2"])

        real_restore = db.restore_message_subtree
        attempts: list[int] = []

        def locked_once(tombstones):
            attempts.append(1)
            if len(attempts) == 1:
                raise sqlite3.OperationalError("database is locked")
            return real_restore(tombstones)

        monkeypatch.setattr(db, "restore_message_subtree", locked_once)
        host.screen.query_one("#console-delete-receipt-undo", Button).press()
        await _wait_until(pilot, lambda: any("still deleted" in n for n in notices))
        assert not any("changed after they were deleted" in n for n in notices)
        assert _deleted_flags(db, conversation_id)["u2"] == 1
        await _wait_until(
            pilot, lambda: bool(host.screen.query("#console-delete-receipt"))
        )

        host.screen.query_one("#console-delete-receipt-undo", Button).press()
        await _wait_until(
            pilot, lambda: set(_deleted_flags(db, conversation_id).values()) == {0}
        )
        assert len(attempts) == 2
