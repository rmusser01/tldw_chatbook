"""Persistence seams behind Console Delete -> Undo (TASK-33628.2).

Real ``CharactersRAGDB`` on in-memory SQLite behind the real
``ChatPersistenceService`` -- the same pair the Console store writes through.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, ConflictError

# The real ChatScreen/store goes through config-participant admission, which the
# per-test sandbox refuses (RecoveryRequired); keep the collection-time profile.
pytestmark = pytest.mark.bootstrap_profile


def _conversation(db: CharactersRAGDB, count: int = 4) -> tuple[str, list[str]]:
    conversation_id = db.add_conversation({"title": "Delete undo"})
    ids: list[str] = []
    parent = None
    for index in range(count):
        parent = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user" if index % 2 == 0 else "assistant",
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"message {index}",
                "parent_message_id": parent,
            }
        )
        ids.append(parent)
    return conversation_id, ids


def _deleted(db: CharactersRAGDB, ids: list[str]) -> list[int]:
    with db.transaction() as conn:
        rows = {
            row["id"]: int(row["deleted"])
            for row in conn.execute(
                f"SELECT id, deleted FROM messages WHERE id IN ({','.join('?' for _ in ids)})",  # nosec B608 - placeholders only
                tuple(ids),
            ).fetchall()
        }
    return [rows[message_id] for message_id in ids]


def test_media_free_subtree_delete_leaves_no_cleanup_warning():
    """AC#4: no recovered-media catalog means no reference can be pending."""
    db = CharactersRAGDB(":memory:", "delete-undo")
    service = ChatPersistenceService(db)
    _conversation_id, ids = _conversation(db)

    rows = service.delete_message_subtree(message_id=ids[1])

    assert {row["message_id"] for row in rows} == set(ids[1:])
    assert not service.recovered_media_cleanup_pending
    assert service.recovered_media_cleanup_warning is None


def test_restore_message_subtree_undeletes_exact_tombstones_and_cursor():
    db = CharactersRAGDB(":memory:", "delete-undo")
    service = ChatPersistenceService(db)
    conversation_id, ids = _conversation(db)
    db.set_conversation_active_cursor(
        conversation_id, active_leaf_message_id=ids[-1], before_message_id=None
    )
    rows = service.delete_message_subtree(message_id=ids[1])
    db.set_conversation_active_leaf(conversation_id, ids[0])
    tombstones = tuple((row["message_id"], row["version"]) for row in rows)

    restored = service.restore_message_subtree(
        tombstones=tombstones,
        conversation_id=conversation_id,
        active_cursor=(ids[-1], None),
    )

    assert _deleted(db, ids) == [0, 0, 0, 0]
    assert {(row["message_id"], row["version"]) for row in restored} == {
        (message_id, version + 1) for message_id, version in tombstones
    }
    assert db.get_conversation_active_cursor(conversation_id) == (ids[-1], None)


def test_restore_message_subtree_refuses_a_changed_tombstone_atomically():
    db = CharactersRAGDB(":memory:", "delete-undo")
    service = ChatPersistenceService(db)
    _conversation_id, ids = _conversation(db)
    rows = service.delete_message_subtree(message_id=ids[1])
    stale = tuple(
        (row["message_id"], row["version"] - (1 if index == 0 else 0))
        for index, row in enumerate(rows)
    )

    with pytest.raises(ConflictError):
        service.restore_message_subtree(tombstones=stale)

    assert _deleted(db, ids) == [0, 1, 1, 1]


def test_restored_rows_are_committed_sync_intents():
    """An undelete is an ordinary committed update a sync projection can read."""
    db = CharactersRAGDB(":memory:", "delete-undo")
    service = ChatPersistenceService(db)
    _conversation_id, ids = _conversation(db)
    rows = service.delete_message_subtree(message_id=ids[2])

    restored = service.restore_message_subtree(
        tombstones=tuple((row["message_id"], row["version"]) for row in rows)
    )

    for row in restored:
        live = db.get_message_by_id(row["message_id"])
        assert live is not None and live["version"] == row["version"]
        intent = db.read_committed_chat_sync_intent(
            message_id=row["message_id"],
            message_version=row["version"],
            payload_hash=CharactersRAGDB._chat_sync_payload_hash_from_row(live),
        )
        assert intent is not None


def test_image_rejection_hint_names_a_non_destructive_path():
    """AC#5: the recovery hint never sends users to the subtree Delete."""
    hint = ConsoleChatController._IMAGE_REJECTION_RECOVERY_HINT

    assert "use Delete" not in hint
    assert "remove that message" not in hint
    assert "vision-capable model" in hint
    assert "/rewind" in hint


# --- Review fixes (TASK-33628.2 checkpoint review) --------------------------
#
# The two cases below drive the real ConsoleChatStore through the same
# delete -> Undo sequence the Console's receipt runs, over the real
# ChatPersistenceService / in-memory CharactersRAGDB pair.


def _tree_nodes(db: CharactersRAGDB, conversation_id: str) -> list:
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.Chat.console_conversation_hydration import (
        console_messages_from_conversation_tree,
    )

    tree = ChatConversationService(db).get_conversation_tree(
        conversation_id, depth_cap=10_000, root_limit=10_000
    )
    return console_messages_from_conversation_tree(tree, db=db)


def _open_store(db: CharactersRAGDB, conversation_id: str):
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.restore_persisted_session(
        title="Delete undo",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=_tree_nodes(db, conversation_id),
        active_leaf_persisted_id=db.get_conversation_active_cursor(conversation_id)[0],
    )
    native = {
        node.persisted_message_id: node.id
        for node in store._nodes_by_session[session.id].values()
    }
    return store, session.id, native


def _delete_then_undo(store, message_id: str) -> None:
    """Run the Console receipt's Delete -> Undo sequence on ``message_id``."""
    from tldw_chatbook.Chat.console_message_delete import (
        delete_subtree_for_undo,
        restore_deleted_subtree,
    )

    deleted, _held = delete_subtree_for_undo(store, message_id)
    restore_deleted_subtree(store, deleted)


def _visible(store, session_id: str) -> list[tuple[str, str]]:
    return [
        (m.persisted_message_id, m.role.value)
        for m in store.messages_for_session(session_id)
    ]


def _seed(db: CharactersRAGDB, rows: list[tuple[str, str, str | None]]) -> str:
    conversation_id = db.add_conversation({"title": "Delete undo"})
    for index, (message_id, role, parent) in enumerate(rows):
        db.add_message(
            {
                "id": message_id,
                "conversation_id": conversation_id,
                "sender": role,
                "role": role,
                "content": f"{message_id} text",
                "parent_message_id": parent,
                "timestamp": f"2026-09-30T00:00:{index:02d}.000000+00:00",
            }
        )
    db.set_conversation_active_cursor(
        conversation_id, active_leaf_message_id=rows[-1][0], before_message_id=None
    )
    return conversation_id


_CHAIN = [
    ("c0", "user", None),
    ("c1", "assistant", "c0"),
    ("c2", "user", "c1"),
    ("c3", "assistant", "c2"),
]


@pytest.mark.parametrize("operation", ["regenerate", "add_variant"])
def test_restored_reply_can_be_regenerated_after_undo(operation):
    """Undo bumps each row's version twice; the store must learn the new one.

    Before the fix the restored node kept its pre-delete version, so the next
    Regenerate (or manual variant) on it raised ConflictError at finalize and
    the paid generation was lost.
    """
    db = CharactersRAGDB(":memory:", "delete-undo")
    conversation_id = _seed(db, _CHAIN)
    store, _session_id, native = _open_store(db, conversation_id)
    reply = native["c3"]

    _delete_then_undo(store, native["c2"])

    if operation == "regenerate":
        token = store.begin_generation_attempt(reply)
        store.begin_variant_stream(reply, generation_token=token)
        store.append_stream_chunk(reply, "regenerated reply")
        store.finalize_variant_stream(reply)
    else:
        store.add_variant(reply, "regenerated reply")

    assert db.get_message_by_id("c3")["content"] == "regenerated reply"
    assert db.get_message_by_id("c3")["version"] == (
        store.get_message(reply).provider_continuation_message_version
    )


def test_undo_restores_every_row_the_delete_tombstoned_including_tool_rows():
    """Undo must restore what the DB delete committed, not only store nodes.

    A persisted tool-role row is never a store node, but the DB subtree
    delete tombstones it with everything else. Restoring only the store's
    nodes left it deleted, and on reopen every message beneath it vanished.
    """
    db = CharactersRAGDB(":memory:", "delete-undo")
    conversation_id = _seed(
        db,
        [
            ("u1", "user", None),
            ("a1", "assistant", "u1"),
            ("t1", "tool", "a1"),
            ("a2", "assistant", "t1"),
            ("u2", "user", "a2"),
            ("a3", "assistant", "u2"),
        ],
    )
    store, session_id, native = _open_store(db, conversation_id)
    before = _visible(store, session_id)
    ids = ["u1", "a1", "t1", "a2", "u2", "a3"]

    _delete_then_undo(store, native["u1"])

    assert _deleted(db, ids) == [0] * len(ids)
    reopened, reopened_session, _native = _open_store(db, conversation_id)
    assert _visible(reopened, reopened_session) == before


def test_undo_refusal_says_changed_only_for_a_version_conflict(monkeypatch):
    """A moved tombstone is final; a storage hiccup keeps Undo on offer."""
    import sqlite3

    from tldw_chatbook.Chat.console_message_delete import (
        ConsoleDeleteUndoError,
        delete_subtree_for_undo,
        restore_deleted_subtree,
    )

    db = CharactersRAGDB(":memory:", "delete-undo")
    conversation_id = _seed(db, _CHAIN)
    store, session_id, native = _open_store(db, conversation_id)
    deleted, _held = delete_subtree_for_undo(store, native["c2"])

    real_restore = db.restore_message_subtree

    def locked(_tombstones):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db, "restore_message_subtree", locked)
    with pytest.raises(ConsoleDeleteUndoError) as transient:
        restore_deleted_subtree(store, deleted)
    assert transient.value.retryable
    assert "changed" not in str(transient.value)
    assert "still deleted" in str(transient.value)
    assert _deleted(db, ["c2", "c3"]) == [1, 1]
    assert [m.persisted_message_id for m in store.messages_for_session(session_id)] == [
        "c0",
        "c1",
    ]

    # Nothing changed, so the retry the receipt re-offers succeeds.
    monkeypatch.setattr(db, "restore_message_subtree", real_restore)
    restore_deleted_subtree(store, deleted)
    assert _deleted(db, ["c2", "c3"]) == [0, 0]

    # A tombstone that moved after the delete is a real conflict: final.
    deleted, _held = delete_subtree_for_undo(store, native["c2"])
    with db.transaction() as conn:
        conn.execute("UPDATE messages SET version = version + 1 WHERE id = 'c3'")
    with pytest.raises(ConsoleDeleteUndoError) as conflict:
        restore_deleted_subtree(store, deleted)
    assert not conflict.value.retryable
    assert "changed after they were deleted" in str(conflict.value)
    assert _deleted(db, ["c2", "c3"]) == [1, 1]


def test_recovered_media_hold_diverts_only_the_calling_thread():
    """A worker thread's release during the hold is not swept into it."""
    import threading

    service = ChatPersistenceService(CharactersRAGDB(":memory:", "delete-undo"))
    other_thread: list[bool] = []

    with service.hold_recovered_media_release() as held:
        worker = threading.Thread(
            target=lambda: other_thread.append(
                service.release_recovered_media_references(["worker-row"])
            )
        )
        worker.start()
        worker.join()
        service.release_recovered_media_references(["own-row"])

    assert held == ["own-row"]
    assert other_thread == [False]


#: Per-message store state that Delete purges but Undo deliberately does not
#: put back, each with the reason it cannot matter for a restored message.
_NOT_RESTORED_BY_UNDO = {
    "_stream_chunks_by_message": "stream buffer; Delete refuses a streaming "
    "message and folds its buffer first",
    "_stream_materialized_counts": "stream buffer bookkeeping, as above",
    "_variant_stream_bases": "exists only while a variant streams",
    "_exchange_blob_cache": "read cache, rebuilt on demand",
    "_terminal_citation_finalizers": "in-flight terminal state of a reply "
    "that is still streaming",
    "_provisional_terminal_selection_ids": "in-flight terminal state, as above",
    "_terminal_persistence_deferred_ids": "in-flight terminal state, as above",
}


def test_undo_accounts_for_every_per_message_registry_delete_purges():
    """Undo stays exact only if it knows every per-node map Delete clears.

    If the store grows a new per-message registry in its delete purge, this
    fails until ``console_message_delete`` captures and restores it, or it is
    listed above with the reason Undo can ignore it.
    """
    import inspect
    import re

    from tldw_chatbook.Chat import console_message_delete as undo
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    purged = set(
        re.findall(
            r"self\.(_\w+)\.(?:pop|discard)\(node_id",
            inspect.getsource(ConsoleChatStore._delete_message),
        )
    ) | set(
        re.findall(
            r"self\.(_\w+)\.(?:pop|discard)\(message_id",
            inspect.getsource(ConsoleChatStore.clear_terminal_citation_state),
        )
    )
    handled = set(
        re.findall(
            r"store\.(_\w+)",
            inspect.getsource(undo.capture_deleted_subtree)
            + inspect.getsource(undo._reinsert),
        )
    )

    assert "_failed_retry_message_ids" in purged  # the scan found the purge
    unaccounted = purged - handled - set(_NOT_RESTORED_BY_UNDO)
    assert not unaccounted, (
        f"Delete purges {sorted(unaccounted)} but Undo neither restores them "
        "nor documents why it need not"
    )
    assert not set(_NOT_RESTORED_BY_UNDO) - purged, "stale exemption"
    assert not set(_NOT_RESTORED_BY_UNDO) & handled, "exempt yet restored"


def test_a_delete_refused_after_its_tombstone_write_rolls_back(monkeypatch):
    """The refusal leaves no committed tombstone, so there is nothing to release.

    PR #2941 review claimed ``store.delete_message`` can commit the subtree's
    tombstones and THEN raise (a pending-dispatch cursor refusal), dropping
    the held recovered-media ids. The refusal raises inside the store's
    ``_dispatch_branch_mutation`` transaction, so the tombstones roll back
    with it: the held ids name live messages, and releasing their references
    (the suggested fix) would orphan media those messages still show.
    """
    from tldw_chatbook.Chat.console_message_delete import delete_subtree_for_undo
    from tldw_chatbook.DB.ChaChaNotes_DB import InputError

    db = CharactersRAGDB(":memory:", "delete-undo")
    conversation_id = _seed(db, _CHAIN)
    store, session_id, native = _open_store(db, conversation_id)
    written: list[list[dict]] = []
    real_delete = store.persistence.delete_message_subtree

    def delete_then_record(**kwargs):
        rows = real_delete(**kwargs)
        written.append(rows)
        return rows

    def refuse(*_args, **_kwargs):
        raise InputError("pending dispatch owns the cursor")

    monkeypatch.setattr(store.persistence, "delete_message_subtree", delete_then_record)
    monkeypatch.setattr(db, "set_conversation_active_leaf", refuse)

    with pytest.raises(ValueError, match="pending dispatch"):
        delete_subtree_for_undo(store, native["c2"])

    # The tombstone write ran (the claim's precondition) ...
    assert {row["message_id"] for row in written[0]} == {"c2", "c3"}
    # ... and rolled back with the refusal: nothing is deleted.
    assert _deleted(db, ["c2", "c3"]) == [0, 0]
    assert not db.get_message_tombstones(["c2", "c3"])
    assert not store.persistence.recovered_media_cleanup_pending
    assert [m.persisted_message_id for m in store.messages_for_session(session_id)] == [
        "c0",
        "c1",
        "c2",
        "c3",
    ]
