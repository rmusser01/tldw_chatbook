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


def _seed(
    db: CharactersRAGDB,
    rows: list[tuple[str, str, str | None]],
    metadata: dict[str, str] | None = None,
) -> str:
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
                "metadata_json": (metadata or {}).get(message_id),
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


# --- TASK-33628.6: legacy flat conversations ----------------------------------
#
# Before branching, every Console row was saved with parent_message_id NULL.
# The store chains those roots into one spine IN MEMORY only
# (``ConsoleChatStore._chain_legacy_flat_roots``), so Delete's prompt, receipt
# and transcript all follow that chain. The DB delete followed parent links
# instead and tombstoned only the selected row, leaving the "later messages"
# the prompt counted live in reopen, search and export.

_FLAT = [
    ("f0", "user", None),
    ("f1", "assistant", None),
    ("f2", "user", None),
    ("f3", "assistant", None),
]

#: A flat prefix with a post-feature tail parented at the last flat row.
_FLAT_PREFIX = [
    ("m0", "user", None),
    ("m1", "assistant", None),
    ("m2", "user", "m1"),
    ("m3", "assistant", "m2"),
]


def _tree_ids(store, session_id: str) -> set[str]:
    """Every persisted id the store loaded, on or off the active path."""
    return {
        node.persisted_message_id
        for node in store._nodes_by_session[session_id].values()
    }


@pytest.mark.parametrize(
    ("rows", "target", "removed"),
    [
        pytest.param(_FLAT, "f1", ["f1", "f2", "f3"], id="flat"),
        pytest.param(_FLAT_PREFIX, "m0", ["m0", "m1", "m2", "m3"], id="flat-prefix"),
    ],
)
def test_flat_conversation_delete_tombstones_exactly_what_it_showed(
    rows, target, removed
):
    """AC#1/#2: the durable delete matches the prompt and the transcript."""
    from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
        export_conversation_to_text,
    )
    from tldw_chatbook.Chat.console_message_delete import (
        console_delete_scope,
        delete_subtree_for_undo,
    )

    db = CharactersRAGDB(":memory:", "flat-delete")
    conversation_id = _seed(db, rows)
    ids = [message_id for message_id, _role, _parent in rows]
    kept = [message_id for message_id in ids if message_id not in removed]
    store, session_id, native = _open_store(db, conversation_id)
    # Precondition: the store really chained the roots into one transcript.
    assert [m for m, _role in _visible(store, session_id)] == ids
    assert db.search_messages_by_content(removed[-1], conversation_id=conversation_id)

    scope = console_delete_scope(store, native[target])
    deleted, _held = delete_subtree_for_undo(store, native[target])

    assert scope.removed_count == deleted.count == len(removed)
    assert [m for m, _role in _visible(store, session_id)] == kept
    assert _deleted(db, ids) == [int(message_id in removed) for message_id in ids]
    assert sorted(message_id for message_id, _version in deleted.tombstones) == removed
    # Reopen loads only what the transcript kept -- on or off the active path.
    reopened, reopened_session, _native = _open_store(db, conversation_id)
    assert _tree_ids(reopened, reopened_session) == set(kept)
    # Search and export agree.
    for message_id in removed:
        assert not db.search_messages_by_content(
            message_id, conversation_id=conversation_id
        )
    exported = export_conversation_to_text(db, conversation_id) or ""
    assert [m for m in ids if f"{m} text" in exported] == kept


def test_flat_conversation_undo_restores_every_deleted_row():
    """AC#3: Undo puts back every row the flat delete tombstoned."""
    db = CharactersRAGDB(":memory:", "flat-delete")
    conversation_id = _seed(db, _FLAT)
    ids = [message_id for message_id, _role, _parent in _FLAT]
    store, session_id, native = _open_store(db, conversation_id)
    before = _visible(store, session_id)

    _delete_then_undo(store, native["f1"])

    assert _deleted(db, ids) == [0, 0, 0, 0]
    assert _visible(store, session_id) == before
    reopened, reopened_session, _native = _open_store(db, conversation_id)
    assert _visible(reopened, reopened_session) == before


def test_flat_delete_keeps_parent_link_descent_under_a_chained_root():
    """A later flat root's hidden parent-linked rows go too, and come back.

    ``ht`` is a tool row under ``h3``: never a store node, and reachable from
    ``h1`` only through the in-memory chain ``h1 -> h2``. So the store cannot
    name it -- only the DB's own parent-link descent from a seeded row can.
    """
    from tldw_chatbook.Chat.console_message_delete import (
        delete_subtree_for_undo,
        restore_deleted_subtree,
    )

    rows = [
        ("h0", "user", None),
        ("h1", "assistant", None),
        ("h2", "user", None),
        ("h3", "assistant", "h2"),
        ("ht", "tool", "h3"),
        ("h4", "assistant", "ht"),
    ]
    ids = [message_id for message_id, _role, _parent in rows]
    db = CharactersRAGDB(":memory:", "flat-delete")
    conversation_id = _seed(db, rows)
    store, session_id, native = _open_store(db, conversation_id)
    assert "ht" not in native  # precondition: the store never names it

    deleted, _held = delete_subtree_for_undo(store, native["h1"])

    assert _deleted(db, ids) == [0, 1, 1, 1, 1, 1]
    assert not db.search_messages_by_content("ht", conversation_id=conversation_id)
    restore_deleted_subtree(store, deleted)
    assert _deleted(db, ids) == [0] * len(ids)


def test_genuine_root_fork_delete_keeps_the_other_root_branch():
    """An un-chained root-level fork must not be swept into the delete."""
    from tldw_chatbook.Chat.console_message_delete import delete_subtree_for_undo

    rows = [
        ("u1", "user", None),
        ("a1", "assistant", "u1"),
        ("u1b", "user", None),
        ("a1b", "assistant", "u1b"),
    ]
    db = CharactersRAGDB(":memory:", "flat-delete")
    conversation_id = _seed(db, rows)
    store, _session_id, native = _open_store(db, conversation_id)

    delete_subtree_for_undo(store, native["u1"])

    assert _deleted(db, ["u1", "a1", "u1b", "a1b"]) == [1, 1, 0, 0]


def test_subtree_delete_seeds_stay_in_the_conversation_and_on_live_rows():
    """Caller-resolved ids never reach another conversation or a tombstone."""
    db = CharactersRAGDB(":memory:", "flat-delete")
    _seed(db, _FLAT)
    _seed(db, [("elsewhere", "user", None)])
    service = ChatPersistenceService(db)
    f3 = db.get_message_by_id("f3")
    db.soft_delete_message("f3", f3["version"])
    f3_tombstone_version = db.get_message_tombstones(["f3"])[0]["version"]

    rows = service.delete_message_subtree(
        message_id="f1", subtree_message_ids=("f1", "f2", "f3", "elsewhere")
    )

    assert sorted(row["message_id"] for row in rows) == ["f1", "f2"]
    assert _deleted(db, ["f0", "f1", "f2", "f3", "elsewhere"]) == [0, 1, 1, 1, 0]
    assert db.get_message_tombstones(["f3"])[0]["version"] == f3_tombstone_version


#: A flat conversation whose first reply was regenerated after branching
#: shipped: ``r1`` hangs under ``f0`` by a real parent link and is the active
#: leaf, so the chained ``f1 -> f2 -> f3`` spine sits OFF the active path.
_FLAT_REGENERATED = [*_FLAT, ("r1", "assistant", "f0")]


@pytest.mark.parametrize(
    ("target", "removed"),
    [
        pytest.param("f0", ["f0", "f1", "f2", "f3", "r1"], id="spans-both-paths"),
        pytest.param("f1", ["f1", "f2", "f3"], id="off-path-chained-root"),
    ],
)
def test_flat_delete_reaches_chained_roots_off_the_active_path(target, removed):
    """The DB delete takes the whole in-memory subtree, not just what is shown.

    The prompt counts off-path rows too (``console_delete_scope`` reports
    them as off-branch), so the durable delete must tombstone them as well.
    """
    from tldw_chatbook.Chat.console_message_delete import (
        console_delete_scope,
        delete_subtree_for_undo,
        restore_deleted_subtree,
    )

    ids = [message_id for message_id, _role, _parent in _FLAT_REGENERATED]
    db = CharactersRAGDB(":memory:", "flat-delete")
    conversation_id = _seed(db, _FLAT_REGENERATED)
    store, session_id, native = _open_store(db, conversation_id)
    # Precondition: the chained spine is loaded but off the active path.
    assert [m for m, _role in _visible(store, session_id)] == ["f0", "r1"]
    assert _tree_ids(store, session_id) == set(ids)

    scope = console_delete_scope(store, native[target])
    deleted, _held = delete_subtree_for_undo(store, native[target])

    assert scope.removed_count == deleted.count == len(removed)
    assert _deleted(db, ids) == [int(message_id in removed) for message_id in ids]
    restore_deleted_subtree(store, deleted)
    assert _deleted(db, ids) == [0] * len(ids)


def test_subtree_delete_descends_by_parent_link_without_statistics():
    """Each recursive step finds children by parent id, not by conversation.

    No ChaChaNotes database runs ``ANALYZE``, so the planner has no
    ``sqlite_stat1``. With ``child.conversation_id = ?`` indexable it chose
    the ``(conversation_id, id)`` index and scanned the whole conversation
    once per subtree row: 1.8 s to delete from a 3,000-message conversation,
    on the event loop. Both the SELECT and the UPDATE must search a
    ``parent_message_id`` index for the child rows.
    """
    db = CharactersRAGDB(":memory:", "flat-delete")
    _seed(db, _FLAT)
    conn = db.get_connection()
    assert (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sqlite_stat1'"
        ).fetchone()
        is None
    )
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    try:
        ChatPersistenceService(db).delete_message_subtree(
            message_id="f1", subtree_message_ids=("f1", "f2", "f3")
        )
    finally:
        conn.set_trace_callback(None)

    # Trigger sub-programs re-report their outer statement; dedupe them.
    recursive = {
        s for s in statements if s.lstrip().startswith("WITH RECURSIVE subtree")
    }
    assert len(recursive) == 2  # the SELECT that reads versions, and the UPDATE
    for statement in recursive:
        plan = [str(row[3]) for row in conn.execute("EXPLAIN QUERY PLAN " + statement)]
        child_steps = [detail for detail in plan if detail.split()[1:2] == ["child"]]
        assert child_steps, plan
        for detail in child_steps:
            assert detail.startswith("SEARCH child USING INDEX"), plan
            assert detail.endswith("(parent_message_id=?)"), plan


# --- TASK-33628.6: a NEW root-level edit fork is marked, not guessed ----------
#
# Editing and resending a conversation's FIRST message forks a new root-level
# USER row (the fork takes the first message's parent: none), and its reply
# hangs under it by a real parent link. The saved tree alone cannot tell that
# fork from a legacy flat run whose last flat USER row was answered later:
# Resend on a broken last turn appends the reply directly under that flat row.
# So the store does not guess from shape. A fork of a root message records
# ``root_fork`` in its local ``metadata_json`` when it is created, the flat
# repair chains every unmarked root exactly as before, and a marked root stays
# its own branch beside the first message. Forks saved before the marker
# existed are unmarked and keep the chained reading.

#: The durable marker, spelled out rather than built with ``MessageMetadata``
#: so a renamed field cannot silently stop reading rows already on disk.
_ROOT_FORK_METADATA = '{"root_fork": true}'

#: Flat rows, then an edit-and-resend of ``f0`` saved after branching shipped.
_FLAT_THEN_FORK = [
    *_FLAT,
    ("e0", "user", None),
    ("e1", "assistant", "e0"),
    ("e2", "user", "e1"),
    ("e3", "assistant", "e2"),
]
_FLAT_IDS = ["f0", "f1", "f2", "f3"]
_FORK_IDS = ["e0", "e1", "e2", "e3"]


class _RecordingGateway:
    """A ready provider that records each request and streams one reply."""

    def __init__(self) -> None:
        self.requests: list[list[dict]] = []

    async def resolve_for_send(self, selection):
        from Tests.console_provider_doubles import provider_resolution

        return provider_resolution(base_url="http://127.0.0.1:9099")

    async def stream_chat(self, resolution, messages, **kwargs):
        self.requests.append([dict(message) for message in messages])
        yield f"reply {len(self.requests)}"


async def _open_console(db: CharactersRAGDB, conversation_id: str):
    """The real controller over the real store, persistence service and DB.

    A conversation saved before Console Library policy existed got its policy
    row from the schema step that added the table; seed the same row (the
    step's own values) so a send can commit, then hydrate it as resume does.
    """
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT OR IGNORE INTO console_conversation_library_policy("
            "conversation_id, auto_retrieve_on_send, assistant_library_access"
            ") VALUES (?, 0, 1)",
            (conversation_id,),
        )
    store, session_id, native = _open_store(db, conversation_id)
    await store.hydrate_session_library_policy(session_id)
    gateway = _RecordingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    return controller, gateway, store, session_id, native


def _transcript(store, session_id: str) -> list[tuple[str, str]]:
    return [(m.role.value, m.content) for m in store.messages_for_session(session_id)]


def _root_count(store, session_id: str) -> int:
    return len(store._children_by_parent[session_id][None])


def test_marked_first_message_fork_reloads_as_its_own_root_branch():
    """A marked fork is a sibling of the first message, not a later flat row."""
    db = CharactersRAGDB(":memory:", "flat-delete")
    conversation_id = _seed(db, _FLAT_THEN_FORK, {"e0": _ROOT_FORK_METADATA})
    store, session_id, native = _open_store(db, conversation_id)

    # The saved cursor sits in the fork, so the fork alone is the transcript.
    assert [m for m, _role in _visible(store, session_id)] == _FORK_IDS
    siblings, index, count = store.siblings_at(native["e0"])
    assert [sibling.persisted_message_id for sibling in siblings] == ["f0", "e0"]
    assert (index, count) == (1, 2)
    # Swiping to the other root branch shows the flat chain, fork excluded.
    store.set_active_leaf(session_id, native["f3"])
    assert [m for m, _role in _visible(store, session_id)] == _FLAT_IDS
    assert store.subtree_message_ids(native["f0"]) == tuple(
        native[message_id] for message_id in _FLAT_IDS
    )


@pytest.mark.parametrize("shown", ["flat", "fork"])
def test_flat_delete_leaves_a_marked_first_message_fork_live(shown):
    """Prompt, receipt, transcript and the durable delete all skip the fork."""
    from tldw_chatbook.Chat.console_message_delete import (
        console_delete_receipt_copy,
        console_delete_scope,
        delete_subtree_for_undo,
    )

    ids = _FLAT_IDS + _FORK_IDS
    db = CharactersRAGDB(":memory:", "flat-delete")
    conversation_id = _seed(db, _FLAT_THEN_FORK, {"e0": _ROOT_FORK_METADATA})
    store, session_id, native = _open_store(db, conversation_id)
    if shown == "flat":
        store.set_active_leaf(session_id, native["f3"])
    visible_before = [m for m, _role in _visible(store, session_id)]

    scope = console_delete_scope(store, native["f1"])
    deleted, _held = delete_subtree_for_undo(store, native["f1"])

    assert scope.removed_count == deleted.count == 3
    assert scope.prompt.startswith("Delete this message and 2 later messages")
    assert console_delete_receipt_copy(deleted.count) == (
        "Deleted 3 messages from transcript."
    )
    assert sorted(message_id for message_id, _version in deleted.tombstones) == [
        "f1",
        "f2",
        "f3",
    ]
    assert _deleted(db, ids) == [0, 1, 1, 1, 0, 0, 0, 0]
    expected_view = ["f0"] if shown == "flat" else visible_before
    assert [m for m, _role in _visible(store, session_id)] == expected_view
    # Reopen keeps the fork and every reply under it.
    reopened, reopened_session, reopened_native = _open_store(db, conversation_id)
    assert _tree_ids(reopened, reopened_session) == {"f0", *_FORK_IDS}
    reopened.set_active_leaf(reopened_session, reopened_native["e3"])
    assert [m for m, _role in _visible(reopened, reopened_session)] == _FORK_IDS


def test_an_unmarked_root_fork_keeps_the_chained_reading():
    """A fork saved before the marker existed reads, deletes and undoes as before.

    Nothing on disk tells it apart from a flat row answered later, so it chains
    after the flat rows: Delete on an earlier flat row counts it, tombstones it
    with its replies, and Undo puts every row back.
    """
    from tldw_chatbook.Chat.console_message_delete import (
        console_delete_scope,
        delete_subtree_for_undo,
        restore_deleted_subtree,
    )

    ids = _FLAT_IDS + _FORK_IDS
    db = CharactersRAGDB(":memory:", "flat-delete")
    conversation_id = _seed(db, _FLAT_THEN_FORK)
    store, session_id, native = _open_store(db, conversation_id)
    assert [m for m, _role in _visible(store, session_id)] == ids
    assert _root_count(store, session_id) == 1

    scope = console_delete_scope(store, native["f1"])
    deleted, _held = delete_subtree_for_undo(store, native["f1"])

    assert scope.removed_count == deleted.count == 7
    assert scope.prompt.startswith("Delete this message and 6 later messages")
    assert sorted(message_id for message_id, _version in deleted.tombstones) == (
        sorted(ids[1:])
    )
    assert _deleted(db, ids) == [0, 1, 1, 1, 1, 1, 1, 1]
    assert [m for m, _role in _visible(store, session_id)] == ["f0"]

    restore_deleted_subtree(store, deleted)

    assert _deleted(db, ids) == [0] * len(ids)
    assert [m for m, _role in _visible(store, session_id)] == ids
    reopened, reopened_session, _native = _open_store(db, conversation_id)
    assert [m for m, _role in _visible(reopened, reopened_session)] == ids


@pytest.mark.parametrize(
    ("rows", "shown"),
    [
        # A reply regenerated mid-run hangs under a flat USER row that an
        # ASSISTANT flat row still follows.
        pytest.param(
            [*_FLAT, ("r3", "assistant", "f2")],
            ["f0", "f1", "f2", "r3"],
            id="regenerated-mid-run-reply",
        ),
        # Two unanswered flat turns; the last was edited and resent, so its
        # fork hangs under the turn before it.
        pytest.param(
            [
                ("f0", "user", None),
                ("f1", "assistant", None),
                ("f2", "user", None),
                ("f3", "user", None),
                ("g3", "user", "f2"),
                ("h3", "assistant", "g3"),
            ],
            ["f0", "f1", "f2", "g3", "h3"],
            id="edited-unanswered-turn",
        ),
        # Resend answered the last flat turn in place, and the chat went on.
        # Same shape as a first-message fork; unmarked, so it is flat data.
        pytest.param(
            [
                ("f0", "user", None),
                ("f1", "assistant", None),
                ("f2", "user", None),
                ("a2", "assistant", "f2"),
                ("u3", "user", "a2"),
                ("a3", "assistant", "u3"),
            ],
            ["f0", "f1", "f2", "a2", "u3", "a3"],
            id="resent-last-flat-turn",
        ),
        pytest.param(
            [*_FLAT, ("f4", "user", None), ("a4", "assistant", "f4")],
            ["f0", "f1", "f2", "f3", "f4", "a4"],
            id="resent-after-a-full-flat-run",
        ),
    ],
)
def test_flat_rows_with_later_children_still_chain(rows, shown):
    """Every unmarked root chains, whatever hangs under it."""
    db = CharactersRAGDB(":memory:", "flat-delete")
    conversation_id = _seed(db, rows)
    store, session_id, _native = _open_store(db, conversation_id)

    assert [m for m, _role in _visible(store, session_id)] == shown
    assert _root_count(store, session_id) == 1


async def test_resend_on_a_legacy_flat_turn_reloads_with_its_whole_history():
    """Resend answers a flat USER row in place; the rows above it stay in view.

    The reply hangs under the NULL-parent flat row by a real parent link -- the
    shape of a first-message edit fork. Reading that shape as a fork showed only
    the rows from the resent turn down after every reopen, and the next send
    built the model's context from that path, so the earlier history silently
    dropped out of what the model saw.
    """
    from tldw_chatbook.Chat.console_turn_resend import resend_target_id, resend_turn

    db = CharactersRAGDB(":memory:", "flat-delete")
    conversation_id = _seed(
        db,
        [("f0", "user", None), ("f1", "assistant", None), ("f2", "user", None)],
    )
    controller, _gateway, store, session_id, native = await _open_console(
        db, conversation_id
    )
    assert resend_target_id(store.messages_for_session(session_id)) == native["f2"]

    assert (await resend_turn(controller, native["f2"])).accepted
    assert (await controller.submit_draft("u3 text")).accepted

    history = [
        ("user", "f0 text"),
        ("assistant", "f1 text"),
        ("user", "f2 text"),
        ("assistant", "reply 1"),
        ("user", "u3 text"),
        ("assistant", "reply 2"),
    ]
    assert _transcript(store, session_id) == history
    reply = store.messages_for_session(session_id)[3]
    assert db.get_message_by_id(reply.persisted_message_id)["parent_message_id"] == (
        "f2"
    )
    assert db.get_message_by_id("f2")["metadata_json"] is None

    controller, gateway, store, session_id, _native = await _open_console(
        db, conversation_id
    )
    assert _transcript(store, session_id) == history
    assert _root_count(store, session_id) == 1
    # The next send carries the whole history to the model.
    assert (await controller.submit_draft("next")).accepted
    sent = [
        (message["role"], message["content"])
        for message in gateway.requests[-1]
        if message["role"] in {"user", "assistant"}
    ]
    assert sent == [*history, ("user", "next")]


@pytest.mark.parametrize("shown", ["flat", "fork"])
async def test_a_new_first_message_edit_fork_is_marked_and_outlives_a_flat_delete(
    shown,
):
    """Edit and resend of the first message marks the fork it saves.

    After a reopen the fork is its own branch beside the first message, and
    Delete on an earlier flat row neither counts nor removes it.
    """
    import json

    from tldw_chatbook.Chat.console_message_delete import (
        console_delete_scope,
        delete_subtree_for_undo,
    )

    db = CharactersRAGDB(":memory:", "flat-delete")
    conversation_id = _seed(db, _FLAT)
    controller, _gateway, store, session_id, native = await _open_console(
        db, conversation_id
    )
    assert [m for m, _role in _visible(store, session_id)] == _FLAT_IDS

    assert (await controller.edit_and_resend_message(native["f0"], "edited")).accepted

    fork = [m.persisted_message_id for m in store.messages_for_session(session_id)]
    assert _transcript(store, session_id) == [
        ("user", "edited"),
        ("assistant", "reply 1"),
    ]
    edited = db.get_message_by_id(fork[0])
    assert edited["parent_message_id"] is None
    assert json.loads(edited["metadata_json"] or "{}").get("root_fork") is True
    assert db.get_message_by_id(fork[1])["parent_message_id"] == fork[0]
    for message_id in _FLAT_IDS:
        assert db.get_message_by_id(message_id)["metadata_json"] is None

    store, session_id, native = _open_store(db, conversation_id)
    assert [m for m, _role in _visible(store, session_id)] == fork
    siblings, index, count = store.siblings_at(native[fork[0]])
    assert [sibling.persisted_message_id for sibling in siblings] == ["f0", fork[0]]
    assert (index, count) == (1, 2)
    if shown == "flat":
        store.set_active_leaf(session_id, native["f3"])
        assert [m for m, _role in _visible(store, session_id)] == _FLAT_IDS

    scope = console_delete_scope(store, native["f1"])
    deleted, _held = delete_subtree_for_undo(store, native["f1"])

    assert scope.removed_count == deleted.count == 3
    assert scope.prompt.startswith("Delete this message and 2 later messages")
    assert sorted(message_id for message_id, _version in deleted.tombstones) == [
        "f1",
        "f2",
        "f3",
    ]
    assert _deleted(db, [*_FLAT_IDS, *fork]) == [0, 1, 1, 1, 0, 0]
    reopened, reopened_session, reopened_native = _open_store(db, conversation_id)
    assert _tree_ids(reopened, reopened_session) == {"f0", *fork}
    reopened.set_active_leaf(reopened_session, reopened_native[fork[1]])
    assert [m for m, _role in _visible(reopened, reopened_session)] == fork


def test_the_root_fork_marker_survives_an_edit_and_keeps_other_metadata():
    """An in-place edit rewrites the row's metadata from the store's copy.

    The marker has to be part of that copy, or the first edit of a fork would
    erase it and the next reopen would chain the fork into the flat rows. The
    row's other metadata rides along unchanged.
    """
    import json

    other = {
        "engine": "realtime",
        "provider": "openai",
        "model": "gpt-realtime",
        "transcript_status": "final",
    }
    db = CharactersRAGDB(":memory:", "flat-delete")
    conversation_id = _seed(
        db, _FLAT_THEN_FORK, {"e0": json.dumps({**other, "root_fork": True})}
    )
    store, _session_id, native = _open_store(db, conversation_id)

    store.update_message_content(native["e0"], "e0 edited")

    stored = json.loads(db.get_message_by_id("e0")["metadata_json"])
    assert stored["root_fork"] is True
    assert {key: stored[key] for key in other} == other
    # Reopen: the edited fork is still its own root beside the first message.
    reopened, reopened_session, reopened_native = _open_store(db, conversation_id)
    assert reopened.get_message(reopened_native["e0"]).content == "e0 edited"
    assert _root_count(reopened, reopened_session) == 2
    siblings, _index, _count = reopened.siblings_at(reopened_native["e0"])
    assert [sibling.persisted_message_id for sibling in siblings] == ["f0", "e0"]
    assert reopened.subtree_message_ids(reopened_native["f0"]) == tuple(
        reopened_native[message_id] for message_id in _FLAT_IDS
    )


def test_flat_delete_and_undo_past_the_sqlite_variable_limit():
    """A delete wider than SQLite's bound-variable limit commits and undoes whole.

    The flat delete seeds the DB descent with the store's whole subtree, so it
    can reach any number of rows. Every id-list statement on the delete and
    Undo paths must stay under the connection's variable limit, or the delete
    rolls back (or Undo cannot find what it committed).
    """
    import sqlite3

    from tldw_chatbook.Chat.console_message_delete import (
        delete_subtree_for_undo,
        restore_deleted_subtree,
    )

    rows = [
        (f"v{index:02d}", "user" if index % 2 == 0 else "assistant", None)
        for index in range(40)
    ]
    db = CharactersRAGDB(":memory:", "flat-delete")
    conversation_id = _seed(db, rows)
    store, session_id, native = _open_store(db, conversation_id)
    before = _visible(store, session_id)
    conn = db.get_connection()

    def deleted_count() -> int:
        return conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ? AND deleted = 1",
            (conversation_id,),
        ).fetchone()[0]

    default_limit = conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 16)
    try:
        deleted, _held = delete_subtree_for_undo(store, native["v01"])
        assert deleted.count == len(deleted.tombstones) == 39
        assert deleted_count() == 39
        restore_deleted_subtree(store, deleted)
        assert deleted_count() == 0
    finally:
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, default_limit)
    assert _visible(store, session_id) == before
    reopened, reopened_session, _native = _open_store(db, conversation_id)
    assert _visible(reopened, reopened_session) == before


@pytest.mark.parametrize("fork_projection", [False, True], ids=["temporary", "fork"])
def test_a_temporary_chat_with_an_edited_first_message_still_saves(fork_projection):
    """Saving a temporary chat keeps working after its first message is resent.

    A forked chat's save stores fork-only facts (here the carried image's label)
    in the same local column and refuses a row that also carries other
    metadata, so a fork projection never marks a root fork. Its rows are all
    parent-linked, so it holds no flat row to tell a fork from. An ordinary
    temporary chat keeps the marker through the save.
    """
    import json

    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleMessageRole,
        MessageAttachment,
    )
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    db = CharactersRAGDB(":memory:", "flat-delete")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.create_session(title="Temporary", ephemeral=True)
    session.fork_projection = fork_projection
    image = (MessageAttachment(b"\x89PNG\r\n\x1a\n", "image/png", "chart.png", 0),)
    user, assistant = ConsoleMessageRole.USER, ConsoleMessageRole.ASSISTANT
    first = store.append_message(
        session.id, role=user, content="first", attachments=image
    )
    store.append_message(session.id, role=assistant, content="first reply")
    edited = store.create_sibling(
        first.id, role=user, content="edited", attachments=image
    )
    store.append_message(session.id, role=assistant, content="edited reply")

    assert store.promote_ephemeral_session(session.id) is not None

    row = db.get_message_by_id(store.get_message(edited.id).persisted_message_id)
    assert row["parent_message_id"] is None
    marker = json.loads(row["metadata_json"] or "{}").get("root_fork", False)
    assert marker is (not fork_projection)
