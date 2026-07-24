"""Store-level tests for Console `/rewind` "summarize up to here" state (SP2).

Task 2 (storage only, no behavior yet): session-level ``(summary,
boundary_native_id)`` accessors mirroring the ``active_leaf_message_id``
pattern -- an in-memory pair (parallel dict, not tree state), a best-effort
write-through of the boundary's *persisted* id via the persistence adapter's
db seam, and a resume mapping from the persisted boundary id back to a
native tree id (dangling -> left unset, fail-open).
"""

import pytest

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


def _store_with_session():
    store = ConsoleChatStore()
    session = store.create_session(title="t")
    store.active_session_id = session.id
    return store, session.id


# --- in-memory set/get/clear -------------------------------------------------


def test_fresh_session_has_no_context_summary():
    store, sid = _store_with_session()
    assert store.session_context_summary(sid) == (None, None)


def test_set_and_get_context_summary_in_memory():
    store, sid = _store_with_session()
    u = store.append_message(sid, role=ConsoleMessageRole.USER, content="hi")
    store.set_session_context_summary(sid, "recap so far", u.id)
    assert store.session_context_summary(sid) == ("recap so far", u.id)


def test_clear_context_summary_resets_to_none_pair():
    store, sid = _store_with_session()
    u = store.append_message(sid, role=ConsoleMessageRole.USER, content="hi")
    store.set_session_context_summary(sid, "recap", u.id)
    store.set_session_context_summary(sid, None, None)
    assert store.session_context_summary(sid) == (None, None)


def test_context_summary_accessors_raise_for_unknown_session():
    store = ConsoleChatStore()
    with pytest.raises(KeyError):
        store.session_context_summary("nope")
    with pytest.raises(KeyError):
        store.set_session_context_summary("nope", "recap", None)


def test_context_summary_is_per_session():
    store = ConsoleChatStore()
    first = store.create_session(title="A")
    second = store.create_session(title="B")
    u = store.append_message(first.id, role=ConsoleMessageRole.USER, content="hi")

    store.set_session_context_summary(first.id, "recap", u.id)

    assert store.session_context_summary(first.id) == ("recap", u.id)
    assert store.session_context_summary(second.id) == (None, None)


def test_close_session_purges_context_summary_state():
    store, sid = _store_with_session()
    u = store.append_message(sid, role=ConsoleMessageRole.USER, content="hi")
    store.set_session_context_summary(sid, "recap", u.id)
    store.close_session(sid)
    assert sid not in store._context_summary_by_session


# --- write-through ------------------------------------------------------------


class _RecordingDB:
    def __init__(self, summary_pair=(None, None)):
        self.calls = []
        self._summary_pair = summary_pair

    def set_conversation_context_summary(self, conversation_id, summary, boundary_message_id):
        self.calls.append((conversation_id, summary, boundary_message_id))

    def get_conversation_context_summary(self, conversation_id):
        return self._summary_pair


class _DBPersistence:
    def __init__(self, db):
        self.db = db


def test_set_context_summary_write_through_records_persisted_boundary_id():
    db = _RecordingDB()
    store = ConsoleChatStore(persistence=_DBPersistence(db))
    session = store.restore_persisted_session(
        title="R",
        workspace_id=None,
        persisted_conversation_id="conv-9",
        all_nodes=[],
        active_leaf_persisted_id=None,
    )
    u = store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    # Simulate a durably-persisted node so write-through has a persisted id.
    store._nodes_by_session[session.id][u.id].persisted_message_id = "pm-1"

    store.set_session_context_summary(session.id, "recap", u.id)

    # The DB seam is called with the PERSISTED id, never the native id.
    assert db.calls[-1] == ("conv-9", "recap", "pm-1")

    # Clearing writes through a None pair.
    store.set_session_context_summary(session.id, None, None)
    assert db.calls[-1] == ("conv-9", None, None)


def test_set_context_summary_boundary_not_yet_persisted_writes_none_id():
    db = _RecordingDB()
    store = ConsoleChatStore(persistence=_DBPersistence(db))
    session = store.restore_persisted_session(
        title="R",
        workspace_id=None,
        persisted_conversation_id="conv-9",
        all_nodes=[],
        active_leaf_persisted_id=None,
    )
    u = store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    # `u` has NOT been durably persisted yet (no persisted_message_id).

    store.set_session_context_summary(session.id, "recap", u.id)

    assert db.calls[-1] == ("conv-9", "recap", None)


def test_set_context_summary_no_write_through_without_persisted_conversation():
    db = _RecordingDB()
    store = ConsoleChatStore(persistence=_DBPersistence(db))
    session = store.create_session(title="local-only")
    u = store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")

    store.set_session_context_summary(session.id, "recap", u.id)

    # No persisted conversation id => nothing written through...
    assert db.calls == []
    # ...but the in-memory pair is still applied.
    assert store.session_context_summary(session.id) == ("recap", u.id)


def test_set_context_summary_no_crash_without_db_seam():
    class _NoDbPersistence:
        pass

    store = ConsoleChatStore(persistence=_NoDbPersistence())
    session = store.create_session(title="t")
    u = store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")

    # Persistence adapter exposes no `.db` seam -> best-effort no-op, no crash.
    store.set_session_context_summary(session.id, "recap", u.id)

    assert store.session_context_summary(session.id) == ("recap", u.id)


def test_set_context_summary_no_crash_with_no_persistence_configured():
    store = ConsoleChatStore()  # persistence=None entirely
    session = store.create_session(title="t")
    u = store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")

    store.set_session_context_summary(session.id, "recap", u.id)

    assert store.session_context_summary(session.id) == ("recap", u.id)


def test_set_context_summary_write_through_failure_does_not_raise():
    class _ExplodingDB:
        def set_conversation_context_summary(self, *args, **kwargs):
            raise RuntimeError("boom")

    store = ConsoleChatStore(persistence=_DBPersistence(_ExplodingDB()))
    session = store.restore_persisted_session(
        title="R",
        workspace_id=None,
        persisted_conversation_id="conv-9",
        all_nodes=[],
        active_leaf_persisted_id=None,
    )
    u = store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")

    # A durable write failure is logged, never raised; the in-memory pair
    # remains authoritative and already applied.
    store.set_session_context_summary(session.id, "recap", u.id)

    assert store.session_context_summary(session.id) == ("recap", u.id)


# --- resume mapping ------------------------------------------------------------


def test_resume_maps_persisted_boundary_to_native_id():
    db = _RecordingDB(summary_pair=("earlier recap", "p1"))
    store = ConsoleChatStore(persistence=_DBPersistence(db))
    m1 = ConsoleChatMessage(
        role=ConsoleMessageRole.USER, content="u1", persisted_message_id="p1"
    )
    m2 = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="a1",
        persisted_message_id="p2",
        parent_message_id="p1",
    )

    session = store.restore_persisted_session(
        title="R",
        workspace_id=None,
        persisted_conversation_id="conv-1",
        all_nodes=[m1, m2],
        active_leaf_persisted_id="p2",
    )

    assert store.session_context_summary(session.id) == ("earlier recap", m1.id)


def test_resume_drops_dangling_boundary():
    db = _RecordingDB(summary_pair=("stale recap", "deleted-message-id"))
    store = ConsoleChatStore(persistence=_DBPersistence(db))
    m1 = ConsoleChatMessage(
        role=ConsoleMessageRole.USER, content="u1", persisted_message_id="p1"
    )

    session = store.restore_persisted_session(
        title="R",
        workspace_id=None,
        persisted_conversation_id="conv-1",
        all_nodes=[m1],
        active_leaf_persisted_id="p1",
    )

    # The stored boundary does not resolve to any node on the loaded tree ->
    # left unset (fail-open), not raised, not defaulted to some other node.
    assert store.session_context_summary(session.id) == (None, None)


def test_resume_leaves_unset_when_no_stored_summary():
    db = _RecordingDB(summary_pair=(None, None))
    store = ConsoleChatStore(persistence=_DBPersistence(db))
    m1 = ConsoleChatMessage(
        role=ConsoleMessageRole.USER, content="u1", persisted_message_id="p1"
    )

    session = store.restore_persisted_session(
        title="R",
        workspace_id=None,
        persisted_conversation_id="conv-1",
        all_nodes=[m1],
        active_leaf_persisted_id="p1",
    )

    assert store.session_context_summary(session.id) == (None, None)


def test_resume_no_crash_without_db_seam():
    store = ConsoleChatStore()  # persistence=None
    m1 = ConsoleChatMessage(
        role=ConsoleMessageRole.USER, content="u1", persisted_message_id="p1"
    )

    session = store.restore_persisted_session(
        title="R",
        workspace_id=None,
        persisted_conversation_id="conv-1",
        all_nodes=[m1],
        active_leaf_persisted_id="p1",
    )

    assert store.session_context_summary(session.id) == (None, None)


def test_resume_read_failure_leaves_unset_and_does_not_raise():
    class _ExplodingDB:
        def get_conversation_context_summary(self, conversation_id):
            raise RuntimeError("boom")

    store = ConsoleChatStore(persistence=_DBPersistence(_ExplodingDB()))
    m1 = ConsoleChatMessage(
        role=ConsoleMessageRole.USER, content="u1", persisted_message_id="p1"
    )

    session = store.restore_persisted_session(
        title="R",
        workspace_id=None,
        persisted_conversation_id="conv-1",
        all_nodes=[m1],
        active_leaf_persisted_id="p1",
    )

    assert store.session_context_summary(session.id) == (None, None)
