"""Tree-model scaffolding tests for ``ConsoleChatStore`` (Phase A, Task 3).

These pin the in-memory conversation-tree plumbing added in Task 3 while the
store still behaves linearly (active path == full transcript). They are the
contract for the tree structures, active-leaf accessors, sibling tracking, the
TOOL-marker display-only rule, tree-aware delete, and restore consistency.
"""

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


def test_linear_append_tracks_tree_and_active_leaf():
    store, sid = _store_with_session()
    u = store.append_message(sid, role=ConsoleMessageRole.USER, content="hi")
    a = store.append_message(sid, role=ConsoleMessageRole.ASSISTANT, content="yo")
    # active leaf is the last appended node; active path is the full linear list
    assert store.active_leaf(sid) == a.id
    assert store.active_path_message_ids(sid) == [u.id, a.id]
    # each node's in-memory PERSISTED parent is unknown without persistence
    assert store.get_message(a.id).parent_message_id is None
    # a single child => no siblings
    sibs, idx, count = store.siblings_at(a.id)
    assert count == 1 and idx == 0
    # the native parent of the assistant node is the user node (tree linkage)
    assert store._native_parent_by_message[a.id] == u.id
    assert store._native_parent_by_message[u.id] is None


def test_set_active_leaf_recomputes_active_path():
    store, sid = _store_with_session()
    u = store.append_message(sid, role=ConsoleMessageRole.USER, content="hi")
    a = store.append_message(sid, role=ConsoleMessageRole.ASSISTANT, content="one")
    store.set_active_leaf(sid, u.id)
    assert store.active_path_message_ids(sid) == [u.id]
    assert [m.content for m in store.messages_for_session(sid)] == ["hi"]
    store.set_active_leaf(sid, a.id)
    assert store.active_path_message_ids(sid) == [u.id, a.id]
    assert [m.content for m in store.messages_for_session(sid)] == ["hi", "one"]


def test_tool_marker_is_display_only():
    store, sid = _store_with_session()
    u = store.append_message(sid, role=ConsoleMessageRole.USER, content="q")
    a = store.append_message(sid, role=ConsoleMessageRole.ASSISTANT, content="a")
    marker = store.append_message(
        sid, role=ConsoleMessageRole.TOOL, content="⚙ tool → ok"
    )
    # display-only: visible in the transcript view, but NOT a tree node and NOT
    # the active leaf
    assert marker.content == "⚙ tool → ok"
    assert store.active_leaf(sid) == a.id
    assert marker.id not in store._nodes_by_session[sid]
    assert marker.id not in store._native_parent_by_message
    view = [m.content for m in store.messages_for_session(sid)]
    assert view == ["q", "a", "⚙ tool → ok"]
    # the next real message parents at the assistant answer, NOT the marker
    nxt = store.append_message(sid, role=ConsoleMessageRole.USER, content="q2")
    assert store._native_parent_by_message[nxt.id] == a.id
    assert store.active_path_message_ids(sid) == [u.id, a.id, nxt.id]
    # recompute rebuilds the view from real tree nodes only -> marker dropped
    # (accepted Phase A limitation)
    assert "⚙ tool → ok" not in [
        m.content for m in store.messages_for_session(sid)
    ]


def test_siblings_tracked_after_fork_via_active_leaf_rewind():
    store, sid = _store_with_session()
    u = store.append_message(sid, role=ConsoleMessageRole.USER, content="hi")
    a1 = store.append_message(sid, role=ConsoleMessageRole.ASSISTANT, content="one")
    # rewind the active leaf, then append -> a second child of `u` (a sibling)
    store.set_active_leaf(sid, u.id)
    a2 = store.append_message(sid, role=ConsoleMessageRole.ASSISTANT, content="two")
    assert store.active_path_message_ids(sid) == [u.id, a2.id]
    # both siblings are discoverable and ordered by creation
    snaps, idx, count = store.siblings_at(a2.id)
    assert count == 2 and idx == 1
    assert [s.content for s in snaps] == ["one", "two"]
    snaps0, idx0, count0 = store.siblings_at(a1.id)
    assert count0 == 2 and idx0 == 0
    # the recomputed active-path tail carries sibling hints for the renderer
    active_tail = store.messages_for_session(sid)[-1]
    assert active_tail.sibling_index == 1 and active_tail.sibling_count == 2


def test_delete_active_leaf_moves_leaf_to_parent():
    store, sid = _store_with_session()
    u = store.append_message(sid, role=ConsoleMessageRole.USER, content="hi")
    a = store.append_message(sid, role=ConsoleMessageRole.ASSISTANT, content="yo")
    store.delete_message(a.id)
    assert store.active_leaf(sid) == u.id
    assert store.active_path_message_ids(sid) == [u.id]
    assert a.id not in store._nodes_by_session[sid]
    assert a.id not in store._native_parent_by_message


def test_delete_off_path_node_leaves_active_path_unchanged():
    store, sid = _store_with_session()
    u = store.append_message(sid, role=ConsoleMessageRole.USER, content="hi")
    a = store.append_message(sid, role=ConsoleMessageRole.ASSISTANT, content="one")
    # rewind the active leaf so `a` is off the active path
    store.set_active_leaf(sid, u.id)
    a2 = store.append_message(sid, role=ConsoleMessageRole.ASSISTANT, content="two")
    assert store.active_path_message_ids(sid) == [u.id, a2.id]
    # deleting the off-path sibling does not disturb the active path
    store.delete_message(a.id)
    assert store.active_path_message_ids(sid) == [u.id, a2.id]
    assert store.active_leaf(sid) == a2.id
    assert a.id not in store._nodes_by_session[sid]


def test_delete_subtree_removes_descendants():
    store, sid = _store_with_session()
    u = store.append_message(sid, role=ConsoleMessageRole.USER, content="hi")
    a = store.append_message(sid, role=ConsoleMessageRole.ASSISTANT, content="a")
    u2 = store.append_message(sid, role=ConsoleMessageRole.USER, content="u2")
    # deleting `a` must also drop its descendant `u2`
    store.delete_message(a.id)
    assert store.active_path_message_ids(sid) == [u.id]
    for gone in (a.id, u2.id):
        assert gone not in store._nodes_by_session[sid]
        assert gone not in store._native_parent_by_message
        assert gone not in store._message_session_index


def test_restore_persisted_session_registers_tree_and_supports_append():
    store = ConsoleChatStore()
    m1 = ConsoleChatMessage(role=ConsoleMessageRole.USER, content="restored-u")
    m2 = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="restored-a")
    session = store.restore_persisted_session(
        title="Resumed",
        workspace_id=None,
        persisted_conversation_id="conv-1",
        messages=[m1, m2],
    )
    # restored nodes are resolvable via the tree (id lookups use _nodes_by_session)
    assert store.get_message(m2.id).content == "restored-a"
    assert store.active_leaf(session.id) == m2.id
    assert store.active_path_message_ids(session.id) == [m1.id, m2.id]
    # appending after restore preserves the restored history (no view wipe)
    nxt = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="follow-up"
    )
    assert [m.content for m in store.messages_for_session(session.id)] == [
        "restored-u",
        "restored-a",
        "follow-up",
    ]
    assert store._native_parent_by_message[nxt.id] == m2.id


def test_restore_state_rebuilds_tree_dicts():
    store = ConsoleChatStore()
    # seed some pre-existing state that restore must clear
    stale = store.create_session(title="Stale")
    store.append_message(stale.id, role=ConsoleMessageRole.USER, content="stale")
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession

    restored_session = ConsoleChatSession(id="session-a", title="Restored")
    restored_message = ConsoleChatMessage(
        id="message-a", role=ConsoleMessageRole.ASSISTANT, content="answer"
    )
    store.restore_state(
        sessions=[restored_session],
        messages_by_session={"session-a": [restored_message]},
        active_session_id="session-a",
    )
    # the stale session's tree state is gone
    assert stale.id not in store._nodes_by_session
    assert stale.id not in store._active_leaf_by_session
    # the restored session is a resolvable linear tree
    assert store.active_leaf("session-a") == "message-a"
    assert store.get_message("message-a").content == "answer"
    # append-after-restore keeps history
    nxt = store.append_message(
        "session-a", role=ConsoleMessageRole.USER, content="next"
    )
    assert store._native_parent_by_message[nxt.id] == "message-a"
    assert [m.content for m in store.messages_for_session("session-a")] == [
        "answer",
        "next",
    ]


def test_close_session_purges_all_tree_structures():
    store, sid = _store_with_session()
    u = store.append_message(sid, role=ConsoleMessageRole.USER, content="hi")
    a = store.append_message(sid, role=ConsoleMessageRole.ASSISTANT, content="one")
    # fork so there is an off-path node that must also be purged
    store.set_active_leaf(sid, u.id)
    a2 = store.append_message(sid, role=ConsoleMessageRole.ASSISTANT, content="two")
    store.close_session(sid)
    assert sid not in store._nodes_by_session
    assert sid not in store._children_by_parent
    assert sid not in store._active_leaf_by_session
    assert sid not in store._messages_by_session
    for mid in (u.id, a.id, a2.id):
        assert mid not in store._message_session_index
        assert mid not in store._native_parent_by_message


class _RecordingDB:
    def __init__(self):
        self.calls = []

    def set_conversation_active_leaf(self, conversation_id, message_id):
        self.calls.append((conversation_id, message_id))


class _DBPersistence:
    def __init__(self, db):
        self.db = db


def test_set_active_leaf_write_through_persists_pointer():
    db = _RecordingDB()
    store = ConsoleChatStore(persistence=_DBPersistence(db))
    session = store.restore_persisted_session(
        title="R",
        workspace_id=None,
        persisted_conversation_id="conv-9",
        messages=[],
    )
    u = store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    # simulate a durably-persisted node so the write-through has a leaf id
    store._nodes_by_session[session.id][u.id].persisted_message_id = "pm-1"
    store.set_active_leaf(session.id, u.id)
    assert db.calls[-1] == ("conv-9", "pm-1")
    # clearing the leaf writes through a None pointer
    store.set_active_leaf(session.id, None)
    assert db.calls[-1] == ("conv-9", None)


def test_set_active_leaf_no_write_through_without_persisted_conversation():
    db = _RecordingDB()
    store = ConsoleChatStore(persistence=_DBPersistence(db))
    session = store.create_session(title="local-only")
    u = store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    store.set_active_leaf(session.id, u.id)
    # no persisted conversation id => nothing written through
    assert db.calls == []
