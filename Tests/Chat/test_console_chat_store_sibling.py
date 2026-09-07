"""``create_sibling`` primitive tests (Phase A, Task 5).

Pins the regenerate-as-sibling primitive: forking a new node alongside an
existing one (sharing its native parent) rather than beneath it, then making
that new node the active leaf so the visible transcript follows the new
branch. Also exercises the "swipe back" recovery path via ``set_active_leaf``
that the mid-conversation case depends on.
"""

from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole


def _s():
    st = ConsoleChatStore()
    ses = st.create_session(title="t")
    st.active_session_id = ses.id
    return st, ses.id


class _RecordingPersistence:
    """Records create_message calls and hands back predictable ``msg-N`` ids."""

    db = None

    def __init__(self):
        self.created = []

    def create_conversation(self, **kw):
        return "conv-1"

    def create_message(
        self,
        *,
        conversation_id,
        sender,
        content,
        image_data,
        image_mime_type,
        message_id=None,
        parent_message_id=None,
        feedback=None,
        attachments=None,
    ):
        pid = f"msg-{len(self.created) + 1}"
        self.created.append(
            {
                "id": pid,
                "parent_message_id": parent_message_id,
                "content": content,
                "sender": sender,
            }
        )
        return pid

    def update_message_content(self, **kw):
        return True


class _RecordingSyncProducer:
    """Records the Sync v2 envelope kwargs enqueued after a durable write."""

    def __init__(self):
        self.enqueued = []

    def enqueue_chat_message(self, **kwargs):
        self.enqueued.append(kwargs)
        return {
            "status": "enqueued",
            "outbox_entry": {
                "outbox_id": len(self.enqueued),
                "envelope": {
                    "payload_hash": f"hash:{kwargs['role']}:{kwargs['content']}"
                },
            },
        }


def test_create_sibling_of_last_assistant_makes_two_children():
    st, sid = _s()
    u = st.append_message(sid, role=ConsoleMessageRole.USER, content="q")
    a = st.append_message(sid, role=ConsoleMessageRole.ASSISTANT, content="ans-1")
    a2 = st.create_sibling(a.id, role=ConsoleMessageRole.ASSISTANT, content="ans-2")
    sibs, idx, count = st.siblings_at(a2.id)
    assert count == 2 and idx == 1
    assert st.active_leaf(sid) == a2.id
    assert st.active_path_message_ids(sid) == [u.id, a2.id]  # tail is the new sibling


def test_create_sibling_midconversation_truncates_visible_tail():
    st, sid = _s()
    u1 = st.append_message(sid, role=ConsoleMessageRole.USER, content="q1")
    a1 = st.append_message(sid, role=ConsoleMessageRole.ASSISTANT, content="a1")
    u2 = st.append_message(sid, role=ConsoleMessageRole.USER, content="q2")
    a2 = st.append_message(sid, role=ConsoleMessageRole.ASSISTANT, content="a2")
    # regenerate a1 (mid-conversation) -> new sibling under u1, tail (u2,a2) drops off-path
    a1b = st.create_sibling(a1.id, role=ConsoleMessageRole.ASSISTANT, content="a1-alt")
    assert st.active_path_message_ids(sid) == [u1.id, a1b.id]
    # swiping back to a1 restores the old tail
    st.set_active_leaf(sid, a2.id)
    assert st.active_path_message_ids(sid) == [u1.id, a1.id, u2.id, a2.id]


def test_eager_persisted_sibling_emits_onpath_sync_sequence():
    """create_sibling(persist=True) with eager content emits the on-path ordinal.

    Regression: persistence ran BEFORE the active-path recompute, so the Sync
    v2 helper (_sync_message_sequence walks the active-path view) saw a stale
    path and emitted ``sequence: None`` for an on-path, sync-eligible message.
    """
    persistence = _RecordingPersistence()
    sync = _RecordingSyncProducer()
    store = ConsoleChatStore(
        persistence=persistence,
        sync_v2_chat_producer=sync,
        sync_v2_server_profile_id="server-a",
    )
    ses = store.create_session(title="t")
    store.active_session_id = ses.id
    u1 = store.append_message(
        ses.id, role=ConsoleMessageRole.USER, content="q", persist=True
    )
    a1 = store.append_message(
        ses.id, role=ConsoleMessageRole.ASSISTANT, content="ans", persist=True
    )
    sync.enqueued.clear()

    sibling = store.create_sibling(
        a1.id, role=ConsoleMessageRole.ASSISTANT, content="eager", persist=True
    )

    # The sibling shares a1's parent (u1); active path is [u1, sibling].
    assert store.active_path_message_ids(ses.id) == [u1.id, sibling.id]
    envelope = sync.enqueued[-1]
    assert envelope["message_id"] == sibling.persisted_message_id
    # On-path ordinal, NOT None: u1 (1) then the eager sibling (2).
    assert envelope["sequence"] == 2
    # Sync parent is the anchor's parent (u1), the nearest persisted ancestor.
    assert envelope["parent_message_id"] == u1.persisted_message_id


def test_interstitial_non_persisted_note_keeps_parent_chain_connected():
    """A non-persisted mid-chain SYSTEM note must not fragment the persisted tree.

    Regression: persistence used the IMMEDIATE tree parent's persisted id. When
    a controller appends a ``persist=False`` SYSTEM interstitial mid-chain, that
    parent's ``persisted_message_id`` is None, so the next real message was
    written as a DB root (parent None) -- Task 8's leaf->root resume walk would
    stop at the break and drop all pre-interstitial history.
    """
    persistence = _RecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    ses = store.create_session(title="t")
    store.active_session_id = ses.id
    store.append_message(
        ses.id, role=ConsoleMessageRole.USER, content="q1", persist=True
    )
    a1 = store.append_message(
        ses.id, role=ConsoleMessageRole.ASSISTANT, content="a1", persist=True
    )
    # Controller-style interstitial appended with the default persist=False.
    store.append_message(
        ses.id, role=ConsoleMessageRole.SYSTEM, content="Response stopped by user."
    )
    u2 = store.append_message(
        ses.id, role=ConsoleMessageRole.USER, content="q2", persist=True
    )

    # Only the three persist=True messages hit durable storage.
    assert [rec["content"] for rec in persistence.created] == ["q1", "a1", "q2"]
    # u2's persisted parent skips the non-persisted note and points at a1.
    assert a1.persisted_message_id is not None
    assert u2.parent_message_id == a1.persisted_message_id
    assert persistence.created[-1]["parent_message_id"] == a1.persisted_message_id
    assert persistence.created[-1]["parent_message_id"] is not None
