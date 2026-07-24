"""Console resume reconstructs the active branch from the stored leaf pointer.

Task 8 (Phase A conversation branching): resuming a saved conversation must
load the WHOLE persisted tree (every branch, on- and off-path) and derive the
visible transcript from the stored ``active_leaf_message_id`` pointer -- not
from a ``children[-1]`` latest-branch walk. Loading all branches is what makes
off-path siblings navigable (swipe) immediately after resume.

Real DB round-trips: a real ``CharactersRAGDB`` behind the real
``ChatConversationService``/``ChatPersistenceService`` and the real ChatScreen
full-tree flatten -- no hand-rolled fakes for the pieces under test.
"""

from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

from Tests.UI.test_destination_shells import _build_test_app


def _persist_branched_conversation(db: CharactersRAGDB):
    """Persist ``U1 -> {A1 (older), A1' (newer)}`` and return the ids.

    Explicit, strictly increasing timestamps pin the sibling order the tree
    service reads (``ORDER BY timestamp ASC``), so ``A1`` is unambiguously the
    older sibling and ``A1'`` the ``children[-1]`` most-recent one.
    """
    service = ChatConversationService(db)
    conversation_id = service.create_conversation(
        id="branch-conv-1",
        title="Branchy",
        scope_type="global",
        state="in-progress",
    )
    u1 = db.add_message(
        {
            "id": "m-u1",
            "conversation_id": conversation_id,
            "sender": "user",
            "role": "user",
            "content": "u1",
            "timestamp": "2026-01-01T00:00:00.000000+00:00",
        }
    )
    a1 = db.add_message(
        {
            "id": "m-a1",
            "conversation_id": conversation_id,
            "parent_message_id": u1,
            "sender": "assistant",
            "role": "assistant",
            "content": "a1",
            "timestamp": "2026-01-01T00:00:01.000000+00:00",
        }
    )
    a1_prime = db.add_message(
        {
            "id": "m-a1-prime",
            "conversation_id": conversation_id,
            "parent_message_id": u1,
            "sender": "assistant",
            "role": "assistant",
            "content": "a1-prime",
            "timestamp": "2026-01-01T00:00:02.000000+00:00",
        }
    )
    return conversation_id, u1, a1, a1_prime


def _persist_flat_legacy_conversation(db: CharactersRAGDB):
    """Persist a legacy FLAT conversation: every message a NULL-parent root.

    Mimics pre-branching persistence, where the base ``_persist_new_message``
    hardcoded ``parent_message_id=None`` for every message. The four rows
    therefore load on resume as four separate roots (all siblings under
    ``None``), not one linear thread. Strictly increasing timestamps pin the
    DB's ``ORDER BY timestamp ASC`` root order.
    """
    service = ChatConversationService(db)
    conversation_id = service.create_conversation(
        id="flat-conv-1",
        title="Flat",
        scope_type="global",
        state="in-progress",
    )
    for i, (content, sender) in enumerate(
        [("u1", "user"), ("a1", "assistant"), ("u2", "user"), ("a2", "assistant")]
    ):
        db.add_message(
            {
                "id": f"m-flat-{i}",
                "conversation_id": conversation_id,
                "sender": sender,
                "role": sender,
                "content": content,
                "timestamp": f"2026-01-01T00:00:0{i}.000000+00:00",
            }
        )
    return conversation_id


def _persist_mixed_legacy_then_branched_conversation(db: CharactersRAGDB):
    """Flat legacy prefix ``[u1,a1,u2,a2]`` (NULL parents) then a post-feature
    continuation ``u3 -> a3`` genuinely parented onto ``a2``.

    Reproduces an old conversation that gained new messages after branching
    landed: the prefix is four roots, the continuation is a real subtree
    hanging off the last flat row.
    """
    service = ChatConversationService(db)
    conversation_id = service.create_conversation(
        id="mixed-conv-1",
        title="Mixed",
        scope_type="global",
        state="in-progress",
    )
    ids = []
    for i, (content, sender) in enumerate(
        [("u1", "user"), ("a1", "assistant"), ("u2", "user"), ("a2", "assistant")]
    ):
        ids.append(
            db.add_message(
                {
                    "id": f"m-mixed-{i}",
                    "conversation_id": conversation_id,
                    "sender": sender,
                    "role": sender,
                    "content": content,
                    "timestamp": f"2026-01-01T00:00:0{i}.000000+00:00",
                }
            )
        )
    a2_id = ids[-1]
    u3 = db.add_message(
        {
            "id": "m-mixed-u3",
            "conversation_id": conversation_id,
            "parent_message_id": a2_id,
            "sender": "user",
            "role": "user",
            "content": "u3",
            "timestamp": "2026-01-01T00:00:05.000000+00:00",
        }
    )
    db.add_message(
        {
            "id": "m-mixed-a3",
            "conversation_id": conversation_id,
            "parent_message_id": u3,
            "sender": "assistant",
            "role": "assistant",
            "content": "a3",
            "timestamp": "2026-01-01T00:00:06.000000+00:00",
        }
    )
    return conversation_id


def _resume_into_store(db: CharactersRAGDB, conversation_id: str):
    """Mirror the production resume plumbing end to end.

    Full-tree flatten via the REAL ChatScreen helper + the stored active-leaf
    pointer, fed into ``restore_persisted_session`` exactly as
    ``_resume_console_workspace_conversation`` does.
    """
    service = ChatConversationService(db)
    tree = service.get_conversation_tree(
        conversation_id, depth_cap=10_000, root_limit=10_000
    )
    screen = ChatScreen(_build_test_app())
    screen.app_instance.chachanotes_db = db
    all_nodes = screen._console_messages_from_conversation_tree(tree)
    active_leaf_id = db.get_conversation_active_leaf(conversation_id)
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.restore_persisted_session(
        title="Branchy",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=all_nodes,
        active_leaf_persisted_id=active_leaf_id,
    )
    return store, session


def test_console_messages_from_conversation_tree_flattens_all_branches():
    """The flatten returns EVERY node (both siblings), each carrying its
    persisted id and persisted parent id -- not just the latest branch."""
    screen = ChatScreen(_build_test_app())
    tree = {
        "conversation": {"title": "Saved"},
        "root_threads": [
            {
                "id": "u1",
                "sender": "user",
                "content": "u1",
                "parent_message_id": None,
                "children": [
                    {
                        "id": "a1",
                        "sender": "assistant",
                        "content": "a1",
                        "parent_message_id": "u1",
                        "children": [],
                    },
                    {
                        "id": "a1-prime",
                        "sender": "assistant",
                        "content": "a1-prime",
                        "parent_message_id": "u1",
                        "children": [],
                    },
                ],
            }
        ],
    }

    messages = screen._console_messages_from_conversation_tree(tree)

    by_pid = {m.persisted_message_id: m for m in messages}
    assert set(by_pid) == {"u1", "a1", "a1-prime"}
    assert by_pid["u1"].parent_message_id is None
    assert by_pid["a1"].parent_message_id == "u1"
    assert by_pid["a1-prime"].parent_message_id == "u1"


def test_resume_reconstructs_older_branch_from_active_leaf():
    """Pointer at the OLDER sibling -> transcript is the older branch, not the
    ``children[-1]`` latest one the pre-Task-8 walk would have shown."""
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        conversation_id, _u1, a1, _a1_prime = _persist_branched_conversation(db)
        db.set_conversation_active_leaf(conversation_id, a1)

        store, session = _resume_into_store(db, conversation_id)

        view = [m.content for m in store.messages_for_session(session.id)]
        assert view == ["u1", "a1"]
    finally:
        db.close_connection()


def test_resume_loads_off_path_siblings_for_swipe():
    """After resuming onto the older branch, the off-path sibling is loaded and
    navigable: ``siblings_at`` reports 2 and ``set_active_leaf`` swaps the view."""
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        conversation_id, _u1, a1, _a1_prime = _persist_branched_conversation(db)
        db.set_conversation_active_leaf(conversation_id, a1)

        store, session = _resume_into_store(db, conversation_id)

        restored_a1 = store.messages_for_session(session.id)[-1]
        assert restored_a1.content == "a1"

        snapshots, index, count = store.siblings_at(restored_a1.id)
        assert count == 2
        assert index == 0
        other = next(s for s in snapshots if s.id != restored_a1.id)
        assert other.content == "a1-prime"

        store.set_active_leaf(session.id, other.id)
        view = [m.content for m in store.messages_for_session(session.id)]
        assert view == ["u1", "a1-prime"]
    finally:
        db.close_connection()


def test_resume_falls_back_to_recent_leaf_and_repairs_pointer_when_missing():
    """No stored pointer -> resume the most-recent ``children[-1]`` branch AND
    repair the durable pointer so the next resume is exact."""
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        conversation_id, _u1, _a1, a1_prime = _persist_branched_conversation(db)
        assert db.get_conversation_active_leaf(conversation_id) is None

        store, session = _resume_into_store(db, conversation_id)

        view = [m.content for m in store.messages_for_session(session.id)]
        assert view == ["u1", "a1-prime"]
        assert db.get_conversation_active_leaf(conversation_id) == a1_prime
    finally:
        db.close_connection()


def test_resume_chains_legacy_flat_roots_into_full_transcript():
    """C1 regression: legacy flat data (every message a NULL-parent root, no
    active-leaf pointer) resumes as the FULL transcript, not truncated to the
    last row, and every message reports a single sibling (no phantom counter).

    Before the fix the active-leaf fallback walked only the LAST root, so the
    transcript collapsed to ``['a2']`` and each row rendered a bogus ``4/4``.
    """
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        conversation_id = _persist_flat_legacy_conversation(db)
        assert db.get_conversation_active_leaf(conversation_id) is None

        store, session = _resume_into_store(db, conversation_id)

        view = store.messages_for_session(session.id)
        assert [m.content for m in view] == ["u1", "a1", "u2", "a2"]
        for message in view:
            _snapshots, _index, count = store.siblings_at(message.id)
            assert count == 1
    finally:
        db.close_connection()


def test_resume_chains_flat_prefix_then_preserves_real_continuation():
    """C1 mixed case: a flat legacy prefix followed by a genuinely-parented
    continuation resumes as the full linear transcript, real subtree intact."""
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        conversation_id = _persist_mixed_legacy_then_branched_conversation(db)

        store, session = _resume_into_store(db, conversation_id)

        view = [m.content for m in store.messages_for_session(session.id)]
        assert view == ["u1", "a1", "u2", "a2", "u3", "a3"]
    finally:
        db.close_connection()


def test_resume_falls_back_when_pointer_dangles():
    """A pointer at a message that no longer exists -> same most-recent-leaf
    fallback and pointer repair as a missing pointer."""
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        conversation_id, _u1, _a1, a1_prime = _persist_branched_conversation(db)
        db.set_conversation_active_leaf(conversation_id, "deleted-message-id")

        store, session = _resume_into_store(db, conversation_id)

        view = [m.content for m in store.messages_for_session(session.id)]
        assert view == ["u1", "a1-prime"]
        assert db.get_conversation_active_leaf(conversation_id) == a1_prime
    finally:
        db.close_connection()
