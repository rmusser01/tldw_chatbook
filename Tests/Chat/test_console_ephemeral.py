"""Temporary (non-persisted) Console conversations: vocabulary and blocked actions."""

import pytest

from tldw_chatbook.Chat.console_ephemeral import (
    EPHEMERAL_BLOCKED_ACTIONS,
    TEMPORARY_LABEL,
    blocked_reason,
)
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.rag_scope import RagScope, ScopeItem, SOURCE_TYPE_MEDIA
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.mark.unit
def test_blocked_reason_only_applies_to_temporary_sessions():
    """A normal chat blocks nothing; a temporary one blocks the audited sinks."""
    for action_id in EPHEMERAL_BLOCKED_ACTIONS:
        assert blocked_reason(action_id, ephemeral=False) is None
        reason = blocked_reason(action_id, ephemeral=True)
        assert isinstance(reason, str) and reason.strip()

    assert blocked_reason("send", ephemeral=True) is None


@pytest.mark.unit
def test_blocked_reasons_name_the_artifact_not_the_feature():
    """Each reason says what would hit disk -- 'disabled' alone teaches nothing."""
    for action_id, reason in EPHEMERAL_BLOCKED_ACTIONS.items():
        assert "temporary chat" in reason, action_id
        assert reason == reason.strip()


@pytest.mark.unit
def test_user_facing_copy_never_overstates_the_guarantee():
    """The promise is local durability only -- not privacy, not anonymity."""
    forbidden = ("private", "anonym", "untracked", "incognito", "secure")
    copy = " ".join([TEMPORARY_LABEL, *EPHEMERAL_BLOCKED_ACTIONS.values()]).lower()
    for word in forbidden:
        assert word not in copy, f"copy overstates the guarantee: {word!r}"


def _row_counts(db: CharactersRAGDB) -> tuple[int, int]:
    """Return (conversations, messages) row counts straight from SQLite."""
    conn = db.get_connection()
    conversations = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
    messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    return conversations, messages


def _run_a_chat(store: ConsoleChatStore, session_id: str) -> None:
    """Drive one complete exchange through the store."""
    store.append_message(
        session_id, role=ConsoleMessageRole.USER, content="hello", persist=True
    )
    store.append_message(
        session_id, role=ConsoleMessageRole.ASSISTANT, content="hi there", persist=True
    )
    store.persist_session_if_needed(session_id)


@pytest.mark.unit
def test_temporary_session_writes_no_rows_while_a_normal_one_does(tmp_path):
    """The gate holds -- proven against a control that DOES write.

    A harness with ``persistence=None`` would pass the "no rows" half of
    this trivially, which is why the normal-session half runs first in the
    same database with the same calls.
    """
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))

        # CONTROL: a normal session must write rows here.
        baseline = _row_counts(db)
        normal = store.create_session(title="Normal chat")
        _run_a_chat(store, normal.id)
        after_normal = _row_counts(db)
        assert after_normal[0] == baseline[0] + 1, "control wrote no conversation row"
        assert after_normal[1] > baseline[1], "control wrote no message rows"
        assert normal.persisted_conversation_id is not None

        # SUBJECT: a temporary session must write nothing.
        temporary = store.create_session(title="Temporary chat", ephemeral=True)
        _run_a_chat(store, temporary.id)
        assert _row_counts(db) == after_normal
        assert temporary.persisted_conversation_id is None
        assert store.persist_session_if_needed(temporary.id) is None

        # The transcript is still fully present in memory.
        assert [m.content for m in store.messages_for_session(temporary.id)] == [
            "hello",
            "hi there",
        ]

        # Closing the tab -- the ordinary way a temporary chat ends -- must
        # not flush anything on the way out.
        store.close_session(temporary.id)
        assert _row_counts(db) == after_normal
    finally:
        db.close()


@pytest.mark.unit
def test_restore_persisted_session_refuses_to_open_the_second_door(tmp_path):
    """``restore_persisted_session`` assigns the id directly -- it must not
    be reachable with ``ephemeral`` set, or the gate has a bypass."""
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        with pytest.raises(ValueError, match="temporary"):
            store.restore_persisted_session(
                title="Restored",
                workspace_id=None,
                persisted_conversation_id="conv-1",
                all_nodes=[],
                ephemeral=True,
            )
    finally:
        db.close()


@pytest.mark.unit
def test_ephemeral_gate_wins_even_if_a_persisted_id_is_already_set(tmp_path):
    """Pins the *placement* of the ephemeral guard, not just its behavior.

    Every reachable code path today keeps ``ephemeral`` and
    ``persisted_conversation_id`` mutually exclusive, so a gate placed
    before OR after the already-persisted check currently produces the
    same observable result for every other test in this file. That means
    none of them would notice if a future refactor slid the
    ``if session.ephemeral`` check down below
    ``if session.persisted_conversation_id is not None`` in
    ``persist_session_if_needed`` -- the suite would stay green while the
    fail-safe silently stopped covering Task 3 (promotion) and Task 4
    (screen-state restore), the two places that will legitimately need to
    reason about a session that is briefly in both states at once.

    This test manufactures that forbidden state by hand -- ``ephemeral``
    and a hand-set ``persisted_conversation_id`` together, reachable only
    by a bug or a future refactor, never through the public API -- and
    asserts the ephemeral check still wins. If the guard is ever moved
    below the already-persisted check, this is the test that catches it;
    do not delete it as redundant with the proof test above.
    """
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.create_session(title="Corrupted state")
        # Reach past the public API to manufacture the forbidden state --
        # `create_session` and `restore_persisted_session` both refuse to
        # produce this combination.
        session.ephemeral = True
        session.persisted_conversation_id = "conv-hand-set"

        assert store.persist_session_if_needed(session.id) is None
    finally:
        db.close()


@pytest.mark.unit
def test_promotion_writes_every_message_in_order(tmp_path):
    """Saving a temporary chat persists exactly what is on screen.

    This session's tree has no branches, so the active path IS the whole
    tree and this test cannot by itself distinguish active-path-only
    promotion from whole-tree promotion -- see
    ``test_promotion_writes_every_node_including_off_path_branches`` below
    for the branching case that does.
    """
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.create_session(title="Temporary chat", ephemeral=True)
        _run_a_chat(store, session.id)
        assert _row_counts(db) == (0, 0)

        conversation_id = store.promote_ephemeral_session(session.id)

        assert conversation_id is not None
        assert session.ephemeral is False
        assert session.persisted_conversation_id == conversation_id
        assert _row_counts(db) == (1, 2)
        persisted = [
            db.get_message_by_id(m.persisted_message_id)["content"]
            for m in store.messages_for_session(session.id)
        ]
        assert persisted == ["hello", "hi there"]
    finally:
        db.close()


@pytest.mark.unit
def test_promotion_writes_every_node_including_off_path_branches(tmp_path):
    """Saving must not silently drop history still reachable by swiping back.

    Regenerating (``create_sibling``) leaves the previous assistant reply
    off the active path but still a real tree node, reachable via
    ``set_active_leaf``. A normal (never-temporary) conversation persists
    that node like any other; a promoted temporary one must come out the
    same way -- otherwise regenerating twice and then saving would silently
    erase the earlier answers, making the promoted conversation unlike one
    that had been saved from the start.
    """
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.create_session(title="Temporary chat", ephemeral=True)
        store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="hello", persist=True
        )
        original_reply = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="hi there",
            persist=True,
        )
        regenerated_reply = store.create_sibling(
            original_reply.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="hi again",
            persist=True,
        )
        assert _row_counts(db) == (0, 0)
        # The active view follows the regenerated branch; the original
        # reply is off-path but still a tree node.
        assert [m.content for m in store.messages_for_session(session.id)] == [
            "hello",
            "hi again",
        ]

        conversation_id = store.promote_ephemeral_session(session.id)

        assert conversation_id is not None
        # 1 conversation, 3 messages: "hello" plus BOTH assistant replies.
        assert _row_counts(db) == (1, 3)
        all_nodes = store._nodes_by_session[session.id]
        assert len(all_nodes) == 3
        assert all(
            node.persisted_message_id is not None for node in all_nodes.values()
        ), "every tree node must be persisted, not just the active path"
        persisted_contents = {
            db.get_message_by_id(node.persisted_message_id)["content"]
            for node in all_nodes.values()
        }
        assert persisted_contents == {"hello", "hi there", "hi again"}

        # Swipe-back must still work after saving: switching the active leaf
        # to the off-path (now-persisted) original reply must not raise and
        # must surface its persisted content.
        store.set_active_leaf(session.id, original_reply.id)
        assert [m.content for m in store.messages_for_session(session.id)] == [
            "hello",
            "hi there",
        ]
    finally:
        db.close()


@pytest.mark.unit
def test_promotion_preserves_persisted_parent_child_structure(tmp_path):
    """The persisted tree must connect exactly like the in-memory one.

    Builds a tree deep enough (root -> reply -> user turn -> two sibling
    replies) that writing a node before its parent is persisted would
    either strand it as a bogus root or -- worse -- silently attach it to
    the wrong, already-persisted ancestor further up the chain. Comparing
    every persisted row's ``parent_message_id`` against the in-memory
    native parent's OWN persisted id (translated through the same
    node) catches both failure modes; a same-order-as-creation write
    produces neither.
    """
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.create_session(title="Temporary chat", ephemeral=True)
        root = store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="U1", persist=True
        )
        reply = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="A1", persist=True
        )
        turn2 = store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="U2", persist=True
        )
        branch_a = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="A2a", persist=True
        )
        branch_b = store.create_sibling(
            branch_a.id, role=ConsoleMessageRole.ASSISTANT, content="A2b", persist=True
        )

        conversation_id = store.promote_ephemeral_session(session.id)

        assert conversation_id is not None
        assert _row_counts(db) == (1, 5)
        rows_by_persisted_id = {
            row["id"]: row
            for row in db.get_messages_for_conversation(conversation_id, limit=100)
        }
        assert len(rows_by_persisted_id) == 5

        all_nodes = store._nodes_by_session[session.id]
        for native_id, node in all_nodes.items():
            assert node.persisted_message_id in rows_by_persisted_id
            native_parent_id = store._native_parent_by_message[native_id]
            expected_parent_persisted_id = (
                all_nodes[native_parent_id].persisted_message_id
                if native_parent_id is not None
                else None
            )
            actual_parent_persisted_id = rows_by_persisted_id[
                node.persisted_message_id
            ]["parent_message_id"]
            assert actual_parent_persisted_id == expected_parent_persisted_id, (
                f"node {node.content!r} persisted with the wrong parent"
            )

        # Spot-check the branch point explicitly (the case an ordering bug
        # would most likely get wrong -- turn2 has TWO persisted children).
        # ``root``/``reply``/``turn2`` are snapshots taken BEFORE promotion
        # (``append_message`` returns a point-in-time copy via
        # ``dataclasses.replace`` -- see ``_snapshot``), so their own
        # ``persisted_message_id`` is stale; only their stable native ``.id``
        # is reused here, looked up fresh through the live ``all_nodes``
        # mapping captured after promotion above.
        turn2_persisted_id = all_nodes[turn2.id].persisted_message_id
        children_of_turn2 = {
            row["content"]
            for row in rows_by_persisted_id.values()
            if row["parent_message_id"] == turn2_persisted_id
        }
        assert children_of_turn2 == {"A2a", "A2b"}
        assert (
            rows_by_persisted_id[all_nodes[reply.id].persisted_message_id][
                "parent_message_id"
            ]
            == all_nodes[root.id].persisted_message_id
        )
        assert (
            rows_by_persisted_id[all_nodes[turn2.id].persisted_message_id][
                "parent_message_id"
            ]
            == all_nodes[reply.id].persisted_message_id
        )
    finally:
        db.close()


@pytest.mark.unit
def test_promotion_is_idempotent(tmp_path):
    """A second Save writes nothing more and does not raise."""
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.create_session(title="Temporary chat", ephemeral=True)
        _run_a_chat(store, session.id)
        first = store.promote_ephemeral_session(session.id)
        after_first = _row_counts(db)

        assert store.promote_ephemeral_session(session.id) is None
        assert _row_counts(db) == after_first
        assert session.persisted_conversation_id == first
    finally:
        db.close()


@pytest.mark.unit
def test_failed_promotion_rolls_back_and_stays_temporary(tmp_path, monkeypatch):
    """A half-saved conversation must never be left in history.

    The failure is injected on the SECOND message write, so the conversation
    row and the first message are already in the transaction when it blows
    up -- exactly the partial state the rollback exists to undo.
    """
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        persistence = ChatPersistenceService(db)
        store = ConsoleChatStore(persistence=persistence)
        session = store.create_session(title="Temporary chat", ephemeral=True)
        _run_a_chat(store, session.id)

        calls = {"n": 0}
        real_create = persistence.create_message

        def failing_create(**kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("disk full")
            return real_create(**kwargs)

        monkeypatch.setattr(persistence, "create_message", failing_create)

        with pytest.raises(RuntimeError, match="disk full"):
            store.promote_ephemeral_session(session.id)

        assert _row_counts(db) == (0, 0), "partial conversation survived"
        assert session.ephemeral is True, "failed save left the chat persisting"
        assert session.persisted_conversation_id is None
        assert all(
            m.persisted_message_id is None
            for m in store.messages_for_session(session.id)
        )
    finally:
        db.close()


@pytest.mark.unit
def test_failed_promotion_restores_the_held_rag_scope(tmp_path, monkeypatch):
    """A failed save must not silently drop the user's scope selection.

    ``persist_session_if_needed`` flushes (and empties) the session's held
    RAG scope as soon as the conversation row is created -- before either
    message write can fail. If promotion rolls back the database but
    leaves the now-empty holder alone, the user's scope selection vanishes
    even though the chat correctly stays temporary. This is reachable in
    normal use: the Console screen puts a scope in the holder precisely
    when there is no persisted conversation, which is always true for a
    temporary chat.
    """
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        persistence = ChatPersistenceService(db)
        store = ConsoleChatStore(persistence=persistence)
        session = store.create_session(title="Temporary chat", ephemeral=True)
        _run_a_chat(store, session.id)

        scope = RagScope(items=(ScopeItem(SOURCE_TYPE_MEDIA, "doc-1"),), updated_at="t1")
        session.rag_scope_holder.set(scope)

        calls = {"n": 0}
        real_create = persistence.create_message

        def failing_create(**kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("disk full")
            return real_create(**kwargs)

        monkeypatch.setattr(persistence, "create_message", failing_create)

        with pytest.raises(RuntimeError, match="disk full"):
            store.promote_ephemeral_session(session.id)

        assert session.ephemeral is True
        assert session.rag_scope_holder.scope == scope, (
            "failed promotion must restore the held scope, not leave it empty"
        )
    finally:
        db.close()


@pytest.mark.unit
def test_promotion_restores_ephemeral_flag_if_persist_returns_none_unexpectedly(
    tmp_path, monkeypatch
):
    """Defensive: an unexpected None from persist_session_if_needed must not
    silently leave the session non-ephemeral with no persisted conversation.

    That state is exactly what the docstring warns about -- a failed save
    that silently starts persisting on the next send. Nothing in
    ``persist_session_if_needed`` reaches this today (its only None-return
    branches are already ruled out once ``ephemeral`` is cleared and
    ``self.persistence`` is known non-None), so this test forces the case
    directly to prove the rollback still fires if a future change ever adds
    one.
    """
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.create_session(title="Temporary chat", ephemeral=True)
        _run_a_chat(store, session.id)

        monkeypatch.setattr(store, "persist_session_if_needed", lambda session_id: None)

        with pytest.raises(RuntimeError):
            store.promote_ephemeral_session(session.id)

        assert session.ephemeral is True
        assert session.persisted_conversation_id is None
        assert _row_counts(db) == (0, 0)
    finally:
        db.close()
