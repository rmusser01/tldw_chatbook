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
