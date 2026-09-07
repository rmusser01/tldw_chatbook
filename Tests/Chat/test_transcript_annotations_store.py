"""Console store seam for transcript annotations (task-17169 slice 2).

Per the maintainer's both-homes decision: every feedback action writes a
trajectory-sidecar audit event; Comment actions ADDITIONALLY persist a
transcript_annotations row (the spec's "Comment ... additionally persists an
annotation"). This file covers the store half of that second write.
"""


from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _store_with_db(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    return db, store


def test_comment_annotation_round_trips_with_a_message_row_key(tmp_path):
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Annotations")
        conversation_id = store.persist_session_if_needed(session.id)
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="the retry loop",
            persist=True,
        )

        annotation_id = store.record_feedback_annotation(
            session.id,
            anchor_message_id=assistant.id,
            quote="the retry loop",
            comment="tighten error paths",
        )

        assert annotation_id
        rows = db.get_transcript_annotations(conversation_id)
        assert len(rows) == 1
        row = rows[0]
        assert row["row_key"] == f"message:{assistant.persisted_message_id}"
        assert row["message_id"] == assistant.persisted_message_id
        assert row["quote_text"] == "the retry loop"
        assert row["comment"] == "tighten error paths"
    finally:
        db.close()


def test_annotation_on_an_unpersisted_anchor_is_skipped(tmp_path):
    """The row_key spike's rule: no durable key, no annotation. TOOL markers
    and ephemeral messages have no persisted id to derive one from."""
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Annotations")
        store.persist_session_if_needed(session.id)
        unpersisted = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="x", persist=False
        )

        assert (
            store.record_feedback_annotation(
                session.id,
                anchor_message_id=unpersisted.id,
                quote="x",
                comment="note",
            )
            is None
        )
    finally:
        db.close()


def test_annotation_on_an_ephemeral_session_is_skipped(tmp_path):
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Ephemeral")
        message = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="x", persist=False
        )
        assert (
            store.record_feedback_annotation(
                session.id,
                anchor_message_id=message.id,
                quote="x",
                comment="note",
            )
            is None
        )
    finally:
        db.close()


def test_annotation_write_never_raises(tmp_path):
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Annotations")
        store.persist_session_if_needed(session.id)
        # Unknown anchor: must be a None, not a KeyError escaping to the
        # dispatch path.
        assert (
            store.record_feedback_annotation(
                session.id,
                anchor_message_id="no-such-message",
                quote="q",
                comment="c",
            )
            is None
        )
    finally:
        db.close()


def test_annotations_survive_a_restart(tmp_path):
    db_path = str(tmp_path / "chachanotes.sqlite")
    db = CharactersRAGDB(db_path, "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.ensure_session(title="Annotations")
        conversation_id = store.persist_session_if_needed(session.id)
        assistant = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="ok", persist=True
        )
        store.record_feedback_annotation(
            session.id,
            anchor_message_id=assistant.id,
            quote="ok",
            comment="revisit",
        )
    finally:
        db.close()

    reopened = CharactersRAGDB(db_path, "test_client")
    try:
        rows = reopened.get_transcript_annotations(conversation_id)
        assert [row["comment"] for row in rows] == ["revisit"]
    finally:
        reopened.close()
