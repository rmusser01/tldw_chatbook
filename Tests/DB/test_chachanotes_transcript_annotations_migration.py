"""V39 -> V40 transcript_annotations migration and accessors (task-17169).

The spec sketch (2026-08-14 console-selection design, "Persistence") names
the anchor column ``session_id``, but native Console session ids are
per-process and do not survive restart -- anchoring on them would orphan
every annotation on reload, which is exactly what the spec's own row_key
paragraph forbids ("a runtime identity regenerated on session reload would
orphan annotations"). The durable identity for a console session is its
persisted conversation, so the column is ``conversation_id``.

Local-only like the trajectory sidecar: sync triggers in this DB are opt-in
per table, and none are added for annotations.
"""

from __future__ import annotations

from pathlib import Path

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

SCHEMA_NAME = "rag_char_chat_schema"

ANNOTATION_COLUMNS = {
    "annotation_id",
    "conversation_id",
    "row_key",
    "message_id",
    "quote_text",
    "comment",
    "created_at",
    "updated_at",
    "deleted",
}


def _version(connection) -> int:
    row = connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (SCHEMA_NAME,),
    ).fetchone()
    return int(row[0])


def _conversation(db: CharactersRAGDB) -> str:
    conv_id = "conv-anno-1"
    db.add_conversation({"id": conv_id, "title": "Annotated"})
    return conv_id


def test_fresh_db_lands_on_v40_with_the_annotations_table(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "test.db", client_id="test")
    try:
        connection = db.get_connection()
        assert _version(connection) == 40
        cols = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(transcript_annotations)")
        }
        assert ANNOTATION_COLUMNS <= cols
        # Anchor lookup index: (conversation_id, row_key).
        indexes = {
            row["name"]
            for row in connection.execute("PRAGMA index_list(transcript_annotations)")
        }
        assert any("conv_row" in name for name in indexes), indexes
        # Local-only: no sync trigger may mention the table.
        triggers = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
                " AND sql LIKE '%transcript_annotations%'"
            )
        }
        assert triggers == set()
    finally:
        db.close()


def test_annotation_round_trip_and_soft_delete(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "test.db", client_id="test")
    try:
        conv_id = _conversation(db)
        annotation_id = db.upsert_transcript_annotation(
            conversation_id=conv_id,
            row_key="message:m-1",
            message_id="m-1",
            quote_text="the retry loop",
            comment="tighten error paths",
        )
        assert annotation_id

        rows = db.get_transcript_annotations(conv_id)
        assert [row["comment"] for row in rows] == ["tighten error paths"]
        assert rows[0]["row_key"] == "message:m-1"
        assert rows[0]["quote_text"] == "the retry loop"

        assert db.soft_delete_transcript_annotation(annotation_id) is True
        assert db.get_transcript_annotations(conv_id) == []
        # Idempotent: deleting an already-deleted (or unknown) id is False.
        assert db.soft_delete_transcript_annotation(annotation_id) is False
    finally:
        db.close()


def test_two_comments_on_one_row_both_survive(tmp_path: Path) -> None:
    """Repeated review of the same row accumulates -- upsert here means
    insert-or-update BY annotation id, never silently replacing a different
    annotation that happens to share the anchor."""
    db = CharactersRAGDB(tmp_path / "test.db", client_id="test")
    try:
        conv_id = _conversation(db)
        first = db.upsert_transcript_annotation(
            conversation_id=conv_id,
            row_key="message:m-1",
            message_id="m-1",
            quote_text="q",
            comment="first pass",
        )
        second = db.upsert_transcript_annotation(
            conversation_id=conv_id,
            row_key="message:m-1",
            message_id="m-1",
            quote_text="q",
            comment="second pass",
        )
        assert first != second
        rows = db.get_transcript_annotations(conv_id)
        assert sorted(row["comment"] for row in rows) == ["first pass", "second pass"]
    finally:
        db.close()


def test_editing_an_annotation_updates_in_place(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "test.db", client_id="test")
    try:
        conv_id = _conversation(db)
        annotation_id = db.upsert_transcript_annotation(
            conversation_id=conv_id,
            row_key="message:m-1",
            message_id="m-1",
            quote_text="q",
            comment="draft",
        )
        db.upsert_transcript_annotation(
            conversation_id=conv_id,
            row_key="message:m-1",
            message_id="m-1",
            quote_text="q",
            comment="final",
            annotation_id=annotation_id,
        )
        rows = db.get_transcript_annotations(conv_id)
        assert [row["comment"] for row in rows] == ["final"]
    finally:
        db.close()


def test_annotations_can_be_filtered_to_one_anchor(tmp_path: Path) -> None:
    """The notes modal reads ONE message's annotations; without the filter a
    heavily annotated conversation is read in full and discarded in Python."""
    db = CharactersRAGDB(tmp_path / "test.db", client_id="test")
    try:
        conv_id = _conversation(db)
        for msg in ("m-1", "m-1", "m-2"):
            db.upsert_transcript_annotation(
                conversation_id=conv_id,
                row_key=f"message:{msg}",
                message_id=msg,
                quote_text="q",
                comment=f"note for {msg}",
            )

        assert len(db.get_transcript_annotations(conv_id)) == 3
        only_m1 = db.get_transcript_annotations(conv_id, message_id="m-1")
        assert len(only_m1) == 2
        assert {row["message_id"] for row in only_m1} == {"m-1"}
        # Soft-deleted rows stay excluded through the filtered path too.
        db.soft_delete_transcript_annotation(only_m1[0]["annotation_id"])
        assert len(db.get_transcript_annotations(conv_id, message_id="m-1")) == 1
    finally:
        db.close()
