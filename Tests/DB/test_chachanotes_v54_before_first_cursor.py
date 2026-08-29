from pathlib import Path

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def test_v54_adds_local_before_message_cursor(tmp_path: Path) -> None:
    path = tmp_path / "v53.sqlite"
    with chachanotes_db_at_version(path, 53):
        pass

    upgraded = CharactersRAGDB(path, client_id="v54-upgrade")
    try:
        with upgraded.get_connection() as conn:
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(conversations)")
            }
            version = conn.execute(
                "SELECT version FROM db_schema_version "
                "WHERE schema_name = 'rag_char_chat_schema'"
            ).fetchone()[0]

        assert version == CharactersRAGDB._CURRENT_SCHEMA_VERSION == 54
        assert "active_leaf_before_message_id" in columns
    finally:
        upgraded.close_connection()


def test_v54_reenters_when_column_exists_but_stamp_is_v53(tmp_path: Path) -> None:
    path = tmp_path / "partial.sqlite"
    with chachanotes_db_at_version(path, 53) as db:
        with db.transaction() as cursor:
            cursor.execute(
                "ALTER TABLE conversations "
                "ADD COLUMN active_leaf_before_message_id TEXT"
            )

    recovered = CharactersRAGDB(path, client_id="v54-recover")
    try:
        assert recovered._get_db_version(recovered.get_connection()) == 54
    finally:
        recovered.close_connection()


def test_cursor_round_trip_and_scalar_compatibility(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "cursor.sqlite", client_id="cursor")
    try:
        conversation_id = db.add_conversation({"title": "Cursor"})

        assert db.get_conversation_active_cursor(conversation_id) == (None, None)
        assert (
            db.set_conversation_active_cursor(
                conversation_id,
                active_leaf_message_id=None,
                before_message_id="root-user",
            )
            is True
        )
        assert db.get_conversation_active_cursor(conversation_id) == (
            None,
            "root-user",
        )

        assert db.set_conversation_active_leaf(conversation_id, "assistant") is None
        assert db.get_conversation_active_cursor(conversation_id) == (
            "assistant",
            None,
        )
        assert db.get_conversation_active_leaf(conversation_id) == "assistant"
        assert (
            db.set_conversation_active_cursor(
                "missing",
                active_leaf_message_id=None,
                before_message_id="root-user",
            )
            is False
        )
    finally:
        db.close_connection()


def test_cursor_get_and_set_ignore_missing_or_deleted_conversations(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "deleted.sqlite", client_id="deleted")
    try:
        conversation_id = db.add_conversation({"title": "Deleted"})
        assert db.soft_delete_conversation(conversation_id, expected_version=1) is True

        assert db.get_conversation_active_cursor("missing") == (None, None)
        assert db.get_conversation_active_cursor(conversation_id) == (None, None)
        assert (
            db.set_conversation_active_cursor(
                conversation_id,
                active_leaf_message_id="assistant",
                before_message_id=None,
            )
            is False
        )
    finally:
        db.close_connection()


def test_cursor_write_is_local_only(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "local.sqlite", client_id="local")
    try:
        conversation_id = db.add_conversation({"title": "Local"})
        with db.get_connection() as conn:
            before = conn.execute(
                "SELECT version, last_modified FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            sync_count_before = conn.execute("SELECT COUNT(*) FROM sync_log").fetchone()[
                0
            ]

        assert (
            db.set_conversation_active_cursor(
                conversation_id,
                active_leaf_message_id=None,
                before_message_id="root-user",
            )
            is True
        )

        with db.get_connection() as conn:
            after = conn.execute(
                "SELECT version, last_modified FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            sync_count_after = conn.execute("SELECT COUNT(*) FROM sync_log").fetchone()[
                0
            ]

        assert tuple(after) == tuple(before)
        assert sync_count_after == sync_count_before
    finally:
        db.close_connection()
