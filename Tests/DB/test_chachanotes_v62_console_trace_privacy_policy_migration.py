"""ChaChaNotes v61 -> v62 scoped Console trace privacy controls."""

from pathlib import Path

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def test_v61_policy_detail_survives_and_new_controls_start_inherit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace-privacy-v61.sqlite"
    original_target = CharactersRAGDB._CURRENT_SCHEMA_VERSION
    CharactersRAGDB._CURRENT_SCHEMA_VERSION = 61
    try:
        historical = CharactersRAGDB(path, client_id="trace-privacy-v61")
        conversation_id = historical.add_conversation({"title": "legacy full"})
        with historical.transaction() as cursor:
            cursor.execute(
                "INSERT INTO console_conversation_capture_policy "
                "(conversation_id, capture_detail) VALUES (?, 'full')",
                (conversation_id,),
            )
        historical.close_connection()
    finally:
        CharactersRAGDB._CURRENT_SCHEMA_VERSION = original_target

    migrated = CharactersRAGDB(path, client_id="trace-privacy-v62")
    try:
        row = migrated.get_connection().execute(
            "SELECT capture_detail, capture_enabled, pii_redaction_enabled "
            "FROM console_conversation_capture_policy WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        assert row is not None
        assert tuple(row) == ("full", None, None)
        assert migrated._get_db_version(migrated.get_connection()) == 62
    finally:
        migrated.close_connection()


def test_v62_policy_requires_at_least_one_sparse_value(tmp_path: Path) -> None:
    database = CharactersRAGDB(tmp_path / "trace-privacy-v62.sqlite", "v62-check")
    conversation_id = database.add_conversation({"title": "privacy"})
    try:
        with database.transaction() as cursor:
            try:
                cursor.execute(
                    "INSERT INTO console_conversation_capture_policy "
                    "(conversation_id) VALUES (?)",
                    (conversation_id,),
                )
            except Exception:
                pass
            else:
                raise AssertionError("empty sparse policy must be rejected")
    finally:
        database.close_connection()
