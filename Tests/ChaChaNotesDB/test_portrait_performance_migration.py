"""The real preceding schema upgrades atomically and remains backup-admissible."""

import sqlite3

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from Tests.Character_Chat.test_portrait_performance import png
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, SchemaError


def test_v82_upgrade_preserves_original_artwork_and_has_no_sync_triggers(tmp_path):
    path = tmp_path / "historical.db"
    with chachanotes_db_at_version(path, 82) as old:
        cid = old.add_character_card({"name": "Ada", "image": png()})
        assert (
            old.execute_query(
                "SELECT count(*) FROM sqlite_schema WHERE name='character_portrait_performances'"
            ).fetchone()[0]
            == 0
        )
    db = CharactersRAGDB(path, "upgrade")
    try:
        assert (
            db._get_db_version(db.get_connection())
            == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        )
        assert db.get_character_card_by_id(cid)["image"] == png()
        assert db.execute_query("PRAGMA foreign_key_check").fetchall() == []
        assert (
            db.execute_query(
                "SELECT name FROM sqlite_schema WHERE type='trigger' AND tbl_name IN ('character_portrait_performances','character_portrait_performance_frames')"
            ).fetchall()
            == []
        )
        from tldw_chatbook.DB.recovery_core_schema import CHACHANOTES_V85_SCHEMAS

        catalog = tuple(
            row[0]
            for row in db.execute_query(
                "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
            )
        )
        assert catalog in CHACHANOTES_V85_SCHEMAS
    finally:
        db.close()


def test_failed_migration_rolls_back_tables_and_version(tmp_path, monkeypatch):
    path = tmp_path / "failed.db"
    with chachanotes_db_at_version(path, 82):
        pass
    original = CharactersRAGDB._execute_migration_statements

    def fail(self, cursor, sql, label):
        original(self, cursor, sql, label)
        if label == "V82→V83":
            raise sqlite3.OperationalError("fixture migration failure")

    monkeypatch.setattr(CharactersRAGDB, "_execute_migration_statements", fail)
    with pytest.raises(SchemaError):
        CharactersRAGDB(path, "failure")
    with sqlite3.connect(path) as conn:
        assert (
            conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'"
            ).fetchone()[0]
            == 82
        )
        assert (
            conn.execute(
                "SELECT name FROM sqlite_schema WHERE name LIKE 'character_portrait_perform%'"
            ).fetchall()
            == []
        )
