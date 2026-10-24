"""A real v82 attachment survives the executed next-version migration."""

import sqlite3

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from Tests.Character_Chat.test_portrait_compositing import scene
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, SchemaError


def test_v82_upgrade_preserves_old_six_frames_and_admits_actual_schema(tmp_path):
    base, _, components = scene()
    from Tests.Character_Chat.test_portrait_compositing import core

    _, recipe, _ = scene()
    frames = core().assemble_components(base, recipe, components)
    path = tmp_path / "v82.db"
    with chachanotes_db_at_version(path, 82) as old:
        cid = old.add_character_card({"name": "Old set", "image": base})
        import hashlib

        with old.transaction() as cursor:
            cursor.execute(
                "INSERT INTO character_portrait_performances (character_id,expression_key,source_sha256,source_kind,source_ref_json,revision,active) VALUES (?,'neutral',?,'portrait','[]',1,1)",
                (cid, hashlib.sha256(base).hexdigest()),
            )
            parent = cursor.lastrowid
            cursor.executemany(
                "INSERT INTO character_portrait_performance_frames VALUES (?,?,?,?,?,?)",
                [(parent, f.role, f.png, f.width, f.height, f.sha256) for f in frames],
            )
    db = CharactersRAGDB(path, "upgraded")
    try:
        assert db._get_db_version(db.get_connection()) == 83
        record = db.get_character_portrait_performance(
            character_id=cid, expression_key="neutral"
        )
        assert (
            record.revision == 1
            and record.frames == frames
            and record.composition_recipe is None
        )
        assert (
            db.get_character_portrait_regions(
                character_id=cid, expression_key="neutral"
            )
            is None
        )
        assert db.execute_query("PRAGMA foreign_key_check").fetchall() == []
        from tldw_chatbook.DB.recovery_core_schema import CHACHANOTES_V83_SCHEMAS

        actual = tuple(
            row[0]
            for row in db.execute_query(
                "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
            )
        )
        assert actual in CHACHANOTES_V83_SCHEMAS
        assert (
            db.execute_query(
                "SELECT name FROM sqlite_schema WHERE type='trigger' AND tbl_name='character_portrait_regions'"
            ).fetchall()
            == []
        )
    finally:
        db.close()


def test_region_migration_failure_rolls_back_added_column_table_and_version(
    tmp_path, monkeypatch
):
    path = tmp_path / "failed.db"
    with chachanotes_db_at_version(path, 82):
        pass
    original = CharactersRAGDB._execute_migration_statements

    def fail(self, cursor, sql, label):
        original(self, cursor, sql, label)
        if label == "V82→V83":
            raise sqlite3.OperationalError("injected migration failure")

    monkeypatch.setattr(CharactersRAGDB, "_execute_migration_statements", fail)
    with pytest.raises(SchemaError):
        CharactersRAGDB(path, "failed")
    with sqlite3.connect(path) as conn:
        assert (
            conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'"
            ).fetchone()[0]
            == 82
        )
        assert not conn.execute(
            "SELECT name FROM sqlite_schema WHERE name='character_portrait_regions'"
        ).fetchall()
        assert "composition_recipe_json" not in {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(character_portrait_performances)"
            )
        }
