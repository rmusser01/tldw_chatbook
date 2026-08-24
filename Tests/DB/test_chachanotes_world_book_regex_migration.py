from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from Tests.ChaChaNotesDB.historical_bootstrap import (
    open_current_chachanotes_from_legacy,
)


def _seed_v21_database(db_path, monkeypatch) -> None:
    with monkeypatch.context() as v21_patch:
        v21_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 21)
        db = CharactersRAGDB(str(db_path), client_id="v21-seed")
        version = db.get_connection().execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (db._SCHEMA_NAME,),
        ).fetchone()
        assert version["version"] == 21
        db.get_connection().executescript(
            """
            DROP TRIGGER world_book_entries_sync_create;
            DROP TRIGGER world_book_entries_sync_update;
            ALTER TABLE world_book_entries DROP COLUMN regex;

            CREATE TRIGGER world_book_entries_sync_create
            AFTER INSERT ON world_book_entries BEGIN
              INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
              VALUES('world_book_entries', CAST(NEW.id AS TEXT), 'create', NEW.last_modified,
                     (SELECT client_id FROM world_books WHERE id = NEW.world_book_id), 1,
                     json_object('id', NEW.id, 'world_book_id', NEW.world_book_id, 'keys', NEW.keys,
                                 'content', NEW.content, 'enabled', NEW.enabled, 'position', NEW.position,
                                 'insertion_order', NEW.insertion_order, 'priority', NEW.priority,
                                 'selective', NEW.selective, 'secondary_keys', NEW.secondary_keys,
                                 'case_sensitive', NEW.case_sensitive, 'extensions', NEW.extensions,
                                 'created_at', NEW.created_at, 'last_modified', NEW.last_modified));
            END;

            CREATE TRIGGER world_book_entries_sync_update
            AFTER UPDATE ON world_book_entries
            WHEN OLD.keys IS NOT NEW.keys OR
                 OLD.content IS NOT NEW.content OR
                 OLD.enabled IS NOT NEW.enabled OR
                 OLD.position IS NOT NEW.position OR
                 OLD.insertion_order IS NOT NEW.insertion_order OR
                 OLD.priority IS NOT NEW.priority OR
                 OLD.selective IS NOT NEW.selective OR
                 OLD.secondary_keys IS NOT NEW.secondary_keys OR
                 OLD.case_sensitive IS NOT NEW.case_sensitive OR
                 OLD.extensions IS NOT NEW.extensions
            BEGIN
              INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
              VALUES('world_book_entries', CAST(NEW.id AS TEXT), 'update', NEW.last_modified,
                     (SELECT client_id FROM world_books WHERE id = NEW.world_book_id), 1,
                     json_object('id', NEW.id, 'world_book_id', NEW.world_book_id, 'keys', NEW.keys,
                                 'content', NEW.content, 'enabled', NEW.enabled, 'position', NEW.position,
                                 'insertion_order', NEW.insertion_order, 'priority', NEW.priority,
                                 'selective', NEW.selective, 'secondary_keys', NEW.secondary_keys,
                                 'case_sensitive', NEW.case_sensitive, 'extensions', NEW.extensions,
                                 'created_at', NEW.created_at, 'last_modified', NEW.last_modified));
            END;
            """
        )
        columns = {
            row[1]
            for row in db.get_connection()
            .execute("PRAGMA table_info(world_book_entries)")
            .fetchall()
        }
        assert "priority" in columns
        assert "regex" not in columns
        for trigger_name in (
            "world_book_entries_sync_create",
            "world_book_entries_sync_update",
        ):
            trigger_sql = db.get_connection().execute(
                "SELECT sql FROM sqlite_master WHERE name = ?", (trigger_name,)
            ).fetchone()["sql"]
            assert "priority" in trigger_sql
            assert "regex" not in trigger_sql
        db.close_connection()


def test_world_book_entries_regex_migrate_v21_to_v22(tmp_path, monkeypatch):
    db_path = tmp_path / "chacha.sqlite"
    _seed_v21_database(db_path, monkeypatch)

    # Reopen the genuine v21 schema with current support.
    migrated = open_current_chachanotes_from_legacy(
        db_path, client_id="test-client"
    )
    mconn = migrated.get_connection()
    version = mconn.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (migrated._SCHEMA_NAME,),
    ).fetchone()
    # A simulated-V21 DB migrates all the way to the current version (which keeps
    # advancing as later migrations are added), not a hardcoded 22.
    assert version["version"] == migrated._CURRENT_SCHEMA_VERSION
    cols = {
        r[1] for r in mconn.execute("PRAGMA table_info(world_book_entries)").fetchall()
    }
    assert "regex" in cols
    for trig in ("world_book_entries_sync_create", "world_book_entries_sync_update"):
        sql = mconn.execute(
            "SELECT sql FROM sqlite_master WHERE name = ?", (trig,)
        ).fetchone()["sql"]
        assert "regex" in sql


def test_fresh_db_has_regex_column_and_triggers(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "fresh.sqlite"), client_id="test-client")
    conn = db.get_connection()
    cols = {
        r[1] for r in conn.execute("PRAGMA table_info(world_book_entries)").fetchall()
    }
    assert "regex" in cols
    create_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'world_book_entries_sync_create'"
    ).fetchone()["sql"]
    assert "regex" in create_sql
