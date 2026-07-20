from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def test_world_book_entries_regex_migrate_v21_to_v22(tmp_path):
    db_path = tmp_path / "chacha.sqlite"
    db = CharactersRAGDB(str(db_path), client_id="test-client")
    conn = db.get_connection()
    # Simulate a V21-shaped DB: drop the V22 additions.
    conn.execute("DROP TRIGGER IF EXISTS world_book_entries_sync_create")
    conn.execute("DROP TRIGGER IF EXISTS world_book_entries_sync_update")
    conn.execute("ALTER TABLE world_book_entries DROP COLUMN regex")
    conn.execute(
        "UPDATE db_schema_version SET version = 21 WHERE schema_name = ?",
        (db._SCHEMA_NAME,),
    )
    conn.commit()
    db.close_connection()

    migrated = CharactersRAGDB(str(db_path), client_id="test-client")
    mconn = migrated.get_connection()
    version = mconn.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (migrated._SCHEMA_NAME,),
    ).fetchone()
    assert version["version"] == migrated._CURRENT_SCHEMA_VERSION == 22
    cols = {r[1] for r in mconn.execute("PRAGMA table_info(world_book_entries)").fetchall()}
    assert "regex" in cols
    for trig in ("world_book_entries_sync_create", "world_book_entries_sync_update"):
        sql = mconn.execute(
            "SELECT sql FROM sqlite_master WHERE name = ?", (trig,)
        ).fetchone()["sql"]
        assert "regex" in sql


def test_fresh_db_has_regex_column_and_triggers(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "fresh.sqlite"), client_id="test-client")
    conn = db.get_connection()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(world_book_entries)").fetchall()}
    assert "regex" in cols
    create_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'world_book_entries_sync_create'"
    ).fetchone()["sql"]
    assert "regex" in create_sql
