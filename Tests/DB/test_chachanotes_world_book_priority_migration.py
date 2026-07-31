from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def test_world_book_entries_priority_migrate_v20_to_v21(tmp_path):
    db_path = tmp_path / "chacha.sqlite"
    db = CharactersRAGDB(str(db_path), client_id="test-client")
    conn = db.get_connection()
    # Simulate a V20-shaped DB: drop the V21 additions.
    conn.execute("DROP TRIGGER IF EXISTS world_book_entries_sync_create")
    conn.execute("DROP TRIGGER IF EXISTS world_book_entries_sync_update")
    conn.execute("ALTER TABLE world_book_entries DROP COLUMN priority")
    # A V20 fixture also predates the V27->V28 character-authority column.
    conn.execute("ALTER TABLE conversations DROP COLUMN assistant_authority_id")
    # A v20 fixture must not retain citation tables introduced at v26→v27.
    for table in (
        "rag_artifact_owner_operations",
        "rag_artifact_owner_leases",
        "rag_source_observations",
        "rag_message_trace_owners",
        "rag_trace_evidence_refs",
        "rag_answer_attempt_payloads",
        "rag_evidence_runs",
        "rag_citation_traces",
        "rag_evidence_snapshots",
        "rag_payload_tombstones",
        "rag_legacy_migration_journal",
        "rag_identity_context",
    ):
        conn.execute(f"DROP TABLE {table}")
    conn.execute(
        "UPDATE db_schema_version SET version = 20 WHERE schema_name = ?",
        (db._SCHEMA_NAME,),
    )
    conn.commit()
    db.close_connection()

    # Reopen → auto-migrates V20→V21.
    migrated = CharactersRAGDB(str(db_path), client_id="test-client")
    mconn = migrated.get_connection()
    version = mconn.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (migrated._SCHEMA_NAME,),
    ).fetchone()
    # A simulated-V20 DB migrates all the way to the current version (which keeps
    # advancing as later migrations are added), not a hardcoded 21.
    assert version["version"] == migrated._CURRENT_SCHEMA_VERSION
    cols = {
        r[1] for r in mconn.execute("PRAGMA table_info(world_book_entries)").fetchall()
    }
    assert "priority" in cols
    create_sql = mconn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'world_book_entries_sync_create'"
    ).fetchone()["sql"]
    assert "priority" in create_sql
    update_sql = mconn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'world_book_entries_sync_update'"
    ).fetchone()["sql"]
    assert "priority" in update_sql


def test_fresh_db_has_priority_column_and_triggers(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "fresh.sqlite"), client_id="test-client")
    conn = db.get_connection()
    cols = {
        r[1] for r in conn.execute("PRAGMA table_info(world_book_entries)").fetchall()
    }
    assert "priority" in cols
    create_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'world_book_entries_sync_create'"
    ).fetchone()["sql"]
    assert "priority" in create_sql
