from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _db(tmp_path):
    return CharactersRAGDB(str(tmp_path / "c.db"), client_id="test-client")


def test_fresh_db_is_v27_with_context_summary_columns(tmp_path):
    db = _db(tmp_path)
    with db.get_connection() as conn:
        version = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = 'rag_char_chat_schema'"
        ).fetchone()["version"]
        cols = {row[1] for row in conn.execute("PRAGMA table_info(conversations)").fetchall()}
    # Fresh databases always reach the current schema, not merely the version
    # where these columns were introduced.
    assert version == 27
    assert "context_summary" in cols
    assert "summary_boundary_message_id" in cols


def test_context_summary_roundtrip_and_default_null(tmp_path):
    db = _db(tmp_path)
    conv_id = db.add_conversation({"title": "t", "character_id": None})
    assert db.get_conversation_context_summary(conv_id) == (None, None)
    db.set_conversation_context_summary(conv_id, "earlier turns recap", "msg-123")
    assert db.get_conversation_context_summary(conv_id) == ("earlier turns recap", "msg-123")
    db.set_conversation_context_summary(conv_id, None, None)
    assert db.get_conversation_context_summary(conv_id) == (None, None)


def test_context_summary_write_does_not_bump_version_or_emit_sync(tmp_path):
    db = _db(tmp_path)
    conv_id = db.add_conversation({"title": "t", "character_id": None})
    with db.get_connection() as conn:
        v_before = conn.execute(
            "SELECT version FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()["version"]
        sync_before = conn.execute(
            "SELECT COUNT(*) AS n FROM sync_log WHERE entity_id = ?", (conv_id,)
        ).fetchone()["n"]
    db.set_conversation_context_summary(conv_id, "recap", "msg-abc")
    with db.get_connection() as conn:
        v_after = conn.execute(
            "SELECT version FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()["version"]
        sync_after = conn.execute(
            "SELECT COUNT(*) AS n FROM sync_log WHERE entity_id = ?", (conv_id,)
        ).fetchone()["n"]
    assert v_after == v_before, "context-summary write must not bump version"
    assert sync_after == sync_before, "context-summary write must not emit a sync_log row"


def test_get_context_summary_missing_conversation_returns_none_pair(tmp_path):
    db = _db(tmp_path)
    assert db.get_conversation_context_summary("does-not-exist") == (None, None)
