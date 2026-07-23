import pytest
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _db(tmp_path):
    return CharactersRAGDB(str(tmp_path / "c.db"), client_id="test-client")


def test_fresh_db_is_v24_with_active_leaf_column(tmp_path):
    db = _db(tmp_path)
    with db.get_connection() as conn:
        version = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = 'rag_char_chat_schema'"
        ).fetchone()["version"]
        cols = {row[1] for row in conn.execute("PRAGMA table_info(conversations)").fetchall()}
    assert version == 24
    assert "active_leaf_message_id" in cols


def test_active_leaf_roundtrip_and_default_null(tmp_path):
    db = _db(tmp_path)
    conv_id = db.add_conversation({"title": "t", "character_id": None})
    assert db.get_conversation_active_leaf(conv_id) is None
    db.set_conversation_active_leaf(conv_id, "msg-123")
    assert db.get_conversation_active_leaf(conv_id) == "msg-123"
    db.set_conversation_active_leaf(conv_id, None)
    assert db.get_conversation_active_leaf(conv_id) is None


def test_active_leaf_write_does_not_bump_version_or_emit_sync(tmp_path):
    db = _db(tmp_path)
    conv_id = db.add_conversation({"title": "t", "character_id": None})
    with db.get_connection() as conn:
        v_before = conn.execute(
            "SELECT version FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()["version"]
        sync_before = conn.execute(
            "SELECT COUNT(*) AS n FROM sync_log WHERE entity_id = ?", (conv_id,)
        ).fetchone()["n"]
    db.set_conversation_active_leaf(conv_id, "msg-abc")
    with db.get_connection() as conn:
        v_after = conn.execute(
            "SELECT version FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()["version"]
        sync_after = conn.execute(
            "SELECT COUNT(*) AS n FROM sync_log WHERE entity_id = ?", (conv_id,)
        ).fetchone()["n"]
    assert v_after == v_before, "active-leaf write must not bump version"
    assert sync_after == sync_before, "active-leaf write must not emit a sync_log row"
