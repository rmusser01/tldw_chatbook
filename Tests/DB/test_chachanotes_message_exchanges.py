"""v41 message_exchanges: local-only, idempotent upsert, cascade delete
(Console Conversation Inspector, task-5).

Local-only means: no sync_log rows are ever written for this table (same
precedent as the v29->v30 usage_json / v39->v40 transcript_annotations
local-only additions), and a hard delete of the parent message cascades
straight through via the FK -- there is no soft-delete/version bookkeeping
for these rows.
"""
import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

# Matches CharactersRAGDB._SCHEMA_NAME, per the sibling migration tests
# (e.g. Tests/DB/test_chachanotes_message_usage_migration.py).
SCHEMA_NAME = "rag_char_chat_schema"


@pytest.fixture
def db():
    database = CharactersRAGDB(":memory:", client_id="message-exchanges-test")
    yield database
    database.close_connection()


def _version(connection) -> int:
    row = connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (SCHEMA_NAME,),
    ).fetchone()
    return int(row[0])


def _seed_message(db) -> str:
    """Create a conversation + message via the DB's real public API and
    return the message id, mirroring the seeding helper used by the v30
    usage_json round-trip test."""
    conv_id = db.add_conversation({"title": "t"})
    msg_id = db.add_message(
        {"conversation_id": conv_id, "sender": "user", "content": "hi"}
    )
    return msg_id


def test_append_and_read_round_trip(db):
    mid = _seed_message(db)
    rows = [
        {"run_tag": "r1", "seq": 0, "status": "complete", "abandoned": False,
         "capture_blob": b"blob0", "created_at": "2026-08-18T00:00:00Z"},
        {"run_tag": "r1", "seq": 1, "status": "stopped", "abandoned": False,
         "capture_blob": b"blob1", "created_at": "2026-08-18T00:00:01Z"},
    ]
    assert db.append_message_exchanges_local(mid, rows) == 2
    stored = db.get_message_exchanges(mid)
    assert [(r["run_tag"], r["seq"], r["capture_blob"]) for r in stored] == [
        ("r1", 0, b"blob0"), ("r1", 1, b"blob1")]


def test_upsert_idempotent_and_updates_in_place(db):
    mid = _seed_message(db)
    row = {"run_tag": "r1", "seq": 0, "status": "complete", "abandoned": False,
           "capture_blob": b"v1", "created_at": "t"}
    db.append_message_exchanges_local(mid, [row])
    db.append_message_exchanges_local(mid, [{**row, "capture_blob": b"v2", "abandoned": True}])
    stored = db.get_message_exchanges(mid)
    assert len(stored) == 1
    assert stored[0]["capture_blob"] == b"v2" and stored[0]["abandoned"]


def test_no_sync_log_rows_written(db):
    mid = _seed_message(db)
    with db.transaction() as cursor:
        before = cursor.execute("SELECT COUNT(*) FROM sync_log").fetchone()[0]
    db.append_message_exchanges_local(mid, [
        {"run_tag": "r1", "seq": 0, "status": "complete", "abandoned": False,
         "capture_blob": b"b", "created_at": "t"}])
    with db.transaction() as cursor:
        after = cursor.execute("SELECT COUNT(*) FROM sync_log").fetchone()[0]
    assert after == before


def test_hard_delete_cascades(db):
    mid = _seed_message(db)
    db.append_message_exchanges_local(mid, [
        {"run_tag": "r1", "seq": 0, "status": "complete", "abandoned": False,
         "capture_blob": b"b", "created_at": "t"}])
    with db.transaction() as cursor:
        cursor.execute("DELETE FROM messages WHERE id = ?", (mid,))
        count = cursor.execute(
            "SELECT COUNT(*) FROM message_exchanges").fetchone()[0]
    assert count == 0


def test_schema_version_is_41(db):
    # Mirrors the house sibling-version test pattern (a local `_version()`
    # helper against db_schema_version -- there is no public accessor).
    assert _version(db.get_connection()) == 41
