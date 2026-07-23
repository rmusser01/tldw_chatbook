import pytest
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

@pytest.fixture
def db(tmp_path):
    d = CharactersRAGDB(tmp_path / "t.db", "test-client")
    yield d
    d.close_connection()

def _mk_message(db):
    conv_id = db.add_conversation({"title": "t"})
    return conv_id, db.add_message({
        "conversation_id": conv_id, "sender": "assistant",
        "content": "[image] a red dragon",
        "image_data": b"png0", "image_mime_type": "image/png",
    })

def _row(pos, seed=1):
    return {"position": pos, "prompt": "a red dragon", "negative_prompt": "blurry",
            "backend": "swarmui", "model": None, "seed": seed, "style": None,
            "params_json": "{}"}

def test_migration_reaches_v25(db):
    with db.transaction() as cur:
        v = cur.execute(
            "SELECT version FROM db_schema_version WHERE schema_name=?",
            ("rag_char_chat_schema",)).fetchone()[0]
    assert v == 25

def test_set_and_batch_get_roundtrip(db):
    _, mid = _mk_message(db)
    db.set_message_generation_metadata(mid, [_row(0, seed=7), _row(1, seed=8)])
    got = db.get_generation_metadata_for_messages([mid])
    assert [r["seed"] for r in got[mid]] == [7, 8]
    assert got[mid][0]["backend"] == "swarmui"

def test_set_is_authoritative_rewrite(db):
    _, mid = _mk_message(db)
    db.set_message_generation_metadata(mid, [_row(0), _row(1)])
    db.set_message_generation_metadata(mid, [_row(0, seed=99)])
    got = db.get_generation_metadata_for_messages([mid])
    assert len(got[mid]) == 1 and got[mid][0]["seed"] == 99

def test_cascade_delete_with_message(db):
    conv_id, mid = _mk_message(db)
    db.set_message_generation_metadata(mid, [_row(0)])
    with db.transaction() as cur:  # hard-delete to exercise ON DELETE CASCADE
        cur.execute("DELETE FROM messages WHERE id=?", (mid,))
    assert db.get_generation_metadata_for_messages([mid]) == {}
