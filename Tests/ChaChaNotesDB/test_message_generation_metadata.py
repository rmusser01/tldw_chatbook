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

def test_set_rejects_duplicate_positions(db):
    _, mid = _mk_message(db)
    with pytest.raises(ValueError):
        db.set_message_generation_metadata(mid, [_row(0), _row(0)])

def test_append_attachment_with_metadata_single_insert(db):
    _, mid = _mk_message(db)
    db.set_message_generation_metadata(mid, [_row(0)])
    pos = db.append_message_attachment_with_metadata(
        mid, data=b"png1", mime_type="image/png", generation_metadata=_row(1, seed=42))
    assert pos == 1
    got = db.get_generation_metadata_for_messages([mid])
    assert [r["position"] for r in got[mid]] == [0, 1]
    with db.transaction() as cur:
        row = cur.execute(
            "SELECT data FROM message_attachments WHERE message_id=? AND position=1",
            (mid,)).fetchone()
    assert row[0] == b"png1"

def test_append_attachment_with_metadata_bumps_version(db):
    _, mid = _mk_message(db)
    before = db.get_message_by_id(mid)
    db.append_message_attachment_with_metadata(
        mid, data=b"png1", mime_type="image/png")
    after = db.get_message_by_id(mid)
    assert after["version"] == before["version"] + 1
    assert after["last_modified"] is not None

def test_append_attachment_rejects_message_without_position_zero_image(db):
    conv_id = db.add_conversation({"title": "t"})
    mid = db.add_message({
        "conversation_id": conv_id, "sender": "assistant", "content": "no image",
    })
    with pytest.raises(ValueError):
        db.append_message_attachment_with_metadata(
            mid, data=b"png1", mime_type="image/png")

def test_swap_makes_kept_variant_position_zero(db):
    _, mid = _mk_message(db)
    db.set_message_generation_metadata(mid, [_row(0, seed=7)])
    db.append_message_attachment_with_metadata(
        mid, data=b"png1", mime_type="image/png", generation_metadata=_row(1, seed=42))
    db.swap_message_attachment_with_scalar(mid, 1)
    msg = db.get_message_by_id(mid)          # read helper: align name to real API
    assert msg["image_data"] == b"png1"       # kept variant now canonical
    with db.transaction() as cur:
        row = cur.execute(
            "SELECT data FROM message_attachments WHERE message_id=? AND position=1",
            (mid,)).fetchone()
    assert row[0] == b"png0"                  # old canonical demoted, bit-identical
    got = db.get_generation_metadata_for_messages([mid])
    by_pos = {r["position"]: r["seed"] for r in got[mid]}
    assert by_pos == {0: 42, 1: 7}            # sidecar re-keyed with the swap

def test_swap_bumps_version_and_leaves_other_attachments_untouched(db):
    _, mid = _mk_message(db)
    db.append_message_attachment_with_metadata(mid, data=b"png1", mime_type="image/png")
    db.append_message_attachment_with_metadata(mid, data=b"png2", mime_type="image/png")
    before = db.get_message_by_id(mid)
    db.swap_message_attachment_with_scalar(mid, 1)
    after = db.get_message_by_id(mid)
    assert after["version"] == before["version"] + 1
    with db.transaction() as cur:
        row = cur.execute(
            "SELECT data FROM message_attachments WHERE message_id=? AND position=2",
            (mid,)).fetchone()
    assert row[0] == b"png2"                  # untouched by the position-1 swap

def test_swap_rejects_bad_position(db):
    _, mid = _mk_message(db)
    with pytest.raises(ValueError):
        db.swap_message_attachment_with_scalar(mid, 0)
    with pytest.raises(ValueError):
        db.swap_message_attachment_with_scalar(mid, 3)   # no such row
