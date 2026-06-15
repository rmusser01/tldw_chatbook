from tldw_chatbook.Sync_Interop.notes_mirror import NotesMirror, MirrorRecord


def test_mirror_upsert_and_get():
    m = NotesMirror(":memory:")
    assert m.get("ds", "note_1") is None
    m.record("ds", "note_1", object_revision=1, object_hash="sha256:a", server_cursor=10)
    rec = m.get("ds", "note_1")
    assert isinstance(rec, MirrorRecord)
    assert rec.object_revision == 1
    assert rec.object_hash == "sha256:a"
    assert rec.server_cursor == 10


def test_mirror_record_is_idempotent_upsert():
    m = NotesMirror(":memory:")
    m.record("ds", "n", object_revision=1, object_hash="sha256:a", server_cursor=10)
    m.record("ds", "n", object_revision=2, object_hash="sha256:b", server_cursor=11)
    rec = m.get("ds", "n")
    assert rec.object_revision == 2 and rec.object_hash == "sha256:b" and rec.server_cursor == 11


def test_mirror_scopes_by_dataset():
    m = NotesMirror(":memory:")
    m.record("ds1", "n", object_revision=1, object_hash="h", server_cursor=1)
    assert m.get("ds2", "n") is None
