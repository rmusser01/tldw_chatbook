import pytest

from tldw_chatbook.Sync_Interop.notes_local_store import InMemoryNotesStore
from tldw_chatbook.Sync_Interop.notes_mirror import NotesMirror
from tldw_chatbook.Sync_Interop.envelope_builder import SyncEnvelopeBuilder
from tldw_chatbook.Sync_Interop.envelope_applier import SyncEnvelopeApplier


def _builder(mirror):
    return SyncEnvelopeBuilder(
        dataset_id="ds_1", device_id="dev_1", dataset_key=b"x" * 32, notes_mirror=mirror,
    )


def test_build_notes_upsert_new_object_has_no_base():
    mirror = NotesMirror(":memory:")
    env = _builder(mirror).build_notes_note_upsert(note_id="n1", title="T", content="B")
    assert env.domain == "notes.note"
    assert env.operation == "upsert"
    assert env.object_id == "n1"
    assert env.payload == {"title": "T", "content": "B"}
    assert env.encryption_policy == "server_trusted_v1"
    assert env.encryption_metadata == {"policy": "server_trusted_v1"}
    assert env.base_object_revision is None and env.base_object_hash is None


def test_build_notes_upsert_update_uses_mirror_base():
    mirror = NotesMirror(":memory:")
    mirror.record("ds_1", "n1", object_revision=1, object_hash="sha256:prev", server_cursor=5)
    env = _builder(mirror).build_notes_note_upsert(note_id="n1", title="T2", content="B2")
    assert env.base_object_revision == 1
    assert env.base_object_hash == "sha256:prev"
    assert env.base_server_cursor == 5


def test_build_notes_tombstone_requires_mirror_base():
    mirror = NotesMirror(":memory:")
    mirror.record("ds_1", "n1", object_revision=2, object_hash="sha256:p", server_cursor=9)
    env = _builder(mirror).build_notes_note_tombstone(note_id="n1")
    assert env.operation == "tombstone" and env.deleted is True
    assert env.base_object_revision == 2 and env.base_object_hash == "sha256:p"


def test_apply_notes_upsert_creates_local_note_and_updates_mirror():
    store = InMemoryNotesStore()
    mirror = NotesMirror(":memory:")
    applier = SyncEnvelopeApplier(local_store=store, notes_mirror=mirror, dataset_id="ds_1")
    from tldw_chatbook.tldw_api import SyncV2Envelope
    env = SyncV2Envelope(
        client_envelope_id="c", dataset_id="ds_1", domain="notes.note", object_id="n1",
        operation="upsert", adapter_version=1, payload={"title": "T", "content": "B"},
        payload_hash="sha256:cur", object_revision=1, server_cursor=10,
        encryption_metadata={"policy": "server_trusted_v1"},
    )
    result = applier.apply(env)
    assert result["status"] == "applied"
    assert store.get("n1") == {"title": "T", "content": "B", "deleted": False}
    assert mirror.get("ds_1", "n1").object_revision == 1

    result2 = applier.apply(env)
    assert result2["status"] in {"applied", "noop"}
    assert store.upsert_calls == 1


def test_apply_notes_upsert_requires_mirror_and_dataset_id():
    from tldw_chatbook.tldw_api import SyncV2Envelope

    store = InMemoryNotesStore()
    applier = SyncEnvelopeApplier(local_store=store)
    env = SyncV2Envelope(
        client_envelope_id="c", dataset_id="ds_1", domain="notes.note", object_id="n1",
        operation="upsert", adapter_version=1, payload={"title": "T", "content": "B"},
        payload_hash="sha256:cur", object_revision=1, server_cursor=10,
    )

    with pytest.raises(ValueError, match="notes_mirror and dataset_id"):
        applier.apply(env)


def test_apply_notes_upsert_rejects_missing_required_payload_fields():
    from tldw_chatbook.tldw_api import SyncV2Envelope

    store = InMemoryNotesStore()
    mirror = NotesMirror(":memory:")
    applier = SyncEnvelopeApplier(local_store=store, notes_mirror=mirror, dataset_id="ds_1")
    env = SyncV2Envelope(
        client_envelope_id="c", dataset_id="ds_1", domain="notes.note", object_id="n1",
        operation="upsert", adapter_version=1, payload={"content": "B"},
        payload_hash="sha256:cur", object_revision=1, server_cursor=10,
    )

    result = applier.apply(env)

    assert result["status"] == "conflict"
    assert result["conflict"]["conflict_type"] == "invalid_notes_payload"
    assert store.get("n1") is None
    assert mirror.get("ds_1", "n1") is None


def test_apply_notes_upsert_sanitizes_payload_before_store():
    from tldw_chatbook.tldw_api import SyncV2Envelope

    store = InMemoryNotesStore()
    mirror = NotesMirror(":memory:")
    applier = SyncEnvelopeApplier(local_store=store, notes_mirror=mirror, dataset_id="ds_1")
    env = SyncV2Envelope(
        client_envelope_id="c", dataset_id="ds_1", domain="notes.note", object_id="n1",
        operation="upsert", adapter_version=1,
        payload={"title": "T\x00", "content": "<script>alert(1)</script>\x00safe"},
        payload_hash="sha256:cur", object_revision=1, server_cursor=10,
    )

    result = applier.apply(env)

    assert result["status"] == "applied"
    assert store.get("n1") == {
        "title": "T",
        "content": "&lt;script&gt;alert(1)&lt;/script&gt;safe",
        "deleted": False,
    }


def test_apply_notes_tombstone_soft_deletes():
    store = InMemoryNotesStore()
    store.upsert_note("n1", {"title": "T", "content": "B"}, object_revision=1)
    mirror = NotesMirror(":memory:")
    applier = SyncEnvelopeApplier(local_store=store, notes_mirror=mirror, dataset_id="ds_1")
    from tldw_chatbook.tldw_api import SyncV2Envelope
    env = SyncV2Envelope(
        client_envelope_id="c", dataset_id="ds_1", domain="notes.note", object_id="n1",
        operation="tombstone", adapter_version=1, deleted=True, payload={"deleted_at": "t"},
        payload_hash="sha256:t", object_revision=2, server_cursor=11,
    )
    applier.apply(env)
    assert store.get("n1")["deleted"] is True


def test_apply_drops_stale_older_revision_envelope():
    from tldw_chatbook.tldw_api import SyncV2Envelope
    store = InMemoryNotesStore()
    mirror = NotesMirror(":memory:")
    mirror.record("ds_1", "n1", object_revision=5, object_hash="sha256:newer", server_cursor=50)
    applier = SyncEnvelopeApplier(local_store=store, notes_mirror=mirror, dataset_id="ds_1")
    stale = SyncV2Envelope(
        client_envelope_id="c", dataset_id="ds_1", domain="notes.note", object_id="n1",
        operation="upsert", adapter_version=1, payload={"title": "OLD", "content": "OLD"},
        payload_hash="sha256:older", object_revision=3, server_cursor=30,
    )
    result = applier.apply(stale)
    assert result["status"] == "noop"
    assert store.upsert_calls == 0
    # mirror not rolled back
    assert mirror.get("ds_1", "n1").object_revision == 5


def test_apply_tombstone_reapply_is_noop():
    from tldw_chatbook.tldw_api import SyncV2Envelope
    store = InMemoryNotesStore()
    mirror = NotesMirror(":memory:")
    applier = SyncEnvelopeApplier(local_store=store, notes_mirror=mirror, dataset_id="ds_1")
    env = SyncV2Envelope(
        client_envelope_id="c", dataset_id="ds_1", domain="notes.note", object_id="n1",
        operation="tombstone", adapter_version=1, deleted=True, payload={"deleted_at": "t"},
        payload_hash="sha256:t", object_revision=2, server_cursor=11,
    )
    applier.apply(env)
    before = store.delete_calls
    applier.apply(env)  # re-apply
    assert store.delete_calls == before  # no second soft-delete
