import pytest
from unittest.mock import AsyncMock

from tldw_chatbook.Sync_Interop.notes_m1_flow import NotesM1SyncFlow
from tldw_chatbook.Sync_Interop.notes_mirror import NotesMirror
from tldw_chatbook.Sync_Interop.notes_local_store import InMemoryNotesStore
from tldw_chatbook.Sync_Interop.envelope_builder import SyncEnvelopeBuilder


@pytest.mark.asyncio
async def test_push_updates_mirror_from_accepted():
    mirror = NotesMirror(":memory:")
    store = InMemoryNotesStore()
    builder = SyncEnvelopeBuilder(dataset_id="ds_1", device_id="dev_1", dataset_key=b"x"*32, notes_mirror=mirror)
    env = builder.build_notes_note_upsert(note_id="n1", title="T", content="B")
    client = AsyncMock()
    client.push_sync_v2_envelopes = AsyncMock(return_value=type("R", (), {
        "accepted": [type("A", (), {
            "client_envelope_id": env.client_envelope_id,
            "object_id": "n1", "object_revision": 1, "server_cursor": 7, "apply_status": "applied",
        })()],
        "idempotent": [], "rejected": [], "conflicts": [], "apply_errors": [], "next_cursor": "7",
    })())
    flow = NotesM1SyncFlow(client=client, builder=builder, mirror=mirror, local_store=store, dataset_id="ds_1", device_id="dev_1")

    result = await flow.push([env])

    assert result["accepted"] == 1
    rec = mirror.get("ds_1", "n1")
    assert rec.object_revision == 1
    assert rec.server_cursor == 7
    assert rec.object_hash == env.payload_hash  # hash path proven, not empty


@pytest.mark.asyncio
async def test_pull_applies_and_advances_cursor():
    from tldw_chatbook.tldw_api import SyncV2Envelope
    mirror = NotesMirror(":memory:")
    store = InMemoryNotesStore()
    builder = SyncEnvelopeBuilder(dataset_id="ds_1", device_id="dev_2", dataset_key=b"x"*32, notes_mirror=mirror)
    client = AsyncMock()
    pulled_env = SyncV2Envelope(
        client_envelope_id="c", dataset_id="ds_1", domain="notes.note", object_id="n9",
        operation="upsert", adapter_version=1, payload={"title": "T", "content": "B"},
        payload_hash="sha256:cur", object_revision=1, server_cursor=12,
    )
    client.pull_sync_v2_envelopes = AsyncMock(return_value=type("P", (), {
        "envelopes": [pulled_env], "next_cursor": "12", "has_more": False,
    })())
    flow = NotesM1SyncFlow(client=client, builder=builder, mirror=mirror, local_store=store, dataset_id="ds_1", device_id="dev_2")

    result = await flow.pull(cursor=0)

    assert result["applied"] == 1
    assert store.get("n9")["title"] == "T"
    assert result["next_cursor"] == "12"
