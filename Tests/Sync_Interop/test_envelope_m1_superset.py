"""P2: SyncV2Envelope supports the M1 superset additively (legacy still works)."""

from tldw_chatbook.tldw_api import SyncV2Envelope


def test_m1_notes_envelope_round_trips_canonical_fields():
    env = SyncV2Envelope(
        client_envelope_id="dev1:notes.note:note_1:h",
        dataset_id="ds_1",
        device_id="dev1",
        domain="notes.note",
        object_id="note_1",
        operation="tombstone",
        adapter_version=1,
        schema_version=1,
        object_revision=2,
        base_server_cursor=10,
        base_object_revision=1,
        base_object_hash="sha256:prev",
        deleted=True,
        payload={"deleted_at": "2026-06-15T00:00:00Z"},
        payload_hash="sha256:cur",
        encryption_metadata={"policy": "server_trusted_v1"},
    )
    dumped = env.model_dump(mode="json")
    assert dumped["object_id"] == "note_1"
    assert dumped["domain"] == "notes.note"
    assert dumped["operation"] == "tombstone"
    assert dumped["object_revision"] == 2
    assert dumped["base_object_hash"] == "sha256:prev"
    assert dumped["payload"] == {"deleted_at": "2026-06-15T00:00:00Z"}
    assert dumped["encryption_metadata"]["policy"] == "server_trusted_v1"


def test_object_id_falls_back_to_entity_id_and_vice_versa():
    legacy = SyncV2Envelope(
        client_envelope_id="c", dataset_id="ds", domain="notes",
        entity_id="e_1", operation="upsert", adapter_version=1, payload_hash="sha256:x",
    )
    assert legacy.object_id == "e_1"
    assert legacy.entity_id == "e_1"
    m1 = SyncV2Envelope(
        client_envelope_id="c", dataset_id="ds", domain="notes.note",
        object_id="o_1", operation="upsert", adapter_version=1, payload_hash="sha256:x",
    )
    assert m1.entity_id == "o_1"
    assert m1.object_id == "o_1"


def test_payload_mirrors_payload_clear():
    env = SyncV2Envelope(
        client_envelope_id="c", dataset_id="ds", domain="notes.note",
        object_id="o", operation="upsert", adapter_version=1, payload_hash="sha256:x",
        payload={"title": "T", "content": "B"},
    )
    assert env.payload_clear == {"title": "T", "content": "B"}


def test_legacy_client_private_envelope_still_valid():
    env = SyncV2Envelope(
        client_envelope_id="c", dataset_id="ds", domain="chat",
        entity_id="m1", operation="upsert", adapter_version=1,
        payload_clear={}, payload_hash="sha256:x",
        encryption_policy="client_private_v1", payload_ciphertext="abc",
    )
    assert env.encryption_policy == "client_private_v1"
    assert env.operation == "upsert"
