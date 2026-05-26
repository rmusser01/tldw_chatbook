from __future__ import annotations

import json

from tldw_chatbook.Sync_Interop.crypto import decrypt_sync_payload, generate_dataset_key
from tldw_chatbook.Sync_Interop.envelope_builder import SyncEnvelopeBuilder


def test_note_body_goes_into_encrypted_payload_without_plaintext_leak() -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key)

    envelope = builder.build_note_upsert(
        note_id="note-1",
        title="Research note",
        body="known private note body",
        status="active",
        tag_ids=["tag-1"],
    )

    serialized = envelope.model_dump_json()
    assert envelope.domain == "notes"
    assert envelope.operation == "upsert"
    assert envelope.payload_ciphertext is not None
    assert envelope.payload_clear == {"status": "active", "tag_ids": ["tag-1"]}
    assert "known private note body" not in serialized
    assert "Research note" not in serialized
    assert decrypt_sync_payload_json(envelope.payload_ciphertext, dataset_key) == {
        "body": "known private note body",
        "title": "Research note",
    }


def test_chat_message_uses_stable_message_identity() -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key)

    envelope = builder.build_chat_message(
        conversation_id="conversation-1",
        message_id="message-1",
        role="assistant",
        content="private answer",
    )

    assert envelope.domain == "chat"
    assert envelope.entity_id == "message-1"
    assert envelope.stable_key == "conversation-1:message-1"
    assert envelope.routing_metadata == {"conversation_id": "conversation-1", "entity_kind": "message"}
    assert decrypt_sync_payload_json(envelope.payload_ciphertext, dataset_key) == {
        "content": "private answer",
        "role": "assistant",
    }


def test_chat_message_preserves_restore_metadata_without_plaintext_leak() -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key)

    envelope = builder.build_chat_message(
        conversation_id="conversation-1",
        message_id="message-2",
        role="assistant",
        content="private regenerated answer",
        parent_message_id="message-1",
        sequence=2,
        variant_turn_id="turn-1",
        variant_index=1,
        variant_count=2,
        selected_variant_id="variant-2",
        base_version="v1",
        entity_version="v2",
    )

    serialized = envelope.model_dump_json()
    assert "private regenerated answer" not in serialized
    assert envelope.base_version == "v1"
    assert envelope.entity_version == "v2"
    assert envelope.routing_metadata == {
        "conversation_id": "conversation-1",
        "entity_kind": "message",
        "parent_message_id": "message-1",
        "selected_variant_id": "variant-2",
        "sequence": 2,
        "variant_count": 2,
        "variant_index": 1,
        "variant_turn_id": "turn-1",
    }
    assert decrypt_sync_payload_json(envelope.payload_ciphertext, dataset_key) == {
        "content": "private regenerated answer",
        "role": "assistant",
    }


def test_workspace_source_ref_add_remove_maps_to_link_unlink() -> None:
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=generate_dataset_key(),
    )

    linked = builder.build_workspace_source_ref(
        workspace_id="workspace-1",
        source_id="source-1",
        operation="link",
    )
    unlinked = builder.build_workspace_source_ref(
        workspace_id="workspace-1",
        source_id="source-1",
        operation="unlink",
    )

    assert linked.domain == "workspaces"
    assert linked.operation == "link"
    assert linked.entity_id == "workspace-1:source-1"
    assert linked.payload_clear == {"workspace_id": "workspace-1", "source_id": "source-1"}
    assert unlinked.operation == "unlink"


def test_source_cache_uses_source_id_and_content_hash_identity() -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key)

    envelope = builder.build_source_cache(
        source_id="source-1",
        content_hash="sha256:content",
        cache_kind="transcript",
        content="private transcript",
    )

    assert envelope.domain == "source_cache"
    assert envelope.entity_id == "source-1:sha256:content"
    assert envelope.stable_key == "source-1:sha256:content"
    assert envelope.payload_clear == {
        "source_id": "source-1",
        "payload_hash": "sha256:content",
        "record_type": "transcript",
    }
    assert decrypt_sync_payload_json(envelope.payload_ciphertext, dataset_key) == {
        "content": "private transcript",
    }


def decrypt_sync_payload_json(payload_ciphertext: str | None, dataset_key: bytes) -> dict:
    assert payload_ciphertext is not None
    return decrypt_sync_payload_json_record(payload_ciphertext, dataset_key)


def decrypt_sync_payload_json_record(payload_ciphertext: str, dataset_key: bytes) -> dict:
    return decrypt_sync_payload(
        json.loads(payload_ciphertext),
        key=dataset_key,
    )
