from __future__ import annotations

import json

import pytest

from tldw_chatbook.Chat.provider_continuation import ContinuationValidationError
from tldw_chatbook.Chat.assistant_generation_state import AssistantGenerationState
from tldw_chatbook.Sync_Interop.crypto import decrypt_sync_payload, generate_dataset_key
from tldw_chatbook.Sync_Interop.envelope_builder import SyncEnvelopeBuilder
from tldw_chatbook.Sync_Interop.hashing import canonical_payload_hash


def _provider_continuation_json(*, canary: str = "private reasoning") -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "checkpoint_revision": 1,
            "provider": "moonshot",
            "protocol": "chat_completions",
            "model": "kimi-k2",
            "api_base_url": "https://api.moonshot.ai/v1",
            "state": "active",
            "rounds": [
                {
                    "assistant_content": "",
                    "reasoning_blocks": [canary],
                    "calls": [
                        {
                            "call_id": "call-1",
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                            "state": "pending",
                        }
                    ],
                }
            ],
        },
        indent=2,
    )


def test_note_body_goes_into_encrypted_payload_without_plaintext_leak() -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key
    )

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
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key
    )

    envelope = builder.build_chat_message(
        conversation_id="conversation-1",
        message_id="message-1",
        role="assistant",
        content="private answer",
    )

    assert envelope.domain == "chat"
    assert envelope.entity_id == "message-1"
    assert envelope.stable_key == "conversation-1:message-1"
    assert envelope.routing_metadata == {
        "conversation_id": "conversation-1",
        "entity_kind": "message",
    }
    assert decrypt_sync_payload_json(envelope.payload_ciphertext, dataset_key) == {
        "assistant_generation_state": None,
        "content": "private answer",
        "role": "assistant",
    }


@pytest.mark.parametrize("state", list(AssistantGenerationState))
def test_chat_message_carries_each_closed_assistant_generation_state(
    state: AssistantGenerationState,
) -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key
    )

    envelope = builder.build_chat_message(
        conversation_id="conversation-1",
        message_id="message-1",
        role="assistant",
        content="",
        provider_continuation_json=(
            _provider_continuation_json()
            if state is AssistantGenerationState.CONTINUATION_ACTIVE
            else None
        ),
        assistant_generation_state=state.value,
    )

    expected = {
        "assistant_generation_state": state.value,
        "content": "",
        "role": "assistant",
    }
    if state is AssistantGenerationState.CONTINUATION_ACTIVE:
        expected["provider_continuation_json"] = json.dumps(
            json.loads(_provider_continuation_json()), separators=(",", ":")
        )
    assert decrypt_sync_payload_json(envelope.payload_ciphertext, dataset_key) == (
        expected
    )


@pytest.mark.parametrize(
    ("role", "state"),
    [("assistant", "unknown"), ("user", "accepted")],
)
def test_chat_message_rejects_malformed_or_wrong_role_generation_state(
    role: str, state: str
) -> None:
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=generate_dataset_key(),
    )

    with pytest.raises(ValueError, match="assistant generation state"):
        builder.build_chat_message(
            conversation_id="conversation-1",
            message_id="message-1",
            role=role,
            content="visible",
            assistant_generation_state=state,
        )


def test_chat_message_rejects_malformed_state_even_with_active_continuation() -> None:
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=generate_dataset_key(),
    )

    with pytest.raises(ValueError, match="assistant generation state"):
        builder.build_chat_message(
            conversation_id="conversation-1",
            message_id="message-1",
            role="assistant",
            content="visible",
            provider_continuation_json=_provider_continuation_json(),
            assistant_generation_state="unknown",
        )


def test_chat_message_rejects_continuation_active_without_active_continuation() -> None:
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=generate_dataset_key(),
    )

    with pytest.raises(ValueError, match="assistant generation state"):
        builder.build_chat_message(
            conversation_id="conversation-1",
            message_id="message-1",
            role="assistant",
            content="",
            assistant_generation_state="continuation_active",
        )


def test_chat_message_preserves_restore_metadata_without_plaintext_leak() -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key
    )

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
        "assistant_generation_state": None,
        "content": "private regenerated answer",
        "role": "assistant",
    }


def test_chat_message_delete_is_a_clear_tombstone_with_exact_version() -> None:
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=generate_dataset_key(),
    )

    envelope = builder.build_chat_message_delete(
        conversation_id="conversation-1",
        message_id="message-2",
        entity_version=7,
    )

    assert envelope.domain == "chat"
    assert envelope.operation == "delete"
    assert envelope.stable_key == "conversation-1:message-2"
    assert envelope.entity_version == 7
    assert envelope.payload_clear == {"deleted": True}
    assert envelope.routing_metadata == {
        "conversation_id": "conversation-1",
        "entity_kind": "message",
    }
    assert envelope.payload_hash == canonical_payload_hash({"deleted": True})


def test_chat_message_canonicalizes_continuation_inside_encrypted_payload() -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key
    )
    canary = "SYNC-CONTINUATION-PRIVATE-CANARY"

    envelope = builder.build_chat_message(
        conversation_id="conversation-1",
        message_id="variant-message-1",
        role="assistant",
        content="visible answer",
        variant_turn_id="turn-1",
        variant_index=1,
        provider_continuation_json=_provider_continuation_json(canary=canary),
    )

    serialized = envelope.model_dump_json()
    payload = decrypt_sync_payload_json(envelope.payload_ciphertext, dataset_key)
    assert canary not in serialized
    assert "provider_continuation_json" not in envelope.routing_metadata
    assert "provider_continuation_json" not in envelope.payload_clear
    assert payload["content"] == "visible answer"
    assert payload["role"] == "assistant"
    assert json.loads(payload["provider_continuation_json"])["rounds"][0][
        "reasoning_blocks"
    ] == [canary]
    assert "\n" not in payload["provider_continuation_json"]


@pytest.mark.parametrize("private_value", [None, ""])
def test_chat_message_omits_empty_continuation_for_legacy_compatibility(
    private_value: str | None,
) -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key
    )

    envelope = builder.build_chat_message(
        conversation_id="conversation-1",
        message_id="message-1",
        role="assistant",
        content="visible answer",
        provider_continuation_json=private_value,
    )

    assert decrypt_sync_payload_json(envelope.payload_ciphertext, dataset_key) == {
        "assistant_generation_state": None,
        "content": "visible answer",
        "role": "assistant",
    }


@pytest.mark.parametrize(
    "private_value",
    [
        pytest.param(False, id="bool-false"),
        pytest.param(0, id="integer-zero"),
        pytest.param({}, id="empty-object"),
        pytest.param([], id="empty-list"),
    ],
)
def test_chat_message_rejects_present_falsey_invalid_continuation(
    private_value: object,
) -> None:
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=generate_dataset_key(),
    )

    with pytest.raises(ContinuationValidationError) as caught:
        builder.build_chat_message(
            conversation_id="conversation-1",
            message_id="message-1",
            role="assistant",
            content="visible answer",
            provider_continuation_json=private_value,  # type: ignore[arg-type]
        )

    assert str(caught.value) == "Invalid continuation data."


def test_chat_message_rejects_invalid_continuation_without_private_error_text() -> None:
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_key=generate_dataset_key(),
    )
    canary = "INVALID-PRIVATE-CANARY"

    with pytest.raises(ContinuationValidationError) as caught:
        builder.build_chat_message(
            conversation_id="conversation-1",
            message_id="message-1",
            role="assistant",
            content="visible answer",
            provider_continuation_json=canary,
        )

    assert canary not in str(caught.value)
    assert canary not in repr(caught.value)


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
    assert linked.payload_clear == {
        "workspace_id": "workspace-1",
        "source_id": "source-1",
    }
    assert unlinked.operation == "unlink"


def test_source_cache_uses_source_id_and_content_hash_identity() -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key
    )

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


def decrypt_sync_payload_json(
    payload_ciphertext: str | None, dataset_key: bytes
) -> dict:
    assert payload_ciphertext is not None
    return decrypt_sync_payload_json_record(payload_ciphertext, dataset_key)


def decrypt_sync_payload_json_record(
    payload_ciphertext: str, dataset_key: bytes
) -> dict:
    return decrypt_sync_payload(
        json.loads(payload_ciphertext),
        key=dataset_key,
    )
