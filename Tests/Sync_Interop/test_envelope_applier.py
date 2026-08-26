from __future__ import annotations

import json

import pytest

from tldw_chatbook.Sync_Interop.crypto import (
    encrypt_sync_payload,
    generate_dataset_key,
)
from tldw_chatbook.Sync_Interop.envelope_applier import SyncEnvelopeApplier
from tldw_chatbook.Sync_Interop.envelope_builder import SyncEnvelopeBuilder
from tldw_chatbook.Sync_Interop.hashing import canonical_payload_hash


class RecordingLocalStore:
    def __init__(self) -> None:
        self.note_hashes: dict[str, str] = {}
        self.note_content: dict[str, dict] = {}
        self.note_metadata: dict[str, dict] = {}
        self.chat_hashes: dict[str, str] = {}
        self.chat_messages: dict[str, dict] = {}
        self.workspace_links: set[tuple[str, str]] = set()
        self.source_cache: dict[str, dict] = {}
        self.conflicts: list[dict] = []

    def get_note_content_hash(self, note_id: str) -> str | None:
        return self.note_hashes.get(note_id)

    def upsert_note_content(
        self, note_id: str, payload: dict, payload_hash: str
    ) -> None:
        self.note_content[note_id] = payload
        self.note_hashes[note_id] = payload_hash

    def upsert_note_metadata(self, note_id: str, metadata: dict) -> None:
        self.note_metadata[note_id] = metadata

    def get_chat_message_hash(self, stable_key: str) -> str | None:
        return self.chat_hashes.get(stable_key)

    def append_chat_message(
        self, stable_key: str, payload: dict, payload_hash: str
    ) -> None:
        self.chat_messages[stable_key] = payload
        self.chat_hashes[stable_key] = payload_hash

    def link_workspace_source(self, workspace_id: str, source_id: str) -> None:
        self.workspace_links.add((workspace_id, source_id))

    def unlink_workspace_source(self, workspace_id: str, source_id: str) -> None:
        self.workspace_links.discard((workspace_id, source_id))

    def upsert_source_cache(
        self, stable_key: str, payload: dict, metadata: dict
    ) -> None:
        self.source_cache[stable_key] = {"payload": payload, "metadata": metadata}

    def record_conflict(self, conflict: dict) -> None:
        self.conflicts.append(conflict)


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
        }
    )


def test_note_applier_records_conflict_instead_of_overwriting_divergent_content() -> (
    None
):
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key
    )
    store = RecordingLocalStore()
    store.note_hashes["note-1"] = "sha256:local-dirty"
    applier = SyncEnvelopeApplier(dataset_key=dataset_key, local_store=store)

    envelope = builder.build_note_upsert(
        note_id="note-1",
        title="Remote",
        body="remote body",
        base_version="sha256:remote-base",
    )
    result = applier.apply(envelope)

    assert result["status"] == "conflict"
    assert store.note_content == {}
    assert store.conflicts[0]["domain"] == "notes"
    assert store.conflicts[0]["conflict_type"] == "encrypted_content_edit"


def test_note_applier_merges_safe_metadata_without_content_overwrite() -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key
    )
    store = RecordingLocalStore()
    applier = SyncEnvelopeApplier(dataset_key=dataset_key, local_store=store)

    envelope = builder.build_note_metadata_update(
        note_id="note-1",
        status="archived",
        tag_ids=["tag-1"],
    )
    result = applier.apply(envelope)

    assert result["status"] == "applied"
    assert store.note_metadata["note-1"] == {"status": "archived", "tag_ids": ["tag-1"]}
    assert store.note_content == {}


def test_legacy_encrypted_apply_conflicts_when_dataset_key_missing() -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key
    )
    store = RecordingLocalStore()
    applier = SyncEnvelopeApplier(local_store=store)

    envelope = builder.build_note_upsert(
        note_id="note-1",
        title="Remote",
        body="remote body",
    )
    result = applier.apply(envelope)

    assert result["status"] == "conflict"
    assert result["conflict"]["conflict_type"] == "missing_dataset_key"
    assert store.note_content == {}


def test_chat_applier_appends_by_stable_id_and_conflicts_on_hash_mismatch() -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key
    )
    store = RecordingLocalStore()
    applier = SyncEnvelopeApplier(dataset_key=dataset_key, local_store=store)

    envelope = builder.build_chat_message(
        conversation_id="conversation-1",
        message_id="message-1",
        role="user",
        content="hello",
    )
    first = applier.apply(envelope)
    second = applier.apply(envelope)
    changed = envelope.model_copy(update={"payload_hash": "sha256:other"})
    conflict = applier.apply(changed)

    assert first["status"] == "applied"
    assert second["status"] == "noop"
    assert store.chat_messages["conversation-1:message-1"] == {
        "assistant_generation_state": None,
        "content": "hello",
        "role": "user",
    }
    assert conflict["status"] == "conflict"
    assert store.conflicts[-1]["domain"] == "chat"


def test_chat_applier_allows_versioned_message_update() -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key
    )
    store = RecordingLocalStore()
    applier = SyncEnvelopeApplier(dataset_key=dataset_key, local_store=store)

    original = builder.build_chat_message(
        conversation_id="conversation-1",
        message_id="message-1",
        role="assistant",
        content="first",
    )
    updated = builder.build_chat_message(
        conversation_id="conversation-1",
        message_id="message-1",
        role="assistant",
        content="second",
        base_version=original.payload_hash,
    )

    first = applier.apply(original)
    second = applier.apply(updated)

    assert first["status"] == "applied"
    assert second["status"] == "applied"
    assert store.chat_messages["conversation-1:message-1"] == {
        "assistant_generation_state": None,
        "content": "second",
        "role": "assistant",
    }
    assert store.conflicts == []


def test_chat_applier_never_matches_an_unreadable_local_hash_sentinel() -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key
    )
    store = RecordingLocalStore()
    stable_key = "conversation-1:message-1"
    store.chat_hashes[stable_key] = "invalid-local-chat-message"
    store.chat_messages[stable_key] = {
        "assistant_generation_state": "complete",
        "content": "unreadable local owner",
        "role": "assistant",
    }
    applier = SyncEnvelopeApplier(dataset_key=dataset_key, local_store=store)
    adversarial = builder.build_chat_message(
        conversation_id="conversation-1",
        message_id="message-1",
        role="assistant",
        content="remote overwrite",
        base_version="invalid-local-chat-message",
    )

    result = applier.apply(adversarial)

    assert result["status"] == "conflict"
    assert result["conflict"]["conflict_type"] == "chat_message_hash_mismatch"
    assert store.chat_messages[stable_key]["content"] == "unreadable local owner"
    assert store.chat_hashes[stable_key] == "invalid-local-chat-message"


def test_legacy_missing_state_apply_uses_upgraded_canonical_hash_for_update() -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key
    )
    store = RecordingLocalStore()
    applier = SyncEnvelopeApplier(dataset_key=dataset_key, local_store=store)
    legacy_payload = {"content": "legacy", "role": "assistant"}
    normalized_payload = {
        "assistant_generation_state": None,
        "content": "legacy",
        "role": "assistant",
    }
    legacy = builder.build_chat_message(
        conversation_id="conversation-1",
        message_id="message-1",
        role="assistant",
        content="legacy",
    ).model_copy(
        update={
            "payload_ciphertext": encrypt_sync_payload(
                legacy_payload, key=dataset_key
            ).model_dump_json(),
            "payload_hash": canonical_payload_hash(legacy_payload),
        }
    )
    normalized_hash = canonical_payload_hash(normalized_payload)
    upgraded = builder.build_chat_message(
        conversation_id="conversation-1",
        message_id="message-1",
        role="assistant",
        content="upgraded",
        base_version=normalized_hash,
    )

    assert applier.apply(legacy) == {"status": "applied"}
    assert store.chat_hashes["conversation-1:message-1"] == normalized_hash
    assert applier.apply(legacy) == {"status": "noop"}
    assert applier.apply(upgraded) == {"status": "applied"}
    assert store.chat_messages["conversation-1:message-1"]["content"] == "upgraded"


def test_chat_applier_attaches_valid_continuation_to_exact_stable_message() -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key
    )
    store = RecordingLocalStore()
    applier = SyncEnvelopeApplier(dataset_key=dataset_key, local_store=store)
    canary = "APPLIER-PRIVATE-CANARY"

    envelope = builder.build_chat_message(
        conversation_id="conversation-1",
        message_id="variant-message-2",
        role="assistant",
        content="winning visible answer",
        variant_turn_id="turn-1",
        variant_index=2,
        provider_continuation_json=_provider_continuation_json(canary=canary),
    )

    result = applier.apply(envelope)

    assert result == {"status": "applied"}
    payload = store.chat_messages["conversation-1:variant-message-2"]
    assert payload["content"] == "winning visible answer"
    assert json.loads(payload["provider_continuation_json"])["rounds"][0][
        "reasoning_blocks"
    ] == [canary]
    assert "conversation-1:variant-message-1" not in store.chat_messages


def test_chat_applier_drops_invalid_private_data_but_applies_visible_message() -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key
    )
    store = RecordingLocalStore()
    applier = SyncEnvelopeApplier(dataset_key=dataset_key, local_store=store)
    canary = "INVALID-REMOTE-PRIVATE-CANARY"
    envelope = builder.build_chat_message(
        conversation_id="conversation-1",
        message_id="message-1",
        role="assistant",
        content="visible survives",
    )
    invalid_payload = {
        "content": "visible survives",
        "role": "assistant",
        "provider_continuation_json": canary,
    }
    envelope = envelope.model_copy(
        update={
            "payload_ciphertext": encrypt_sync_payload(
                invalid_payload, key=dataset_key
            ).model_dump_json(),
            "payload_hash": builder._payload_hash(invalid_payload),
        }
    )

    result = applier.apply(envelope)

    assert result["status"] == "applied"
    assert result["warning"] == "Exact tool continuation was discarded."
    assert store.chat_messages["conversation-1:message-1"] == {
        "assistant_generation_state": None,
        "content": "visible survives",
        "role": "assistant",
    }
    assert canary not in json.dumps(result)


@pytest.mark.parametrize(
    "private_value",
    [
        pytest.param("", id="empty-string"),
        pytest.param(False, id="bool-false"),
        pytest.param(0, id="integer-zero"),
        pytest.param({}, id="empty-object"),
        pytest.param([], id="empty-list"),
    ],
)
def test_chat_applier_warns_for_present_falsey_invalid_private_data(
    private_value: object,
) -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key
    )
    store = RecordingLocalStore()
    applier = SyncEnvelopeApplier(dataset_key=dataset_key, local_store=store)
    envelope = builder.build_chat_message(
        conversation_id="conversation-1",
        message_id="message-1",
        role="assistant",
        content="visible survives",
    )
    invalid_payload = {
        "content": "visible survives",
        "role": "assistant",
        "provider_continuation_json": private_value,
    }
    envelope = envelope.model_copy(
        update={
            "payload_ciphertext": encrypt_sync_payload(
                invalid_payload, key=dataset_key
            ).model_dump_json(),
            "payload_hash": builder._payload_hash(invalid_payload),
        }
    )

    result = applier.apply(envelope)

    assert result == {
        "status": "applied",
        "warning": "Exact tool continuation was discarded.",
    }
    assert store.chat_messages["conversation-1:message-1"] == {
        "assistant_generation_state": None,
        "content": "visible survives",
        "role": "assistant",
    }


@pytest.mark.parametrize("include_private_key", [False, True], ids=["absent", "none"])
def test_chat_applier_accepts_legacy_missing_private_data_without_warning(
    include_private_key: bool,
) -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key
    )
    store = RecordingLocalStore()
    applier = SyncEnvelopeApplier(dataset_key=dataset_key, local_store=store)
    envelope = builder.build_chat_message(
        conversation_id="conversation-1",
        message_id="message-1",
        role="assistant",
        content="legacy visible",
    )
    payload = {"content": "legacy visible", "role": "assistant"}
    if include_private_key:
        payload["provider_continuation_json"] = None
        envelope = envelope.model_copy(
            update={
                "payload_ciphertext": encrypt_sync_payload(
                    payload, key=dataset_key
                ).model_dump_json(),
                "payload_hash": builder._payload_hash(payload),
            }
        )

    assert applier.apply(envelope) == {"status": "applied"}
    assert store.chat_messages["conversation-1:message-1"] == {
        "assistant_generation_state": None,
        "content": "legacy visible",
        "role": "assistant",
    }


@pytest.mark.parametrize(
    ("role", "state"),
    [
        ("assistant", "unknown"),
        ("user", "accepted"),
    ],
)
def test_chat_applier_rejects_invalid_generation_state_payloads(
    role: str, state: str
) -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key
    )
    store = RecordingLocalStore()
    applier = SyncEnvelopeApplier(dataset_key=dataset_key, local_store=store)
    envelope = builder.build_chat_message(
        conversation_id="conversation-1",
        message_id="message-1",
        role="assistant",
        content="visible",
    )
    payload = {
        "assistant_generation_state": state,
        "content": "visible",
        "role": role,
    }
    envelope = envelope.model_copy(
        update={
            "payload_ciphertext": encrypt_sync_payload(
                payload, key=dataset_key
            ).model_dump_json(),
            "payload_hash": builder._payload_hash(payload),
        }
    )

    result = applier.apply(envelope)

    assert result["status"] == "conflict"
    assert result["conflict"]["conflict_type"] == "invalid_chat_message_payload"
    assert store.chat_messages == {}


def test_chat_applier_rejects_unknown_payload_keys_after_legacy_normalization() -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key
    )
    store = RecordingLocalStore()
    applier = SyncEnvelopeApplier(dataset_key=dataset_key, local_store=store)
    envelope = builder.build_chat_message(
        conversation_id="conversation-1",
        message_id="message-1",
        role="assistant",
        content="visible",
    )
    payload = {"content": "visible", "role": "assistant", "unexpected": True}
    envelope = envelope.model_copy(
        update={
            "payload_ciphertext": encrypt_sync_payload(
                payload, key=dataset_key
            ).model_dump_json(),
            "payload_hash": builder._payload_hash(payload),
        }
    )

    result = applier.apply(envelope)

    assert result["status"] == "conflict"
    assert result["conflict"]["conflict_type"] == "invalid_chat_message_payload"
    assert store.chat_messages == {}


def test_chat_conflict_never_field_merges_content_and_continuation() -> None:
    dataset_key = generate_dataset_key()
    first_builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key
    )
    second_builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-2", dataset_key=dataset_key
    )
    store = RecordingLocalStore()
    applier = SyncEnvelopeApplier(dataset_key=dataset_key, local_store=store)
    local = first_builder.build_chat_message(
        conversation_id="conversation-1",
        message_id="message-1",
        role="assistant",
        content="local winner",
        provider_continuation_json=_provider_continuation_json(canary="LOCAL-OWNER"),
    )
    remote = second_builder.build_chat_message(
        conversation_id="conversation-1",
        message_id="message-1",
        role="assistant",
        content="remote loser",
        provider_continuation_json=_provider_continuation_json(canary="REMOTE-OWNER"),
        base_version="sha256:stale-base",
    )

    assert applier.apply(local)["status"] == "applied"
    assert applier.apply(remote)["status"] == "conflict"

    winner = store.chat_messages["conversation-1:message-1"]
    assert winner["content"] == "local winner"
    assert json.loads(winner["provider_continuation_json"])["rounds"][0][
        "reasoning_blocks"
    ] == ["LOCAL-OWNER"]
    assert "REMOTE-OWNER" not in json.dumps(winner)


def test_workspace_and_source_cache_appliers_route_to_local_store() -> None:
    dataset_key = generate_dataset_key()
    builder = SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-1", dataset_key=dataset_key
    )
    store = RecordingLocalStore()
    applier = SyncEnvelopeApplier(dataset_key=dataset_key, local_store=store)

    link = builder.build_workspace_source_ref(
        workspace_id="workspace-1",
        source_id="source-1",
        operation="link",
    )
    cache = builder.build_source_cache(
        source_id="source-1",
        content_hash="sha256:content",
        cache_kind="transcript",
        content="private transcript",
    )

    assert applier.apply(link)["status"] == "applied"
    assert ("workspace-1", "source-1") in store.workspace_links
    assert applier.apply(cache)["status"] == "applied"
    assert store.source_cache["source-1:sha256:content"]["payload"] == {
        "content": "private transcript"
    }
