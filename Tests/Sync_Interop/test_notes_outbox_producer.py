from __future__ import annotations

import json

import pytest

from tldw_chatbook.Sync_Interop.crypto import decrypt_sync_payload, generate_dataset_key
from tldw_chatbook.Sync_Interop.notes_outbox_producer import NotesSyncV2OutboxProducer
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository


class _AllowAllNotesAuthority:
    @staticmethod
    def is_note_dispatchable(note_id: str) -> bool:
        return bool(note_id)


def test_notes_producer_enqueues_encrypted_note_upsert_and_updates_summary(
    tmp_path,
) -> None:
    dataset_key = generate_dataset_key()
    repo = _local_first_repo(tmp_path, dataset_key=dataset_key)
    producer = NotesSyncV2OutboxProducer(
        state_repository=repo,
        dataset_keys={"dataset-1": dataset_key},
        notes_db=_AllowAllNotesAuthority(),
    )

    result = producer.enqueue_note_upsert(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        note_id="note-1",
        title="Private title",
        content="Private body",
        status="active",
        entity_version=1,
    )

    entries = repo.list_pending_sync_v2_outbox_envelopes(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        dataset_id="dataset-1",
    )
    summary = repo.get_sync_v2_profile_summary(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
    )
    envelope = entries[0]["envelope"]

    assert result["status"] == "enqueued"
    assert len(entries) == 1
    assert entries[0]["domain"] == "notes"
    assert summary["outbox"]["pending"] == 1
    assert summary["outbox"]["by_domain"]["notes"]["pending"] == 1
    serialized = json.dumps(entries[0]["envelope"])
    assert "Private title" not in serialized
    assert "Private body" not in serialized
    assert envelope["payload_clear"] == {"status": "active"}
    assert envelope["routing_metadata"] == {"entity_kind": "note"}
    decrypted_payload = _decrypt_payload(envelope["payload_ciphertext"], dataset_key)
    assert decrypted_payload == {
        "body": "Private body",
        "title": "Private title",
    }
    assert not _contains_forbidden_folder_key(envelope)
    assert not _contains_forbidden_folder_key(decrypted_payload)

    producer.enqueue_note_upsert(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        note_id="note-1",
        title="Private title",
        content="Private body",
        status="active",
        entity_version=1,
    )
    assert (
        len(
            repo.list_pending_sync_v2_outbox_envelopes(
                server_profile_id="server-a",
                authenticated_principal_id="user-a",
                workspace_scope=None,
                dataset_id="dataset-1",
            )
        )
        == 1
    )


def test_notes_producer_enqueues_delete_without_plaintext_payload(tmp_path) -> None:
    dataset_key = generate_dataset_key()
    repo = _local_first_repo(tmp_path, dataset_key=dataset_key)
    producer = NotesSyncV2OutboxProducer(
        state_repository=repo,
        dataset_keys={"dataset-1": dataset_key},
        notes_db=_AllowAllNotesAuthority(),
    )

    result = producer.enqueue_note_delete(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        note_id="note-1",
        base_version=3,
        entity_version=4,
    )

    entries = repo.list_pending_sync_v2_outbox_envelopes(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        dataset_id="dataset-1",
    )
    envelope = entries[0]["envelope"]

    assert result["status"] == "enqueued"
    assert envelope["operation"] == "delete"
    assert envelope["payload_ciphertext"] is None
    assert envelope["payload_clear"] == {"deleted": True}
    assert envelope["base_version"] == 3
    assert envelope["entity_version"] == 4


def test_notes_producer_skips_without_local_first_profile_or_dataset_key(
    tmp_path,
) -> None:
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    repo.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        profile_mode="server_frontend",
        device_id="device-1",
        dataset_id="dataset-1",
    )
    producer = NotesSyncV2OutboxProducer(
        state_repository=repo,
        dataset_keys={},
        notes_db=_AllowAllNotesAuthority(),
    )

    result = producer.enqueue_note_upsert(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        note_id="note-1",
        title="Local only",
        content="No sync",
    )

    assert result == {"status": "skipped", "reason": "profile_not_local_first"}
    assert (
        repo.list_sync_v2_outbox_entries(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope=None,
            dataset_id="dataset-1",
        )
        == []
    )


def test_notes_producer_fails_closed_without_notes_dispatch_authority(tmp_path) -> None:
    dataset_key = generate_dataset_key()
    repo = _local_first_repo(tmp_path, dataset_key=dataset_key)
    producer = NotesSyncV2OutboxProducer(
        state_repository=repo,
        dataset_keys={"dataset-1": dataset_key},
    )

    result = producer.enqueue_note_upsert(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        note_id="note-unauthorized",
        title="Must not leave",
        content="No Notes owner was injected",
        entity_version=1,
    )

    assert result == {"status": "skipped", "reason": "notes_authority_unavailable"}
    assert repo.list_sync_v2_outbox_entries(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        dataset_id="dataset-1",
    ) == []


def test_notes_producer_observes_keys_inserted_into_shared_cache_after_wiring(
    tmp_path,
) -> None:
    dataset_key = generate_dataset_key()
    repo = _local_first_repo(tmp_path, dataset_key=dataset_key)
    shared_keys: dict[str, bytes] = {}
    producer = NotesSyncV2OutboxProducer(
        state_repository=repo,
        dataset_keys=shared_keys,
        notes_db=_AllowAllNotesAuthority(),
    )

    assert producer.dataset_keys is shared_keys
    skipped = producer.enqueue_note_upsert(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        note_id="note-late-key",
        title="Late key",
        content="Unavailable at startup",
    )
    assert skipped == {"status": "skipped", "reason": "dataset_key_unavailable"}

    shared_keys["dataset-1"] = dataset_key
    enqueued = producer.enqueue_note_upsert(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        note_id="note-late-key",
        title="Late key",
        content="Available after recovery",
    )

    assert enqueued["status"] == "enqueued"
    assert len(
        repo.list_pending_sync_v2_outbox_envelopes(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope=None,
            dataset_id="dataset-1",
        )
    ) == 1


def test_notes_producer_without_key_cache_owns_a_private_mapping(tmp_path) -> None:
    repo = _local_first_repo(tmp_path, dataset_key=generate_dataset_key())

    first = NotesSyncV2OutboxProducer(state_repository=repo)
    second = NotesSyncV2OutboxProducer(state_repository=repo)

    assert first.dataset_keys == {}
    assert second.dataset_keys == {}
    assert first.dataset_keys is not second.dataset_keys


def _local_first_repo(tmp_path, *, dataset_key: bytes) -> SyncStateRepository:
    del dataset_key
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    repo.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        profile_mode="local_first",
        device_id="device-1",
        dataset_id="dataset-1",
    )
    return repo


def _decrypt_payload(payload_ciphertext: str, dataset_key: bytes) -> dict:
    return decrypt_sync_payload(json.loads(payload_ciphertext), key=dataset_key)


def _recursive_dict_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for nested in value.values():
            keys.update(_recursive_dict_keys(nested))
        return keys
    if isinstance(value, (list, tuple)):
        keys: set[str] = set()
        for nested in value:
            keys.update(_recursive_dict_keys(nested))
        return keys
    return set()


def _contains_forbidden_folder_key(value: object) -> bool:
    forbidden_tokens = ("folder", "membership", "owner_id", "binding")
    return any(
        token in key.casefold()
        for key in _recursive_dict_keys(value)
        for token in forbidden_tokens
    )


@pytest.mark.parametrize(
    "key",
    ["folder_id", "memberships", "managed_owner_id", "ROOT_BINDING"],
)
def test_folder_key_guard_rejects_derivative_and_case_variant_keys(key: str) -> None:
    assert _contains_forbidden_folder_key({"nested": [{key: "value"}]})


def test_folder_key_guard_does_not_match_payload_values() -> None:
    assert not _contains_forbidden_folder_key(
        {"title": "folder_id membership owner_id root_binding"}
    )
