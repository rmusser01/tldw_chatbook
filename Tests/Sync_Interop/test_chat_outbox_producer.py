from __future__ import annotations

import json

from tldw_chatbook.Sync_Interop.chat_outbox_producer import ChatSyncV2OutboxProducer
from tldw_chatbook.Sync_Interop.crypto import decrypt_sync_payload, generate_dataset_key
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository


def test_chat_producer_enqueues_encrypted_message_and_updates_summary(tmp_path) -> None:
    dataset_key = generate_dataset_key()
    repo = _local_first_repo(tmp_path)
    producer = ChatSyncV2OutboxProducer(
        state_repository=repo,
        dataset_keys={"dataset-1": dataset_key},
    )

    result = producer.enqueue_chat_message(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        conversation_id="conversation-1",
        message_id="message-2",
        role="assistant",
        content="Private answer",
        parent_message_id="message-1",
        sequence=2,
        variant_turn_id="turn-1",
        variant_index=1,
        variant_count=2,
        selected_variant_id="variant-2",
        entity_version=3,
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
    assert entries[0]["domain"] == "chat"
    assert summary["outbox"]["pending"] == 1
    assert summary["outbox"]["by_domain"]["chat"]["pending"] == 1
    serialized = json.dumps(envelope)
    assert "Private answer" not in serialized
    assert envelope["stable_key"] == "conversation-1:message-2"
    assert envelope["routing_metadata"] == {
        "conversation_id": "conversation-1",
        "entity_kind": "message",
        "parent_message_id": "message-1",
        "selected_variant_id": "variant-2",
        "sequence": 2,
        "variant_count": 2,
        "variant_index": 1,
        "variant_turn_id": "turn-1",
    }
    assert _decrypt_payload(envelope["payload_ciphertext"], dataset_key) == {
        "content": "Private answer",
        "role": "assistant",
    }

    producer.enqueue_chat_message(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        conversation_id="conversation-1",
        message_id="message-2",
        role="assistant",
        content="Private answer",
        parent_message_id="message-1",
        sequence=2,
        variant_turn_id="turn-1",
        variant_index=1,
        variant_count=2,
        selected_variant_id="variant-2",
        entity_version=3,
    )
    assert len(
        repo.list_pending_sync_v2_outbox_envelopes(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope=None,
            dataset_id="dataset-1",
        )
    ) == 1


def test_chat_producer_skips_without_local_first_profile_or_dataset_key(tmp_path) -> None:
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    repo.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        profile_mode="server_frontend",
        device_id="device-1",
        dataset_id="dataset-1",
    )
    producer = ChatSyncV2OutboxProducer(state_repository=repo, dataset_keys={})

    result = producer.enqueue_chat_message(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        conversation_id="conversation-1",
        message_id="message-1",
        role="user",
        content="No sync",
    )

    assert result == {"status": "skipped", "reason": "profile_not_local_first"}
    assert repo.list_sync_v2_outbox_entries(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        dataset_id="dataset-1",
    ) == []


def _local_first_repo(tmp_path) -> SyncStateRepository:
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
