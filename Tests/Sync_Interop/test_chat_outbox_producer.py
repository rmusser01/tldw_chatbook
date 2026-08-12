from __future__ import annotations

import json

import pytest

from tldw_chatbook.Chat.provider_continuation import (
    dump_provider_continuation_json,
    parse_provider_continuation_json,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Sync_Interop.chat_outbox_producer import ChatSyncV2OutboxProducer
from tldw_chatbook.Sync_Interop.crypto import decrypt_sync_payload, generate_dataset_key
from tldw_chatbook.Sync_Interop.hashing import canonical_payload_hash
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


def test_chat_producer_skips_without_local_first_profile_or_dataset_key(
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
    assert (
        repo.list_sync_v2_outbox_entries(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope=None,
            dataset_id="dataset-1",
        )
        == []
    )


def test_chat_producer_does_not_transport_or_infer_local_character_authority(
    tmp_path,
) -> None:
    db = CharactersRAGDB(tmp_path / "chat.db", client_id="sync-test")
    try:
        character_id = db.add_character_card({"name": "Local Character"})
        conversation_id = db.add_conversation(
            {
                "character_id": character_id,
                "assistant_kind": "character",
                "assistant_id": str(character_id),
                "runtime_backend": "local",
            }
        )
        conversation = db.get_conversation_by_id(conversation_id)
        authority_id = conversation["assistant_authority_id"]
        assert authority_id == db.get_local_authority_id()

        dataset_key = generate_dataset_key()
        repo = _local_first_repo(tmp_path)
        producer = ChatSyncV2OutboxProducer(
            state_repository=repo,
            dataset_keys={"dataset-1": dataset_key},
        )

        producer.enqueue_chat_message(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope=None,
            conversation_id=conversation_id,
            message_id="message-1",
            role="assistant",
            content="Local answer",
        )

        envelope = repo.list_pending_sync_v2_outbox_envelopes(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope=None,
            dataset_id="dataset-1",
        )[0]["envelope"]
        payload = _decrypt_payload(envelope["payload_ciphertext"], dataset_key)

        assert "assistant_authority_id" not in envelope["routing_metadata"]
        assert "assistant_authority_id" not in payload
        assert authority_id not in json.dumps(envelope)
    finally:
        db.close_connection()


def test_chachanotes_source_reads_only_exact_committed_message_intent(tmp_path) -> None:
    canary = "SOURCE-PRIVATE-CANARY"
    private_json = _provider_continuation_json(canary=canary)
    db = CharactersRAGDB(tmp_path / "source.db", client_id="sync-source")
    try:
        conversation_id = db.add_conversation({"title": "Source proof"})
        message_id = "assistant-source-1"
        db.create_assistant_with_continuation(
            message_id=message_id,
            conversation_id=conversation_id,
            parent_message_id=None,
            content="visible answer",
            provider_continuation_json=private_json,
        )
        canonical_private = dump_provider_continuation_json(
            parse_provider_continuation_json(private_json)
        )
        payload_hash = canonical_payload_hash(
            {
                "content": "visible answer",
                "provider_continuation_json": canonical_private,
                "role": "assistant",
            }
        )

        source = db.read_committed_chat_sync_intent(
            message_id=message_id,
            message_version=1,
            payload_hash=payload_hash,
        )

        assert source is not None
        assert source.message_id == message_id
        assert source.conversation_id == conversation_id
        assert source.message_version == 1
        assert source.payload_hash == payload_hash
        assert source.role == "assistant"
        assert source.content == "visible answer"
        assert source.provider_continuation_json == canonical_private
        assert canary not in repr(source)
        assert (
            db.read_committed_chat_sync_intent(
                message_id=message_id,
                message_version=2,
                payload_hash=payload_hash,
            )
            is None
        )
        assert (
            db.read_committed_chat_sync_intent(
                message_id=message_id,
                message_version=1,
                payload_hash="sha256:" + "0" * 64,
            )
            is None
        )
    finally:
        db.close_connection()


def test_chachanotes_source_rejects_uncommitted_ambiguous_and_deleted_intents(
    tmp_path,
) -> None:
    private_json = _provider_continuation_json()
    db = CharactersRAGDB(tmp_path / "source-rejections.db", client_id="sync-source")
    try:
        conversation_id = db.add_conversation({"title": "Source proof"})
        message_id = "assistant-source-2"
        db.create_assistant_with_continuation(
            message_id=message_id,
            conversation_id=conversation_id,
            parent_message_id=None,
            content="visible answer",
            provider_continuation_json=private_json,
        )
        canonical_private = dump_provider_continuation_json(
            parse_provider_continuation_json(private_json)
        )
        payload_hash = canonical_payload_hash(
            {
                "content": "visible answer",
                "provider_continuation_json": canonical_private,
                "role": "assistant",
            }
        )
        connection = db.get_connection()
        connection.execute(
            """
            INSERT INTO sync_log (
                entity, entity_id, operation, timestamp, client_id, version, payload
            )
            SELECT entity, entity_id, operation, timestamp, client_id, version, payload
              FROM sync_log
             WHERE entity = 'messages' AND entity_id = ? AND version = 1
            """,
            (message_id,),
        )
        connection.commit()

        assert (
            db.read_committed_chat_sync_intent(
                message_id=message_id,
                message_version=1,
                payload_hash=payload_hash,
            )
            is None
        )

        connection.execute(
            "DELETE FROM sync_log WHERE entity = 'messages' AND entity_id = ?",
            (message_id,),
        )
        connection.commit()
        db.update_provider_continuation(
            message_id=message_id,
            expected_message_version=1,
            provider_continuation_json=None,
        )
        connection.execute("BEGIN")
        assert (
            db.read_committed_chat_sync_intent(
                message_id=message_id,
                message_version=2,
                payload_hash=canonical_payload_hash(
                    {"content": "visible answer", "role": "assistant"}
                ),
            )
            is None
        )
        connection.rollback()

        db.soft_delete_message(message_id, expected_version=2)
        deleted_hash = canonical_payload_hash(
            {"content": "visible answer", "role": "assistant"}
        )
        assert (
            db.read_committed_chat_sync_intent(
                message_id=message_id,
                message_version=3,
                payload_hash=deleted_hash,
            )
            is None
        )
    finally:
        db.close_connection()


def test_chat_producer_reconciles_only_from_exact_source_proof(tmp_path) -> None:
    private_json = _provider_continuation_json(canary="PROJECTION-PRIVATE-CANARY")
    db = CharactersRAGDB(tmp_path / "source-projection.db", client_id="sync-source")
    try:
        conversation_id = db.add_conversation({"title": "Projection"})
        message_id = "assistant-projection-1"
        db.create_assistant_with_continuation(
            message_id=message_id,
            conversation_id=conversation_id,
            parent_message_id=None,
            content="visible answer",
            provider_continuation_json=private_json,
        )
        canonical_private = dump_provider_continuation_json(
            parse_provider_continuation_json(private_json)
        )
        payload_hash = canonical_payload_hash(
            {
                "content": "visible answer",
                "provider_continuation_json": canonical_private,
                "role": "assistant",
            }
        )
        dataset_key = generate_dataset_key()
        repo = _local_first_repo(tmp_path)
        producer = ChatSyncV2OutboxProducer(
            state_repository=repo,
            dataset_keys={"dataset-1": dataset_key},
            source=db,
        )

        result = producer.reconcile_chat_message_intent(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope=None,
            message_id=message_id,
            message_version=1,
            payload_hash=payload_hash,
        )

        assert result["status"] == "enqueued"
        assert result["receipt"]["source_version"] == 1
        envelope = result["outbox_entry"]["envelope"]
        assert _decrypt_payload(envelope["payload_ciphertext"], dataset_key) == {
            "content": "visible answer",
            "provider_continuation_json": canonical_private,
            "role": "assistant",
        }
        serialized = json.dumps(result)
        assert "PROJECTION-PRIVATE-CANARY" not in serialized
        assert producer.reconcile_chat_message_intent(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope=None,
            message_id=message_id,
            message_version=2,
            payload_hash=payload_hash,
        ) == {"status": "skipped", "reason": "source_intent_unavailable"}
    finally:
        db.close_connection()


def test_reconcile_resumes_after_commit_before_producer_return(
    tmp_path, monkeypatch
) -> None:
    db = CharactersRAGDB(tmp_path / "source-crash.db", client_id="sync-source")
    try:
        conversation_id = db.add_conversation({"title": "Crash recovery"})
        message_id = "assistant-crash-1"
        db.create_assistant_with_continuation(
            message_id=message_id,
            conversation_id=conversation_id,
            parent_message_id=None,
            content="visible answer",
            provider_continuation_json=_provider_continuation_json(),
        )
        source_row = db.get_message_by_id(message_id)
        payload_hash = canonical_payload_hash(
            {
                "content": source_row["content"],
                "provider_continuation_json": source_row["provider_continuation_json"],
                "role": "assistant",
            }
        )
        repo = _local_first_repo(tmp_path)
        producer = ChatSyncV2OutboxProducer(
            state_repository=repo,
            dataset_keys={"dataset-1": generate_dataset_key()},
            source=db,
        )
        original = repo.enqueue_sync_v2_outbox_envelope_with_source_receipt
        crashed = False

        def commit_then_die(**kwargs):
            nonlocal crashed
            result = original(**kwargs)
            if not crashed:
                crashed = True
                raise RuntimeError("simulated process death")
            return result

        monkeypatch.setattr(
            repo,
            "enqueue_sync_v2_outbox_envelope_with_source_receipt",
            commit_then_die,
        )
        arguments = {
            "server_profile_id": "server-a",
            "authenticated_principal_id": "user-a",
            "workspace_scope": None,
            "message_id": message_id,
            "message_version": 1,
            "payload_hash": payload_hash,
        }
        with pytest.raises(RuntimeError, match="simulated process death"):
            producer.reconcile_chat_message_intent(**arguments)

        resumed = producer.reconcile_chat_message_intent(**arguments)
        assert resumed["status"] == "enqueued"
        assert (
            len(
                repo.list_sync_v2_outbox_entries(
                    server_profile_id="server-a",
                    authenticated_principal_id="user-a",
                    workspace_scope=None,
                    dataset_id="dataset-1",
                )
            )
            == 1
        )
        with repo._get_connection() as conn:
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM sync_v2_source_projection_receipts"
                ).fetchone()[0]
                == 1
            )
    finally:
        db.close_connection()


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


def _decrypt_payload(payload_ciphertext: str, dataset_key: bytes) -> dict:
    return decrypt_sync_payload(json.loads(payload_ciphertext), key=dataset_key)
