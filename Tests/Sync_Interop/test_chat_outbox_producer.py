from __future__ import annotations

import json
import sqlite3

import pytest

from tldw_chatbook.Chat.assistant_generation_state import AssistantGenerationState
from tldw_chatbook.Chat.provider_continuation import (
    dump_provider_continuation_json,
    parse_provider_continuation_json,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Sync_Interop.chat_outbox_producer import ChatSyncV2OutboxProducer
from tldw_chatbook.Sync_Interop.crypto import decrypt_sync_payload, generate_dataset_key
from tldw_chatbook.Sync_Interop.hashing import canonical_payload_hash
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository


@pytest.mark.parametrize(
    "state", [None, *(item.value for item in AssistantGenerationState)]
)
def test_source_proof_and_sync_v2_outbox_carry_explicit_generation_state(
    tmp_path, state: str | None
) -> None:
    db = CharactersRAGDB(tmp_path / f"state-{state}.db", client_id="sync-source")
    try:
        conversation_id = db.add_conversation({"title": "State proof"})
        private_json = (
            _provider_continuation_json()
            if state == AssistantGenerationState.CONTINUATION_ACTIVE.value
            else None
        )
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "assistant",
                "role": "assistant",
                "content": "visible answer",
                "provider_continuation_json": private_json,
                "assistant_generation_state": state,
            }
        )
        assert message_id is not None
        payload = {
            "assistant_generation_state": state,
            "content": "visible answer",
            "role": "assistant",
        }
        if private_json is not None:
            payload["provider_continuation_json"] = dump_provider_continuation_json(
                parse_provider_continuation_json(private_json)
            )
        payload_hash = canonical_payload_hash(payload)
        source = db.read_committed_chat_sync_intent(
            message_id=str(message_id),
            message_version=1,
            payload_hash=payload_hash,
        )
        assert source is not None
        assert source.assistant_generation_state == state

        dataset_key = generate_dataset_key()
        repo = _local_first_repo(tmp_path)
        result = ChatSyncV2OutboxProducer(
            state_repository=repo,
            dataset_keys={"dataset-1": dataset_key},
            source=db,
        ).reconcile_chat_message_intent(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope=None,
            message_id=str(message_id),
            message_version=1,
            payload_hash=payload_hash,
        )

        assert result["status"] == "enqueued"
        assert _decrypt_payload(
            result["outbox_entry"]["envelope"]["payload_ciphertext"], dataset_key
        ) == payload
    finally:
        db.close_connection()


def test_source_proof_normalizes_only_legacy_missing_generation_state(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "legacy-state.db", client_id="sync-source")
    try:
        conversation_id = db.add_conversation({"title": "Legacy state"})
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "assistant",
                "role": "assistant",
                "content": "legacy answer",
            }
        )
        assert message_id is not None
        connection = db.get_connection()
        row = connection.execute(
            "SELECT payload FROM sync_log WHERE entity = 'messages' AND entity_id = ?",
            (message_id,),
        ).fetchone()
        legacy_payload = json.loads(row["payload"])
        legacy_payload.pop("assistant_generation_state")
        connection.execute(
            "UPDATE sync_log SET payload = ? WHERE entity = 'messages' AND entity_id = ?",
            (json.dumps(legacy_payload), message_id),
        )
        connection.commit()
        payload_hash = canonical_payload_hash(
            {
                "assistant_generation_state": None,
                "content": "legacy answer",
                "role": "assistant",
            }
        )

        assert db.read_committed_chat_sync_intent(
            message_id=str(message_id),
            message_version=1,
            payload_hash=payload_hash,
        ) is not None

        legacy_payload["unexpected"] = True
        connection.execute(
            "UPDATE sync_log SET payload = ? WHERE entity = 'messages' AND entity_id = ?",
            (json.dumps(legacy_payload), message_id),
        )
        connection.commit()
        assert db.read_committed_chat_sync_intent(
            message_id=str(message_id),
            message_version=1,
            payload_hash=payload_hash,
        ) is None
    finally:
        db.close_connection()


def test_source_proof_rejects_malformed_wrong_role_and_mismatched_state(
    tmp_path,
) -> None:
    db = CharactersRAGDB(tmp_path / "bad-state.db", client_id="sync-source")
    try:
        conversation_id = db.add_conversation({"title": "Bad state"})
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "role": "user",
                "content": "visible",
            }
        )
        assert message_id is not None
        payload_hash = canonical_payload_hash(
            {
                "assistant_generation_state": None,
                "content": "visible",
                "role": "user",
            }
        )
        connection = db.get_connection()
        for state in ("accepted", "unknown"):
            payload = json.loads(
                connection.execute(
                    "SELECT payload FROM sync_log WHERE entity = 'messages' "
                    "AND entity_id = ?",
                    (message_id,),
                ).fetchone()["payload"]
            )
            payload["assistant_generation_state"] = state
            connection.execute(
                "UPDATE sync_log SET payload = ? WHERE entity = 'messages' "
                "AND entity_id = ?",
                (json.dumps(payload), message_id),
            )
            connection.commit()
            assert db.read_committed_chat_sync_intent(
                message_id=str(message_id),
                message_version=1,
                payload_hash=payload_hash,
            ) is None
    finally:
        db.close_connection()


def test_source_proof_rejects_illegal_nonassistant_persisted_state(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "bad-row-state.db", client_id="sync-source")
    try:
        conversation_id = db.add_conversation({"title": "Bad row state"})
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "role": "user",
                "content": "visible",
            }
        )
        assert message_id is not None
        payload_hash = canonical_payload_hash(
            {
                "assistant_generation_state": None,
                "content": "visible",
                "role": "user",
            }
        )
        assert (
            db.read_committed_chat_sync_intent(
                message_id=str(message_id),
                message_version=1,
                payload_hash=payload_hash,
            )
            is not None
        )
        corrupt_statement = (
            "UPDATE messages SET assistant_generation_state = 'accepted' WHERE id = ?"
        )
        with pytest.raises(sqlite3.IntegrityError, match="semantic mutation"):
            with db.transaction() as cursor:
                cursor.execute(corrupt_statement, (message_id,))
        with db.transaction() as cursor:
            cursor.execute("DROP TRIGGER messages_sync_update")
            # Deliberately inject an invalid persisted row, not a supported
            # application write. Keep the guard installed and scope the fixture
            # authorization to this message and this transaction only.
            authorization = db._semantic_mutation_authorization_for_coordinator(
                cursor.connection
            )
            with authorization._authorize(
                message_id=message_id, operations={"message_update"}
            ):
                cursor.execute(corrupt_statement, (message_id,))
            with pytest.raises(sqlite3.IntegrityError, match="semantic mutation"):
                cursor.execute(
                    "UPDATE messages SET assistant_generation_state = NULL WHERE id = ?",
                    (message_id,),
                )

        db.close_connection()
        row = db.get_message_by_id(message_id)
        assert row["role"] == "user"
        assert row["assistant_generation_state"] == "accepted"
        assert row["version"] == 1
        assert (
            db.read_committed_chat_sync_intent(
                message_id=str(message_id),
                message_version=1,
                payload_hash=payload_hash,
            )
            is None
        )
    finally:
        db.close_connection()


def test_source_proof_rejects_continuation_active_without_continuation(
    tmp_path,
) -> None:
    db = CharactersRAGDB(tmp_path / "missing-continuation.db", client_id="sync-source")
    try:
        conversation_id = db.add_conversation({"title": "Missing continuation"})
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "assistant",
                "role": "assistant",
                "content": "",
                "assistant_generation_state": "continuation_active",
            }
        )
        assert message_id is not None
        payload_hash = canonical_payload_hash(
            {
                "assistant_generation_state": "continuation_active",
                "content": "",
                "role": "assistant",
            }
        )

        assert db.read_committed_chat_sync_intent(
            message_id=str(message_id),
            message_version=1,
            payload_hash=payload_hash,
        ) is None
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    ("role", "delete_state"),
    [
        ("assistant", "unknown"),
        ("user", "accepted"),
        ("assistant", "continuation_active"),
    ],
)
def test_undelete_source_proof_rejects_invalid_prior_delete_state(
    tmp_path, role: str, delete_state: str
) -> None:
    db = CharactersRAGDB(
        tmp_path / f"bad-delete-{role}-{delete_state}.db", client_id="sync-source"
    )
    try:
        conversation_id = db.add_conversation({"title": "Bad delete"})
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": role,
                "role": role,
                "content": "visible",
            }
        )
        assert message_id is not None
        assert db.soft_delete_message(str(message_id), expected_version=1)
        connection = db.get_connection()
        deleted_payload = json.loads(
            connection.execute(
                "SELECT payload FROM sync_log WHERE entity = 'messages' "
                "AND entity_id = ? AND version = 2",
                (message_id,),
            ).fetchone()["payload"]
        )
        deleted_payload["assistant_generation_state"] = delete_state
        connection.execute(
            "UPDATE sync_log SET payload = ? WHERE entity = 'messages' "
            "AND entity_id = ? AND version = 2",
            (json.dumps(deleted_payload), message_id),
        )
        connection.commit()
        with db.transaction() as connection:
            connection.execute("DROP TRIGGER messages_au")
            connection.execute(
                "UPDATE messages SET deleted = 0, version = 3, last_modified = ? "
                "WHERE id = ?",
                (db._get_current_utc_timestamp_iso(), message_id),
            )
        payload_hash = canonical_payload_hash(
            {
                "assistant_generation_state": None,
                "content": "visible",
                "role": role,
            }
        )

        assert db.read_committed_chat_sync_intent(
            message_id=str(message_id),
            message_version=3,
            payload_hash=payload_hash,
        ) is None
    finally:
        db.close_connection()


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
        "assistant_generation_state": None,
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


def test_legacy_direct_chat_enqueue_rejects_thinking_before_repository_access(
    tmp_path, monkeypatch
) -> None:
    repo = _local_first_repo(tmp_path)
    profile_accesses: list[str] = []

    def unexpected_profile_access(**_kwargs):
        profile_accesses.append("accessed")
        raise AssertionError("repository must not be consulted")

    monkeypatch.setattr(repo, "get_sync_v2_profile_state", unexpected_profile_access)
    producer = ChatSyncV2OutboxProducer(
        state_repository=repo,
        dataset_keys={"dataset-1": generate_dataset_key()},
    )

    with pytest.raises(ValueError, match="committed-intent reconciliation"):
        producer.enqueue_chat_message(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope=None,
            conversation_id="conversation-1",
            message_id="message-1",
            role="assistant",
            content="visible",
            thinking_blocks_json="UNCOMMITTED-THINKING-CANARY",
        )

    assert profile_accesses == []
    assert (
        repo.list_sync_v2_outbox_entries(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope=None,
            dataset_id="dataset-1",
        )
        == []
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
                "assistant_generation_state": "continuation_active",
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
        assert source.base_payload_hash is None
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


def test_chachanotes_source_read_uses_database_transaction(
    tmp_path, monkeypatch
) -> None:
    private_json = _provider_continuation_json()
    db = CharactersRAGDB(tmp_path / "source-transaction.db", client_id="sync-source")
    try:
        conversation_id = db.add_conversation({"title": "Source transaction"})
        message_id = "assistant-source-transaction"
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
                "assistant_generation_state": "continuation_active",
                "content": "visible answer",
                "provider_continuation_json": canonical_private,
                "role": "assistant",
            }
        )
        real_transaction = db.transaction
        transaction_calls = 0

        def recording_transaction():
            nonlocal transaction_calls
            transaction_calls += 1
            return real_transaction()

        monkeypatch.setattr(db, "transaction", recording_transaction)

        assert (
            db.read_committed_chat_sync_intent(
                message_id=message_id,
                message_version=1,
                payload_hash=payload_hash,
            )
            is not None
        )
        assert transaction_calls == 1
    finally:
        db.close_connection()


def test_message_tombstone_read_uses_database_transaction(tmp_path, monkeypatch) -> None:
    db = CharactersRAGDB(
        tmp_path / "tombstone-transaction.db", client_id="sync-source"
    )
    try:
        conversation_id = db.add_conversation({"title": "Tombstone transaction"})
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "role": "user",
                "content": "delete me",
            }
        )
        assert message_id is not None
        assert db.soft_delete_message(str(message_id), expected_version=1)
        real_transaction = db.transaction
        transaction_calls = 0

        def recording_transaction():
            nonlocal transaction_calls
            transaction_calls += 1
            return real_transaction()

        monkeypatch.setattr(db, "transaction", recording_transaction)

        assert db.get_message_tombstones([str(message_id)]) == [
            {
                "message_id": str(message_id),
                "conversation_id": conversation_id,
                "version": 2,
            }
        ]
        assert transaction_calls == 1
    finally:
        db.close_connection()


def test_delete_source_and_current_intent_list_use_database_transactions(
    tmp_path, monkeypatch
) -> None:
    db = CharactersRAGDB(
        tmp_path / "delete-source-transaction.db", client_id="sync-source"
    )
    try:
        conversation_id = db.add_conversation({"title": "Delete source transaction"})
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "role": "user",
                "content": "delete me",
            }
        )
        assert message_id is not None
        assert db.soft_delete_message(str(message_id), expected_version=1)
        real_transaction = db.transaction
        transaction_calls = 0

        def recording_transaction():
            nonlocal transaction_calls
            transaction_calls += 1
            return real_transaction()

        monkeypatch.setattr(db, "transaction", recording_transaction)
        deleted_hash = canonical_payload_hash({"deleted": True})

        assert (
            db.read_committed_chat_delete_intent(
                message_id=str(message_id),
                message_version=2,
                payload_hash=deleted_hash,
            )
            is not None
        )
        intents = db.list_current_committed_chat_sync_intents(conversation_id)

        assert intents == [
            {
                "message_id": str(message_id),
                "message_version": 2,
                "operation": "delete",
                "payload_hash": deleted_hash,
            }
        ]
        assert transaction_calls >= 3
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
                "assistant_generation_state": "continuation_active",
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
                    {
                        "assistant_generation_state": None,
                        "content": "visible answer",
                        "role": "assistant",
                    }
                ),
            )
            is None
        )
        connection.rollback()

        db.soft_delete_message(message_id, expected_version=2)
        deleted_hash = canonical_payload_hash(
            {
                "assistant_generation_state": None,
                "content": "visible answer",
                "role": "assistant",
            }
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
                "assistant_generation_state": "continuation_active",
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
            "assistant_generation_state": "continuation_active",
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


def test_reconcile_same_payload_later_version_gets_distinct_outbox_entry(
    tmp_path,
) -> None:
    db = CharactersRAGDB(tmp_path / "source-version.db", client_id="sync-source")
    try:
        conversation_id = db.add_conversation({"title": "Version projection"})
        message_id = "assistant-version-1"
        db.create_assistant_with_continuation(
            message_id=message_id,
            conversation_id=conversation_id,
            parent_message_id=None,
            content="unchanged visible answer",
            provider_continuation_json=_provider_continuation_json(),
        )
        source_row = db.get_message_by_id(message_id)
        payload_hash = canonical_payload_hash(
            {
                "assistant_generation_state": "continuation_active",
                "content": source_row["content"],
                "provider_continuation_json": source_row["provider_continuation_json"],
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
        scope = {
            "server_profile_id": "server-a",
            "authenticated_principal_id": "user-a",
            "workspace_scope": None,
        }

        first = producer.reconcile_chat_message_intent(
            **scope,
            message_id=message_id,
            message_version=1,
            payload_hash=payload_hash,
        )
        first_id = first["outbox_entry"]["client_envelope_id"]
        assert repo.mark_sync_v2_outbox_push_results(
            **scope,
            dataset_id="dataset-1",
            accepted=[{"client_envelope_id": first_id}],
            rejected=[],
            conflicts=[],
        ) == {"dispatched": 1, "retained": 0}
        first_dispatched = repo.list_sync_v2_outbox_entries(
            **scope, dataset_id="dataset-1"
        )[0]

        assert db.update_message(message_id, {"ranking": 1}, expected_version=1)
        second = producer.reconcile_chat_message_intent(
            **scope,
            message_id=message_id,
            message_version=2,
            payload_hash=payload_hash,
        )
        entries = repo.list_sync_v2_outbox_entries(**scope, dataset_id="dataset-1")

        assert len(entries) == 2
        assert second["outbox_entry"]["client_envelope_id"] != first_id
        assert entries[0] == first_dispatched
        assert entries[0]["status"] == "dispatched"
        assert entries[0]["envelope"]["entity_version"] == 1
        assert entries[1]["status"] == "pending"
        assert entries[1]["attempt_count"] == 0
        assert entries[1]["envelope"]["entity_version"] == 2
        assert entries[1]["envelope"]["base_version"] == payload_hash
        assert _decrypt_payload(
            entries[0]["envelope"]["payload_ciphertext"], dataset_key
        ) == _decrypt_payload(entries[1]["envelope"]["payload_ciphertext"], dataset_key)
        with repo._get_connection() as conn:
            receipts = conn.execute(
                """
                SELECT source_version, client_envelope_id
                  FROM sync_v2_source_projection_receipts
                 ORDER BY source_version
                """
            ).fetchall()
        assert [
            (row["source_version"], row["client_envelope_id"]) for row in receipts
        ] == [
            (1, first_id),
            (2, second["outbox_entry"]["client_envelope_id"]),
        ]
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
                "assistant_generation_state": "continuation_active",
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
