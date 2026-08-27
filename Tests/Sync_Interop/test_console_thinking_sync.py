"""Console thinking is one canonical whole-record Sync v2 field."""

from __future__ import annotations

import json

import pytest

from tldw_chatbook.Chat.provider_continuation import (
    ContinuationCall,
    ContinuationRound,
    ProviderContinuationCheckpoint,
    dump_provider_continuation_json,
)
from tldw_chatbook.Chat.thinking_blocks import (
    DisplayableThinkingBlock,
    ProprietaryThinkingBlock,
    ThinkingEnvelope,
    dump_thinking_blocks_json,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Sync_Interop.chat_outbox_producer import ChatSyncV2OutboxProducer
from tldw_chatbook.Sync_Interop.crypto import (
    decrypt_sync_payload,
    encrypt_sync_payload,
    generate_dataset_key,
)
from tldw_chatbook.Sync_Interop.envelope_applier import SyncEnvelopeApplier
from tldw_chatbook.Sync_Interop.envelope_builder import SyncEnvelopeBuilder
from tldw_chatbook.Sync_Interop.hashing import canonical_payload_hash
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from tldw_chatbook.tldw_api import SyncV2Envelope


def _thinking(*, proprietary: bool = False, text: str = "visible reasoning") -> str:
    common = {
        "block_id": "reasoning-1",
        "round_ordinal": 0,
        "provider": "llama_cpp",
        "model": "test-model",
        "protocol": "chat_completions",
        "source_format": "think_tag",
        "status": "complete",
    }
    block = (
        ProprietaryThinkingBlock(**common)
        if proprietary
        else DisplayableThinkingBlock(text=text, **common)
    )
    canonical = dump_thinking_blocks_json(ThinkingEnvelope((block,)))
    assert canonical is not None
    return canonical


def _continuation() -> str:
    return dump_provider_continuation_json(
        ProviderContinuationCheckpoint(
            schema_version=1,
            checkpoint_revision=1,
            provider="moonshot",
            protocol="chat_completions",
            model="kimi-k3",
            api_base_url="https://api.moonshot.ai/v1",
            state="complete",
            rounds=(
                ContinuationRound(
                    assistant_content="answer",
                    reasoning_blocks=("PRIVATE-CONTINUATION-CANARY",),
                    calls=(),
                ),
            ),
        )
    )


def _active_continuation() -> str:
    return dump_provider_continuation_json(
        ProviderContinuationCheckpoint(
            schema_version=1,
            checkpoint_revision=1,
            provider="moonshot",
            protocol="chat_completions",
            model="kimi-k3",
            api_base_url="https://api.moonshot.ai/v1",
            state="active",
            rounds=(
                ContinuationRound(
                    assistant_content="",
                    reasoning_blocks=(),
                    calls=(
                        ContinuationCall(
                            call_id="call-1",
                            name="calculator",
                            arguments='{"expression":"2+2"}',
                            state="pending",
                        ),
                    ),
                ),
            ),
        )
    )


def _sync_repository(tmp_path) -> SyncStateRepository:
    tmp_path.mkdir(parents=True, exist_ok=True)
    repository = SyncStateRepository(tmp_path / "sync-state.db")
    repository.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        profile_mode="local_first",
        device_id="source-device",
        dataset_id="dataset-1",
    )
    return repository


def _source_message(
    tmp_path, *, proprietary: bool = False
) -> tuple[CharactersRAGDB, str, str, str]:
    db = CharactersRAGDB(tmp_path / "source.db", client_id="source-device")
    conversation_id = db.add_conversation(
        {"id": "conversation-1", "title": "Thinking sync"}
    )
    thinking = _thinking(proprietary=proprietary)
    message_id = db.add_message(
        {
            "id": "assistant-1",
            "conversation_id": conversation_id,
            "sender": "assistant",
            "role": "assistant",
            "content": "answer",
            "thinking_blocks_json": thinking,
            "provider_continuation_json": _continuation(),
            "assistant_generation_state": "complete",
        }
    )
    assert message_id == "assistant-1"
    payload = {
        "assistant_generation_state": "complete",
        "content": "answer",
        "provider_continuation_json": _continuation(),
        "role": "assistant",
        "thinking_blocks_json": thinking,
    }
    return db, conversation_id, message_id, canonical_payload_hash(payload)


def _target_db(tmp_path) -> CharactersRAGDB:
    db = CharactersRAGDB(tmp_path / "target.db", client_id="target-device")
    assert (
        db.add_conversation({"id": "conversation-1", "title": "Thinking sync"})
        == "conversation-1"
    )
    return db


def _reconcile(
    tmp_path,
    *,
    source: CharactersRAGDB,
    message_id: str,
    message_version: int,
    payload_hash: str,
    dataset_key: bytes,
) -> tuple[dict, SyncV2Envelope]:
    result = ChatSyncV2OutboxProducer(
        state_repository=_sync_repository(tmp_path),
        dataset_keys={"dataset-1": dataset_key},
        source=source,
    ).reconcile_chat_message_intent(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        message_id=message_id,
        message_version=message_version,
        payload_hash=payload_hash,
    )
    assert result["status"] == "enqueued"
    envelope = SyncV2Envelope.model_validate(result["outbox_entry"]["envelope"])
    return result, envelope


@pytest.mark.parametrize(
    "proprietary", [False, True], ids=["displayable", "proprietary"]
)
def test_real_source_outbox_and_target_round_trip_complete_generation(
    tmp_path, proprietary: bool
) -> None:
    source, _conversation_id, message_id, payload_hash = _source_message(
        tmp_path, proprietary=proprietary
    )
    target = _target_db(tmp_path)
    dataset_key = generate_dataset_key()
    try:
        result, envelope = _reconcile(
            tmp_path,
            source=source,
            message_id=message_id,
            message_version=1,
            payload_hash=payload_hash,
            dataset_key=dataset_key,
        )
        payload = decrypt_sync_payload(
            json.loads(result["outbox_entry"]["envelope"]["payload_ciphertext"]),
            key=dataset_key,
        )

        assert payload["thinking_blocks_json"] == _thinking(proprietary=proprietary)
        assert SyncEnvelopeApplier(dataset_key=dataset_key, local_store=target).apply(
            envelope
        ) == {"status": "applied"}
        target_row = target.get_message_by_id(message_id)
        source_row = source.get_message_by_id(message_id)
        assert target_row is not None and source_row is not None
        assert {
            key: target_row[key]
            for key in (
                "content",
                "thinking_blocks_json",
                "provider_continuation_json",
                "assistant_generation_state",
            )
        } == {
            key: source_row[key]
            for key in (
                "content",
                "thinking_blocks_json",
                "provider_continuation_json",
                "assistant_generation_state",
            )
        }
    finally:
        source.close_connection()
        target.close_connection()


def test_thinking_deletion_round_trips_as_whole_record(tmp_path) -> None:
    source, _conversation_id, message_id, payload_hash = _source_message(tmp_path)
    target = _target_db(tmp_path)
    dataset_key = generate_dataset_key()
    try:
        _result, initial = _reconcile(
            tmp_path,
            source=source,
            message_id=message_id,
            message_version=1,
            payload_hash=payload_hash,
            dataset_key=dataset_key,
        )
        applier = SyncEnvelopeApplier(dataset_key=dataset_key, local_store=target)
        assert applier.apply(initial)["status"] == "applied"
        assert source.soft_delete_message(message_id, expected_version=1)
        delete_hash = canonical_payload_hash({"deleted": True})
        deleted_result = ChatSyncV2OutboxProducer(
            state_repository=_sync_repository(tmp_path),
            dataset_keys={"dataset-1": dataset_key},
            source=source,
        ).reconcile_chat_message_delete_intent(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope=None,
            message_id=message_id,
            message_version=2,
            payload_hash=delete_hash,
        )
        assert deleted_result["status"] == "enqueued"
        deleted = SyncV2Envelope.model_validate(
            deleted_result["outbox_entry"]["envelope"]
        )
        assert deleted.base_version == initial.payload_hash == payload_hash
        remaining = _sync_repository(tmp_path).list_sync_v2_outbox_entries(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope=None,
            dataset_id="dataset-1",
        )
        assert len(remaining) == 1
        assert remaining[0]["envelope"]["operation"] == "delete"
        assert remaining[0]["envelope"]["object_id"] == message_id
        assert remaining[0]["envelope"]["payload_ciphertext"] is None
        assert "visible reasoning" not in json.dumps(remaining)

        assert applier.apply(deleted)["status"] == "applied"
        raw = (
            target.get_connection()
            .execute(
                "SELECT deleted, thinking_blocks_json FROM messages WHERE id = ?",
                (message_id,),
            )
            .fetchone()
        )
        assert dict(raw) == {"deleted": 1, "thinking_blocks_json": None}
        source_raw = (
            source.get_connection()
            .execute(
                "SELECT thinking_blocks_json FROM messages WHERE id = ?",
                (message_id,),
            )
            .fetchone()
        )
        delete_intent = (
            source.get_connection()
            .execute(
                "SELECT payload FROM sync_log WHERE entity = 'messages' "
                "AND entity_id = ? AND operation = 'delete' AND version = 2",
                (message_id,),
            )
            .fetchone()
        )
        all_intents = (
            source.get_connection()
            .execute(
                "SELECT operation, payload FROM sync_log WHERE entity = 'messages' "
                "AND entity_id = ? ORDER BY change_id",
                (message_id,),
            )
            .fetchall()
        )
        assert source_raw["thinking_blocks_json"] is None
        assert json.loads(delete_intent["payload"])["base_payload_hash"] == payload_hash
        assert [row["operation"] for row in all_intents] == ["delete"]
        assert "thinking_blocks_json" not in json.loads(all_intents[0]["payload"])
        assert "visible reasoning" not in delete_intent["payload"]
    finally:
        source.close_connection()
        target.close_connection()


def test_uninstrumented_v50_delete_without_base_proof_fails_closed(tmp_path) -> None:
    source, _conversation_id, message_id, _payload_hash = _source_message(tmp_path)
    try:
        source.get_connection().execute(
            "UPDATE messages SET deleted = 1, thinking_blocks_json = NULL, "
            "last_modified = ?, version = 2, client_id = ? "
            "WHERE id = ? AND version = 1 AND deleted = 0",
            ("2026-08-26T00:00:00.000Z", "source-device", message_id),
        )
        source.get_connection().commit()
        delete_hash = canonical_payload_hash({"deleted": True})

        assert (
            source.read_committed_chat_delete_intent(
                message_id=message_id,
                message_version=2,
                payload_hash=delete_hash,
            )
            is None
        )
        result = ChatSyncV2OutboxProducer(
            state_repository=_sync_repository(tmp_path / "raw-delete"),
            dataset_keys={"dataset-1": generate_dataset_key()},
            source=source,
        ).reconcile_chat_message_delete_intent(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope=None,
            message_id=message_id,
            message_version=2,
            payload_hash=delete_hash,
        )
        assert result == {"status": "skipped", "reason": "source_intent_unavailable"}
    finally:
        source.close_connection()


@pytest.mark.parametrize(
    "delete_path",
    ["direct", "subtree", "edit-descendant", "continuation-discard"],
)
def test_every_message_tombstone_path_commits_content_free_base_hash(
    tmp_path, delete_path: str
) -> None:
    db = CharactersRAGDB(tmp_path / f"{delete_path}.db", client_id="source-device")
    try:
        conversation_id = db.add_conversation({"title": delete_path})
        parent_id = db.add_message(
            {
                "id": "parent",
                "conversation_id": conversation_id,
                "sender": "user",
                "role": "user",
                "content": "question",
            }
        )
        continuation_discard = delete_path == "continuation-discard"
        content = "" if continuation_discard else "answer"
        continuation = (
            _active_continuation() if continuation_discard else _continuation()
        )
        generation_state = "continuation_active" if continuation_discard else "complete"
        message_id = db.add_message(
            {
                "id": "assistant-1",
                "conversation_id": conversation_id,
                "parent_message_id": parent_id,
                "sender": "assistant",
                "role": "assistant",
                "content": content,
                "thinking_blocks_json": _thinking(text="DELETE-CANARY"),
                "provider_continuation_json": continuation,
                "assistant_generation_state": generation_state,
            }
        )
        live_payload = {
            "assistant_generation_state": generation_state,
            "content": content,
            "provider_continuation_json": continuation,
            "role": "assistant",
            "thinking_blocks_json": _thinking(text="DELETE-CANARY"),
        }
        expected_base_hash = canonical_payload_hash(live_payload)

        if delete_path == "direct":
            assert db.soft_delete_message(message_id, expected_version=1)
        elif delete_path == "subtree":
            db.soft_delete_message_subtree(message_id, expected_version=1)
        elif delete_path == "edit-descendant":
            assert db.update_message(
                parent_id,
                {"content": "edited question"},
                expected_version=1,
                preserve_descendants=False,
            )
        else:
            assert db.update_provider_continuation(
                message_id=message_id,
                expected_message_version=1,
                provider_continuation_json=None,
                content="",
                assistant_generation_state="discarded",
            )

        row = (
            db.get_connection()
            .execute(
                "SELECT deleted, thinking_blocks_json, version FROM messages WHERE id = ?",
                (message_id,),
            )
            .fetchone()
        )
        intent = (
            db.get_connection()
            .execute(
                "SELECT payload FROM sync_log WHERE entity = 'messages' "
                "AND entity_id = ? AND operation = 'delete' AND version = ?",
                (message_id, row["version"]),
            )
            .fetchone()
        )
        all_intents = (
            db.get_connection()
            .execute(
                "SELECT operation, payload FROM sync_log WHERE entity = 'messages' "
                "AND entity_id = ? ORDER BY change_id",
                (message_id,),
            )
            .fetchall()
        )
        payload = json.loads(intent["payload"])
        assert row["deleted"] == 1
        assert row["thinking_blocks_json"] is None
        assert payload["base_payload_hash"] == expected_base_hash
        assert [item["operation"] for item in all_intents] == ["delete"]
        assert "thinking_blocks_json" not in json.loads(all_intents[0]["payload"])
        assert "DELETE-CANARY" not in intent["payload"]
    finally:
        db.close_connection()


def test_conflict_never_merges_thinking_blocks_between_whole_records(tmp_path) -> None:
    target = _target_db(tmp_path)
    dataset_key = generate_dataset_key()
    first = SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-a", dataset_key=dataset_key
    ).build_chat_message(
        conversation_id="conversation-1",
        message_id="assistant-1",
        role="assistant",
        content="local answer",
        thinking_blocks_json=_thinking(text="LOCAL-THINKING"),
        assistant_generation_state="complete",
    )
    divergent = SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-b", dataset_key=dataset_key
    ).build_chat_message(
        conversation_id="conversation-1",
        message_id="assistant-1",
        role="assistant",
        content="remote answer",
        thinking_blocks_json=_thinking(text="REMOTE-THINKING"),
        assistant_generation_state="complete",
        base_version="sha256:stale",
    )
    try:
        applier = SyncEnvelopeApplier(dataset_key=dataset_key, local_store=target)
        assert applier.apply(first)["status"] == "applied"
        assert applier.apply(divergent)["status"] == "conflict"
        row = target.get_message_by_id("assistant-1")
        assert row["content"] == "local answer"
        assert row["thinking_blocks_json"] == _thinking(text="LOCAL-THINKING")
    finally:
        target.close_connection()


@pytest.mark.parametrize(
    "thinking",
    [
        json.dumps({"version": 1, "blocks": [{"text": "MALFORMED-CANARY"}]}),
        json.dumps({"version": 99, "future": "UNSUPPORTED-CANARY"}),
    ],
    ids=["malformed-known", "unsupported-version"],
)
def test_invalid_incoming_thinking_is_rejected_before_target_transaction(
    tmp_path, thinking: str
) -> None:
    target = _target_db(tmp_path)
    dataset_key = generate_dataset_key()
    payload = {
        "assistant_generation_state": "complete",
        "content": "must not land",
        "role": "assistant",
        "thinking_blocks_json": thinking,
    }
    valid = SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-a", dataset_key=dataset_key
    ).build_chat_message(
        conversation_id="conversation-1",
        message_id="assistant-1",
        role="assistant",
        content="placeholder",
    )
    forged = valid.model_copy(
        update={
            "payload_ciphertext": encrypt_sync_payload(
                payload, key=dataset_key
            ).model_dump_json(),
            "payload_hash": canonical_payload_hash(payload),
        }
    )
    try:
        before = target.get_connection().total_changes
        result = SyncEnvelopeApplier(dataset_key=dataset_key, local_store=target).apply(
            forged
        )

        assert result["status"] == "conflict"
        assert result["conflict"]["conflict_type"] == "invalid_chat_message_payload"
        assert target.get_connection().total_changes == before
        assert target.get_message_by_id("assistant-1") is None
        assert "CANARY" not in json.dumps(result)
    finally:
        target.close_connection()


def test_conversation_policy_uses_existing_whole_conversation_sync_intent(
    tmp_path,
) -> None:
    db = CharactersRAGDB(tmp_path / "policy.db", client_id="source-device")
    try:
        conversation_id = db.add_conversation(
            {"title": "Policy", "thinking_history_policy": "include"}
        )
        created = (
            db.get_connection()
            .execute(
                "SELECT payload FROM sync_log WHERE entity = 'conversations' "
                "AND entity_id = ? ORDER BY change_id DESC LIMIT 1",
                (conversation_id,),
            )
            .fetchone()
        )
        assert json.loads(created["payload"])["thinking_history_policy"] == "include"

        assert db.update_conversation(
            conversation_id,
            {"thinking_history_policy": "exclude"},
            expected_version=1,
        )
        updated = (
            db.get_connection()
            .execute(
                "SELECT payload FROM sync_log WHERE entity = 'conversations' "
                "AND entity_id = ? ORDER BY change_id DESC LIMIT 1",
                (conversation_id,),
            )
            .fetchone()
        )
        assert json.loads(updated["payload"])["thinking_history_policy"] == "exclude"
    finally:
        db.close_connection()
