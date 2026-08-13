from __future__ import annotations

import json

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.provider_continuation import (
    dump_provider_continuation_json,
    parse_provider_continuation_json,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Sync_Interop.chat_outbox_producer import ChatSyncV2OutboxProducer
from tldw_chatbook.Sync_Interop.crypto import decrypt_sync_payload, generate_dataset_key
from tldw_chatbook.Sync_Interop.hashing import canonical_payload_hash
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository


def test_barrier_accepts_committed_local_intent_when_portable_sync_is_absent(
    tmp_path,
) -> None:
    db, message_id, payload_hash = _source_message(tmp_path)
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))

        result = store.ensure_provider_continuation_durable(
            message_id=message_id,
            message_version=1,
            payload_hash=payload_hash,
        )

        assert result.ready is True
        assert result.reason == "local_intent_durable"
        assert "PRIVATE-CONTINUATION-CANARY" not in repr(result)
    finally:
        db.close_connection()


def test_barrier_requires_only_atomic_local_projection_not_remote_ack(tmp_path) -> None:
    db, message_id, payload_hash = _source_message(tmp_path)
    try:
        repo = _configured_repo(tmp_path / "sync-state.db")
        producer = ChatSyncV2OutboxProducer(
            state_repository=repo,
            dataset_keys={"dataset-1": generate_dataset_key()},
            source=db,
        )
        store = ConsoleChatStore(
            persistence=ChatPersistenceService(db),
            sync_v2_chat_producer=producer,
            sync_v2_server_profile_id="server-a",
            sync_v2_authenticated_principal_id="user-a",
            sync_v2_workspace_scope="workspace-1",
        )

        result = store.ensure_provider_continuation_durable(
            message_id=message_id,
            message_version=1,
            payload_hash=payload_hash,
        )

        assert result.ready is True
        assert result.reason == "portable_projection_durable"
        assert repo.list_sync_v2_outbox_entries(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            dataset_id="dataset-1",
            status="pending",
        )
    finally:
        db.close_connection()


def test_barrier_blocks_configured_memory_unavailable_and_stale_source(
    tmp_path,
) -> None:
    db, message_id, payload_hash = _source_message(tmp_path)
    try:
        memory_repo = _configured_repo(":memory:")
        memory_producer = ChatSyncV2OutboxProducer(
            state_repository=memory_repo,
            dataset_keys={"dataset-1": generate_dataset_key()},
            source=db,
        )
        memory_store = ConsoleChatStore(
            persistence=ChatPersistenceService(db),
            sync_v2_chat_producer=memory_producer,
            sync_v2_server_profile_id="server-a",
            sync_v2_authenticated_principal_id="user-a",
            sync_v2_workspace_scope="workspace-1",
        )
        unavailable_store = ConsoleChatStore(
            persistence=ChatPersistenceService(db),
            sync_v2_server_profile_id="server-a",
        )

        results = [
            memory_store.ensure_provider_continuation_durable(
                message_id=message_id,
                message_version=1,
                payload_hash=payload_hash,
            ),
            unavailable_store.ensure_provider_continuation_durable(
                message_id=message_id,
                message_version=1,
                payload_hash=payload_hash,
            ),
            ConsoleChatStore(
                persistence=ChatPersistenceService(db)
            ).ensure_provider_continuation_durable(
                message_id=message_id,
                message_version=2,
                payload_hash=payload_hash,
            ),
        ]

        assert [result.ready for result in results] == [False, False, False]
        assert all(len(result.reason) <= 160 for result in results)
        assert "PRIVATE-CONTINUATION-CANARY" not in repr(results)
    finally:
        db.close_connection()


def test_barrier_blocks_wrong_scope_and_orphaned_receipt(tmp_path, monkeypatch) -> None:
    db, message_id, payload_hash = _source_message(tmp_path)
    try:
        repo = _configured_repo(tmp_path / "sync-state.db")
        producer = ChatSyncV2OutboxProducer(
            state_repository=repo,
            dataset_keys={"dataset-1": generate_dataset_key()},
            source=db,
        )
        wrong_scope = ConsoleChatStore(
            persistence=ChatPersistenceService(db),
            sync_v2_chat_producer=producer,
            sync_v2_server_profile_id="server-a",
            sync_v2_authenticated_principal_id="user-b",
            sync_v2_workspace_scope="workspace-1",
        ).ensure_provider_continuation_durable(
            message_id=message_id,
            message_version=1,
            payload_hash=payload_hash,
        )
        projected = producer.reconcile_chat_message_intent(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            message_id=message_id,
            message_version=1,
            payload_hash=payload_hash,
        )
        with repo._get_connection() as conn:
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute("DELETE FROM sync_v2_local_outbox")
        monkeypatch.setattr(
            producer,
            "reconcile_chat_message_intent",
            lambda **_kwargs: projected,
        )
        orphaned = ConsoleChatStore(
            persistence=ChatPersistenceService(db),
            sync_v2_chat_producer=producer,
            sync_v2_server_profile_id="server-a",
            sync_v2_authenticated_principal_id="user-a",
            sync_v2_workspace_scope="workspace-1",
        ).ensure_provider_continuation_durable(
            message_id=message_id,
            message_version=1,
            payload_hash=payload_hash,
        )

        assert wrong_scope.ready is False
        assert orphaned.ready is False
    finally:
        db.close_connection()


def test_clear_and_later_edit_project_distinct_whole_message_versions(tmp_path) -> None:
    db, message_id, first_hash = _source_message(tmp_path)
    try:
        dataset_key = generate_dataset_key()
        repo = _configured_repo(tmp_path / "sync-state.db")
        producer = ChatSyncV2OutboxProducer(
            state_repository=repo,
            dataset_keys={"dataset-1": dataset_key},
            source=db,
        )
        scope = {
            "server_profile_id": "server-a",
            "authenticated_principal_id": "user-a",
            "workspace_scope": "workspace-1",
        }
        producer.reconcile_chat_message_intent(
            **scope,
            message_id=message_id,
            message_version=1,
            payload_hash=first_hash,
        )
        db.update_provider_continuation(
            message_id=message_id,
            expected_message_version=1,
            provider_continuation_json=None,
        )
        cleared_hash = canonical_payload_hash(
            {"content": "visible answer", "role": "assistant"}
        )
        producer.reconcile_chat_message_intent(
            **scope,
            message_id=message_id,
            message_version=2,
            payload_hash=cleared_hash,
        )

        entries = repo.list_sync_v2_outbox_entries(**scope, dataset_id="dataset-1")
        payloads = [
            decrypt_sync_payload(
                json.loads(entry["envelope"]["payload_ciphertext"]), key=dataset_key
            )
            for entry in entries
        ]
        assert len(entries) == 2
        assert "provider_continuation_json" in payloads[0]
        assert payloads[0]["content"] == "visible answer"
        assert payloads[1] == {"content": "visible answer", "role": "assistant"}
        with repo._get_connection() as conn:
            assert [
                row[0]
                for row in conn.execute(
                    "SELECT source_version FROM sync_v2_source_projection_receipts "
                    "ORDER BY source_version"
                ).fetchall()
            ] == [1, 2]
    finally:
        db.close_connection()


def test_visible_edit_keeps_checkpoint_on_its_exact_new_message_version(
    tmp_path,
) -> None:
    db, message_id, first_hash = _source_message(tmp_path)
    try:
        dataset_key = generate_dataset_key()
        repo = _configured_repo(tmp_path / "sync-state.db")
        producer = ChatSyncV2OutboxProducer(
            state_repository=repo,
            dataset_keys={"dataset-1": dataset_key},
            source=db,
        )
        scope = {
            "server_profile_id": "server-a",
            "authenticated_principal_id": "user-a",
            "workspace_scope": "workspace-1",
        }
        producer.reconcile_chat_message_intent(
            **scope,
            message_id=message_id,
            message_version=1,
            payload_hash=first_hash,
        )
        db.update_message(
            message_id,
            {"content": "edited visible answer"},
            expected_version=1,
        )
        edited_hash = _current_message_payload_hash(db, message_id)

        producer.reconcile_chat_message_intent(
            **scope,
            message_id=message_id,
            message_version=2,
            payload_hash=edited_hash,
        )

        entries = repo.list_sync_v2_outbox_entries(**scope, dataset_id="dataset-1")
        edited_payload = decrypt_sync_payload(
            json.loads(entries[1]["envelope"]["payload_ciphertext"]), key=dataset_key
        )
        assert edited_payload["content"] == "edited visible answer"
        assert (
            "PRIVATE-CONTINUATION-CANARY"
            in edited_payload["provider_continuation_json"]
        )
        assert [entry["envelope"]["entity_version"] for entry in entries] == [1, 2]
    finally:
        db.close_connection()


def test_delete_is_not_resumable_and_undelete_projects_new_whole_version(
    tmp_path,
) -> None:
    db, message_id, first_hash = _source_message(tmp_path)
    try:
        repo = _configured_repo(tmp_path / "sync-state.db")
        producer = ChatSyncV2OutboxProducer(
            state_repository=repo,
            dataset_keys={"dataset-1": generate_dataset_key()},
            source=db,
        )
        scope = {
            "server_profile_id": "server-a",
            "authenticated_principal_id": "user-a",
            "workspace_scope": "workspace-1",
        }
        producer.reconcile_chat_message_intent(
            **scope,
            message_id=message_id,
            message_version=1,
            payload_hash=first_hash,
        )
        db.soft_delete_message(message_id, expected_version=1)
        assert producer.reconcile_chat_message_intent(
            **scope,
            message_id=message_id,
            message_version=2,
            payload_hash=first_hash,
        ) == {"status": "skipped", "reason": "source_intent_unavailable"}
        with db.transaction() as conn:
            # This test exercises the sync undelete trigger directly. The legacy
            # messages FTS trigger assumes an indexed old row and cannot express
            # undelete for a tombstone, so keep that unrelated index path out.
            conn.execute("DROP TRIGGER messages_au")
            conn.execute(
                """UPDATE messages
                      SET deleted = 0, version = 3, last_modified = ?
                    WHERE id = ? AND version = 2 AND deleted = 1""",
                (db._get_current_utc_timestamp_iso(), message_id),
            )
        undeleted_hash = _current_message_payload_hash(db, message_id)
        assert (
            producer.reconcile_chat_message_intent(
                **scope,
                message_id=message_id,
                message_version=3,
                payload_hash=undeleted_hash,
            )["status"]
            == "enqueued"
        )
        with repo._get_connection() as conn:
            versions = [
                row[0]
                for row in conn.execute(
                    "SELECT source_version FROM sync_v2_source_projection_receipts "
                    "ORDER BY source_version"
                ).fetchall()
            ]
        assert versions == [1, 3]
    finally:
        db.close_connection()


def test_committed_delete_projects_exact_sync_v2_tombstone(tmp_path) -> None:
    db, message_id, _first_hash = _source_message(tmp_path)
    try:
        repo = _configured_repo(tmp_path / "delete-sync-state.db")
        producer = ChatSyncV2OutboxProducer(
            state_repository=repo,
            dataset_keys={"dataset-1": generate_dataset_key()},
            source=db,
        )
        deleted_hash = canonical_payload_hash({"deleted": True})
        tombstones = db.soft_delete_message_subtree(message_id, expected_version=1)

        result = producer.reconcile_chat_message_delete_intent(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            message_id=message_id,
            message_version=tombstones[0]["version"],
            payload_hash=deleted_hash,
        )

        assert result["status"] == "enqueued"
        envelope = result["outbox_entry"]["envelope"]
        assert envelope["operation"] == "delete"
        assert envelope["entity_version"] == 2
        assert envelope["payload_clear"] == {"deleted": True}
        assert envelope["payload_ciphertext"] is None
        assert result["receipt"]["source_version"] == 2
    finally:
        db.close_connection()


def test_branch_variants_keep_private_checkpoint_on_distinct_stable_ids(
    tmp_path,
) -> None:
    db = CharactersRAGDB(tmp_path / "branches.db", client_id="source-client")
    try:
        conversation_id = db.add_conversation({"title": "Branches"})
        for message_id, canary in (
            ("variant-message-1", "VARIANT-ONE"),
            ("variant-message-2", "VARIANT-TWO"),
        ):
            db.create_assistant_with_continuation(
                message_id=message_id,
                conversation_id=conversation_id,
                parent_message_id=None,
                content=f"answer for {message_id}",
                provider_continuation_json=_provider_continuation_json().replace(
                    "PRIVATE-CONTINUATION-CANARY", canary
                ),
            )
        repo = _configured_repo(tmp_path / "sync-state.db")
        producer = ChatSyncV2OutboxProducer(
            state_repository=repo,
            dataset_keys={"dataset-1": generate_dataset_key()},
            source=db,
        )
        for message_id in ("variant-message-1", "variant-message-2"):
            assert (
                producer.reconcile_chat_message_intent(
                    server_profile_id="server-a",
                    authenticated_principal_id="user-a",
                    workspace_scope="workspace-1",
                    message_id=message_id,
                    message_version=1,
                    payload_hash=_current_message_payload_hash(db, message_id),
                )["status"]
                == "enqueued"
            )

        entries = repo.list_sync_v2_outbox_entries(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            dataset_id="dataset-1",
        )
        assert [entry["envelope"]["stable_key"] for entry in entries] == [
            f"{conversation_id}:variant-message-1",
            f"{conversation_id}:variant-message-2",
        ]
    finally:
        db.close_connection()


def _source_message(tmp_path) -> tuple[CharactersRAGDB, str, str]:
    db = CharactersRAGDB(tmp_path / "source.db", client_id="source-client")
    conversation_id = db.add_conversation({"title": "Continuation"})
    message_id = "assistant-continuation-1"
    private_json = _provider_continuation_json()
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
    return (
        db,
        message_id,
        canonical_payload_hash(
            {
                "content": "visible answer",
                "provider_continuation_json": canonical_private,
                "role": "assistant",
            }
        ),
    )


def _configured_repo(path) -> SyncStateRepository:
    repo = SyncStateRepository(path)
    repo.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        profile_mode="local_first",
        device_id="device-1",
        dataset_id="dataset-1",
    )
    return repo


def _current_message_payload_hash(db: CharactersRAGDB, message_id: str) -> str:
    row = db.get_message_by_id(message_id)
    payload = {"content": row["content"], "role": "assistant"}
    if row["provider_continuation_json"] is not None:
        payload["provider_continuation_json"] = row["provider_continuation_json"]
    return canonical_payload_hash(payload)


def _provider_continuation_json() -> str:
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
                    "reasoning_blocks": ["PRIVATE-CONTINUATION-CANARY"],
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
