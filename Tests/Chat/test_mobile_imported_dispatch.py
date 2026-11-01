"""Actual native recovery consumers of committed passive Chat provenance."""

from dataclasses import replace

import pytest

from Tests.Chat.test_console_dispatch_recovery import (
    _acceptance,
    _database,
    _insert,
    _NoReplayGateway,
    _restored_store,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleDispatchRecoveryActionId,
    ConsoleDispatchRecoveryKind,
)
from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget
from tldw_chatbook.DB.mobile_consolidation_receipts import (
    _SPECS,
    _encode_owner_receipt,
    _row_digest,
)


def _install_chat_history(db, table_ids):
    """Independent native consumer fixture, not an actual producer E2E claim."""
    children = []
    with db.transaction(immediate=True) as cursor:
        for table, identities in table_ids.items():
            spec = _SPECS[table]
            assert len(spec.identity_columns) == 1
            key = spec.identity_columns[0]
            for identity in identities:
                selector = "message_id" if table == "fleet_progress_messages" else key
                columns = ",".join(
                    f'CASE typeof("{column}") WHEN \'text\' THEN CAST("{column}" AS TEXT) ELSE "{column}" END AS "{column}"'
                    for column in spec.columns
                )
                row = cursor.execute(
                    f'SELECT {columns} FROM "{table}" WHERE "{selector}"=?', (identity,)
                ).fetchone()
                assert row is not None
                if table in {"messages", "fleet_progress_messages"}:
                    children.append(
                        {
                            "version": 1,
                            "kind": "identity",
                            "data": {
                                "domain": "message"
                                if table == "messages"
                                else "fleet_message",
                                "original": "original-" + identity,
                                "mapped": identity,
                            },
                        }
                    )
                children.append(
                    {
                        "version": 1,
                        "kind": "row",
                        "data": {
                            "table": table,
                            "original": [
                                row[key]
                                if type(row[key]) is int
                                else "original-" + identity
                            ],
                            "mapped": [row[key]],
                            "source_sha256": "d" * 64,
                            "mapped_sha256": _row_digest(
                                table, dict(row), _TraceReadBudget()
                            ),
                        },
                    }
                )
        encoded = _encode_owner_receipt(
            "chat",
            children,
            source_manifest_sha256="a" * 64,
            canonical_manifest_sha256="b" * 64,
            source_runs_version=None,
            command_history_count=None,
            budget=_TraceReadBudget(),
        )
        cursor.executemany(
            "INSERT INTO mobile_consolidation_batches VALUES (?,?,?,?,?)",
            (
                ("import-operation", "c" * 64, key, raw, "2026-10-27T00:00:00Z")
                for key, raw in encoded
            ),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["manual", "queued"])
async def test_imported_checkpoint_offers_only_actual_discard_then_fresh_acceptance(
    tmp_path, origin
):
    db, conversation, repository = _database(tmp_path / "chat.db")
    try:
        _insert(
            db,
            repository,
            _acceptance(
                conversation,
                origin=origin,
                queue_entry_id="historical-queue" if origin == "queued" else None,
            ),
        )
        _install_chat_history(
            db,
            {
                "messages": ("user-1", "assistant-1"),
                "console_dispatch_checkpoints": ("assistant-1",),
            },
        )
        store, session = _restored_store(db, conversation)
        recovery = store.dispatch_recovery_for_session(session)
        assert recovery.kind is ConsoleDispatchRecoveryKind.QUARANTINED
        assert recovery.error_code == "imported_history"
        assert recovery.checkpoint is not None
        assert [(action.action_id, action.enabled) for action in recovery.actions] == [
            (ConsoleDispatchRecoveryActionId.DISCARD, True)
        ]
        assert recovery.queue_entry_id is recovery.preparation_id is None
        assert not store.dispatch_recovery_needs_queue_hydration(session)
        gateway = _NoReplayGateway(db)
        controller = ConsoleChatController(
            store=store, provider_gateway=gateway, agent_runtime_enabled=False
        )
        assert not controller._hydrate_dispatch_recovery_queue(session, force=True)
        assert not (await controller.retry_dispatch_recovery(session)).accepted
        assert (await controller.discard_dispatch_recovery(session)).accepted
        assert gateway.resolve_calls == 0 and gateway.provider_states == []
        assert db.get_message_by_id("user-1")["content"] == "hello"
        assert (
            db.get_message_by_id("assistant-1")["assistant_generation_state"]
            == "discarded"
        )
        assert (
            db.get_connection()
            .execute("SELECT count(*) FROM console_dispatch_checkpoints")
            .fetchone()[0]
            == 0
        )
        fresh = replace(
            _acceptance(conversation),
            user_message_id="fresh-user",
            assistant_message_id="fresh-assistant",
            preparation_id="fresh-preparation",
        )
        assert _insert(db, repository, fresh).assistant_message_id == "fresh-assistant"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_imported_discard_real_cas_conflict_keeps_only_discard(
    tmp_path, monkeypatch
):
    from Tests.Chat.test_console_dispatch_recovery import ConsoleDispatchRepository

    db, conversation, repository = _database(tmp_path / "cas.db")
    try:
        _insert(db, repository, _acceptance(conversation))
        _install_chat_history(
            db,
            {
                "messages": ("user-1", "assistant-1"),
                "console_dispatch_checkpoints": ("assistant-1",),
            },
        )
        store, session = _restored_store(db, conversation)
        controller = ConsoleChatController(
            store=store,
            provider_gateway=_NoReplayGateway(db),
            agent_runtime_enabled=False,
        )
        settle = ConsoleDispatchRepository._settle_with_assistant_uncoordinated

        def stale_revision(owner, settlement):
            return settle(
                owner,
                replace(
                    settlement,
                    expected_checkpoint_revision=settlement.expected_checkpoint_revision
                    + 1,
                ),
            )

        with monkeypatch.context() as fault:
            fault.setattr(
                ConsoleDispatchRepository,
                "_settle_with_assistant_uncoordinated",
                stale_revision,
            )
            assert not (await controller.discard_dispatch_recovery(session)).accepted
        assert (
            db.get_message_by_id("assistant-1")["assistant_generation_state"]
            == "accepted"
        )
        recovery = store.dispatch_recovery_for_session(session)
        assert [(action.action_id, action.enabled) for action in recovery.actions] == [
            (ConsoleDispatchRecoveryActionId.DISCARD, True)
        ]
        assert (await controller.discard_dispatch_recovery(session)).accepted
    finally:
        db.close()


def test_incomplete_imported_checkpoint_receipt_offers_no_action(tmp_path):
    db, conversation, repository = _database(tmp_path / "invalid.db")
    try:
        _insert(db, repository, _acceptance(conversation))
        _install_chat_history(
            db,
            {
                "messages": ("user-1", "assistant-1"),
                "console_dispatch_checkpoints": ("assistant-1",),
            },
        )
        with db.transaction() as cursor:
            cursor.execute(
                "DELETE FROM mobile_consolidation_batches WHERE batch_key='chat.entry:0'"
            )
        recovery = repository.reconcile_for_session(conversation)
        assert recovery.kind is ConsoleDispatchRecoveryKind.QUARANTINED
        assert recovery.actions == () and recovery.checkpoint is None
        assert (
            db.get_message_by_id("assistant-1")["assistant_generation_state"]
            == "accepted"
        )
    finally:
        db.close()


def test_imported_native_continuation_remains_visible_without_current_recovery(
    tmp_path,
):
    from Tests.Chat.test_console_dispatch_continuation_handoff import (
        _continuation,
        _deepseek_acceptance,
    )
    from Tests.Chat.test_console_dispatch_recovery import _start
    from tldw_chatbook.Chat.console_dispatch_checkpoint import (
        ConsoleContinuationHandoff,
        ConsoleDispatchResultStatus,
    )
    from tldw_chatbook.Chat.provider_continuation import dump_provider_continuation_json

    db, conversation, repository = _database(tmp_path / "continuation.db")
    try:
        started = _start(
            repository, _insert(db, repository, _deepseek_acceptance(conversation))
        )
        raw = dump_provider_continuation_json(_continuation())
        result = repository.handoff_to_provider_continuation(
            ConsoleContinuationHandoff(
                assistant_message_id=started.assistant_message_id,
                expected_checkpoint_revision=started.checkpoint_revision,
                expected_user_message_version=started.user_message_version,
                expected_assistant_message_version=started.assistant_message_version,
                provider_continuation_json=raw,
            )
        )
        assert result.status is ConsoleDispatchResultStatus.COMMITTED
        _install_chat_history(db, {"messages": ("user-1", "assistant-1")})
        before = db.get_message_by_id("assistant-1")
        recovery = repository.reconcile_for_session(conversation)
        assert recovery.kind is ConsoleDispatchRecoveryKind.QUARANTINED
        assert recovery.error_code == "imported_history"
        assert recovery.assistant_message_id == "assistant-1" and recovery.actions == ()
        assert recovery.checkpoint is None
        assert (
            "read-only history" in recovery.visible_copy
            and "invalid" not in recovery.visible_copy
        )
        assert (
            repository.provider_continuation_owner_snapshot(
                conversation_id=conversation, assistant_message_id="assistant-1"
            )
            is None
        )
        assert (
            repository.normalize_provider_continuation_owner(
                conversation_id=conversation,
                assistant_message_id="assistant-1",
                expected_message_version=before["version"],
                expected_state=before["assistant_generation_state"],
                provider_continuation_json=raw,
            ).status
            is ConsoleDispatchResultStatus.CONFLICT
        )
        assert db.get_message_by_id("assistant-1") == before
    finally:
        db.close()


@pytest.mark.parametrize("kind", ["hook", "machine"])
@pytest.mark.parametrize("imported", [False, True])
def test_current_followup_acceptance_cannot_derive_from_imported_parent(
    tmp_path, kind, imported
):
    from tldw_chatbook.Agents.hooks_v2.continuations import ContinuationReceipt
    from tldw_chatbook.Chat.console_dispatch_repository import (
        ConsoleDispatchCheckpointValidationError,
    )
    from tldw_chatbook.Chat.response_rules.corrections import MachineFollowupReceipt

    db, conversation, repository = _database(tmp_path / "followup.db")
    try:
        parent = db.add_message(
            {
                "conversation_id": conversation,
                "sender": "assistant",
                "role": "assistant",
                "content": "Retained parent",
            }
        )
        if imported:
            _install_chat_history(db, {"messages": (parent,)})
        receipt = ContinuationReceipt("parent-turn", "stop-event", parent, "chain", 1)
        machine = MachineFollowupReceipt(
            "operation",
            "parent-turn",
            "settlement",
            parent,
            "chain",
            1,
            1,
            ("native",),
            db.get_message_by_id(parent)["version"],
        )
        acceptance = replace(
            _acceptance(conversation, origin="queued", queue_entry_id="fresh-queue"),
            parent_message_id=parent,
            continuation_receipt=receipt if kind == "hook" else None,
            machine_followup_receipt=machine if kind == "machine" else None,
        )
        before = (
            db.get_connection().execute("SELECT count(*) FROM messages").fetchone()[0]
        )
        if imported:
            with pytest.raises(
                ConsoleDispatchCheckpointValidationError,
                match="Imported follow-up parent",
            ):
                _insert(db, repository, acceptance)
            assert (
                db.get_connection()
                .execute("SELECT count(*) FROM messages")
                .fetchone()[0]
                == before
            )
            assert (
                db.get_connection()
                .execute("SELECT count(*) FROM console_dispatch_checkpoints")
                .fetchone()[0]
                == 0
            )
        else:
            assert (
                _insert(db, repository, acceptance).assistant_message_id
                == "assistant-1"
            )
    finally:
        db.close()


def test_native_hook_user_metadata_keeps_exact_audit_shape(tmp_path):
    import json
    from dataclasses import replace

    from tldw_chatbook.Agents.hooks_v2.continuations import ContinuationReceipt
    from tldw_chatbook.Backup_Recovery.mobile_content_settings import _message_metadata

    db, conversation, repository = _database(tmp_path / "hook-metadata.db")
    try:
        parent = db.add_message(
            {
                "conversation_id": conversation,
                "sender": "assistant",
                "role": "assistant",
                "content": "Native parent",
            }
        )
        receipt = ContinuationReceipt("parent-turn", "stop-event", parent, "chain", 1)
        acceptance = replace(
            _acceptance(conversation, origin="queued", queue_entry_id="queue"),
            parent_message_id=parent,
            continuation_receipt=receipt,
        )
        _insert(db, repository, acceptance)
        raw = db.get_message_by_id(acceptance.user_message_id)["metadata_json"]

        def no_new_authority(*args):
            pytest.fail(
                "historical hook audit metadata minted a new identity/authority"
            )

        assert (
            _message_metadata(
                None, raw, no_new_authority, message_id=acceptance.user_message_id
            )
            == raw
        )
        from tldw_chatbook.Backup_Recovery.mobile_content_workflows import (
            _hook_metadata_join,
        )

        hook = dict(
            db.get_connection()
            .execute("SELECT * FROM console_hook_continuation_receipts")
            .fetchone()
        )
        messages = {
            row["id"]: dict(row)
            for row in db.get_connection().execute("SELECT * FROM messages")
        }
        _hook_metadata_join(hook, messages)
        for bad in (
            hook | {"stop_event_id": "unrelated"},
            hook | {"parent_assistant_message_id": "unrelated"},
        ):
            with pytest.raises(ValueError, match="hook_message_relation"):
                _hook_metadata_join(bad, messages)
        original = json.loads(raw)
        assert original == {
            "origin": "hook",
            "initiator": "hook_continuation",
            "parent_turn_id": receipt.parent_turn_id,
            "stop_event_id": receipt.stop_event_id,
        }
        for changed in (
            original | {"unknown": 1},
            original | {"parent_turn_id": ""},
            original | {"stop_event_id": "x" * 257},
            original | {"initiator": "other"},
            original | {"parent_turn_id": True},
        ):
            with pytest.raises(ValueError):
                _message_metadata(
                    None,
                    json.dumps(changed),
                    no_new_authority,
                    message_id=acceptance.user_message_id,
                )
    finally:
        db.close()
