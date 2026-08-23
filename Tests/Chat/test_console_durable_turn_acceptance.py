"""Task 14: one durable Console turn-acceptance transaction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleDispatchReconstructability,
    ConsoleDurableTurnAcceptance,
    ConsoleEgressClass,
    ConsoleLibraryItemScopeSnapshot,
    ConsoleProviderIntent,
    ConsoleResolvedDestination,
    ConsoleTurnLibraryAuthority,
)
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyDefaults,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.Chat.console_turn_context import (
    ConsoleTurnConfigurationSnapshot,
    ConsoleTurnExecutionContext,
)
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparation,
    ConsoleTurnPreparationState,
    preparation_actions,
)
from tldw_chatbook.Chat.console_transaction_contribution import (
    ConsoleTransactionWriter,
)
from tldw_chatbook.Chat.library_preparation import (
    LibraryPreparationContribution,
    LibraryPreparationEvent,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, TransactionContextManager


@dataclass
class _SequenceThenInsertContribution:
    calls: int = 0

    def durable_acceptance_fingerprint(self) -> dict[str, object]:
        """Return only immutable contribution inputs, excluding test counters."""
        return {"event_kind": "library_preparation", "outcome": "bypassed"}

    def write(
        self,
        *,
        writer: ConsoleTransactionWriter,
        conversation_id: str,
        message_ids: dict[str, str],
    ) -> None:
        self.calls += 1
        sequence = writer.next_trajectory_sequence()
        writer.execute(
            "INSERT INTO message_trajectory_metadata "
            "(message_id, conversation_id, turn_id, seq, event_kind, payload_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                message_ids["user"],
                conversation_id,
                message_ids["user"],
                sequence,
                "library_preparation",
                '{"version":1,"outcome":"bypassed","attempt_id":"attempt-1",'
                '"result_count":0,"source_types":["notes","media","conversations"]}',
            ),
        )


def _authority(
    *,
    source: str = "new_session",
    revision: int | None = None,
) -> ConsoleTurnLibraryAuthority:
    return ConsoleTurnLibraryAuthority(
        policy=ConsoleLibraryPolicySnapshot(
            auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC,
            assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
            policy_revision=revision,
            source=source,  # type: ignore[arg-type]
        ),
        direct_library_tools=True,
        source_types=("notes", "media", "conversations"),
        scope_snapshot=ConsoleLibraryItemScopeSnapshot((), (), True),
        provider_intent=ConsoleProviderIntent(
            "llama_cpp", "test-model", "http://127.0.0.1:9099"
        ),
        attempt_id="attempt-1",
    )


def _context(session_id: str, authority: ConsoleTurnLibraryAuthority):
    return ConsoleTurnExecutionContext(
        configuration=ConsoleTurnConfigurationSnapshot.capture(
            session_id=session_id,
            provider_selection=ConsoleProviderSelection(
                provider="llama_cpp", explicit_model="test-model"
            ),
        ),
        library_authority=authority,
        resolved_destination=ConsoleResolvedDestination(
            provider="llama_cpp",
            model="test-model",
            endpoint_identity="http://127.0.0.1:9099",
            egress_class=ConsoleEgressClass.ON_DEVICE,
        ),
    )


def _ready_store(
    tmp_path: Path,
    *,
    existing: bool = False,
    attachments: bool = True,
    contribution: object | None = None,
) -> tuple[
    CharactersRAGDB,
    ChatPersistenceService,
    ConsoleChatStore,
    ConsoleTurnPreparation,
    ConsoleDurableTurnAcceptance,
]:
    db = CharactersRAGDB(tmp_path / "acceptance.sqlite", client_id="task14-test")
    service = ChatPersistenceService(db)
    store = ConsoleChatStore(
        persistence=service,
        library_policy_defaults=ConsoleLibraryPolicyDefaults(
            ConsoleAutoRetrieve.AUTOMATIC,
            ConsoleAssistantLibraryAccess.ALLOWED,
        ),
    )
    session = store.create_session(session_id="session-1", title="Chat 1")
    if existing:
        conversation_id = service.create_conversation(
            conversation_id="conversation-existing",
            conversation_title="Existing",
        )
        session.persisted_conversation_id = conversation_id
        policy = service.console_library_policy_repository.insert(
            conversation_id,
            store.session_library_policy_candidate(session.id),
        )
        assert policy.snapshot.policy_revision == 1
        authority = _authority(source="durable", revision=1)
    else:
        authority = _authority()
        conversation_id = ""
    echo = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="exact captured draft",
        persist=False,
    )
    preparation = ConsoleTurnPreparation(
        preparation_id="preparation-1",
        attempt_id="attempt-1",
        session_id=session.id,
        origin="manual",
        queue_entry_id=None,
        executed_draft="exact captured draft",
        execution_context=_context(session.id, authority),
        transient_user_message_id=echo.id,
        attachment_ids=("attachment-1",) if attachments else (),
        evidence_ids=(),
        prefill_id=None,
        queue_generation=None,
        pre_send_title="Chat 1",
        pre_send_conversation_id=(conversation_id if existing else None),
        state=ConsoleTurnPreparationState.COMMITTING,
        pause_kind=None,
        one_shot_bypass=False,
        ephemeral=False,
    )
    assert store.begin_preparation(preparation) is preparation
    if not existing:
        identity = store.stage_durable_turn_identity(
            session.id,
            "preparation-1",
            title="Atomic hello",
        )
        conversation_id = identity.conversation_id
    if contribution is None:
        contribution = LibraryPreparationContribution(
            LibraryPreparationEvent(
                version=1,
                outcome="bypassed",
                attempt_id="attempt-1",
                result_count=0,
                source_types=("notes", "media", "conversations"),
            )
        )
    attachment_rows = (
        (
            {
                "position": 0,
                "data": b"first-image",
                "mime_type": "image/png",
                "display_name": "first.png",
            },
            {
                "position": 1,
                "data": b"second-image",
                "mime_type": "image/png",
                "display_name": "second.png",
            },
        )
        if attachments
        else ()
    )
    acceptance = ConsoleDurableTurnAcceptance(
        conversation_id=conversation_id,
        user_message_id=echo.id,
        assistant_message_id="assistant-1",
        parent_message_id=None,
        user_content="exact captured draft",
        attachments=attachment_rows,
        preparation_id="preparation-1",
        attempt_id="attempt-1",
        origin="manual",
        queue_entry_id=None,
        frozen_authority=authority,
        resolved_destination=_context(session.id, authority).resolved_destination,
        reconstructability=ConsoleDispatchReconstructability(
            attachments_reconstructable=True,
            evidence_reconstructable=True,
            prefill_reconstructable=True,
            opaque_reference="opaque:preparation-1",
        ),
        contributions=(contribution,),  # type: ignore[arg-type]
    )
    return db, service, store, preparation, acceptance


def _database_snapshot(db: CharactersRAGDB) -> dict[str, tuple[tuple[Any, ...], ...]]:
    connection = db.get_connection()
    tables = (
        "conversations",
        "console_conversation_library_policy",
        "messages",
        "message_attachments",
        "console_dispatch_checkpoints",
        "message_trajectory_metadata",
        "sync_log",
    )
    result: dict[str, tuple[tuple[Any, ...], ...]] = {}
    for table in tables:
        columns = tuple(
            row[1]
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        )
        rows = connection.execute(f"SELECT * FROM {table}").fetchall()
        result[table] = tuple(
            sorted((tuple(row[column] for column in columns) for row in rows), key=repr)
        )
    return result


def _memory_snapshot(
    store: ConsoleChatStore, preparation_id: str
) -> tuple[object, ...]:
    session = store.sessions()[0]
    return (
        session.persisted_conversation_id,
        session.title,
        session.draft,
        tuple(session.pending_attachments),
        session.one_shot_prefill,
        tuple(
            (
                row.id,
                row.role,
                row.content,
                row.persisted_message_id,
                row.parent_message_id,
            )
            for row in store.messages_for_session(session.id)
        ),
        store.preparation_by_id(preparation_id),
    )


def _install_failure(
    db: CharactersRAGDB,
    service: ChatPersistenceService,
    point: str,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[], None]:
    connection = db.get_connection()
    trigger = f"task14_fail_{point}"
    when = ""
    table = point
    if point in {"user", "assistant"}:
        table = "messages"
        when = f" WHEN NEW.role = '{point}'"
    elif point == "attachment":
        table = "message_attachments"
    elif point == "checkpoint":
        table = "console_dispatch_checkpoints"
    elif point == "policy":
        table = "console_conversation_library_policy"
    elif point == "conversation":
        table = "conversations"
    elif point == "contribution_insert":
        table = "message_trajectory_metadata"
        when = " WHEN NEW.event_kind = 'library_preparation'"
    if point == "commit":
        connection.execute(
            "CREATE TABLE task14_commit_fault ("
            "parent_id TEXT REFERENCES conversations(id) "
            "DEFERRABLE INITIALLY DEFERRED)"
        )
        connection.commit()
        original_exit = TransactionContextManager.__exit__

        def fail_actual_sqlite_commit(self, exc_type, exc_val, exc_tb):
            if exc_type is None and self.is_outermost_transaction:
                assert self.conn is not None
                self.conn.execute(
                    "INSERT INTO task14_commit_fault(parent_id) VALUES (?)",
                    ("task14-missing-parent",),
                )
            return original_exit(self, exc_type, exc_val, exc_tb)

        monkeypatch.setattr(
            TransactionContextManager, "__exit__", fail_actual_sqlite_commit
        )

        def cleanup_commit_fault() -> None:
            monkeypatch.setattr(TransactionContextManager, "__exit__", original_exit)
            connection.execute("DROP TABLE task14_commit_fault")
            connection.commit()

        return cleanup_commit_fault
    connection.execute(
        f"CREATE TEMP TRIGGER {trigger} BEFORE INSERT ON {table}{when} "
        "BEGIN SELECT RAISE(ABORT, 'task14 injected failure'); END"
    )
    return lambda: connection.execute(f"DROP TRIGGER {trigger}")


@pytest.mark.parametrize(
    "failure_point",
    (
        "conversation",
        "policy",
        "user",
        "attachment",
        "assistant",
        "checkpoint",
        "contribution_insert",
        "commit",
    ),
)
def test_every_new_conversation_write_or_commit_failure_rolls_back_exactly_and_retries_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    db, service, store, _preparation, acceptance = _ready_store(tmp_path)
    before_db = _database_snapshot(db)
    before_memory = _memory_snapshot(store, acceptance.preparation_id)
    cleanup = _install_failure(db, service, failure_point, monkeypatch)

    with pytest.raises(
        Exception,
        match="injected|task14|could not be committed|Commit failed|FOREIGN KEY",
    ):
        store.commit_durable_turn(acceptance)

    assert _database_snapshot(db) == before_db
    after_memory = _memory_snapshot(store, acceptance.preparation_id)
    assert after_memory[:-1] == before_memory[:-1]
    paused = store.preparation_by_id(acceptance.preparation_id)
    assert paused is not None
    assert paused.state is ConsoleTurnPreparationState.PAUSED
    assert paused.pause_kind is ConsolePreparationPauseKind.PERSISTENCE
    assert preparation_actions(paused) == ("retry", "cancel")
    cleanup()

    committed = store.commit_durable_turn(acceptance)

    assert committed.identity.conversation_id == acceptance.conversation_id
    assert committed.identity.title == "Atomic hello"
    assert committed.user_message_version == 1
    assert committed.assistant_message_version == 1
    assert committed.checkpoint.checkpoint_revision == 1
    assert db.get_connection().in_transaction is False
    assert store.sessions()[0].persisted_conversation_id is None
    assert store.sessions()[0].title == "Chat 1"
    assert (
        len(db.get_messages_for_conversation(acceptance.conversation_id, limit=20)) == 2
    )
    assert (
        db.get_connection()
        .execute(
            "SELECT COUNT(*) FROM console_dispatch_checkpoints WHERE preparation_id = ?",
            (acceptance.preparation_id,),
        )
        .fetchone()[0]
        == 1
    )
    assert (
        db.get_connection()
        .execute(
            "SELECT COUNT(*) FROM message_trajectory_metadata "
            "WHERE conversation_id = ? AND event_kind = 'library_preparation'",
            (acceptance.conversation_id,),
        )
        .fetchone()[0]
        == 1
    )


def test_sequence_allocation_failure_rolls_back_existing_conversation_byte_exact(
    tmp_path: Path,
) -> None:
    contribution = _SequenceThenInsertContribution()
    db, _service, store, _preparation, acceptance = _ready_store(
        tmp_path,
        existing=True,
        attachments=False,
        contribution=contribution,
    )
    existing_message_id = db.add_message(
        {
            "id": "existing-user",
            "conversation_id": acceptance.conversation_id,
            "sender": "user",
            "content": "existing",
        }
    )
    db.get_connection().execute(
        "INSERT INTO message_trajectory_metadata "
        "(message_id, conversation_id, turn_id, seq, event_kind) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            existing_message_id,
            acceptance.conversation_id,
            existing_message_id,
            9223372036854775807,
            "existing",
        ),
    )
    db.get_connection().commit()
    before = _database_snapshot(db)

    with pytest.raises(Exception):
        store.commit_durable_turn(acceptance)

    assert _database_snapshot(db) == before
    assert contribution.calls == 1
    paused = store.preparation_by_id(acceptance.preparation_id)
    assert paused is not None
    assert paused.pause_kind is ConsolePreparationPauseKind.PERSISTENCE


def test_existing_conversation_policy_mismatch_fails_closed_without_version_drift(
    tmp_path: Path,
) -> None:
    db, _service, store, _preparation, acceptance = _ready_store(
        tmp_path,
        existing=True,
        attachments=False,
    )
    db.get_connection().execute(
        "UPDATE console_conversation_library_policy "
        "SET policy_revision = 2 WHERE conversation_id = ?",
        (acceptance.conversation_id,),
    )
    db.get_connection().commit()
    before = _database_snapshot(db)

    with pytest.raises(Exception):
        store.commit_durable_turn(acceptance)

    assert _database_snapshot(db) == before
    assert store.preparation_by_id(acceptance.preparation_id).pause_kind is (
        ConsolePreparationPauseKind.PERSISTENCE
    )


def test_success_persists_exact_attachment_state_hash_sync_intent_and_private_checkpoint(
    tmp_path: Path,
) -> None:
    db, _service, store, _preparation, acceptance = _ready_store(tmp_path)

    commit = store.commit_durable_turn(acceptance)

    user = db.get_message_by_id(commit.user_message_id)
    assistant = db.get_message_by_id(commit.assistant_message_id)
    assert user is not None and assistant is not None
    assert user["content"] == "exact captured draft"
    assert user["image_data"] == b"first-image"
    assert user["version"] == commit.user_message_version == 1
    assert assistant["content"] == ""
    assert assistant["assistant_generation_state"] == "accepted"
    assert assistant["version"] == commit.assistant_message_version == 1
    assert assistant["deleted"] == user["deleted"] == 0
    attachments = db.get_attachments_for_messages([commit.user_message_id])
    assert [
        (row["position"], row["data"]) for row in attachments[commit.user_message_id]
    ] == [(1, b"second-image")]
    sync_payloads = [
        json.loads(row[0])
        for row in db.get_connection().execute(
            "SELECT payload FROM sync_log WHERE entity = 'messages' ORDER BY rowid"
        )
    ]
    assert [
        payload["assistant_generation_state"] for payload in sync_payloads[-2:]
    ] == [
        None,
        "accepted",
    ]
    checkpoint = (
        db.get_connection()
        .execute(
            "SELECT frozen_authority_json, resolved_destination_json, "
            "reconstructability_json FROM console_dispatch_checkpoints "
            "WHERE preparation_id = ?",
            (acceptance.preparation_id,),
        )
        .fetchone()
    )
    serialized = " ".join(str(value) for value in checkpoint)
    for forbidden in (
        "exact captured draft",
        "first-image",
        "second-image",
        "api_key",
        "source body",
        "provider_request",
    ):
        assert forbidden not in serialized
