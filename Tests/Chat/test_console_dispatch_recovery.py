from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import tldw_chatbook.Chat.console_chat_models as recovery_models
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleAssistantSettlement,
    ConsoleDispatchCheckpoint,
    ConsoleDispatchCheckpointState,
    ConsoleDispatchReconstructability,
    ConsoleDispatchTransition,
    ConsoleDurableTurnAcceptance,
    ConsoleEgressClass,
    ConsoleLibraryItemScopeSnapshot,
    ConsoleProviderIntent,
    ConsoleResolvedDestination,
    ConsoleTurnLibraryAuthority,
)
from tldw_chatbook.Chat.console_dispatch_repository import ConsoleDispatchRepository
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.Chat.provider_continuation import (
    dump_provider_continuation_json,
    parse_provider_continuation_json,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


UNRECONSTRUCTABLE_REASON = (
    "Retry response is unavailable because one-shot prefill or transient evidence "
    "cannot be reconstructed exactly."
)
DISCARD_COPY = "Response discarded."


def _recovery_symbols():
    required = (
        "ConsoleDispatchRecoveryActionId",
        "ConsoleDispatchRecoveryKind",
        "ConsoleDispatchRecoveryState",
    )
    missing = [name for name in required if not hasattr(recovery_models, name)]
    assert not missing, f"dispatch recovery model is missing: {', '.join(missing)}"
    return tuple(getattr(recovery_models, name) for name in required)


def _authority(*, attempt_id: str = "attempt-1") -> ConsoleTurnLibraryAuthority:
    return ConsoleTurnLibraryAuthority(
        policy=ConsoleLibraryPolicySnapshot(
            auto_retrieve=ConsoleAutoRetrieve.NEVER,
            assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
            policy_revision=1,
            source="durable",
        ),
        direct_library_tools=False,
        source_types=("notes", "media", "conversations"),
        scope_snapshot=ConsoleLibraryItemScopeSnapshot(
            note_ids=(),
            media_ids=(),
            conversations_allowed=True,
        ),
        provider_intent=ConsoleProviderIntent(
            provider="llama_cpp",
            model="test-model",
            endpoint="http://127.0.0.1:9099",
        ),
        attempt_id=attempt_id,
    )


def _destination() -> ConsoleResolvedDestination:
    return ConsoleResolvedDestination(
        provider="llama_cpp",
        model="test-model",
        endpoint_identity="http://127.0.0.1:9099",
        egress_class=ConsoleEgressClass.ON_DEVICE,
    )


def _reconstructability() -> ConsoleDispatchReconstructability:
    return ConsoleDispatchReconstructability(
        attachments_reconstructable=True,
        evidence_reconstructable=True,
        prefill_reconstructable=True,
        opaque_reference="opaque:turn-1",
    )


def _acceptance(
    conversation_id: str,
    *,
    state_truth: ConsoleDispatchReconstructability | None = None,
    origin: str = "manual",
    queue_entry_id: str | None = None,
) -> ConsoleDurableTurnAcceptance:
    return ConsoleDurableTurnAcceptance(
        conversation_id=conversation_id,
        user_message_id="user-1",
        assistant_message_id="assistant-1",
        parent_message_id=None,
        user_content="hello",
        attachments=(),
        preparation_id="preparation-1",
        attempt_id="attempt-1",
        origin=origin,
        queue_entry_id=queue_entry_id,
        frozen_authority=_authority(),
        resolved_destination=_destination(),
        reconstructability=state_truth or _reconstructability(),
        contributions=(),
    )


def _database(path: Path) -> tuple[CharactersRAGDB, str, ConsoleDispatchRepository]:
    db = CharactersRAGDB(path, client_id="dispatch-recovery-test")
    conversation_id = db.add_conversation({"title": "recovery"})
    assert conversation_id is not None
    with db.transaction(immediate=True) as cursor:
        cursor.execute(
            "INSERT INTO console_conversation_library_policy "
            "(conversation_id, auto_retrieve_on_send, assistant_library_access, "
            "policy_revision, updated_at) "
            "VALUES (?, 0, 0, 1, CURRENT_TIMESTAMP)",
            (conversation_id,),
        )
    return db, conversation_id, ConsoleDispatchRepository(db)


def _insert(
    db: CharactersRAGDB,
    repository: ConsoleDispatchRepository,
    acceptance: ConsoleDurableTurnAcceptance,
) -> ConsoleDispatchCheckpoint:
    with db.transaction(immediate=True) as cursor:
        return repository.insert_with_messages(cursor, acceptance)


def _reconcile(repository: ConsoleDispatchRepository, conversation_id: str):
    reconcile = getattr(repository, "reconcile_for_session", None)
    assert callable(reconcile), "dispatch repository has no deterministic loader"
    return reconcile(conversation_id)


def _start(
    repository: ConsoleDispatchRepository,
    checkpoint: ConsoleDispatchCheckpoint,
) -> ConsoleDispatchCheckpoint:
    result = repository.cas_state(
        ConsoleDispatchTransition(
            assistant_message_id=checkpoint.assistant_message_id,
            expected_state=ConsoleDispatchCheckpointState.ACCEPTED,
            expected_checkpoint_revision=checkpoint.checkpoint_revision,
            expected_user_message_version=checkpoint.user_message_version,
            expected_assistant_message_version=checkpoint.assistant_message_version,
            new_state=ConsoleDispatchCheckpointState.DISPATCH_STARTED,
            new_attempt_id="attempt-2",
        )
    )
    assert result.checkpoint is not None
    return result.checkpoint


def _active_continuation_json() -> str:
    raw = json.dumps(
        {
            "schema_version": 1,
            "checkpoint_revision": 1,
            "provider": "deepseek",
            "protocol": "responses",
            "model": "deepseek-v4-flash",
            "api_base_url": "https://api.deepseek.com/v1",
            "state": "active",
            "rounds": [
                {
                    "assistant_content": "",
                    "reasoning_blocks": [],
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
    canonical = dump_provider_continuation_json(parse_provider_continuation_json(raw))
    assert canonical is not None
    return canonical


def _action_projection(recovery) -> tuple[tuple[str, str, bool, str], ...]:
    return tuple(
        (
            action.action_id.value,
            action.label,
            action.enabled,
            action.disabled_reason,
        )
        for action in recovery.actions
    )


def test_loader_hydrates_valid_accepted_checkpoint_with_exact_actions(
    tmp_path: Path,
) -> None:
    _action_id, kind, _state = _recovery_symbols()
    db, conversation_id, repository = _database(tmp_path / "accepted.sqlite")
    checkpoint = _insert(db, repository, _acceptance(conversation_id))

    recovery = _reconcile(repository, conversation_id)

    assert recovery.kind is kind.ACCEPTED
    assert recovery.checkpoint == checkpoint
    assert recovery.visible_copy == "Response accepted; waiting for dispatch."
    assert _action_projection(recovery) == (
        ("retry_response", "Retry response", True, ""),
        ("discard", "Discard", True, ""),
    )


def test_loader_hydrates_dispatch_started_without_auto_replay(
    tmp_path: Path,
) -> None:
    _action_id, kind, _state = _recovery_symbols()
    db, conversation_id, repository = _database(tmp_path / "started.sqlite")
    started = _start(repository, _insert(db, repository, _acceptance(conversation_id)))

    recovery = _reconcile(repository, conversation_id)

    assert recovery.kind is kind.DISPATCH_STARTED
    assert recovery.checkpoint == started
    assert recovery.visible_copy == (
        "Response delivery status is unknown on the source device."
    )
    assert recovery.warning == (
        "Retry anyway may send a duplicate request because delivery status is unknown."
    )
    assert _action_projection(recovery) == (
        ("retry_anyway", "Retry anyway", True, ""),
        ("discard", "Discard", True, ""),
    )


@pytest.mark.parametrize("missing", ["prefill", "evidence"])
def test_loader_disables_retry_when_transient_input_is_not_reconstructable(
    tmp_path: Path,
    missing: str,
) -> None:
    _recovery_symbols()
    truth = replace(
        _reconstructability(),
        prefill_reconstructable=missing != "prefill",
        evidence_reconstructable=missing != "evidence",
    )
    db, conversation_id, repository = _database(
        tmp_path / f"unreconstructable-{missing}.sqlite"
    )
    _insert(db, repository, _acceptance(conversation_id, state_truth=truth))

    recovery = _reconcile(repository, conversation_id)

    retry, discard = recovery.actions
    assert retry.enabled is False
    assert retry.disabled_reason == UNRECONSTRUCTABLE_REASON
    assert discard.enabled is True


def test_valid_continuation_without_checkpoint_is_authoritative_but_task15_inert(
    tmp_path: Path,
) -> None:
    _action_id, kind, _state = _recovery_symbols()
    db, conversation_id, repository = _database(tmp_path / "continuation.sqlite")
    inserted = _insert(db, repository, _acceptance(conversation_id))
    connection = db.get_connection()
    connection.execute(
        "DELETE FROM console_dispatch_checkpoints WHERE assistant_message_id = ?",
        (inserted.assistant_message_id,),
    )
    connection.execute(
        "UPDATE messages SET provider_continuation_json = ?, "
        "assistant_generation_state = NULL WHERE id = ?",
        (_active_continuation_json(), inserted.assistant_message_id),
    )
    connection.commit()

    recovery = _reconcile(repository, conversation_id)

    assert recovery.kind is kind.CONTINUATION
    assert recovery.actions == ()
    assert recovery.checkpoint is None


def test_continuation_wins_both_owner_race_and_checkpoint_cleanup_is_atomic(
    tmp_path: Path,
) -> None:
    _action_id, kind, _state = _recovery_symbols()
    db, conversation_id, repository = _database(tmp_path / "both.sqlite")
    inserted = _insert(db, repository, _acceptance(conversation_id))
    connection = db.get_connection()
    connection.execute(
        "UPDATE messages SET provider_continuation_json = ? WHERE id = ?",
        (_active_continuation_json(), inserted.assistant_message_id),
    )
    connection.commit()

    recovery = _reconcile(repository, conversation_id)

    assert recovery.kind is kind.CONTINUATION
    assert recovery.actions == ()
    row = db.get_message_by_id(inserted.assistant_message_id)
    assert row is not None
    assert row["assistant_generation_state"] == "continuation_active"
    assert row["version"] == 2
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM console_dispatch_checkpoints"
        ).fetchone()[0]
        == 0
    )


def test_both_owner_cleanup_failure_preserves_both_owners(tmp_path: Path) -> None:
    _recovery_symbols()
    db, conversation_id, repository = _database(tmp_path / "both-rollback.sqlite")
    inserted = _insert(db, repository, _acceptance(conversation_id))
    connection = db.get_connection()
    continuation = _active_continuation_json()
    connection.execute(
        "UPDATE messages SET provider_continuation_json = ? WHERE id = ?",
        (continuation, inserted.assistant_message_id),
    )
    connection.execute(
        "CREATE TRIGGER fail_recovery_cleanup BEFORE DELETE ON "
        "console_dispatch_checkpoints BEGIN SELECT RAISE(ABORT, 'fail'); END"
    )
    connection.commit()

    recovery = _reconcile(repository, conversation_id)

    assert recovery.error_code == "checkpoint_reconcile_error"
    row = db.get_message_by_id(inserted.assistant_message_id)
    assert row is not None
    assert (row["assistant_generation_state"], row["version"]) == ("accepted", 1)
    assert row["provider_continuation_json"] == continuation
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM console_dispatch_checkpoints"
        ).fetchone()[0]
        == 1
    )


def test_terminal_assistant_wins_stale_checkpoint_without_retry_actions(
    tmp_path: Path,
) -> None:
    _recovery_symbols()
    db, conversation_id, repository = _database(tmp_path / "terminal.sqlite")
    inserted = _insert(db, repository, _acceptance(conversation_id))
    connection = db.get_connection()
    connection.execute(
        "UPDATE messages SET content = 'finished', assistant_generation_state = "
        "'complete', version = 2 WHERE id = ?",
        (inserted.assistant_message_id,),
    )
    connection.commit()

    recovery = _reconcile(repository, conversation_id)

    assert recovery is None
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM console_dispatch_checkpoints"
        ).fetchone()[0]
        == 0
    )


@pytest.mark.parametrize(
    ("corruption", "expected_code"),
    [
        ("wrong_user_role", "invalid_checkpoint_owner"),
        ("cross_conversation", "invalid_checkpoint_owner"),
        ("missing_user", "invalid_checkpoint_owner"),
        ("wrong_assistant_role", "invalid_checkpoint_owner"),
        ("assistant_version", "invalid_checkpoint_owner"),
        ("assistant_state", "invalid_checkpoint_owner"),
    ],
)
def test_invalid_pairs_quarantine_without_deleting_unrelated_rows(
    tmp_path: Path,
    corruption: str,
    expected_code: str,
) -> None:
    _action_id, kind, _state = _recovery_symbols()
    db, conversation_id, repository = _database(tmp_path / f"{corruption}.sqlite")
    inserted = _insert(db, repository, _acceptance(conversation_id))
    connection = db.get_connection()
    other_id = db.add_conversation({"title": "other"})
    assert other_id is not None
    connection.commit()
    connection.execute("PRAGMA foreign_keys = OFF")
    if corruption == "wrong_user_role":
        connection.execute(
            "UPDATE messages SET role = 'assistant' WHERE id = ?",
            (inserted.user_message_id,),
        )
    elif corruption == "cross_conversation":
        connection.execute(
            "UPDATE messages SET conversation_id = ? WHERE id = ?",
            (other_id, inserted.user_message_id),
        )
    elif corruption == "missing_user":
        connection.execute(
            "DELETE FROM messages WHERE id = ?", (inserted.user_message_id,)
        )
    elif corruption == "wrong_assistant_role":
        connection.execute(
            "UPDATE messages SET role = 'user' WHERE id = ?",
            (inserted.assistant_message_id,),
        )
    elif corruption == "assistant_version":
        connection.execute(
            "UPDATE messages SET version = 2 WHERE id = ?",
            (inserted.assistant_message_id,),
        )
    else:
        connection.execute(
            "UPDATE messages SET assistant_generation_state = 'dispatch_started' "
            "WHERE id = ?",
            (inserted.assistant_message_id,),
        )
    connection.commit()
    before_other = connection.execute(
        "SELECT COUNT(*) FROM conversations WHERE id = ?", (other_id,)
    ).fetchone()[0]

    recovery = _reconcile(repository, conversation_id)

    assert recovery.kind is kind.QUARANTINED
    assert recovery.error_code == expected_code
    assert recovery.actions == ()
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM console_dispatch_checkpoints"
        ).fetchone()[0]
        == 1
    )
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM conversations WHERE id = ?", (other_id,)
        ).fetchone()[0]
        == before_other
    )


def test_orphan_continuation_active_is_quarantined(tmp_path: Path) -> None:
    _action_id, kind, _state = _recovery_symbols()
    db, conversation_id, repository = _database(tmp_path / "orphan.sqlite")
    inserted = _insert(db, repository, _acceptance(conversation_id))
    connection = db.get_connection()
    connection.execute("DELETE FROM console_dispatch_checkpoints")
    connection.execute(
        "UPDATE messages SET assistant_generation_state = 'continuation_active' "
        "WHERE id = ?",
        (inserted.assistant_message_id,),
    )
    connection.commit()

    recovery = _reconcile(repository, conversation_id)

    assert recovery.kind is kind.QUARANTINED
    assert recovery.error_code == "orphan_continuation"
    assert recovery.actions == ()


@pytest.mark.parametrize(
    ("assistant_state", "kind_name", "copy"),
    [
        (
            "accepted",
            "REMOTE_ACCEPTED",
            "Response accepted on another device; waiting for dispatch.",
        ),
        (
            "dispatch_started",
            "REMOTE_DISPATCH_STARTED",
            "Response delivery status is unknown on the source device.",
        ),
    ],
)
def test_checkpoint_free_active_states_are_inert_source_device_projections(
    tmp_path: Path,
    assistant_state: str,
    kind_name: str,
    copy: str,
) -> None:
    _action_id, kind, _state = _recovery_symbols()
    db, conversation_id, repository = _database(
        tmp_path / f"remote-{assistant_state}.sqlite"
    )
    inserted = _insert(db, repository, _acceptance(conversation_id))
    connection = db.get_connection()
    connection.execute("DELETE FROM console_dispatch_checkpoints")
    connection.execute(
        "UPDATE messages SET assistant_generation_state = ? WHERE id = ?",
        (assistant_state, inserted.assistant_message_id),
    )
    connection.commit()

    recovery = _reconcile(repository, conversation_id)

    assert recovery.kind is getattr(kind, kind_name)
    assert recovery.visible_copy == copy
    assert recovery.actions == ()
    assert recovery.checkpoint is None


@pytest.mark.parametrize(
    "assistant_state", [None, "complete", "stopped", "failed", "discarded"]
)
def test_checkpoint_free_terminal_or_null_state_is_ordinary_load(
    tmp_path: Path,
    assistant_state: str | None,
) -> None:
    _recovery_symbols()
    db, conversation_id, repository = _database(
        tmp_path / f"ordinary-{assistant_state}.sqlite"
    )
    inserted = _insert(db, repository, _acceptance(conversation_id))
    connection = db.get_connection()
    connection.execute("DELETE FROM console_dispatch_checkpoints")
    connection.execute(
        "UPDATE messages SET assistant_generation_state = ? WHERE id = ?",
        (assistant_state, inserted.assistant_message_id),
    )
    connection.commit()

    assert _reconcile(repository, conversation_id) is None


def _restored_store(
    db: CharactersRAGDB,
    conversation_id: str,
) -> tuple[ConsoleChatStore, str]:
    rows = db.get_messages_for_conversation(conversation_id, limit=100)
    nodes = [
        ConsoleChatMessage(
            id=str(row["id"]),
            role=ConsoleMessageRole(str(row["role"])),
            content=str(row.get("content") or ""),
            persisted_message_id=str(row["id"]),
            parent_message_id=(
                str(row["parent_message_id"])
                if row.get("parent_message_id") is not None
                else None
            ),
        )
        for row in rows
    ]
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.restore_persisted_session(
        title="recovery",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=nodes,
        active_leaf_persisted_id=db.get_conversation_active_leaf(conversation_id),
    )
    return store, session.id


def test_store_hydrates_recovery_before_publishing_restored_session(
    tmp_path: Path,
) -> None:
    _action_id, kind, _state = _recovery_symbols()
    db, conversation_id, repository = _database(tmp_path / "store-load.sqlite")
    checkpoint = _insert(db, repository, _acceptance(conversation_id))

    store, session_id = _restored_store(db, conversation_id)
    recovery = store.dispatch_recovery_for_session(session_id)

    assert recovery.kind is kind.ACCEPTED
    assert recovery.checkpoint == checkpoint


class _NoReplayGateway:
    def __init__(self, db: CharactersRAGDB) -> None:
        self.db = db
        self.resolve_calls = 0
        self.provider_states: list[tuple[str, int]] = []

    async def resolve_for_send(self, _selection):
        self.resolve_calls += 1
        return SimpleNamespace(
            ready=True,
            visible_copy="",
            resolved_destination=_destination(),
            provider="llama_cpp",
            model="test-model",
            base_url="http://127.0.0.1:9099",
        )

    async def stream_chat(self, _resolution, _messages, **_kwargs):
        row = (
            self.db.get_connection()
            .execute(
                "SELECT assistant_generation_state, version FROM messages "
                "WHERE id = 'assistant-1'"
            )
            .fetchone()
        )
        self.provider_states.append((row[0], row[1]))
        yield "recovered"


class _SettlementFaultGateway(_NoReplayGateway):
    def __init__(self, db: CharactersRAGDB, *, outcome: str) -> None:
        super().__init__(db)
        self.outcome = outcome
        self.started = asyncio.Event()
        self.never_release = asyncio.Event()
        self.assistant_before_terminal: tuple[object, ...] | None = None
        self.checkpoint_before_terminal: tuple[object, ...] | None = None

    async def stream_chat(self, _resolution, _messages, **_kwargs):
        connection = self.db.get_connection()
        self.assistant_before_terminal = tuple(
            connection.execute(
                "SELECT * FROM messages WHERE id = 'assistant-1'"
            ).fetchone()
        )
        self.checkpoint_before_terminal = tuple(
            connection.execute(
                "SELECT * FROM console_dispatch_checkpoints "
                "WHERE assistant_message_id = 'assistant-1'"
            ).fetchone()
        )
        self.started.set()
        if self.outcome == "failure":
            raise RuntimeError("provider failed")
        yield "partial" if self.outcome == "cancel" else "recovered"
        if self.outcome == "cancel":
            await self.never_release.wait()


async def _patch_exact_retry_context(
    monkeypatch: pytest.MonkeyPatch,
    controller: ConsoleChatController,
    gateway: _NoReplayGateway,
) -> None:
    async def exact_context(_session_id, recovery):
        return SimpleNamespace(
            resolution=await gateway.resolve_for_send(None),
            authority=replace(
                recovery.checkpoint.frozen_authority, attempt_id="attempt-retry"
            ),
            destination=recovery.checkpoint.resolved_destination,
            provider_messages=[{"role": "user", "content": "hello"}],
        )

    monkeypatch.setattr(controller, "_resolve_dispatch_retry_context", exact_context)


def _assert_terminal_fault_retained(
    db: CharactersRAGDB,
    store: ConsoleChatStore,
    session_id: str,
    gateway: _SettlementFaultGateway,
) -> None:
    connection = db.get_connection()
    assert (
        tuple(
            connection.execute(
                "SELECT * FROM messages WHERE id = 'assistant-1'"
            ).fetchone()
        )
        == gateway.assistant_before_terminal
    )
    assert (
        tuple(
            connection.execute(
                "SELECT * FROM console_dispatch_checkpoints "
                "WHERE assistant_message_id = 'assistant-1'"
            ).fetchone()
        )
        == gateway.checkpoint_before_terminal
    )
    assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    recovery = store.dispatch_recovery_for_session(session_id)
    assert recovery is not None
    assert recovery.assistant_message_id == "assistant-1"
    assert all(action.enabled for action in recovery.actions)
    assistant = store.get_message("assistant-1")
    assert (assistant.content, assistant.status) == ("", "complete")


@pytest.mark.asyncio
async def test_accepted_retry_cas_precedes_provider_and_reuses_exact_owners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _recovery_symbols()
    db, conversation_id, repository = _database(tmp_path / "retry.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    store, session_id = _restored_store(db, conversation_id)
    gateway = _NoReplayGateway(db)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
        base_url="http://127.0.0.1:9099",
        agent_runtime_enabled=False,
    )

    await _patch_exact_retry_context(monkeypatch, controller, gateway)
    before = (
        db.get_connection()
        .execute("SELECT id, role FROM messages ORDER BY timestamp, id")
        .fetchall()
    )

    result = await controller.retry_dispatch_recovery(session_id)

    assert result.accepted is True
    assert gateway.provider_states == [("dispatch_started", 2)]
    after = (
        db.get_connection()
        .execute("SELECT id, role FROM messages ORDER BY timestamp, id")
        .fetchall()
    )
    assert [tuple(row) for row in after] == [tuple(row) for row in before]


@pytest.mark.asyncio
async def test_dispatch_started_never_auto_replays_and_requires_retry_anyway(
    tmp_path: Path,
) -> None:
    _recovery_symbols()
    db, conversation_id, repository = _database(tmp_path / "no-auto.sqlite")
    _start(repository, _insert(db, repository, _acceptance(conversation_id)))
    store, session_id = _restored_store(db, conversation_id)
    gateway = _NoReplayGateway(db)

    ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
        base_url="http://127.0.0.1:9099",
        agent_runtime_enabled=False,
    )
    await __import__("asyncio").sleep(0)

    assert gateway.resolve_calls == 0
    assert gateway.provider_states == []
    assert store.dispatch_recovery_for_session(session_id).actions[0].label == (
        "Retry anyway"
    )


@pytest.mark.asyncio
async def test_discard_atomically_settles_same_assistant_and_retains_user(
    tmp_path: Path,
) -> None:
    _recovery_symbols()
    db, conversation_id, repository = _database(tmp_path / "discard.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    store, session_id = _restored_store(db, conversation_id)
    gateway = _NoReplayGateway(db)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_runtime_enabled=False,
    )

    result = await controller.discard_dispatch_recovery(session_id)

    assert result.accepted is True
    assert gateway.resolve_calls == 0
    assert db.get_message_by_id("user-1") is not None
    assistant = db.get_message_by_id("assistant-1")
    assert assistant is not None
    assert (
        assistant["content"],
        assistant["assistant_generation_state"],
        assistant["version"],
    ) == (DISCARD_COPY, "discarded", 2)
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
        .fetchone()[0]
        == 0
    )
    assert store.dispatch_recovery_for_session(session_id) is None


def test_terminal_settlement_preserves_local_metadata_and_usage_atomically(
    tmp_path: Path,
) -> None:
    db, conversation_id, repository = _database(tmp_path / "terminal-metadata.sqlite")
    started = _start(
        repository,
        _insert(db, repository, _acceptance(conversation_id)),
    )
    metadata_json = '{"origin":"task15-test"}'
    usage_json = '{"uncached_input":7,"output":3}'

    result = repository.settle_with_assistant(
        ConsoleAssistantSettlement(
            assistant_message_id=started.assistant_message_id,
            expected_checkpoint_state=started.state,
            expected_checkpoint_revision=started.checkpoint_revision,
            expected_user_message_version=started.user_message_version,
            expected_assistant_message_version=started.assistant_message_version,
            terminal_state="complete",
            content="settled",
            metadata_json=metadata_json,
            usage_json=usage_json,
        )
    )

    assert result.status.value == "committed"
    row = (
        db.get_connection()
        .execute(
            "SELECT content, assistant_generation_state, metadata_json, usage_json "
            "FROM messages WHERE id = ?",
            (started.assistant_message_id,),
        )
        .fetchone()
    )
    assert tuple(row) == ("settled", "complete", metadata_json, usage_json)
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
        .fetchone()[0]
        == 0
    )


@pytest.mark.asyncio
async def test_discard_delete_failure_rolls_back_terminal_and_restores_actions(
    tmp_path: Path,
) -> None:
    _recovery_symbols()
    db, conversation_id, repository = _database(tmp_path / "discard-fail.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    store, session_id = _restored_store(db, conversation_id)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_NoReplayGateway(db),
        agent_runtime_enabled=False,
    )
    connection = db.get_connection()
    assistant_before = tuple(
        connection.execute("SELECT * FROM messages WHERE id = 'assistant-1'").fetchone()
    )
    checkpoint_before = tuple(
        connection.execute(
            "SELECT * FROM console_dispatch_checkpoints "
            "WHERE assistant_message_id = 'assistant-1'"
        ).fetchone()
    )
    db.get_connection().execute(
        "CREATE TRIGGER fail_discard_delete BEFORE DELETE ON "
        "console_dispatch_checkpoints BEGIN SELECT RAISE(ABORT, 'fail'); END"
    )
    db.get_connection().commit()

    result = await controller.discard_dispatch_recovery(session_id)

    assert result.accepted is False
    assert (
        tuple(
            connection.execute(
                "SELECT * FROM messages WHERE id = 'assistant-1'"
            ).fetchone()
        )
        == assistant_before
    )
    assert (
        tuple(
            connection.execute(
                "SELECT * FROM console_dispatch_checkpoints "
                "WHERE assistant_message_id = 'assistant-1'"
            ).fetchone()
        )
        == checkpoint_before
    )
    recovery = store.dispatch_recovery_for_session(session_id)
    assert recovery is not None
    assert all(action.enabled for action in recovery.actions)
    assert controller.run_state_for(session_id).status is ConsoleRunStatus.BLOCKED


@pytest.mark.parametrize("provider_outcome", ["success", "failure"])
@pytest.mark.asyncio
async def test_retry_terminal_delete_failure_retains_exact_preterminal_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider_outcome: str,
) -> None:
    db, conversation_id, repository = _database(
        tmp_path / f"retry-terminal-{provider_outcome}.sqlite"
    )
    _insert(db, repository, _acceptance(conversation_id))
    store, session_id = _restored_store(db, conversation_id)
    gateway = _SettlementFaultGateway(db, outcome=provider_outcome)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
        base_url="http://127.0.0.1:9099",
        agent_runtime_enabled=False,
    )
    await _patch_exact_retry_context(monkeypatch, controller, gateway)
    db.get_connection().execute(
        "CREATE TRIGGER fail_retry_terminal_delete BEFORE DELETE ON "
        "console_dispatch_checkpoints BEGIN SELECT RAISE(ABORT, 'fail'); END"
    )
    db.get_connection().commit()

    result = await controller.retry_dispatch_recovery(session_id)

    assert result.accepted is False
    assert result.visible_copy == "Response recovery failed. Try again or discard."
    assert controller.run_state_for(session_id).status is ConsoleRunStatus.BLOCKED
    _assert_terminal_fault_retained(db, store, session_id, gateway)


@pytest.mark.asyncio
async def test_retry_cancel_settlement_failure_retains_exact_preterminal_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, conversation_id, repository = _database(
        tmp_path / "retry-terminal-cancel.sqlite"
    )
    _insert(db, repository, _acceptance(conversation_id))
    store, session_id = _restored_store(db, conversation_id)
    gateway = _SettlementFaultGateway(db, outcome="cancel")
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
        base_url="http://127.0.0.1:9099",
        agent_runtime_enabled=False,
    )
    await _patch_exact_retry_context(monkeypatch, controller, gateway)
    db.get_connection().execute(
        "CREATE TRIGGER fail_retry_cancel_delete BEFORE DELETE ON "
        "console_dispatch_checkpoints BEGIN SELECT RAISE(ABORT, 'fail'); END"
    )
    db.get_connection().commit()

    task = asyncio.create_task(controller.retry_dispatch_recovery(session_id))
    await asyncio.wait_for(gateway.started.wait(), timeout=1)
    await asyncio.sleep(0)
    stopped = controller.stop_active_run(record_user_stop=False)
    result = await asyncio.wait_for(task, timeout=1)

    assert stopped is True
    assert result.accepted is False
    assert result.visible_copy == "Response recovery failed. Try again or discard."
    assert controller.run_state_for(session_id).status is ConsoleRunStatus.BLOCKED
    _assert_terminal_fault_retained(db, store, session_id, gateway)


@pytest.mark.parametrize("guard", ["user_version", "assistant_version", "deleted"])
@pytest.mark.asyncio
async def test_discard_rejects_changed_or_deleted_owner_without_half_settlement(
    tmp_path: Path,
    guard: str,
) -> None:
    _recovery_symbols()
    db, conversation_id, repository = _database(tmp_path / f"guard-{guard}.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    store, session_id = _restored_store(db, conversation_id)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_NoReplayGateway(db),
        agent_runtime_enabled=False,
    )
    connection = db.get_connection()
    if guard == "user_version":
        connection.execute("UPDATE messages SET version = 2 WHERE id = 'user-1'")
    elif guard == "assistant_version":
        connection.execute("UPDATE messages SET version = 2 WHERE id = 'assistant-1'")
    else:
        connection.execute("UPDATE messages SET deleted = 1 WHERE id = 'assistant-1'")
    connection.commit()

    result = await controller.discard_dispatch_recovery(session_id)

    assert result.accepted is False
    row = connection.execute(
        "SELECT content, assistant_generation_state FROM messages "
        "WHERE id = 'assistant-1'"
    ).fetchone()
    assert tuple(row) == ("", "accepted")
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM console_dispatch_checkpoints"
        ).fetchone()[0]
        == 1
    )
