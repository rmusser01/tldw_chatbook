from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from Tests.console_resource_fixtures import (
    close_owned_console_resources as close_owned_console_resources,
)

import tldw_chatbook.Chat.console_chat_models as recovery_models
from tldw_chatbook.Canvas.models import CanvasScope
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_canvas_controller import ConsoleCanvasController
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_chat_store import (
    ConsoleChatStore,
    ConsoleDispatchSettlementError,
)
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


def _raw_semantic_corruption(
    db: CharactersRAGDB,
    sql: str,
    params: tuple[object, ...] = (),
):
    """Execute one deliberate corruption statement, then restore the real guard."""
    connection = db.get_connection()
    authorization = db._semantic_mutation_authorization_for_coordinator(connection)
    connection.create_function(
        "console_semantic_mutation_authorized", 2, lambda *_args: 1
    )
    try:
        return connection.execute(sql, params)
    finally:
        connection.create_function(
            "console_semantic_mutation_authorized",
            2,
            authorization._sqlite_authorized,
        )


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
    _raw_semantic_corruption(
        db,
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
    _raw_semantic_corruption(
        db,
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
    _raw_semantic_corruption(
        db,
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
    _raw_semantic_corruption(
        db,
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
        _raw_semantic_corruption(
            db,
            "UPDATE messages SET role = 'assistant' WHERE id = ?",
            (inserted.user_message_id,),
        )
    elif corruption == "cross_conversation":
        _raw_semantic_corruption(
            db,
            "UPDATE messages SET conversation_id = ? WHERE id = ?",
            (other_id, inserted.user_message_id),
        )
    elif corruption == "missing_user":
        _raw_semantic_corruption(
            db, "DELETE FROM messages WHERE id = ?", (inserted.user_message_id,)
        )
    elif corruption == "wrong_assistant_role":
        _raw_semantic_corruption(
            db,
            "UPDATE messages SET role = 'user' WHERE id = ?",
            (inserted.assistant_message_id,),
        )
    elif corruption == "assistant_version":
        _raw_semantic_corruption(
            db,
            "UPDATE messages SET version = 2 WHERE id = ?",
            (inserted.assistant_message_id,),
        )
    else:
        _raw_semantic_corruption(
            db,
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
    _raw_semantic_corruption(
        db,
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
    _raw_semantic_corruption(
        db,
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
    _raw_semantic_corruption(
        db,
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


def test_dispatch_retry_cas_gap_rejects_fork_until_runtime_owner_is_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action_id, _kind, _state = _recovery_symbols()
    db, conversation_id, repository = _database(tmp_path / "fork-cas-gap.sqlite")
    inserted = _insert(db, repository, _acceptance(conversation_id))
    store, session_id = _restored_store(db, conversation_id)
    recovery = store.dispatch_recovery_for_session(session_id)
    assert recovery is not None
    claimed = store.claim_dispatch_recovery_action(
        session_id,
        action_id.RETRY_RESPONSE,
    )
    assert claimed is not None
    committed = Event()
    release = Event()
    failures: list[BaseException] = []
    store_repository = store.persistence.console_dispatch_repository
    original = store_repository.cas_state

    def blocking_cas(transition):
        result = original(transition)
        committed.set()
        assert release.wait(2)
        return result

    monkeypatch.setattr(store_repository, "cas_state", blocking_cas)

    def transition() -> None:
        try:
            store.transition_dispatch_recovery_for_retry(
                session_id,
                assistant_message_id=inserted.assistant_message_id,
                new_attempt_id="attempt-2",
            )
        except BaseException as exc:  # pragma: no cover - assertion reports it
            failures.append(exc)

    thread = Thread(target=transition)
    thread.start()
    assert committed.wait(2)
    eligibility = store.fork_eligibility(inserted.assistant_message_id)
    assert eligibility.eligible is False
    assert "changing" in eligibility.reason.lower()
    release.set()
    thread.join(2)
    assert not thread.is_alive()
    assert failures == []
    assert session_id not in store._fork_source_transitions


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


def test_terminal_settlement_commits_canvas_revision_with_assistant_message(
    tmp_path: Path,
) -> None:
    db, conversation_id, repository = _database(tmp_path / "terminal-canvas.sqlite")
    started = _start(repository, _insert(db, repository, _acceptance(conversation_id)))
    canvas = ConsoleCanvasController()
    scope = CanvasScope(
        session_id="session-1",
        conversation_id=conversation_id,
        active_message_ids=("user-1",),
        selected_canvas_id=None,
        selected_revision_id=None,
        run_id="run-1",
    )
    canvas.register_run(scope, assistant_message_id="assistant-1", temporary=False)
    created = canvas.create_canvas(
        scope, tool_call_id="call-1", title="Atomic", html="<p>atomic</p>"
    )
    staged = canvas.finish_run("run-1", "done")
    assert staged is not None and staged.contribution is not None

    result = repository.settle_with_assistant(
        ConsoleAssistantSettlement(
            assistant_message_id=started.assistant_message_id,
            expected_checkpoint_state=started.state,
            expected_checkpoint_revision=started.checkpoint_revision,
            expected_user_message_version=started.user_message_version,
            expected_assistant_message_version=started.assistant_message_version,
            terminal_state="complete",
            content="",
            metadata_json=staged.metadata_json,
            contributions=(staged.contribution,),
        )
    )

    assert result.status.value == "committed"
    message = db.get_message_by_id("assistant-1")
    revision = db.get_connection().execute(
        "SELECT origin_message_id, html FROM canvas_revisions WHERE id = ?",
        (created.revision.revision_id,),
    ).fetchone()
    assert message["assistant_generation_state"] == "complete"
    assert tuple(revision) == ("assistant-1", "<p>atomic</p>")


def test_canvas_revision_failure_rolls_back_terminal_message(
    tmp_path: Path,
) -> None:
    db, conversation_id, repository = _database(tmp_path / "terminal-canvas-fail.sqlite")
    started = _start(repository, _insert(db, repository, _acceptance(conversation_id)))
    canvas = ConsoleCanvasController()
    scope = CanvasScope(
        session_id="session-1",
        conversation_id=conversation_id,
        active_message_ids=("user-1",),
        selected_canvas_id=None,
        selected_revision_id=None,
        run_id="run-1",
    )
    canvas.register_run(scope, assistant_message_id="assistant-1", temporary=False)
    canvas.create_canvas(
        scope, tool_call_id="call-1", title="Atomic", html="<p>atomic</p>"
    )
    staged = canvas.finish_run("run-1", "done")
    assert staged is not None and staged.contribution is not None
    db.get_connection().execute(
        "CREATE TRIGGER fail_canvas_revision BEFORE INSERT ON canvas_revisions "
        "BEGIN SELECT RAISE(ABORT, 'injected revision failure'); END"
    )

    with pytest.raises(Exception, match="injected revision failure"):
        repository.settle_with_assistant(
            ConsoleAssistantSettlement(
                assistant_message_id=started.assistant_message_id,
                expected_checkpoint_state=started.state,
                expected_checkpoint_revision=started.checkpoint_revision,
                expected_user_message_version=started.user_message_version,
                expected_assistant_message_version=started.assistant_message_version,
                terminal_state="complete",
                content="settled",
                metadata_json=staged.metadata_json,
                contributions=(staged.contribution,),
            )
        )

    message = db.get_message_by_id("assistant-1")
    assert message["content"] == ""
    assert message["assistant_generation_state"] == "dispatch_started"
    assert db.get_connection().execute(
        "SELECT COUNT(*) FROM canvas_revisions"
    ).fetchone()[0] == 0


def test_terminal_message_failure_never_writes_canvas_revision(
    tmp_path: Path,
) -> None:
    db, conversation_id, repository = _database(tmp_path / "terminal-message-fail.sqlite")
    started = _start(repository, _insert(db, repository, _acceptance(conversation_id)))
    canvas = ConsoleCanvasController()
    scope = CanvasScope(
        session_id="session-1",
        conversation_id=conversation_id,
        active_message_ids=("user-1",),
        selected_canvas_id=None,
        selected_revision_id=None,
        run_id="run-1",
    )
    canvas.register_run(scope, assistant_message_id="assistant-1", temporary=False)
    canvas.create_canvas(
        scope, tool_call_id="call-1", title="Atomic", html="<p>atomic</p>"
    )
    staged = canvas.finish_run("run-1", "done")
    assert staged is not None and staged.contribution is not None
    db.get_connection().execute(
        "CREATE TRIGGER fail_terminal_message BEFORE UPDATE ON messages "
        "WHEN OLD.id = 'assistant-1' "
        "BEGIN SELECT RAISE(ABORT, 'injected message failure'); END"
    )

    with pytest.raises(Exception, match="injected message failure"):
        repository.settle_with_assistant(
            ConsoleAssistantSettlement(
                assistant_message_id=started.assistant_message_id,
                expected_checkpoint_state=started.state,
                expected_checkpoint_revision=started.checkpoint_revision,
                expected_user_message_version=started.user_message_version,
                expected_assistant_message_version=started.assistant_message_version,
                terminal_state="complete",
                content="settled",
                metadata_json=staged.metadata_json,
                contributions=(staged.contribution,),
            )
        )

    assert db.get_connection().execute(
        "SELECT COUNT(*) FROM canvas_revisions"
    ).fetchone()[0] == 0


def test_store_confirms_canvas_stage_only_after_terminal_transaction(
    tmp_path: Path,
) -> None:
    db, conversation_id, repository = _database(tmp_path / "store-terminal-canvas.sqlite")
    started = _start(repository, _insert(db, repository, _acceptance(conversation_id)))
    store, session_id = _restored_store(db, conversation_id)
    claimed = store.publish_durable_dispatch_checkpoint(session_id, started, in_flight=True)
    assert claimed.in_flight
    assistant = store._message_or_raise(started.assistant_message_id)
    canvas = ConsoleCanvasController()
    canvas_scope = CanvasScope(
        session_id=session_id,
        conversation_id=conversation_id,
        active_message_ids=("user-1",),
        selected_canvas_id=None,
        selected_revision_id=None,
        run_id="run-1",
    )
    canvas.register_run(
        canvas_scope, assistant_message_id=assistant.id, temporary=False
    )
    created = canvas.create_canvas(
        canvas_scope,
        tool_call_id="call-1",
        title="Store atomic",
        html="<p>store atomic</p>",
    )
    canvas.finish_run("run-1", "done")
    store.canvas_turn_controller = canvas
    assistant.status = "streaming"
    assistant.assistant_generation_state = "streaming"
    completed = store.mark_message_complete(assistant.id)

    assert completed.status == "complete"
    assert canvas.settlement_for_assistant(assistant.id).state.value == "committed"
    assert db.get_connection().execute(
        "SELECT COUNT(*) FROM canvas_revisions WHERE id = ?",
        (created.revision.revision_id,),
    ).fetchone()[0] == 1
    metadata = json.loads(db.get_message_by_id("assistant-1")["metadata_json"])
    assert metadata["canvas_cards"][0]["title"] == "Store atomic"


def test_store_closes_successful_canvas_run_without_mutations(tmp_path: Path) -> None:
    db, conversation_id, repository = _database(tmp_path / "store-empty-canvas.sqlite")
    started = _start(repository, _insert(db, repository, _acceptance(conversation_id)))
    store, session_id = _restored_store(db, conversation_id)
    store.publish_durable_dispatch_checkpoint(session_id, started, in_flight=True)
    assistant = store._message_or_raise(started.assistant_message_id)
    canvas = ConsoleCanvasController()
    canvas.register_run(
        CanvasScope(
            session_id=session_id,
            conversation_id=conversation_id,
            active_message_ids=("user-1",),
            selected_canvas_id=None,
            selected_revision_id=None,
            run_id="run-1",
        ),
        assistant_message_id=assistant.id,
        temporary=False,
    )
    canvas.finish_run("run-1", "done")
    store.canvas_turn_controller = canvas
    assistant.status = "streaming"
    assistant.assistant_generation_state = "streaming"

    completed = store.mark_message_complete(assistant.id)

    assert completed.status == "complete"
    assert canvas.settlement_for_assistant(assistant.id).state.value == "committed"


def test_store_retains_canvas_stage_when_terminal_transaction_fails(
    tmp_path: Path,
) -> None:
    db, conversation_id, repository = _database(tmp_path / "store-terminal-canvas-fail.sqlite")
    started = _start(repository, _insert(db, repository, _acceptance(conversation_id)))
    store, session_id = _restored_store(db, conversation_id)
    store.publish_durable_dispatch_checkpoint(session_id, started, in_flight=True)
    assistant = store._message_or_raise(started.assistant_message_id)
    canvas = ConsoleCanvasController()
    canvas_scope = CanvasScope(
        session_id=session_id,
        conversation_id=conversation_id,
        active_message_ids=("user-1",),
        selected_canvas_id=None,
        selected_revision_id=None,
        run_id="run-1",
    )
    canvas.register_run(canvas_scope, assistant_message_id=assistant.id, temporary=False)
    canvas.create_canvas(
        canvas_scope, tool_call_id="call-1", title="Fail", html="<p>fail</p>"
    )
    canvas.finish_run("run-1", "done")
    store.canvas_turn_controller = canvas
    assistant.status = "streaming"
    assistant.assistant_generation_state = "streaming"
    db.get_connection().execute(
        "CREATE TRIGGER fail_canvas_revision BEFORE INSERT ON canvas_revisions "
        "BEGIN SELECT RAISE(ABORT, 'injected revision failure'); END"
    )

    with pytest.raises(ConsoleDispatchSettlementError):
        store.mark_message_complete(assistant.id)

    assert canvas.settlement_for_assistant(assistant.id).state.value == "ready"
    assert db.get_message_by_id("assistant-1")["assistant_generation_state"] == (
        "dispatch_started"
    )

    db.get_connection().execute("DROP TRIGGER fail_canvas_revision")
    completed = store.mark_message_complete(assistant.id)

    assert completed.status == "complete"
    assert canvas.settlement_for_assistant(assistant.id).state.value == "committed"


def test_post_commit_dispatch_owner_change_reconciles_terminal_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _recovery_symbols()
    db, conversation_id, repository = _database(
        tmp_path / "post-commit-owner-change.sqlite"
    )
    started = _start(
        repository,
        _insert(db, repository, _acceptance(conversation_id)),
    )
    store, session_id = _restored_store(db, conversation_id)
    claimed = store.publish_durable_dispatch_checkpoint(
        session_id, started, in_flight=True
    )
    assert claimed.in_flight
    assistant = store._message_or_raise(started.assistant_message_id)
    generation_token = store.begin_generation_attempt(assistant.id)
    assert store._dispatch_recovery_generation_tokens[session_id] == generation_token
    assistant.content = "settled despite sidecar"
    assistant.status = "streaming"
    assistant.assistant_generation_state = "streaming"
    canvas = ConsoleCanvasController()
    canvas_scope = CanvasScope(
        session_id=session_id,
        conversation_id=conversation_id,
        active_message_ids=("user-1",),
        selected_canvas_id=None,
        selected_revision_id=None,
        run_id="postcommit-canvas-run",
    )
    canvas.register_run(
        canvas_scope, assistant_message_id=assistant.id, temporary=False
    )
    created = canvas.create_canvas(
        canvas_scope,
        tool_call_id="postcommit-canvas-call",
        title="Postcommit",
        html="<p>postcommit private source</p>",
    )
    canvas.finish_run("postcommit-canvas-run", "done")
    store.canvas_turn_controller = canvas
    persisted_repository = store.persistence.console_dispatch_repository
    original = persisted_repository.settle_with_assistant

    def commit_then_replace_owner(settlement):
        result = original(settlement)
        current = store._dispatch_recoveries_by_session[session_id]
        store._dispatch_recoveries_by_session[session_id] = replace(current)
        return result

    monkeypatch.setattr(
        persisted_repository, "settle_with_assistant", commit_then_replace_owner
    )

    with pytest.raises(RuntimeError, match="owner changed during settlement"):
        store.mark_message_complete(assistant.id)

    row = db.get_message_by_id(assistant.persisted_message_id)
    current = store.get_message(assistant.id)
    assert row["content"] == current.content == "settled despite sidecar"
    assert row["assistant_generation_state"] == "complete"
    assert canvas.settlement_for_assistant(assistant.id).state.value == "committed"
    assert db.get_connection().execute(
        "SELECT COUNT(*) FROM canvas_revisions WHERE id = ?",
        (created.revision.revision_id,),
    ).fetchone()[0] == 1
    assert current.status == current.assistant_generation_state == "complete"
    assert current.provider_continuation_message_version == row["version"] == 3
    assert store.dispatch_recovery_for_session(session_id) is None
    assert store.dispatch_recovery_for_presentation(session_id) is None
    assert not store.dispatch_recovery_blocks_submission(session_id)
    assert session_id not in store._dispatch_recovery_message_baselines
    assert session_id not in store._dispatch_recovery_generation_tokens
    assert session_id not in store._dispatch_recovery_queue_hydration_pending
    assert (
        db.get_connection()
        .execute(
            "SELECT COUNT(*) FROM console_dispatch_checkpoints "
            "WHERE assistant_message_id = ?",
            (assistant.persisted_message_id,),
        )
        .fetchone()[0]
        == 0
    )


def test_dispatch_reconcile_read_apply_is_atomic_with_new_durable_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale no-checkpoint read cannot erase a newly published owner."""

    db, conversation_id, repository = _database(tmp_path / "dispatch-toctou.sqlite")
    first = _start(
        repository,
        _insert(db, repository, _acceptance(conversation_id)),
    )
    settled = repository.settle_with_assistant(
        ConsoleAssistantSettlement(
            assistant_message_id=first.assistant_message_id,
            expected_checkpoint_state=first.state,
            expected_checkpoint_revision=first.checkpoint_revision,
            expected_user_message_version=first.user_message_version,
            expected_assistant_message_version=first.assistant_message_version,
            terminal_state="complete",
            content="first answer",
            metadata_json=None,
        )
    )
    assert settled.status.value == "committed"
    store, session_id = _restored_store(db, conversation_id)
    user = store.append_message(
        session_id,
        role=ConsoleMessageRole.USER,
        content="next question",
        persist=False,
    )
    assistant = store.append_message(
        session_id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=False,
    )
    acceptance = replace(
        _acceptance(conversation_id),
        user_message_id=user.id,
        assistant_message_id=assistant.id,
        parent_message_id="assistant-1",
        user_content="next question",
        preparation_id="preparation-2",
        attempt_id="attempt-2",
        frozen_authority=_authority(attempt_id="attempt-2"),
    )
    persisted_repository = store.persistence.console_dispatch_repository
    original_reconcile = persisted_repository.reconcile_for_session
    read_done = Event()
    release_read = Event()

    def paused_empty_reconcile(target_conversation_id: str):
        recovery = original_reconcile(target_conversation_id)
        assert recovery is None
        read_done.set()
        assert release_read.wait(5)
        return recovery

    monkeypatch.setattr(
        persisted_repository, "reconcile_for_session", paused_empty_reconcile
    )
    hydration_failure: list[BaseException] = []

    def reconcile_runtime() -> None:
        try:
            store._hydrate_dispatch_recovery(session_id, conversation_id)
        except BaseException as exc:
            hydration_failure.append(exc)

    hydration = Thread(target=reconcile_runtime)
    hydration.start()
    assert read_done.wait(5)
    second = _start(repository, _insert(db, repository, acceptance))
    publish_started = Event()
    publish_done = Event()
    publish_failure: list[BaseException] = []

    def publish_new_owner() -> None:
        publish_started.set()
        try:
            store.publish_durable_dispatch_checkpoint(
                session_id, second, in_flight=True
            )
        except BaseException as exc:
            publish_failure.append(exc)
        finally:
            publish_done.set()

    publisher = Thread(target=publish_new_owner)
    publisher.start()
    assert publish_started.wait(5)
    assert not publish_done.wait(0.1)
    release_read.set()
    hydration.join(5)
    publisher.join(5)

    assert not hydration.is_alive()
    assert not publisher.is_alive()
    assert not hydration_failure
    assert not publish_failure
    token = store.begin_generation_attempt(assistant.id)
    assert store._dispatch_recovery_generation_tokens[session_id] == token
    assert store.mark_dispatch_recovery_needed(
        session_id, assistant.id, generation_token=token
    )
    runtime = store.dispatch_recovery_for_session(session_id)
    presentation = store.dispatch_recovery_for_presentation(session_id)
    assert runtime is not None and runtime.assistant_message_id == assistant.id
    assert presentation is not None
    assert presentation.assistant_message_id == assistant.id
    assert session_id not in store._dispatch_recovery_generation_tokens
    checkpoint_row = (
        db.get_connection()
        .execute("SELECT assistant_message_id FROM console_dispatch_checkpoints")
        .fetchone()
    )
    assert checkpoint_row["assistant_message_id"] == assistant.id


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
        _raw_semantic_corruption(
            db, "UPDATE messages SET version = 2 WHERE id = 'user-1'"
        )
    elif guard == "assistant_version":
        _raw_semantic_corruption(
            db, "UPDATE messages SET version = 2 WHERE id = 'assistant-1'"
        )
    else:
        _raw_semantic_corruption(
            db, "UPDATE messages SET deleted = 1 WHERE id = 'assistant-1'"
        )
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
