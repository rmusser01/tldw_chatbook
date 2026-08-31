"""Task 16: exclusive dispatch-to-continuation ownership and rebind."""

from __future__ import annotations

from dataclasses import replace
from contextlib import contextmanager
from pathlib import Path
import sqlite3

import pytest

from Tests.Chat.test_console_dispatch_recovery import (
    _acceptance,
    _database,
    _insert,
    _raw_semantic_corruption,
    _restored_store,
    _start,
)
from tldw_chatbook.Agents.agent_models import (
    RUN_CANCELLED,
    RUN_ERROR,
    AgentConfig,
    ContinuationEventContext,
    ModelTurn,
    ToolBatchReady,
    ToolCall,
    ToolLoadSelection,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents import agent_service as agent_service_module
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleDispatchRecoveryKind,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleAssistantSettlement,
    ConsoleContinuationHandoff,
    ConsoleDispatchResultStatus,
    ConsoleEgressClass,
    ConsoleProviderIntent,
    ConsoleResolvedDestination,
)
from tldw_chatbook.Chat.thinking_blocks import (
    DisplayableThinkingBlock,
    ThinkingEnvelope,
    dump_thinking_blocks_json,
)
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationCall,
    ContinuationRound,
    ProviderContinuationCheckpoint,
    dump_provider_continuation_json,
    parse_provider_continuation_json,
)
from tldw_chatbook.Sync_Interop.chat_outbox_producer import (
    ChatSyncV2OutboxProducer,
)
from tldw_chatbook.Sync_Interop.crypto import generate_dataset_key
from tldw_chatbook.Sync_Interop.hashing import canonical_payload_hash
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


_SCOPE = {
    "server_profile_id": "server-a",
    "authenticated_principal_id": "user-a",
    "workspace_scope": "workspace-a",
}
_TOOL_SCHEMA = ToolSchema(
    id="builtin:calculator",
    name="calculator",
    description="math",
    parameters={"type": "object"},
)
_CONFIG = AgentConfig(
    model="deepseek-v4-flash",
    system_prompt="system",
    allowed_tools=("calculator",),
)


def _continuation(
    *,
    revision: int = 1,
    state: str = "active",
    call_state: str = "pending",
    content: str = "",
) -> ProviderContinuationCheckpoint:
    result = None
    if call_state in {"completed", "failed"}:
        from tldw_chatbook.Chat.provider_continuation import ContinuationResult

        result = ContinuationResult("4")
    return ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=revision,
        provider="deepseek",
        protocol="responses",
        model="deepseek-v4-flash",
        api_base_url="https://api.deepseek.com/v1",
        state=state,  # type: ignore[arg-type]
        rounds=(
            ContinuationRound(
                assistant_content=content,
                reasoning_blocks=("private-reasoning",),
                calls=(
                    ContinuationCall(
                        call_id="call-1",
                        name="calculator",
                        arguments='{"expression":"2+2"}',
                        state=call_state,  # type: ignore[arg-type]
                        result=result,
                    ),
                ),
            ),
        ),
    )


def _thinking(*, status: str = "complete") -> str:
    return dump_thinking_blocks_json(
        ThinkingEnvelope(
            (
                DisplayableThinkingBlock(
                    block_id="reasoning-1",
                    round_ordinal=0,
                    provider="deepseek",
                    model="deepseek-v4-flash",
                    protocol="responses",
                    source_format="reasoning_content",
                    status=status,  # type: ignore[arg-type]
                    text="bounded reasoning",
                ),
            )
        )
    )


def _deepseek_acceptance(conversation_id: str):
    acceptance = _acceptance(conversation_id)
    return replace(
        acceptance,
        frozen_authority=replace(
            acceptance.frozen_authority,
            provider_intent=ConsoleProviderIntent(
                provider="deepseek",
                model="deepseek-v4-flash",
                endpoint="https://api.deepseek.com/v1",
            ),
        ),
        resolved_destination=ConsoleResolvedDestination(
            provider="deepseek",
            model="deepseek-v4-flash",
            endpoint_identity="https://api.deepseek.com/v1",
            egress_class=ConsoleEgressClass.PUBLIC_NETWORK,
        ),
    )


def _moonshot_acceptance(conversation_id: str):
    acceptance = _acceptance(conversation_id)
    return replace(
        acceptance,
        frozen_authority=replace(
            acceptance.frozen_authority,
            provider_intent=ConsoleProviderIntent(
                provider="moonshot",
                model="kimi-k3",
                endpoint="https://api.moonshot.ai/v1",
            ),
        ),
        resolved_destination=ConsoleResolvedDestination(
            provider="moonshot",
            model="kimi-k3",
            endpoint_identity="https://api.moonshot.ai/v1",
            egress_class=ConsoleEgressClass.PUBLIC_NETWORK,
        ),
    )


def _portable_store(
    db,
    conversation_id: str,
    tmp_path: Path,
) -> tuple[ConsoleChatStore, str, SyncStateRepository, ChatSyncV2OutboxProducer]:
    repository = SyncStateRepository(tmp_path / "sync-v2.sqlite")
    repository.set_sync_v2_profile_state(
        **_SCOPE,
        profile_mode="local_first",
        device_id="device-a",
        dataset_id="dataset-a",
    )
    producer = ChatSyncV2OutboxProducer(
        state_repository=repository,
        dataset_keys={"dataset-a": generate_dataset_key()},
        source=db,
    )
    rows = db.get_messages_for_conversation(conversation_id, limit=100)
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage

    nodes = []
    for row in rows:
        node = ConsoleChatMessage(
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
        node.assistant_generation_state = row.get("assistant_generation_state")
        nodes.append(node)
    store = ConsoleChatStore(
        persistence=ChatPersistenceService(db),
        sync_v2_chat_producer=producer,
        sync_v2_server_profile_id=_SCOPE["server_profile_id"],
        sync_v2_authenticated_principal_id=_SCOPE["authenticated_principal_id"],
        sync_v2_workspace_scope=_SCOPE["workspace_scope"],
    )
    session = store.restore_persisted_session(
        title="handoff",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=nodes,
        active_leaf_persisted_id=db.get_conversation_active_leaf(conversation_id),
    )
    return store, session.id, repository, producer


def _started_portable(tmp_path: Path):
    db, conversation_id, repository = _database(tmp_path / "chat.sqlite")
    checkpoint = _start(
        repository,
        _insert(db, repository, _deepseek_acceptance(conversation_id)),
    )
    store, session_id, sync_repository, producer = _portable_store(
        db, conversation_id, tmp_path
    )
    store.publish_durable_dispatch_checkpoint(
        session_id,
        checkpoint,
        in_flight=True,
    )
    return (
        db,
        conversation_id,
        repository,
        checkpoint,
        store,
        session_id,
        sync_repository,
        producer,
    )


def _event() -> ToolBatchReady:
    return ToolBatchReady(
        ContinuationEventContext("assistant-1", "run-1", "primary", "persistent"),
        _continuation(),
        None,
    )


def _checkpoint_count(db) -> int:
    return int(
        db.get_connection()
        .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
        .fetchone()[0]
    )


def test_terminal_settlement_commits_thinking_with_terminal_status(
    tmp_path: Path,
) -> None:
    db, conversation_id, repository = _database(tmp_path / "thinking-settle.sqlite")
    started = _start(
        repository,
        _insert(db, repository, _acceptance(conversation_id)),
    )
    canonical = _thinking(status="stopped")

    result = repository.settle_with_assistant(
        ConsoleAssistantSettlement(
            assistant_message_id=started.assistant_message_id,
            expected_checkpoint_state=started.state,
            expected_checkpoint_revision=started.checkpoint_revision,
            expected_user_message_version=started.user_message_version,
            expected_assistant_message_version=started.assistant_message_version,
            terminal_state="stopped",
            content="partial answer",
            metadata_json=None,
            thinking_blocks_json=canonical,
        )
    )
    row = db.get_message_by_id(started.assistant_message_id)

    assert result.status is ConsoleDispatchResultStatus.COMMITTED
    assert row["assistant_generation_state"] == "stopped"
    assert row["thinking_blocks_json"] == canonical
    assert result.committed_payload_hash == canonical_payload_hash(
        {
            "assistant_generation_state": "stopped",
            "content": "partial answer",
            "role": "assistant",
            "thinking_blocks_json": canonical,
        }
    )


def test_continuation_handoff_preserves_existing_thinking(tmp_path: Path) -> None:
    db, conversation_id, repository = _database(tmp_path / "thinking-handoff.sqlite")
    started = _start(
        repository,
        _insert(db, repository, _deepseek_acceptance(conversation_id)),
    )
    canonical_thinking = _thinking()
    _raw_semantic_corruption(
        db,
        "UPDATE messages SET thinking_blocks_json = ? WHERE id = ?",
        (canonical_thinking, started.assistant_message_id),
    )
    canonical_continuation = dump_provider_continuation_json(_continuation())

    result = repository.handoff_to_provider_continuation(
        ConsoleContinuationHandoff(
            assistant_message_id=started.assistant_message_id,
            expected_checkpoint_revision=started.checkpoint_revision,
            expected_user_message_version=started.user_message_version,
            expected_assistant_message_version=started.assistant_message_version,
            provider_continuation_json=canonical_continuation,
        )
    )
    row = db.get_message_by_id(started.assistant_message_id)

    assert result.status is ConsoleDispatchResultStatus.COMMITTED
    assert row["thinking_blocks_json"] == canonical_thinking
    assert result.committed_payload_hash == canonical_payload_hash(
        {
            "assistant_generation_state": "continuation_active",
            "content": "",
            "provider_continuation_json": canonical_continuation,
            "role": "assistant",
            "thinking_blocks_json": canonical_thinking,
        }
    )


def _projected_message_versions(repository: SyncStateRepository) -> list[int]:
    with repository._get_connection() as connection:
        rows = connection.execute(
            "SELECT source_version FROM sync_v2_source_projection_receipts "
            "WHERE domain = 'chat' AND source_entity_id = 'assistant-1' "
            "ORDER BY source_version"
        ).fetchall()
    return [int(row[0]) for row in rows]


def test_first_tool_batch_uses_atomic_handoff_and_publishes_committed_proof(
    tmp_path: Path,
) -> None:
    """Kills the legacy update-without-checkpoint-delete production path."""
    (
        db,
        _conversation_id,
        _repository,
        _checkpoint,
        store,
        session_id,
        sync_repository,
        _producer,
    ) = _started_portable(tmp_path)

    store.persist_provider_continuation_event(_event())

    row = db.get_message_by_id("assistant-1")
    assert row is not None
    canonical = dump_provider_continuation_json(_continuation())
    expected_hash = canonical_payload_hash(
        {
            "assistant_generation_state": "continuation_active",
            "content": "",
            "provider_continuation_json": canonical,
            "role": "assistant",
        }
    )
    assert (
        row["assistant_generation_state"],
        row["version"],
        row["provider_continuation_json"],
    ) == ("continuation_active", 3, canonical)
    assert _checkpoint_count(db) == 0
    assert (
        db.read_committed_chat_sync_intent(
            message_id="assistant-1",
            message_version=3,
            payload_hash=expected_hash,
        )
        is not None
    )
    # Restore projects the accepted v2 owner first; handoff then projects
    # exactly one continuation-active v3 owner without rewriting either.
    assert _projected_message_versions(sync_repository) == [2, 3]
    live = store.get_message("assistant-1")
    assert live.assistant_generation_state == "continuation_active"
    assert live.provider_continuation == _continuation()
    assert live.provider_continuation_message_version == 3
    assert live.provider_continuation_actions_enabled is True
    assert store.dispatch_recovery_for_session(session_id) is None


def test_pure_runtime_commits_handoff_before_any_tool_observer(tmp_path: Path) -> None:
    """Kills moving tool review/invocation before the local handoff commit."""
    db, _, _, _, store, session_id, _, _ = _started_portable(tmp_path)
    invoked: list[str] = []
    stop = False
    call = ToolCall(
        "calculator", {"expression": "2+2"}, "call-1", '{"expression":"2+2"}'
    )

    def invoke_tool(actual: ToolCall) -> ToolResult:
        nonlocal stop
        row = db.get_message_by_id("assistant-1")
        assert row is not None
        assert row["assistant_generation_state"] == "continuation_active"
        assert _checkpoint_count(db) == 0
        assert store.dispatch_recovery_for_session(session_id) is None
        invoked.append(actual.call_id)
        stop = True
        return ToolResult(ok=True, content="4")

    outcome = run_agent_loop(
        _CONFIG,
        [{"role": "user", "content": "calculate"}],
        [_TOOL_SCHEMA],
        LoopDeps(
            call_model=lambda _messages, _schemas: ModelTurn(
                tool_calls=(call,),
                assistant_message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression":"2+2"}',
                            },
                        }
                    ],
                },
                provider_continuation=_continuation(),
            ),
            invoke_tool=invoke_tool,
            spawn=lambda _task: ToolResult(ok=True),
            find_tools=lambda _query: [],
            load_schemas=lambda _ids, _messages, _call: ToolLoadSelection(),
            should_cancel=lambda: stop,
            clock=lambda: 0.0,
            review_tool_calls=lambda _calls: {},
            continuation_context=_event().context,
            persist_provider_continuation=store.persist_provider_continuation_event,
        ),
    )

    assert outcome.status == RUN_CANCELLED
    assert invoked == ["call-1"]


def test_agent_service_wires_the_exact_dispatch_owner_into_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills leaving AgentService on the legacy continuation persistence path."""
    db, conversation_id, _, _, store, _, _, _ = _started_portable(tmp_path)

    def exercise_real_callback(config, messages, active, deps, **_kwargs):
        deps.persist_provider_continuation(
            ToolBatchReady(deps.continuation_context, _continuation(), None)
        )
        row = db.get_message_by_id("assistant-1")
        assert row is not None
        assert row["assistant_generation_state"] == "continuation_active"
        assert _checkpoint_count(db) == 0
        return agent_service_module.RunOutcome(status="done", steps=[], final_text="")

    monkeypatch.setattr(agent_service_module, "run_agent_loop", exercise_real_callback)
    service = AgentService(
        db=AgentRunsDB(tmp_path / "agent-runs.sqlite", client_id="task16"),
        registry=ToolCatalogRegistry(),
        chat_call=lambda **_kwargs: pytest.fail("provider must be driven by loop"),
        persist_provider_continuation=store.persist_provider_continuation_event,
    )

    _run_id, outcome = service.run_turn(
        conversation_id=conversation_id,
        messages=[{"role": "user", "content": "calculate"}],
        config=_CONFIG,
        api_endpoint="deepseek",
        assistant_message_id="assistant-1",
        continuation_owner_message_id="assistant-1",
    )

    assert outcome.status == "done"


@pytest.mark.parametrize("boundary", ["message_write", "checkpoint_delete"])
def test_local_handoff_statement_failure_runs_zero_tools_and_retains_dispatch(
    tmp_path: Path,
    boundary: str,
) -> None:
    """Kills split handoff writes and any fail-open runtime dispatch."""
    db, _, _, checkpoint, store, session_id, _, _ = _started_portable(tmp_path)
    table = (
        "messages" if boundary == "message_write" else "console_dispatch_checkpoints"
    )
    operation = "UPDATE" if boundary == "message_write" else "DELETE"
    db.get_connection().execute(
        f"CREATE TRIGGER fail_{boundary} BEFORE {operation} ON {table} "
        "BEGIN SELECT RAISE(ABORT, 'handoff fault'); END"
    )
    db.get_connection().commit()
    invoked: list[str] = []
    call = ToolCall(
        "calculator", {"expression": "2+2"}, "call-1", '{"expression":"2+2"}'
    )
    outcome = run_agent_loop(
        _CONFIG,
        [],
        [_TOOL_SCHEMA],
        LoopDeps(
            call_model=lambda _messages, _schemas: ModelTurn(
                tool_calls=(call,),
                assistant_message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression":"2+2"}',
                            },
                        }
                    ],
                },
                provider_continuation=_continuation(),
            ),
            invoke_tool=lambda actual: (
                invoked.append(actual.call_id) or ToolResult(ok=True, content="4")
            ),
            spawn=lambda _task: ToolResult(ok=True),
            find_tools=lambda _query: [],
            load_schemas=lambda _ids, _messages, _call: ToolLoadSelection(),
            should_cancel=lambda: False,
            clock=lambda: 0.0,
            continuation_context=_event().context,
            persist_provider_continuation=store.persist_provider_continuation_event,
        ),
    )

    row = db.get_message_by_id("assistant-1")
    recovery = store.dispatch_recovery_for_session(session_id)
    assert outcome.status == RUN_ERROR
    assert invoked == []
    assert row is not None
    assert (
        row["assistant_generation_state"],
        row["version"],
        row["provider_continuation_json"],
    ) == ("dispatch_started", 2, None)
    assert _checkpoint_count(db) == 1
    assert recovery is not None
    assert recovery.kind is ConsoleDispatchRecoveryKind.DISPATCH_STARTED
    assert recovery.checkpoint == checkpoint
    assert recovery.runtime_active is False
    assert recovery.recovery_needed is True


def test_sqlite_commit_failure_runs_zero_tools_and_retains_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills publication before the real SQLite transaction exits."""
    db, _, repository, checkpoint, store, session_id, _, _ = _started_portable(tmp_path)
    original_transaction = repository.db.transaction

    @contextmanager
    def fail_commit(*, immediate: bool = False):
        connection = db.get_connection()
        connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield connection.cursor()
        except Exception:
            connection.rollback()
            raise
        connection.rollback()
        raise sqlite3.OperationalError("simulated commit failure")

    monkeypatch.setattr(repository.db, "transaction", fail_commit)
    # Store and test share the exact ChatPersistenceService repository.
    store.persistence.console_dispatch_repository = repository

    with pytest.raises(RuntimeError, match="continuation"):
        store.persist_provider_continuation_event(_event())

    row = db.get_message_by_id("assistant-1")
    recovery = store.dispatch_recovery_for_session(session_id)
    assert row is not None
    assert (row["assistant_generation_state"], row["version"]) == (
        "dispatch_started",
        2,
    )
    assert row["provider_continuation_json"] is None
    assert _checkpoint_count(db) == 1
    assert recovery is not None and recovery.checkpoint == checkpoint
    monkeypatch.setattr(repository.db, "transaction", original_transaction)


def test_post_commit_sync_projection_failure_keeps_only_actionable_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills compensation back to dispatch after the separate Sync-v2 barrier."""
    db, _, _, _, store, session_id, _, producer = _started_portable(tmp_path)
    monkeypatch.setattr(
        producer,
        "reconcile_chat_message_intent",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("projection fault")),
    )

    with pytest.raises(RuntimeError, match="Portable sync projection failed"):
        store.persist_provider_continuation_event(_event())

    row = db.get_message_by_id("assistant-1")
    assert row is not None
    assert row["assistant_generation_state"] == "continuation_active"
    assert row["provider_continuation_json"] is not None
    assert _checkpoint_count(db) == 0
    assert store.dispatch_recovery_for_session(session_id) is None
    live = store.get_message("assistant-1")
    assert live.provider_continuation == _continuation()
    assert live.provider_continuation_actions_enabled is True
    assert live.provider_continuation_warning == (
        "Portable sync projection failed; restore local sync state and retry."
    )


@pytest.mark.parametrize("drift", ["provider", "model", "endpoint"])
def test_handoff_rejects_continuation_that_changes_frozen_destination(
    tmp_path: Path,
    drift: str,
) -> None:
    """Kills accepting a provider continuation outside the frozen dispatch target."""
    db, _, _, checkpoint, store, session_id, _, _ = _started_portable(tmp_path)
    if drift == "provider":
        changed = replace(_continuation(), provider="zai", protocol="chat_completions")
    elif drift == "model":
        changed = replace(_continuation(), model="deepseek-other")
    else:
        changed = replace(_continuation(), api_base_url="https://other.example/v1")

    with pytest.raises(RuntimeError, match="continuation"):
        store.persist_provider_continuation_event(replace(_event(), checkpoint=changed))

    assert db.get_message_by_id("assistant-1")["provider_continuation_json"] is None
    assert _checkpoint_count(db) == 1
    recovery = store.dispatch_recovery_for_session(session_id)
    assert recovery is not None and recovery.checkpoint == checkpoint


def test_reasoning_only_complete_continuation_uses_terminal_settlement(
    tmp_path: Path,
) -> None:
    """Kills creating continuation-active recovery for a tool-free final turn."""
    db, conversation_id, repository = _database(tmp_path / "reasoning.sqlite")
    checkpoint = _start(
        repository,
        _insert(db, repository, _moonshot_acceptance(conversation_id)),
    )
    store, session_id, _, _ = _portable_store(db, conversation_id, tmp_path)
    store.publish_durable_dispatch_checkpoint(
        session_id,
        checkpoint,
        in_flight=True,
    )
    complete = ProviderContinuationCheckpoint(
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
                reasoning_blocks=("private",),
                calls=(),
            ),
        ),
    )
    from tldw_chatbook.Agents.agent_models import FinalContinuation

    store.persist_provider_continuation_event(
        FinalContinuation(_event().context, complete, None, "answer")
    )

    row = db.get_message_by_id("assistant-1")
    assert row is not None
    assert (
        row["assistant_generation_state"],
        row["content"],
        parse_provider_continuation_json(row["provider_continuation_json"]),
    ) == ("complete", "answer", complete)
    assert _checkpoint_count(db) == 0
    assert store.dispatch_recovery_for_session(session_id) is None


def _install_legacy_owner(db, conversation_id: str, *, state: str | None) -> None:
    connection = db.get_connection()
    connection.execute("DELETE FROM console_dispatch_checkpoints")
    _raw_semantic_corruption(
        db,
        "UPDATE messages SET provider_continuation_json = ?, "
        "assistant_generation_state = ?, version = 7 WHERE id = 'assistant-1'",
        (dump_provider_continuation_json(_continuation()), state),
    )
    connection.execute(
        "UPDATE conversations SET active_leaf_message_id = 'assistant-1' WHERE id = ?",
        (conversation_id,),
    )
    connection.commit()


@pytest.mark.parametrize("state", [None, "complete", "accepted"])
def test_legacy_continuation_normalizes_and_rebinds_before_actions(
    tmp_path: Path,
    state: str | None,
) -> None:
    """Kills action enablement before the committed normalization proof."""
    db, conversation_id, repository = _database(tmp_path / "legacy.sqlite")
    _insert(db, repository, _deepseek_acceptance(conversation_id))
    _install_legacy_owner(db, conversation_id, state=state)

    store, session_id = _restored_store(db, conversation_id)

    row = db.get_message_by_id("assistant-1")
    owner = store.get_message("assistant-1")
    assert row is not None
    assert (row["assistant_generation_state"], row["version"]) == (
        "continuation_active",
        8,
    )
    assert owner.provider_continuation == _continuation()
    assert owner.provider_continuation_message_version == 8
    assert owner.provider_continuation_actions_enabled is True
    assert store.dispatch_recovery_for_session(session_id) is None
    expected_hash = canonical_payload_hash(
        {
            "assistant_generation_state": "continuation_active",
            "content": "",
            "provider_continuation_json": dump_provider_continuation_json(
                _continuation()
            ),
            "role": "assistant",
        }
    )
    assert (
        db.read_committed_chat_sync_intent(
            message_id="assistant-1",
            message_version=8,
            payload_hash=expected_hash,
        )
        is not None
    )


def test_normalization_rollback_confirms_identical_owner_before_actions(
    tmp_path: Path,
) -> None:
    """Kills trusting a failed normalization without a fresh durable read."""
    db, conversation_id, repository = _database(tmp_path / "rollback.sqlite")
    _insert(db, repository, _deepseek_acceptance(conversation_id))
    _install_legacy_owner(db, conversation_id, state=None)
    connection = db.get_connection()
    connection.execute(
        "CREATE TRIGGER fail_normalization BEFORE UPDATE ON messages "
        "WHEN OLD.id = 'assistant-1' BEGIN SELECT RAISE(ABORT, 'rollback'); END"
    )
    connection.commit()

    store, _session_id = _restored_store(db, conversation_id)

    row = db.get_message_by_id("assistant-1")
    owner = store.get_message("assistant-1")
    assert row is not None
    assert (row["assistant_generation_state"], row["version"]) == (None, 7)
    assert owner.provider_continuation == _continuation()
    assert owner.provider_continuation_message_version == 7
    assert owner.provider_continuation_actions_enabled is True
    assert owner.provider_continuation_warning is not None


def test_normalized_legacy_owner_can_be_discarded_with_rebound_version(
    tmp_path: Path,
) -> None:
    """Kills retaining the pre-normalization version on the Discard handle."""
    db, conversation_id, repository = _database(tmp_path / "discard.sqlite")
    _insert(db, repository, _deepseek_acceptance(conversation_id))
    _install_legacy_owner(db, conversation_id, state="accepted")
    store, session_id = _restored_store(db, conversation_id)
    owner = store.get_message("assistant-1")

    assert owner.provider_continuation_actions_enabled is True
    assert owner.provider_continuation_message_version == 8
    assert store.discard_provider_continuation(
        owner.id,
        expected_message_version=8,
    )
    assert db.get_message_by_id("assistant-1") is None
    assert store.provider_continuation_recovery_message(session_id) is None


@pytest.mark.parametrize("race", ["version", "deleted", "replacement", "invalid"])
def test_normalization_conflict_rereads_and_never_leaves_stale_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    race: str,
) -> None:
    """Kills stale Resume/Discard after version/deletion/identity conflicts."""
    db, conversation_id, repository = _database(tmp_path / f"race-{race}.sqlite")
    _insert(db, repository, _deepseek_acceptance(conversation_id))
    _install_legacy_owner(db, conversation_id, state=None)
    real = repository.normalize_provider_continuation_owner

    def mutate_then_normalize(**kwargs):
        connection = db.get_connection()
        if race == "version":
            _raw_semantic_corruption(
                db, "UPDATE messages SET version = 8 WHERE id = 'assistant-1'"
            )
        elif race == "deleted":
            _raw_semantic_corruption(
                db,
                "UPDATE messages SET deleted = 1, version = 8 WHERE id = 'assistant-1'",
            )
        elif race == "replacement":
            replacement = replace(_continuation(), checkpoint_revision=2)
            _raw_semantic_corruption(
                db,
                "UPDATE messages SET provider_continuation_json = ?, version = 8 "
                "WHERE id = 'assistant-1'",
                (dump_provider_continuation_json(replacement),),
            )
        else:
            _raw_semantic_corruption(
                db,
                "UPDATE messages SET provider_continuation_json = '{\"bad\":true}', "
                "version = 8 WHERE id = 'assistant-1'",
            )
        connection.commit()
        monkeypatch.setattr(repository, "normalize_provider_continuation_owner", real)
        return real(**kwargs)

    monkeypatch.setattr(
        repository, "normalize_provider_continuation_owner", mutate_then_normalize
    )
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService

    persistence = ChatPersistenceService(db)
    persistence.console_dispatch_repository = repository
    rows = db.get_messages_for_conversation(conversation_id, limit=100)
    from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage

    nodes = []
    for row in rows:
        node = ConsoleChatMessage(
            id=str(row["id"]),
            role=ConsoleMessageRole(str(row["role"])),
            content=str(row.get("content") or ""),
            persisted_message_id=str(row["id"]),
            parent_message_id=row.get("parent_message_id"),
        )
        node.assistant_generation_state = row.get("assistant_generation_state")
        nodes.append(node)
    store = ConsoleChatStore(persistence=persistence)
    store.restore_persisted_session(
        title="race",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=nodes,
        active_leaf_persisted_id="assistant-1",
    )

    owner = store.get_message("assistant-1")
    if race == "version":
        assert owner.provider_continuation_actions_enabled is True
        assert owner.provider_continuation_message_version == 9
    else:
        assert owner.provider_continuation_actions_enabled is False
        assert owner.provider_continuation_message_version is None


def test_dual_owner_precedence_returns_continuation_and_no_dispatch_actions(
    tmp_path: Path,
) -> None:
    """Kills exposing two Retry/Discard owner surfaces after reconciliation."""
    db, conversation_id, repository = _database(tmp_path / "dual.sqlite")
    _insert(db, repository, _deepseek_acceptance(conversation_id))
    _raw_semantic_corruption(
        db,
        "UPDATE messages SET provider_continuation_json = ? WHERE id = 'assistant-1'",
        (dump_provider_continuation_json(_continuation()),),
    )
    db.get_connection().commit()

    recovery = repository.reconcile_for_session(conversation_id)

    assert recovery is not None
    assert recovery.kind is ConsoleDispatchRecoveryKind.CONTINUATION
    assert recovery.actions == ()
    assert _checkpoint_count(db) == 0
    row = db.get_message_by_id("assistant-1")
    assert row is not None
    assert row["assistant_generation_state"] == "continuation_active"


@pytest.mark.parametrize("guard", ["user_version", "assistant_version", "deleted"])
def test_handoff_expected_message_and_delete_guards_fail_closed(
    tmp_path: Path,
    guard: str,
) -> None:
    """Kills removal of either message-version or deleted=0 handoff predicate."""
    db, _, repository, checkpoint, store, session_id, _, _ = _started_portable(tmp_path)
    connection = db.get_connection()
    if guard == "user_version":
        _raw_semantic_corruption(
            db, "UPDATE messages SET version = 2 WHERE id = 'user-1'"
        )
    elif guard == "assistant_version":
        _raw_semantic_corruption(
            db, "UPDATE messages SET version = 3 WHERE id = 'assistant-1'"
        )
    else:
        _raw_semantic_corruption(
            db, "UPDATE messages SET deleted = 1 WHERE id = 'assistant-1'"
        )
    connection.commit()

    with pytest.raises(RuntimeError, match="[Cc]ontinuation"):
        store.persist_provider_continuation_event(_event())

    assert _checkpoint_count(db) == 1
    recovery = store.dispatch_recovery_for_session(session_id)
    assert recovery is not None and recovery.checkpoint == checkpoint
    result = repository.read_for_session(checkpoint.conversation_id)
    assert result.status in {
        ConsoleDispatchResultStatus.COMMITTED,
        ConsoleDispatchResultStatus.QUARANTINED,
    }
