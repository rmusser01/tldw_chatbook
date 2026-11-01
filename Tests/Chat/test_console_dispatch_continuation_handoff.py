"""Task 16: exclusive dispatch-to-continuation ownership and rebind."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest

from Tests.Chat.test_console_dispatch_recovery import (
    _acceptance,
    _database,
    _insert,
    _raw_semantic_corruption,
    _restored_store,
    _start,
)
from tldw_chatbook.Agents import agent_service as agent_service_module
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
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop
from tldw_chatbook.Agents.agent_service import AgentService
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
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationCall,
    ContinuationRestoreTarget,
    ContinuationResult,
    ContinuationRound,
    ProviderContinuationCheckpoint,
    dump_provider_continuation_json,
    parse_provider_continuation_json,
)
from tldw_chatbook.Chat.thinking_blocks import (
    DisplayableThinkingBlock,
    ThinkingEnvelope,
    dump_thinking_blocks_json,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Sync_Interop.chat_outbox_producer import (
    ChatSyncV2OutboxProducer,
)
from tldw_chatbook.Sync_Interop.crypto import generate_dataset_key
from tldw_chatbook.Sync_Interop.hashing import canonical_payload_hash
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository

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
        schema_version=3,
        selected_provider="deepseek",
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
            endpoint_identity="https://api.deepseek.com",
            continuation_provider="deepseek",
            continuation_protocol="responses",
            continuation_base_url="https://api.deepseek.com/v1",
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
            endpoint_identity="https://api.moonshot.ai",
            continuation_provider="moonshot",
            continuation_protocol="chat_completions",
            continuation_base_url="https://api.moonshot.ai/v1",
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

    display_revision = store.display_projection_revision(session_id)
    store.persist_provider_continuation_event(
        _event(), continuation_target=_DEEPSEEK_RUN_TARGET
    )
    assert store.display_projection_revision(session_id) > display_revision

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
            persist_provider_continuation=partial(
                store.persist_provider_continuation_event,
                continuation_target=_DEEPSEEK_RUN_TARGET,
            ),
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
        persist_provider_continuation=partial(
            store.persist_provider_continuation_event,
            continuation_target=_DEEPSEEK_RUN_TARGET,
        ),
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
            persist_provider_continuation=partial(
                store.persist_provider_continuation_event,
                continuation_target=_DEEPSEEK_RUN_TARGET,
            ),
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
        store.persist_provider_continuation_event(
            _event(), continuation_target=_DEEPSEEK_RUN_TARGET
        )

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
        store.persist_provider_continuation_event(
            _event(), continuation_target=_DEEPSEEK_RUN_TARGET
        )

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
        store.persist_provider_continuation_event(
            replace(_event(), checkpoint=changed),
            continuation_target=_DEEPSEEK_RUN_TARGET,
        )

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
        schema_version=3,
        selected_provider="moonshot",
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
        FinalContinuation(_event().context, complete, None, "answer"),
        continuation_target=_MOONSHOT_RUN_TARGET,
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
        store.persist_provider_continuation_event(
            _event(), continuation_target=_DEEPSEEK_RUN_TARGET
        )

    assert _checkpoint_count(db) == 1
    recovery = store.dispatch_recovery_for_session(session_id)
    assert recovery is not None and recovery.checkpoint == checkpoint
    result = repository.read_for_session(checkpoint.conversation_id)
    assert result.status in {
        ConsoleDispatchResultStatus.COMMITTED,
        ConsoleDispatchResultStatus.QUARANTINED,
    }


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "native_tool",
    [
        pytest.param(True, id="injected_resolution_native"),
        pytest.param(False, id="injected_resolution_text"),
        pytest.param("resume", id="injected_resolution_reopen"),
        pytest.param("resume_again", id="restored_then_another_native_round"),
        pytest.param(
            "normal",
            id="normal_configured_resolver",
            marks=pytest.mark.owned_http_client,
        ),
    ],
)
async def test_custom_selected_identity_survives_actual_console_handoff(
    tmp_path, monkeypatch, native_tool
):
    """Normal configured send plus isolated injected-resolution A/B controls."""
    resume_again = native_tool == "resume_again"
    if resume_again:
        native_tool = "resume"
    normal_resolver = native_tool == "normal"
    if normal_resolver:
        native_tool = True
    import io
    import json
    import socket

    import urllib3

    from Tests.console_provider_doubles import with_destination
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderResolution,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Utils.token_counter import ContextWindowResolution

    selected = "custom-ep:identity-a"
    model = "identity-model"
    requests = []

    class Gateway(ConsoleProviderGateway):
        async def resolve_for_send(self, selection):
            return with_destination(
                ConsoleProviderResolution(
                    provider=selected,
                    selected_provider=selected,
                    execution_key="custom-hosted",
                    base_url="http://127.0.0.1:8010/v1",
                    model=model,
                    ready=True,
                    readiness_key="custom-hosted",
                    max_tokens=1024,
                    streaming=True,
                    context_window=ContextWindowResolution(32768, "fixture", True),
                    continuation_protocol="chat_completions",
                )
            )

    if normal_resolver:
        profile = {
            "custom_endpoints": {
                "identity-a": {
                    "display_name": "Identity fixture",
                    "family": "openai_compatible",
                    "base_url": "http://127.0.0.1:8010/v1",
                    "models": [model],
                }
            },
        }
        # Only configuration input is supplied: resolution/readiness/classification
        # and context/admission use their unmodified production implementations.
        gateway = ConsoleProviderGateway(config_provider=lambda: profile, environ={})
    else:
        gateway = Gateway(config_provider=dict, environ={})
    chat = CharactersRAGDB(str(tmp_path / "chat.db"), client_id="identity")
    store = ConsoleChatStore(persistence=ChatPersistenceService(chat))
    runs = AgentRunsDB(tmp_path / "runs.db", client_id="identity")
    bridge = ConsoleAgentBridge(
        agent_runs_db=runs, store=store, provider_gateway=gateway
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider=selected,
        model=model,
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )
    session = store.ensure_session()
    if native_tool == "resume":
        persist = store.persist_provider_continuation_event

        def interrupt_after_commit(event, **kwargs):
            persist(event, **kwargs)
            if isinstance(event, ToolBatchReady):
                raise RuntimeError("simulated interruption after durable pending batch")

        monkeypatch.setattr(
            store, "persist_provider_continuation_event", interrupt_after_commit
        )

    def denied(*args, **kwargs):
        raise AssertionError("network prohibited")

    def physical_response(pool, connection, method, url, *args, **kwargs):
        request = json.loads(kwargs.get("body", args[0] if args else None))
        assert method == "POST"
        assert pool.host == "127.0.0.1" and pool.port == 8010
        assert url == "/v1/chat/completions"
        requests.append(request)
        if normal_resolver and len(requests) == 1:
            row = (
                chat.get_connection()
                .execute(
                    "SELECT resolved_destination_json FROM console_dispatch_checkpoints"
                )
                .fetchone()
            )
            frozen = json.loads(row[0])
            assert frozen == {
                "schema_version": 2,
                "provider": selected,
                "model": model,
                "endpoint_identity": "http://127.0.0.1:8010",
                "egress_class": "on_device",
                "continuation_provider": "custom-hosted",
                "continuation_protocol": "chat_completions",
                "continuation_base_url": "http://127.0.0.1:8010/v1/chat/completions",
            }
        first_tool = native_tool and (
            len(requests) == 1 or (resume_again and len(requests) == 2)
        )
        if len(requests) == 2:
            # Reading the actual owner here proves commit precedes replay.
            row = next(
                row
                for row in chat.get_messages_for_conversation(
                    store._session_or_raise(session.id).persisted_conversation_id,
                    limit=100,
                )
                if row.get("provider_continuation_json")
            )
            private = json.loads(row["provider_continuation_json"])
            assert private["schema_version"] == 3
            assert private["provider"] == "custom-hosted"
            assert private["protocol"] == "chat_completions"
            assert private["model"] == model
            live = store.get_message(row["id"]).provider_continuation
            assert live.schema_version == 3 and live.selected_provider == selected
            assert live.rounds[0].calls[0].state == "completed"
            assert private["selected_provider"] == selected
            assert private["api_base_url"] == "http://127.0.0.1:8010/v1"
            assert private["rounds"][0]["calls"][0]["state"] == "completed"
            calls = [
                call
                for message in request["messages"]
                for call in message.get("tool_calls", [])
            ]
            results = [
                message
                for message in request["messages"]
                if message.get("role") == "tool"
            ]
            assert calls == [
                {
                    "id": "call_check",
                    "type": "function",
                    "function": {"name": "check_agents", "arguments": "{}"},
                }
            ]
            assert len(results) == 1
            assert results[0]["tool_call_id"] == "call_check"
            assert results[0]["content"] == private["rounds"][0]["calls"][0]["result"]
            assert store.dispatch_recovery_for_session(session.id) is None
        if len(requests) == 3:
            assert resume_again
            expected_ids = ["call_check", "call_check_again"]
            assert [
                call["id"]
                for message in request["messages"]
                for call in message.get("tool_calls", [])
            ] == expected_ids
            assert [
                message["tool_call_id"]
                for message in request["messages"]
                if message["role"] == "tool"
            ] == expected_ids
        delta = (
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_check_again"
                        if len(requests) == 2
                        else "call_check",
                        "type": "function",
                        "function": {"name": "check_agents", "arguments": "{}"},
                    }
                ]
            }
            if first_tool
            else {"content": "Finished."}
        )
        chunks = [
            {
                "id": "fixture",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            },
            {
                "id": "fixture",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls" if first_tool else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        ]
        body = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
        return urllib3.response.HTTPResponse(
            body=io.BytesIO((body + "data: [DONE]\n\n").encode()),
            status=200,
            headers={"Content-Type": "text/event-stream"},
            preload_content=False,
        )

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    monkeypatch.setattr(
        urllib3.connectionpool.HTTPConnectionPool, "_make_request", physical_response
    )
    metadata_requests = []
    if normal_resolver:
        import httpx

        async def metadata_response(transport, request):
            metadata_requests.append((request.method, str(request.url)))
            assert request.method == "GET"
            assert request.url.host == "127.0.0.1" and request.url.port == 8010
            assert request.url.path == "/v1/models"
            return httpx.Response(
                200, json={"data": [{"id": model, "context_length": 32768}]}
            )

        monkeypatch.setattr(
            httpx.AsyncHTTPTransport, "handle_async_request", metadata_response
        )
    try:
        result = await controller.submit_draft(
            "Inspect active agents.", session_id=session.id
        )
        assert result.accepted and result.provider_started
        if normal_resolver:
            import asyncio

            await asyncio.gather(*tuple(gateway._context_window_refreshes))
            assert metadata_requests == [("GET", "http://127.0.0.1:8010/v1/models")]
        assert len(requests) == (2 if native_tool is True else 1)
        owners = [
            message
            for message in store.messages_for_session(session.id)
            if message.role is ConsoleMessageRole.ASSISTANT
        ]
        conversation_id = store._session_or_raise(session.id).persisted_conversation_id
        owner_id = owners[-1].id
        if native_tool != "resume":
            assert owners[-1].content == "Finished."
        if native_tool:
            assert owners[-1].provider_continuation.schema_version == 3
            assert owners[-1].provider_continuation.selected_provider == selected
            assert owners[-1].provider_continuation.state == (
                "active" if native_tool == "resume" else "complete"
            )
            if native_tool == "resume":
                assert (
                    owners[-1].provider_continuation.rounds[0].calls[0].state
                    == "pending"
                )
        else:
            assert owners[-1].provider_continuation is None
        assert store.dispatch_recovery_for_session(session.id) is None
    finally:
        await controller.shutdown()
        await gateway.aclose()
        with chat.quiesce_connections(timeout_seconds=2):
            assert chat.registered_connection_count() == 0
        runs.close()

    if not native_tool or normal_resolver:
        return
    chat = CharactersRAGDB(str(tmp_path / "chat.db"), client_id="identity-reopened")
    store, restored_id = _restored_store(chat, conversation_id)
    session = store._session_or_raise(restored_id)
    assert _checkpoint_count(chat) == 0
    runs = AgentRunsDB(tmp_path / "runs-reopened.db", client_id="identity-reopened")
    gateway = Gateway(config_provider=dict, environ={})
    bridge = ConsoleAgentBridge(
        agent_runs_db=runs, store=store, provider_gateway=gateway
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider=selected,
        model=model,
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )
    try:
        owner = store.get_message(owner_id)
        assert owner.provider_continuation.selected_provider == "custom-ep:identity-a"
        selected = "custom-ep:identity-b"
        resolution_b = await gateway.resolve_for_send(None)
        if native_tool == "resume":
            version = owner.provider_continuation_message_version
            assert owner.provider_continuation_actions_enabled, (
                owner.assistant_generation_state,
                owner.generation_projection_quarantined,
                version,
            )
            assert version == store.persistence.get_message_version(owner_id)
            assert not await controller.recover_provider_continuation(
                "resume", owner_id, version
            )
            assert len(requests) == 1
            assert owner.provider_continuation.rounds[0].calls[0].state == "pending"
            selected = "custom-ep:identity-a"
            assert await controller.recover_provider_continuation(
                "resume", owner_id, version
            ), store.get_message(owner_id).provider_continuation_warning
            assert len(requests) == (3 if resume_again else 2)
            final = store.get_message(owner_id).provider_continuation
            assert final.state == "complete" and final.schema_version == 3
            if resume_again:
                assert [
                    call.call_id for round_ in final.rounds for call in round_.calls
                ] == ["call_check", "call_check_again"]
        else:
            sidecar_b, _ = controller._provider_continuation_history_for_resolution(
                session.id, resolution_b
            )
            assert sidecar_b == ()
            assert owner.content == "Finished."
            selected = "custom-ep:identity-a"
            resolution_a = await gateway.resolve_for_send(None)
            sidecar_a, target_a = (
                controller._provider_continuation_history_for_resolution(
                    session.id, resolution_a
                )
            )
            assert len(sidecar_a) == 1
            prepared = gateway.prepare_chat_request(
                resolution_a,
                [
                    {"role": "assistant", "content": owner.content, "_owner": owner.id},
                    {"role": "user", "content": "Continue."},
                ],
                continuation_sidecar=sidecar_a,
                continuation_target=target_a,
                continuation_owner_key="_owner",
            )
            forwarded = gateway._chat_api_kwargs_from_prepared(resolution_a, prepared)
            assert forwarded["provider_continuations"][0] == owner.provider_continuation
            assert (
                forwarded["provider_continuations"][0].rounds[0].calls[0].result
                is not None
            )
    finally:
        await controller.shutdown()
        await gateway.aclose()
        with chat.quiesce_connections(timeout_seconds=2):
            assert chat.registered_connection_count() == 0
        runs.close()


_DEEPSEEK_RUN_TARGET = ContinuationRestoreTarget(
    provider="deepseek",
    protocol="responses",
    model="deepseek-v4-flash",
    api_base_url="https://api.deepseek.com/v1",
    selected_provider="deepseek",
)
_MOONSHOT_RUN_TARGET = ContinuationRestoreTarget(
    provider="moonshot",
    protocol="chat_completions",
    model="kimi-k3",
    api_base_url="https://api.moonshot.ai/v1",
    selected_provider="moonshot",
)


@pytest.mark.parametrize(
    "drift", ["raw", "model", "base", "protocol", "legacy_destination"]
)
def test_repository_independently_refuses_changed_frozen_native_identity(
    tmp_path, drift
):
    db, conversation_id, repository = _database(tmp_path / "identity-refusal.sqlite")
    acceptance = _deepseek_acceptance(conversation_id)
    if drift == "legacy_destination":
        acceptance = replace(
            acceptance,
            resolved_destination=replace(
                acceptance.resolved_destination,
                continuation_provider=None,
                continuation_protocol=None,
                continuation_base_url=None,
            ),
        )
    started = _start(repository, _insert(db, repository, acceptance))
    candidate = _continuation()
    changes = {
        "raw": {"selected_provider": "custom-ep:other"},
        "model": {"model": "other-model"},
        "base": {"api_base_url": "https://api.deepseek.com/v2"},
        "protocol": {"protocol": "chat_completions"},
        "legacy_destination": {},
    }
    candidate = replace(candidate, **changes[drift])
    try:
        result = repository.handoff_to_provider_continuation(
            ConsoleContinuationHandoff(
                assistant_message_id=started.assistant_message_id,
                expected_checkpoint_revision=started.checkpoint_revision,
                expected_user_message_version=started.user_message_version,
                expected_assistant_message_version=started.assistant_message_version,
                provider_continuation_json=dump_provider_continuation_json(candidate),
            )
        )
        assert result.status is ConsoleDispatchResultStatus.CONFLICT
        assert _checkpoint_count(db) == 1
        assert (
            db.get_message_by_id(started.assistant_message_id)[
                "provider_continuation_json"
            ]
            is None
        )
    finally:
        db.close_connection()


def test_store_rejects_missing_or_foreign_run_authority_before_signing(tmp_path):
    db, _, _, _, store, _, _, _ = _started_portable(tmp_path)
    sealed = []

    def finalizer(*args):
        sealed.append(True)
        raise AssertionError("must refuse before signing")

    for target in (
        None,
        replace(_DEEPSEEK_RUN_TARGET, selected_provider="custom-ep:other"),
    ):
        with pytest.raises(RuntimeError):
            store.persist_provider_continuation_event(
                _event(), continuation_target=target, checkpoint_finalizer=finalizer
            )
    assert sealed == []
    assert _checkpoint_count(db) == 1
    store.persist_provider_continuation_event(
        _event(), continuation_target=_DEEPSEEK_RUN_TARGET
    )
    current = store.get_message("assistant-1").provider_continuation
    from tldw_chatbook.Agents.agent_models import FinalContinuation

    downgraded = replace(
        current,
        schema_version=1,
        selected_provider=None,
        checkpoint_revision=2,
        state="complete",
        rounds=(
            replace(
                current.rounds[0],
                calls=(
                    replace(
                        current.rounds[0].calls[0],
                        state="completed",
                        result=ContinuationResult("4"),
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(RuntimeError, match="binding cannot be erased"):
        store.persist_provider_continuation_event(
            FinalContinuation(_event().context, downgraded, 1, "done"),
            continuation_target=_DEEPSEEK_RUN_TARGET,
        )
    assert store.get_message("assistant-1").provider_continuation == current


@pytest.mark.bootstrap_profile
@pytest.mark.owned_http_client
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("round_count", "batch_size"), [(1, 1), (2, 1), (3, 1), (2, 2)]
)
@pytest.mark.parametrize("capture_on", [False, True], ids=["capture-off", "capture-on"])
async def test_normal_console_multiple_native_rounds(
    tmp_path,
    monkeypatch,
    round_count,
    batch_size,
    capture_on,
    fault=None,
    subsequent_turn=False,
    replay_fault=None,
    known_credential="",
):
    import io
    import json
    import socket
    from unittest.mock import patch

    import httpx
    import urllib3

    from tldw_chatbook.Chat import console_agent_bridge, console_chat_controller
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_prepared_request import CONTINUATION_OWNER_KEY
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
    from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
    from tldw_chatbook.Chat.console_trace_provenance import ConsoleTraceCaptureMode
    from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    def denied(*args, **kwargs):
        raise AssertionError("network prohibited")

    retry_mode = fault in {"retry", "retry_changed"}
    later_turn = False
    trace_errors = []
    executed = []
    retry_refusals = []
    wire = []
    frames = []
    metadata = []
    selected = "custom-ep:multiround"
    model = "synthetic-qwen"
    profile = {
        "custom_endpoints": {
            "multiround": {
                "display_name": "Multiround fixture",
                "family": "openai_compatible",
                "base_url": "http://127.0.0.1:8010/v1",
                "models": [model],
            }
        }
    }
    if known_credential:
        profile["custom_endpoints"]["multiround"]["api_key"] = known_credential
    chat = CharactersRAGDB(str(tmp_path / "chat.db"), client_id="multiround")
    shadows = []
    if capture_on and not subsequent_turn:
        original_preparation = console_chat_controller.ConsoleTurnPreparation

        def preparation(**kwargs):
            kwargs["capture_mode"] = ConsoleTraceCaptureMode.CAPTURE_ON
            return original_preparation(**kwargs)

        monkeypatch.setattr(
            console_chat_controller, "ConsoleTurnPreparation", preparation
        )
    gateway = ConsoleProviderGateway(
        config_provider=lambda: profile,
        environ={},
        trace_call_boundary_factory=ConsoleTraceBoundaryFactory(chat)
        if capture_on
        else None,
        trace_shadow_sink=shadows.append,
    )
    store = ConsoleChatStore(persistence=ChatPersistenceService(chat))
    runs = AgentRunsDB(tmp_path / "runs.db", client_id="diagnosis")
    bridge = ConsoleAgentBridge(
        agent_runs_db=runs, store=store, provider_gateway=gateway
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider=selected,
        model=model,
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )
    session = store.ensure_session()
    fault_snapshots = []
    if retry_mode or subsequent_turn:
        from tldw_chatbook.Agents.agent_service import AgentService

        original_invoke = AgentService._make_invoke_tool

        def observed_invoke(service, *args, **kwargs):
            invoke = original_invoke(service, *args, **kwargs)

            def counted(call):
                executed.append(call.call_id)
                result = invoke(call)
                if known_credential:
                    from dataclasses import replace

                    result = replace(
                        result, content="Read file value: " + known_credential
                    )
                return result

            return counted

        monkeypatch.setattr(AgentService, "_make_invoke_tool", observed_invoke)
    if fault == "private_projection":
        from tldw_chatbook.Agents.agent_models import ToolRecordProjection
        from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry

        original_project = ToolCatalogRegistry.project_tool_record

        def private_project(registry, audience, call, result):
            if audience == "continuation":
                return ToolRecordProjection(
                    arguments={"safe_reference": call.call_id},
                    content="projected continuation result",
                    error="projected error",
                    ok=None if result is None else result.ok,
                )
            return original_project(registry, audience, call, result)

        monkeypatch.setattr(
            ToolCatalogRegistry,
            "has_tool_record_projection",
            lambda registry, name: True,
        )
        monkeypatch.setattr(ToolCatalogRegistry, "project_tool_record", private_project)
    if fault is not None:
        from dataclasses import replace

        from tldw_chatbook.Chat.console_prepared_request import thaw_json
        from tldw_chatbook.Chat.console_trace_service import ConsoleTraceService

        tables = (
            "console_trace_surface_nodes",
            "console_trace_surface_replacements",
            "console_trace_artifacts",
            "console_trace_request_headers",
        )

        def snapshot(cursor):
            return tuple(
                cursor.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]
                for table in tables
            )

        original_factory = ConsoleTraceBoundaryFactory.__call__

        def faulty_factory(factory, request, resolution, route):
            if len(wire) == 2:
                with chat.transaction() as cursor:
                    fault_snapshots.append(snapshot(cursor))
                if fault == "cold":
                    factory.service = ConsoleTraceService(factory.repository)
                elif fault == "forged":
                    request = replace(
                        request,
                        live_native_custody=replace(
                            request.live_native_custody, issuer=object()
                        ),
                    )
                elif fault == "prefix":
                    rows = thaw_json(request.messages_payload)
                    next(row for row in rows if row.get("tool_call_id") == "call_1_0")[
                        "content"
                    ] = "changed history"
                    request = replace(request, messages_payload=tuple(rows))
                elif fault == "target":
                    resolution = replace(
                        resolution, selected_provider="custom-ep:foreign"
                    )
                elif fault == "old_round":
                    group = request.continuation_groups[-1]
                    changed = replace(
                        group.checkpoint,
                        rounds=(
                            replace(group.rounds[0], reasoning_blocks=("changed",)),
                        )
                        + group.rounds[1:],
                    )
                    request = replace(
                        request,
                        continuation_groups=(
                            replace(group, checkpoint=changed, rounds=changed.rounds),
                        ),
                    )
            return original_factory(factory, request, resolution, route)

        monkeypatch.setattr(ConsoleTraceBoundaryFactory, "__call__", faulty_factory)
        original_surface = ConsoleTraceService.prepare_current_surface_delta

        def changed_retry_surface(service, cursor, **kwargs):
            if fault == "retry_changed" and len(wire) == 4:
                fault_snapshots.append(snapshot(cursor))
                values = list(kwargs["values"])
                index = next(
                    index
                    for index, row in enumerate(values)
                    if hasattr(row, "get") and row.get("tool_call_id") == "call_1_0"
                )
                values[index] = {**values[index], "content": "changed retry result"}
                kwargs["values"] = tuple(values)
                try:
                    return original_surface(service, cursor, **kwargs)
                except ValueError as exc:
                    retry_refusals.append(str(exc))
                    raise
            return original_surface(service, cursor, **kwargs)

        monkeypatch.setattr(
            ConsoleTraceService, "prepare_current_surface_delta", changed_retry_surface
        )
        original_bind = ConsoleTraceService.bind_and_mark_dispatch

        def faulty_bind(service, database, **kwargs):
            delta = kwargs["surface_delta"]
            if delta.live_native is not None:
                if fault == "stale":
                    delta.live_native_custody.issuer._live_native_custody = None
                elif fault == "version":
                    with chat.transaction() as cursor:
                        cursor.execute(
                            "UPDATE messages SET version = version + 1 WHERE id = ?",
                            (delta.live_native_custody.owner_message_id,),
                        )
            if delta.live_native is not None and fault == "readback":
                from contextlib import contextmanager

                original_transaction = database.transaction
                lost = False

                @contextmanager
                def transaction(*args, **options):
                    nonlocal lost
                    with original_transaction(*args, **options) as cursor:
                        yield cursor
                    if not lost:
                        lost = True
                        raise OSError("commit acknowledgment lost")

                with monkeypatch.context() as commit_patch:
                    commit_patch.setattr(database, "transaction", transaction)
                    return original_bind(service, database, **kwargs)
            return original_bind(service, database, **kwargs)

        monkeypatch.setattr(ConsoleTraceService, "bind_and_mark_dispatch", faulty_bind)
        original_append = ConsoleTraceService._append_descriptor

        def faulty_append(service, cursor, **kwargs):
            if (
                fault == "rollback"
                and kwargs["tail"] is not None
                and kwargs["tail"].component_kind == "continuation_delta"
            ):
                assert (
                    cursor.execute(
                        "SELECT COUNT(*) FROM console_trace_surface_replacements"
                    ).fetchone()[0]
                    == 1
                )
                raise OSError("injected append failure after replacement")
            return original_append(service, cursor, **kwargs)

        monkeypatch.setattr(ConsoleTraceService, "_append_descriptor", faulty_append)
    native_builder = console_agent_bridge.build_console_request

    def observed_builder(messages, *args, **kwargs):
        groups = kwargs.get("continuation_groups", ())
        row = {
            "groups": [
                {
                    "owner": g.owner_message_id,
                    "rounds": len(g.rounds),
                    "revision": g.checkpoint.checkpoint_revision,
                }
                for g in groups
            ],
            "rows": [
                {
                    "role": r.get("role"),
                    "owner": r.get(CONTINUATION_OWNER_KEY),
                    "call_ids": [c.get("id") for c in r.get("tool_calls", [])],
                    "tool_call_id": r.get("tool_call_id"),
                }
                for r in messages
            ],
        }
        frames.append(row)
        prepared = native_builder(messages, *args, **kwargs)
        if groups:
            assert len(
                [message for message in messages if message.get(CONTINUATION_OWNER_KEY)]
            ) == len(groups)
            assert len(groups) == (
                2
                if later_turn and any(g.checkpoint.state != "complete" for g in groups)
                else 1
            )
            assert (prepared.provenance is not None) is capture_on
        return prepared

    monkeypatch.setattr(console_agent_bridge, "build_console_request", observed_builder)

    async def fake_metadata(transport, request):
        assert (
            request.method == "GET"
            and request.url.host == "127.0.0.1"
            and request.url.path == "/v1/models"
        )
        metadata.append(str(request.url))
        return httpx.Response(
            200, json={"data": [{"id": model, "context_length": 32768}]}
        )

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", fake_metadata)

    def fake_native(pool, connection, method, url, *args, **kwargs):
        request = json.loads(kwargs.get("body", args[0] if args else None))
        wire.append(request)
        assert (
            method == "POST"
            and pool.host == "127.0.0.1"
            and pool.port == 8010
            and url == "/v1/chat/completions"
        )
        index = len(wire)
        if capture_on:
            expected_shadows = (
                (2 if index <= 4 else index - 2) if retry_mode and index >= 2 else index
            )
            assert len(shadows) == expected_shadows and shadows[-1].available
            assert (
                shadows[-1].boundary_kwargs["api_base_url"]
                == "http://127.0.0.1:8010/v1"
            )
            with chat.transaction() as cursor:
                header = cursor.execute(
                    "SELECT provider_name, model_name, endpoint_identity "
                    "FROM console_trace_request_headers ORDER BY rowid DESC LIMIT 1"
                ).fetchone()
                assert tuple(header) == (
                    "custom-hosted",
                    model,
                    "http://127.0.0.1:8010/v1/chat/completions",
                )
                assert (
                    cursor.execute(
                        "SELECT COUNT(*) FROM console_trace_surface_nodes WHERE reference_kind = 'omission'"
                    ).fetchone()[0]
                    == 0
                )
        if retry_mode and 2 <= index <= 4:
            assert request == wire[1]
            return urllib3.response.HTTPResponse(
                body=io.BytesIO(b'{"error":{"message":"offline transient"}}'),
                status=503,
                headers={"Content-Type": "application/json"},
                preload_content=False,
            )
        if retry_mode and index > 4:
            if index == 5:
                assert request == wire[1]
            index -= 3
        if later_turn:
            index -= round_count + 1
        assert index <= round_count + 1
        prefix = "next_" if later_turn else ""
        previous_ids = (
            [
                f"call_{round_}_{call}"
                for round_ in range(1, round_count + 1)
                for call in range(batch_size)
            ]
            if later_turn
            else []
        ) + [
            f"{prefix}call_{round_}_{call}"
            for round_ in range(1, index)
            for call in range(batch_size)
        ]
        assert [
            call["id"]
            for message in request["messages"]
            for call in message.get("tool_calls", [])
        ] == previous_ids
        assert [
            message["tool_call_id"]
            for message in request["messages"]
            if message["role"] == "tool"
        ] == previous_ids
        tool = (
            ["check_agents", "get_current_datetime", "check_agents"][index - 1]
            if index <= round_count
            else None
        )
        if (retry_mode or subsequent_turn) and tool:
            tool = "get_current_datetime"
        delta = (
            {
                "tool_calls": [
                    {
                        "index": call,
                        "id": f"{prefix}call_{index}_{call}",
                        "type": "function",
                        "function": {"name": tool, "arguments": "{}"},
                    }
                    for call in range(batch_size)
                ]
            }
            if tool
            else {
                "content": "Synthetic later final."
                if later_turn
                else "Synthetic final."
            }
        )
        events = [
            {
                "id": "synthetic",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            },
            {
                "id": "synthetic",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls" if tool else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        ]
        body = (
            "".join("data: " + json.dumps(e) + "\n\n" for e in events)
            + "data: [DONE]\n\n"
        )
        return urllib3.response.HTTPResponse(
            body=io.BytesIO(body.encode()),
            status=200,
            headers={"Content-Type": "text/event-stream"},
            preload_content=False,
        )

    if subsequent_turn:
        from copy import deepcopy

        from tldw_chatbook.Chat.console_prepared_request import thaw_json
        from tldw_chatbook.Chat.console_trace_service import ConsoleTraceService

        original_prepare = ConsoleTraceService.prepare_current_surface_delta
        replay_snapshots = []

        def replay_snapshot(cursor):
            return tuple(
                cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in (
                    "console_trace_calls",
                    "console_trace_surface_nodes",
                    "console_trace_surface_replacements",
                )
            )

        def probe_replay(service, cursor, **kwargs):
            advance = replay_fault in {
                "history_result",
                "history_order",
                "history_attachment",
                "foreign_parent",
                "stale_parent",
            }
            if (
                not replay_fault
                or not later_turn
                or len(wire) != round_count + 1 + int(advance)
            ):
                return original_prepare(service, cursor, **kwargs)
            replay_snapshots.append(replay_snapshot(cursor))
            values = thaw_json(kwargs["values"])
            if replay_fault in {"history_attachment", "completed_attachment"}:
                attachment = next(
                    value
                    for value in values
                    if isinstance(value, dict) and value.get("state") == "complete"
                )
                attachment["selected_provider"] = "custom-ep:foreign"
            elif replay_fault in {"history_result", "history_order", "replay_order"}:
                index = next(
                    i
                    for i, value in enumerate(values)
                    if hasattr(value, "get") and value.get("tool_call_id") == "call_1_0"
                )
                if replay_fault == "history_result":
                    values[index] = {
                        **values[index],
                        "content": "changed historical result",
                    }
                else:
                    values[index - 1], values[index] = values[index], values[index - 1]
            kwargs["values"] = tuple(values)
            parent = None
            old = None
            if replay_fault in {"foreign_parent", "stale_parent"}:
                checkpoint = service.current_surface_checkpoint(kwargs["segment_id"])
                parent = service._parent_capabilities[id(checkpoint)]
                from dataclasses import replace

                old = parent
                service._parent_capabilities[id(checkpoint)] = replace(
                    parent,
                    **(
                        {"owner_id": "foreign"}
                        if replay_fault == "foreign_parent"
                        else {"head_id": "stale"}
                    ),
                )
            resolve = ConsoleTraceService._resolve_reference_values
            if replay_fault == "recorded_round":

                def changed_round(service, cursor, keys, **options):
                    result = resolve(service, cursor, keys, **options)
                    for key, value in tuple(result.items()):
                        if (
                            key[0] in {"continuation", "continuation_delta"}
                            and key[1] == "artifact"
                        ):
                            value = deepcopy(value)
                            value["rounds"][0]["reasoning_blocks"] = [
                                "tampered recorded reasoning"
                            ]
                            result[key] = value
                    return result

                monkeypatch.setattr(
                    ConsoleTraceService, "_resolve_reference_values", changed_round
                )
            try:
                return original_prepare(service, cursor, **kwargs)
            finally:
                if replay_fault == "recorded_round":
                    monkeypatch.setattr(
                        ConsoleTraceService, "_resolve_reference_values", resolve
                    )
                if parent is not None:
                    service._parent_capabilities[id(checkpoint)] = old

        monkeypatch.setattr(
            ConsoleTraceService, "prepare_current_surface_delta", probe_replay
        )
        original_boundary = ConsoleTraceBoundaryFactory.__call__

        def observe_boundary(factory, *args, **kwargs):
            try:
                return original_boundary(factory, *args, **kwargs)
            except Exception as exc:
                tb = exc.__traceback__
                while tb.tb_next is not None:
                    tb = tb.tb_next
                trace_errors.append(
                    type(exc).__name__ + ":" + str(exc) + ":" + str(tb.tb_lineno)
                )
                raise

        monkeypatch.setattr(ConsoleTraceBoundaryFactory, "__call__", observe_boundary)
        from tldw_chatbook.Chat.console_trace_service import ConsoleTraceService

        original_bind = ConsoleTraceService.bind_and_mark_dispatch

        def observe_bind(service, *args, **kwargs):
            try:
                return original_bind(service, *args, **kwargs)
            except Exception as exc:
                tb = exc.__traceback__
                while tb.tb_next is not None:
                    tb = tb.tb_next
                trace_errors.append(
                    "bind:"
                    + type(exc).__name__
                    + ":"
                    + str(exc)
                    + ":"
                    + str(tb.tb_lineno)
                )
                raise

        monkeypatch.setattr(ConsoleTraceService, "bind_and_mark_dispatch", observe_bind)
        original_persist = ConsoleTraceService.persist_request

        def observe_persist(service, *args, **kwargs):
            try:
                return original_persist(service, *args, **kwargs)
            except Exception as exc:
                tb = exc.__traceback__
                while tb.tb_next is not None:
                    tb = tb.tb_next
                trace_errors.append(
                    "persist:"
                    + type(exc).__name__
                    + ":"
                    + str(exc)
                    + ":"
                    + str(tb.tb_lineno)
                )
                raise

        monkeypatch.setattr(ConsoleTraceService, "persist_request", observe_persist)

    from Tests import network_guard

    # Keep the repository's socket guard: unlike a blanket replacement it
    # permits Windows asyncio's socketpair bootstrap, while denying egress.
    assert not network_guard.is_allowed() and not network_guard.is_loopback_only()
    try:
        with (
            patch.object(socket, "getaddrinfo", denied),
            patch.object(
                urllib3.connectionpool.HTTPConnectionPool, "_make_request", fake_native
            ),
        ):
            result = await controller.submit_draft(
                "Inspect agents twice, then finish.", session_id=session.id
            )
        assert result.accepted and result.provider_started
        if fault is not None and fault not in {
            "private_projection",
            "readback",
            "retry",
        }:
            assert len(wire) == (4 if retry_mode else 2), (
                trace_errors,
                [
                    (m.role, m.content[:500])
                    for m in store.messages_for_session(session.id)
                ],
            )
            if retry_mode:
                assert retry_refusals == ["live_native_retry_surface"]
                assert executed == ["call_1_0"]
            assert fault_snapshots
            with chat.transaction() as cursor:
                assert snapshot(cursor) == fault_snapshots[-1]
                assert (
                    cursor.execute(
                        "SELECT COUNT(*) FROM console_trace_calls WHERE dispatch_started_at IS NOT NULL"
                    ).fetchone()[0]
                    == 2
                )
            return
        assert len(wire) == round_count + 1 + (3 if retry_mode else 0), (
            frames,
            trace_errors,
            [(m.role, m.content[:500]) for m in store.messages_for_session(session.id)],
        )
        if retry_mode:
            assert executed == ["call_1_0", "call_2_0"]
            assert all(request == wire[1] for request in wire[2:5])
        owners = [
            message
            for message in store.messages_for_session(session.id)
            if message.provider_continuation
        ]
        assert len(owners) == 1
        owner = owners[0]
        assert {group["owner"] for frame in frames for group in frame["groups"]} == {
            owner.id
        }
        checkpoint = owner.provider_continuation
        assert owner.content == "Synthetic final."
        assert checkpoint.schema_version == 3 and checkpoint.state == "complete"
        assert (
            checkpoint.provider == "custom-hosted"
            and checkpoint.selected_provider == selected
        )
        assert (
            checkpoint.api_base_url == "http://127.0.0.1:8010/v1"
            and checkpoint.model == model
        )
        calls = [call for round_ in checkpoint.rounds for call in round_.calls]
        assert [call.call_id for call in calls] == [
            f"call_{round_}_{call}"
            for round_ in range(1, round_count + 1)
            for call in range(batch_size)
        ]
        assert all(
            call.state == "completed" and call.result is not None for call in calls
        )
        if fault == "private_projection":
            assert all(
                json.loads(call.arguments) == {"safe_reference": call.call_id}
                for call in calls
            )
            assert all(
                call.result.value == "projected continuation result" for call in calls
            )
            assert all(
                row["content"] != "projected continuation result"
                for row in wire[-1]["messages"]
                if row.get("role") == "tool"
            )
        if capture_on:
            captured = ConsoleTraceNativeReader(chat).read_calls(owner.id)
            assert len(captured) == round_count + 1 + int(retry_mode)
            if retry_mode:
                assert captured[1].capture.request == captured[2].capture.request
                with chat.transaction() as cursor:
                    states = cursor.execute(
                        "SELECT state, outcome, surface_node_id FROM console_trace_calls ORDER BY call_sequence"
                    ).fetchall()
                assert [(row[0], row[1]) for row in states] == [
                    ("complete", "complete"),
                    ("error", "error"),
                    ("complete", "complete"),
                    ("complete", "complete"),
                ]
                assert states[1][2] == states[2][2]
            for physical_index, call in enumerate(captured):
                index = physical_index - int(retry_mode and physical_index >= 2)
                assert call.verification_status == "verified"
                assert call.capture.status == (
                    "error" if retry_mode and physical_index == 1 else "complete"
                )
                assert call.capture.provider == "custom-hosted"
                assert call.capture.model == model
                assert (
                    call.capture.request["api_base_url"]
                    == "http://127.0.0.1:8010/v1/chat/completions"
                )
                assert [
                    message["content"]
                    for message in call.capture.request["messages_payload"]
                    if message.get("role") == "user"
                ] == ["Inspect agents twice, then finish."]
                if index:
                    (continuation,) = call.capture.request["provider_continuations"]
                    assert continuation["schema_version"] == 3
                    assert continuation["selected_provider"] == selected
                    assert continuation["provider"] == "custom-hosted"
                    assert continuation["model"] == model
                    assert continuation["api_base_url"] == "http://127.0.0.1:8010/v1"
                    assert len(continuation["rounds"]) == index
                    assert [
                        item["call_id"]
                        for round_ in continuation["rounds"]
                        for item in round_["calls"]
                    ] == [
                        f"call_{round_}_{item}"
                        for round_ in range(1, index + 1)
                        for item in range(batch_size)
                    ]
            with chat.transaction() as cursor:
                artifacts = cursor.execute(
                    "SELECT a.normalization_version, a.sanitized_bytes FROM console_trace_surface_nodes n "
                    "JOIN console_trace_artifacts a ON a.artifact_id = n.artifact_id "
                    "WHERE n.component_kind IN ('continuation', 'continuation_delta') ORDER BY n.sequence"
                ).fetchall()
            assert len(artifacts) == round_count
            assert [row[0] for row in artifacts[1:]] == [
                "native-continuation-round-v1"
            ] * (round_count - 1)
            for index, row in enumerate(artifacts[1:], 2):
                encoded = json.loads(row[1])
                assert encoded["prior_round_count"] == index - 1
                assert "rounds" not in encoded["checkpoint"]
                assert [call["call_id"] for call in encoded["round"]["calls"]] == [
                    f"call_{index}_{call}" for call in range(batch_size)
                ]
        assert store.dispatch_recovery_for_session(session.id) is None
        assert metadata
        if subsequent_turn:
            later_turn = True
            with (
                patch.object(socket, "getaddrinfo", denied),
                patch.object(
                    urllib3.connectionpool.HTTPConnectionPool,
                    "_make_request",
                    fake_native,
                ),
            ):
                result = await controller.submit_draft(
                    "Continue with a fresh inspection.", session_id=session.id
                )
            assert result.accepted and result.provider_started
            if replay_fault:
                advance = replay_fault in {
                    "history_result",
                    "history_order",
                    "history_attachment",
                    "foreign_parent",
                    "stale_parent",
                }
                assert len(wire) == round_count + 1 + int(advance), trace_errors
                assert len(replay_snapshots) == 1 and trace_errors
                expected_error = {
                    "completed_attachment": "completed_replay_attachment",
                    "recorded_round": "completed_replay_prior_attachment",
                    "replay_order": "completed_replay_order",
                    "history_result": "completed_replay_value",
                    "history_order": "live_native_history_prefix",
                    "history_attachment": "live_native_history_prefix",
                    "foreign_parent": "live_native_history_lineage",
                    "stale_parent": "live_native_history_lineage",
                }[replay_fault]
                assert any(expected_error in error for error in trace_errors), (
                    trace_errors
                )
                with chat.transaction() as cursor:
                    assert replay_snapshot(cursor) == replay_snapshots[0]
                return
            assert len(wire) == 2 * (round_count + 1), (frames, trace_errors)
            assert executed == [
                f"{prefix}call_{round_}_{call}"
                for prefix in ("", "next_")
                for round_ in range(1, round_count + 1)
                for call in range(batch_size)
            ]
            later_owner = [
                m
                for m in store.messages_for_session(session.id)
                if m.provider_continuation
            ][-1]
            assert (
                later_owner.id != owner.id
                and later_owner.content == "Synthetic later final."
            )
            captured = ConsoleTraceNativeReader(chat).read_calls(later_owner.id)
            assert len(captured) == round_count + 1
            for call, sent in zip(captured, wire[round_count + 1 :], strict=True):
                assert call.verification_status == "verified"
                request = call.capture.request
                rows = (
                    [{"role": "system", "content": request["system_message"]}]
                    if request.get("system_message")
                    else []
                ) + request["messages_payload"]
                if known_credential:
                    from tldw_chatbook.Chat.console_trace_redaction import (
                        CredentialSanitizer,
                    )

                    expected = CredentialSanitizer(
                        known_credentials=(known_credential,)
                    ).sanitize(sent["messages"])
                    assert expected.available and rows == expected.value
                    assert known_credential not in repr(call.capture)
                else:
                    assert rows == sent["messages"]
                assert (
                    sum(row.get("content") == "Synthetic final." for row in rows) == 1
                )
                assert request["provider_continuations"][0]["state"] == "complete"
            assert len(captured[-1].capture.request["provider_continuations"]) == 2
            if known_credential:
                assert known_credential in repr(wire[-1]["messages"])
                with chat.transaction() as cursor:
                    assert all(
                        known_credential.encode() not in row[0]
                        for row in cursor.execute(
                            "SELECT sanitized_bytes FROM console_trace_artifacts"
                        )
                    )
    finally:
        if replay_fault in {"completed_attachment", "recorded_round", "replay_order"}:
            # A refused initial trace keeps the existing pending-dispatch guard.
            assert store.dispatch_recovery_for_session(session.id) is not None
            with pytest.raises(
                ValueError,
                match=r"^Resolve pending dispatch before changing conversation branches\.$",
            ):
                await controller.shutdown()
        else:
            await controller.shutdown()
        await gateway.aclose()
        with chat.quiesce_connections(timeout_seconds=2):
            assert chat.registered_connection_count() == 0
        runs.close()


@pytest.mark.bootstrap_profile
@pytest.mark.owned_http_client
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    ["cold", "forged", "prefix", "target", "old_round", "stale", "version", "rollback"],
)
async def test_live_native_capture_refuses_changed_custody_atomically(
    tmp_path, monkeypatch, fault
):
    await test_normal_console_multiple_native_rounds(
        tmp_path, monkeypatch, 2, 1, True, fault
    )


@pytest.mark.bootstrap_profile
@pytest.mark.owned_http_client
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["private_projection", "readback"])
async def test_live_native_capture_private_projection_and_commit_readback(
    tmp_path, monkeypatch, mode
):
    await test_normal_console_multiple_native_rounds(
        tmp_path, monkeypatch, 2, 1, True, mode
    )


@pytest.mark.bootstrap_profile
@pytest.mark.owned_http_client
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["retry", "retry_changed"])
async def test_live_native_capture_failed_request_retry(tmp_path, monkeypatch, mode):
    await test_normal_console_multiple_native_rounds(
        tmp_path, monkeypatch, 2, 1, True, mode
    )


@pytest.mark.bootstrap_profile
@pytest.mark.owned_http_client
@pytest.mark.asyncio
@pytest.mark.parametrize("detail", ["safe", "full"])
async def test_normal_manual_completed_history_and_new_native_rounds(
    tmp_path, monkeypatch, detail, replay_fault=None, known_credential=""
):
    import os
    from pathlib import Path

    from tldw_chatbook import config

    assert config.get_cli_config_path().is_relative_to(
        Path(os.environ["TLDW_TEST_CONFIG_ROOT"])
    )
    old_enabled = config.get_cli_setting("console", "exchange_capture", True)
    old_detail = config.get_cli_setting("console", "exchange_capture_detail", "safe")
    try:
        config.save_setting_to_cli_config("console", "exchange_capture", True)
        config.save_setting_to_cli_config("console", "exchange_capture_detail", detail)
        await test_normal_console_multiple_native_rounds(
            tmp_path,
            monkeypatch,
            2,
            1,
            True,
            subsequent_turn=True,
            replay_fault=replay_fault,
            known_credential=known_credential,
        )
    finally:
        config.save_setting_to_cli_config("console", "exchange_capture", old_enabled)
        config.save_setting_to_cli_config(
            "console", "exchange_capture_detail", old_detail
        )


@pytest.mark.bootstrap_profile
@pytest.mark.owned_http_client
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "replay_fault",
    [
        "completed_attachment",
        "recorded_round",
        "replay_order",
        "history_result",
        "history_order",
        "history_attachment",
        "foreign_parent",
        "stale_parent",
    ],
)
async def test_completed_replay_refuses_changed_history_before_dispatch(
    tmp_path, monkeypatch, replay_fault
):
    await test_normal_manual_completed_history_and_new_native_rounds(
        tmp_path, monkeypatch, "safe", replay_fault
    )


@pytest.mark.bootstrap_profile
@pytest.mark.owned_http_client
@pytest.mark.asyncio
@pytest.mark.parametrize("detail", ["safe", "full"])
async def test_completed_replay_known_credential_manual_rounds(
    tmp_path, monkeypatch, detail
):
    await test_normal_manual_completed_history_and_new_native_rounds(
        tmp_path,
        monkeypatch,
        detail,
        known_credential="syntheticOpaqueCredentialValue42",
    )


@pytest.mark.parametrize("owner", ["same", "foreign"])
def test_dispatch_destination_cross_check_binds_only_its_own_owner(owner):
    """A dispatch checkpoint constrains only the assistant message it owns.

    The same owner with a mismatched destination still refuses; another
    message's dispatch is irrelevant to this run, whose own host-captured
    target has already been validated.
    """
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_continuation_binding import (
        validate_durable_continuation_binding,
    )

    message = SimpleNamespace(id="assistant-b", provider_continuation=None)
    mismatched = SimpleNamespace(
        assistant_message_id="assistant-b" if owner == "same" else "assistant-a",
        checkpoint=SimpleNamespace(
            resolved_destination=SimpleNamespace(
                provider="deepseek",
                model="deepseek-v4-flash",
                continuation_provider="deepseek",
                continuation_protocol="responses",
                continuation_base_url="https://api.deepseek.com/v2",
            )
        ),
    )

    def bind():
        validate_durable_continuation_binding(
            message,
            _continuation(),
            _DEEPSEEK_RUN_TARGET,
            mismatched,
            has_finalizer=False,
        )

    if owner == "same":
        with pytest.raises(RuntimeError, match="destination mismatch"):
            bind()
    else:
        bind()
