"""Task 16 review ratchets for continuation ownership and terminalization."""

from __future__ import annotations

from pathlib import Path

import pytest

from Tests.Chat.test_console_dispatch_continuation_handoff import (
    _continuation,
    _deepseek_acceptance,
)
from Tests.Chat.test_console_dispatch_recovery import (
    _database,
    _insert,
    _raw_semantic_corruption,
)
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleDispatchRecoveryKind,
    ConsoleMessageRole,
    ConsoleRunState,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleEgressClass,
    ConsoleResolvedDestination,
)
from tldw_chatbook.Chat.console_prompt_queue_coordinator import _PromptChain
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderResolution,
    ProviderToolCalls,
    ProviderTurnMetadata,
)
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationRound,
    ProviderContinuationCheckpoint,
    dump_provider_continuation_json,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def _install_checkpoint_free_continuation(
    db,
    conversation_id: str,
    *,
    owner_id: str = "assistant-1",
) -> None:
    connection = db.get_connection()
    connection.execute("DELETE FROM console_dispatch_checkpoints")
    _raw_semantic_corruption(
        db,
        "UPDATE messages SET provider_continuation_json = ?, "
        "assistant_generation_state = NULL, version = 7 WHERE id = ?",
        (dump_provider_continuation_json(_continuation()), owner_id),
    )
    connection.execute(
        "UPDATE conversations SET active_leaf_message_id = ? WHERE id = ?",
        (owner_id, conversation_id),
    )
    connection.commit()


def _append_active_path_pair(
    db,
    conversation_id: str,
    *,
    assistant_continuation: bool,
) -> None:
    assert (
        db.add_message(
            {
                "id": "user-2",
                "conversation_id": conversation_id,
                "parent_message_id": "assistant-1",
                "sender": "user",
                "role": "user",
                "content": "follow up",
            }
        )
        == "user-2"
    )
    assistant = {
        "id": "assistant-2",
        "conversation_id": conversation_id,
        "parent_message_id": "user-2",
        "sender": "assistant",
        "role": "assistant",
        "content": "" if assistant_continuation else "later answer",
        "assistant_generation_state": (
            "continuation_active" if assistant_continuation else "complete"
        ),
    }
    if assistant_continuation:
        assistant["provider_continuation_json"] = dump_provider_continuation_json(
            _continuation(revision=2)
        )
    assert db.add_message(assistant) == "assistant-2"
    db.get_connection().execute(
        "UPDATE conversations SET active_leaf_message_id = 'assistant-2' WHERE id = ?",
        (conversation_id,),
    )
    db.get_connection().commit()


def _nodes(db, conversation_id: str) -> list[ConsoleChatMessage]:
    return [
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
            assistant_generation_state=row.get("assistant_generation_state"),
        )
        for row in db.get_messages_for_conversation(conversation_id, limit=100)
    ]


def _restore(db, conversation_id: str, nodes: list[ConsoleChatMessage]):
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.restore_persisted_session(
        title="continuation review",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=nodes,
        active_leaf_persisted_id=db.get_conversation_active_leaf(conversation_id),
    )
    return store, session.id


def test_checkpoint_free_restore_finds_and_normalizes_sole_earlier_active_owner(
    tmp_path: Path,
) -> None:
    """Kills limiting checkpoint-free continuation lookup to the active leaf."""
    db, conversation_id, repository = _database(tmp_path / "earlier-owner.sqlite")
    _insert(db, repository, _deepseek_acceptance(conversation_id))
    _install_checkpoint_free_continuation(db, conversation_id)
    _append_active_path_pair(
        db,
        conversation_id,
        assistant_continuation=False,
    )

    store, session_id = _restore(db, conversation_id, _nodes(db, conversation_id))

    durable = db.get_message_by_id("assistant-1")
    owner = store.get_message("assistant-1")
    assert durable is not None
    assert (durable["assistant_generation_state"], durable["version"]) == (
        "continuation_active",
        8,
    )
    assert owner.provider_continuation == _continuation()
    assert owner.provider_continuation_message_version == 8
    assert owner.provider_continuation_actions_enabled is True
    assert store.provider_continuation_recovery_message(session_id).id == "assistant-1"
    assert store.dispatch_recovery_for_session(session_id) is None


def test_checkpoint_free_restore_quarantines_multiple_active_path_continuations(
    tmp_path: Path,
) -> None:
    """Kills choosing the leaf when two active-path continuations claim ownership."""
    db, conversation_id, repository = _database(tmp_path / "duplicate-owner.sqlite")
    _insert(db, repository, _deepseek_acceptance(conversation_id))
    _install_checkpoint_free_continuation(db, conversation_id)
    _append_active_path_pair(
        db,
        conversation_id,
        assistant_continuation=True,
    )

    recovery = repository.reconcile_for_session(conversation_id)

    assert recovery is not None
    assert recovery.kind is ConsoleDispatchRecoveryKind.QUARANTINED
    assert recovery.error_code == "duplicate_active_path_owner"
    assert recovery.actions == ()
    assert db.get_message_by_id("assistant-1")["provider_continuation_json"]
    assert db.get_message_by_id("assistant-2")["provider_continuation_json"]


def test_checkpoint_free_restore_quarantines_earlier_orphan_continuation_state(
    tmp_path: Path,
) -> None:
    """Kills inspecting private continuation integrity on the leaf alone."""
    db, conversation_id, repository = _database(tmp_path / "earlier-orphan.sqlite")
    _insert(db, repository, _deepseek_acceptance(conversation_id))
    _install_checkpoint_free_continuation(db, conversation_id)
    _raw_semantic_corruption(
        db,
        "UPDATE messages SET provider_continuation_json = NULL, "
        "assistant_generation_state = 'continuation_active' WHERE id = 'assistant-1'",
    )
    db.get_connection().commit()
    _append_active_path_pair(
        db,
        conversation_id,
        assistant_continuation=False,
    )

    recovery = repository.reconcile_for_session(conversation_id)

    assert recovery is not None
    assert recovery.kind is ConsoleDispatchRecoveryKind.QUARANTINED
    assert recovery.assistant_message_id == "assistant-1"
    assert recovery.error_code == "orphan_continuation"
    assert recovery.actions == ()


def test_continuation_row_read_failure_remains_blocking_until_exact_reread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills deleting the neutral owner when private continuation hydration fails."""
    db, conversation_id, repository = _database(tmp_path / "read-failure.sqlite")
    _insert(db, repository, _deepseek_acceptance(conversation_id))
    _install_checkpoint_free_continuation(db, conversation_id)
    nodes = _nodes(db, conversation_id)
    healthy_reader = db.get_messages_for_conversation

    def fail_read(*_args, **_kwargs):
        raise RuntimeError("injected continuation row read failure")

    monkeypatch.setattr(db, "get_messages_for_conversation", fail_read)
    blocked, blocked_session_id = _restore(db, conversation_id, nodes)

    recovery = blocked.dispatch_recovery_for_session(blocked_session_id)
    assert recovery is not None
    assert recovery.kind is ConsoleDispatchRecoveryKind.QUARANTINED
    assert recovery.assistant_message_id == "assistant-1"
    assert recovery.error_code == "continuation_hydration_error"
    assert recovery.actions == ()
    assert blocked.dispatch_recovery_for_presentation(blocked_session_id) == recovery
    assert blocked.dispatch_recovery_blocks_submission(blocked_session_id) is True
    assert blocked.provider_continuation_recovery_message(blocked_session_id) is None
    assert db.get_message_by_id("assistant-1")["version"] == 7

    monkeypatch.setattr(db, "get_messages_for_conversation", healthy_reader)
    recovered, recovered_session_id = _restore(
        db,
        conversation_id,
        _nodes(db, conversation_id),
    )
    owner = recovered.provider_continuation_recovery_message(recovered_session_id)
    assert owner is not None
    assert owner.id == "assistant-1"
    assert owner.provider_continuation_actions_enabled is True
    assert recovered.dispatch_recovery_for_session(recovered_session_id) is None


def _complete_reasoning_continuation() -> ProviderContinuationCheckpoint:
    return ProviderContinuationCheckpoint(
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
                reasoning_blocks=("private reasoning",),
                calls=(),
            ),
        ),
    )


def test_checkpoint_free_restore_ignores_completed_continuation_history(
    tmp_path: Path,
) -> None:
    """Completed private history is not an active or invalid recovery owner."""
    db, conversation_id, repository = _database(tmp_path / "complete-history.sqlite")
    _insert(db, repository, _deepseek_acceptance(conversation_id))
    connection = db.get_connection()
    connection.execute("DELETE FROM console_dispatch_checkpoints")
    _raw_semantic_corruption(
        db,
        "UPDATE messages SET content = 'answer', provider_continuation_json = ?, "
        "assistant_generation_state = 'complete', version = 3 WHERE id = 'assistant-1'",
        (dump_provider_continuation_json(_complete_reasoning_continuation()),),
    )
    connection.commit()

    assert repository.reconcile_for_session(conversation_id) is None


class _ReasoningOnlyGateway:
    def __init__(self) -> None:
        self.calls = 0

    async def resolve_for_send(self, _selection):
        return ConsoleProviderResolution(
            provider="moonshot",
            base_url="https://api.moonshot.ai/v1",
            model="kimi-k3",
            ready=True,
            readiness_key="moonshot",
            execution_key="moonshot",
            continuation_protocol="chat_completions",
            resolved_destination=ConsoleResolvedDestination(
                provider="moonshot",
                model="kimi-k3",
                endpoint_identity="https://api.moonshot.ai/v1",
                egress_class=ConsoleEgressClass.PUBLIC_NETWORK,
            ),
        )

    async def stream_chat(self, _resolution, _messages, **_kwargs):
        self.calls += 1
        yield "answer"
        yield ProviderToolCalls(
            (),
            metadata=ProviderTurnMetadata(
                finish_reason="stop",
                provider_continuation=_complete_reasoning_continuation(),
            ),
        )


def _reasoning_controller(tmp_path: Path):
    from tldw_chatbook.Chat.prompt_history import PromptHistory
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(tmp_path / "chat.sqlite", client_id="task16-review")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    store.create_session(
        session_id="session-1",
        title="reasoning",
        project_instruction_state=ProjectInstructionControlState.legacy_disabled(),
    )
    gateway = _ReasoningOnlyGateway()
    bridge = ConsoleAgentBridge(
        agent_runs_db=AgentRunsDB(
            tmp_path / "agent-runs.sqlite",
            client_id="task16-review",
        ),
        store=store,
        provider_gateway=gateway,
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="moonshot",
        model="kimi-k3",
        base_url="https://api.moonshot.ai/v1",
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )
    controller.prompt_history = PromptHistory(tmp_path / "history.jsonl")
    return db, store, controller, gateway


def _assert_reasoning_turn_completed_once(db, store, *, expected_calls: int) -> None:
    session = store.sessions()[0]
    assert session.persisted_conversation_id is not None
    rows = db.get_messages_for_conversation(
        session.persisted_conversation_id,
        limit=100,
    )
    assistants = [row for row in rows if row["role"] == "assistant"]
    assert len(assistants) == expected_calls
    assert all(
        (row["content"], row["assistant_generation_state"], row["version"])
        == ("answer", "complete", 3)
        for row in assistants
    )
    assert all(row["provider_continuation_json"] is not None for row in assistants)
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
        .fetchone()[0]
        == 0
    )
    assert store.dispatch_recovery_for_session("session-1") is None


@pytest.mark.asyncio
async def test_manual_reasoning_only_final_event_is_not_terminalized_twice(
    tmp_path: Path,
) -> None:
    """Kills calling mark_message_complete after atomic FinalContinuation settle."""
    db, store, controller, gateway = _reasoning_controller(tmp_path)

    result = await controller.submit_draft("reason about this", session_id="session-1")

    assert result.accepted is True
    assert result.provider_started is True
    assert result.visible_copy == "answer"
    assert controller.run_state_for("session-1").status is ConsoleRunStatus.COMPLETED
    assert gateway.calls == 1
    _assert_reasoning_turn_completed_once(db, store, expected_calls=1)


@pytest.mark.asyncio
async def test_queued_reasoning_only_final_event_settles_once(
    tmp_path: Path,
) -> None:
    """Kills terminal double-write failures that pause an otherwise healthy queue."""
    db, store, controller, gateway = _reasoning_controller(tmp_path)
    coordinator = controller.prompt_queue_coordinator
    registry = coordinator.registry
    begun = registry.begin_chain("session-1", context_epoch=0, expected_revision=0)
    first = registry.admit(
        "session-1",
        text="queued one",
        expected_revision=begun.snapshot.revision,
    )
    assert first.entry_id is not None
    coordinator._chains["session-1"] = _PromptChain()
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.COMPLETED),
        session_id="session-1",
    )

    await coordinator._drain_waiting("session-1", ConsoleRunStatus.COMPLETED)

    snapshot = registry.snapshot("session-1")
    assert snapshot.total_count == 0
    assert coordinator.dispatch_recovery_blocks_queue("session-1") is False
    assert controller.run_state_for("session-1").status is ConsoleRunStatus.COMPLETED
    assert gateway.calls == 1
    _assert_reasoning_turn_completed_once(db, store, expected_calls=1)
