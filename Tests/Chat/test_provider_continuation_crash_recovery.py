"""Joined crash-recovery evidence for durable provider continuation.

The focused component suites pin each individual boundary.  This module keeps
one restart-level matrix so a future change cannot make the pieces green while
breaking their joined safety property: durable state is exact, portable intent
is idempotent, and restored terminal/ambiguous calls never dispatch again.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import pytest

from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    ContinuationEventContext,
    ModelTurn,
    ToolCall,
    ToolBatchReady,
    ToolCallExecuting,
    ToolCallFinished,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop
from tldw_chatbook.Agents.run_log_eviction import bound_history_for_send
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationResult,
    ContinuationRestoreTarget,
    continuation_owner_group,
    dump_provider_continuation_json,
    parse_provider_continuation_json,
    read_provider_continuation_json,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_chatbook.Sync_Interop.hashing import canonical_payload_hash
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from Tests.Chat.test_console_dispatch_continuation_handoff import (
    _portable_store as _portable_dispatch_store,
    _started_portable,
)


CrashBoundary = Literal[
    "before_assistant_checkpoint_commit",
    "after_commit_before_sync_projection",
    "after_projection_before_acknowledgement",
    "before_executing",
    "during_side_effect",
    "after_result_commit",
    "before_next_provider_request",
]

_BOUNDARIES: tuple[CrashBoundary, ...] = (
    "before_assistant_checkpoint_commit",
    "after_commit_before_sync_projection",
    "after_projection_before_acknowledgement",
    "before_executing",
    "during_side_effect",
    "after_result_commit",
    "before_next_provider_request",
)
_SYNC_BOUNDARIES = {
    "after_commit_before_sync_projection",
    "after_projection_before_acknowledgement",
}
@dataclass(frozen=True)
class _CrashSnapshot:
    boundary: CrashBoundary
    restored_call_state: str | None
    interrupted_owner_id: str | None
    outbox_before_restart: int
    outbox_after_reconciliation: int


class _SimulatedProcessDeath(BaseException):
    """Escape runtime exception handling at an exact crash boundary."""


def _checkpoint_payload(
    call_state: Literal["pending", "executing", "completed", "failed"] = "pending",
    *,
    checkpoint_revision: int = 1,
    state: Literal["active", "complete"] = "active",
) -> dict[str, object]:
    call: dict[str, object] = {
        "call_id": "call-1",
        "name": "calculator",
        "arguments": '{"expression":"2+2"}',
        "state": call_state,
    }
    if call_state in {"completed", "failed"}:
        call["result"] = "4" if call_state == "completed" else "recorded failure"
    return {
        "schema_version": 1,
        "checkpoint_revision": checkpoint_revision,
        "provider": "deepseek",
        "protocol": "responses",
        "model": "deepseek-v4-flash",
        "api_base_url": "https://api.deepseek.com/v1",
        "state": state,
        "rounds": [
            {
                "assistant_content": "",
                "reasoning_blocks": ["PRIVATE-CRASH-REASONING-CANARY"],
                "calls": [call],
            }
        ],
    }


def _checkpoint(
    call_state: Literal["pending", "executing", "completed", "failed"] = "pending",
    *,
    checkpoint_revision: int = 1,
    state: Literal["active", "complete"] = "active",
):
    return parse_provider_continuation_json(
        _checkpoint_payload(
            call_state,
            checkpoint_revision=checkpoint_revision,
            state=state,
        )
    )


def _continuation_receipt_count(
    repository: SyncStateRepository, owner_message_id: str
) -> int:
    with repository._get_connection() as connection:
        return connection.execute(
            "SELECT COUNT(*) FROM sync_v2_source_projection_receipts "
            "WHERE domain = 'chat' AND source_entity_id = ?",
            (owner_message_id,),
        ).fetchone()[0]


def _payload_hash(row: dict) -> str:
    return canonical_payload_hash(
        {
            "content": row["content"],
            "provider_continuation_json": row["provider_continuation_json"],
            "role": "assistant",
        }
    )


def _crash_snapshot(tmp_path: Path, boundary: CrashBoundary) -> _CrashSnapshot:
    boundary_path = tmp_path / boundary
    boundary_path.mkdir()
    (
        database,
        conversation_id,
        _dispatch_repository,
        _dispatch_checkpoint,
        store,
        session_id,
        repository,
        producer,
    ) = _started_portable(boundary_path)
    owner = store.get_message("assistant-1")
    context = ContinuationEventContext(
        owner.id,
        "run-a",
        "primary",
        "persistent",
    )

    if boundary in _SYNC_BOUNDARIES:
        reconcile = producer.reconcile_chat_message_intent

        if boundary == "after_commit_before_sync_projection":

            def fail_before_projection(**_kwargs):
                raise RuntimeError("simulated process death before projection")

            producer.reconcile_chat_message_intent = fail_before_projection
        else:

            def fail_after_projection(**kwargs):
                reconcile(**kwargs)
                raise RuntimeError("simulated process death before acknowledgement")

            producer.reconcile_chat_message_intent = fail_after_projection

    if boundary != "before_assistant_checkpoint_commit":
        try:
            store.persist_provider_continuation_event(
                ToolBatchReady(context, _checkpoint(), None)
            )
        except RuntimeError:
            assert boundary in _SYNC_BOUNDARIES

    if boundary == "during_side_effect":
        store.persist_provider_continuation_event(
            ToolCallExecuting(context, "call-1", 1)
        )
    elif boundary in {"after_result_commit", "before_next_provider_request"}:
        store.persist_provider_continuation_event(
            ToolCallExecuting(context, "call-1", 1)
        )
        store.persist_provider_continuation_event(
            ToolCallFinished(
                context,
                "call-1",
                2,
                "completed",
                ContinuationResult("4"),
            )
        )

    # The restored accepted v2 owner is projected before the first provider
    # event. Only v3+ receipts belong to ADR-063 continuation ownership.
    outbox_before_restart = max(
        0,
        _continuation_receipt_count(repository, owner.id) - 1,
    )
    database.close_connection()

    restarted_db = CharactersRAGDB(database.db_path, f"restart-{boundary}")
    row = restarted_db.get_message_by_id(owner.id)
    (
        restored_store,
        restored_session_id,
        restarted_repository,
        _restarted_producer,
    ) = _portable_dispatch_store(
        restarted_db,
        conversation_id,
        boundary_path,
    )
    interrupted = restored_store.interrupted_provider_continuation_message(
        restored_session_id
    )
    restored_call_state = None
    if row is not None and row["provider_continuation_json"] is not None:
        checkpoint = parse_provider_continuation_json(row["provider_continuation_json"])
        restored_call_state = checkpoint.rounds[-1].calls[-1].state

    outbox_after_reconciliation = max(
        0,
        _continuation_receipt_count(restarted_repository, owner.id) - 1,
    )

    snapshot = _CrashSnapshot(
        boundary=boundary,
        restored_call_state=restored_call_state,
        interrupted_owner_id=interrupted.id if interrupted else None,
        outbox_before_restart=outbox_before_restart,
        outbox_after_reconciliation=outbox_after_reconciliation,
    )
    restarted_db.close_connection()
    return snapshot


def test_every_approved_crash_boundary_has_an_exact_restored_state(tmp_path) -> None:
    snapshots = [_crash_snapshot(tmp_path, boundary) for boundary in _BOUNDARIES]

    assert [snapshot.restored_call_state for snapshot in snapshots] == [
        None,
        "pending",
        "pending",
        "pending",
        "executing",
        "completed",
        "completed",
    ]
    assert [snapshot.interrupted_owner_id is not None for snapshot in snapshots] == [
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    assert [snapshot.outbox_before_restart for snapshot in snapshots] == [
        0,
        0,
        1,
        1,
        2,
        3,
        3,
    ]
    assert [snapshot.outbox_after_reconciliation for snapshot in snapshots] == [
        0,
        1,
        1,
        1,
        2,
        3,
        3,
    ]


@pytest.mark.parametrize(
    ("boundary", "expected_state", "expected_model_attempts"),
    [
        ("during_side_effect", "executing", 1),
        ("after_result_commit", "completed", 2),
        ("before_next_provider_request", "completed", 3),
    ],
)
def test_runtime_crash_hooks_restart_without_repeating_side_effects(
    tmp_path: Path,
    boundary: CrashBoundary,
    expected_state: str,
    expected_model_attempts: int,
) -> None:
    (
        database,
        conversation_id,
        _dispatch_repository,
        _dispatch_checkpoint,
        store,
        _session_id,
        _sync_repository,
        _producer,
    ) = _started_portable(tmp_path)
    owner = store.get_message("assistant-1")
    context = ContinuationEventContext(
        owner.id,
        "run-a",
        "primary",
        "persistent",
    )
    schema = ToolSchema(
        id="builtin:calculator",
        name="calculator",
        description="math",
        parameters={"type": "object"},
    )
    call = ToolCall(
        "calculator",
        {"expression": "2+2"},
        "call-1",
        '{"expression":"2+2"}',
    )
    side_effects: list[str] = []
    model_attempts: list[str] = []
    successful_model_calls: list[str] = []

    def call_model(messages, _schemas):
        model_attempts.append("original")
        if len(model_attempts) > 1:
            if boundary == "before_next_provider_request":
                raise _SimulatedProcessDeath
            pytest.fail("provider request crossed an earlier crash boundary")
        successful_model_calls.append("original")
        return ModelTurn(
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
            provider_continuation=_checkpoint(),
        )

    def invoke_tool(actual: ToolCall) -> ToolResult:
        side_effects.append(actual.call_id)
        if boundary == "during_side_effect":
            raise _SimulatedProcessDeath
        return ToolResult(ok=True, content="4")

    def persist(event) -> None:
        store.persist_provider_continuation_event(event)
        if boundary == "after_result_commit" and isinstance(
            event, ToolCallFinished
        ):
            raise _SimulatedProcessDeath

    with pytest.raises(_SimulatedProcessDeath):
        run_agent_loop(
            AgentConfig(
                model="deepseek-v4-flash",
                system_prompt="system",
                allowed_tools=("calculator",),
            ),
            [{"role": "user", "content": "Use the calculator"}],
            [schema],
            LoopDeps(
                call_model=call_model,
                invoke_tool=invoke_tool,
                spawn=lambda task: ToolResult(ok=True, content=task),
                find_tools=lambda query: [],
                load_schemas=lambda ids: [],
                should_cancel=lambda: False,
                clock=lambda: 0.0,
                review_tool_calls=lambda calls: {},
                continuation_context=context,
                persist_provider_continuation=persist,
            ),
        )
    database.close_connection()

    restarted_db = CharactersRAGDB(database.db_path, f"restart-runtime-{boundary}")
    row = restarted_db.get_message_by_id(owner.id)
    assert row is not None
    restored = parse_provider_continuation_json(row["provider_continuation_json"])
    assert restored.rounds[-1].calls[-1].state == expected_state
    restarted_store, _, _, _ = _portable_dispatch_store(
        restarted_db,
        conversation_id,
        tmp_path,
    )

    def resumed_model(messages, _schemas):
        model_attempts.append("restarted")
        successful_model_calls.append("restarted")
        final = replace(
            restored,
            checkpoint_revision=restored.checkpoint_revision + 1,
            state="complete",
        )
        return ModelTurn(text="done", provider_continuation=final)

    outcome = run_agent_loop(
        AgentConfig(
            model="deepseek-v4-flash",
            system_prompt="system",
            allowed_tools=("calculator",),
        ),
        [{"role": "user", "content": "continue"}],
        [schema],
        LoopDeps(
            call_model=resumed_model,
            invoke_tool=invoke_tool,
            spawn=lambda task: ToolResult(ok=True, content=task),
            find_tools=lambda query: [],
            load_schemas=lambda ids: [],
            should_cancel=lambda: False,
            clock=lambda: 0.0,
            review_tool_calls=lambda calls: {},
            continuation_context=context,
            persist_provider_continuation=(
                restarted_store.persist_provider_continuation_event
            ),
            expand_provider_continuation=lambda checkpoint: [
                {
                    "role": "tool",
                    "tool_call_id": canonical.call_id,
                    "content": canonical.result.value,
                }
                for round_ in checkpoint.rounds
                for canonical in round_.calls
                if canonical.result is not None
            ],
        ),
        restore_provider_continuation=restored,
        restore_provider_target=ContinuationRestoreTarget(
            "deepseek",
            "deepseek-v4-flash",
            "responses",
            "https://api.deepseek.com/v1",
        ),
        resume_provider_continuation=True,
    )

    assert side_effects == ["call-1"]
    if boundary == "during_side_effect":
        assert outcome.status == "stuck"
        assert successful_model_calls == ["original"]
    else:
        assert outcome.status == "done"
        assert successful_model_calls == ["original", "restarted"]
    assert len(model_attempts) == expected_model_attempts
    restarted_db.close_connection()


@pytest.mark.parametrize(
    (
        "call_state",
        "expected_invocations",
        "expected_model_calls",
        "expected_replay",
    ),
    [
        ("pending", 1, 0, None),
        ("executing", 0, 0, None),
        ("completed", 0, 1, "4"),
        ("failed", 0, 1, "recorded failure"),
    ],
)
def test_restore_state_mutations_cannot_duplicate_a_terminal_or_ambiguous_call(
    call_state: Literal["pending", "executing", "completed", "failed"],
    expected_invocations: int,
    expected_model_calls: int,
    expected_replay: str | None,
) -> None:
    checkpoint = _checkpoint(call_state)
    invocations: list[str] = []
    model_calls: list[list[dict]] = []
    events = []
    schema = ToolSchema(
        id="builtin:calculator",
        name="calculator",
        description="math",
        parameters={"type": "object"},
    )

    def call_model(messages, _schemas):
        model_calls.append(list(messages))
        return ModelTurn(text="next provider reply")

    outcome = run_agent_loop(
        AgentConfig(
            model="deepseek-v4-flash",
            system_prompt="system",
            allowed_tools=("calculator",),
        ),
        [{"role": "user", "content": "continue"}],
        [schema],
        LoopDeps(
            call_model=call_model,
            invoke_tool=lambda call: (
                invocations.append(call.call_id) or ToolResult(ok=True, content="4")
            ),
            spawn=lambda task: ToolResult(ok=True, content=task),
            find_tools=lambda query: [],
            load_schemas=lambda ids: [],
            should_cancel=lambda: len(events) >= 2,
            clock=lambda: 0.0,
            review_tool_calls=lambda calls: {},
            continuation_context=ContinuationEventContext(
                "owner-a", "run-a", "primary", "persistent"
            ),
            persist_provider_continuation=events.append,
            expand_provider_continuation=lambda restored: [
                {
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": call.result.value,
                }
                for round_ in restored.rounds
                for call in round_.calls
                if call.result is not None
            ],
        ),
        restore_provider_continuation=checkpoint,
        restore_provider_target=ContinuationRestoreTarget(
            "deepseek",
            "deepseek-v4-flash",
            "responses",
            "https://api.deepseek.com/v1",
        ),
        resume_provider_continuation=True,
    )

    assert len(invocations) == expected_invocations, outcome
    assert len(model_calls) == expected_model_calls, outcome
    if call_state == "executing":
        assert outcome.status == "stuck"
        assert "ambiguous" in outcome.steps[-1].summary
    if expected_replay is not None:
        assert any(
            row.get("tool_call_id") == "call-1"
            and row.get("content") == expected_replay
            for row in model_calls[0]
        )


@pytest.mark.parametrize("invalid_kind", ["unknown_version", "oversized_reasoning"])
def test_invalid_private_state_is_bounded_and_visible_owner_stays_usable(
    invalid_kind: str,
) -> None:
    payload = _checkpoint_payload()
    if invalid_kind == "unknown_version":
        payload["schema_version"] = 999
    else:
        payload["rounds"][0]["reasoning_blocks"] = [  # type: ignore[index]
            "x" * (4 * 1024 * 1024 + 1)
        ]

    safe = read_provider_continuation_json(payload)

    assert safe.checkpoint is None
    assert safe.warning == "Exact tool continuation was discarded."
    assert "PRIVATE-CRASH-REASONING-CANARY" not in repr(safe)


def test_stale_whole_record_update_preserves_branch_variant_owner(tmp_path) -> None:
    database = CharactersRAGDB(tmp_path / "conflict.db", "owner-client")
    try:
        conversation_id = database.add_conversation({"title": "Variants"})
        owner_id = "assistant-owner"
        sibling_id = "assistant-sibling"
        private_json = dump_provider_continuation_json(_checkpoint())
        assert private_json is not None
        database.create_assistant_with_continuation(
            message_id=owner_id,
            conversation_id=conversation_id,
            parent_message_id=None,
            content="owner visible",
            provider_continuation_json=private_json,
        )
        database.add_message(
            {
                "id": sibling_id,
                "conversation_id": conversation_id,
                "sender": "assistant",
                "content": "sibling visible",
            }
        )
        with database.transaction() as connection:
            connection.execute(
                "UPDATE messages SET variant_number = 1, "
                "is_selected_variant = 1, total_variants = 2 WHERE id = ?",
                (owner_id,),
            )
            connection.execute(
                "UPDATE messages SET variant_of = ?, variant_number = 2, "
                "is_selected_variant = 0, total_variants = 2 WHERE id = ?",
                (owner_id, sibling_id),
            )
        database.update_message(
            owner_id,
            {"content": "owner edited"},
            expected_version=1,
        )

        with pytest.raises(ConflictError):
            database.update_provider_continuation(
                message_id=owner_id,
                expected_message_version=1,
                provider_continuation_json=None,
            )

        owner = database.get_message_by_id(owner_id)
        sibling = database.get_message_by_id(sibling_id)
        with database.transaction() as connection:
            variant_rows = {
                row["id"]: row
                for row in connection.execute(
                    "SELECT id, variant_of, variant_number FROM messages "
                    "WHERE id IN (?, ?)",
                    (owner_id, sibling_id),
                ).fetchall()
            }
        assert owner is not None and owner["provider_continuation_json"] is not None
        assert sibling is not None and sibling["provider_continuation_json"] is None
        assert variant_rows[owner_id]["variant_number"] == 1
        assert variant_rows[sibling_id]["variant_of"] == owner_id
    finally:
        database.close_connection()


def test_private_history_and_visible_owner_are_one_eviction_unit() -> None:
    payload = _checkpoint_payload("completed", state="complete")
    payload["rounds"][0]["reasoning_blocks"] = [  # type: ignore[index]
        "PRIVATE-CRASH-REASONING-CANARY " * 80
    ]
    complete = parse_provider_continuation_json(payload)
    group = continuation_owner_group(
        {"id": "assistant-old", "role": "assistant", "content": "old answer"},
        complete,
    )
    payload = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "old answer",
            "owner": "assistant-old",
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "4"},
        {"role": "assistant", "content": "new answer", "owner": "assistant-new"},
        {"role": "tool", "tool_call_id": "call-2", "content": "new result"},
    ]

    evicted = bound_history_for_send(
        payload,
        model="deepseek-v4-flash",
        provider="deepseek",
        native=True,
        enabled=True,
        response_reservation=0,
        window=540,
        count_fn=lambda rows, _model: sum(
            len(str(row.get("content", "")).split()) for row in rows
        ),
        min_recent_rounds=1,
        continuation_groups=(group,),
        continuation_owner_key="owner",
    )
    detached = replace(group, owner_message_id="missing-owner")

    assert not any(row.get("owner") == "assistant-old" for row in evicted)
    assert not any(row.get("tool_call_id") == "call-1" for row in evicted)
    assert (
        bound_history_for_send(
            payload,
            model="deepseek-v4-flash",
            provider="deepseek",
            native=True,
            enabled=True,
            response_reservation=0,
            window=540,
            count_fn=lambda rows, _model: sum(
                len(str(row.get("content", "")).split()) for row in rows
            ),
            min_recent_rounds=1,
            continuation_groups=(detached,),
            continuation_owner_key="owner",
        )
        is payload
    )
    assert "PRIVATE-CRASH-REASONING-CANARY" not in repr(evicted)
