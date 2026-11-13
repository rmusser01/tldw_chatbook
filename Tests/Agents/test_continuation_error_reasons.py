"""TASK-34100.3 AC#3: every continuation-error exit names its precondition.

Before this task all of the agent loop's continuation-error exits recorded the
same "Provider continuation could not be persisted" summary, and the store's
exception was swallowed by ``persist_continuation``. A live Moonshot failure
(gap-02) therefore could not say which precondition failed. Each exit now
records a distinct, code-owned reason in the error step's ``field_states`` and
in the log; a store failure adds the store's own precondition code. Neither
ever carries the exception text (ADR-029: messages are not metadata).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from loguru import logger

from Tests.Agents.test_provider_continuation_runtime import (
    CALCULATOR,
    CONFIG,
    _deps,
    _kimi_checkpoint,
    _native_turn,
    _pending_call,
)
from Tests.Agents.test_provider_continuation_runtime import (
    _checkpoint as _runtime_checkpoint,
)
from tldw_chatbook.Agents.agent_models import (
    RUN_DONE,
    RUN_ERROR,
    STEP_ERROR,
    ContinuationEventContext,
    FinalContinuation,
    ModelTurn,
    ToolBatchReady,
    ToolCall,
    ToolCallExecuting,
    ToolCallFinished,
    ToolResult,
)
from tldw_chatbook.Agents.agent_runtime import run_agent_loop
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationCall,
    ContinuationRestoreTarget,
    ContinuationResult,
    ContinuationRound,
    ProviderContinuationCheckpoint,
)
from tldw_chatbook.Chat.provider_continuation_diagnostics import (
    ACKNOWLEDGE_FAILED,
    STORE_FAILURE_CODES,
    UNRECOGNIZED_STORE_FAILURE,
    UNWRITTEN_STORE_FAILURES,
    store_failure_code,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHAT = _REPO_ROOT / "tldw_chatbook" / "Chat"
_STORE = _CHAT / "console_chat_store.py"
#: Every function on the store's continuation persist path that raises one of
#: its fixed failure sentences. Dev moved two of them out of the store
#: (TASK-35582's binding module, and the dispatch settlement body), and
#: TASK-37653 put the managed pre-seal (``_presealed_continuation_finalizer``,
#: applied by ``_continuation_event_owner_scope``) on the path.
_PERSIST_PATH = {
    _STORE: {
        "persist_provider_continuation_event",
        "_continuation_write_value",
        "_presealed_continuation_finalizer",
        "ensure_provider_continuation_durable",
    },
    _CHAT / "console_continuation_binding.py": {
        "continuation_event_value",
        "validate_durable_continuation_binding",
    },
    _CHAT / "console_native_commit.py": {"_store_settle_dispatch_recovery"},
}


@pytest.fixture
def warnings_log():
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink)


def _final_round(content: str = "answer") -> ContinuationRound:
    return ContinuationRound(content, ("private",), ())


def _run(turns, *, persist, context=None, invoke=None):
    order: list[str] = []
    return run_agent_loop(
        CONFIG,
        [{"role": "user", "content": "go"}],
        [CALCULATOR],
        _deps(
            turns,
            order=order,
            persist=persist,
            context=context,
            invoke=invoke or (lambda call: pytest.fail("no tool expected")),
        ),
    )


# Every value a Trace ``field_states`` entry may hold (Trace export/import).
_TRACE_FIELD_STATES = frozenset(
    {"observed", "redacted", "truncated", "omitted", "not_available", "capture_failed"}
)


def _reasons(field_states) -> dict[str, str]:
    """Decode a step's reason codes, checking every state stays exportable."""
    from tldw_chatbook.Chat.provider_continuation_diagnostics import (
        continuation_reasons,
    )

    assert set(field_states.values()) <= _TRACE_FIELD_STATES, field_states
    return continuation_reasons(field_states)


def _error_states(outcome) -> dict[str, str]:
    errors = [step for step in outcome.steps if step.kind == STEP_ERROR]
    assert errors, "the run must record an error step"
    return _reasons(errors[-1].field_states)


def _tool_then_final(final_checkpoint):
    """A k3 tool round followed by a final turn carrying ``final_checkpoint``."""
    call = ToolCall("calculator", {"expression": "2+2"}, "a", '{"expression":"2+2"}')
    pending = ContinuationRound(
        "",
        ("tool reasoning",),
        (ContinuationCall("a", "calculator", '{"expression":"2+2"}', "pending"),),
    )
    initial = _kimi_checkpoint(revision=1, state="active", rounds=(pending,))
    return [
        _native_turn((call,), initial),
        ModelTurn(text="visible answer", provider_continuation=final_checkpoint),
    ]


def _completed_tool_round() -> ContinuationRound:
    return ContinuationRound(
        "",
        ("tool reasoning",),
        (
            ContinuationCall(
                "a",
                "calculator",
                '{"expression":"2+2"}',
                "completed",
                ContinuationResult("4"),
            ),
        ),
    )


@pytest.mark.parametrize(
    ("label", "turn", "context", "expected"),
    [
        (
            "unserializable",
            ModelTurn(
                text="answer",
                provider_continuation=ProviderContinuationCheckpoint(
                    schema_version=99,  # type: ignore[arg-type]
                    checkpoint_revision=1,
                    provider="moonshot",
                    protocol="chat_completions",
                    model="kimi-k3",
                    api_base_url="https://api.moonshot.ai/v1",
                    state="complete",
                    rounds=(_final_round(),),
                ),
            ),
            None,
            "final_checkpoint_unserializable",
        ),
        (
            "persistent owner missing",
            ModelTurn(
                text="answer",
                provider_continuation=_kimi_checkpoint(
                    revision=1, state="complete", rounds=(_final_round(),)
                ),
            ),
            ContinuationEventContext(
                owner_message_id=None,
                run_id="run-1",
                agent_kind="primary",
                durability="persistent",
            ),
            "final_context_invalid",
        ),
        (
            "first revision is not 1",
            ModelTurn(
                text="answer",
                provider_continuation=_kimi_checkpoint(
                    revision=2, state="complete", rounds=(_final_round(),)
                ),
            ),
            None,
            "final_first_revision_invalid",
        ),
        (
            "assistant content differs from the delivered text",
            ModelTurn(
                text="answer",
                provider_continuation=_kimi_checkpoint(
                    revision=1,
                    state="complete",
                    rounds=(_final_round("different answer"),),
                ),
            ),
            None,
            "final_first_content_mismatch",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_each_tool_free_final_exit_records_its_own_reason(
    label, turn, context, expected, warnings_log
) -> None:
    events: list[object] = []

    outcome = _run([turn], persist=events.append, context=context)

    assert outcome.status == RUN_ERROR, label
    assert events == []
    assert _error_states(outcome) == {"continuation_error": expected}
    assert any(f"reason={expected}" in line for line in warnings_log), warnings_log


def test_post_tool_final_update_mismatch_records_its_reason(warnings_log) -> None:
    wrong_revision = _kimi_checkpoint(
        revision=9,
        state="complete",
        rounds=(_completed_tool_round(), _final_round("visible answer")),
    )
    events: list[object] = []

    outcome = _run(
        _tool_then_final(wrong_revision),
        persist=events.append,
        invoke=lambda call: ToolResult(ok=True, content="4"),
    )

    assert outcome.status == RUN_ERROR
    assert _error_states(outcome) == {"continuation_error": "final_update_invalid"}


def test_post_tool_final_without_checkpoint_records_its_reason(warnings_log) -> None:
    outcome = _run(
        _tool_then_final(None),
        persist=lambda event: None,
        invoke=lambda call: ToolResult(ok=True, content="4"),
    )

    assert outcome.status == RUN_ERROR
    assert _error_states(outcome) == {"continuation_error": "final_checkpoint_missing"}


# --- Review v5r1 TE-5: restore and transition exits, driven, not just parsed. ---

_DEEPSEEK_TARGET = ContinuationRestoreTarget(
    "deepseek", "deepseek-v4-flash", "responses", "https://api.deepseek.com/v1"
)


def _restore_checkpoint() -> ProviderContinuationCheckpoint:
    return _runtime_checkpoint(_pending_call())


#: A persistent run that names no owner message cannot hold a continuation.
_OWNERLESS = ContinuationEventContext(
    owner_message_id=None,
    run_id="run-1",
    agent_kind="primary",
    durability="persistent",
)


def _restore(checkpoint, target, *, expand=None, context=None):
    deps = _deps(
        [ModelTurn(text="must not run")],
        order=[],
        persist=lambda event: None,
        invoke=lambda call: pytest.fail("no tool may run after a failed restore"),
        expand=expand,
        context=context,
    )
    return run_agent_loop(
        CONFIG,
        [{"role": "user", "content": "go"}],
        [CALCULATOR],
        deps,
        restore_provider_continuation=checkpoint,
        restore_provider_target=target,
        resume_provider_continuation=True,
    )


@pytest.mark.parametrize(
    ("target", "context", "expand", "expected"),
    [
        (None, None, None, "restore_target_invalid"),
        (
            ContinuationRestoreTarget(
                "deepseek", "deepseek-v4-pro", "responses", "https://api.deepseek.com/v1"
            ),
            None,
            None,
            "restore_target_invalid",
        ),
        (_DEEPSEEK_TARGET, _OWNERLESS, None, "restore_context_invalid"),
        (_DEEPSEEK_TARGET, None, None, "restore_history_unavailable"),
        (
            _DEEPSEEK_TARGET,
            None,
            lambda checkpoint: "not a row list",
            "restore_history_unavailable",
        ),
    ],
    ids=[
        "no target",
        "another model",
        "ownerless context",
        "no expander",
        "expander returns no rows",
    ],
)
def test_each_restore_exit_records_its_own_reason(
    target, context, expand, expected, warnings_log
) -> None:
    outcome = _restore(_restore_checkpoint(), target, expand=expand, context=context)

    assert outcome.status == RUN_ERROR
    assert _error_states(outcome) == {"continuation_error": expected}
    assert any(f"reason={expected}" in line for line in warnings_log), warnings_log


@pytest.mark.parametrize(
    ("failing_event", "expected"),
    [
        (ToolCallExecuting, "executing_transition_failed"),
        (ToolCallFinished, "result_transition_failed"),
    ],
)
def test_each_tool_transition_exit_records_its_own_reason(
    failing_event, expected, warnings_log
) -> None:
    """The batch barrier holds, then one call's transition cannot be stored."""

    def persist(event) -> None:
        if isinstance(event, failing_event):
            raise RuntimeError("Continuation revision conflict.")

    call = ToolCall("calculator", {"expression": "2+2"}, "a", '{"expression":"2+2"}')
    initial = _kimi_checkpoint(
        revision=1,
        state="active",
        rounds=(
            ContinuationRound(
                "",
                ("tool reasoning",),
                (
                    ContinuationCall(
                        "a", "calculator", '{"expression":"2+2"}', "pending"
                    ),
                ),
            ),
        ),
    )
    invoked: list[ToolCall] = []

    outcome = _run(
        [_native_turn((call,), initial)],
        persist=persist,
        invoke=lambda actual: invoked.append(actual) or ToolResult(ok=True, content="4"),
    )

    assert outcome.status == RUN_ERROR
    assert _error_states(outcome) == {
        "continuation_error": expected,
        "continuation_store_failure": "revision_conflict",
    }
    # A call whose "executing" mark was not stored never runs.
    assert len(invoked) == (0 if failing_event is ToolCallExecuting else 1)
    assert any(f"reason={expected}" in line for line in warnings_log), warnings_log


def test_store_failure_is_logged_by_code_never_by_text(warnings_log) -> None:
    """The store's exception is logged, but only as code-owned metadata."""

    def persist(event) -> None:
        if isinstance(event, ToolBatchReady):
            raise RuntimeError("Continuation revision conflict.")

    call = ToolCall("calculator", {"expression": "2+2"}, "a", '{"expression":"2+2"}')
    initial = _kimi_checkpoint(
        revision=1,
        state="active",
        rounds=(
            ContinuationRound(
                "",
                ("tool reasoning",),
                (
                    ContinuationCall(
                        "a", "calculator", '{"expression":"2+2"}', "pending"
                    ),
                ),
            ),
        ),
    )

    outcome = _run(
        [_native_turn((call,), initial)],
        persist=persist,
        invoke=lambda actual: pytest.fail("no tool may run after a failed barrier"),
    )

    assert outcome.status == RUN_ERROR
    assert _error_states(outcome) == {
        "continuation_error": "batch_persist_failed",
        "continuation_store_failure": "revision_conflict",
    }
    joined = "\n".join(warnings_log)
    assert "store_failure=revision_conflict" in joined
    assert "error_type=RuntimeError" in joined
    assert "Continuation revision conflict." not in joined


def test_unknown_store_text_is_never_logged(warnings_log) -> None:
    def persist(event) -> None:
        raise RuntimeError("PRIVATE-CANARY reasoning text")

    call = ToolCall("calculator", {"expression": "2+2"}, "a", '{"expression":"2+2"}')
    initial = _kimi_checkpoint(
        revision=1,
        state="active",
        rounds=(
            ContinuationRound(
                "",
                ("tool reasoning",),
                (
                    ContinuationCall(
                        "a", "calculator", '{"expression":"2+2"}', "pending"
                    ),
                ),
            ),
        ),
    )

    outcome = _run(
        [_native_turn((call,), initial)],
        persist=persist,
        invoke=lambda actual: pytest.fail("no tool may run after a failed barrier"),
    )

    assert _error_states(outcome)["continuation_store_failure"] == (
        UNRECOGNIZED_STORE_FAILURE
    )
    assert all("PRIVATE-CANARY" not in line for line in warnings_log)
    assert all("PRIVATE-CANARY" not in step.summary for step in outcome.steps)


def test_store_failure_code_ignores_non_runtime_text() -> None:
    assert store_failure_code(ValueError("Continuation revision conflict.")) == (
        UNRECOGNIZED_STORE_FAILURE
    )
    assert store_failure_code(RuntimeError("Continuation revision conflict.")) == (
        "revision_conflict"
    )


def _literal_strings(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        return []
    return []


def _store_precondition_messages() -> set[str]:
    """Every fixed failure text on the store's continuation persist path."""
    messages: set[str] = set()
    for path, wanted in _PERSIST_PATH.items():
        messages.update(_failure_messages(path, wanted))
    return messages


def _failure_messages(path: Path, wanted: set[str]) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    }
    assert found == wanted, f"{path.name} lost {sorted(wanted - found)}"
    messages: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in wanted:
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Raise)
                and isinstance(inner.exc, ast.Call)
                and isinstance(inner.exc.func, ast.Name)
                and inner.exc.func.id == "RuntimeError"
                and inner.exc.args
            ):
                messages.update(_literal_strings(inner.exc.args[0]))
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "ContinuationDurabilityResult"
                and len(inner.args) == 2
                and isinstance(inner.args[0], ast.Constant)
                and inner.args[0].value is False
            ):
                messages.update(_literal_strings(inner.args[1]))
    return messages


def test_every_store_precondition_message_has_a_code() -> None:
    """A new store failure text must be given a code, or it logs 'unrecognized'."""
    messages = _store_precondition_messages()

    assert len(messages) >= 20, sorted(messages)
    missing = sorted(
        message for message in messages if message not in STORE_FAILURE_CODES
    )
    assert missing == []
    assert all(
        code.replace("_", "").isalnum() and code.islower()
        for code in STORE_FAILURE_CODES.values()
    )


# --- AC#2: the optional reasoning-replay checkpoint never costs the reply. ---


def _first_final_turn() -> ModelTurn:
    return ModelTurn(
        text="answer",
        provider_continuation=_kimi_checkpoint(
            revision=1, state="complete", rounds=(_final_round(),)
        ),
    )


def _run_with_trace(turns, *, persist):
    order: list[str] = []
    traced: list[object] = []
    deps = _deps(
        turns,
        order=order,
        persist=persist,
        invoke=lambda call: pytest.fail("no tool expected"),
    )
    deps.on_trace_step = traced.append
    outcome = run_agent_loop(
        CONFIG, [{"role": "user", "content": "go"}], [CALCULATOR], deps
    )
    return outcome, traced


@pytest.mark.parametrize(
    ("store_text", "store_code"),
    [
        (
            "Provider continuation terminal settlement failed.",
            "terminal_settlement_failed",
        ),
        ("Continuation revision conflict.", "revision_conflict"),
        ("Durable continuation storage is unavailable.", "create_unavailable"),
    ],
)
def test_unstored_reasoning_checkpoint_keeps_the_delivered_reply(
    store_text, store_code, warnings_log
) -> None:
    """A tool-free first final whose optional checkpoint was not written."""

    def persist(event) -> None:
        raise RuntimeError(store_text)

    outcome, traced = _run_with_trace([_first_final_turn()], persist=persist)

    assert outcome.status == RUN_DONE
    assert outcome.final_text == "answer"
    assert not [step for step in outcome.steps if step.kind == STEP_ERROR]
    notes = [
        step for step in traced if step.summary == "Reasoning not preserved for replay"
    ]
    assert len(notes) == 1
    assert notes[0].field_states["payload"] == "omitted"
    assert _reasons(notes[0].field_states) == {
        "continuation_error": "final_persist_failed",
        "continuation_store_failure": store_code,
    }
    assert any(
        f"Reasoning not preserved for replay: store_failure={store_code}" in line
        for line in warnings_log
    )


@pytest.mark.parametrize(
    ("store_text", "store_code"),
    [
        # Raised after the store committed the row: not a clean "unwritten".
        (
            "Local continuation intent is stale or unavailable; save and retry.",
            "local_intent_stale",
        ),
        ("Dispatch recovery owner changed during settlement.", "settlement_owner_changed"),
        # Raised after the managed finalizer ran (TASK-37653's pre-seal).
        (
            "Continuation owner changed while sealing; reload and retry.",
            "seal_owner_changed",
        ),
        # Unknown text: fail closed, as before.
        ("PRIVATE-CANARY", UNRECOGNIZED_STORE_FAILURE),
    ],
)
def test_possibly_written_checkpoint_still_fails_closed(store_text, store_code) -> None:
    def persist(event) -> None:
        raise RuntimeError(store_text)

    outcome, traced = _run_with_trace([_first_final_turn()], persist=persist)

    assert outcome.status == RUN_ERROR
    assert _error_states(outcome) == {
        "continuation_error": "final_persist_failed",
        "continuation_store_failure": store_code,
    }
    assert not [
        step for step in traced if step.summary == "Reasoning not preserved for replay"
    ]


@pytest.mark.parametrize(
    "acknowledge_text",
    [
        # Each would name an unwritten store failure if it reached the code
        # table; raised after the commit, it must not.
        "Continuation revision conflict.",
        "Provider continuation terminal settlement failed.",
        "PRIVATE-CANARY",
    ],
)
def test_acknowledge_failure_after_the_commit_still_fails_closed(
    acknowledge_text, warnings_log
) -> None:
    """Pre-merge review M6: the acknowledge hook runs after the store committed.

    A failure there is not an unwritten checkpoint, whatever its text, so the
    keep-reply path must never say "Reasoning not preserved for replay": the
    reasoning WAS saved. It gets its own code, outside the unwritten set.
    """
    persisted: list[object] = []

    def acknowledge(original, event, model_result) -> None:
        raise RuntimeError(acknowledge_text)

    order: list[str] = []
    traced: list[object] = []
    deps = _deps(
        [_first_final_turn()],
        order=order,
        persist=persisted.append,
        invoke=lambda call: pytest.fail("no tool expected"),
    )
    deps.acknowledge_provider_continuation = acknowledge
    deps.on_trace_step = traced.append
    outcome = run_agent_loop(
        CONFIG, [{"role": "user", "content": "go"}], [CALCULATOR], deps
    )

    assert len(persisted) == 1, "the store committed the checkpoint"
    assert outcome.status == RUN_ERROR
    assert _error_states(outcome) == {
        "continuation_error": "final_persist_failed",
        "continuation_store_failure": ACKNOWLEDGE_FAILED,
    }
    assert ACKNOWLEDGE_FAILED not in UNWRITTEN_STORE_FAILURES
    assert not [
        step for step in traced if step.summary == "Reasoning not preserved for replay"
    ]
    assert any(
        f"store_failure={ACKNOWLEDGE_FAILED}" in line for line in warnings_log
    ), warnings_log
    assert not any("PRIVATE-CANARY" in line for line in warnings_log)


@pytest.mark.parametrize(
    ("store_text", "store_code"),
    [
        ("Durable continuation host target is unavailable.", "host_target_unavailable"),
        ("Durable continuation destination is unavailable.", "destination_unavailable"),
        ("Provider continuation destination mismatch.", "destination_mismatch"),
        ("Provider continuation binding cannot be erased.", "binding_erased"),
        (
            "Managed continuation finalizer is unavailable.",
            "managed_finalizer_unavailable",
        ),
    ],
)
def test_destination_binding_refusals_still_fail_closed(store_text, store_code) -> None:
    """TASK-35582's binding checks guard the destination the user accepted.

    They refuse before any write, but a refusal means the checkpoint names a
    different provider, origin or binding than the accepted dispatch, so the
    run must not quietly complete as an ordinary reply (review v5r1 F9).
    """

    def persist(event) -> None:
        raise RuntimeError(store_text)

    outcome, traced = _run_with_trace([_first_final_turn()], persist=persist)

    assert outcome.status == RUN_ERROR
    assert _error_states(outcome) == {
        "continuation_error": "final_persist_failed",
        "continuation_store_failure": store_code,
    }
    assert not [
        step for step in traced if step.summary == "Reasoning not preserved for replay"
    ]


def test_post_tool_final_persist_failure_still_fails_closed() -> None:
    """Only the first, tool-free checkpoint is optional; a tool run's is not."""

    def persist(event) -> None:
        if isinstance(event, FinalContinuation):
            raise RuntimeError("Provider continuation terminal settlement failed.")

    final = _kimi_checkpoint(
        revision=4,
        state="complete",
        rounds=(_completed_tool_round(), _final_round("visible answer")),
    )
    outcome = _run(
        _tool_then_final(final),
        persist=persist,
        invoke=lambda call: ToolResult(ok=True, content="4"),
    )

    assert outcome.status == RUN_ERROR
    assert _error_states(outcome) == {
        "continuation_error": "final_persist_failed",
        "continuation_store_failure": "terminal_settlement_failed",
    }


# --- Review round 1 (F1): the reason codes survive the persisted run trace. ---


def _revision_conflict_on_batch(event) -> None:
    if isinstance(event, ToolBatchReady):
        raise RuntimeError("Continuation revision conflict.")


def _pending_tool_turn() -> ModelTurn:
    call = ToolCall("calculator", {"expression": "2+2"}, "a", '{"expression":"2+2"}')
    pending = ContinuationRound(
        "",
        ("tool reasoning",),
        (ContinuationCall("a", "calculator", '{"expression":"2+2"}', "pending"),),
    )
    return _native_turn(
        (call,), _kimi_checkpoint(revision=1, state="active", rounds=(pending,))
    )


def _terminal_settlement_conflict(event) -> None:
    raise RuntimeError("Provider continuation terminal settlement failed.")


@pytest.mark.parametrize(
    "profile", ["safe_summary", "redacted_diagnostic", "full_trace"]
)
def test_reason_codes_survive_trace_export_and_import(tmp_path, profile) -> None:
    """A run that recorded a reason code can still be exported and re-imported.

    The codes used to be stored as ``field_states`` VALUES, which the Trace
    export and import reject (only observed/redacted/truncated/omitted/
    not_available/capture_failed are field states). So the one conversation a
    user would share to report the failure could not be exported at all.
    This drives the real runtime exits, the real AgentRunsDB rows and the
    real derive -> preflight -> build -> import path.
    """
    from tldw_chatbook.Agents.agent_service import _safe_agent_step_record
    from tldw_chatbook.Chat.provider_continuation_diagnostics import (
        continuation_reasons,
    )
    from tldw_chatbook.Chat.trajectory import derive_trajectory
    from tldw_chatbook.Chat.trajectory_export import (
        build_trace_export,
        preflight_trace_export,
    )
    from tldw_chatbook.Chat.trajectory_import import load_imported_trace
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    failed = _run(
        [_pending_tool_turn()],
        persist=_revision_conflict_on_batch,
        invoke=lambda call: pytest.fail("no tool may run after a failed barrier"),
    )
    kept, traced = _run_with_trace(
        [_first_final_turn()], persist=_terminal_settlement_conflict
    )
    assert failed.status == RUN_ERROR
    assert kept.status == RUN_DONE

    db = AgentRunsDB(tmp_path / "agent_runs.db")
    try:
        for steps in (failed.steps, [*kept.steps, *traced]):
            run_id = db.create_run(conversation_id="conv-34100", agent_kind="primary")
            db.insert_steps_at_indices(
                run_id,
                [(step.index, _safe_agent_step_record(run_id, step)) for step in steps],
            )
        runs = [dict(run) for run in db.list_runs("conv-34100", agent_kind="primary")]
    finally:
        db.close()
    persisted = [
        {**step, "run_id": run["id"], "conversation_id": "conv-34100"}
        for run in runs
        for step in run["steps"]
    ]
    snapshot = derive_trajectory(
        [], {}, [], [], [], agent_runs=runs, agent_steps=persisted
    )

    preflight = preflight_trace_export(snapshot, profile=profile)
    bundle = build_trace_export(
        snapshot, preflight=preflight, confirm_full=profile == "full_trace"
    )
    imported = load_imported_trace(bundle)

    reasons = sorted(
        tuple(sorted(continuation_reasons(record.field_states).items()))
        for turn in imported.snapshot.turns
        for record in turn.records
        if continuation_reasons(record.field_states)
    )
    assert reasons == sorted(
        [
            (
                ("continuation_error", "batch_persist_failed"),
                ("continuation_store_failure", "revision_conflict"),
            ),
            (
                ("continuation_error", "final_persist_failed"),
                ("continuation_store_failure", "terminal_settlement_failed"),
            ),
        ]
    )


# --- Review round 1 (F3/TE-1): every exit keeps its own code. ---

_RUNTIME = _REPO_ROOT / "tldw_chatbook" / "Agents" / "agent_runtime.py"


def _runtime_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_every_continuation_error_exit_passes_a_distinct_code() -> None:
    """A copy-pasted code would make two exits indistinguishable in the log.

    Every ``continuation_error(...)`` call in ``run_agent_loop`` passes either
    its own string literal or the result of ``_first_final_continuation_problem``
    (whose return literals are themselves distinct). Nothing else may name an
    exit, so a new call site must bring a new code.
    """
    tree = ast.parse(_RUNTIME.read_text(encoding="utf-8"))
    loop = _runtime_function(tree, "run_agent_loop")
    literals: list[str] = []
    shape_sites = 0
    for node in ast.walk(loop):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "continuation_error"
        ):
            continue
        assert len(node.args) == 1 and not node.keywords, ast.unparse(node)
        (argument,) = node.args
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            literals.append(argument.value)
        else:
            assert isinstance(argument, ast.Name), ast.unparse(node)
            assert argument.id == "shape_problem", ast.unparse(node)
            shape_sites += 1
    helper = _runtime_function(tree, "_first_final_continuation_problem")
    shape_codes = [
        node.value.value
        for node in ast.walk(helper)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]

    assert shape_sites == 1
    # 20 exits, plus dev's instruction-deferral transition (105f593355).
    assert len(literals) == 21, sorted(literals)
    assert len(shape_codes) == 6, shape_codes
    duplicates = sorted(
        code for code in set(literals) if literals.count(code) > 1
    ) + sorted(code for code in set(shape_codes) if shape_codes.count(code) > 1)
    assert duplicates == []
    assert set(literals).isdisjoint(shape_codes)
    assert all(
        code.replace("_", "").isalnum() and code.islower()
        for code in (*literals, *shape_codes)
    )


def _pending_round(call_id: str = "a") -> ContinuationRound:
    return ContinuationRound(
        "",
        ("tool reasoning",),
        (ContinuationCall(call_id, "calculator", '{"expression":"2+2"}', "pending"),),
    )


def _completed_call_round(content: str = "answer") -> ContinuationRound:
    return ContinuationRound(
        content,
        ("private",),
        (
            ContinuationCall(
                "a",
                "calculator",
                '{"expression":"2+2"}',
                "completed",
                ContinuationResult("4"),
            ),
        ),
    )


def _deepseek_complete_checkpoint() -> ProviderContinuationCheckpoint:
    return ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=1,
        provider="deepseek",
        protocol="responses",
        model="deepseek-v4-flash",
        api_base_url="https://api.deepseek.com/v1",
        state="complete",
        rounds=(_completed_call_round(),),
    )


# Each candidate below is a checkpoint the serializer accepts (a call-free
# round must be a moonshot reasoning round, so most shapes carry a completed
# call), so the run reaches the first-final shape checks themselves.
@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        pytest.param(
            _kimi_checkpoint(revision=1, state="active", rounds=(_pending_round(),)),
            "final_state_not_complete",
            id="state-not-complete",
        ),
        pytest.param(
            _deepseek_complete_checkpoint(),
            "final_first_provider_unsupported",
            id="provider-unsupported",
        ),
        pytest.param(
            _kimi_checkpoint(
                revision=1,
                state="complete",
                rounds=(_completed_call_round(),),
                model="kimi-latest",
            ),
            "final_first_model_unsupported",
            id="model-unsupported",
        ),
        pytest.param(
            _kimi_checkpoint(
                revision=1,
                state="complete",
                rounds=(_completed_call_round(""), _final_round()),
            ),
            "final_first_round_count_invalid",
            id="round-count",
        ),
        pytest.param(
            _kimi_checkpoint(
                revision=1,
                state="complete",
                rounds=(_completed_call_round(),),
                model="kimi-k2.6",
            ),
            "final_first_round_has_calls",
            id="round-has-calls",
        ),
    ],
)
def test_remaining_tool_free_final_exits_record_their_reason(
    checkpoint, expected, warnings_log
) -> None:
    events: list[object] = []

    outcome = _run(
        [ModelTurn(text="answer", provider_continuation=checkpoint)],
        persist=events.append,
    )

    assert outcome.status == RUN_ERROR
    assert events == []
    assert _error_states(outcome) == {"continuation_error": expected}
    assert any(f"reason={expected}" in line for line in warnings_log), warnings_log


def _tool_turn(checkpoint: ProviderContinuationCheckpoint) -> ModelTurn:
    call = ToolCall("calculator", {"expression": "2+2"}, "a", '{"expression":"2+2"}')
    return _native_turn((call,), checkpoint)


@pytest.mark.parametrize(
    ("turns", "context", "expected"),
    [
        pytest.param(
            [
                _tool_turn(
                    _kimi_checkpoint(
                        revision=1, state="active", rounds=(_pending_round(),)
                    )
                )
            ],
            ContinuationEventContext(
                owner_message_id=None,
                run_id="run-1",
                agent_kind="primary",
                durability="persistent",
            ),
            "batch_context_invalid",
            id="batch-context-invalid",
        ),
        pytest.param(
            [
                _tool_turn(
                    _kimi_checkpoint(
                        revision=1, state="active", rounds=(_pending_round("b"),)
                    )
                )
            ],
            None,
            "batch_calls_mismatch",
            id="batch-calls-mismatch",
        ),
        pytest.param(
            [
                _tool_turn(
                    _kimi_checkpoint(
                        revision=2, state="active", rounds=(_pending_round(),)
                    )
                )
            ],
            None,
            "batch_revision_invalid",
            id="batch-revision-invalid",
        ),
        pytest.param(
            [
                _tool_turn(
                    _kimi_checkpoint(
                        revision=1, state="active", rounds=(_pending_round(),)
                    )
                ),
                ModelTurn(
                    tool_calls=(
                        ToolCall(
                            "calculator",
                            {"expression": "3+3"},
                            "b",
                            '{"expression":"3+3"}',
                        ),
                    ),
                    assistant_message={"role": "assistant", "content": ""},
                ),
            ],
            None,
            "batch_checkpoint_missing",
            id="batch-checkpoint-missing",
        ),
    ],
)
def test_batch_exits_record_their_reason(turns, context, expected) -> None:
    outcome = _run(
        turns,
        persist=lambda event: None,
        context=context,
        invoke=lambda call: ToolResult(ok=True, content="4"),
    )

    assert outcome.status == RUN_ERROR
    assert _error_states(outcome) == {"continuation_error": expected}


def test_store_failure_cause_class_is_logged(warnings_log) -> None:
    """The store's chained cause is named by class, never by its text."""

    def persist(event) -> None:
        if isinstance(event, ToolBatchReady):
            try:
                raise ValueError("PRIVATE-CANARY cause")
            except ValueError as cause:
                raise RuntimeError("Continuation revision conflict.") from cause

    outcome = _run(
        [_pending_tool_turn()],
        persist=persist,
        invoke=lambda call: pytest.fail("no tool may run after a failed barrier"),
    )

    assert outcome.status == RUN_ERROR
    joined = "\n".join(warnings_log)
    assert "cause_type=ValueError" in joined
    assert "PRIVATE-CANARY" not in joined
