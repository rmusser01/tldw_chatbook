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
    ToolResult,
)
from tldw_chatbook.Agents.agent_runtime import run_agent_loop
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationCall,
    ContinuationResult,
    ContinuationRound,
    ProviderContinuationCheckpoint,
)
from tldw_chatbook.Chat.provider_continuation_diagnostics import (
    STORE_FAILURE_CODES,
    UNRECOGNIZED_STORE_FAILURE,
    store_failure_code,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STORE = _REPO_ROOT / "tldw_chatbook" / "Chat" / "console_chat_store.py"


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


def _error_states(outcome) -> dict[str, str]:
    errors = [step for step in outcome.steps if step.kind == STEP_ERROR]
    assert errors, "the run must record an error step"
    return dict(errors[-1].field_states)


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
    tree = ast.parse(_STORE.read_text(encoding="utf-8"))
    wanted = {
        "persist_provider_continuation_event",
        "_continuation_event_value",
        "ensure_provider_continuation_durable",
        "_settle_dispatch_recovery",
    }
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
    assert notes[0].field_states == {
        "payload": "omitted",
        "continuation_error": "final_persist_failed",
        "continuation_store_failure": store_code,
    }
    assert any(
        f"Reasoning not preserved for replay: store_failure={store_code}" in line
        for line in warnings_log
    )


@pytest.mark.parametrize(
    "store_text",
    [
        # Raised after the store committed the row: not a clean "unwritten".
        "Local continuation intent is stale or unavailable; save and retry.",
        "Dispatch recovery owner changed during settlement.",
        # Unknown text: fail closed, as before.
        "PRIVATE-CANARY",
    ],
)
def test_possibly_written_checkpoint_still_fails_closed(store_text) -> None:
    def persist(event) -> None:
        raise RuntimeError(store_text)

    outcome, traced = _run_with_trace([_first_final_turn()], persist=persist)

    assert outcome.status == RUN_ERROR
    assert _error_states(outcome)["continuation_error"] == "final_persist_failed"
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
