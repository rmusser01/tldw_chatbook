"""Incremental, idempotent persistence of agent trace steps."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

import tldw_chatbook.Agents.agent_service as agent_service_module
from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    AgentStep,
    ModelTurn,
    ToolResult,
)
from tldw_chatbook.Agents.agent_runtime import LoopDeps, safe_utc_timestamp
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
import tldw_chatbook.DB.AgentRuns_DB as agent_runs_db_module
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def _reply(text: str = "done") -> dict:
    return {"choices": [{"message": {"content": text}}]}


def _service(db: AgentRunsDB, **kwargs) -> AgentService:
    chat_call = kwargs.pop("chat_call", lambda **_kwargs: _reply())
    return AgentService(
        db,
        ToolCatalogRegistry(),
        chat_call=chat_call,
        **kwargs,
    )


def _run(service: AgentService):
    return service.run_turn(
        conversation_id="conversation-1",
        messages=[{"role": "user", "content": "finish"}],
        config=AgentConfig(model="model", system_prompt="system"),
        api_endpoint="llama_cpp",
    )


@pytest.fixture()
def db(tmp_path) -> AgentRunsDB:
    return AgentRunsDB(tmp_path / "agent-runs.db", client_id="test")


def test_first_step_is_durable_before_run_finalization(db: AgentRunsDB) -> None:
    observed = []

    def on_step(step, _agent_kind, run_id):
        run = db.get_run(run_id)
        observed.append((run["status"], run["steps"], step.index))

    _run_id, outcome = _run(_service(db, on_step=on_step))

    assert outcome.status == "done"
    assert observed[0][0] == "running"
    assert observed[0][1][0]["index"] == observed[0][2] == 0


def test_step_timestamp_uses_injected_utc_wall_clock_not_budget_clock(
    db: AgentRunsDB,
) -> None:
    fixed = datetime(2026, 8, 22, 12, 34, 56, 123456, tzinfo=timezone.utc)
    observed = []
    service = _service(
        db,
        clock=lambda: -1234.5,
        wall_clock=lambda: fixed,
        on_step=lambda step, _kind, _run_id: observed.append(step.created_at),
    )

    _run(service)

    assert observed == ["2026-08-22T12:34:56.123456Z"]
    parsed = datetime.fromisoformat(observed[0].replace("Z", "+00:00"))
    assert parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def test_loopdeps_preserves_pre_wall_clock_positional_continuation_slot() -> None:
    def continuation(_messages, _schemas, _checkpoint):
        return ModelTurn(text="done")

    deps = LoopDeps(
        lambda _messages, _schemas: ModelTurn(text="done"),
        lambda _call: ToolResult(ok=True),
        lambda _task: ToolResult(ok=True),
        lambda _query: [],
        lambda _ids: [],
        lambda: False,
        lambda: 0.0,
        continuation,
    )

    assert deps.call_model_with_continuation is continuation


def test_raising_wall_clock_falls_back_without_aborting_run(db: AgentRunsDB) -> None:
    def fail_wall_clock():
        raise RuntimeError("clock unavailable")

    run_id, outcome = _run(_service(db, wall_clock=fail_wall_clock))

    durable_timestamp = db.get_run(run_id)["steps"][0]["created_at"]
    assert outcome.status == "done"
    assert outcome.steps[0].created_at == durable_timestamp
    parsed = datetime.fromisoformat(durable_timestamp.replace("Z", "+00:00"))
    assert parsed.utcoffset() == timezone.utc.utcoffset(parsed)


@pytest.mark.parametrize(
    "invalid_value",
    [None, "2026-08-22T13:00:00Z", datetime(2001, 1, 1, 0, 0, 0)],
)
def test_safe_utc_timestamp_rejects_invalid_or_naive_clock_values(
    invalid_value,
) -> None:
    before = datetime.now(timezone.utc)
    timestamp = safe_utc_timestamp(lambda: invalid_value)
    after = datetime.now(timezone.utc)

    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    assert before <= parsed <= after


def test_safe_utc_timestamp_converts_aware_non_utc_value() -> None:
    ist = timezone(timedelta(hours=5, minutes=30))

    timestamp = safe_utc_timestamp(
        lambda: datetime(2026, 8, 22, 18, 30, 0, 123456, tzinfo=ist)
    )

    assert timestamp == "2026-08-22T13:00:00.123456Z"


def test_terminal_error_step_without_live_timestamp_uses_wall_clock_fallback(
    db: AgentRunsDB,
) -> None:
    fixed = datetime(2026, 8, 22, 13, 0, 0, tzinfo=timezone.utc)

    def fail_provider(**_kwargs):
        raise RuntimeError("provider failed")

    run_id, outcome = _run(
        _service(db, chat_call=fail_provider, wall_clock=lambda: fixed)
    )

    assert outcome.status == "error"
    durable_timestamp = db.get_run(run_id)["steps"][0]["created_at"]
    assert outcome.steps[0].created_at == durable_timestamp
    assert durable_timestamp == "2026-08-22T13:00:00.000000Z"


def test_live_serialization_failure_still_notifies_ui_and_finalizes_status(
    db: AgentRunsDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = []
    serialize = agent_service_module.dataclasses.asdict

    def fail_step_serialization(value):
        if isinstance(value, AgentStep):
            raise TypeError("cannot serialize")
        return serialize(value)

    monkeypatch.setattr(
        agent_service_module.dataclasses, "asdict", fail_step_serialization
    )
    run_id, outcome = _run(
        _service(
            db,
            on_step=lambda step, _kind, _run_id: observed.append(step.index),
        )
    )

    run = db.get_run(run_id)
    assert outcome.status == "done"
    assert observed == [0]
    assert run["status"] == "done" and run["result"] == "done"


def test_live_and_terminal_trace_write_failure_still_finalizes_status(
    db: AgentRunsDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    warnings = []

    class UnprintableTraceError(RuntimeError):
        def __str__(self):
            raise AssertionError("exception text must not be rendered")

    def fail_trace_write(_run_id, _steps):
        raise UnprintableTraceError("SECRET_TRACE_PAYLOAD")

    monkeypatch.setattr(db, "insert_steps_at_indices", fail_trace_write)
    monkeypatch.setattr(
        agent_service_module.logger,
        "warning",
        lambda message, *args: warnings.append((message, args)),
    )
    run_id, outcome = _run(_service(db))

    run = db.get_run(run_id)
    assert outcome.status == "done"
    assert run["status"] == "done" and run["result"] == "done"
    assert run["steps"] == []
    assert len(warnings) == 2
    assert all("UnprintableTraceError" in args for _message, args in warnings)
    assert all("SECRET_TRACE_PAYLOAD" not in repr(item) for item in warnings)


def test_failed_incremental_write_does_not_abort_and_terminal_write_recovers(
    db: AgentRunsDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    durable_insert = db.insert_steps_at_indices
    calls = 0
    replies = [
        _reply(
            "```tool_call\n"
            + json.dumps({"name": "missing", "arguments": {}})
            + "\n```"
        ),
        _reply(),
    ]

    def fail_once(run_id, steps):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated transient write failure")
        return durable_insert(run_id, steps)

    monkeypatch.setattr(db, "insert_steps_at_indices", fail_once)

    run_id, outcome = _run(
        _service(db, chat_call=lambda **_kwargs: replies.pop(0))
    )

    assert outcome.status == "done"
    assert calls == len(outcome.steps) + 1
    assert [step["index"] for step in db.get_run(run_id)["steps"]] == [
        step.index for step in outcome.steps
    ]


def test_ui_callback_failure_does_not_abort_incremental_durability(
    db: AgentRunsDB,
) -> None:
    def fail_ui(_step, _agent_kind, _run_id):
        raise RuntimeError("broken UI callback")

    run_id, outcome = _run(_service(db, on_step=fail_ui))

    assert outcome.status == "done"
    assert [step["index"] for step in db.get_run(run_id)["steps"]] == [0]


def test_terminal_recovery_does_not_duplicate_successful_incremental_rows(
    db: AgentRunsDB,
) -> None:
    replies = [
        _reply(
            "```tool_call\n"
            + json.dumps({"name": "missing", "arguments": {}})
            + "\n```"
        ),
        _reply(),
    ]
    run_id, outcome = _run(
        _service(db, chat_call=lambda **_kwargs: replies.pop(0))
    )

    expected_indices = [step.index for step in outcome.steps]
    assert len(expected_indices) > 1
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT seq, payload FROM agent_run_steps WHERE run_id = ? ORDER BY seq",
            (run_id,),
        ).fetchall()
    assert [row["seq"] for row in rows] == expected_indices
    assert [step["index"] for step in db.get_run(run_id)["steps"]] == expected_indices


def test_explicit_index_insert_is_idempotent_and_first_writer_wins(
    db: AgentRunsDB,
) -> None:
    run_id = db.create_run(conversation_id="c", agent_kind="primary")
    original = {"index": 4, "kind": "model", "summary": "original"}
    same_value_different_order = {
        "summary": "original",
        "kind": "model",
        "index": 4,
    }

    db.insert_steps_at_indices(run_id, [(4, original)])
    db.insert_steps_at_indices(run_id, [(4, same_value_different_order)])

    with db.connection() as conn:
        rows = conn.execute(
            "SELECT seq, payload FROM agent_run_steps WHERE run_id = ?", (run_id,)
        ).fetchall()
    assert [row["seq"] for row in rows] == [4]
    assert rows[0]["payload"] == (
        '{"index":4,"kind":"model","summary":"original"}'
    )
    assert db.get_run(run_id)["steps"] == [original]


def test_explicit_index_insert_rejects_divergent_stored_payload(
    db: AgentRunsDB,
) -> None:
    assert hasattr(agent_runs_db_module, "AgentStepConflictError")
    run_id = db.create_run(conversation_id="c", agent_kind="primary")
    db.insert_steps_at_indices(
        run_id, [(4, {"index": 4, "kind": "model", "summary": "original"})]
    )

    with pytest.raises(agent_runs_db_module.AgentStepConflictError):
        db.insert_steps_at_indices(
            run_id,
            [(4, {"index": 4, "kind": "model", "summary": "replacement"})],
        )

    assert db.get_run(run_id)["steps"][0]["summary"] == "original"


def test_mixed_terminal_recovery_commits_missing_rows_before_reporting_conflict(
    db: AgentRunsDB,
) -> None:
    run_id = db.create_run(conversation_id="c", agent_kind="primary")
    db.insert_steps_at_indices(
        run_id, [(0, {"index": 0, "kind": "model", "summary": "durable"})]
    )

    with pytest.raises(agent_runs_db_module.AgentStepConflictError) as raised:
        db.insert_steps_at_indices(
            run_id,
            [
                (
                    0,
                    {
                        "index": 0,
                        "kind": "model",
                        "summary": "SECRET_DIVERGENT_PAYLOAD",
                    },
                ),
                (1, {"index": 1, "kind": "tool_call", "summary": "missing"}),
            ],
        )

    assert "0" in str(raised.value)
    assert "SECRET_DIVERGENT_PAYLOAD" not in str(raised.value)
    assert db.get_run(run_id)["steps"] == [
        {"index": 0, "kind": "model", "summary": "durable"},
        {"index": 1, "kind": "tool_call", "summary": "missing"},
    ]


def test_explicit_index_insert_deduplicates_identical_batch_entries(
    db: AgentRunsDB,
) -> None:
    run_id = db.create_run(conversation_id="c", agent_kind="primary")
    first = {"index": 2, "kind": "model", "summary": "same"}
    reordered = {"summary": "same", "kind": "model", "index": 2}

    db.insert_steps_at_indices(run_id, [(2, first), (2, reordered)])

    assert db.get_run(run_id)["steps"] == [first]


def test_explicit_index_insert_rejects_divergent_duplicate_batch_entries(
    db: AgentRunsDB,
) -> None:
    assert hasattr(agent_runs_db_module, "AgentStepConflictError")
    run_id = db.create_run(conversation_id="c", agent_kind="primary")

    with pytest.raises(agent_runs_db_module.AgentStepConflictError):
        db.insert_steps_at_indices(
            run_id,
            [
                (2, {"index": 2, "kind": "model", "summary": "first"}),
                (2, {"index": 2, "kind": "model", "summary": "second"}),
            ],
        )

    assert db.get_run(run_id)["steps"] == []


@pytest.mark.parametrize(
    ("seq", "payload", "error"),
    [
        (True, {"index": True}, TypeError),
        (-1, {"index": -1}, ValueError),
        ("0", {"index": 0}, TypeError),
        (0.0, {"index": 0}, TypeError),
        (0, [], TypeError),
        (0, {}, ValueError),
        (0, {"index": True}, TypeError),
        (0, {"index": "0"}, TypeError),
        (0, {"index": 0.0}, TypeError),
        (0, {"index": 1}, ValueError),
    ],
)
def test_explicit_index_insert_validates_index_and_payload_before_write(
    db: AgentRunsDB, seq, payload, error
) -> None:
    run_id = db.create_run(conversation_id="c", agent_kind="primary")

    with pytest.raises(error):
        db.insert_steps_at_indices(run_id, [(seq, payload)])

    assert db.get_run(run_id)["steps"] == []


def test_explicit_index_validation_occurs_before_transaction(
    db: AgentRunsDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = db.create_run(conversation_id="c", agent_kind="primary")

    def transaction_started():
        raise AssertionError("transaction opened before validation")

    monkeypatch.setattr(db, "transaction", transaction_started)
    with pytest.raises(ValueError, match="non-negative"):
        db.insert_steps_at_indices(run_id, [(-1, {"index": -1})])


def test_explicit_index_json_serialization_occurs_before_transaction(
    db: AgentRunsDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = db.create_run(conversation_id="c", agent_kind="primary")

    def transaction_started():
        raise AssertionError("transaction opened before serialization")

    monkeypatch.setattr(db, "transaction", transaction_started)
    with pytest.raises(TypeError):
        db.insert_steps_at_indices(
            run_id, [(0, {"index": 0, "value": object()})]
        )


def test_explicit_step_insert_preserves_terminal_lifecycle_timestamp_and_wake(
    db: AgentRunsDB,
) -> None:
    parent_id = db.create_run(conversation_id="wake", agent_kind="primary")
    child_id = db.create_run(
        conversation_id="wake",
        agent_kind="subagent",
        parent_run_id=parent_id,
    )
    db.set_status(child_id, "done", result="collected in turn")
    db.set_status(parent_id, "done", result="parent done")
    with db.connection() as conn:
        conn.execute(
            "UPDATE agent_runs SET updated_at = ? WHERE id = ?",
            ("2000-01-01T00:00:00.000000Z", child_id),
        )
        conn.execute(
            "UPDATE agent_runs SET updated_at = ? WHERE id = ?",
            ("2001-01-01T00:00:00.000000Z", parent_id),
        )

    assert db.undelivered_wake_runs("wake") == []
    db.insert_steps_at_indices(
        child_id, [(0, {"index": 0, "kind": "model", "summary": "late"})]
    )

    assert db.get_run(child_id)["updated_at"] == "2000-01-01T00:00:00.000000Z"
    assert db.undelivered_wake_runs("wake") == []


def test_explicit_index_insert_rejects_unknown_run(db: AgentRunsDB) -> None:
    with pytest.raises(KeyError, match="Unknown run id: missing"):
        db.insert_steps_at_indices(
            "missing", [(0, {"index": 0, "kind": "model"})]
        )


def test_legacy_append_steps_keeps_allocating_after_existing_rows(
    db: AgentRunsDB,
) -> None:
    run_id = db.create_run(conversation_id="c", agent_kind="primary")

    db.append_steps(run_id, [{"index": 10, "kind": "model"}])
    db.append_steps(run_id, [{"index": 11, "kind": "model"}])

    with db.connection() as conn:
        rows = conn.execute(
            "SELECT seq FROM agent_run_steps WHERE run_id = ? ORDER BY seq", (run_id,)
        ).fetchall()
    assert [row["seq"] for row in rows] == [0, 1]
    assert [step["index"] for step in db.get_run(run_id)["steps"]] == [10, 11]
