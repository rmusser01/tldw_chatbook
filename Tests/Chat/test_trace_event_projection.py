"""Contract tests for the Trace v2 causal event projection."""

from __future__ import annotations

from dataclasses import dataclass

from tldw_chatbook.Chat.trajectory import derive_trajectory


def _records(snapshot):
    return [record for turn in snapshot.turns for record in turn.records]


def _snapshot(**sources):
    return derive_trajectory(
        messages=sources.pop("messages", []),
        usage_by_id={},
        traj_rows=sources.pop("traj_rows", []),
        variant_sets=[],
        compaction_records=[],
        **sources,
    )


@dataclass(frozen=True)
class Sidecar:
    message_id: str
    conversation_id: str
    turn_id: str
    seq: int | None
    event_kind: str
    step_started_at: float | None = None
    first_token_at: float | None = None
    completed_at: float | None = None
    model: str | None = None
    provider: str | None = None
    payload_json: str | None = None
    status: str | None = None
    parent_event_id: str | None = None
    source_event_id: str | None = None
    replacement_event_id: str | None = None
    field_states: dict[str, str] | None = None
    sensitivity: str | None = None


def test_message_envelope_has_stable_source_identity_and_owner_sequence() -> None:
    message = {
        "id": "u1",
        "conversation_id": "conv-1",
        "sender": "user",
        "content": "Inspect this run",
        "timestamp": "2026-08-22T12:00:00Z",
        "parent_message_id": None,
        "deleted": False,
    }
    row = Sidecar(
        message_id="u1",
        conversation_id="conv-1",
        turn_id="turn-1",
        seq=17,
        event_kind="user",
        status="complete",
        field_states={"content_preview": "observed"},
        sensitivity="conversation_content",
    )

    record = _records(_snapshot(messages=[message], traj_rows=[row]))[0]

    assert record.event_id == "message:u1"
    assert record.conversation_id == "conv-1"
    assert record.source_seq == 17
    assert record.seq == 1
    assert record.label == "User message"
    assert record.status == "complete"
    assert record.actor_kind == "user"
    assert record.actor_id == "user"
    assert record.turn_id == "turn-1"
    assert record.observed_at == 1_787_400_000.0
    assert record.field_states == {"content_preview": "observed"}
    assert record.sensitivity == "conversation_content"

    reordered = _records(_snapshot(messages=[message], traj_rows=[row]))[0]
    assert reordered.event_id == record.event_id


def test_message_parent_and_sidecar_lineage_links_are_preserved() -> None:
    messages = [
        {
            "id": "u1",
            "sender": "user",
            "content": "first",
            "timestamp": 1.0,
            "parent_message_id": None,
            "deleted": False,
        },
        {
            "id": "a1",
            "sender": "assistant",
            "content": "second",
            "timestamp": 2.0,
            "parent_message_id": "u1",
            "deleted": False,
        },
    ]
    rows = [
        Sidecar("u1", "conv-1", "turn-1", 1, "user"),
        Sidecar("a1", "conv-1", "turn-1", 2, "assistant"),
        Sidecar(
            "a1",
            "conv-1",
            "turn-1",
            3,
            "model_retry",
            parent_event_id="message:a1",
            source_event_id="trajectory:conv-1:2",
            replacement_event_id="trajectory:conv-1:4",
            payload_json='{"attempt": 2}',
        ),
    ]

    by_id = {
        record.event_id: record
        for record in _records(_snapshot(messages=messages, traj_rows=rows))
    }

    assert by_id["message:a1"].parent_event_id == "message:u1"
    retry = by_id["trajectory:conv-1:3"]
    assert retry.kind == "model_retry"
    assert retry.label == "Model retry"
    assert retry.source_seq == 3
    assert retry.parent_event_id == "message:a1"
    assert retry.source_event_id == "trajectory:conv-1:2"
    assert retry.replacement_event_id == "trajectory:conv-1:4"
    assert retry.payload == {"attempt": 2}


def test_agent_run_and_step_adapters_preserve_actor_run_and_field_state() -> None:
    run = {
        "id": "run-1",
        "conversation_id": "conv-1",
        "agent_kind": "subagent",
        "agent_definition_id": "researcher",
        "task": "Find evidence",
        "status": "running",
        "created_at": "2026-08-22T12:00:00Z",
        "source_seq": 4,
        "turn_id": "turn-1",
        "field_states": {"result": "not_available"},
        "sensitivity": "diagnostic",
    }
    step = {
        "run_id": "run-1",
        "conversation_id": "conv-1",
        "index": 9,
        "kind": "tool_call",
        "summary": "Search docs",
        "created_at": "2026-08-22T12:00:01Z",
        "turn_id": "turn-1",
        "status": "started",
    }

    by_id = {
        record.event_id: record
        for record in _records(_snapshot(agent_runs=[run], agent_steps=[step]))
    }

    run_record = by_id["agent-run:run-1"]
    assert run_record.run_id == "run-1"
    assert run_record.actor_kind == "subagent"
    assert run_record.actor_id == "researcher"
    assert run_record.source_seq == 4
    assert run_record.label == "Agent run"
    assert run_record.status == "running"
    assert run_record.field_states == {"result": "not_available"}
    assert run_record.sensitivity == "diagnostic"

    step_record = by_id["agent-step:run-1:9"]
    assert step_record.run_id == "run-1"
    assert step_record.source_seq == 9
    assert step_record.parent_event_id == "agent-run:run-1"
    assert step_record.label == "Tool call"
    assert step_record.status == "started"


def test_retrieval_adapter_uses_fixed_identity_and_safe_metadata() -> None:
    retrieval = {
        "run_id": "rag-7",
        "conversation_id": "conv-1",
        "turn_id": "turn-1",
        "run_ordinal": 2,
        "stage": "hybrid_search",
        "status": "complete",
        "started_at": "2026-08-22T12:00:00Z",
        "ended_at": "2026-08-22T12:00:02Z",
        "field_states": {"query": "redacted"},
        "sensitivity": "retrieval_metadata",
    }

    (record,) = _records(_snapshot(retrieval_runs=[retrieval]))

    assert record.event_id == "retrieval-run:rag-7"
    assert record.kind == "retrieval_run"
    assert record.label == "Retrieval run"
    assert record.source_seq == 2
    assert record.status == "complete"
    assert record.observed_at == 1_787_400_000.0
    assert record.field_states == {"query": "redacted"}
    assert record.sensitivity == "retrieval_metadata"


def test_causal_parent_precedes_concurrent_child() -> None:
    parent_run = {
        "id": "parent",
        "conversation_id": "conv-1",
        "agent_kind": "primary",
        "status": "running",
        "created_at": "2026-08-22T12:00:00Z",
    }
    child_run = {
        "id": "child",
        "conversation_id": "conv-1",
        "agent_kind": "subagent",
        "parent_run_id": "parent",
        "spawn_event_id": "agent-step:parent:4",
        "status": "running",
        # Deliberately earlier: causality must outrank wall-clock ordering.
        "created_at": "2026-08-22T11:59:59Z",
    }
    child_step = {
        "run_id": "child",
        "conversation_id": "conv-1",
        "index": 0,
        "kind": "model",
        "summary": "child working",
        "created_at": "2026-08-22T12:00:01Z",
    }
    parent_spawn_step = {
        "run_id": "parent",
        "conversation_id": "conv-1",
        "index": 4,
        "kind": "spawn",
        "summary": "spawn child",
        "created_at": "2026-08-22T12:00:02Z",
    }

    snapshot = _snapshot(
        agent_runs=[parent_run, child_run],
        agent_steps=[child_step, parent_spawn_step],
    )
    ids = [record.event_id for record in _records(snapshot)]

    assert ids.index("agent-step:parent:4") < ids.index("agent-run:child")
    assert ids.index("agent-run:child") < ids.index("agent-step:child:0")


def test_concurrent_events_use_stable_event_id_tie_breaker() -> None:
    runs = [
        {
            "id": "z-run",
            "conversation_id": "conv-1",
            "agent_kind": "primary",
            "status": "running",
            "created_at": "2026-08-22T12:00:00Z",
        },
        {
            "id": "a-run",
            "conversation_id": "conv-1",
            "agent_kind": "primary",
            "status": "running",
            "created_at": "2026-08-22T12:00:00Z",
        },
    ]

    forward = [record.event_id for record in _records(_snapshot(agent_runs=runs))]
    reverse = [
        record.event_id for record in _records(_snapshot(agent_runs=reversed(runs)))
    ]

    assert forward == reverse == ["agent-run:a-run", "agent-run:z-run"]


def test_cycles_and_orphans_degrade_deterministically_without_dropping_events() -> None:
    runs = [
        {
            "id": "a",
            "conversation_id": "conv-1",
            "parent_event_id": "agent-run:b",
            "created_at": 1,
        },
        {
            "id": "b",
            "conversation_id": "conv-1",
            "parent_event_id": "agent-run:a",
            "created_at": 1,
        },
        {
            "id": "orphan",
            "conversation_id": "conv-1",
            "parent_event_id": "agent-run:missing",
            "created_at": 1,
        },
    ]

    ids = [record.event_id for record in _records(_snapshot(agent_runs=runs))]

    assert ids == ["agent-run:a", "agent-run:b", "agent-run:orphan"]


def test_unknown_sidecar_kind_is_preserved_instead_of_discarded() -> None:
    messages = [
        {
            "id": "a1",
            "sender": "assistant",
            "content": "answer",
            "timestamp": 1,
            "parent_message_id": None,
            "deleted": False,
        }
    ]
    rows = [
        Sidecar("a1", "conv-1", "turn-1", 1, "assistant"),
        Sidecar(
            "a1",
            "conv-1",
            "turn-1",
            2,
            "future_observable_event",
            payload_json='{"safe": true}',
        ),
    ]

    records = _records(_snapshot(messages=messages, traj_rows=rows))

    assert [record.kind for record in records] == [
        "assistant",
        "future_observable_event",
    ]
    assert records[1].event_id == "trajectory:conv-1:2"
    assert records[1].label == "Future observable event"
    assert records[1].payload == {"safe": True}


def test_legacy_callers_can_omit_all_new_sources() -> None:
    empty = derive_trajectory([], {}, [], [], [])
    assert empty.turns == ()

    message_only = derive_trajectory(
        [
            {
                "id": "legacy",
                "sender": "user",
                "content": "hello",
                "timestamp": 1,
                "parent_message_id": None,
                "deleted": False,
            }
        ],
        {},
        [],
        [],
        [],
    )
    (record,) = _records(message_only)
    assert record.event_id == "message:legacy"
    assert record.source_seq is None
    assert record.seq == 1


def test_message_causality_and_owner_sequence_apply_without_external_events() -> None:
    messages = [
        {
            "id": "child",
            "conversation_id": "conv-1",
            "sender": "assistant",
            "content": "child",
            "timestamp": 1,
            "parent_message_id": "parent",
            "deleted": False,
        },
        {
            "id": "parent",
            "conversation_id": "conv-1",
            "sender": "assistant",
            "content": "parent",
            "timestamp": 2,
            "parent_message_id": None,
            "deleted": False,
        },
    ]
    rows = [
        Sidecar("child", "conv-1", "turn-1", 2, "assistant"),
        Sidecar("parent", "conv-1", "turn-1", 1, "assistant"),
    ]

    ids = [record.event_id for record in _records(_snapshot(messages=messages, traj_rows=rows))]

    assert ids == ["message:parent", "message:child"]


def test_colliding_legacy_compactions_and_external_event_all_survive() -> None:
    messages = [
        {
            "id": "u1",
            "conversation_id": "conv-1",
            "sender": "user",
            "content": "hello",
            "timestamp": 1,
            "parent_message_id": None,
            "deleted": False,
        }
    ]
    compactions = [
        {
            "conversation_id": "conv-1",
            "purpose": "conversation_compaction",
            "status": "succeeded",
            "started_at": 2,
            "finished_at": 3,
        },
        {
            "conversation_id": "conv-1",
            "purpose": "conversation_compaction",
            "status": "failed",
            "started_at": 4,
            "finished_at": 5,
        },
    ]

    snapshot = derive_trajectory(
        messages=messages,
        usage_by_id={},
        traj_rows=[],
        variant_sets=[],
        compaction_records=compactions,
        agent_runs=[{"id": "run-1", "conversation_id": "conv-1"}],
    )
    records = _records(snapshot)
    compaction_ids = [record.event_id for record in records if record.kind == "compaction"]

    assert len(records) == 4
    assert len(compaction_ids) == 2
    assert len(set(compaction_ids)) == 2


def test_repeated_no_sequence_sidecars_keep_distinct_stable_ids() -> None:
    message = {
        "id": "a1",
        "conversation_id": "conv-1",
        "sender": "assistant",
        "content": "answer",
        "timestamp": 1,
        "parent_message_id": None,
        "deleted": False,
    }
    rows = [
        Sidecar("a1", "conv-1", "turn-1", 1, "assistant"),
        Sidecar("a1", "conv-1", "turn-1", None, "future_event"),
        Sidecar("a1", "conv-1", "turn-1", None, "future_event"),
    ]

    records = _records(
        _snapshot(
            messages=[message],
            traj_rows=rows,
            agent_runs=[{"id": "run-1", "conversation_id": "conv-1"}],
        )
    )
    event_ids = [record.event_id for record in records if record.kind == "future_event"]

    assert len(event_ids) == 2
    assert len(set(event_ids)) == 2


def test_delayed_external_event_does_not_fragment_an_existing_turn() -> None:
    messages = [
        {"id": "u1", "sender": "user", "content": "q1", "timestamp": 1, "deleted": False},
        {"id": "a1", "sender": "assistant", "content": "a1", "timestamp": 2, "deleted": False},
        {"id": "u2", "sender": "user", "content": "q2", "timestamp": 3, "deleted": False},
        {"id": "a2", "sender": "assistant", "content": "a2", "timestamp": 4, "deleted": False},
    ]
    rows = [
        Sidecar("u1", "conv-1", "t1", 1, "user"),
        Sidecar("a1", "conv-1", "t1", 2, "assistant"),
        Sidecar("u2", "conv-1", "t2", 3, "user"),
        Sidecar("a2", "conv-1", "t2", 4, "assistant"),
    ]

    snapshot = _snapshot(
        messages=messages,
        traj_rows=rows,
        agent_runs=[
            {
                "id": "late",
                "conversation_id": "conv-1",
                "turn_id": "t1",
                "created_at": 5,
            }
        ],
    )

    assert [turn.turn_id for turn in snapshot.turns] == ["t1", "t2"]
    assert snapshot.turns[0].records[0].event_id == "message:u1"
    assert any(record.event_id == "agent-run:late" for record in snapshot.turns[0].records)


def test_real_agent_owner_shapes_preserve_actor_tool_metadata_and_outcome() -> None:
    run = {
        "id": "run-1",
        "conversation_id": "conv-1",
        "agent_kind": "subagent",
        "agent_definition": "researcher",
    }
    step = {
        "run_id": "run-1",
        "conversation_id": "conv-1",
        "index": 3,
        "kind": "tool_result",
        "summary": "Read completed",
        "tool_name": "fs_read",
        "args": {"path": "README.md"},
        "result": "ok",
        "tool_outcome": "success",
        "created_at": "2026-08-22T12:00:00Z",
    }

    by_id = {
        record.event_id: record
        for record in _records(_snapshot(agent_runs=[run], agent_steps=[step]))
    }

    assert by_id["agent-run:run-1"].actor_id == "researcher"
    record = by_id["agent-step:run-1:3"]
    assert record.payload == {
        "tool_name": "fs_read",
        "tool_outcome": "success",
    }
    assert record.status == "success"
    assert record.field_states == {
        "summary": "observed",
        "args": "omitted_sensitive",
        "result": "omitted_sensitive",
        "tool_outcome": "observed",
    }
    assert record.sensitivity == "tool_content"


def test_agent_step_payload_does_not_copy_sensitive_tool_content() -> None:
    step = {
        "run_id": "run-1",
        "index": 1,
        "kind": "tool_result",
        "tool_name": "remote_api",
        "args": {"api_key": "do-not-project"},
        "result": "hidden provider reasoning",
        "tool_outcome": "success",
    }

    (record,) = _records(_snapshot(agent_steps=[step]))

    assert "do-not-project" not in repr(record.payload)
    assert "hidden provider reasoning" not in repr(record.payload)
