"""Real-seam evidence for durable Trace child-agent lineage."""

from __future__ import annotations

import json
import threading

import pytest

from Tests.Agents.test_fleet_runtime import FLEET_CFG, make_fleet_service
from Tests.Agents.test_agent_service import fence
from tldw_chatbook.Agents.agent_models import (
    RUN_DONE,
    RUN_ERROR,
    SPAWN_TOOL_NAME,
    WAIT_AGENTS_TOOL_NAME,
)
from tldw_chatbook.Agents import run_log as run_log_module
from tldw_chatbook.Agents.run_log_search import load_records
from tldw_chatbook.Chat.trajectory import derive_trajectory
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def _records(snapshot):
    return [record for turn in snapshot.turns for record in turn.records]


@pytest.fixture()
def db(tmp_path):
    return AgentRunsDB(tmp_path / "runs.db", client_id="trace-lineage")


def test_parallel_children_reload_with_precise_spawn_causes_and_safe_tasks(
    db, tmp_path, monkeypatch
):
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    barrier = threading.Barrier(2, timeout=5)
    private_task = (
        "reasoning_content: secret plan; read /Users/alice/private.txt; "
        "api_key=sk-test-lineage-secret"
    )

    def finish(text):
        def reply():
            barrier.wait()
            return text

        return reply

    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": private_task}),
            fence(SPAWN_TOOL_NAME, {"task": "compare public evidence"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "parent complete",
        ],
        {
            private_task: [finish("private child complete")],
            "compare public evidence": [finish("public child complete")],
        },
    )

    parent_id, outcome = service.run_turn(
        conversation_id="trace-lineage",
        messages=[{"role": "user", "content": "delegate in parallel"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE

    rows = db.list_runs("trace-lineage", include_superseded=True)
    parent = next(row for row in rows if row["id"] == parent_id)
    children = [row for row in rows if row["agent_kind"] == "subagent"]
    spawn_events = {
        f"agent-step:{parent_id}:{step['index']}"
        for step in parent["steps"]
        if step["kind"] == "spawn"
    }
    assert len(children) == len(spawn_events) == 2
    assert {child["spawn_event_id"] for child in children} == spawn_events
    assert all(child["parent_run_id"] == parent_id for child in children)
    assert all(child["status"] == RUN_DONE for child in children)
    assert [
        step["kind"]
        for step in parent["steps"]
        if step["kind"].startswith("agent_run_")
    ] == ["agent_run_created", "agent_run_started", "agent_run_completed"]
    for child in children:
        assert [
            step["kind"]
            for step in child["steps"]
            if step["kind"].startswith("agent_run_")
        ] == [
            "agent_run_reserved",
            "agent_run_created",
            "agent_run_started",
            "agent_run_completed",
        ]

    durable_text = json.dumps(rows, sort_keys=True)
    for forbidden in (
        "reasoning_content",
        "/Users/alice/private.txt",
        "sk-test-lineage-secret",
    ):
        assert forbidden not in durable_text
    records = load_records(service.run_log_writer.log_dir)
    logged = "\n".join(record.content for record in records)
    for forbidden in (
        "reasoning_content",
        "/Users/alice/private.txt",
        "sk-test-lineage-secret",
        *(handle.handle_id for handle in coordinator.snapshot()),
    ):
        assert forbidden not in logged
        assert forbidden not in durable_text
    assert "parent complete" in logged
    assert "public child complete" in logged

    db_path = db.db_path
    db.close()
    reopened = AgentRunsDB(db_path, client_id="lineage-reload")
    reloaded = reopened.list_runs("trace-lineage", include_superseded=True)
    assert {
        row["spawn_event_id"]
        for row in reloaded
        if row["agent_kind"] == "subagent"
    } == spawn_events

    agent_steps = [
        {**step, "run_id": row["id"], "conversation_id": "trace-lineage"}
        for row in reloaded
        for step in row["steps"]
    ]
    snapshot = derive_trajectory(
        messages=[],
        usage_by_id={},
        traj_rows=[],
        variant_sets=[],
        compaction_records=[],
        agent_runs=reloaded,
        agent_steps=agent_steps,
    )
    event_ids = [record.event_id for record in _records(snapshot)]
    for child in (row for row in reloaded if row["agent_kind"] == "subagent"):
        assert event_ids.index(child["spawn_event_id"]) < event_ids.index(
            f"agent-run:{child['id']}"
        )
        child_sequences = [
            record.source_seq
            for record in _records(snapshot)
            if record.run_id == child["id"]
            and record.event_id.startswith("agent-step:")
        ]
        assert child_sequences == sorted(child_sequences)
    reopened.close()


@pytest.mark.parametrize(
    ("failed_kind", "successor_kind"),
    [
        ("agent_run_reserved", "agent_run_created"),
        ("agent_run_created", "agent_run_started"),
    ],
)
def test_fleet_create_capture_failure_chains_to_actual_diagnostic(
    db, monkeypatch, failed_kind, successor_kind
):
    original_insert = db.insert_steps_at_indices
    failed = False

    def fail_lifecycle_once(run_id, indexed_steps):
        nonlocal failed
        is_child = db.get_run(run_id)["agent_kind"] == "subagent"
        if (
            not failed
            and is_child
            and any(
                step["kind"] == failed_kind
                for _index, step in indexed_steps
            )
        ):
            failed = True
            raise RuntimeError("simulated fleet lifecycle failure")
        return original_insert(run_id, indexed_steps)

    monkeypatch.setattr(db, "insert_steps_at_indices", fail_lifecycle_once)
    service, _chat, _coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "inspect"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "parent done",
        ],
        {"inspect": ["child done"]},
    )
    _parent_id, outcome = service.run_turn(
        conversation_id=f"fleet-capture-failure-{failed_kind}",
        messages=[{"role": "user", "content": "delegate"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    child_id = next(
        row["id"]
        for row in db.list_runs(f"fleet-capture-failure-{failed_kind}")
        if row["agent_kind"] == "subagent"
    )

    path = db.db_path
    db.close()
    reopened = AgentRunsDB(path, client_id="fleet-failure-reload")
    row = reopened.get_run(child_id)
    assert row["status"] == RUN_DONE
    steps = row["steps"]
    diagnostics = [step for step in steps if step["kind"] == "capture_failed"]
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    lifecycle_kinds = [
        step["kind"] for step in steps if step["kind"].startswith("agent_run_")
    ]
    assert len(lifecycle_kinds) == len(set(lifecycle_kinds))
    successor = next(step for step in steps if step["kind"] == successor_kind)
    diagnostic_id = f"agent-step:{child_id}:{diagnostic['index']}"
    assert diagnostic["field_states"][failed_kind] == "not_observed"
    assert successor["parent_event_id"] == diagnostic_id
    assert not any(step["kind"] == failed_kind for step in steps)
    rows = reopened.list_runs(
        f"fleet-capture-failure-{failed_kind}", include_superseded=True
    )
    event_ids = {
        f"agent-step:{candidate['id']}:{step['index']}"
        for candidate in rows
        for step in candidate["steps"]
    } | {f"agent-run:{candidate['id']}" for candidate in rows}
    for step in steps:
        assert step["parent_event_id"] in event_ids
        assert step["source_event_id"] is None or step["source_event_id"] in event_ids
    reopened.close()


def test_failed_child_and_completed_primary_project_after_reload(db):
    def explode():
        raise RuntimeError("provider exploded with api_key=sk-private-error")

    service, _chat, _coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "failing child"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "parent recovered safely",
        ],
        {"failing child": [explode]},
    )
    parent_id, outcome = service.run_turn(
        conversation_id="trace-failure",
        messages=[{"role": "user", "content": "delegate"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    child = next(
        row
        for row in db.list_runs("trace-failure")
        if row["agent_kind"] == "subagent"
    )
    assert child["status"] == RUN_ERROR

    path = db.db_path
    db.close()
    reopened = AgentRunsDB(path, client_id="failure-reload")
    runs = reopened.list_runs("trace-failure", include_superseded=True)
    steps = [
        {**step, "run_id": row["id"], "conversation_id": "trace-failure"}
        for row in runs
        for step in row["steps"]
    ]
    records = _records(
        derive_trajectory(
            messages=[],
            usage_by_id={},
            traj_rows=[],
            variant_sets=[],
            compaction_records=[],
            agent_runs=runs,
            agent_steps=steps,
        )
    )
    child_kinds = [record.kind for record in records if record.run_id == child["id"]]
    assert "agent_run_created" in child_kinds
    assert "agent_run_started" in child_kinds
    assert "agent_run_failed" in child_kinds
    causal_kinds = [
        record.kind
        for record in records
        if record.run_id == child["id"]
        and record.kind
        in {
            "agent_run_created",
            "agent_run_started",
            "model_request_started",
            "model_error",
            "error",
            "agent_run_failed",
        }
    ]
    assert causal_kinds == [
        "agent_run_created",
        "agent_run_started",
        "model_request_started",
        "model_error",
        "error",
        "agent_run_failed",
    ]
    child_records = {
        record.kind: record
        for record in records
        if record.run_id == child["id"]
    }
    service_error = child_records["error"]
    model_error = child_records["model_error"]
    failed = child_records["agent_run_failed"]
    assert service_error.parent_event_id == model_error.event_id
    assert service_error.source_event_id == model_error.event_id
    assert failed.parent_event_id == service_error.event_id
    assert failed.source_event_id == service_error.event_id
    parent_kinds = [record.kind for record in records if record.run_id == parent_id]
    assert "agent_run_completed" in parent_kinds
    event_ids = {record.event_id for record in records}
    for record in records:
        for link in (record.parent_event_id, record.source_event_id):
            assert link is None or link in event_ids, (
                record.event_id,
                record.kind,
                link,
                sorted(event_ids),
            )
    serialized = json.dumps(runs, sort_keys=True)
    assert "sk-private-error" not in serialized
    reopened.close()


def test_failure_after_control_steps_allocates_a_distinct_causal_error(db):
    def explode():
        raise RuntimeError("second provider call failed")

    service, _chat, _coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "partially progressed child"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "parent recovered safely",
        ],
        {
            "partially progressed child": [
                fence("calculator", {"expression": "1+1"}),
                explode,
            ]
        },
    )
    service.run_turn(
        conversation_id="trace-failure-after-step",
        messages=[{"role": "user", "content": "delegate"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    child = next(
        row
        for row in db.list_runs("trace-failure-after-step")
        if row["agent_kind"] == "subagent"
    )
    path = db.db_path
    db.close()
    reopened = AgentRunsDB(path, client_id="failure-after-step-reload")
    row = reopened.get_run(child["id"])
    steps = row["steps"]
    indices = [step["index"] for step in steps]
    assert len(indices) == len(set(indices))
    error = next(step for step in steps if step["kind"] == "error")
    model_error = next(step for step in steps if step["kind"] == "model_error")
    failed = next(step for step in steps if step["kind"] == "agent_run_failed")
    error_event_id = f"agent-step:{child['id']}:{error['index']}"
    model_error_event_id = f"agent-step:{child['id']}:{model_error['index']}"
    assert error["owner_seq"] == model_error["owner_seq"] + 1
    assert error["parent_event_id"] == model_error_event_id
    assert error["source_event_id"] == model_error_event_id
    assert failed["parent_event_id"] == error_event_id
    assert failed["source_event_id"] == error_event_id
    records = _records(
        derive_trajectory(
            messages=[],
            usage_by_id={},
            traj_rows=[],
            variant_sets=[],
            compaction_records=[],
            agent_runs=[row],
            agent_steps=[
                {
                    **step,
                    "run_id": child["id"],
                    "conversation_id": "trace-failure-after-step",
                }
                for step in steps
            ],
        )
    )
    causal = [
        record.kind
        for record in records
        if record.kind in {"model_error", "error", "agent_run_failed"}
    ]
    assert causal == ["model_error", "error", "agent_run_failed"]
    reopened.close()
