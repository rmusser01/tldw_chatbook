"""Real-seam evidence for durable Trace child-agent lineage."""

from __future__ import annotations

import json
import threading

import pytest

from Tests.Agents.test_fleet_runtime import FLEET_CFG, make_fleet_service
from Tests.Agents.test_agent_service import fence
from tldw_chatbook.Agents.agent_models import (
    RUN_DONE,
    SPAWN_TOOL_NAME,
    WAIT_AGENTS_TOOL_NAME,
)
from tldw_chatbook.Agents import run_log as run_log_module
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

    service, _chat, _coordinator = make_fleet_service(
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

    durable_text = json.dumps(rows, sort_keys=True)
    for forbidden in (
        "reasoning_content",
        "/Users/alice/private.txt",
        "sk-test-lineage-secret",
    ):
        assert forbidden not in durable_text
        assert forbidden not in "".join(
            path.read_text(encoding="utf-8") for path in tmp_path.rglob("*.jsonl")
        )

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


def test_projected_run_statuses_cover_terminal_and_continuation_lineage():
    runs = [
        {
            "id": "old",
            "conversation_id": "c",
            "agent_kind": "subagent",
            "status": "error",
            "created_at": 1,
            "updated_at": 2,
        },
        {
            "id": "resumed",
            "conversation_id": "c",
            "agent_kind": "subagent",
            "parent_run_id": "primary",
            "spawn_event_id": "agent-step:primary:8",
            "resumed_from_run_id": "old",
            "status": "cancelled",
            "created_at": 3,
            "updated_at": 4,
        },
        {
            "id": "superseded",
            "conversation_id": "c",
            "agent_kind": "primary",
            "status": "superseded",
            "created_at": 5,
            "updated_at": 6,
        },
    ]
    steps = [
        {
            "run_id": "primary",
            "conversation_id": "c",
            "index": 8,
            "owner_seq": 8,
            "kind": "spawn",
            "created_at": 2.5,
        }
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
    by_id = {record.event_id: record for record in records}
    assert by_id["agent-run:old"].status == "error"
    assert by_id["agent-run:resumed"].status == "cancelled"
    assert by_id["agent-run:resumed"].parent_event_id == "agent-step:primary:8"
    assert by_id["agent-run:resumed"].source_event_id == "agent-run:old"
    assert by_id["agent-run:superseded"].status == "superseded"
