"""Real SQLite capacity and per-chat lifecycle admission boundaries."""

import json
from contextlib import closing
from typing import get_args

import pytest

from Tests.Agents.workflow_support import make_workflow_request, plan_data
from Tests.DB.test_workflow_attempts import (
    workflow_fixture as workflow_fixture,  # noqa: PLC0414
)
from tldw_chatbook.Agents.workflow_models import (
    RECORD_BYTES,
    WorkflowDecision,
    WorkflowPlan,
    WorkflowStatus,
)

pytestmark = pytest.mark.bootstrap_profile


def _rows(db):
    with db.connection() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return {
            table[0]: [
                tuple(row) for row in conn.execute(f'SELECT * FROM "{table[0]}"')
            ]
            for table in tables
        }


def _retained(conn, workflow_id=None):
    total = 0
    for table, column, key in (
        ("workflow_runs", "length(CAST(request_json AS BLOB))", "id"),
        (
            "workflow_plan_revisions",
            "length(CAST(payload_json AS BLOB))",
            "workflow_id",
        ),
        ("workflow_decisions", "length(CAST(payload_json AS BLOB))", "workflow_id"),
        (
            "workflow_payload_reservations",
            "CASE WHEN state!='removed' THEN bytes ELSE 0 END",
            "workflow_id",
        ),
        ("workflow_phase_attempts", "result_bytes", "workflow_id"),
    ):
        total += conn.execute(
            f"SELECT COALESCE(SUM({column}),0) FROM {table}"
            + (f" WHERE {key}=?" if workflow_id else ""),
            (workflow_id,) if workflow_id else (),
        ).fetchone()[0]
    return total


@pytest.mark.parametrize("kind", ["initial", "remaining"])
@pytest.mark.parametrize("ceiling", ["WORKFLOW_BYTES", "TOTAL_BYTES"])
@pytest.mark.parametrize("shortfall", [0, 1])
def test_decision_and_all_result_reservations_share_capacity(
    workflow, tmp_path, monkeypatch, kind, ceiling, shortfall
):
    from tldw_chatbook.DB import workflow_runs

    db, snapshot, _ = workflow
    data = plan_data(tmp_path)
    data["request"] = snapshot.request.model_dump(mode="json")
    task = data["phases"][0]["tasks"][0]
    data["phases"][0]["tasks"].append(
        dict(
            task,
            id="task-2",
            outputs=["other.py"],
            footprint=[{"path": "other.py", "access": "write"}],
        )
    )
    snapshot = db.workflow_runs.save_plan(
        snapshot.id,
        WorkflowPlan.model_validate_json(json.dumps(data)),
        expected_control_revision=snapshot.control_revision,
    )
    db.workflow_runs.create(
        snapshot.request.model_copy(update={"conversation_id": "other-chat"}),
        launch_id="other-chat-launch",
    )
    owner = "owner"
    if kind == "remaining":
        db.automatic_work.prepare_workflow_phase(
            snapshot.id,
            revision_id=2,
            phase_id="phase-1",
            owner_id=owner,
            session_id="session",
            command_id="original-unused",
        )
        owner = "replacement"
        db.automatic_work.recover(current_owner_id=owner)
        snapshot = db.workflow_runs.get(snapshot.id)
        with db.connection() as conn:
            assert db.workflow_runs._remaining_partition(
                conn, snapshot.id, 2, "phase-1"
            ) == ({}, ("task-1", "task-2"))
    decision = WorkflowDecision(
        id="joint-capacity",
        workflow_id=snapshot.id,
        plan_revision=2,
        phase_id="phase-1",
        control_revision=snapshot.control_revision,
        action="approve",
        reason="ready" if kind == "initial" else "remaining_ready",
    )
    with db.connection() as conn:
        retained = _retained(conn, snapshot.id if ceiling == "WORKFLOW_BYTES" else None)
    limit = (
        retained
        + 2 * RECORD_BYTES
        + len(decision.canonical_json().encode())
        - shortfall
    )
    monkeypatch.setattr(workflow_runs, ceiling, limit)
    args = dict(
        revision_id=2,
        phase_id="phase-1",
        purpose="execution",
        owner_id=owner,
        session_id="session",
        command_id="joint-capacity-command",
        **{"_" + kind: decision},
    )
    before = _rows(db)
    if shortfall:
        with pytest.raises(ValueError, match="^workflow_payload_capacity$"):
            db.automatic_work._prepare_workflow(snapshot.id, **args)
        assert _rows(db) == before
        return
    attempt = db.automatic_work._prepare_workflow(snapshot.id, **args)
    with db.connection() as conn:
        db.workflow_runs._capacity(conn, snapshot.id, 0)
        assert (
            _retained(conn, snapshot.id if ceiling == "WORKFLOW_BYTES" else None)
            == limit
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM workflow_payload_reservations WHERE state='reserved'"
            ).fetchone()[0]
            == 2
        )
    assert db.automatic_work.accept_workflow_phase(attempt.id, owner_id=owner)
    accepted = db.automatic_work.read_workflow_attempt(attempt.id, owner_id=owner)
    before = _rows(db)
    monkeypatch.setattr(workflow_runs, ceiling, 0)
    assert db.automatic_work._prepare_workflow(snapshot.id, **args) == accepted
    assert not db.automatic_work.accept_workflow_phase(attempt.id, owner_id=owner)
    assert _rows(db) == before
    with pytest.raises(ValueError, match="workflow_command_conflict"):
        db.automatic_work._prepare_workflow(
            snapshot.id,
            **dict(
                args, **{"_" + kind: decision.model_copy(update={"reason": "changed"})}
            ),
        )
    assert _rows(db) == before


@pytest.mark.parametrize("status", get_args(WorkflowStatus))
def test_only_terminal_workflows_allow_another_launch_in_the_chat(tmp_path, status):
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    request = make_workflow_request(tmp_path)
    with closing(AgentRunsDB(tmp_path / "lifecycle.db")) as db:
        first = db.workflow_runs.create(request, launch_id="first")
        if status == "removed":
            first = db.workflow_runs.set_status(
                first.id,
                status="closed",
                expected_control_revision=first.control_revision,
            )
            first = db.workflow_runs.remove_payloads(
                first.id, expected_control_revision=first.control_revision
            )
        elif status != first.status:
            first = db.workflow_runs.set_status(
                first.id,
                status=status,
                expected_control_revision=first.control_revision,
            )
        before = _rows(db)
        assert db.workflow_runs.create(request, launch_id="first") == first
        assert _rows(db) == before
        with pytest.raises(ValueError, match="launch_payload_conflict"):
            db.workflow_runs.create(
                request.model_copy(update={"objective": "different"}), launch_id="first"
            )
        if status in {"stopped", "completed", "closed", "removed"}:
            second = db.workflow_runs.create(request, launch_id="second")
            assert second.id != first.id and second.chain_id != first.chain_id
            assert db.workflow_runs.get(first.id) == first
            assert db.workflow_runs.create(request, launch_id="first") == first
        else:
            with pytest.raises(
                RuntimeError, match="^workflow_already_exists$"
            ) as refused:
                db.workflow_runs.create(request, launch_id="second")
            assert refused.value.workflow_id == first.id
            assert _rows(db) == before
        assert db.list_runs(request.conversation_id) == []


def test_concurrent_new_launches_serialize_the_same_chat_predicate(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    path = tmp_path / "concurrent.db"
    with closing(AgentRunsDB(path)):
        pass
    entered = Barrier(2)
    request = make_workflow_request(tmp_path)

    def create(launch):
        with closing(AgentRunsDB(path)) as db:
            entered.wait(5)
            try:
                return "created", db.workflow_runs.create(request, launch_id=launch).id
            except RuntimeError as error:
                assert str(error) == "workflow_already_exists"
                return "refused", error.workflow_id

    with ThreadPoolExecutor(max_workers=2) as workers:
        outcomes = list(workers.map(create, ("one", "two")))
    assert sorted(kind for kind, _ in outcomes) == ["created", "refused"]
    assert outcomes[0][1] == outcomes[1][1]
    with closing(AgentRunsDB(path)) as db:
        rows = _rows(db)
        assert len(rows["workflow_runs"]) == len(rows["automatic_work_chains"]) == 1
        assert not rows["automatic_wake_attempts"]
