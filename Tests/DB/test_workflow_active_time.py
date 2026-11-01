"""Workflow union time excludes only positively drained durable waits."""

import pytest

from Tests.DB.test_workflow_attempts import prepare, workflow_fixture  # noqa: F401

pytestmark = pytest.mark.bootstrap_profile


def test_parallel_calls_share_one_interval(workflow):
    db, snap, clock = workflow
    a = prepare(db, snap)
    db.automatic_work.accept_workflow_phase(a.id, owner_id="owner")
    clock[:] = [1010.0, 10.0]
    assert db.automatic_work.workflow_active_seconds(snap.id) == 10.0
    db.automatic_work.check_active(snap.chain_id, owner_id="owner")
    clock[:] = [1020.0, 20.0]
    assert db.automatic_work.workflow_active_seconds(snap.id) == 20.0
    assert db.automatic_work.snapshot(snap.chain_id).deadline_at is None


def test_clock_reversal_is_review_required(workflow):
    db, snap, clock = workflow
    a = prepare(db, snap)
    db.automatic_work.accept_workflow_phase(a.id, owner_id="owner")
    clock[:] = [999.0, 3.0]
    with pytest.raises(RuntimeError, match="clock_reversed"):
        db.automatic_work.check_active(snap.chain_id, owner_id="owner")
    assert db.automatic_work.snapshot(snap.chain_id).status == "review_required"


def test_foreign_open_interval_cannot_become_zero(workflow, monkeypatch):
    from tldw_chatbook.DB import automatic_work

    db, snap, clock = workflow
    a = prepare(db, snap)
    db.automatic_work.accept_workflow_phase(a.id, owner_id="owner")
    monkeypatch.setattr(automatic_work, "_CLOCK_OWNER_ID", "another-process")
    clock[:] = [1005.0, 1.0]
    with pytest.raises(RuntimeError, match="clock_unknown"):
        db.automatic_work.check_active(snap.chain_id, owner_id="owner")
    assert db.automatic_work.snapshot(snap.chain_id).status == "review_required"


def test_overnight_review_does_not_spend_or_refill_allowance(workflow, tmp_path):
    import json

    from Tests.Agents.workflow_support import make_workflow_request, plan_data
    from tldw_chatbook.Agents.workflow_models import WorkflowPlan

    db, _, clock = workflow
    snap = db.workflow_runs.create(
        make_workflow_request(tmp_path), launch_id="planning"
    )
    a = db.automatic_work.prepare_workflow_planning(
        snap.id, owner_id="owner", session_id="session", command_id="plan-command"
    )
    db.automatic_work.accept_workflow_phase(a.id, owner_id="owner")
    clock[:] = [1010.0, 10.0]
    assert not db.automatic_work.settle_workflow_phase(
        a.id, owner_id="owner", drained=False
    )
    assert db.automatic_work.workflow_active_seconds(snap.id) == 10
    plan = WorkflowPlan.model_validate_json(json.dumps(plan_data(tmp_path)))
    assert db.automatic_work.settle_workflow_phase(
        a.id, owner_id="owner", drained=True, plan=plan
    )
    clock[:] = [100000.0, 99000.0]
    assert db.automatic_work.workflow_active_seconds(snap.id) == 10
    b = prepare(db, snap)
    db.automatic_work.accept_workflow_phase(b.id, owner_id="owner")
    clock[:] = [100005.0, 99005.0]
    assert db.automatic_work.workflow_active_seconds(snap.id) == 15
    assert db.automatic_work.snapshot(snap.chain_id).used["generation"] == 2


def test_live_reduction_exhaustion_survives_later_raise(workflow):
    from dataclasses import replace

    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkLimits

    db, snap, clock = workflow
    a = prepare(db, snap)
    db.automatic_work.accept_workflow_phase(a.id, owner_id="owner")
    clock[:] = [1001.0, 20.0]
    with pytest.raises(RuntimeError, match="wall_budget"):
        db.automatic_work.check_active(
            snap.chain_id,
            owner_id="owner",
            limits=replace(AutomaticWorkLimits(), wall_seconds=15),
        )
    with pytest.raises(RuntimeError, match="wall_budget"):
        db.automatic_work.check_active(
            snap.chain_id,
            owner_id="owner",
            limits=replace(AutomaticWorkLimits(), wall_seconds=9999),
        )
    assert db.automatic_work.workflow_active_seconds(snap.id) == 20


def test_recovery_retains_open_interval_charge(workflow):
    db, snap, clock = workflow
    a = prepare(db, snap)
    db.automatic_work.accept_workflow_phase(a.id, owner_id="owner")
    clock[:] = [1010.0, 10.0]
    db.automatic_work.recover(current_owner_id="replacement")
    with db.connection() as conn:
        interval = conn.execute(
            "SELECT * FROM workflow_active_intervals WHERE workflow_id=?", (snap.id,)
        ).fetchone()
        assert interval["ended_at"] is None
        assert interval["charged_seconds"] >= 900
    assert db.workflow_runs.get(snap.id).status == "recovery_required"


def test_parallel_task_durations_count_once_and_unknown_drain_stays_open(
    workflow, tmp_path
):
    import copy
    import json

    from Tests.Agents.workflow_support import plan_data
    from tldw_chatbook.Agents.workflow_models import WorkflowPlan, WorkflowTaskResult

    db, snap, clock = workflow
    data = plan_data(tmp_path)
    second = copy.deepcopy(data["phases"][0]["tasks"][0])
    second["id"] = "task-2"
    data["phases"][0]["tasks"].append(second)
    db.workflow_runs.save_plan(
        snap.id,
        WorkflowPlan.model_validate_json(json.dumps(data)),
        expected_control_revision=2,
    )
    a = prepare(db, snap, revision_id=2)
    db.automatic_work.accept_workflow_phase(a.id, owner_id="owner")
    claims = []
    for task_id in ("task-1", "task-2"):
        claim = db.automatic_work.claim_workflow_task(
            a.id, task_id=task_id, owner_id="owner"
        )
        assert db.automatic_work.commit(claim.reservation_id, owner_id="owner")
        run = db.create_run(
            conversation_id=snap.request.conversation_id,
            agent_kind="subagent",
            work_chain_id=snap.chain_id,
            run_id=claim.launch_run_id,
        )
        db.automatic_work.bind_workflow_task_run(
            a.id, task_id=task_id, run_id=run, owner_id="owner"
        )
        claims.append((claim, run))
    clock[:] = [1010.0, 10.0]
    for claim, run in claims:
        db.workflow_runs.publish_result(
            WorkflowTaskResult(
                workflow_id=snap.id,
                attempt_id=claim.id,
                run_id=run,
                status="completed",
                summary="done",
                handoff="saved",
            )
        )
    db.workflow_runs.record_retirement(
        snap.id, claims[0][0].id, run_id=claims[0][1], drained=True
    )
    with pytest.raises(RuntimeError, match="workflow_not_settled"):
        db.automatic_work.settle_workflow_phase(a.id, owner_id="owner", drained=True)
    clock[:] = [1015.0, 15.0]
    db.workflow_runs.record_retirement(
        snap.id, claims[1][0].id, run_id=claims[1][1], drained=True
    )
    assert db.automatic_work.settle_workflow_phase(a.id, owner_id="owner", drained=True)
    assert db.automatic_work.workflow_active_seconds(snap.id) == 15
