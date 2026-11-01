"""Native workflow admission uses durable exact-owner claims, never survivor wakes."""

from contextlib import closing

import pytest

from Tests.DB.test_workflow_runs import planned

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture(name="workflow")
def workflow_fixture(tmp_path, monkeypatch):
    from tldw_chatbook.Agents import run_log
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.automatic_work import AutomaticWorkLedger

    monkeypatch.setattr(
        run_log,
        "_setting",
        lambda key, default: True if key == "workflow_runs_enabled" else default,
    )
    with closing(AgentRunsDB(tmp_path / "attempts.db")) as db:
        clock = [1000.0, 0.0]
        db.automatic_work = AutomaticWorkLedger(
            db, wall_clock=lambda: clock[0], monotonic_clock=lambda: clock[1]
        )
        db.automatic_work.recover(current_owner_id="owner")
        yield db, planned(db, tmp_path), clock


def prepare(db, snap, **changes):
    values = {
        "revision_id": 1,
        "phase_id": "phase-1",
        "owner_id": "owner",
        "session_id": "session",
        "command_id": "approve-1",
    }
    values.update(changes)
    return db.automatic_work.prepare_workflow_phase(snap.id, **values)


def test_repeated_phase_and_task_claim_preserve_original_identity(workflow):
    db, snap, _ = workflow
    first = prepare(db, snap)
    assert prepare(db, snap) == first
    assert first.state == "prepared"
    assert first.chain_id == snap.chain_id
    assert db.automatic_work.accept_workflow_phase(first.id, owner_id="owner")
    assert not db.automatic_work.accept_workflow_phase(first.id, owner_id="owner")
    one = db.automatic_work.claim_workflow_task(
        first.id, task_id="task-1", owner_id="owner"
    )
    assert (
        db.automatic_work.claim_workflow_task(
            first.id, task_id="task-1", owner_id="owner"
        )
        == one
    )
    assert db.automatic_work.snapshot(snap.chain_id).reserved["child_launch"] == 1
    assert db.automatic_work.commit(one.reservation_id, owner_id="owner")
    assert not db.automatic_work.commit(one.reservation_id, owner_id="owner")
    counts = db.automatic_work.snapshot(snap.chain_id).used
    assert counts["generation"] == counts["child_launch"] == 1
    with db.connection() as conn:
        assert (
            conn.execute("SELECT count(*) FROM automatic_wake_claims").fetchone()[0]
            == 0
        )


def test_conflicting_command_and_wrong_owner_refuse(workflow):
    db, snap, _ = workflow
    first = prepare(db, snap)
    with pytest.raises(ValueError, match="conflict"):
        prepare(db, snap, session_id="other")
    with pytest.raises((ValueError, RuntimeError), match="owner"):
        db.automatic_work.accept_workflow_phase(first.id, owner_id="other")
    with pytest.raises(RuntimeError):
        db.automatic_work.claim_workflow_task(
            first.id, task_id="task-1", owner_id="owner"
        )


def test_control_revocation_wins_before_dispatch(workflow):
    db, snap, _ = workflow
    first = prepare(db, snap)
    assert db.automatic_work.accept_workflow_phase(first.id, owner_id="owner")
    current = db.workflow_runs.get(snap.id)
    db.workflow_runs.set_status(
        snap.id, status="paused", expected_control_revision=current.control_revision
    )
    with pytest.raises(RuntimeError):
        db.automatic_work.claim_workflow_task(
            first.id, task_id="task-1", owner_id="owner"
        )


def test_foreign_recovery_retains_prepared_charge_and_never_replays(workflow):
    db, snap, _ = workflow
    first = prepare(db, snap)
    db.automatic_work.recover(current_owner_id="replacement")
    with pytest.raises(RuntimeError):
        db.automatic_work.accept_workflow_phase(first.id, owner_id="owner")
    assert db.automatic_work.snapshot(snap.chain_id).reserved["generation"] == 1


def test_planning_is_a_real_attempt_without_a_fake_plan(workflow, tmp_path):
    from Tests.Agents.workflow_support import make_workflow_request

    db, _, _ = workflow
    snap = db.workflow_runs.create(
        make_workflow_request(tmp_path), launch_id="planning"
    )
    args = {"owner_id": "owner", "session_id": "session", "command_id": "plan-command"}
    first = db.automatic_work.prepare_workflow_planning(snap.id, **args)
    assert (
        first.purpose == "planning"
        and first.phase_id is None
        and first.revision_id is None
    )
    assert db.automatic_work.prepare_workflow_planning(snap.id, **args) == first
    assert db.workflow_runs.get(snap.id).plans == ()
    assert db.automatic_work.accept_workflow_phase(first.id, owner_id="owner")
    with pytest.raises(RuntimeError, match="planning_has_no_tasks"):
        db.automatic_work.claim_workflow_task(
            first.id, task_id="task-1", owner_id="owner"
        )


def test_acceptance_commits_under_full_sync_and_rolls_back_failure(workflow):
    import sqlite3

    db, snap, _ = workflow
    first = prepare(db, snap)
    seen = []
    with db.connection() as conn:
        conn.create_function(
            "observe_full",
            0,
            lambda: seen.append(conn.execute("PRAGMA synchronous").fetchone()[0]) or 1,
        )
        conn.execute(
            "CREATE TEMP TRIGGER reject_accept BEFORE UPDATE OF state ON automatic_wake_attempts WHEN NEW.state='accepted' BEGIN SELECT observe_full(); SELECT RAISE(ABORT,'reject acceptance'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="reject acceptance"):
        db.automatic_work.accept_workflow_phase(first.id, owner_id="owner")
    assert seen == [2]
    assert (
        db.automatic_work.read_workflow_attempt(first.id, owner_id="owner").state
        == "prepared"
    )
    assert db.automatic_work.workflow_active_seconds(snap.id) == 0
    assert db.automatic_work.snapshot(snap.chain_id).used["generation"] == 0


def test_claimed_native_run_identity_cannot_be_replaced(workflow):
    db, snap, _ = workflow
    first = prepare(db, snap)
    db.automatic_work.accept_workflow_phase(first.id, owner_id="owner")
    claimed = db.automatic_work.claim_workflow_task(
        first.id, task_id="task-1", owner_id="owner"
    )
    run = db.create_run(
        conversation_id=snap.request.conversation_id,
        agent_kind="subagent",
        work_chain_id=snap.chain_id,
        run_id=claimed.launch_run_id,
    )
    db.automatic_work.bind_workflow_task_run(
        first.id, task_id="task-1", run_id=run, owner_id="owner"
    )
    repeated = db.automatic_work.claim_workflow_task(
        first.id, task_id="task-1", owner_id="owner"
    )
    assert repeated.id == claimed.id and repeated.run_id == run
    other = db.create_run(
        conversation_id=snap.request.conversation_id,
        agent_kind="subagent",
        work_chain_id=snap.chain_id,
    )
    with pytest.raises(ValueError, match="conflict"):
        db.automatic_work.bind_workflow_task_run(
            first.id, task_id="task-1", run_id=other, owner_id="owner"
        )


def test_accepted_database_id_cannot_reconstruct_runtime_context(workflow):
    from tldw_chatbook.Agents.automatic_work_runtime import AutomaticWorkContext

    db, snap, _ = workflow
    first = prepare(db, snap)
    db.automatic_work.accept_workflow_phase(first.id, owner_id="owner")
    context = AutomaticWorkContext(
        db.automatic_work,
        snap.chain_id,
        "owner",
        first.id,
        attempt_kind="workflow_phase",
    )
    with pytest.raises(RuntimeError, match="acceptance_required"):
        context.mark_accepted()


def test_workflow_policy_follows_immutable_chain_after_removal(workflow):
    db, snap, _ = workflow
    assert db.automatic_work.policy_kind(snap.chain_id) == "workflow_phase"
    db.workflow_runs.set_status(snap.id, status="stopped", expected_control_revision=2)
    db.workflow_runs.remove_payloads(snap.id, expected_control_revision=3)
    assert db.automatic_work.policy_kind(snap.chain_id) == "workflow_phase"


def test_planning_settlement_failure_retains_acceptance_capacity_and_time(
    workflow, tmp_path
):
    import json
    import sqlite3

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
    with db.connection() as conn:
        conn.execute(
            "CREATE TEMP TRIGGER fail_plan BEFORE INSERT ON workflow_plan_revisions BEGIN SELECT RAISE(ABORT,'uncertain commit'); END"
        )
    clock[:] = [1015.0, 15.0]
    with pytest.raises(sqlite3.IntegrityError, match="uncertain commit"):
        db.automatic_work.settle_workflow_phase(
            a.id,
            owner_id="owner",
            drained=True,
            plan=WorkflowPlan.model_validate_json(json.dumps(plan_data(tmp_path))),
        )
    assert (
        db.automatic_work.read_workflow_attempt(a.id, owner_id="owner").state
        == "accepted"
    )
    assert db.automatic_work.workflow_active_seconds(snap.id) == 15
    assert db.workflow_runs.get(snap.id).plans == ()
    with db.connection() as conn:
        assert (
            conn.execute(
                "SELECT result_bytes FROM workflow_phase_attempts WHERE native_attempt_id=?",
                (a.id,),
            ).fetchone()[0]
            == 131072
        )
        assert (
            conn.execute(
                "SELECT ended_at FROM workflow_active_intervals WHERE workflow_id=?",
                (snap.id,),
            ).fetchone()[0]
            is None
        )


def test_native_commit_rechecks_control_after_task_claim(workflow):
    db, snap, _ = workflow
    a = prepare(db, snap)
    db.automatic_work.accept_workflow_phase(a.id, owner_id="owner")
    claim = db.automatic_work.claim_workflow_task(
        a.id, task_id="task-1", owner_id="owner"
    )
    db.workflow_runs.set_status(snap.id, status="paused", expected_control_revision=2)
    with pytest.raises(RuntimeError, match="workflow_authority_changed"):
        db.automatic_work.commit(claim.reservation_id, owner_id="owner")
    assert db.automatic_work.snapshot(snap.chain_id).used["child_launch"] == 0


def test_private_custody_issues_once_after_exact_live_owner_acceptance(workflow):
    from threading import Event
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_workflow_runs import _accept_workflow_phase

    db, snap, _ = workflow
    a = prepare(db, snap)
    session = SimpleNamespace(
        id="session", persisted_conversation_id=snap.request.conversation_id
    )
    controller = SimpleNamespace(
        _disposed=False,
        _shutdown_requested=Event(),
        _agent_bridge=SimpleNamespace(runs_db=db),
        _fleet_wake=SimpleNamespace(runtime_owner_id="owner"),
        store=SimpleNamespace(sessions=lambda: [session]),
    )
    with pytest.raises(RuntimeError, match="workflow_session_changed"):
        _accept_workflow_phase(controller, a.id, session_id="foreign")
    assert (
        db.automatic_work.read_workflow_attempt(a.id, owner_id="owner").state
        == "prepared"
    )
    authorization = _accept_workflow_phase(controller, a.id, session_id="session")
    assert authorization.context.check().chain_id == snap.chain_id
    assert _accept_workflow_phase(controller, a.id, session_id="session") is None
    controller.store.sessions = lambda: [
        SimpleNamespace(
            id="session", persisted_conversation_id=snap.request.conversation_id
        )
    ]
    with pytest.raises(RuntimeError, match="workflow_session_changed"):
        authorization.context.check()
    controller.store.sessions = lambda: [session]
    db.workflow_runs.set_status(snap.id, status="paused", expected_control_revision=2)
    with pytest.raises(RuntimeError, match="workflow_authority_changed"):
        authorization.context.check()


@pytest.mark.parametrize(
    "field",
    [
        "generations",
        "child_launches",
        "model_calls",
        "budget_tokens",
        "output_tokens",
        "wall_seconds",
    ],
)
def test_zero_workflow_ceiling_refuses_before_acceptance(workflow, monkeypatch, field):
    from tldw_chatbook.Agents import run_log

    db, snap, _ = workflow
    monkeypatch.setattr(
        run_log,
        "_setting",
        lambda key, default: (
            True
            if key == "workflow_runs_enabled"
            else 0
            if key == "max_workflow_" + field
            else default
        ),
    )
    with pytest.raises(RuntimeError, match="budget"):
        prepare(db, snap)
    assert db.automatic_work.snapshot(snap.chain_id).used["generation"] == 0


@pytest.mark.parametrize(
    "damage", ["native_kind", "task_chain", "run_identity", "clock"]
)
def test_recovery_rejects_corrupt_workflow_admission_links(workflow, damage):
    from tldw_chatbook.DB.recovery_workflow_schema import validate_workflow_rows

    db, snap, _ = workflow
    a = prepare(db, snap)
    db.automatic_work.accept_workflow_phase(a.id, owner_id="owner")
    claim = db.automatic_work.claim_workflow_task(
        a.id, task_id="task-1", owner_id="owner"
    )
    with db.connection() as conn:
        assert validate_workflow_rows(conn) == ()
        if damage == "native_kind":
            conn.execute(
                "UPDATE automatic_wake_attempts SET attempt_kind='fleet_wake' WHERE id=?",
                (a.id,),
            )
        elif damage == "task_chain":
            conn.execute(
                "UPDATE automatic_work_reservations SET kind='tokens' WHERE id=?",
                (claim.reservation_id,),
            )
        elif damage == "run_identity":
            run = db.create_run(
                conversation_id=snap.request.conversation_id,
                agent_kind="subagent",
                work_chain_id=snap.chain_id,
            )
            conn.execute(
                "UPDATE workflow_task_attempts SET run_id=? WHERE id=?", (run, claim.id)
            )
        else:
            conn.execute(
                "UPDATE workflow_active_intervals SET observed_monotonic=?",
                (float("inf"),),
            )
        assert validate_workflow_rows(conn)
