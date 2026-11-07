"""Real FULL transactions, immutable history and required-result capacity."""

import json
import sqlite3
from contextlib import closing

import pytest

pytestmark = pytest.mark.bootstrap_profile


def test_duplicate_start_preserves_identity_and_no_dispatch(tmp_path):
    from Tests.Agents.workflow_support import make_workflow_request
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.workflow_runs import WorkflowRunsStore

    with closing(AgentRunsDB(tmp_path / "runs.db")) as db:
        store = WorkflowRunsStore(db)
        first = store.create(make_workflow_request(tmp_path), launch_id="start-1")
        again = store.create(make_workflow_request(tmp_path), launch_id="start-1")
        assert (first.id, first.chain_id) == (again.id, again.chain_id)
        assert first.status == "awaiting_plan"
        assert db.list_runs(first.request.conversation_id) == []
        assert first.accounting.used == {
            "generation": 0,
            "child_launch": 0,
            "model_call": 0,
            "tokens": 0,
        }
        with pytest.raises(ValueError, match="launch_payload_conflict"):
            store.create(
                make_workflow_request(tmp_path).model_copy(
                    update={"objective": "other"}
                ),
                launch_id="start-1",
            )


@pytest.mark.parametrize("status", ["completed", "stopped", "closed"])
def test_terminal_history_cannot_reopen_after_another_workflow_starts(tmp_path, status):
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    with closing(AgentRunsDB(tmp_path / "runs.db")) as db:
        original = planned(db, tmp_path)
        terminal = db.workflow_runs.set_status(
            original.id,
            status=status,
            expected_control_revision=original.control_revision,
        )
        successor = db.workflow_runs.create(original.request, launch_id="successor")
        with pytest.raises(ValueError, match="workflow_terminal"):
            db.workflow_runs.set_status(
                terminal.id,
                status="awaiting_approval",
                expected_control_revision=terminal.control_revision,
            )
        with pytest.raises(ValueError, match="workflow_terminal"):
            db.workflow_runs.save_plan(
                terminal.id,
                original.plans[-1],
                expected_control_revision=terminal.control_revision,
            )
        assert db.workflow_runs.get(terminal.id) == terminal
        assert db.workflow_runs.get(successor.id) == successor
        removed = db.workflow_runs.remove_payloads(
            terminal.id, expected_control_revision=terminal.control_revision
        )
        assert removed.status == "removed" and removed.accounting == terminal.accounting


def test_create_rolls_back_chain_on_sql_failure(tmp_path):
    from Tests.Agents.workflow_support import make_workflow_request
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    with closing(AgentRunsDB(tmp_path / "runs.db")) as db:
        with db.connection() as conn:
            conn.execute(
                "CREATE TRIGGER refuse_workflow BEFORE INSERT ON workflow_runs BEGIN SELECT RAISE(ABORT, 'test refusal'); END"
            )
        with pytest.raises(sqlite3.IntegrityError, match="test refusal"):
            db.workflow_runs.create(make_workflow_request(tmp_path), launch_id="start")
        with db.connection() as conn:
            assert (
                conn.execute("SELECT count(*) FROM automatic_work_chains").fetchone()[0]
                == 0
            )


def planned(db, tmp_path, launch="start", *, conversation_id="workflow-test-chat"):
    from Tests.Agents.workflow_support import make_workflow_request, plan_data
    from tldw_chatbook.Agents.workflow_models import WorkflowPlan

    store = db.workflow_runs
    request = make_workflow_request(tmp_path).model_copy(
        update={"conversation_id": conversation_id}
    )
    snapshot = store.create(request, launch_id=launch)
    data = plan_data(tmp_path)
    data["request"] = request.model_dump(mode="json")
    return store.save_plan(
        snapshot.id,
        WorkflowPlan.model_validate_json(json.dumps(data)),
        expected_control_revision=1,
    )


def attempt(db, snapshot, run_id=None):
    from tldw_chatbook.Agents.workflow_models import WorkflowTaskAttempt

    return db.workflow_runs.record_attempt(
        WorkflowTaskAttempt(
            id="attempt-" + snapshot.id,
            workflow_id=snapshot.id,
            plan_revision=1,
            phase_id="phase-1",
            task_id="task-1",
            ordinal=1,
            run_id=run_id,
        )
    )


def test_plan_cas_separate_progress_and_reopen(tmp_path):
    from Tests.Agents.workflow_support import plan_data
    from tldw_chatbook.Agents.workflow_models import WorkflowPlan
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    path = tmp_path / "runs.db"
    with closing(AgentRunsDB(path)) as db:
        snap = planned(db, tmp_path)
        assert (snap.control_revision, snap.progress_cursor, len(snap.plans)) == (
            2,
            1,
            1,
        )
        with pytest.raises(ValueError, match="revision_conflict"):
            db.workflow_runs.save_plan(
                snap.id,
                WorkflowPlan.model_validate_json(json.dumps(plan_data(tmp_path))),
                expected_control_revision=1,
            )
        with db.connection() as conn:
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                conn.execute("UPDATE workflow_plan_revisions SET payload_json='{}'")
        db.workflow_runs.set_status(
            snap.id, status="paused", expected_control_revision=2
        )
    with closing(AgentRunsDB(path)) as db:
        saved = db.workflow_runs.get(snap.id)
        assert saved.status == "paused" and saved.control_revision == 3
        assert len(saved.plans) == 1 and saved.progress_cursor == 1
        assert (
            len(
                db.workflow_runs.list_for_conversation(
                    "workflow-test-chat", limit=50, offset=0
                )
            )
            == 1
        )
        with pytest.raises(ValueError):
            db.workflow_runs.list_for_conversation(
                "workflow-test-chat", limit=51, offset=0
            )


def test_private_results_require_reservation_and_exact_owner(tmp_path):
    from tldw_chatbook.Agents.workflow_models import WorkflowTaskResult
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    with closing(AgentRunsDB(tmp_path / "runs.db")) as db:
        snap = planned(db, tmp_path)
        run = db.create_run(
            conversation_id=snap.request.conversation_id,
            agent_kind="subagent",
            work_chain_id=snap.chain_id,
        )
        a = attempt(db, snap, run)
        result = WorkflowTaskResult(
            workflow_id=snap.id,
            attempt_id=a.id,
            run_id=run,
            status="failed",
            summary="real command failed",
            handoff="retained source",
            checks=(),
        )
        with pytest.raises(ValueError, match="unreserved"):
            db.workflow_runs.publish_result(result)
        db.workflow_runs.reserve_result(snap.id, a.id, bytes=131072)
        with pytest.raises(ValueError, match="ownership"):
            db.workflow_runs.publish_result(
                result.model_copy(update={"run_id": "foreign"})
            )
        db.workflow_runs.publish_result(result)
        assert db.workflow_runs.read_result(snap.id, a.id) == result
        assert db.workflow_runs.get(snap.id).control_revision == 2
        with pytest.raises(ValueError, match="active"):
            db.workflow_runs.remove_payloads(snap.id, expected_control_revision=2)
        db.workflow_runs.record_retirement(snap.id, a.id, run_id=run, drained=True)
        db.workflow_runs.set_status(
            snap.id, status="stopped", expected_control_revision=2
        )
        removed = db.workflow_runs.remove_payloads(snap.id, expected_control_revision=3)
        assert (
            removed.status == "removed" and removed.accounting.chain_id == snap.chain_id
        )
        with pytest.raises(ValueError, match="missing"):
            db.workflow_runs.read_result(snap.id, a.id)
        with db.connection() as conn:
            assert (
                conn.execute(
                    "SELECT state FROM workflow_payload_reservations WHERE attempt_id=?",
                    (a.id,),
                ).fetchone()[0]
                == "removed"
            )
            assert (
                conn.execute("SELECT count(*) FROM workflow_task_attempts").fetchone()[
                    0
                ]
                == 1
            )


def test_capacity_and_active_evidence_retention(tmp_path):
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    with closing(AgentRunsDB(tmp_path / "runs.db")) as db:
        snap = planned(db, tmp_path)
        a = attempt(db, snap)
        with pytest.raises(ValueError, match="record"):
            db.workflow_runs.reserve_result(snap.id, a.id, bytes=131073)
        reservation = db.workflow_runs.reserve_result(snap.id, a.id, bytes=131072)
        assert reservation.bytes == 131072
        assert (
            db.workflow_runs.reserve_result(snap.id, a.id, bytes=131072) == reservation
        )
        for status in ("paused", "recovery_required"):
            snap = db.workflow_runs.set_status(
                snap.id, status=status, expected_control_revision=snap.control_revision
            )
            with pytest.raises(ValueError, match="active"):
                db.workflow_runs.remove_payloads(
                    snap.id, expected_control_revision=snap.control_revision
                )


def test_required_results_refuse_undersized_and_multibyte_without_success(tmp_path):
    from tldw_chatbook.Agents.workflow_models import WorkflowTaskResult
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    with closing(AgentRunsDB(tmp_path / "runs.db")) as db:
        snap = planned(db, tmp_path)
        run = db.create_run(
            conversation_id=snap.request.conversation_id,
            agent_kind="subagent",
            work_chain_id=snap.chain_id,
        )
        a = attempt(db, snap, run)
        db.workflow_runs.reserve_result(snap.id, a.id, bytes=128)
        result = WorkflowTaskResult(
            workflow_id=snap.id,
            attempt_id=a.id,
            run_id=run,
            status="completed",
            summary="s",
            handoff="é" * 100,
        )
        with pytest.raises(ValueError, match="capacity"):
            db.workflow_runs.publish_result(result)
        with pytest.raises(ValueError, match="missing"):
            db.workflow_runs.read_result(snap.id, a.id)
        with pytest.raises(ValueError):
            WorkflowTaskResult(
                workflow_id=snap.id,
                attempt_id=a.id,
                run_id=run,
                status="completed",
                summary="s",
                handoff="é" * 65536,
            )


def test_workflow_and_aggregate_capacity_refuse_without_partial_reservations(tmp_path):
    from tldw_chatbook.Agents.workflow_models import WorkflowTaskAttempt
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    with closing(AgentRunsDB(tmp_path / "runs.db")) as db:
        snapshots = [
            planned(db, tmp_path, launch=f"start-{i}", conversation_id=f"chat-{i}")
            for i in range(34)
        ]
        # Real owned rows represent full pre-dispatch slots, without allocating 128 MiB of log text.
        with db.automatic_work.transaction() as conn:
            for i, snap in enumerate(snapshots[:33]):
                for ordinal in range(1, 32):
                    aid = f"reserved-{i}-{ordinal}"
                    conn.execute(
                        "INSERT INTO workflow_task_attempts VALUES (?,?,1,'phase-1','task-1',?,NULL,NULL,'prepared','pending')",
                        (aid, snap.id, ordinal),
                    )
                    conn.execute(
                        "INSERT INTO workflow_payload_reservations VALUES (?,?,131072,'reserved')",
                        (aid, snap.id),
                    )
        for snap, aid in (
            (snapshots[0], "workflow-full"),
            (snapshots[-1], "aggregate-full"),
        ):
            db.workflow_runs.record_attempt(
                WorkflowTaskAttempt(
                    id=aid,
                    workflow_id=snap.id,
                    plan_revision=1,
                    phase_id="phase-1",
                    task_id="task-1",
                    ordinal=40,
                )
            )
            with pytest.raises(ValueError, match="workflow_payload_capacity"):
                db.workflow_runs.reserve_result(snap.id, aid)
        with db.connection() as conn:
            assert (
                conn.execute(
                    "SELECT count(*) FROM workflow_payload_reservations"
                ).fetchone()[0]
                == 1023
            )


def test_decision_cas_duplicate_identity_and_no_budget_refill(tmp_path):
    from tldw_chatbook.Agents.workflow_models import WorkflowDecision
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    with closing(AgentRunsDB(tmp_path / "runs.db")) as db:
        snap = planned(db, tmp_path)
        decision = WorkflowDecision(
            id="decision",
            workflow_id=snap.id,
            plan_revision=1,
            control_revision=2,
            phase_id="phase-1",
            action="approve",
            reason="user_review",
        )
        saved = db.workflow_runs.save_decision(decision)
        again = db.workflow_runs.save_decision(decision)
        assert saved.control_revision == again.control_revision == 3
        assert saved.status == "awaiting_approval"
        assert saved.accounting == snap.accounting
        assert db.list_runs(snap.request.conversation_id) == []
        with pytest.raises(ValueError, match="conflict"):
            db.workflow_runs.save_decision(decision.model_copy(update={"id": "stale"}))


def test_missing_published_payload_fails_snapshot_instead_of_empty_success(tmp_path):
    from Tests.Backup_Recovery.test_workflow_recovery_schema import saved_result
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    with closing(AgentRunsDB(tmp_path / "runs.db")) as db:
        snap, a, _ = saved_result(db, tmp_path)
        with db.connection() as conn:
            conn.execute("DELETE FROM workflow_results WHERE attempt_id=?", (a.id,))
        with pytest.raises(ValueError, match="missing"):
            db.workflow_runs.get(snap.id)


def test_terminal_result_alone_does_not_retire_physical_owner(tmp_path):
    from Tests.Backup_Recovery.test_workflow_recovery_schema import saved_result
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    with closing(AgentRunsDB(tmp_path / "runs.db")) as db:
        snap, a, result = saved_result(db, tmp_path)
        db.set_status(result.run_id, "done", "logical completion only")
        with pytest.raises(ValueError, match="active"):
            db.workflow_runs.set_status(
                snap.id, status="stopped", expected_control_revision=2
            )
        with pytest.raises(ValueError, match="ownership"):
            db.workflow_runs.record_retirement(
                snap.id, a.id, run_id="foreign", drained=True
            )
        db.workflow_runs.record_retirement(
            snap.id, a.id, run_id=result.run_id, drained=True
        )
        stopped = db.workflow_runs.set_status(
            snap.id, status="stopped", expected_control_revision=2
        )
        assert (
            db.workflow_runs.remove_payloads(
                snap.id, expected_control_revision=stopped.control_revision
            ).status
            == "removed"
        )


def test_snapshot_refuses_tampered_structural_plan_rows(tmp_path):
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    with closing(AgentRunsDB(tmp_path / "runs.db")) as db:
        snap = planned(db, tmp_path)
        with db.connection() as conn:
            conn.execute("UPDATE workflow_tasks SET task_id='foreign'")
        with pytest.raises(ValueError, match="ownership"):
            db.workflow_runs.get(snap.id)


@pytest.mark.parametrize(
    "damage",
    [
        "lost_completed_pair",
        "lost_failed_pair",
        "lost_uncertain_pair",
        "active_tombstone",
        "removed_undrained",
        "removed_reserved",
        "removed_missing_tombstone",
        "removed_uncertain_chain",
    ],
)
def test_snapshot_rejects_missing_required_history_from_attempts(tmp_path, damage):
    from Tests.Backup_Recovery.test_workflow_recovery_schema import saved_result
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    with closing(AgentRunsDB(tmp_path / "runs.db")) as db:
        snap, a, _ = saved_result(db, tmp_path)
        with db.connection() as conn:
            conn.execute("DELETE FROM workflow_results WHERE attempt_id=?", (a.id,))
            if damage.startswith("lost_") or damage == "removed_missing_tombstone":
                conn.execute(
                    "DELETE FROM workflow_payload_reservations WHERE attempt_id=?",
                    (a.id,),
                )
            else:
                conn.execute(
                    "UPDATE workflow_payload_reservations SET state=? WHERE attempt_id=?",
                    ("reserved" if damage == "removed_reserved" else "removed", a.id),
                )
            if damage.startswith("lost_"):
                conn.execute(
                    "UPDATE workflow_task_attempts SET status=? WHERE id=?",
                    (damage.split("_")[1], a.id),
                )
            if damage.startswith("removed_"):
                conn.execute(
                    "UPDATE workflow_runs SET status='removed' WHERE id=?", (snap.id,)
                )
                if damage != "removed_undrained":
                    conn.execute(
                        "UPDATE workflow_task_attempts SET retirement='drained' WHERE id=?",
                        (a.id,),
                    )
            if damage == "removed_uncertain_chain":
                conn.execute(
                    "UPDATE automatic_work_chains SET status='review_required' WHERE id=?",
                    (snap.chain_id,),
                )
            assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        with pytest.raises(ValueError, match="workflow_required_result_missing"):
            db.workflow_runs.get(snap.id)


@pytest.mark.parametrize("reserved", [False, True])
def test_prepared_attempt_without_result_remains_readable(tmp_path, reserved):
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    with closing(AgentRunsDB(tmp_path / "runs.db")) as db:
        snap = planned(db, tmp_path)
        a = attempt(db, snap)
        if reserved:
            db.workflow_runs.reserve_result(snap.id, a.id)
        saved = db.workflow_runs.get(snap.id)
        assert saved.status == "awaiting_approval" and saved.results == ()


def test_snapshot_refuses_disagreeing_attempt_and_result_outcomes(tmp_path):
    from Tests.Backup_Recovery.test_workflow_recovery_schema import saved_result
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    with closing(AgentRunsDB(tmp_path / "runs.db")) as db:
        snap, a, _ = saved_result(db, tmp_path)
        with db.connection() as conn:
            conn.execute(
                "UPDATE workflow_task_attempts SET status='completed' WHERE id=?",
                (a.id,),
            )
        with pytest.raises(ValueError, match="invalid_workflow_result_ownership"):
            db.workflow_runs.get(snap.id)
