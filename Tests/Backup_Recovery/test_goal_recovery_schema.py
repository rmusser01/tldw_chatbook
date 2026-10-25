"""Current goal archives retain exact schema and inert durable authority."""

import sqlite3
from contextlib import closing
from threading import Event

import pytest

from Tests.agent_runs_owner import owned_agent_runs_db
from Tests.Agents.test_goal_models import request
from Tests.Backup_Recovery.test_agent_runs_recovery_schema import owner
from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate


def test_goal_archive_is_exact_and_restore_never_accepts_work(tmp_path):
    path = tmp_path / "goals.db"
    db = owned_agent_runs_db(path)
    goal = db.goal_runs.create(request(), launch_id="launch")
    goal = db.goal_runs.set_provisioning(goal, status="ready")
    attempt = db.automatic_work.prepare_goal_iteration(goal.id, owner_id="before")
    db.close()
    before = path.read_bytes()
    assert validate_candidate(owner(), path, Event(), migrate=True) == ()
    assert path.read_bytes() == before
    db = owned_agent_runs_db(path)
    try:
        db.automatic_work.recover(current_owner_id="after")
        with db.connection() as conn:
            assert (
                conn.execute(
                    "SELECT state FROM automatic_wake_attempts WHERE id=?",
                    (attempt.id,),
                ).fetchone()[0]
                == "review_required"
            )
        assert db.automatic_work.snapshot(goal.chain_id).uncertain
        assert db.list_runs(goal.conversation_id) == []
    finally:
        db.close()


@pytest.mark.parametrize("case", ["missing_goal_table", "old_stamp", "extra_column"])
def test_hybrid_goal_catalog_refuses_without_rewrite(tmp_path, case):
    path = tmp_path / "hybrid.db"
    owned_agent_runs_db(path).close()
    with closing(sqlite3.connect(path)) as conn:
        if case == "missing_goal_table":
            conn.execute("DROP TABLE goal_waits")
        elif case == "old_stamp":
            conn.execute("DELETE FROM schema_version WHERE version=24")
        else:
            conn.execute("ALTER TABLE goal_runs ADD COLUMN unqualified TEXT")
        conn.commit()
    before = path.read_bytes()
    assert validate_candidate(owner(), path, Event(), migrate=True)
    assert path.read_bytes() == before


@pytest.mark.parametrize("corruption", ["attempt_kind", "status", "reservation_owner"])
def test_goal_archive_rejects_mismatched_domain_without_rewrite(tmp_path, corruption):
    path = tmp_path / "invalid-domain.db"
    db = owned_agent_runs_db(path)
    goal = db.goal_runs.create(request(), launch_id="domain")
    goal = db.goal_runs.set_provisioning(goal, status="ready")
    attempt = db.automatic_work.prepare_goal_iteration(goal.id, owner_id="before")
    with db.connection() as conn:
        if corruption == "attempt_kind":
            conn.execute(
                "UPDATE automatic_wake_attempts SET attempt_kind='fleet_wake' WHERE id=?",
                (attempt.id,),
            )
        elif corruption == "status":
            conn.execute(
                "UPDATE goal_runs SET status='invented' WHERE id=?", (goal.id,)
            )
    if corruption == "reservation_owner":
        other = db.goal_runs.create(request(), launch_id="other-owner")
        with db.connection() as conn:
            conn.execute(
                "UPDATE goal_payload_reservations SET goal_id=? WHERE attempt_id=?",
                (other.id, attempt.id),
            )
    db.close()
    before = path.read_bytes()
    assert validate_candidate(owner(), path, Event(), migrate=True)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "field",
    ["id", "goal_id", "attempt_id", "run_id", "checkpoint_id", "checkpoint_ordinal"],
)
def test_goal_archive_rejects_foreign_payload_identity(tmp_path, field):
    from tldw_chatbook.Agents.goal_models import (
        GoalCheckpoint,
        GoalDecision,
        GoalEvidence,
        IterationReport,
    )

    path = tmp_path / "foreign-payload.db"
    db = owned_agent_runs_db(path)
    goal = db.goal_runs.create(request(), launch_id="payload")
    goal = db.goal_runs.set_provisioning(goal, status="ready")
    attempt = db.automatic_work.prepare_goal_iteration(goal.id, owner_id="before")
    run_id = db.create_run(
        conversation_id=goal.conversation_id,
        agent_kind="primary",
        work_chain_id=goal.chain_id,
    )
    evidence = GoalEvidence(
        id="evidence",
        goal_id=goal.id,
        attempt_id=attempt.id,
        run_id=run_id,
        verifier_id="check",
        source_digest="s",
        checked_manifest="m",
        verifier_sha256="h",
        passed=True,
        fresh=True,
        reason="passed",
    )
    checkpoint = GoalCheckpoint(
        id=attempt.id,
        ordinal=1,
        report=IterationReport(),
        decision=GoalDecision(action="continue", reason="fixture"),
    )
    with db.transaction() as conn:
        conn.execute(
            "UPDATE goal_iterations SET run_id=? WHERE attempt_id=?",
            (run_id, attempt.id),
        )
        conn.execute(
            "INSERT INTO goal_evidence VALUES (?,?,?,?)",
            (evidence.id, goal.id, attempt.id, evidence.model_dump_json()),
        )
        conn.execute(
            "INSERT INTO goal_checkpoints VALUES (?,?,?,?)",
            (attempt.id, goal.id, "fingerprint", checkpoint.model_dump_json()),
        )
    db.close()
    assert validate_candidate(owner(), path, Event(), migrate=False) == ()
    with closing(sqlite3.connect(path)) as conn:
        if field.startswith("checkpoint_"):
            key = field.removeprefix("checkpoint_")
            corrupted = checkpoint.model_copy(
                update={key: 2 if key == "ordinal" else "foreign"}
            )
            conn.execute(
                "UPDATE goal_checkpoints SET payload_json=?",
                (corrupted.model_dump_json(),),
            )
        else:
            corrupted = evidence.model_copy(update={field: "foreign"})
            conn.execute(
                "UPDATE goal_evidence SET payload_json=?",
                (corrupted.model_dump_json(),),
            )
        conn.commit()
    before = path.read_bytes()
    assert validate_candidate(owner(), path, Event(), migrate=False)
    assert path.read_bytes() == before

    reopened = owned_agent_runs_db(path)
    try:
        with pytest.raises(ValueError, match="invalid_goal_ownership"):
            if field.startswith("checkpoint_"):
                reopened.goal_runs.get(goal.id)
            else:
                reopened.goal_runs.evidence(goal.id)
    finally:
        reopened.close()


@pytest.mark.parametrize("corruption", ["report_owner", "iteration_run_owner"])
def test_goal_archive_rejects_cross_owner_report_or_native_run(tmp_path, corruption):
    from tldw_chatbook.Agents.goal_models import GoalReport

    path = tmp_path / "cross-owner.db"
    db = owned_agent_runs_db(path)
    goal = db.goal_runs.create(request(), launch_id="first")
    goal = db.goal_runs.set_provisioning(goal, status="ready")
    attempt = db.automatic_work.prepare_goal_iteration(goal.id, owner_id="owner")
    other = db.goal_runs.create(request(), launch_id="second")
    foreign_run = db.create_run(
        conversation_id=other.conversation_id,
        agent_kind="primary",
        work_chain_id=other.chain_id,
    )
    with db.transaction() as conn:
        if corruption == "report_owner":
            iteration_id = conn.execute(
                "SELECT id FROM goal_iterations WHERE attempt_id=?", (attempt.id,)
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO goal_reports VALUES (?,?,?,?)",
                (
                    "foreign-report",
                    other.id,
                    iteration_id,
                    GoalReport().model_dump_json(),
                ),
            )
        else:
            conn.execute(
                "UPDATE goal_iterations SET run_id=? WHERE attempt_id=?",
                (foreign_run, attempt.id),
            )
    if corruption == "report_owner":
        with pytest.raises(ValueError, match="invalid_goal_ownership"):
            db.goal_runs.get(other.id)
    db.close()
    before = path.read_bytes()
    assert validate_candidate(owner(), path, Event(), migrate=False)
    assert path.read_bytes() == before
