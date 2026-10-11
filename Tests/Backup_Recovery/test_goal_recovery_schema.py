"""Current goal archives retain exact schema and inert durable authority."""

import sqlite3
from contextlib import closing
from threading import Event

import pytest

from Tests.Agents.test_goal_models import request
from Tests.Backup_Recovery.test_agent_runs_recovery_schema import owner
from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def test_goal_archive_is_exact_and_restore_never_accepts_work(tmp_path):
    path = tmp_path / "goals.db"
    db = AgentRunsDB(path)
    goal = db.goal_runs.create(request(), launch_id="launch")
    goal = db.goal_runs.set_provisioning(goal, status="ready")
    attempt = db.automatic_work.prepare_goal_iteration(goal.id, owner_id="before")
    db.close()
    before = path.read_bytes()
    assert validate_candidate(owner(), path, Event(), migrate=True) == ()
    assert path.read_bytes() == before
    db = AgentRunsDB(path)
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
    AgentRunsDB(path).close()
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


@pytest.mark.parametrize("corruption", ["attempt_kind", "status"])
def test_goal_archive_rejects_mismatched_domain_without_rewrite(tmp_path, corruption):
    path = tmp_path / "invalid-domain.db"
    db = AgentRunsDB(path)
    goal = db.goal_runs.create(request(), launch_id="domain")
    goal = db.goal_runs.set_provisioning(goal, status="ready")
    attempt = db.automatic_work.prepare_goal_iteration(goal.id, owner_id="before")
    with db.connection() as conn:
        if corruption == "attempt_kind":
            conn.execute(
                "UPDATE automatic_wake_attempts SET attempt_kind='fleet_wake' WHERE id=?",
                (attempt.id,),
            )
        else:
            conn.execute(
                "UPDATE goal_runs SET status='invented' WHERE id=?", (goal.id,)
            )
    db.close()
    before = path.read_bytes()
    assert validate_candidate(owner(), path, Event(), migrate=True)
    assert path.read_bytes() == before
