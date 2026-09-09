"""Real launch identity, rollback, reopening and revision conflicts."""

import importlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from Tests.Agents.test_goal_models import request
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def store(db):
    return importlib.import_module("tldw_chatbook.DB.goal_runs").GoalRunsStore(db)


def test_repeated_launch_and_reopen_preserve_one_allowance_and_private_report(tmp_path):
    path = tmp_path / "runs.db"
    db = AgentRunsDB(path)
    first = store(db).create(request(), launch_id="start-1")
    again = store(db).create(request(), launch_id="start-1")
    assert (again.id, again.chain_id, again.conversation_id) == (
        first.id,
        first.chain_id,
        first.conversation_id,
    )
    with pytest.raises(ValueError, match="launch_payload_conflict"):
        store(db).create(request(criteria="different"), launch_id="start-1")
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO goal_reports (id, goal_id, payload_json) VALUES (?, ?, ?)",
            ("report", first.id, '{"summary":"retained private result"}'),
        )
    db.close()
    reopened = AgentRunsDB(path)
    result = store(reopened).get(first.id)
    assert result.request.criteria == "Validation exits zero"
    assert result.chain_id == first.chain_id
    assert result.iteration_count == 0
    assert result.reports[0].summary == "retained private result"
    assert result.accounting.available["generation"] == 3
    assert result.accounting.available["child_launch"] == 0
    assert reopened.automatic_work.snapshot(first.chain_id).started_at is None
    reopened.close()


def test_chain_failure_rolls_back_both_launch_writes(tmp_path, monkeypatch):
    db = AgentRunsDB(tmp_path / "runs.db")
    ledger = db.automatic_work
    original = getattr(ledger, "_create_chain", None)
    assert original is not None, "chain allocation needs a connection-taking helper"

    def fail_after_chain(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected after chain insert")

    monkeypatch.setattr(ledger, "_create_chain", fail_after_chain)
    with pytest.raises(RuntimeError, match="injected"):
        store(db).create(request(), launch_id="start")
    with db.connection() as conn:
        assert conn.execute("SELECT count(*) FROM goal_runs").fetchone()[0] == 0
        assert (
            conn.execute("SELECT count(*) FROM automatic_work_chains").fetchone()[0]
            == 0
        )
    db.close()


def test_concurrent_launches_and_stale_revision_cannot_fork_identity(tmp_path):
    path = tmp_path / "runs.db"
    AgentRunsDB(path).close()

    def launch(_):
        db = AgentRunsDB(path)
        try:
            return store(db).create(request(), launch_id="start")
        finally:
            db.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, again = pool.map(launch, range(2))
    assert first.id == again.id
    db = AgentRunsDB(path)
    saved = store(db).set_provisioning(first, status="ready")
    assert saved.revision == first.revision + 1
    with pytest.raises(ValueError, match="revision_conflict"):
        store(db).set_provisioning(
            again, status="paused", pause_reason="binding_missing"
        )
    with (
        db.transaction() as conn,
        pytest.raises(sqlite3.IntegrityError, match="immutable"),
    ):
        conn.execute("UPDATE goal_runs SET request_json=? WHERE id=?", ("{}", first.id))
    db.close()
