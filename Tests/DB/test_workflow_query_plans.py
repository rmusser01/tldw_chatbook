"""Workflow paging, active clocks and task claims use their indexes without stats."""

from contextlib import closing

import sqlite3

import pytest

from Tests.DB.test_workflow_runs import attempt, planned
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

pytestmark = pytest.mark.bootstrap_profile


def test_workflow_indexes_are_used_without_sqlite_stat1(tmp_path):
    with closing(AgentRunsDB(tmp_path / "plans.db")) as db:
        for index in range(32):
            snapshot = planned(
                db, tmp_path, f"launch-{index}", conversation_id=f"chat-{index}"
            )
            task = attempt(db, snapshot)
            with db.transaction() as conn:
                conn.execute(
                    "INSERT INTO workflow_phase_attempts(id,workflow_id,plan_revision,phase_id,ordinal,status) VALUES (?, ?, 1, 'phase-1', 1, 'prepared')",
                    (f"phase-{index}", snapshot.id),
                )
                conn.execute(
                    "INSERT INTO workflow_task_claims(task_attempt_id,phase_attempt_id,launch_run_id) VALUES (?, ?, ?)",
                    (task.id, f"phase-{index}", f"future-run-{index}"),
                )
                conn.execute(
                    "INSERT INTO workflow_active_intervals(id,workflow_id,owner_id,started_at) VALUES (?, ?, ?, 1)",
                    (f"clock-{index}", snapshot.id, "owner"),
                )
        with db.connection() as conn:
            assert (
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='sqlite_stat1'"
                ).fetchone()
                is None
            )
            statements = []
            conn.set_trace_callback(statements.append)
            try:
                assert len(db.workflow_runs.list_for_conversation("chat-3")) == 1
            finally:
                conn.set_trace_callback(None)
            page = next(
                sql for sql in statements if sql.startswith("SELECT id,conversation_id")
            )
            probes = (
                ("idx_workflow_conversation", page, ()),
                # AutomaticWorkLedger's live-clock guard and task-claim lookup.
                (
                    "idx_workflow_open_interval",
                    "SELECT 1 FROM workflow_active_intervals WHERE workflow_id=? AND ended_at IS NULL",
                    (snapshot.id,),
                ),
                (
                    "idx_workflow_task_phase",
                    "SELECT 1 FROM workflow_task_claims WHERE phase_attempt_id=?",
                    ("phase-3",),
                ),
            )
            for index, query, params in probes:
                plan = [
                    row[3]
                    for row in conn.execute("EXPLAIN QUERY PLAN " + query, params)
                ]
                assert any(index in detail and "SEARCH" in detail for detail in plan), (
                    plan
                )
                print(index, plan)
            with pytest.raises(
                sqlite3.IntegrityError,
                match="UNIQUE constraint failed: workflow_active_intervals.workflow_id",
            ):
                conn.execute(
                    "INSERT INTO workflow_active_intervals(id,workflow_id,owner_id,started_at) VALUES (?, ?, ?, 1)",
                    ("duplicate-open-clock", snapshot.id, "owner"),
                )
