"""Orchestration upgrades preserve upstream step, spawn, and receipt records."""

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from Tests.DB.fixtures.agent_runs_history import install_historical
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def test_ordinary_run_storage_defers_automatic_execution_modules():
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import sys
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
db = AgentRunsDB(':memory:')
run_id = db.create_run(conversation_id='c', agent_kind='primary')
db.set_status(run_id, 'done', budget_tokens=23)
assert db.get_run(run_id)['budget_tokens'] == 23
assert 'tldw_chatbook.DB.automatic_work' not in sys.modules
assert 'tldw_chatbook.Agents.automatic_work_budget' not in sys.modules
chain_id = db.automatic_work.create_chain('c', root_submission_id='root')
assert db.automatic_work.snapshot(chain_id).limits.generations == 3
db.close()
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _upstream_v15_database(path):
    run_id, receipt_id = "stable-child", "stable-receipt"
    with sqlite3.connect(path) as conn:
        install_historical(conn, 15)
        conn.execute(
            "INSERT INTO agent_runs(id,conversation_id,agent_kind,spawn_event_id,status,result,created_at,updated_at) "
            "VALUES (?, 'conversation', 'subagent', 'spawn-event', 'done', 'saved result', 'then', 'then')",
            (run_id,),
        )
        conn.execute(
            "INSERT INTO agent_run_steps(run_id,seq,payload,created_at) VALUES (?,0,?,'then')",
            (run_id, '{"index":0,"kind":"note","summary":"saved step"}'),
        )
        conn.execute(
            "INSERT INTO console_activity_receipts(activity_id,origin,logical_outcome_id,transition_revision,conversation_id,run_id,status,created_at) "
            "VALUES (?, 'fleet_survivor', 'outcome', 1, 'conversation', ?, 'done', 'then')",
            (receipt_id, run_id),
        )
    return run_id, receipt_id


@pytest.mark.parametrize("standalone", [False, True])
def test_v15_upgrade_preserves_upstream_records_and_authority(tmp_path, standalone):
    path = tmp_path / "runs.db"
    run_id, receipt_id = _upstream_v15_database(path)
    if standalone:
        # task-19565: the standalone path applies EVERY agent_runs migration
        # file -- they are the hand-applied upgrade route for databases the
        # app cannot open, so an unexercised file in the family is a file
        # that can silently rot. v18->v21 were previously app-path-only.
        for version, name in (
            (16, "agent_runs_v15_to_v16_budget_tokens.sql"),
            (17, "agent_runs_v16_to_v17_automatic_work.sql"),
            (18, "agent_runs_v17_to_v18_runtime_owner.sql"),
            (19, "agent_runs_v18_to_v19_definition_wall_seconds.sql"),
            (20, "agent_runs_v19_to_v20_worktree_recovery.sql"),
            (21, "agent_runs_v20_to_v21_agent_routing.sql"),
        ):
            with sqlite3.connect(path) as conn:
                conn.executescript(
                    (Path("tldw_chatbook/DB/migrations") / name).read_text()
                )
                assert (
                    conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[
                        0
                    ]
                    == version
                )
    for _ in range(2):
        db = AgentRunsDB(path)
        try:
            row = db.get_run(run_id)
            assert row["status"] == "done"
            assert row["result"] == "saved result"
            assert row["spawn_event_id"] == "spawn-event"
            assert row["steps"][0]["summary"] == "saved step"
            assert row["budget_tokens"] is None
            assert row["work_chain_id"] is None
            assert db.list_unseen_console_activity()[0]["activity_id"] == receipt_id
            with db.connection() as conn:
                assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
                assert (
                    conn.execute(
                        "SELECT COUNT(*) FROM automatic_work_chains"
                    ).fetchone()[0]
                    == 0
                )
                assert (
                    conn.execute(
                        "SELECT COUNT(*) FROM automatic_work_runtime_owner"
                    ).fetchone()[0]
                    == 0
                )
                assert (
                    conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[
                        0
                    ]
                    == AgentRunsDB._CURRENT_SCHEMA_VERSION
                )
                assert (
                    conn.execute("SELECT COUNT(*) FROM agent_worktrees").fetchone()[0]
                    == 0
                )
                assert (
                    conn.execute(
                        "SELECT COUNT(*) FROM agent_definitions WHERE max_wall_seconds IS NOT NULL"
                    ).fetchone()[0]
                    == 0
                )
        finally:
            db.close()


def test_terminal_budget_is_atomic_with_upstream_lifecycle_and_visible_in_metadata(
    tmp_path,
):
    db = AgentRunsDB(tmp_path / "runs.db")
    try:
        chain_id = db.automatic_work.create_chain("c", root_submission_id="root")
        run_id = db.create_run(
            conversation_id="c", agent_kind="subagent", work_chain_id=chain_id
        )
        step = {"index": 10_000_010, "kind": "agent_run_completed"}
        assert db.set_terminal_with_step(
            run_id, "done", "answer", step, budget_tokens=23
        )
        assert not db.set_terminal_with_step(
            run_id, "done", "late answer", step, budget_tokens=46
        )
        row = db.get_run_metadata(run_id)
        assert row["budget_tokens"] == 23
        assert row["work_chain_id"] == chain_id
        assert row["result"] == "answer"
        assert "steps" not in row
        assert db.get_run(run_id)["steps"] == [step]
    finally:
        db.close()
