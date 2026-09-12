"""Orchestration upgrades preserve upstream step, spawn, and receipt records."""

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

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
    db = AgentRunsDB(path)
    run_id = db.create_run(
        conversation_id="conversation",
        agent_kind="subagent",
        spawn_event_id="spawn-event",
        run_id="stable-child",
    )
    db.append_steps(run_id, [{"index": 0, "kind": "note", "summary": "saved step"}])
    db.set_status(run_id, "done", "saved result")
    receipt_id, _ = db.publish_console_activity(
        origin="fleet_survivor",
        logical_outcome_id="outcome",
        status="done",
        session_id=None,
        conversation_id="conversation",
        run_id=run_id,
    )
    with db.transaction() as conn:
        names = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' "
            "AND name LIKE 'automatic_%'"
        ).fetchall()
        for (name,) in names:
            conn.execute('DROP TRIGGER "' + name.replace('"', '""') + '"')
        for table in (
            "automatic_wake_claims",
            "automatic_wake_attempts",
            "automatic_work_reservations",
            "automatic_work_runtime_owner",
        ):
            conn.execute(f"DROP TABLE {table}")
        conn.execute("ALTER TABLE agent_runs DROP COLUMN work_chain_id")
        conn.execute("ALTER TABLE agent_runs DROP COLUMN budget_tokens")
        conn.execute("DROP TABLE automatic_work_chains")
        conn.execute("DELETE FROM schema_version WHERE version > 15")
    db.close()
    return run_id, receipt_id


@pytest.mark.parametrize("standalone", [False, True])
def test_v15_upgrade_preserves_upstream_records_and_authority(tmp_path, standalone):
    path = tmp_path / "runs.db"
    run_id, receipt_id = _upstream_v15_database(path)
    if standalone:
        for version, name in (
            (16, "agent_runs_v15_to_v16_budget_tokens.sql"),
            (17, "agent_runs_v16_to_v17_automatic_work.sql"),
            (18, "agent_runs_v17_to_v18_runtime_owner.sql"),
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
                    == 18
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
