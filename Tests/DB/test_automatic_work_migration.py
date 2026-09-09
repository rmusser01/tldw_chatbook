"""Both the runtime migration and standalone v13->v14 artifact preserve history."""

import sqlite3
from pathlib import Path

import pytest

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def v13_database(path):
    db = AgentRunsDB(path)
    run_id = db.create_run(conversation_id="legacy", agent_kind="primary")
    db.set_status(run_id, "done", "original result", budget_tokens=23)
    with db.transaction() as conn:
        triggers = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'automatic_%'"
            )
        ]
        for name in triggers:
            # Only schema-owned identifiers read from this newly created DB.
            conn.execute('DROP TRIGGER "' + name.replace('"', '""') + '"')
        conn.execute("DROP TABLE automatic_wake_claims")
        conn.execute("DROP TABLE automatic_wake_attempts")
        conn.execute("DROP TABLE automatic_work_reservations")
        conn.execute("DROP TABLE IF EXISTS automatic_work_runtime_owner")
        conn.execute("ALTER TABLE agent_runs DROP COLUMN work_chain_id")
        conn.execute("DROP TABLE automatic_work_chains")
        conn.execute("DELETE FROM schema_version WHERE version>=14")
        assert (
            conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 13
        )
        assert "work_chain_id" not in {
            row[1] for row in conn.execute("PRAGMA table_info(agent_runs)")
        }
        assert (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE name LIKE 'automatic_%'"
            ).fetchall()
            == []
        )
    db.close()
    return run_id


@pytest.mark.parametrize("standalone", [False, True])
def test_v13_upgrade_preserves_legacy_rows_without_giving_them_allowance(
    tmp_path, standalone
):
    path = tmp_path / "runs.sqlite"
    run_id = v13_database(path)
    if standalone:
        script = Path(
            "tldw_chatbook/DB/migrations/agent_runs_v13_to_v14_automatic_work.sql"
        ).read_text()
        with sqlite3.connect(path) as conn:
            conn.executescript(script)
            assert (
                conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
                == 14
            )
    db = AgentRunsDB(path)
    legacy = db.get_run(run_id)
    assert legacy["result"] == "original result"
    assert legacy["budget_tokens"] == 23
    assert legacy["work_chain_id"] is None
    with db.connection() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM automatic_work_chains").fetchone()[0]
            == 0
        )
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    chain_id = db.automatic_work.create_chain("new", root_submission_id="new")
    db.close()
    reopened = AgentRunsDB(path)
    assert reopened.automatic_work.snapshot(chain_id).limits.generations == 3
    assert reopened.get_run(run_id)["work_chain_id"] is None
    reopened.close()
