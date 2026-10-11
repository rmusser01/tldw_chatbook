"""v23 upgrade keeps existing fleet attempts and their default kind."""

import sqlite3
from pathlib import Path

import pytest

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


@pytest.mark.parametrize("standalone", [False, True])
def test_v23_goal_upgrade_preserves_fleet_history(tmp_path, standalone):
    path = tmp_path / "runs.db"
    db = AgentRunsDB(path)
    chain = db.automatic_work.create_chain("conversation", root_submission_id="root")
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO automatic_work_reservations (id, chain_id, owner_id, kind, amount, state, created_at, updated_at) VALUES ('res', ?, 'owner', 'generation', 1, 'committed', 1, 1)",
            (chain,),
        )
        conn.execute(
            "INSERT INTO automatic_wake_attempts (id, chain_id, conversation_id, session_id, owner_id, generation_reservation_id, run_ids_json, state, created_at) VALUES ('attempt', ?, 'conversation', 'session', 'owner', 'res', '[]', 'completed', 1)",
            (chain,),
        )
        # Reconstruct the previous schema from a real database, retaining rows.
        for name in (
            "goal_waits",
            "goal_payload_reservations",
            "goal_evidence",
            "goal_checkpoints",
            "goal_reports",
            "goal_iterations",
            "goal_runs",
        ):
            conn.execute(f"DROP TABLE IF EXISTS {name}")
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(automatic_wake_attempts)")
        }
        if "attempt_kind" in columns:
            conn.execute("ALTER TABLE automatic_wake_attempts DROP COLUMN attempt_kind")
        conn.execute("DELETE FROM schema_version WHERE version>23")
    db.close()
    if standalone:
        sql = Path(
            "tldw_chatbook/DB/migrations/agent_runs_v23_to_v24_goal_runs.sql"
        ).read_text()
        with sqlite3.connect(path) as conn:
            conn.executescript(sql)
    db = AgentRunsDB(path)
    with db.connection() as conn:
        row = conn.execute(
            "SELECT * FROM automatic_wake_attempts WHERE id='attempt'"
        ).fetchone()
        assert row["attempt_kind"] == "fleet_wake"
        assert row["state"] == "completed"
        assert row["chain_id"] == chain
        assert (
            conn.execute("SELECT max(version) FROM schema_version").fetchone()[0]
            == AgentRunsDB._CURRENT_SCHEMA_VERSION
        )
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    db.close()
    AgentRunsDB(path).close()  # guarded runtime migration is repeatable


def test_standalone_goal_migration_matches_runtime_catalog(tmp_path):
    """The installed audit SQL and guarded constructor produce identical DDL."""
    from tldw_chatbook.DB.recovery_operations import _AGENT_RUNS_SCHEMA

    predecessor = next(
        schema for version, schema in _AGENT_RUNS_SCHEMA if version == 23
    )
    paths = [tmp_path / "runtime.db", tmp_path / "standalone.db"]
    for path in paths:
        with sqlite3.connect(path) as conn:
            for sql in sorted(
                predecessor, key=lambda sql: not sql.startswith("CREATE TABLE")
            ):
                if not sql.startswith("CREATE TABLE sqlite_sequence"):
                    conn.execute(sql)
            conn.execute("INSERT INTO schema_version(version) VALUES (23)")
    with sqlite3.connect(paths[1]) as conn:
        conn.executescript(
            Path(
                "tldw_chatbook/DB/migrations/agent_runs_v23_to_v24_goal_runs.sql"
            ).read_text()
        )
    AgentRunsDB(paths[0]).close()
    catalogs = []
    for path in paths:
        with sqlite3.connect(path) as conn:
            catalogs.append(
                conn.execute(
                    "SELECT type,name,tbl_name,sql FROM sqlite_schema ORDER BY type,name"
                ).fetchall()
            )
    assert catalogs[0] == catalogs[1]
