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
    goal = Path("tldw_chatbook/DB/migrations/agent_runs_v23_to_v24_goal_runs.sql")
    for path in paths:
        with sqlite3.connect(path) as conn:
            conn.executescript(goal.read_text())
            installed = tuple(
                row[0]
                for row in conn.execute(
                    "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL "
                    "ORDER BY type,name"
                )
            )
        # The goal SQL lands on a recovery-qualified v24 catalog.
        assert any(
            sorted(installed) == sorted(schema)
            for version, schema in _AGENT_RUNS_SCHEMA
            if version == 24
        )
    # From that v24 database, the constructor's v24->v25 upgrade and the
    # standalone review-boundary SQL (ADR-281) produce identical DDL.
    with sqlite3.connect(paths[1]) as conn:
        conn.executescript(
            Path(
                "tldw_chatbook/DB/migrations/agent_runs_v24_to_v25_review_boundaries.sql"
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


@pytest.mark.parametrize("upgrade", ["recovery", "constructor"])
def test_v25_recovery_accepts_both_upgrades_and_still_checks_goal_rows(
    tmp_path, upgrade
):
    """ADR-281's v25 keeps v24 goal validation; both upgrade lineages qualify."""
    from threading import Event

    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
    from tldw_chatbook.DB.recovery_operations import (
        _AGENT_RUNS_SCHEMA,
        recovery_adapters,
    )

    owner = next(
        item for item in recovery_adapters() if item.owner_id == "db.agent_runs"
    )
    path = tmp_path / "v24.db"
    schema = next(catalog for version, catalog in _AGENT_RUNS_SCHEMA if version == 24)
    with sqlite3.connect(path) as conn:
        for sql in sorted(schema, key=lambda sql: not sql.startswith("CREATE TABLE")):
            if not sql.startswith("CREATE TABLE sqlite_sequence"):
                conn.execute(sql)
        conn.execute("INSERT INTO schema_version(version) VALUES (24)")
    path.chmod(0o600)
    if upgrade == "constructor":
        AgentRunsDB(path).close()
    assert validate_candidate(owner, path, Event(), migrate=True) == ()
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone() == (
            25,
        )
        conn.execute(
            "INSERT INTO automatic_work_chains(id,conversation_id,root_submission_id,limits_json,created_at,last_observed_at) VALUES ('chain','chat','manual','{}',0,0)"
        )
        conn.execute(
            "INSERT INTO automatic_work_reservations(id,chain_id,owner_id,kind,amount,state,created_at,updated_at) VALUES ('reserve','chain','owner','generation',1,'committed',0,0)"
        )
        # A goal attempt with no owning iteration is invalid goal ownership.
        conn.execute(
            "INSERT INTO automatic_wake_attempts(id,chain_id,conversation_id,session_id,owner_id,generation_reservation_id,run_ids_json,state,created_at,attempt_kind) VALUES ('orphan','chain','chat','native','owner','reserve','[]','completed',0,'goal_iteration')"
        )
    with sqlite3.connect(path) as conn:  # the constructor left WAL mode on
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    assert validate_candidate(owner, path, Event(), migrate=False) == (
        "invalid_goal_ownership",
    )


@pytest.mark.parametrize("rows", [(), (None,)])
def test_goal_row_validation_refuses_a_missing_schema_version(rows):
    """Admission runs first today; a missing version must still fail closed."""
    from tldw_chatbook.DB.recovery_goal_schema import validate_goal_rows

    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE schema_version (version INTEGER)")
    for version in rows:
        connection.execute("INSERT INTO schema_version VALUES (?)", (version,))
    assert validate_goal_rows(connection) == ("unsupported_schema_version",)
