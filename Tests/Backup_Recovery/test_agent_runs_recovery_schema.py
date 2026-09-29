"""Merged AgentRuns catalogs remain exact, with a bounded legacy migration."""

import sqlite3
from contextlib import closing
from pathlib import Path
from threading import Event

import pytest

from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.recovery_operations import _AGENT_RUNS_SCHEMA, recovery_adapters

# Literal CREATE statement from the native v19/v20 constructor, before v21.
_HISTORICAL_DEFINITIONS = "CREATE TABLE IF NOT EXISTS agent_definitions (\n                    id TEXT PRIMARY KEY,\n                    name TEXT NOT NULL,\n                    description TEXT NOT NULL DEFAULT '',\n                    instructions TEXT NOT NULL DEFAULT '',\n                    tool_allowlist TEXT NOT NULL DEFAULT '[]',\n                    model TEXT NOT NULL DEFAULT '',\n                    enabled INTEGER NOT NULL DEFAULT 1,\n                    max_wall_seconds REAL,\n                    deleted INTEGER NOT NULL DEFAULT 0,\n                    created_at TEXT NOT NULL,\n                    updated_at TEXT NOT NULL\n                )"

def owner():
    return next(row for row in recovery_adapters() if row.owner_id == "db.agent_runs")


def legacy(path: Path, version=18):
    schema = next(sql for version, sql in _AGENT_RUNS_SCHEMA if version == 18)
    with closing(sqlite3.connect(path)) as connection:
        # This literal catalog is the already qualified actual v18 layout.
        for sql in sorted(schema, key=lambda sql: not sql.startswith("CREATE TABLE")):
            if sql.startswith("CREATE TABLE sqlite_sequence"):
                continue
            if version in (19, 20) and sql.startswith("CREATE TABLE agent_definitions "):
                sql = _HISTORICAL_DEFINITIONS
            connection.execute(sql)
        if version == 20:
            from tldw_chatbook.DB.recovery_operations import (
                _AGENT_WORKTREES_INDEX,
                _AGENT_WORKTREES_TABLE,
            )

            connection.execute(_AGENT_WORKTREES_TABLE)
            connection.execute(_AGENT_WORKTREES_INDEX)
        connection.execute("INSERT INTO schema_version VALUES (?)", (version,))
        connection.execute(
            "INSERT INTO agent_definitions(id,name,created_at,updated_at) VALUES ('kept','Saved preset','then','then')"
        )
        connection.execute(
            "INSERT INTO agent_runs(id,conversation_id,agent_kind,status,result,created_at,updated_at) "
            "VALUES ('retained-run','saved-conversation','primary','completed','saved result','then','then')"
        )
        connection.commit()
    path.chmod(0o600)
    return path


@pytest.mark.parametrize("route", ["legacy", "fresh", "native_migrated", "native_v19", "native_v20"])
def test_exact_agent_catalogs_validate_without_rewriting(tmp_path, route):
    path = tmp_path / "agents.db"
    if route != "fresh":
        legacy(path, {"native_v19": 19, "native_v20": 20}.get(route, 18))
    if route != "legacy":
        from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

        database = AgentRunsDB(path)
        database.close()
    before = path.read_bytes()
    assert validate_candidate(owner(), path, Event(), migrate=False) == ()
    assert path.read_bytes() == before


def test_legacy_agent_candidate_migration_preserves_history(tmp_path):
    path = legacy(tmp_path / "agents.db")
    assert validate_candidate(owner(), path, Event(), migrate=True) == ()
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone() == (AgentRunsDB._CURRENT_SCHEMA_VERSION,)
        assert connection.execute(
            "SELECT id,name,provider,params_json,max_wall_seconds FROM agent_definitions"
        ).fetchall() == [("kept", "Saved preset", "", "{}", None)]
        assert connection.execute(
            "SELECT id,conversation_id,status,result,resolved_provider,resolved_model,resolved_base_url,resolved_params_json FROM agent_runs"
        ).fetchall() == [
            (
                "retained-run",
                "saved-conversation",
                "completed",
                "saved result",
                None,
                None,
                None,
                None,
            )
        ]
    assert validate_candidate(owner(), path, Event(), migrate=False) == ()


@pytest.mark.parametrize(
    "alteration",
    [
        "extra_table",
        "future_version",
        "relabeled_legacy",
        "v21_stamped_v22",
        "v22_stamped_v21",
    ],
)
def test_unknown_agent_schema_or_version_refuses(tmp_path, alteration):
    path = tmp_path / "agents.db"
    if alteration == "v21_stamped_v22":
        schema = next(schema for version, schema in _AGENT_RUNS_SCHEMA if version == 21)
        with closing(sqlite3.connect(path)) as connection:
            for sql in sorted(
                schema, key=lambda sql: not sql.startswith("CREATE TABLE")
            ):
                if not sql.startswith("CREATE TABLE sqlite_sequence"):
                    connection.execute(sql)
            connection.execute("INSERT INTO schema_version VALUES (22)")
            connection.commit()
        path.chmod(0o600)
    elif alteration == "relabeled_legacy":
        legacy(path)
    else:
        from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

        database = AgentRunsDB(path)
        database.close()
    with closing(sqlite3.connect(path)) as connection:
        if alteration == "extra_table":
            connection.execute("CREATE TABLE unrecognized(payload TEXT)")
        elif alteration == "v22_stamped_v21":
            connection.execute("DELETE FROM schema_version WHERE version=22")
        elif alteration != "v21_stamped_v22":
            connection.execute(
                "INSERT INTO schema_version VALUES (?)",
                (
                    21
                    if alteration == "relabeled_legacy"
                    else AgentRunsDB._CURRENT_SCHEMA_VERSION + 1,
                ),
            )
        connection.commit()
    before = path.read_bytes()
    assert validate_candidate(owner(), path, Event(), migrate=True)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "case",
    [
        "outside_window",
        "wrong_owner",
        "wrong_table",
        "wrong_index",
        "wrong_database",
        "wrong_reindex",
        "row_delete",
        "unrelated_trigger",
        "writable_schema",
        "other_database_trigger",
    ],
)
def test_agent_migration_authority_is_exact_and_temporary(tmp_path, case):
    from tldw_chatbook.DB.private_sqlite import open_recovery_validation

    path = legacy(tmp_path / "agents.db")
    with open_recovery_validation(
        "db.agent_runs", path, writable=True, with_restrictions=True
    ) as (connection, restrictions):
        restrictions.migrating = case != "outside_window"
        restrictions.migration_owner = (
            "research.local" if case == "wrong_owner" else "db.agent_runs"
        )
        sql = {
            "outside_window": "CREATE TABLE agent_worktrees(run_id TEXT PRIMARY KEY)",
            "wrong_owner": "CREATE TABLE agent_worktrees(run_id TEXT PRIMARY KEY)",
            "wrong_table": "CREATE TABLE unrelated(payload TEXT)",
            "wrong_index": "CREATE INDEX unrelated ON agent_definitions(name)",
            "wrong_database": "CREATE TEMP TABLE agent_worktrees(run_id TEXT PRIMARY KEY)",
            "wrong_reindex": "REINDEX idx_agent_definitions_name",
            "row_delete": "DELETE FROM agent_definitions",
            "unrelated_trigger": "DROP TRIGGER automatic_run_chain_immutable",
            "writable_schema": "PRAGMA writable_schema=ON",
            "other_database_trigger": "CREATE TEMP TRIGGER automatic_chain_root_insert AFTER INSERT ON agent_definitions BEGIN SELECT 1; END",
        }[case]
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute(sql)


def test_agent_migration_cancellation_rolls_back_all_schema_and_rows(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import sqlite_validation as validation

    path = legacy(tmp_path / "agents.db")
    before = path.read_bytes()
    cancel = Event()
    native = validation._Restrictions.authorize

    def interrupted(self, action, first, second, database, source):
        if self.migrating and action == sqlite3.SQLITE_ALTER_TABLE:
            cancel.set()
        return native(self, action, first, second, database, source)

    monkeypatch.setattr(validation._Restrictions, "authorize", interrupted)
    assert validate_candidate(owner(), path, cancel, migrate=True) == ("cancelled",)
    assert path.read_bytes() == before
    assert validate_candidate(owner(), path, Event(), migrate=False) == ()


def test_fixed_v21_to_v22_recovery_route_preserves_historical_variants(tmp_path):
    for index, schema in enumerate(
        sql for version, sql in _AGENT_RUNS_SCHEMA if version == 21
    ):
        path = tmp_path / f"predecessor-{index}.db"
        with closing(sqlite3.connect(path)) as connection:
            for sql in sorted(
                schema, key=lambda sql: not sql.startswith("CREATE TABLE")
            ):
                if sql.startswith("CREATE TABLE sqlite_sequence"):
                    continue
                connection.execute(sql)
            connection.execute("INSERT INTO schema_version VALUES (21)")
            connection.execute(
                "INSERT INTO agent_definitions(id,name,max_wall_seconds,provider,params_json,created_at,updated_at) VALUES ('kept','Kept',12.5,'openai','{}','then','then')"
            )
            connection.execute(
                "INSERT INTO automatic_work_chains(id,conversation_id,root_submission_id,limits_json,created_at,last_observed_at) VALUES ('kept-chain','conversation','root','{}',1.0,2.0)"
            )
            connection.execute(
                "INSERT INTO automatic_work_reservations(id,chain_id,owner_id,kind,amount,state,actual_amount,created_at,updated_at) VALUES ('kept-reservation','kept-chain','owner','tokens',25,'settled',17,1.0,2.0)"
            )
            connection.commit()
        path.chmod(0o600)
        assert validate_candidate(owner(), path, Event(), migrate=True) == ()
        with closing(sqlite3.connect(path)) as connection:
            assert connection.execute(
                "SELECT MAX(version) FROM schema_version"
            ).fetchone() == (22,)
            assert connection.execute(
                "SELECT max_wall_seconds,provider,params_json FROM agent_definitions"
            ).fetchall() == [(12.5, "openai", "{}")]
            assert connection.execute(
                "SELECT id,conversation_id,root_submission_id,limits_json,created_at,last_observed_at,allowance_root_chain_id FROM automatic_work_chains"
            ).fetchall() == [
                ("kept-chain", "conversation", "root", "{}", 1.0, 2.0, None)
            ]
            assert connection.execute(
                "SELECT id,chain_id,owner_id,kind,amount,state,actual_amount,created_at,updated_at FROM automatic_work_reservations"
            ).fetchall() == [
                (
                    "kept-reservation",
                    "kept-chain",
                    "owner",
                    "tokens",
                    25,
                    "settled",
                    17,
                    1.0,
                    2.0,
                )
            ]
            assert not connection.execute("PRAGMA foreign_key_check").fetchall()
        assert validate_candidate(owner(), path, Event(), migrate=False) == ()
