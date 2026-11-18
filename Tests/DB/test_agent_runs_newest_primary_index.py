"""The original newest-primary readers use a bounded, recovery-owned index."""

import sqlite3
from contextlib import closing
from pathlib import Path
from threading import Event

import pytest

from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.recovery_operations import _AGENT_RUNS_SCHEMA, recovery_adapters

pytestmark = pytest.mark.bootstrap_profile

INDEX = "idx_agent_runs_newest_primary"
MIGRATION = Path(
    "tldw_chatbook/DB/migrations/agent_runs_v31_to_v32_newest_primary_index.sql"
)


def _owner():
    return next(
        owner for owner in recovery_adapters() if owner.owner_id == "db.agent_runs"
    )


def _catalog(conn):
    return tuple(
        row[0]
        for row in conn.execute(
            "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
        )
    )


def _v31(path):
    """Build a genuine frozen predecessor, never downgrade a current schema."""
    catalog = next(catalog for version, catalog in _AGENT_RUNS_SCHEMA if version == 31)
    with closing(sqlite3.connect(path)) as conn:
        for sql in sorted(catalog, key=lambda sql: not sql.startswith("CREATE TABLE")):
            if not sql.startswith("CREATE TABLE sqlite_sequence"):
                conn.execute(sql)
        conn.executemany(
            "INSERT INTO schema_version(version) VALUES (?)",
            [(version,) for version in range(4, 32)],
        )
        conn.execute(
            "INSERT INTO agent_runs(id,conversation_id,agent_kind,status,created_at,updated_at,steps) "
            "VALUES ('retained','conversation','primary','done','then','then','[]')"
        )
        conn.commit()
    path.chmod(0o600)


@pytest.mark.parametrize("route", ["constructor", "recovery", "standalone"])
def test_v31_additive_index_upgrade_preserves_history(tmp_path, route):
    path = tmp_path / "runs.db"
    _v31(path)
    before = path.read_bytes()
    assert validate_candidate(_owner(), path, Event(), migrate=False) == ()
    assert path.read_bytes() == before
    with closing(sqlite3.connect(path)) as conn:
        old_catalog = _catalog(conn)
        old_rows = conn.execute("SELECT * FROM agent_runs").fetchall()
    if route == "constructor":
        AgentRunsDB(path).close()
        AgentRunsDB(path).close()
    elif route == "recovery":
        assert validate_candidate(_owner(), path, Event(), migrate=True) == ()
    else:
        with closing(sqlite3.connect(path)) as conn:
            conn.executescript(MIGRATION.read_text(encoding="utf-8"))
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT * FROM agent_runs").fetchall() == old_rows
        assert set(_catalog(conn)) - set(old_catalog) == {
            conn.execute(
                "SELECT sql FROM sqlite_schema WHERE name=?", (INDEX,)
            ).fetchone()[0]
        }
        assert set(old_catalog) <= set(_catalog(conn))
        assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone() == (
            32,
        )
        assert _catalog(conn) in [
            catalog for version, catalog in _AGENT_RUNS_SCHEMA if version == 32
        ]
    assert validate_candidate(_owner(), path, Event(), migrate=False) == ()


def test_newest_primary_ties_filters_and_external_updates(tmp_path):
    with closing(AgentRunsDB(tmp_path / "runs.db")) as db:
        with db.connection() as conn:
            conn.executemany(
                "INSERT INTO agent_runs(id,conversation_id,agent_kind,status,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?)",
                [
                    ("a", "c", "primary", "done", "2026-01-01", "then"),
                    ("b", "c", "primary", "error", "2026-01-01", "then"),
                    ("newer-child", "c", "subagent", "done", "2026-01-03", "then"),
                    ("superseded", "c", "primary", "superseded", "2026-01-04", "then"),
                    ("other", "other", "primary", "done", "2026-01-05", "then"),
                ],
            )
        for expected in ("b", "superseded", "a"):
            assert db.latest_primary_run_metadata("c")["id"] == expected
            assert db.latest_primary_run("c")["id"] == expected
            assert [
                row["id"]
                for row in db.list_runs(
                    "c", include_superseded=False, agent_kind="primary", limit=1
                )
            ] == [expected]
            with db.connection() as conn:
                if expected == "b":
                    conn.execute(
                        "UPDATE agent_runs SET status='done' WHERE id='superseded'"
                    )
                elif expected == "superseded":
                    conn.execute(
                        "UPDATE agent_runs SET agent_kind='subagent' WHERE id='superseded'"
                    )
                    conn.execute(
                        "UPDATE agent_runs SET status='superseded' WHERE id='b'"
                    )
        assert db.latest_primary_run_metadata("missing") is None
        assert len(db.list_runs("c")) == 4
        assert [
            row["id"]
            for row in db.list_runs("c", False, limit=1, agent_kind="subagent")
        ] == ["superseded"]
        assert [
            row["id"] for row in db.list_runs("c", False, limit=1, agent_kind="primary")
        ] == ["a"]


def test_original_newest_queries_avoid_history_sort_and_linear_work(tmp_path):
    with closing(AgentRunsDB(tmp_path / "runs.db")) as db:
        with db.connection() as conn:
            conn.executemany(
                "INSERT INTO agent_runs(id,conversation_id,agent_kind,status,created_at,updated_at) "
                "VALUES (?,'c','primary','done',?,?)",
                [(f"run-{n:05}", f"{n:05}", f"{n:05}") for n in range(2000)],
            )
            queries = []
            conn.set_trace_callback(queries.append)
            try:
                assert db.latest_primary_run_metadata("c")["id"] == "run-01999"
                assert db.latest_primary_run("c")["id"] == "run-01999"
                assert (
                    db.list_runs("c", False, limit=1, agent_kind="primary")[0]["id"]
                    == "run-01999"
                )
            finally:
                conn.set_trace_callback(None)
            newest = [
                sql
                for sql in queries
                if "FROM agent_runs WHERE" in sql and "LIMIT 1" in sql
            ]
            assert len(newest) == 3
            for sql in newest:
                plan = [row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + sql)]
                assert any(INDEX in row for row in plan), plan
                assert not any("TEMP B-TREE" in row for row in plan), plan
                work = 0

                def progress():
                    nonlocal work
                    work += 1
                    return 0

                conn.set_progress_handler(progress, 1)
                try:
                    assert conn.execute(sql).fetchone()["id"] == "run-01999"
                finally:
                    conn.set_progress_handler(None, 0)
                assert work < 200, work


@pytest.mark.parametrize(
    "damage", ["missing", "wrong_predicate", "wrong_order", "wrong_stamp"]
)
def test_newest_index_recovery_requires_exact_catalog(tmp_path, damage):
    path = tmp_path / "runs.db"
    AgentRunsDB(path).close()
    with closing(sqlite3.connect(path)) as conn:
        if damage == "wrong_stamp":
            conn.execute("DELETE FROM schema_version WHERE version=32")
        else:
            conn.execute(f"DROP INDEX {INDEX}")
            if damage == "wrong_predicate":
                conn.execute(
                    f"CREATE INDEX {INDEX} ON agent_runs(conversation_id,created_at DESC,id DESC)"
                )
            elif damage == "wrong_order":
                conn.execute(
                    f"CREATE INDEX {INDEX} ON agent_runs(conversation_id,created_at,id) "
                    "WHERE agent_kind='primary' AND status != 'superseded'"
                )
        conn.commit()
    before = path.read_bytes()
    assert validate_candidate(_owner(), path, Event(), migrate=True)
    assert path.read_bytes() == before


def test_index_migration_cancellation_rolls_back(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import sqlite_validation as validation

    path = tmp_path / "runs.db"
    _v31(path)
    before = path.read_bytes()
    cancel = Event()
    native = validation._Restrictions.authorize

    def interrupted(self, action, first, second, database, source):
        if self.migrating and action == sqlite3.SQLITE_CREATE_INDEX and first == INDEX:
            cancel.set()
        return native(self, action, first, second, database, source)

    monkeypatch.setattr(validation._Restrictions, "authorize", interrupted)
    assert validate_candidate(_owner(), path, cancel, migrate=True) == ("cancelled",)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "case", ["outside_window", "wrong_owner", "wrong_table", "wrong_name"]
)
def test_index_migration_authority_is_exact(tmp_path, case):
    from tldw_chatbook.DB.private_sqlite import open_recovery_validation

    path = tmp_path / "runs.db"
    _v31(path)
    with open_recovery_validation(
        "db.agent_runs", path, writable=True, with_restrictions=True
    ) as (conn, restrictions):
        restrictions.migrating = case != "outside_window"
        restrictions.migration_owner = (
            "research.local" if case == "wrong_owner" else "db.agent_runs"
        )
        name = "unapproved_index" if case == "wrong_name" else INDEX
        table = "agent_definitions" if case == "wrong_table" else "agent_runs"
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute(f"CREATE INDEX {name} ON {table}(id)")
