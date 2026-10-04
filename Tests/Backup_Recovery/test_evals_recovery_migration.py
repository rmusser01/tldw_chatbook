"""A v5 Evals backup restores through the declared v5 -> v6 step (TASK-19566 F8).

``Evals/recovery.py`` declares the step so an older Evals backup stays
restorable. The step is five ``ALTER TABLE ... DROP COLUMN version``
statements, and the restore sandbox's authorizer refused them: DROP COLUMN
reports the dropped column in the authorizer's database slot and rewrites
the temp schema, both of which fail the sandbox's main-database-only rule.
A genuine v5 database validated as-is but came back
``sqlite_validation_unavailable`` with ``migrate=True`` (PR #3002 review).
"""

import sqlite3
from contextlib import closing
from pathlib import Path
from threading import Event

import pytest

from tldw_chatbook.Backup_Recovery.sqlite_validation import (
    _EVALS_VERSION_COLUMN_TABLES,
    _Restrictions,
    validate_candidate,
)
from tldw_chatbook.Evals.recovery import _SCHEMA, recovery_adapters

_SHADOW_SUFFIXES = ("_fts_config'", "_fts_data'", "_fts_docsize'", "_fts_idx'")


def _owner():
    return next(row for row in recovery_adapters() if row.owner_id == "db.evals")


def _build_declared_v5(path: Path) -> Path:
    """Create a v5 Evals database from the catalog the recovery policy declares."""
    catalog = next(sql for version, sql in _SCHEMA if version == 5)

    def is_shadow(sql: str) -> bool:
        # FTS5 creates its own shadow tables with the virtual table.
        return sql.startswith("CREATE TABLE '") and sql.split("(", 1)[
            0
        ].rstrip().endswith(_SHADOW_SUFFIXES)

    ordered = sorted(
        (sql for sql in catalog if not is_shadow(sql)),
        key=lambda sql: (
            not sql.startswith("CREATE TABLE"),
            not sql.startswith("CREATE VIRTUAL TABLE"),
            not sql.startswith("CREATE INDEX"),
        ),
    )
    with closing(sqlite3.connect(path)) as connection:
        for sql in ordered:
            connection.execute(sql)
        connection.execute(
            "INSERT INTO eval_tasks "
            "(id, name, task_type, config_format, config_data, client_id) "
            "VALUES ('kept-task', 'Saved task', 'question_answer', 'custom', '{}', 'fixture')"
        )
        connection.execute("PRAGMA user_version = 5")
        connection.commit()
    path.chmod(0o600)
    return path


def _columns(path: Path, table: str) -> set[str]:
    with closing(sqlite3.connect(path)) as connection:
        return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def test_declared_v5_fixture_is_a_valid_v5_candidate(tmp_path):
    path = _build_declared_v5(tmp_path / "evals.db")
    before = path.read_bytes()

    assert validate_candidate(_owner(), path, Event(), migrate=False) == ()
    assert path.read_bytes() == before
    assert all("version" in _columns(path, t) for t in _EVALS_VERSION_COLUMN_TABLES)


def test_v5_evals_candidate_migrates_to_v6_and_keeps_its_rows(tmp_path):
    path = _build_declared_v5(tmp_path / "evals.db")

    assert validate_candidate(_owner(), path, Event(), migrate=True) == ()

    assert not any("version" in _columns(path, t) for t in _EVALS_VERSION_COLUMN_TABLES)
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (6,)
        assert connection.execute(
            "SELECT name FROM eval_tasks WHERE id = 'kept-task'"
        ).fetchone() == ("Saved task",)


@pytest.mark.parametrize(
    "migrating, owner, call, allowed",
    [
        (
            True,
            "db.evals",
            (sqlite3.SQLITE_ALTER_TABLE, "main", "eval_tasks", "version"),
            True,
        ),
        (
            True,
            "db.evals",
            (sqlite3.SQLITE_UPDATE, "sqlite_temp_master", "sql", "temp"),
            True,
        ),
        (
            True,
            "db.evals",
            (sqlite3.SQLITE_READ, "sqlite_temp_master", "sql", "temp"),
            True,
        ),
        # Another column, another table, another schema table, another database.
        (
            True,
            "db.evals",
            (sqlite3.SQLITE_ALTER_TABLE, "main", "eval_tasks", "name"),
            False,
        ),
        (
            True,
            "db.evals",
            (sqlite3.SQLITE_ALTER_TABLE, "main", "eval_results", "version"),
            False,
        ),
        (
            True,
            "db.evals",
            (sqlite3.SQLITE_UPDATE, "sqlite_temp_master", "name", "temp"),
            False,
        ),
        (
            True,
            "db.evals",
            (sqlite3.SQLITE_UPDATE, "eval_tasks", "name", "temp"),
            False,
        ),
        (
            True,
            "db.evals",
            (sqlite3.SQLITE_DELETE, "sqlite_temp_master", None, "temp"),
            False,
        ),
        # The gate must be open, and open for this owner.
        (
            False,
            "db.evals",
            (sqlite3.SQLITE_ALTER_TABLE, "main", "eval_tasks", "version"),
            False,
        ),
        (
            True,
            "db.agent_runs",
            (sqlite3.SQLITE_ALTER_TABLE, "main", "eval_tasks", "version"),
            False,
        ),
    ],
)
def test_evals_drop_column_authority_is_exact(migrating, owner, call, allowed):
    restrictions = object.__new__(_Restrictions)
    restrictions.migrating = migrating
    restrictions.migration_owner = owner

    assert restrictions._evals_drop_column_step(*call) is allowed
