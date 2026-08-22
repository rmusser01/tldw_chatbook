"""ChaChaNotes v43 -> v44: recoverable copy of a discarded sync side.

task-19554. The Notes sync engine resolved a ``both_changed`` conflict by
overwriting one side wholesale while ``sync_conflicts`` kept only a SHA-256 of
it, so the discarded text was unrecoverable. This migration adds the three
columns (``losing_side``, ``losing_content``, ``preserved_file_path``) that
make the row a real second copy behind the on-disk sidecar.

This module also carries the repo's EXACT current-schema-version pin, which
moved here from ``test_chachanotes_message_exchanges.py`` when v44 landed:
the pin belongs to the newest migration's own file, so a schema bump touches
the file that caused it rather than an unrelated older one (older files assert
``>= their own version`` instead). Updating the number here is a deliberate
schema-review act.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, SchemaError

SCHEMA_NAME = CharactersRAGDB._SCHEMA_NAME
NEW_COLUMNS = ("losing_side", "losing_content", "preserved_file_path")


def _version(connection: sqlite3.Connection) -> int:
    return connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (SCHEMA_NAME,),
    ).fetchone()[0]


def _columns(connection: sqlite3.Connection) -> dict[str, tuple]:
    return {
        row[1]: tuple(row)
        for row in connection.execute("PRAGMA table_info(sync_conflicts)")
    }


@pytest.fixture
def db(tmp_path: Path):
    instance = CharactersRAGDB(tmp_path / "chachanotes.db", client_id="v44-test")
    yield instance
    instance.close_connection()


def test_schema_version_is_44(db):
    """The one exact current-version pin (see this module's docstring)."""
    assert _version(db.get_connection()) == 44
    assert CharactersRAGDB._CURRENT_SCHEMA_VERSION == 44


def test_fresh_schema_has_the_preservation_columns(db):
    columns = _columns(db.get_connection())
    for name in NEW_COLUMNS:
        assert name in columns, f"{name} missing from sync_conflicts"
        assert columns[name][2].upper() == "TEXT", columns[name]
        assert columns[name][3] == 0, "the columns must be nullable"
        assert columns[name][4] is None, "no default -- absence must be NULL"


def test_migrate_from_v43_to_v44_requires_version_43(db):
    """A fresh DB is already at 44, so re-entering the step must refuse."""
    with pytest.raises(SchemaError, match="requires schema version"):
        db._migrate_from_v43_to_v44(db.get_connection())


def test_upgrade_from_a_real_v43_database_adds_the_columns(tmp_path: Path):
    """A genuine v43 database, with rows in it, migrates and keeps them."""
    db_path = tmp_path / "chachanotes.db"
    with chachanotes_db_at_version(db_path, 43) as historical:
        connection = historical.get_connection()
        assert _version(connection) == 43
        for name in NEW_COLUMNS:
            assert name not in _columns(connection), (
                f"{name} must NOT exist before the migration, or this test "
                f"proves nothing"
            )
        with historical.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO sync_sessions(
                    session_id, sync_root_folder, sync_direction,
                    conflict_resolution, status, client_id
                ) VALUES ('s-1', '/tmp/root', 'bidirectional', 'newer_wins',
                          'completed', 'v43-client')
                """
            )
            cursor.execute(
                """
                INSERT INTO sync_conflicts(
                    session_id, file_path, conflict_type,
                    db_content_hash, disk_content_hash
                ) VALUES ('s-1', 'note.md', 'both_changed', 'aaa', 'bbb')
                """
            )

    migrated = CharactersRAGDB(db_path, client_id="v44-upgrade")
    try:
        connection = migrated.get_connection()
        # Dynamic, not a literal 44: this reopen is UNPATCHED, so it replays
        # the chain to whatever the current version is. Pinning the literal
        # here made a schema bump red this test on version arithmetic and
        # short-circuit the column/row assertions below it -- the exact shape
        # task-19568 removed from the persona-visual migration test. The one
        # deliberate exact pin stays in ``test_schema_version_is_44``.
        assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        columns = _columns(connection)
        for name in NEW_COLUMNS:
            assert name in columns
        row = connection.execute(
            "SELECT db_content_hash, disk_content_hash, losing_side, "
            "losing_content, preserved_file_path FROM sync_conflicts"
        ).fetchone()
        assert tuple(row) == ("aaa", "bbb", None, None, None), (
            "the pre-existing row must survive with NULL preservation fields"
        )
    finally:
        migrated.close_connection()


def test_upgrade_is_re_enterable_after_a_half_applied_run(tmp_path: Path):
    """The task-19553 brick shape: one ADD COLUMN applied, stamp still 43.

    ``_execute_migration_statements`` skips an already-satisfied ADD COLUMN,
    so this must land on 44 rather than dying on ``duplicate column name``
    every launch forever.
    """
    db_path = tmp_path / "half_applied.db"
    with chachanotes_db_at_version(db_path, 43):
        pass

    connection = sqlite3.connect(str(db_path))
    try:
        connection.execute("ALTER TABLE sync_conflicts ADD COLUMN losing_side TEXT")
        connection.commit()
        assert _version(connection) == 43, "a half-applied step never bumps"
    finally:
        connection.close()

    migrated = CharactersRAGDB(db_path, client_id="v44-reentry")
    try:
        # Dynamic for the same reason as the sibling test above: the reopen
        # is unpatched, so the literal would red on the next schema bump and
        # skip the NEW_COLUMNS check below it.
        assert (
            _version(migrated.get_connection())
            == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        )
        columns = _columns(migrated.get_connection())
        for name in NEW_COLUMNS:
            assert name in columns
    finally:
        migrated.close_connection()
