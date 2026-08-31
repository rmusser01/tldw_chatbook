"""ChaChaNotes v61 Agent Lessons seed-state migration coverage."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, SchemaError
from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version


SEED_TABLE = "agent_lessons_seed_state"
V59_TO_V60_SHA256 = "1b6011914bed2c3dc5928806d29eab6ce5f4698c101b1719575f3044a5766dc2"
FINGERPRINT = "a" * 64
EXPECTED_COLUMNS = (
    "profile_id",
    "dataset_id",
    "scope_mode",
    "state",
    "folder_sync_id",
    "seed_fingerprint",
)


def _schema_version(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (CharactersRAGDB._SCHEMA_NAME,),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


def _seed_schema(connection: sqlite3.Connection) -> tuple[object, ...]:
    table_sql = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (SEED_TABLE,),
    ).fetchone()
    assert table_sql is not None
    return (
        tuple(tuple(row) for row in connection.execute(f"PRAGMA table_info({SEED_TABLE})")),
        str(table_sql[0]),
    )


def _organization_counts(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in ("notes", "note_folders", "keywords")
    }


def _seed_real_v60(path: Path) -> dict[str, int]:
    with chachanotes_db_at_version(path, 60, client_id="agent-lessons-v60-seed") as db:
        connection = db.get_connection()
        assert _schema_version(connection) == 60
        assert SEED_TABLE not in _table_names(connection)
        return _organization_counts(connection)


def test_v59_to_v60_publication_migration_remains_byte_identical() -> None:
    migration = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "DB"
        / "migrations"
        / "chachanotes_v59_to_v60_note_sync_publication_intents.sql"
    )
    assert hashlib.sha256(migration.read_bytes()).hexdigest() == V59_TO_V60_SHA256


def test_real_v60_reopen_adds_empty_content_free_seed_state(tmp_path: Path) -> None:
    path = tmp_path / "agent-lessons-v60.sqlite"
    before_counts = _seed_real_v60(path)

    migrated = CharactersRAGDB(path, client_id="agent-lessons-v61-migrate")
    try:
        connection = migrated.get_connection()
        assert CharactersRAGDB._CURRENT_SCHEMA_VERSION == 61
        assert _schema_version(connection) == 61
        assert SEED_TABLE in _table_names(connection)
        assert tuple(
            str(row[1]) for row in connection.execute(f"PRAGMA table_info({SEED_TABLE})")
        ) == EXPECTED_COLUMNS
        assert connection.execute(f"SELECT COUNT(*) FROM {SEED_TABLE}").fetchone()[0] == 0
        assert _organization_counts(connection) == before_counts
        for name in ("Agent_Lessons", "agent-lesson"):
            assert connection.execute(
                "SELECT COUNT(*) FROM note_folders WHERE name = ?", (name,)
            ).fetchone()[0] == 0
            assert connection.execute(
                "SELECT COUNT(*) FROM keywords WHERE keyword = ?", (name,)
            ).fetchone()[0] == 0

        connection.execute(
            f"INSERT INTO {SEED_TABLE}("
            "profile_id, dataset_id, scope_mode, state, folder_sync_id, seed_fingerprint"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            ("profile-a", "dataset-a", "synchronized", "unknown", None, FINGERPRINT),
        )
        connection.execute(
            f"UPDATE {SEED_TABLE} SET state = 'not_seeded' "
            "WHERE profile_id = 'profile-a' AND dataset_id = 'dataset-a'"
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                f"UPDATE {SEED_TABLE} SET state = 'unknown' "
                "WHERE profile_id = 'profile-a' AND dataset_id = 'dataset-a'"
            )
        connection.execute(
            f"UPDATE {SEED_TABLE} SET state = 'seeded', folder_sync_id = 'folder-a' "
            "WHERE profile_id = 'profile-a' AND dataset_id = 'dataset-a'"
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                f"UPDATE {SEED_TABLE} SET state = 'not_seeded' "
                "WHERE profile_id = 'profile-a' AND dataset_id = 'dataset-a'"
            )
        connection.execute(
            f"INSERT INTO {SEED_TABLE}("
            "profile_id, dataset_id, scope_mode, state, folder_sync_id, seed_fingerprint"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            ("profile-a", "local", "local_only", "seeded", None, FINGERPRINT),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                f"UPDATE {SEED_TABLE} SET state = 'unknown' "
                "WHERE profile_id = 'profile-a' AND dataset_id = 'local'"
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                f"INSERT INTO {SEED_TABLE}("
                "profile_id, dataset_id, scope_mode, state, folder_sync_id, seed_fingerprint"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                ("profile-a", "dataset-a", "synchronized", "seeded", None, FINGERPRINT),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                f"INSERT INTO {SEED_TABLE}("
                "profile_id, dataset_id, scope_mode, state, folder_sync_id, seed_fingerprint"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                ("profile-b", "dataset-b", "remote", "unknown", None, FINGERPRINT),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                f"INSERT INTO {SEED_TABLE}("
                "profile_id, dataset_id, scope_mode, state, folder_sync_id, seed_fingerprint"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                ("profile-b", "dataset-b", "synchronized", "reset", None, FINGERPRINT),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                f"INSERT INTO {SEED_TABLE}("
                "profile_id, dataset_id, scope_mode, state, folder_sync_id, seed_fingerprint"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                ("profile-b", "dataset-b", "synchronized", "unknown", None, "NOT-A-DIGEST"),
            )
        connection.commit()
    finally:
        migrated.close_connection()

    reopened = CharactersRAGDB(path, client_id="agent-lessons-v61-reopen")
    try:
        connection = reopened.get_connection()
        assert _schema_version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        assert SEED_TABLE in _table_names(connection)
        rows = connection.execute(
            f"SELECT profile_id, dataset_id, scope_mode, state, folder_sync_id "
            f"FROM {SEED_TABLE} ORDER BY profile_id, dataset_id"
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            ("profile-a", "dataset-a", "synchronized", "seeded", "folder-a"),
            ("profile-a", "local", "local_only", "seeded", None),
        ]
        assert _organization_counts(connection) == before_counts
    finally:
        reopened.close_connection()


def test_v61_migration_failure_rolls_back_seed_state_and_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "agent-lessons-v61-rollback.sqlite"
    before_counts = _seed_real_v60(path)
    real_execute = CharactersRAGDB._execute_migration_statements

    def fail_after_seed_ddl(
        self: CharactersRAGDB,
        cursor: sqlite3.Cursor,
        script: str,
        label: str,
    ) -> None:
        real_execute(self, cursor, script, label)
        if label == "V60→V61":
            raise RuntimeError("injected after Agent Lessons seed DDL")

    monkeypatch.setattr(
        CharactersRAGDB,
        "_execute_migration_statements",
        fail_after_seed_ddl,
    )
    with pytest.raises(SchemaError, match=r"V60.*V61"):
        CharactersRAGDB(path, client_id="agent-lessons-v61-failure")

    with sqlite3.connect(path) as connection:
        assert _schema_version(connection) == 60
        assert SEED_TABLE not in _table_names(connection)
        assert _organization_counts(connection) == before_counts


def test_seed_fingerprint_is_opaque_and_service_category_owned() -> None:
    categories = (
        "coordinator_created",
        "exact_root_reuse",
        "remote_history_upsert",
    )
    fingerprints = {
        category: hashlib.sha256(
            f"agent-lessons-seed:v1:{category}".encode("ascii")
        ).hexdigest()
        for category in categories
    }
    assert len(set(fingerprints.values())) == len(categories)

    db = CharactersRAGDB(":memory:", client_id="agent-lessons-category-digests")
    try:
        connection = db.get_connection()
        for category in categories:
            connection.execute(
                f"INSERT INTO {SEED_TABLE}("
                "profile_id, dataset_id, scope_mode, state, folder_sync_id, "
                "seed_fingerprint) VALUES (?, ?, 'local_only', 'seeded', NULL, ?)",
                (f"profile-{category}", "local", fingerprints[category]),
            )
        stored = {
            str(row[0])
            for row in connection.execute(
                f"SELECT seed_fingerprint FROM {SEED_TABLE}"
            )
        }
        columns = {
            str(row[1])
            for row in connection.execute(f"PRAGMA table_info({SEED_TABLE})")
        }
        assert stored == set(fingerprints.values())
        assert not ({"evidence_category", "source_category", "content"} & columns)
    finally:
        db.close_connection()


def test_fresh_v61_seed_schema_matches_real_v60_migration(tmp_path: Path) -> None:
    path = tmp_path / "agent-lessons-v61-parity.sqlite"
    _seed_real_v60(path)
    migrated = CharactersRAGDB(path, client_id="agent-lessons-v61-parity-migrated")
    fresh = CharactersRAGDB(":memory:", client_id="agent-lessons-v61-parity-fresh")
    try:
        assert _seed_schema(migrated.get_connection()) == _seed_schema(
            fresh.get_connection()
        )
        assert _organization_counts(migrated.get_connection()) == {
            "notes": 0,
            "note_folders": 0,
            "keywords": 0,
        }
        assert _organization_counts(fresh.get_connection()) == {
            "notes": 0,
            "note_folders": 0,
            "keywords": 0,
        }
    finally:
        migrated.close_connection()
        fresh.close_connection()
