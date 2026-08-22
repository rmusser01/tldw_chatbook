"""ChaChaNotes v44 -> v45 portable Actor Pack persistence coverage.

This module carried the repo's EXACT current-schema-version pin while v45 was
the newest migration. TASK-19564's `sync_log` retention step was renumbered to
v45 -> v46 after landing concurrently, so the pin moved on to
`Tests/DB/test_chachanotes_sync_log_retention_migration.py` and the assertions
here relaxed to `>= 45`. That is TASK-19554's convention, repaired by
TASK-19568: the exact pin belongs to the NEWEST migration's own file, and an
older file that keeps a literal `==` reds on version arithmetic and
short-circuits the real column/row assertions below it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, SchemaError


TABLES = {"actor_portable_identities", "actor_pack_persona_intents"}

EXPECTED_COLUMNS = {
    "actor_portable_identities": (
        "actor_kind",
        "local_actor_id",
        "portable_uuid",
        "source_portable_uuid",
        "created_at",
        "updated_at",
        "version",
    ),
    "actor_pack_persona_intents": (
        "intent_id",
        "persona_id",
        "operation",
        "state",
        "old_profile_json",
        "new_profile_json",
        "old_profile_sha256",
        "new_profile_sha256",
        "old_store_sha256",
        "new_store_sha256",
        "old_registry_uuid",
        "new_registry_uuid",
        "quarantine_reason",
        "created_at",
        "updated_at",
    ),
}


def _version(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (CharactersRAGDB._SCHEMA_NAME,),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


def test_real_v44_upgrade_installs_actor_pack_tables(tmp_path: Path) -> None:
    path = tmp_path / "actor-pack-v44.db"
    with chachanotes_db_at_version(path, 44, client_id="actor-pack-v44") as old:
        connection = old.get_connection()
        assert _version(connection) == 44
        assert not (_tables(connection) & TABLES)

    migrated = CharactersRAGDB(path, client_id="actor-pack-v45")
    try:
        connection = migrated.get_connection()
        # Dynamic, not a literal 45: this reopen is unpatched, so it replays
        # the chain to whatever the current version is (v46 since TASK-19564).
        assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        assert CharactersRAGDB._CURRENT_SCHEMA_VERSION >= 45
        assert TABLES <= _tables(connection)
        for table, expected in EXPECTED_COLUMNS.items():
            columns = tuple(
                str(row[1])
                for row in connection.execute(f"PRAGMA table_info('{table}')")
            )
            assert columns == expected
    finally:
        migrated.close_connection()


def test_fresh_database_contains_same_v45_schema(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "fresh.db", client_id="actor-pack-fresh")
    try:
        assert _version(db.get_connection()) >= 45
        assert TABLES <= _tables(db.get_connection())
    finally:
        db.close_connection()


def test_registry_enforces_cross_kind_uuid_uniqueness_and_copy_provenance(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "registry.db", client_id="actor-pack-registry")
    connection = db.get_connection()
    try:
        connection.execute(
            """
            INSERT INTO actor_portable_identities(
                actor_kind, local_actor_id, portable_uuid
            ) VALUES ('character', '7', ?)
            """,
            ("123e4567-e89b-42d3-a456-426614174000",),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO actor_portable_identities(
                    actor_kind, local_actor_id, portable_uuid
                ) VALUES ('persona', 'guide', ?)
                """,
                ("123e4567-e89b-42d3-a456-426614174000",),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO actor_portable_identities(
                    actor_kind, local_actor_id, portable_uuid, source_portable_uuid
                ) VALUES ('persona', 'copy', ?, ?)
                """,
                (
                    "223e4567-e89b-42d3-a456-426614174000",
                    "223e4567-e89b-42d3-a456-426614174000",
                ),
            )
    finally:
        connection.rollback()
        db.close_connection()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("actor_kind", "server"),
        ("portable_uuid", "123E4567-E89B-42D3-A456-426614174000"),
        ("portable_uuid", "not-a-uuid"),
        ("portable_uuid", "123e4567-e89b-42d3-a456-42661417-000"),
        ("source_portable_uuid", "not-a-uuid"),
    ],
)
def test_registry_rejects_noncanonical_storage_values(
    column: str, value: str, tmp_path: Path
) -> None:
    db = CharactersRAGDB(tmp_path / f"bad-{column}.db", client_id="actor-pack-bad")
    connection = db.get_connection()
    fields = {
        "actor_kind": "character",
        "local_actor_id": "7",
        "portable_uuid": "123e4567-e89b-42d3-a456-426614174000",
        "source_portable_uuid": None,
    }
    fields[column] = value
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO actor_portable_identities(
                    actor_kind, local_actor_id, portable_uuid, source_portable_uuid
                ) VALUES (?, ?, ?, ?)
                """,
                tuple(fields.values()),
            )
    finally:
        connection.rollback()
        db.close_connection()


def test_intent_state_and_digest_shape_are_enforced_by_storage(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "intent.db", client_id="actor-pack-intent")
    connection = db.get_connection()
    valid = (
        "a" * 32,
        "guide",
        "create",
        "prepared",
        None,
        '{"id":"guide"}',
        None,
        "a" * 64,
        "b" * 64,
        "c" * 64,
        None,
        "123e4567-e89b-42d3-a456-426614174000",
        None,
    )
    statement = """
        INSERT INTO actor_pack_persona_intents(
            intent_id, persona_id, operation, state,
            old_profile_json, new_profile_json,
            old_profile_sha256, new_profile_sha256,
            old_store_sha256, new_store_sha256,
            old_registry_uuid, new_registry_uuid, quarantine_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    try:
        connection.execute(statement, valid)
        for index, bad in ((2, "delete"), (3, "unknown"), (7, "not-a-digest")):
            row = list(valid)
            row[0] = f"{index}" * 32
            row[index] = bad
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(statement, tuple(row))
    finally:
        connection.rollback()
        db.close_connection()


def test_v44_to_v45_step_refuses_wrong_entry_version(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "current.db", client_id="actor-pack-current")
    try:
        with pytest.raises(SchemaError, match="requires schema version"):
            db._migrate_from_v44_to_v45(db.get_connection())
    finally:
        db.close_connection()
