"""ChaChaNotes v57 -> v58 portable Notes organization migration coverage."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import uuid

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, SchemaError
from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version


RESOURCE_TABLES = ("keywords", "keyword_collections", "note_folders")
STATE_TABLES = {
    "notes_organization_sync_intents",
    "notes_organization_heads",
    "notes_organization_sync_checkpoints",
    "notes_organization_adoption_reviews",
    "note_folder_sync_suppressions",
}
EXPECTED_INDEXES = {
    "uq_keywords_sync_id",
    "uq_keyword_collections_sync_id",
    "uq_note_folders_sync_id",
    "idx_notes_organization_intents_pending",
    "idx_notes_organization_heads_cursor",
    "idx_notes_organization_adoption_reviews_open",
}


def _schema_version(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (CharactersRAGDB._SCHEMA_NAME,),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _intent_columns(connection: sqlite3.Connection) -> list[tuple[object, ...]]:
    return [
        tuple(row)
        for row in connection.execute(
            "PRAGMA table_info(notes_organization_sync_intents)"
        ).fetchall()
    ]


def _is_canonical_uuid4(value: str) -> bool:
    parsed = uuid.UUID(value)
    return parsed.version == 4 and str(parsed) == value and value == value.lower()


def _seed_real_v57(
    path: Path,
) -> tuple[
    dict[str, list[tuple[object, str]]],
    list[tuple[str, str, str, str, str, int, int, int]],
]:
    seeded: dict[str, list[tuple[object, str]]] = {}
    with chachanotes_db_at_version(path, 57, client_id="v57-seed") as db:
        connection = db.get_connection()
        connection.execute(
            "INSERT INTO keywords(keyword, deleted) VALUES ('Active keyword', 0)"
        )
        connection.execute(
            "INSERT INTO keywords(keyword, deleted) VALUES ('Deleted keyword', 1)"
        )
        connection.execute(
            "INSERT INTO keyword_collections(name, deleted) VALUES ('Active collection', 0)"
        )
        connection.execute(
            "INSERT INTO keyword_collections(name, deleted) VALUES ('Deleted collection', 1)"
        )
        connection.execute(
            """
            INSERT INTO note_folders(
                id, parent_id, name, normalized_name, path, normalized_path,
                version, deleted, created_at, modified_at
            ) VALUES
                ('folder-active', NULL, 'Active', 'active', '/Active', '/active',
                 3, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
                ('folder-deleted', NULL, 'Deleted', 'deleted', '/Deleted', '/deleted',
                 7, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """
        )
        note_id = str(db.add_note("Membership target", "Body"))
        connection.execute(
            """
            INSERT INTO note_folder_memberships(
                id, folder_id, note_id, ownership, owner_id, owner_active,
                version, deleted, created_at, modified_at
            ) VALUES
                ('membership-active', 'folder-active', ?, 'manual', '', 1,
                 2, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
                ('membership-deleted', 'folder-deleted', ?, 'managed',
                 'legacy-owner', 0, 5, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (note_id, note_id),
        )
        connection.commit()
        for table in RESOURCE_TABLES:
            label_column = {
                "keywords": "keyword",
                "keyword_collections": "name",
                "note_folders": "name",
            }[table]
            seeded[table] = [
                (row[0], str(row[1]))
                for row in connection.execute(
                    f"SELECT id, {label_column} FROM {table} ORDER BY id"
                )
            ]
        assert _schema_version(connection) == 57
        assert all(
            "sync_id"
            not in {
                str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
            }
            for table in RESOURCE_TABLES
        )
        memberships = [
            tuple(row)
            for row in connection.execute(
                """
                SELECT id, folder_id, note_id, ownership, owner_id, owner_active,
                       version, deleted
                  FROM note_folder_memberships
                 ORDER BY id
                """
            )
        ]
    return seeded, memberships


def _resource_rows(
    connection: sqlite3.Connection,
) -> dict[str, list[tuple[object, str, str]]]:
    rows: dict[str, list[tuple[object, str, str]]] = {}
    for table in RESOURCE_TABLES:
        label_column = {
            "keywords": "keyword",
            "keyword_collections": "name",
            "note_folders": "name",
        }[table]
        rows[table] = [
            (row[0], str(row[1]), str(row[2]))
            for row in connection.execute(
                f"SELECT id, {label_column}, sync_id FROM {table} ORDER BY id"
            )
        ]
    return rows


def test_real_v57_reopen_backfills_stable_unique_uuid4_ids(tmp_path: Path) -> None:
    path = tmp_path / "v57.db"
    original, original_memberships = _seed_real_v57(path)

    with chachanotes_db_at_version(path, 58, client_id="v58-open") as migrated:
        connection = migrated.get_connection()
        first = _resource_rows(connection)
        assert _schema_version(connection) == 58
        assert CharactersRAGDB._CURRENT_SCHEMA_VERSION == 58
        assert {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        } >= STATE_TABLES
        assert {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        } >= EXPECTED_INDEXES
        routing_column = next(
            row
            for row in _intent_columns(connection)
            if row[1] == "routing_metadata_json"
        )
        assert routing_column[2:5] == ("TEXT", 1, "'{}'")
        sequence_column = next(
            row for row in _intent_columns(connection) if row[1] == "intent_sequence"
        )
        assert sequence_column[2:5] == ("INTEGER", 1, None)
        predecessor_column = next(
            row
            for row in _intent_columns(connection)
            if row[1] == "predecessor_intent_id"
        )
        assert predecessor_column[2:5] == ("TEXT", 0, None)
        base_cursor_column = next(
            row for row in _intent_columns(connection) if row[1] == "base_server_cursor"
        )
        assert base_cursor_column[2:5] == ("TEXT", 0, None)

        assigned_ids = [row[2] for table_rows in first.values() for row in table_rows]
        assert len(assigned_ids) == len(set(assigned_ids))
        assert all(_is_canonical_uuid4(value) for value in assigned_ids)
        assert {
            table: [(row[0], row[1]) for row in table_rows]
            for table, table_rows in first.items()
        } == original
        first_memberships = [
            tuple(row)
            for row in connection.execute(
                """
                SELECT id, folder_id, note_id, ownership, owner_id, owner_active,
                       version, deleted
                  FROM note_folder_memberships
                 ORDER BY id
                """
            )
        ]
        assert first_memberships == original_memberships
    with chachanotes_db_at_version(path, 58, client_id="v58-reopen") as reopened:
        connection = reopened.get_connection()
        assert _resource_rows(connection) == first
        assert [
            tuple(row)
            for row in connection.execute(
                """
                SELECT id, folder_id, note_id, ownership, owner_id, owner_active,
                       version, deleted
                  FROM note_folder_memberships
                 ORDER BY id
                """
            )
        ] == original_memberships


def test_fresh_v58_schema_matches_migration_constraints() -> None:
    with chachanotes_db_at_version(
        ":memory:", 58, client_id="fresh-v58"
    ) as db:
        connection = db.get_connection()
        assert _schema_version(connection) == 58
        for table in RESOURCE_TABLES:
            assert "sync_id" in {
                str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
            }
        routing_column = next(
            row
            for row in _intent_columns(connection)
            if row[1] == "routing_metadata_json"
        )
        assert routing_column[2:5] == ("TEXT", 1, "'{}'")
        sequence_column = next(
            row for row in _intent_columns(connection) if row[1] == "intent_sequence"
        )
        assert sequence_column[2:5] == ("INTEGER", 1, None)
        predecessor_column = next(
            row
            for row in _intent_columns(connection)
            if row[1] == "predecessor_intent_id"
        )
        assert predecessor_column[2:5] == ("TEXT", 0, None)
        base_cursor_column = next(
            row for row in _intent_columns(connection) if row[1] == "base_server_cursor"
        )
        assert base_cursor_column[2:5] == ("TEXT", 0, None)

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO notes_organization_sync_intents(
                    intent_id, intent_sequence, server_profile_id, dataset_id, domain, object_id,
                    operation, schema_version, encryption_policy, payload_json,
                    payload_hash, source_version, created_at,
                    outbox_client_envelope_id
                ) VALUES (
                    'intent-a', 1, 'profile', 'dataset', 'notes.keyword', 'object',
                    'upsert', 1, 'server_trusted_v1', '{}', ?, 1, 'now',
                    'different-envelope'
                )
                """,
                ("a" * 64,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO notes_organization_sync_intents(
                    intent_id, intent_sequence, server_profile_id, dataset_id, domain, object_id,
                    operation, schema_version, encryption_policy, payload_json,
                    payload_hash, base_server_cursor, source_version, created_at
                ) VALUES (
                    'intent-incomplete-base', 1, 'profile', 'dataset',
                    'notes.keyword', 'object', 'upsert', 1,
                    'server_trusted_v1', '{}', ?, '17', 1, 'now'
                )
                """,
                ("b" * 64,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO notes_organization_sync_intents(
                    intent_id, intent_sequence, server_profile_id, dataset_id,
                    domain, object_id, operation, schema_version,
                    encryption_policy, payload_json, payload_hash,
                    base_object_revision, base_object_hash, source_version,
                    created_at
                ) VALUES (
                    'intent-base-without-cursor', 2, 'profile', 'dataset',
                    'notes.keyword', 'object', 'upsert', 1,
                    'server_trusted_v1', '{}', ?, 1, ?, 2, 'now'
                )
                """,
                ("c" * 64, "d" * 64),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO notes_organization_heads(
                    server_profile_id, dataset_id, domain, object_id, operation,
                    schema_version, encryption_policy, payload_json, payload_hash,
                    object_revision, object_hash, server_cursor, deleted,
                    apply_state, updated_at
                ) VALUES (
                    'profile', 'dataset', 'notes.folder', 'object', 'upsert',
                    1, 'server_trusted_v1', '{}', ?, 1, ?, 'cursor', 1,
                    'applied', 'now'
                )
                """,
                ("a" * 64, "b" * 64),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO notes_organization_sync_checkpoints(
                    server_profile_id, dataset_id, local_state, server_state,
                    captured_count, expected_count, inventory_phase, updated_at
                ) VALUES (
                    'profile', 'dataset', 'ready', 'initializing', 0, 0,
                    'complete', 'now'
                )
                """
            )

        note_id = str(db.add_note("Suppressed", "Body"))
        connection.execute(
            """
            INSERT INTO note_folder_sync_suppressions(note_id, folder_sync_id, created_at)
            VALUES (?, '11111111-1111-4111-8111-111111111111', 'now')
            """,
            (note_id,),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO note_folder_sync_suppressions(
                    note_id, folder_sync_id, created_at
                ) VALUES (?, '11111111-1111-4111-8111-111111111111', 'later')
                """,
                (note_id,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO note_folder_sync_suppressions(
                    note_id, folder_sync_id, created_at
                ) VALUES ('missing-note', '22222222-2222-4222-8222-222222222222', 'now')
                """
            )


def test_real_v57_reopen_matches_fresh_v58_intent_schema(tmp_path: Path) -> None:
    path = tmp_path / "v57-equivalence.db"
    _seed_real_v57(path)

    with chachanotes_db_at_version(
        path, 58, client_id="v58-migrated-shape"
    ) as migrated, chachanotes_db_at_version(
        ":memory:", 58, client_id="v58-fresh-shape"
    ) as fresh:
        assert _intent_columns(migrated.get_connection()) == _intent_columns(
            fresh.get_connection()
        )


def test_v58_failure_during_uuid_backfill_rolls_back_schema_data_and_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "rollback.db"
    original, original_memberships = _seed_real_v57(path)
    real_uuid4 = uuid.uuid4
    calls = 0

    def fail_during_backfill() -> uuid.UUID:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected UUID allocation failure")
        return real_uuid4()

    monkeypatch.setattr(
        "tldw_chatbook.DB.ChaChaNotes_DB.uuid.uuid4", fail_during_backfill
    )
    with pytest.raises(SchemaError, match=r"V57.*V58"):
        CharactersRAGDB(path, client_id="failed-v58-open")

    with sqlite3.connect(path) as connection:
        assert _schema_version(connection) == 57
        unchanged: dict[str, list[tuple[object, str]]] = {}
        for table in RESOURCE_TABLES:
            label_column = {
                "keywords": "keyword",
                "keyword_collections": "name",
                "note_folders": "name",
            }[table]
            unchanged[table] = [
                (row[0], str(row[1]))
                for row in connection.execute(
                    f"SELECT id, {label_column} FROM {table} ORDER BY id"
                )
            ]
        assert unchanged == original
        assert [
            tuple(row)
            for row in connection.execute(
                """
                SELECT id, folder_id, note_id, ownership, owner_id, owner_active,
                       version, deleted
                  FROM note_folder_memberships
                 ORDER BY id
                """
            )
        ] == original_memberships
        assert all(
            "sync_id"
            not in {
                str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
            }
            for table in RESOURCE_TABLES
        )
        assert not (
            STATE_TABLES
            & {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        )
