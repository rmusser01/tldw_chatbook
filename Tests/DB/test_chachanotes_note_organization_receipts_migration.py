"""ChaChaNotes v59 receipts and v60 publication-intent migration coverage."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import uuid

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, SchemaError
from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version


RECEIPT_TABLE = "note_organization_receipts"
RECEIPT_INDEX = "uq_note_organization_receipts_unresolved_note"
PUBLICATION_TABLE = "note_sync_publication_intents"
PUBLICATION_INDEX = "idx_note_sync_publication_intents_pending"
LINK_LOOKUP_INDEXES = {
    "idx_notes_organization_heads_note_subject",
    "idx_notes_organization_intents_note_subject_latest",
}
ORGANIZATION_VERSION = "a" * 64
EXPECTED_COLUMNS = (
    "receipt_id",
    "note_id",
    "requested_folder_name",
    "requested_folder_sync_id",
    "requested_keywords_json",
    "review_id",
    "collision_ids_json",
    "note_version",
    "organization_version",
    "state",
    "created_at",
    "updated_at",
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


def _index_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        )
    }


def _receipt_columns(connection: sqlite3.Connection) -> tuple[str, ...]:
    return tuple(
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({RECEIPT_TABLE})")
    )


def _receipt_schema(connection: sqlite3.Connection) -> tuple[object, ...]:
    table_sql = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (RECEIPT_TABLE,),
    ).fetchone()
    index_sql = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
        (RECEIPT_INDEX,),
    ).fetchone()
    assert table_sql is not None and index_sql is not None
    return (
        tuple(tuple(row) for row in connection.execute(f"PRAGMA table_info({RECEIPT_TABLE})")),
        str(table_sql[0]),
        tuple(tuple(row) for row in connection.execute(f"PRAGMA index_list({RECEIPT_TABLE})")),
        tuple(tuple(row) for row in connection.execute(f"PRAGMA index_info({RECEIPT_INDEX})")),
        str(index_sql[0]),
    )


def _publication_schema(connection: sqlite3.Connection) -> tuple[object, ...]:
    table_sql = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (PUBLICATION_TABLE,),
    ).fetchone()
    index_sql = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
        (PUBLICATION_INDEX,),
    ).fetchone()
    assert table_sql is not None and index_sql is not None
    return (
        tuple(
            tuple(row)
            for row in connection.execute(f"PRAGMA table_info({PUBLICATION_TABLE})")
        ),
        str(table_sql[0]),
        tuple(
            tuple(row)
            for row in connection.execute(f"PRAGMA index_list({PUBLICATION_TABLE})")
        ),
        tuple(
            tuple(row)
            for row in connection.execute(f"PRAGMA index_info({PUBLICATION_INDEX})")
        ),
        str(index_sql[0]),
    )


def _organization_row_counts(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in ("notes", "note_folders", "keywords", "keyword_collections")
    }


def _null_organization_sync_ids(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        table: int(
            connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE sync_id IS NULL"
            ).fetchone()[0]
        )
        for table in ("keywords", "keyword_collections", "note_folders")
    }


def _seed_real_v58(path: Path) -> tuple[str, dict[str, int]]:
    with chachanotes_db_at_version(path, 58, client_id="receipt-v58-seed") as db:
        connection = db.get_connection()
        note_id = str(db.add_note("Existing note", "Existing body"))
        connection.execute(
            "INSERT INTO keywords(keyword, deleted) VALUES ('existing-keyword', 0)"
        )
        connection.execute(
            "INSERT INTO keyword_collections(name, deleted) VALUES ('existing-collection', 0)"
        )
        connection.execute(
            """
            INSERT INTO note_folders(
                id, parent_id, name, normalized_name, path, normalized_path,
                version, deleted, created_at, modified_at, sync_id
            ) VALUES (
                'existing-folder', NULL, 'Existing', 'existing', '/Existing',
                '/existing', 1, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
                NULL
            )
            """
        )
        connection.commit()
        assert _schema_version(connection) == 58
        assert RECEIPT_TABLE not in _table_names(connection)
        assert _null_organization_sync_ids(connection) == {
            "keywords": 1,
            "keyword_collections": 1,
            "note_folders": 1,
        }
        return note_id, _organization_row_counts(connection)


def _seed_real_v59(path: Path) -> tuple[str, tuple[object, ...]]:
    with chachanotes_db_at_version(path, 59, client_id="publication-v59-seed") as db:
        connection = db.get_connection()
        note_id = str(db.add_note("Existing v59 note", "Existing v59 body"))
        _insert_receipt(
            connection,
            receipt_id="existing-v59-receipt",
            note_id=note_id,
        )
        connection.commit()
        assert _schema_version(connection) == 59
        assert RECEIPT_TABLE in _table_names(connection)
        assert PUBLICATION_TABLE not in _table_names(connection)
        assert PUBLICATION_INDEX not in _index_names(connection)
        return note_id, _receipt_schema(connection)


def _insert_receipt(
    connection: sqlite3.Connection,
    *,
    receipt_id: str,
    note_id: str,
    organization_version: str = ORGANIZATION_VERSION,
    state: str = "pending_organization",
    review_id: str | None = None,
    collision_ids_json: str = "[]",
) -> None:
    connection.execute(
        f"""
        INSERT INTO {RECEIPT_TABLE}(
            receipt_id, note_id, requested_folder_name,
            requested_folder_sync_id, requested_keywords_json, review_id,
            collision_ids_json, note_version, organization_version, state,
            created_at, updated_at
        ) VALUES (?, ?, 'Agent_Lessons', ?, '["agent-lesson"]', ?, ?, 3, ?, ?, 'now', 'now')
        """,
        (
            receipt_id,
            note_id,
            "22222222-2222-4222-8222-222222222222",
            review_id,
            collision_ids_json,
            organization_version,
            state,
        ),
    )


def test_real_v58_reopen_adds_empty_content_free_receipt_state(tmp_path: Path) -> None:
    path = tmp_path / "receipt-v58.sqlite"
    note_id, before_counts = _seed_real_v58(path)

    migrated = CharactersRAGDB(path, client_id="receipt-v59-migrate")
    try:
        connection = migrated.get_connection()
        assert CharactersRAGDB._CURRENT_SCHEMA_VERSION >= 60
        assert _schema_version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        assert RECEIPT_TABLE in _table_names(connection)
        assert PUBLICATION_TABLE in _table_names(connection)
        assert RECEIPT_INDEX in _index_names(connection)
        assert PUBLICATION_INDEX in _index_names(connection)
        assert LINK_LOOKUP_INDEXES <= _index_names(connection)
        assert _receipt_columns(connection) == EXPECTED_COLUMNS
        assert _organization_row_counts(connection) == before_counts
        assert _null_organization_sync_ids(connection) == {
            "keywords": 0,
            "keyword_collections": 0,
            "note_folders": 0,
        }
        assert connection.execute(f"SELECT COUNT(*) FROM {RECEIPT_TABLE}").fetchone()[0] == 0
        assert connection.execute(
            f"SELECT COUNT(*) FROM {PUBLICATION_TABLE}"
        ).fetchone()[0] == 0

        column_names = {name.casefold() for name in _receipt_columns(connection)}
        forbidden_fragments = ("body", "content", "file", "path", "secret", "token", "credential")
        assert not {
            name
            for name in column_names
            if any(fragment in name for fragment in forbidden_fragments)
        }

        _insert_receipt(connection, receipt_id="receipt-1", note_id=note_id)
        connection.commit()
    finally:
        migrated.close_connection()

    reopened = CharactersRAGDB(path, client_id="receipt-v59-reopen")
    try:
        connection = reopened.get_connection()
        assert _schema_version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        assert _organization_row_counts(connection) == before_counts
        row = connection.execute(
            f"SELECT receipt_id, note_id, requested_keywords_json, state FROM {RECEIPT_TABLE}"
        ).fetchone()
        assert tuple(row) == (
            "receipt-1",
            note_id,
            '["agent-lesson"]',
            "pending_organization",
        )
    finally:
        reopened.close_connection()


def test_fresh_v59_receipt_schema_matches_real_v58_migration(tmp_path: Path) -> None:
    path = tmp_path / "receipt-parity.sqlite"
    _seed_real_v58(path)
    migrated = CharactersRAGDB(path, client_id="receipt-parity-migrated")
    fresh = CharactersRAGDB(":memory:", client_id="receipt-parity-fresh")
    try:
        assert _receipt_schema(migrated.get_connection()) == _receipt_schema(
            fresh.get_connection()
        )
        assert _publication_schema(migrated.get_connection()) == _publication_schema(
            fresh.get_connection()
        )
        migrated_indexes = {
            str(row[0]): str(row[1])
            for row in migrated.get_connection().execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'index' "
                "AND name IN (?, ?)",
                tuple(sorted(LINK_LOOKUP_INDEXES)),
            )
        }
        fresh_indexes = {
            str(row[0]): str(row[1])
            for row in fresh.get_connection().execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'index' "
                "AND name IN (?, ?)",
                tuple(sorted(LINK_LOOKUP_INDEXES)),
            )
        }
        assert migrated_indexes == fresh_indexes
        assert set(migrated_indexes) == LINK_LOOKUP_INDEXES
        for name, sql in migrated_indexes.items():
            normalized = " ".join(sql.casefold().split())
            assert (
                "when domain = 'notes.folder_link' "
                "then json_extract(payload_json, '$.note_id')"
            ) in normalized
            assert (
                "when domain = 'notes.keyword_link' "
                "and json_extract(payload_json, '$.subject_type') = 'note' "
                "then json_extract(payload_json, '$.subject_id')"
            ) in normalized
            assert (
                "where domain = 'notes.folder_link' "
                "or (domain = 'notes.keyword_link' "
                "and json_extract(payload_json, '$.subject_type') = 'note')"
            ) in normalized
            if name == "idx_notes_organization_intents_note_subject_latest":
                assert "intent_sequence desc" in normalized
    finally:
        migrated.close_connection()
        fresh.close_connection()


def test_real_v59_reopen_adds_publication_intents_without_rewriting_receipts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "publication-v59.sqlite"
    note_id, receipt_schema = _seed_real_v59(path)

    migrated = CharactersRAGDB(path, client_id="publication-v60-migrate")
    try:
        connection = migrated.get_connection()
        assert _schema_version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        assert _receipt_schema(connection) == receipt_schema
        assert PUBLICATION_TABLE in _table_names(connection)
        assert PUBLICATION_INDEX in _index_names(connection)
        assert tuple(
            str(row[2])
            for row in connection.execute(f"PRAGMA index_info({PUBLICATION_INDEX})")
        ) == (
            "server_profile_id",
            "dataset_id",
            "note_id",
            "entity_version",
            "intent_id",
        )
        plan = connection.execute(
            f"EXPLAIN QUERY PLAN SELECT intent_id FROM {PUBLICATION_TABLE} "
            "WHERE server_profile_id = ? AND dataset_id = ? "
            "AND acknowledged_at IS NULL AND cancelled_at IS NULL "
            "ORDER BY note_id, entity_version, intent_id",
            ("server-a", "dataset-a"),
        ).fetchall()
        assert any(
            f"USING INDEX {PUBLICATION_INDEX}" in str(row[3]) for row in plan
        )
        assert connection.execute(
            f"SELECT COUNT(*) FROM {PUBLICATION_TABLE}"
        ).fetchone()[0] == 0
        receipt = connection.execute(
            f"SELECT receipt_id, note_id FROM {RECEIPT_TABLE}"
        ).fetchone()
        assert tuple(receipt) == ("existing-v59-receipt", note_id)
    finally:
        migrated.close_connection()

    reopened = CharactersRAGDB(path, client_id="publication-v60-reopen")
    try:
        assert (
            _schema_version(reopened.get_connection())
            == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        )
        assert _publication_schema(reopened.get_connection())
    finally:
        reopened.close_connection()


def test_v60_migration_failure_rolls_back_publication_state_and_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "publication-v60-rollback.sqlite"
    _seed_real_v59(path)
    real_execute = CharactersRAGDB._execute_migration_statements

    def fail_after_publication_ddl(
        self: CharactersRAGDB,
        cursor: sqlite3.Cursor,
        script: str,
        label: str,
    ) -> None:
        real_execute(self, cursor, script, label)
        if label == "V59→V60":
            raise RuntimeError("injected after publication DDL")

    monkeypatch.setattr(
        CharactersRAGDB,
        "_execute_migration_statements",
        fail_after_publication_ddl,
    )
    with pytest.raises(SchemaError, match=r"V59.*V60"):
        CharactersRAGDB(path, client_id="publication-v60-failure")

    with sqlite3.connect(path) as connection:
        assert _schema_version(connection) == 59
        assert PUBLICATION_TABLE not in _table_names(connection)
        assert PUBLICATION_INDEX not in _index_names(connection)
        assert connection.execute(
            f"SELECT COUNT(*) FROM {RECEIPT_TABLE}"
        ).fetchone()[0] == 1


def test_current_schema_reopen_repairs_missing_organization_sync_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipt-v59-repair.sqlite"
    db = CharactersRAGDB(path, client_id="receipt-v59-repair-seed")
    try:
        connection = db.get_connection()
        keyword_id = db.add_keyword("late-null-keyword")
        assert keyword_id is not None
        connection.execute(
            "UPDATE keywords SET sync_id = NULL WHERE id = ?", (keyword_id,)
        )
        connection.commit()
        assert _schema_version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        assert _null_organization_sync_ids(connection)["keywords"] == 1
    finally:
        db.close_connection()

    repaired = CharactersRAGDB(path, client_id="receipt-v59-repair-open")
    try:
        connection = repaired.get_connection()
        assert _schema_version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        assert _null_organization_sync_ids(connection) == {
            "keywords": 0,
            "keyword_collections": 0,
            "note_folders": 0,
        }
        repaired_sync_id = connection.execute(
            "SELECT sync_id FROM keywords WHERE id = ?", (keyword_id,)
        ).fetchone()[0]
        parsed = uuid.UUID(repaired_sync_id)
        assert str(parsed) == repaired_sync_id
        assert parsed.version == 4
    finally:
        repaired.close_connection()


def test_current_schema_reopen_restores_link_lookup_indexes_and_query_plans(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipt-v59-index-repair.sqlite"
    db = CharactersRAGDB(path, client_id="receipt-v59-index-seed")
    try:
        note_id = str(db.add_note("Index repair", "Body"))
        connection = db.get_connection()
        for index_name in sorted(LINK_LOOKUP_INDEXES):
            connection.execute(f"DROP INDEX {index_name}")
        connection.commit()
        assert not (LINK_LOOKUP_INDEXES & _index_names(connection))
        assert _schema_version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
    finally:
        db.close_connection()

    reopened = CharactersRAGDB(path, client_id="receipt-v59-index-open")
    try:
        connection = reopened.get_connection()
        assert LINK_LOOKUP_INDEXES <= _index_names(connection)

        traced: list[str] = []
        connection.set_trace_callback(traced.append)
        try:
            assert reopened.get_library_note_text(
                note_id, start=0, max_chars=20
            ) is not None
        finally:
            connection.set_trace_callback(None)
        lookups = [
            statement
            for statement in traced
            if "FROM notes_organization_heads" in statement
            or "FROM notes_organization_sync_intents" in statement
        ]
        details = [
            str(row[3])
            for statement in lookups
            for row in connection.execute(f"EXPLAIN QUERY PLAN {statement}")
        ]
        combined = "\n".join(details)
        assert "idx_notes_organization_heads_note_subject" in combined
        assert "idx_notes_organization_intents_note_subject_latest" in combined
        assert "SCAN notes_organization_heads" not in combined
        assert "SCAN notes_organization_sync_intents" not in combined
        assert "USE TEMP B-TREE" not in combined
    finally:
        reopened.close_connection()


def test_receipt_constraints_enforce_one_unresolved_state_per_note() -> None:
    db = CharactersRAGDB(":memory:", client_id="receipt-constraints")
    try:
        connection = db.get_connection()
        first_note_id = str(db.add_note("First", "Body"))
        second_note_id = str(db.add_note("Second", "Body"))

        _insert_receipt(connection, receipt_id="pending", note_id=first_note_id)
        with pytest.raises(sqlite3.IntegrityError):
            _insert_receipt(
                connection,
                receipt_id="duplicate-note",
                note_id=first_note_id,
                state="placement_review",
                review_id="review-1",
                collision_ids_json='["collision-1"]',
            )
        connection.rollback()

        invalid_states = (
            ("unknown", None, "[]"),
            ("pending_organization", "review-not-allowed", "[]"),
            ("pending_organization", None, '["collision-not-allowed"]'),
            ("placement_review", None, '["collision-1"]'),
            ("placement_review", "review-1", "[]"),
        )
        for index, (state, review_id, collision_ids_json) in enumerate(invalid_states):
            with pytest.raises(sqlite3.IntegrityError):
                _insert_receipt(
                    connection,
                    receipt_id=f"bad-state-{index}",
                    note_id=second_note_id,
                    state=state,
                    review_id=review_id,
                    collision_ids_json=collision_ids_json,
                )
            connection.rollback()

        _insert_receipt(
            connection,
            receipt_id="placement",
            note_id=second_note_id,
            state="placement_review",
            review_id="review-1",
            collision_ids_json='["collision-1"]',
        )
        connection.commit()
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    "invalid_version",
    ("a" * 63, "a" * 65, "A" * 64, "g" * 64, "opaque-version"),
)
def test_receipt_rejects_non_sha256_organization_versions(invalid_version: str) -> None:
    db = CharactersRAGDB(":memory:", client_id="receipt-hash-constraint")
    try:
        connection = db.get_connection()
        note_id = str(db.add_note("Hash target", "Body"))
        with pytest.raises(sqlite3.IntegrityError):
            _insert_receipt(
                connection,
                receipt_id="bad-hash",
                note_id=note_id,
                organization_version=invalid_version,
            )
    finally:
        db.close_connection()


def test_v59_migration_failure_rolls_back_table_index_data_and_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "receipt-rollback.sqlite"
    _, before_counts = _seed_real_v58(path)
    real_repair = CharactersRAGDB._repair_missing_notes_organization_sync_ids

    def fail_after_portable_id_repair(
        connection: sqlite3.Connection | sqlite3.Cursor,
    ) -> None:
        real_repair(connection)
        raise RuntimeError("injected after portable ID repair")

    monkeypatch.setattr(
        CharactersRAGDB,
        "_repair_missing_notes_organization_sync_ids",
        staticmethod(fail_after_portable_id_repair),
    )
    with pytest.raises(SchemaError, match=r"V58.*V59"):
        CharactersRAGDB(path, client_id="receipt-v59-failure")

    with sqlite3.connect(path) as connection:
        assert _schema_version(connection) == 58
        assert RECEIPT_TABLE not in _table_names(connection)
        assert RECEIPT_INDEX not in _index_names(connection)
        assert not (LINK_LOOKUP_INDEXES & _index_names(connection))
        assert _organization_row_counts(connection) == before_counts
        assert _null_organization_sync_ids(connection) == {
            "keywords": 1,
            "keyword_collections": 1,
            "note_folders": 1,
        }
