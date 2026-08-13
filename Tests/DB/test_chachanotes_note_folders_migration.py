"""ChaChaNotes V35 -> V36 local note-folder schema migration coverage."""

from pathlib import Path
import sqlite3

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, SchemaError


EXPECTED_FOLDER_TABLES = {"note_folders", "note_folder_memberships"}


def _schema_version(db: CharactersRAGDB) -> int:
    row = db.get_connection().execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (db._SCHEMA_NAME,),
    ).fetchone()
    assert row is not None
    return int(row["version"])


def _table_names(db: CharactersRAGDB) -> set[str]:
    rows = db.get_connection().execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {str(row["name"]) for row in rows}


def _seed_v35(path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    with monkeypatch.context() as v35:
        v35.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 35)
        db = CharactersRAGDB(path, client_id="v35-seed")
        note_id = db.add_note("Existing", "Body")
        assert _schema_version(db) == 35
        db.close_connection()
    return str(note_id)


def _insert_folder(connection: sqlite3.Connection, folder_id: str = "folder-1") -> None:
    connection.execute(
        """
        INSERT INTO note_folders(
            id, parent_id, name, normalized_name, path, normalized_path,
            created_at, modified_at
        ) VALUES (?, NULL, 'Folder', 'folder', '/Folder', '/folder',
                  CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """,
        (folder_id,),
    )


def test_fresh_database_has_v36_folder_schema(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "fresh.db", client_id="fresh")
    try:
        assert _schema_version(db) == 36
        assert EXPECTED_FOLDER_TABLES <= _table_names(db)
    finally:
        db.close_connection()


def test_v35_database_migrates_without_assigning_existing_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "v35.db"
    note_id = _seed_v35(path, monkeypatch)

    migrated = CharactersRAGDB(path, client_id="v36-open")
    try:
        count = migrated.get_connection().execute(
            "SELECT COUNT(*) AS count FROM note_folder_memberships"
        ).fetchone()["count"]

        assert _schema_version(migrated) == 36
        assert migrated.get_note_by_id(note_id) is not None
        assert count == 0
    finally:
        migrated.close_connection()


def test_database_rejects_duplicate_active_folder_normalized_path(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "duplicate-path.db", client_id="constraints")
    try:
        connection = db.get_connection()
        _insert_folder(connection)

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO note_folders(
                    id, parent_id, name, normalized_name, path, normalized_path,
                    created_at, modified_at
                ) VALUES (
                    'folder-2', NULL, 'Folder copy', 'folder copy',
                    '/Folder copy', '/folder', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                )
                """
            )
    finally:
        db.close_connection()


def test_database_rejects_folder_that_is_its_own_parent(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "self-parent.db", client_id="constraints")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            db.get_connection().execute(
                """
                INSERT INTO note_folders(
                    id, parent_id, name, normalized_name, path, normalized_path,
                    created_at, modified_at
                ) VALUES (
                    'folder-1', 'folder-1', 'Folder', 'folder', '/Folder',
                    '/folder', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                )
                """
            )
    finally:
        db.close_connection()


def test_database_rejects_managed_membership_without_owner_id(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "managed-owner.db", client_id="constraints")
    try:
        note_id = str(db.add_note("Existing", "Body"))
        connection = db.get_connection()
        _insert_folder(connection)

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO note_folder_memberships(
                    id, folder_id, note_id, ownership, owner_id,
                    created_at, modified_at
                ) VALUES (?, 'folder-1', ?, 'managed', '', CURRENT_TIMESTAMP,
                          CURRENT_TIMESTAMP)
                """,
                ("membership-1", note_id),
            )
    finally:
        db.close_connection()


def test_database_rejects_manual_membership_with_owner_id(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "manual-owner.db", client_id="constraints")
    try:
        note_id = str(db.add_note("Existing", "Body"))
        connection = db.get_connection()
        _insert_folder(connection)

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO note_folder_memberships(
                    id, folder_id, note_id, ownership, owner_id,
                    created_at, modified_at
                ) VALUES (?, 'folder-1', ?, 'manual', 'owner-1', CURRENT_TIMESTAMP,
                          CURRENT_TIMESTAMP)
                """,
                ("membership-1", note_id),
            )
    finally:
        db.close_connection()


def test_opening_already_v36_database_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "idempotent.db"
    db = CharactersRAGDB(path, client_id="first-open")
    try:
        note_id = str(db.add_note("Existing", "Body"))
        connection = db.get_connection()
        _insert_folder(connection)
        connection.execute(
            """
            INSERT INTO note_folder_memberships(
                id, folder_id, note_id, ownership, owner_id, created_at, modified_at
            ) VALUES (?, 'folder-1', ?, 'manual', '', CURRENT_TIMESTAMP,
                      CURRENT_TIMESTAMP)
            """,
            ("membership-1", note_id),
        )
        connection.commit()
        before = (
            _schema_version(db),
            _table_names(db),
            connection.execute("SELECT COUNT(*) FROM note_folders").fetchone()[0],
            connection.execute("SELECT COUNT(*) FROM note_folder_memberships").fetchone()[
                0
            ],
        )
    finally:
        db.close_connection()

    reopened = CharactersRAGDB(path, client_id="second-open")
    try:
        connection = reopened.get_connection()
        after = (
            _schema_version(reopened),
            _table_names(reopened),
            connection.execute("SELECT COUNT(*) FROM note_folders").fetchone()[0],
            connection.execute("SELECT COUNT(*) FROM note_folder_memberships").fetchone()[
                0
            ],
        )
        assert after == before
    finally:
        reopened.close_connection()


def test_v35_to_v36_failure_rolls_back_schema_and_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "rollback.db"
    _seed_v35(path, monkeypatch)
    migration_path = (
        Path(__file__).parents[2]
        / "tldw_chatbook"
        / "DB"
        / "migrations"
        / "chachanotes_v35_to_v36_note_folders.sql"
    )
    original_read_text = Path.read_text

    def read_text_with_invalid_v36_sql(path_to_read: Path, *args, **kwargs) -> str:
        if path_to_read == migration_path:
            return """
            CREATE TABLE note_folders(id TEXT PRIMARY KEY);
            INVALID SQL;
            """
        return original_read_text(path_to_read, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text_with_invalid_v36_sql)

    with pytest.raises(SchemaError, match=r"V35.*V36"):
        CharactersRAGDB(path, client_id="failed-v36-open")

    with sqlite3.connect(path) as connection:
        version = connection.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    assert version == 35
    assert not (EXPECTED_FOLDER_TABLES & tables)
