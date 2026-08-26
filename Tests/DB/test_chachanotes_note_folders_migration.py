"""ChaChaNotes V35 -> V36 local note-folder schema migration coverage."""

import sqlite3
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, SchemaError
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from Tests.ChaChaNotesDB.historical_bootstrap import (
    chachanotes_db_at_version,
    open_current_chachanotes_from_legacy,
)

EXPECTED_FOLDER_TABLES = {"note_folders", "note_folder_memberships"}
MANAGED_OWNER_INDEX = "idx_note_folder_memberships_managed_owner"


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


def _seed_v35(path: Path) -> str:
    # This file's patched-version bootstrap idiom became the shared primitive
    # in task-16840 (Tests/ChaChaNotesDB/historical_bootstrap.py); seed
    # through it so there is exactly one implementation.
    with chachanotes_db_at_version(path, 35, client_id="v35-seed") as db:
        note_id = db.add_note("Existing", "Body")
        assert _schema_version(db) == 35
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
        assert _schema_version(db) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        assert EXPECTED_FOLDER_TABLES <= _table_names(db)
    finally:
        db.close_connection()


def test_managed_owner_operations_use_the_owner_lookup_index(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "owner-index.db", client_id="owner-index")
    try:
        connection = db.get_connection()
        index_names = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA index_list('note_folder_memberships')"
            ).fetchall()
        }
        plan = connection.execute(
            "EXPLAIN QUERY PLAN "
            "SELECT id, folder_id, note_id, version "
            "FROM note_folder_memberships "
            "WHERE ownership = 'managed' AND owner_id = ? AND deleted = 0 "
            "ORDER BY folder_id, note_id, id",
            ("root-a",),
        ).fetchall()
        details = " ".join(str(row["detail"]) for row in plan)

        assert MANAGED_OWNER_INDEX in index_names
        assert MANAGED_OWNER_INDEX in details
    finally:
        db.close_connection()


def test_v35_database_migrates_without_assigning_existing_notes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v35.db"
    note_id = _seed_v35(path)

    migrated = open_current_chachanotes_from_legacy(
        path, client_id="v36-open"
    )
    try:
        count = migrated.get_connection().execute(
            "SELECT COUNT(*) AS count FROM note_folder_memberships"
        ).fetchone()["count"]

        assert _schema_version(migrated) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
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
    _seed_v35(path)
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


def test_chachanotes_backup_preserves_owned_folders_and_supports_restore_review(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.UI.Tools_Settings_Window import SETTINGS_DATABASES

    source_path = tmp_path / "source.db"
    backup_path = tmp_path / "backup.db"
    source = CharactersRAGDB(source_path, client_id="backup-source")
    repository = LocalNoteFolderRepository(source)
    folder = repository.create_folder(name="Folder", parent_id=None)
    note_id = source.add_note("Preserved", "Exact content")
    assert note_id is not None
    manual = repository.attach_manual(folder_id=folder.folder_id, note_id=note_id)
    managed = repository.reconcile_managed(
        owner_id="restored-root", desired=((folder.folder_id, note_id),)
    )[0]
    removable = repository.reconcile_managed(
        owner_id="remove-root", desired=((folder.folder_id, note_id),)
    )[0]
    note_before = tuple(
        source.get_connection()
        .execute(
            "SELECT id, title, content, deleted, version FROM notes WHERE id = ?",
            (note_id,),
        )
        .fetchone()
    )
    with sqlite3.connect(backup_path) as destination:
        source.get_connection().backup(destination)
    source.close_connection()

    restored_db = CharactersRAGDB(backup_path, client_id="backup-restored")
    restored = LocalNoteFolderRepository(restored_db)
    try:
        assert restored.get_folder(folder.folder_id) == folder
        restored_memberships = restored.list_memberships(
            note_ids=(note_id,), include_inactive=True
        )
        assert {item.membership_id for item in restored_memberships} == {
            manual.membership_id,
            managed.membership_id,
            removable.membership_id,
        }

        assert restored.mark_unknown_owners_inactive(active_owner_ids=()) == 2
        assert restored.list_memberships(note_ids=(note_id,)) == (manual,)
        review_owners = [review.owner_id for review in restored.list_restore_reviews()]
        assert review_owners == ["remove-root", "restored-root"]
        assert {
            item.membership_id
            for item in restored.list_memberships(
                note_ids=(note_id,), include_inactive=True
            )
        } == {manual.membership_id, managed.membership_id, removable.membership_id}

        assert restored.remove_owner_memberships(owner_id="remove-root") == 1
        assert restored.convert_owner_to_manual(owner_id="restored-root") == 1
        final_memberships = restored.list_memberships(
            note_ids=(note_id,), include_inactive=True
        )
        assert final_memberships == (manual,)
        note_after = tuple(
            restored_db.get_connection()
            .execute(
                "SELECT id, title, content, deleted, version FROM notes WHERE id = ?",
                (note_id,),
            )
            .fetchone()
        )
        assert note_after == note_before
    finally:
        restored_db.close_connection()

    database_names = {name for name, _display, _stem in SETTINGS_DATABASES}
    assert "chachanotes" in database_names
    assert not any("sync" in name or "folder" in name for name in database_names)
