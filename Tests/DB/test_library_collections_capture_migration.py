"""Atomic schema-v2 migration coverage for Local Collections captures."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from tldw_chatbook.DB.Library_Collections_DB import (
    LibraryCollectionsDB,
    LibraryCollectionsSchemaError,
)


CAPTURE_TABLES = {
    "collection_capture_highlights",
    "collection_capture_item_tags",
    "collection_capture_items",
    "collection_capture_note_links",
    "collection_capture_offline_files",
    "collection_capture_saved_searches",
    "collection_capture_scavenge_state",
    "collection_capture_search",
    "collection_capture_tags",
}


def _open(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _create_v1_fixture(path: Path) -> None:
    with _open(path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY NOT NULL
            );
            INSERT INTO schema_version (version) VALUES (1);

            CREATE TABLE library_collections (
                collection_id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT
            );

            CREATE TABLE library_collection_items (
                membership_id TEXT PRIMARY KEY,
                collection_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY(collection_id)
                    REFERENCES library_collections(collection_id)
                    ON DELETE CASCADE,
                UNIQUE(collection_id, source_type, source_id)
            );

            INSERT INTO library_collections VALUES
                ('active', 'Active set', 'kept active', '2026-01-01', '2026-01-02', NULL),
                ('deleted', 'Deleted set', 'kept deleted', '2026-01-03', '2026-01-04', '2026-01-05');
            INSERT INTO library_collection_items VALUES
                ('member-active', 'active', 'media', '41', 'Stored title', '2026-01-06'),
                ('member-deleted', 'deleted', 'note', 'n-7', 'Deleted member', '2026-01-07');
            """
        )


def _schema_objects(path: Path) -> set[tuple[str, str]]:
    with _open(path) as connection:
        return {
            (str(row["type"]), str(row["name"]))
            for row in connection.execute(
                "SELECT type, name FROM sqlite_schema "
                "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
            )
        }


def _legacy_rows(path: Path) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    with _open(path) as connection:
        collections = [
            tuple(row)
            for row in connection.execute(
                "SELECT * FROM library_collections ORDER BY collection_id"
            )
        ]
        memberships = [
            tuple(row)
            for row in connection.execute(
                "SELECT * FROM library_collection_items ORDER BY membership_id"
            )
        ]
    return collections, memberships


def test_fresh_database_creates_v1_compatibility_and_capture_v2(tmp_path: Path) -> None:
    path = tmp_path / "collections.db"

    database = LibraryCollectionsDB(path)

    assert database.get_schema_version() == 2
    assert database.has_compatible_legacy_schema() is True
    database.require_capture_schema()
    objects = _schema_objects(path)
    assert CAPTURE_TABLES <= {name for kind, name in objects if kind == "table"}
    assert {"library_collections", "library_collection_items"} <= {
        name for kind, name in objects if kind == "table"
    }
    database.close()


def test_real_v1_fixture_migrates_without_changing_legacy_values(tmp_path: Path) -> None:
    path = tmp_path / "collections.db"
    _create_v1_fixture(path)
    before = _legacy_rows(path)

    database = LibraryCollectionsDB(path)

    assert database.get_schema_version() == 2
    assert _legacy_rows(path) == before
    assert database.has_compatible_legacy_schema() is True
    database.require_capture_schema()
    database.close()


def test_capture_ddl_failure_rolls_back_every_v2_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "collections.db"
    _create_v1_fixture(path)
    before_rows = _legacy_rows(path)
    original_ddl = LibraryCollectionsDB._CAPTURE_SCHEMA_DDL
    monkeypatch.setattr(
        LibraryCollectionsDB,
        "_CAPTURE_SCHEMA_DDL",
        (*original_ddl[:3], "CREATE TABL deliberately_invalid", *original_ddl[3:]),
    )

    with pytest.raises(sqlite3.OperationalError):
        LibraryCollectionsDB(path)

    with _open(path) as connection:
        assert connection.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 1
    assert _legacy_rows(path) == before_rows
    assert CAPTURE_TABLES.isdisjoint(
        {name for kind, name in _schema_objects(path) if kind == "table"}
    )


def test_two_concurrent_openers_publish_one_complete_v2_schema(tmp_path: Path) -> None:
    path = tmp_path / "collections.db"
    _create_v1_fixture(path)
    ready = threading.Barrier(3)
    errors: list[BaseException] = []

    def open_database() -> None:
        ready.wait()
        try:
            database = LibraryCollectionsDB(path)
            database.require_capture_schema()
            database.close()
        except BaseException as exc:  # noqa: BLE001 - reported by the parent test
            errors.append(exc)

    threads = [threading.Thread(target=open_database) for _ in range(2)]
    for thread in threads:
        thread.start()
    ready.wait()
    for thread in threads:
        thread.join(timeout=10)

    assert all(thread.is_alive() is False for thread in threads)
    assert errors == []
    with _open(path) as connection:
        assert [
            row[0]
            for row in connection.execute(
                "SELECT version FROM schema_version ORDER BY version"
            )
        ] == [1, 2]
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert CAPTURE_TABLES <= {
        name for kind, name in _schema_objects(path) if kind == "table"
    }


def test_v2_reopen_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "collections.db"
    first = LibraryCollectionsDB(path)
    first.close()
    before = _schema_objects(path)

    second = LibraryCollectionsDB(path)

    assert second.get_schema_version() == 2
    assert _schema_objects(path) == before
    second.close()


def test_future_schema_is_refused_without_writing(tmp_path: Path) -> None:
    path = tmp_path / "collections.db"
    _create_v1_fixture(path)
    with _open(path) as connection:
        connection.execute("INSERT INTO schema_version (version) VALUES (3)")
    before_objects = _schema_objects(path)
    before_rows = _legacy_rows(path)

    with pytest.raises(LibraryCollectionsSchemaError) as caught:
        LibraryCollectionsDB(path)

    assert caught.value.reason == "schema_too_new"
    assert _schema_objects(path) == before_objects
    assert _legacy_rows(path) == before_rows
    with _open(path) as connection:
        assert [
            row[0]
            for row in connection.execute(
                "SELECT version FROM schema_version ORDER BY version"
            )
        ] == [1, 3]


def test_capture_schema_gate_requires_owned_search_triggers(tmp_path: Path) -> None:
    path = tmp_path / "collections.db"
    database = LibraryCollectionsDB(path)
    with database.connection() as connection:
        connection.execute("DROP TRIGGER collection_capture_items_search_ai")

    with pytest.raises(LibraryCollectionsSchemaError) as caught:
        database.require_capture_schema()

    assert caught.value.reason == "capture_schema_unavailable"
    database.close()


def test_capture_schema_foreign_keys_are_valid(tmp_path: Path) -> None:
    path = tmp_path / "collections.db"
    database = LibraryCollectionsDB(path)

    with database.connection() as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO collection_capture_item_tags "
                "(authority_key, capture_id, tag_id) VALUES (?, ?, ?)",
                ("local:one", "missing", 1),
            )
    database.close()


def test_capture_search_triggers_follow_item_and_tag_changes(tmp_path: Path) -> None:
    path = tmp_path / "collections.db"
    database = LibraryCollectionsDB(path)

    with database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO collection_capture_items (
                authority_key, capture_id, submitted_url, canonical_url, domain,
                title, summary, freeform_note, text_content, status, favorite,
                processing_state, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "local:one",
                "capture-1",
                "https://example.org/submitted",
                "https://example.org/canonical",
                "example.org",
                "Reader foundation",
                "Summary",
                "A note",
                "Readable body",
                "saved",
                0,
                "ready",
                "2026-01-01",
                "2026-01-01",
            ),
        )
        connection.execute(
            "INSERT INTO collection_capture_tags "
            "(authority_key, tag_id, normalized_name, display_name) VALUES (?, ?, ?, ?)",
            ("local:one", 1, "research", "Research"),
        )
        connection.execute(
            "INSERT INTO collection_capture_item_tags "
            "(authority_key, capture_id, tag_id) VALUES (?, ?, ?)",
            ("local:one", "capture-1", 1),
        )

    with database.connection() as connection:
        assert connection.execute(
            "SELECT capture_id FROM collection_capture_search "
            "WHERE collection_capture_search MATCH ?",
            ('"reader"',),
        ).fetchone()[0] == "capture-1"
        assert connection.execute(
            "SELECT capture_id FROM collection_capture_search "
            "WHERE collection_capture_search MATCH ?",
            ('"research"',),
        ).fetchone()[0] == "capture-1"

    with database.transaction() as connection:
        connection.execute(
            "UPDATE collection_capture_tags SET display_name = ? "
            "WHERE authority_key = ? AND tag_id = ?",
            ("Analysis", "local:one", 1),
        )
    with database.connection() as connection:
        assert connection.execute(
            "SELECT capture_id FROM collection_capture_search "
            "WHERE collection_capture_search MATCH ?",
            ('"analysis"',),
        ).fetchone()[0] == "capture-1"
        assert connection.execute(
            "SELECT capture_id FROM collection_capture_search "
            "WHERE collection_capture_search MATCH ?",
            ('"research"',),
        ).fetchone() is None

    with database.transaction() as connection:
        connection.execute(
            "DELETE FROM collection_capture_tags "
            "WHERE authority_key = ? AND tag_id = ?",
            ("local:one", 1),
        )
    with database.connection() as connection:
        assert connection.execute(
            "SELECT capture_id FROM collection_capture_search "
            "WHERE collection_capture_search MATCH ?",
            ('"analysis"',),
        ).fetchone() is None

    with database.transaction() as connection:
        connection.execute(
            "UPDATE collection_capture_items SET title = ?, revision = revision + 1 "
            "WHERE authority_key = ? AND capture_id = ?",
            ("Changed heading", "local:one", "capture-1"),
        )
    with database.connection() as connection:
        assert connection.execute(
            "SELECT capture_id FROM collection_capture_search "
            "WHERE collection_capture_search MATCH ?",
            ('"changed"',),
        ).fetchone()[0] == "capture-1"

    with database.transaction() as connection:
        connection.execute(
            "DELETE FROM collection_capture_items WHERE authority_key = ? AND capture_id = ?",
            ("local:one", "capture-1"),
        )
    with database.connection() as connection:
        assert connection.execute(
            "SELECT capture_id FROM collection_capture_search "
            "WHERE collection_capture_search MATCH ?",
            ('"changed"',),
        ).fetchone() is None
    database.close()


@pytest.mark.parametrize(
    ("query", "expected_index"),
    [
        (
            "SELECT capture_id FROM collection_capture_items "
            "WHERE authority_key = ? ORDER BY updated_at DESC, capture_id DESC LIMIT 20",
            "idx_collection_capture_items_updated_page",
        ),
        (
            "SELECT capture_id FROM collection_capture_items "
            "WHERE authority_key = ? ORDER BY created_at DESC, capture_id DESC LIMIT 20",
            "idx_collection_capture_items_created_page",
        ),
        (
            "SELECT capture_id FROM collection_capture_items "
            "WHERE authority_key = ? ORDER BY published_at DESC, capture_id DESC LIMIT 20",
            "idx_collection_capture_items_published_page",
        ),
        (
            "SELECT capture_id FROM collection_capture_items "
            "WHERE authority_key = ? "
            "ORDER BY title COLLATE NOCASE, capture_id LIMIT 20",
            "idx_collection_capture_items_title_page",
        ),
        (
            "SELECT capture_id FROM collection_capture_items "
            "WHERE authority_key = ? AND status = ? "
            "ORDER BY updated_at DESC, capture_id DESC LIMIT 20",
            "idx_collection_capture_items_status_page",
        ),
        (
            "SELECT capture_id FROM collection_capture_items "
            "WHERE authority_key = ? AND favorite = ? "
            "ORDER BY updated_at DESC, capture_id DESC LIMIT 20",
            "idx_collection_capture_items_favorite_page",
        ),
        (
            "SELECT capture_id FROM collection_capture_items "
            "WHERE authority_key = ? AND domain = ? "
            "ORDER BY updated_at DESC, capture_id DESC LIMIT 20",
            "idx_collection_capture_items_domain_page",
        ),
    ],
)
def test_fixed_capture_page_queries_use_bounded_indexes(
    tmp_path: Path,
    query: str,
    expected_index: str,
) -> None:
    database = LibraryCollectionsDB(tmp_path / "collections.db")
    params: tuple[object, ...] = ("local:one",)
    if "status = ?" in query:
        params = (*params, "saved")
    elif "favorite = ?" in query:
        params = (*params, 1)
    elif "domain = ?" in query:
        params = (*params, "example.org")

    with database.connection() as connection:
        plan = " ".join(
            str(row[3]) for row in connection.execute(f"EXPLAIN QUERY PLAN {query}", params)
        )

    assert expected_index in plan
    assert "TEMP B-TREE" not in plan.upper()
    database.close()
