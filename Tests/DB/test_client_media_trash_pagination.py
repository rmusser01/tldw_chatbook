"""Focused database-contract tests for Library Media Trash pagination."""

from __future__ import annotations

import sqlite3
import threading

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase


@pytest.fixture
def media_db(tmp_path):
    database_path = tmp_path / "media.db"
    database = MediaDatabase(db_path=database_path, client_id="trash-pagination-test")
    yield database, database_path
    database.close_connection()


def _add_trashed_media(
    database: MediaDatabase,
    *,
    title: str,
    media_type: str,
    content: str = "private-content",
) -> int:
    media_id, _uuid, _message = database.add_media_with_keywords(
        title=title,
        media_type=media_type,
        content=f"{content}-{title}",
        keywords=[],
    )
    assert media_id is not None
    assert database.mark_as_trash(media_id)
    return media_id


def _seed_trash(database: MediaDatabase, *, count: int) -> list[int]:
    return [
        _add_trashed_media(
            database,
            title=f"Trash {index:02d}",
            media_type="pdf" if index % 2 else "audio",
            content=f"private-content-{index}",
        )
        for index in range(count)
    ]


def test_library_trash_pages_filter_before_slicing_and_echo_coordinates(media_db):
    database, _path = media_db
    media_ids = _seed_trash(database, count=45)

    pages = [
        database.list_library_media_trash_page(limit=20, offset=offset)
        for offset in (0, 20, 40)
    ]

    assert [page["total"] for page in pages] == [45, 45, 45]
    assert [page["limit"] for page in pages] == [20, 20, 20]
    assert [page["offset"] for page in pages] == [0, 20, 40]
    assert [len(page["items"]) for page in pages] == [20, 20, 5]
    assert all(
        set(item) == {"id", "title", "type", "trash_date"}
        for page in pages
        for item in page["items"]
    )
    assert [item["id"] for page in pages for item in page["items"]] == list(
        reversed(media_ids)
    )
    assert all("private-content" not in repr(item) for page in pages for item in page["items"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"limit": True},
        {"limit": 0},
        {"limit": 19},
        {"limit": 21},
        {"limit": "20"},
        {"limit": 2**63},
        {"offset": True},
        {"offset": "0"},
        {"offset": -1},
        {"offset": 2**63},
    ],
)
def test_library_trash_rejects_invalid_coordinates_before_sql(media_db, kwargs):
    database, _path = media_db
    statements: list[str] = []
    connection = database.get_connection()
    connection.set_trace_callback(statements.append)
    try:
        with pytest.raises(ValueError):
            database.list_library_media_trash_page(**kwargs)
    finally:
        connection.set_trace_callback(None)

    assert statements == []


def test_library_trash_title_query_matches_like_metacharacters_literally(media_db):
    database, _path = media_db
    percent_id = _add_trashed_media(database, title="budget 100%", media_type="pdf")
    underscore_id = _add_trashed_media(database, title="under_score", media_type="pdf")
    slash_id = _add_trashed_media(database, title=r"slash\\title", media_type="pdf")
    _add_trashed_media(database, title="ordinary title", media_type="pdf")

    assert [item["id"] for item in database.list_library_media_trash_page(query="%")["items"]] == [percent_id]
    assert [item["id"] for item in database.list_library_media_trash_page(query="_")["items"]] == [underscore_id]
    assert [item["id"] for item in database.list_library_media_trash_page(query=r"\\")["items"]] == [slash_id]


def test_library_trash_filter_is_trimmed_exact_and_case_sensitive(media_db):
    database, _path = media_db
    pdf_id = _add_trashed_media(database, title="Padded PDF", media_type=" pdf ")
    _add_trashed_media(database, title="Upper PDF", media_type="PDF")
    _add_trashed_media(database, title="Audio", media_type="audio")

    page = database.list_library_media_trash_page(media_type="  pdf  ")

    assert page["total"] == 1
    assert [item["id"] for item in page["items"]] == [pdf_id]
    assert database.list_library_media_trash_page(media_type="PDF")["total"] == 1
    assert database.list_library_media_trash_page(media_type="Pdf")["total"] == 0


def test_library_trash_facets_are_complete_trimmed_unique_and_independent(media_db):
    database, _path = media_db
    _add_trashed_media(database, title="One", media_type=" pdf ")
    _add_trashed_media(database, title="Two", media_type="audio")
    _add_trashed_media(database, title="Three", media_type="audio")
    _add_trashed_media(database, title="Blank", media_type=" \t ")
    active_id, _uuid, _message = database.add_media_with_keywords(
        title="Active only", media_type="active-private", content="active", keywords=[]
    )
    deleted_id = _add_trashed_media(
        database, title="Deleted only", media_type="deleted-private"
    )
    assert active_id is not None
    connection = database.get_connection()
    connection.execute(
        "UPDATE Media SET deleted = 1, version = version + 1 WHERE id = ?",
        (deleted_id,),
    )
    connection.commit()

    page = database.list_library_media_trash_page(
        query="One", media_type="pdf", offset=0
    )

    assert page["total"] == 1
    assert page["types"] == ["audio", "pdf"]


def test_library_trash_order_is_null_last_and_uses_id_tie_breaker(media_db):
    database, _path = media_db
    older_id = _add_trashed_media(database, title="Older", media_type="pdf")
    first_tie_id = _add_trashed_media(database, title="First tie", media_type="pdf")
    second_tie_id = _add_trashed_media(database, title="Second tie", media_type="pdf")
    no_trash_date_id = _add_trashed_media(database, title="No trash date", media_type="pdf")
    connection = database.get_connection()
    connection.executemany(
        "UPDATE Media SET trash_date = ?, last_modified = ?, version = version + 1 WHERE id = ?",
        [
            ("2026-08-01T00:00:00+00:00", "2026-08-01T00:00:00+00:00", older_id),
            ("2026-08-02T00:00:00+00:00", "2026-08-03T00:00:00+00:00", first_tie_id),
            ("2026-08-02T00:00:00+00:00", "2026-08-03T00:00:00+00:00", second_tie_id),
            (None, "2026-08-04T00:00:00+00:00", no_trash_date_id),
        ],
    )
    connection.commit()
    statements: list[str] = []
    connection.set_trace_callback(statements.append)
    try:
        page = database.list_library_media_trash_page()
    finally:
        connection.set_trace_callback(None)

    assert [item["id"] for item in page["items"]] == [
        second_tie_id,
        first_tie_id,
        older_id,
        no_trash_date_id,
    ]
    row_select = next(sql for sql in statements if "FROM Media" in sql and "LIMIT 20" in sql)
    assert "trash_date IS NULL ASC" in row_select
    assert "last_modified IS NULL ASC" in row_select
    assert "id DESC" in row_select


def test_library_trash_order_sorts_null_last_modified_last_in_real_rows(media_db):
    database, _path = media_db
    connection = database.get_connection()
    connection.execute("PRAGMA foreign_keys = OFF")
    connection.execute("DROP TABLE Media")
    connection.execute(
        """
        CREATE TABLE Media (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            type TEXT NOT NULL,
            trash_date DATETIME,
            last_modified DATETIME,
            deleted BOOLEAN NOT NULL DEFAULT 0,
            is_trash BOOLEAN NOT NULL DEFAULT 0
        )
        """
    )
    connection.executemany(
        """
        INSERT INTO Media (id, title, type, trash_date, last_modified, deleted, is_trash)
        VALUES (?, ?, ?, ?, ?, 0, 1)
        """,
        [
            (1, "Older modification", "pdf", "2026-08-02T00:00:00+00:00", "2026-08-01T00:00:00+00:00"),
            (2, "Newest modification", "pdf", "2026-08-02T00:00:00+00:00", "2026-08-03T00:00:00+00:00"),
            (3, "Missing modification", "pdf", "2026-08-02T00:00:00+00:00", None),
        ],
    )
    connection.commit()

    page = database.list_library_media_trash_page()

    assert [item["id"] for item in page["items"]] == [2, 1, 3]


@pytest.mark.parametrize(
    "query",
    ["\x00", "x" * 201],
)
def test_library_trash_rejects_invalid_query(media_db, query):
    database, _path = media_db

    with pytest.raises(ValueError, match="Library Media Trash query is invalid"):
        database.list_library_media_trash_page(query=query)


def test_library_trash_count_rows_and_facets_share_wal_read_snapshot(media_db):
    database, database_path = media_db
    media_ids = _seed_trash(database, count=45)
    connection = database.get_connection()
    connection.execute(
        "UPDATE Media SET type = ?, version = version + 1 WHERE id = ?",
        ("writer-only", media_ids[-1]),
    )
    connection.commit()
    writer_started = threading.Event()
    writer_done = threading.Event()
    writer_errors: list[Exception] = []

    def writer() -> None:
        try:
            with sqlite3.connect(database_path, timeout=5) as writer_connection:
                assert writer_started.wait(5)
                writer_connection.execute(
                    "UPDATE Media SET deleted = 1, version = version + 1 WHERE id = ?",
                    (media_ids[-1],),
                )
        except Exception as error:  # surfaced in the test thread below
            writer_errors.append(error)
        finally:
            writer_done.set()

    worker = threading.Thread(target=writer)
    worker.start()
    count_started = False
    progress_calls = 0
    write_completed_during_read = False

    def trace(sql: str) -> None:
        nonlocal count_started
        if "SELECT COUNT(*) AS count FROM Media" in sql:
            count_started = True

    def coordinate_write() -> int:
        nonlocal progress_calls, write_completed_during_read
        if count_started and not writer_started.is_set():
            progress_calls += 1
            if progress_calls >= 20:
                writer_started.set()
                assert writer_done.wait(5)
                write_completed_during_read = True
        return 0

    connection.set_trace_callback(trace)
    connection.set_progress_handler(coordinate_write, 1)
    try:
        page = database.list_library_media_trash_page(offset=40)
    finally:
        connection.set_progress_handler(None, 0)
        connection.set_trace_callback(None)
        writer_started.set()
        worker.join(5)

    assert not worker.is_alive()
    assert writer_errors == []
    assert write_completed_during_read
    assert page["total"] == 45
    assert len(page["items"]) == 5
    assert "writer-only" in page["types"]
