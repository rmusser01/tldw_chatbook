"""Focused database-contract tests for Library Media pagination."""

from __future__ import annotations

import sqlite3
import threading

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase


@pytest.fixture
def media_db(tmp_path):
    database_path = tmp_path / "media.db"
    database = MediaDatabase(db_path=database_path, client_id="pagination-test")
    yield database, database_path
    database.close_connection()


def _seed_media(database: MediaDatabase, count: int = 45) -> list[int]:
    media_ids = []
    for index in range(count):
        media_id, _uuid, _message = database.add_media_with_keywords(
            title="Same title",
            media_type=("article", "audio", "video")[index % 3],
            content=f"same searchable needle unique-{index:03}",
            ingestion_date="2026-08-16T12:00:00+00:00",
            keywords=[],
        )
        assert media_id is not None
        media_ids.append(media_id)

    connection = database.get_connection()
    connection.execute(
        "UPDATE Media SET last_modified = ?, version = version + 1",
        ("2026-08-16T12:00:00+00:00",),
    )
    connection.commit()
    return media_ids


def test_exact_offsets_return_20_20_5_summary_rows_and_stable_ids(media_db):
    database, _path = media_db
    media_ids = _seed_media(database)

    pages = [
        database.search_media_db(
            None,
            results_per_page=20,
            offset=offset,
            library_summary=True,
        )
        for offset in (0, 20, 40)
    ]

    assert [len(rows) for rows, _total in pages] == [20, 20, 5]
    assert [total for _rows, total in pages] == [45, 45, 45]
    rows = [row for page, _total in pages for row in page]
    assert [row["id"] for row in rows] == list(reversed(media_ids))
    assert all(set(row) == {"id", "title", "type", "last_modified"} for row in rows)
    assert all(type(row["id"]) is int and row["id"] > 0 for row in rows)
    assert len({row["id"] for row in rows}) == 45


def test_omitted_offset_keeps_legacy_page_coordinates_and_broad_rows(media_db):
    database, _path = media_db
    media_ids = _seed_media(database)

    rows, total = database.search_media_db(None, page=3, results_per_page=20)

    assert total == 45
    assert [row["id"] for row in rows] == list(reversed(media_ids))[40:]
    assert "uuid" in rows[0]


def test_filters_apply_before_count_and_offset(media_db):
    database, _path = media_db
    media_ids = _seed_media(database)
    expected = [media_id for index, media_id in enumerate(media_ids) if index % 3 == 0]

    rows, total = database.search_media_db(
        None,
        media_types=["article"],
        results_per_page=10,
        offset=10,
        library_summary=True,
    )

    assert total == 15
    assert [row["id"] for row in rows] == list(reversed(expected))[10:]


@pytest.mark.parametrize(
    ("sort_by", "expected_ids"),
    [
        ("date_desc", "descending"),
        ("date_asc", "ascending"),
        ("title_asc", "ascending"),
        ("title_desc", "descending"),
        ("last_modified_asc", "ascending"),
        ("last_modified_desc", "descending"),
        ("relevance", "descending"),
    ],
)
def test_supported_sorts_end_with_stable_id_tie_breaker(
    media_db, sort_by, expected_ids
):
    database, _path = media_db
    media_ids = _seed_media(database, 6)

    rows, total = database.search_media_db(
        "needle" if sort_by == "relevance" else None,
        sort_by=sort_by,
        results_per_page=20,
        offset=0,
        library_summary=True,
    )

    assert total == 6
    expected = media_ids if expected_ids == "ascending" else list(reversed(media_ids))
    assert [row["id"] for row in rows] == expected
    assert all(set(row) == {"id", "title", "type", "last_modified"} for row in rows)


def test_offset_is_bound_directly_without_prefix_fetch(media_db):
    database, _path = media_db
    _seed_media(database)
    statements: list[str] = []
    connection = database.get_connection()
    connection.set_trace_callback(statements.append)
    try:
        rows, total = database.search_media_db(
            None,
            results_per_page=20,
            offset=40,
            library_summary=True,
        )
    finally:
        connection.set_trace_callback(None)

    assert total == 45
    assert len(rows) == 5
    page_selects = [sql for sql in statements if " LIMIT 20 OFFSET 40" in sql]
    assert len(page_selects) == 1
    for private_column in (
        "m.content",
        "m.url",
        "m.author",
        "m.transcription_model",
        "m.transcription_provenance_json",
        "m.content_hash",
        "m.client_id",
    ):
        assert private_column not in page_selects[0]
    assert not any("LIMIT 60 OFFSET 0" in sql for sql in statements)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"results_per_page": True},
        {"results_per_page": 0},
        {"results_per_page": 2**63},
        {"page": True},
        {"page": 0},
        {"page": 2**63, "results_per_page": 2},
        {"offset": True},
        {"offset": -1},
        {"offset": 2**63},
    ],
)
def test_invalid_pagination_values_fail_before_sql(media_db, kwargs):
    database, _path = media_db
    statements: list[str] = []
    connection = database.get_connection()
    connection.set_trace_callback(statements.append)
    try:
        with pytest.raises(ValueError):
            database.search_media_db(None, **kwargs)
    finally:
        connection.set_trace_callback(None)

    assert statements == []


def test_count_and_page_share_one_read_snapshot_during_wal_write(media_db):
    database, database_path = media_db
    media_ids = _seed_media(database)
    writer_started = threading.Event()
    writer_done = threading.Event()
    writer_errors: list[Exception] = []

    def writer() -> None:
        try:
            with sqlite3.connect(database_path, timeout=5) as connection:
                assert writer_started.wait(5)
                connection.execute(
                    "UPDATE Media SET deleted = 1, version = version + 1 WHERE id = ?",
                    (media_ids[-1],),
                )
        except Exception as error:  # surfaced in the test thread below
            writer_errors.append(error)
        finally:
            writer_done.set()

    worker = threading.Thread(target=writer)
    worker.start()
    connection = database.get_connection()
    count_started = False
    progress_calls = 0
    write_completed_during_count = False

    def trace(sql: str) -> None:
        nonlocal count_started
        if sql.lstrip().startswith("SELECT COUNT(DISTINCT m.id)"):
            count_started = True

    def coordinate_write() -> int:
        nonlocal progress_calls, write_completed_during_count
        if count_started and not writer_started.is_set():
            progress_calls += 1
            if progress_calls >= 20:
                writer_started.set()
                assert writer_done.wait(5)
                write_completed_during_count = True
        return 0

    connection.set_trace_callback(trace)
    connection.set_progress_handler(coordinate_write, 1)
    try:
        rows, total = database.search_media_db(
            None,
            results_per_page=20,
            offset=40,
            library_summary=True,
        )
    finally:
        connection.set_progress_handler(None, 0)
        connection.set_trace_callback(None)
        writer_started.set()
        worker.join(5)

    assert not worker.is_alive()
    assert writer_errors == []
    assert write_completed_during_count
    assert total == 45
    assert len(rows) == 5


def test_distinct_types_are_complete_for_active_media_only(media_db):
    database, _path = media_db
    _seed_media(database, 6)
    trashed_id, _uuid, _message = database.add_media_with_keywords(
        title="Trashed", media_type="private-trashed-type", content="trashed"
    )
    deleted_id, _uuid, _message = database.add_media_with_keywords(
        title="Deleted", media_type="private-deleted-type", content="deleted"
    )
    assert trashed_id is not None and deleted_id is not None
    connection = database.get_connection()
    connection.execute(
        "UPDATE Media SET is_trash = 1, version = version + 1 WHERE id = ?",
        (trashed_id,),
    )
    connection.execute(
        "UPDATE Media SET deleted = 1, version = version + 1 WHERE id = ?",
        (deleted_id,),
    )
    connection.commit()

    assert database.get_distinct_media_types() == ["article", "audio", "video"]


def test_distinct_types_exclude_whitespace_only_and_preserve_nonblank_verbatim(
    media_db,
):
    database, _path = media_db
    for index, media_type in enumerate(
        ("   ", "\t", "\n", "\N{NO-BREAK SPACE}", " pdf ")
    ):
        media_id, _uuid, _message = database.add_media_with_keywords(
            title=f"Type edge {index}",
            media_type=media_type,
            content=f"type-edge-{index}",
            keywords=[],
        )
        assert media_id is not None

    assert database.get_distinct_media_types() == [" pdf "]
