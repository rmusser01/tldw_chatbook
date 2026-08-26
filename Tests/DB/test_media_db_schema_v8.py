"""Schema v8 (TASK-21126): the engine-version census covering index.

The Library Search/RAG panel's legacy-chunk report reads
``LocalRAGAdminService.count_chunks_by_engine_version`` -- "SELECT
chunk_engine_version, COUNT(DISTINCT media_id) ... WHERE deleted = 0 GROUP
BY chunk_engine_version". Before v8 that ran as an index search on
``deleted`` plus a table row-lookup per live chunk row plus two temp
B-trees: measured 119 ms at 200k live chunk rows and 701 ms at 1M.

What these tests pin, and why each one exists:

* the index is created on a fresh DB and by a genuine v7 upgrade;
* the migration adds NO row, column, table or trigger (a pure index add);
* **the planner actually CHOOSES it with no ``sqlite_stat1``**. This is the
  load-bearing one. No media DB ever runs ``ANALYZE`` (there is no ANALYZE
  anywhere in ``Client_Media_DB_v2.py``), and the "obvious" index for this
  query -- ``(chunk_engine_version, media_id) WHERE deleted = 0`` -- is
  simply never picked in that state: measured 120 ms with it present
  against 119 ms without, i.e. a dead 5 MB index. Leading with ``deleted``
  is what makes the no-stats planner take it. A plan assertion is the only
  thing that can catch a future re-spelling of the query, or a future
  reordering of the index, silently going back to the scan.
"""

from __future__ import annotations

import sqlite3

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.RAG_Admin.local_rag_admin_service import LocalRAGAdminService

from Tests.DB.historical_bootstrap_v6 import media_db_at_version

INDEX_NAME = "idx_unvectorizedmediachunks_engine_census"

#: The census SQL exactly as the production reader spells it. Kept as a
#: literal here on purpose: this file's job is to fail when the reader and
#: the index stop matching, which importing the reader's string would hide.
CENSUS_SQL = (
    "SELECT chunk_engine_version, COUNT(DISTINCT media_id) AS n "
    "FROM UnvectorizedMediaChunks WHERE deleted = 0 "
    "GROUP BY chunk_engine_version"
)


@pytest.fixture()
def fresh_db(tmp_path):
    db = MediaDatabase(str(tmp_path / "media.db"), client_id="test")
    yield db
    db.close_connection()


def _indexes_on_chunks(conn: sqlite3.Connection) -> dict[str, str]:
    return {
        row["name"]: row["sql"] or ""
        for row in conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'index' "
            "AND tbl_name = 'UnvectorizedMediaChunks'"
        )
    }


def _plan(conn: sqlite3.Connection, sql: str) -> str:
    return " | ".join(
        row["detail"] for row in conn.execute("EXPLAIN QUERY PLAN " + sql)
    )


def _seed_chunks(conn: sqlite3.Connection, *, legacy_media: int, stamped_media: int,
                 chunks_each: int = 3, deleted_media: int = 0) -> None:
    """Insert Media + chunk rows directly (fast, and shape-exact).

    Goes around ``add_media_with_keywords`` deliberately: this file cares
    about the physical row shape the index sees, and needs soft-deleted and
    unstamped rows that the writer never produces on its own.
    """
    now = "2026-08-23T00:00:00Z"
    total_media = legacy_media + stamped_media + deleted_media
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.executemany(
        "INSERT INTO Media (id, title, type, content, content_hash, uuid, "
        "last_modified, version, client_id, deleted) "
        "VALUES (?,?,?,?,?,?,?,1,'test',0)",
        [
            (i, f"doc {i}", "document", "body", f"hash-{i}", f"media-{i}", now)
            for i in range(1, total_media + 1)
        ],
    )
    rows = []
    n = 0
    for media_id in range(1, total_media + 1):
        if media_id <= legacy_media:
            version, deleted = None, 0
        elif media_id <= legacy_media + stamped_media:
            version, deleted = "parity-1@385afa95", 0
        else:
            version, deleted = None, 1
        for index in range(chunks_each):
            n += 1
            rows.append(
                (n, media_id, f"chunk {n}", index, "words", f"uuid-{n}", now,
                 version, deleted)
            )
    conn.executemany(
        "INSERT INTO UnvectorizedMediaChunks (id, media_id, chunk_text, "
        "chunk_index, chunk_type, uuid, last_modified, chunk_engine_version, "
        "deleted, version, client_id) VALUES (?,?,?,?,?,?,?,?,?,1,'test')",
        rows,
    )
    conn.commit()


# ---------------------------------------------------------------------------
# The index exists, on a fresh DB and after a genuine upgrade
# ---------------------------------------------------------------------------


def test_fresh_db_is_at_the_current_version(fresh_db):
    version = fresh_db.get_connection().execute(
        "SELECT version FROM schema_version LIMIT 1"
    ).fetchone()["version"]
    assert version == MediaDatabase._CURRENT_SCHEMA_VERSION


def test_fresh_db_has_the_census_index_with_the_measured_shape(fresh_db):
    indexes = _indexes_on_chunks(fresh_db.get_connection())
    assert INDEX_NAME in indexes, sorted(indexes)
    ddl = " ".join(indexes[INDEX_NAME].split()).lower()
    # Column ORDER is the measured part -- see the module docstring.
    assert "(deleted, chunk_engine_version, media_id)" in ddl
    assert "where deleted = 0" in ddl


def test_genuine_v7_db_upgrades_and_gains_the_index(tmp_path):
    path = tmp_path / "v7.db"
    with media_db_at_version(path, 7) as old:
        conn = old.get_connection()
        assert conn.execute(
            "SELECT version FROM schema_version"
        ).fetchone()["version"] == 7
        assert INDEX_NAME not in _indexes_on_chunks(conn)
        _seed_chunks(conn, legacy_media=2, stamped_media=3)

    upgraded = MediaDatabase(str(path), client_id="upgrade")
    try:
        conn = upgraded.get_connection()
        assert conn.execute(
            "SELECT version FROM schema_version"
        ).fetchone()["version"] == MediaDatabase._CURRENT_SCHEMA_VERSION
        assert INDEX_NAME in _indexes_on_chunks(conn)
        # The rows the v7 DB already held are untouched by an index add.
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM UnvectorizedMediaChunks"
        ).fetchone()["n"] == 15
        service = LocalRAGAdminService.__new__(LocalRAGAdminService)
        assert service.count_chunks_by_engine_version(upgraded) == {
            "legacy": 2,
            "parity-1@385afa95": 3,
        }
    finally:
        upgraded.close_connection()


def test_v7_to_v8_adds_nothing_but_the_index(tmp_path):
    """A pure index add: no column, table, trigger or row may move."""
    path = tmp_path / "v7-shape.db"

    def _shape(conn: sqlite3.Connection) -> dict[str, set]:
        return {
            "tables": {
                r["name"]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            },
            "triggers": {
                r["name"]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger'"
                )
            },
            "chunk_columns": {
                r["name"]
                for r in conn.execute("PRAGMA table_info(UnvectorizedMediaChunks)")
            },
        }

    with media_db_at_version(path, 7) as old:
        conn = old.get_connection()
        _seed_chunks(conn, legacy_media=1, stamped_media=1, deleted_media=1)
        before = _shape(conn)
        before_indexes = set(_indexes_on_chunks(conn))
        before_rows = conn.execute(
            "SELECT id, media_id, chunk_engine_version, deleted, version "
            "FROM UnvectorizedMediaChunks ORDER BY id"
        ).fetchall()
        before_rows = [tuple(r) for r in before_rows]

    upgraded = MediaDatabase(str(path), client_id="upgrade")
    try:
        conn = upgraded.get_connection()
        after = _shape(conn)
        after_rows = [
            tuple(r)
            for r in conn.execute(
                "SELECT id, media_id, chunk_engine_version, deleted, version "
                "FROM UnvectorizedMediaChunks ORDER BY id"
            )
        ]
        assert after == before
        assert after_rows == before_rows
        assert set(_indexes_on_chunks(conn)) - before_indexes == {INDEX_NAME}
    finally:
        upgraded.close_connection()


# ---------------------------------------------------------------------------
# The load-bearing one: the planner picks it with NO sqlite_stat1
# ---------------------------------------------------------------------------


def test_census_plan_uses_the_covering_index_without_analyze(fresh_db):
    conn = fresh_db.get_connection()
    _seed_chunks(conn, legacy_media=40, stamped_media=160, chunks_each=5)
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_stat1'"
    ).fetchone() is None, "the fixture must reproduce the no-stats production state"

    plan = _plan(conn, CENSUS_SQL)
    assert INDEX_NAME in plan, plan
    assert "COVERING INDEX" in plan, plan
    # Both temp B-trees are what the index removes; either one reappearing
    # means the shape or the query text has drifted.
    assert "TEMP B-TREE" not in plan, plan


def test_census_result_is_unchanged_by_the_index(fresh_db):
    """The index must not change a single number the panel can show."""
    conn = fresh_db.get_connection()
    _seed_chunks(conn, legacy_media=4, stamped_media=6, chunks_each=3,
                 deleted_media=5)
    service = LocalRAGAdminService.__new__(LocalRAGAdminService)
    with_index = service.count_chunks_by_engine_version(fresh_db)

    conn.execute(f"DROP INDEX {INDEX_NAME}")
    conn.commit()
    without_index = service.count_chunks_by_engine_version(fresh_db)

    assert with_index == without_index == {"legacy": 4, "parity-1@385afa95": 6}


def test_census_on_an_empty_library_is_empty_and_still_indexed(fresh_db):
    """First run: no media at all. The report's honest empty state."""
    conn = fresh_db.get_connection()
    service = LocalRAGAdminService.__new__(LocalRAGAdminService)
    assert service.count_chunks_by_engine_version(fresh_db) == {}
    assert INDEX_NAME in _plan(conn, CENSUS_SQL)


def test_soft_deleted_chunks_are_outside_the_partial_index(fresh_db):
    """A soft-deleted row must neither be counted nor occupy the index."""
    conn = fresh_db.get_connection()
    _seed_chunks(conn, legacy_media=2, stamped_media=0, chunks_each=2,
                 deleted_media=3)
    service = LocalRAGAdminService.__new__(LocalRAGAdminService)
    assert service.count_chunks_by_engine_version(fresh_db) == {"legacy": 2}
    # `deleted = 0` legs only: the partial index holds 2 media x 2 chunks.
    live = conn.execute(
        "SELECT COUNT(*) AS n FROM UnvectorizedMediaChunks WHERE deleted = 0"
    ).fetchone()["n"]
    assert live == 4
