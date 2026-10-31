"""B7: browse ordering is index-served (no temp B-tree per page).

The Library's conversation browse pages run
``ORDER BY last_modified DESC, id DESC LIMIT ? OFFSET ?``
(``search_conversations_page``, ``locate_conversation_page``,
``list_all_active_conversations``). Without sort-serving indexes every page
materialized a TEMP B-TREE over the full filtered set just to take the first
``limit`` rows. This pins sort-free page plans on BOTH a fresh bootstrap and
a chain-migrated DB (the versioned-migration reach the repo uses for index
additions -- see ``Tests/ChaChaNotesDB/test_index_census.py``), per scope:

* scoped browses (every page query carries the archive clause) are served by
  ``idx_conversations_archived_browse_order`` (archived equality prefix puts
  the remaining index order exactly on last_modified DESC, id DESC);
* unscoped browses are served by ``idx_conversations_last_modified``.

EXPLAIN evidence for the shape (5k-row fixture, ADR-216): planner choice
without these indexes was idx_conversations_archive + TEMP B-TREE at
~3.1 ms/query; index-served plans run ~0.3-0.6 ms/query.
"""

import sqlite3

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import (
    MINIMUM_BOOTSTRAP_VERSION,
    chachanotes_db_at_version,
    open_current_chachanotes_from_legacy,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

#: Page-query shapes per ``_conversation_search_filter`` scope clause.
_SCOPED_PAGE_SQL = (
    "SELECT * FROM conversations "
    "WHERE archived = 0 "
    "ORDER BY last_modified DESC, id DESC LIMIT 20 OFFSET 0"
)
_UNSCOPED_PAGE_SQL = (
    "SELECT * FROM conversations "
    "ORDER BY last_modified DESC, id DESC LIMIT 20 OFFSET 0"
)


def _plan_lines(conn: sqlite3.Connection, sql: str) -> list[str]:
    return [
        str(row[-1])
        for row in conn.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()
    ]


def _assert_no_order_sort(lines: list[str], context: str) -> None:
    joined = "\n".join(lines).upper()
    assert "TEMP B-TREE FOR ORDER BY" not in joined, f"{context}:\n{joined}"


def _assert_scopes_index_served(conn: sqlite3.Connection) -> None:
    scoped = _plan_lines(conn, _SCOPED_PAGE_SQL)
    _assert_no_order_sort(scoped, "scoped browse page")
    assert "IDX_CONVERSATIONS_ARCHIVED_BROWSE_ORDER" in "\n".join(scoped).upper(), (
        scoped
    )
    unscoped = _plan_lines(conn, _UNSCOPED_PAGE_SQL)
    _assert_no_order_sort(unscoped, "unscoped browse page")
    assert "IDX_CONVERSATIONS_LAST_MODIFIED" in "\n".join(unscoped).upper(), (
        unscoped
    )


def test_fresh_bootstrap_browse_order_uses_index():
    db = CharactersRAGDB(":memory:", client_id="b7-browse-index")
    try:
        _assert_scopes_index_served(db.get_connection())
    finally:
        db.close_connection()


def test_chain_migrated_db_reaches_browse_order_index(tmp_path):
    """A v4-era DB reopened today must gain the indexes through the chain."""
    db_path = tmp_path / "b7_chain.sqlite"
    with chachanotes_db_at_version(db_path, MINIMUM_BOOTSTRAP_VERSION):
        pass  # bootstrap a genuinely-v4 DB, then close it
    db = open_current_chachanotes_from_legacy(
        db_path, client_id="b7-browse-index-chain"
    )
    try:
        _assert_scopes_index_served(db.get_connection())
    finally:
        db.close_connection()


def _seed_conversations(conn: sqlite3.Connection, count: int) -> None:
    for i in range(count):
        conn.execute(
            """
            INSERT INTO conversations (
                id, root_id, title, created_at, last_modified,
                deleted, archived, client_id, version
            ) VALUES (?, ?, ?, ?, ?, 0, 0, 'b7', 1)
            """,
            (f"conv-{i}", f"conv-{i}", f"t{i}",
             f"2026-01-01T00:00:00", f"2026-02-01T00:00:{i:02d}"),
        )


def test_browse_page_actually_returns_rows_in_order():
    """The index serves real pages: rows come back newest-first."""
    db = CharactersRAGDB(":memory:", client_id="b7-browse-order")
    try:
        conn = db.get_connection()
        _seed_conversations(conn, 5)
        rows = conn.execute(_SCOPED_PAGE_SQL).fetchall()
        assert [r["id"] for r in rows] == [
            "conv-4", "conv-3", "conv-2", "conv-1", "conv-0"
        ]
    finally:
        db.close_connection()


def test_populated_browse_pages_stay_sort_free():
    """With rows and ANALYZE stats, page plans must still be sort-free."""
    db = CharactersRAGDB(":memory:", client_id="b7-browse-populated")
    try:
        conn = db.get_connection()
        rows = []
        for i in range(500):
            ts = f"2026-03-01T{i // 3600:02d}:{(i // 60) % 60:02d}:{i % 60:02d}"
            rows.append((f"conv-{i:04d}", f"conv-{i:04d}", f"t{i}", ts, ts))
        conn.executemany(
            """
            INSERT INTO conversations (
                id, root_id, title, created_at, last_modified,
                deleted, archived, client_id, version
            ) VALUES (?, ?, ?, ?, ?, 0, 0, 'b7', 1)
            """,
            rows,
        )
        conn.execute("ANALYZE")
        _assert_scopes_index_served(conn)
    finally:
        db.close_connection()
