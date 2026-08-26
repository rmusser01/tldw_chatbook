"""SQLite contracts for the Watchlists Reader's stable item snapshots."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.item_dates import effective_date
from tldw_chatbook.Subscriptions.watchlist_item_page import WatchlistItemCursor


@pytest.fixture
def db(tmp_path: Path):
    owner = SubscriptionsDB(tmp_path / "subscriptions.db")
    try:
        yield owner
    finally:
        owner.close()


def _source(db: SubscriptionsDB, name: str) -> int:
    return db.add_subscription(name=name, type="rss", source=f"https://{name}.test")


def _watchlist(db: SubscriptionsDB, name: str) -> int:
    with db.transaction() as conn:
        return int(conn.execute("INSERT INTO watchlists (name) VALUES (?)", (name,)).lastrowid)


def _link(db: SubscriptionsDB, watchlist_id: int, subscription_id: int) -> None:
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO watchlist_sources (watchlist_id, subscription_id) VALUES (?, ?)",
            (watchlist_id, subscription_id),
        )


def _item(
    db: SubscriptionsDB,
    subscription_id: int,
    slug: str,
    *,
    title: str | None = None,
    content: str = "reader body",
    status: str = "new",
    published: str | None = "2026-08-14T12:00:00Z",
    created: str = "2026-08-14T12:05:00Z",
    run_id: int | None = None,
    flagged: bool = False,
) -> int:
    with db.transaction() as conn:
        return int(
            conn.execute(
                """
                INSERT INTO subscription_items (
                    subscription_id, url, title, content, status, published_date,
                    created_at, run_id, is_flagged
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    subscription_id,
                    f"https://items.test/{slug}",
                    title or slug,
                    content,
                    status,
                    published,
                    created,
                    run_id,
                    int(flagged),
                ),
            ).lastrowid
        )


def _page(db: SubscriptionsDB, **kwargs):
    return db.get_reader_items_page(**kwargs)


def _all_ids(db: SubscriptionsDB, **kwargs) -> list[int]:
    page = _page(db, **kwargs)
    ids = [row["id"] for row in page.items]
    while page.has_more:
        page = _page(
            db,
            **kwargs,
            snapshot_max_item_id=page.snapshot_max_item_id,
            after=page.next_cursor,
        )
        ids.extend(row["id"] for row in page.items)
    return ids


def test_reader_traverses_descending_ties_then_null_date_sink_and_projects_effective_date(
    db: SubscriptionsDB,
) -> None:
    source_id = _source(db, "ties")
    oldest = _item(db, source_id, "oldest", published="2026-01-01T00:00:00Z")
    tied_one = _item(db, source_id, "tied-one", published="2026-01-02T00:00:00Z")
    tied_two = _item(db, source_id, "tied-two", published="2026-01-02T00:00:00Z")
    null_one = _item(db, source_id, "null-one", published="bad", created="also-bad")
    null_two = _item(db, source_id, "null-two", published=None, created="also-bad")

    page = _page(db, limit=2)

    assert [row["id"] for row in page.items] == [tied_two, tied_one]
    assert all("effective_date" in row for row in page.items)
    assert page.next_cursor == WatchlistItemCursor(page.items[-1]["effective_date"], tied_one)
    assert _all_ids(db, limit=2) == [tied_two, tied_one, oldest, null_two, null_one]


def test_reader_first_page_captures_matching_watermark_count_lookahead_and_never_uses_offset(
    db: SubscriptionsDB,
) -> None:
    source_id = _source(db, "snapshot")
    _item(db, source_id, "old", published="2026-01-01T00:00:00Z")
    middle = _item(db, source_id, "middle", published="2026-01-02T00:00:00Z")
    newest = _item(db, source_id, "newest", published="2026-01-03T00:00:00Z")
    _item(db, source_id, "decoy", status="reviewed", published="2026-01-04T00:00:00Z")
    statements: list[str] = []
    db.conn.set_trace_callback(statements.append)
    try:
        page = _page(db, status="new", limit=2)
    finally:
        db.conn.set_trace_callback(None)

    assert page.snapshot_max_item_id == newest
    assert page.snapshot_count == 3
    assert [row["id"] for row in page.items] == [newest, middle]
    assert page.has_more is True
    assert page.next_cursor == WatchlistItemCursor(page.items[-1]["effective_date"], middle)
    assert all("OFFSET" not in statement.upper() for statement in statements)


def test_reader_snapshot_excludes_later_inserts_and_survives_deleted_mounted_row(
    db: SubscriptionsDB,
) -> None:
    source_id = _source(db, "continuation")
    ids = [
        _item(db, source_id, f"item-{day}", published=f"2026-01-0{day}T00:00:00Z")
        for day in range(1, 5)
    ]
    first = _page(db, limit=2)
    late = _item(db, source_id, "late", published="2030-01-01T00:00:00Z")
    with db.transaction() as conn:
        conn.execute("DELETE FROM subscription_items WHERE id = ?", (first.items[0]["id"],))

    second = _page(
        db,
        limit=2,
        snapshot_max_item_id=first.snapshot_max_item_id,
        after=first.next_cursor,
    )

    assert late not in [row["id"] for row in second.items]
    assert [row["id"] for row in second.items] == [ids[1], ids[0]]
    assert second.snapshot_count is None


def test_reader_intersects_every_existing_query_dimension(db: SubscriptionsDB) -> None:
    in_scope = _source(db, "in-scope")
    other = _source(db, "other")
    watchlist_id = _watchlist(db, "scope")
    _link(db, watchlist_id, in_scope)
    wanted = _item(
        db, in_scope, "wanted", title="needle", status="new", run_id=7,
        flagged=True, published="2026-08-15T12:00:00Z",
    )
    _item(db, in_scope, "wrong-status", title="needle", status="ignored", run_id=7, flagged=True, published="2026-08-15T12:00:00Z")
    _item(db, in_scope, "wrong-run", title="needle", status="new", run_id=8, flagged=True, published="2026-08-15T12:00:00Z")
    _item(db, other, "wrong-source", title="needle", status="new", run_id=7, flagged=True, published="2026-08-15T12:00:00Z")

    page = _page(
        db, subscription_id=in_scope, status=None, statuses=["new", "reviewed"],
        run_id=7, watchlist_id=watchlist_id, is_flagged=True, search="needle",
        since="2026-08-15T00:00:00Z",
    )

    assert [row["id"] for row in page.items] == [wanted]
    unassigned = _page(db, unassigned_only=True)
    assert [row["id"] for row in unassigned.items] == [
        row["id"] for row in _page(db, subscription_id=other).items
    ]


def test_reader_validates_inputs_and_empty_first_page_is_safe(db: SubscriptionsDB) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _page(db, limit=0)
    with pytest.raises(ValueError, match="either status or statuses"):
        _page(db, status="new", statuses=["new"])
    with pytest.raises(ValueError, match="positive"):
        _page(db, snapshot_max_item_id=3, after=WatchlistItemCursor(None, 0))
    with pytest.raises(ValueError, match="watermark"):
        _page(db, snapshot_max_item_id=2, after=WatchlistItemCursor(None, 3))
    with pytest.raises(ValueError, match="watermark"):
        _page(db, after=WatchlistItemCursor(None, 1))

    page = _page(db)
    assert page.items == ()
    assert page.snapshot_max_item_id == 0
    assert page.snapshot_count == 0
    assert page.next_cursor is None


def test_reader_fts_and_like_fallback_have_complete_snapshot_and_arrival_parity(
    db: SubscriptionsDB,
) -> None:
    source_id = _source(db, "search")
    _item(db, source_id, "old", title="needle old", published="2026-01-01T00:00:00Z")
    _item(db, source_id, "new", content="needle newest", published="2026-01-02T00:00:00Z")
    _item(db, source_id, "decoy", title="not it", published="2026-01-03T00:00:00Z")
    fts = _page(db, search="needle", limit=1)
    _item(db, source_id, "arrival", title="needle arrival", published="2030-01-01T00:00:00Z")
    fts_arrivals = db.count_reader_item_arrivals(
        search="needle", snapshot_max_item_id=fts.snapshot_max_item_id
    )
    with db.transaction() as conn:
        conn.execute("DROP TABLE subscription_items_fts")
    fallback = _page(
        db,
        search="needle",
        limit=1,
        snapshot_max_item_id=fts.snapshot_max_item_id,
    )
    fallback_arrivals = db.count_reader_item_arrivals(
        search="needle", snapshot_max_item_id=fts.snapshot_max_item_id
    )

    assert fallback.snapshot_max_item_id == fts.snapshot_max_item_id
    assert fallback.snapshot_count == fts.snapshot_count
    assert [row["id"] for row in fallback.items] == [row["id"] for row in fts.items]
    assert fallback.has_more == fts.has_more
    assert fallback.next_cursor == fts.next_cursor
    assert fts_arrivals == fallback_arrivals == 1


def test_reader_arrivals_use_same_scope_and_only_new_rows(db: SubscriptionsDB) -> None:
    source_id = _source(db, "arrivals")
    initial = _item(db, source_id, "initial", status="new")
    first = _page(db, subscription_id=source_id, status="new")
    matching_later = _item(db, source_id, "matching", status="new")
    _item(db, source_id, "out-of-scope", status="reviewed")
    db.mark_item_status(initial, "reviewed")

    assert db.count_reader_item_arrivals(
        subscription_id=source_id, status="new", snapshot_max_item_id=first.snapshot_max_item_id
    ) == 1
    assert matching_later > first.snapshot_max_item_id


def test_reader_sqlite_order_and_cursor_boundaries_match_python_item_dates(
    db: SubscriptionsDB,
) -> None:
    source_id = _source(db, "dates")
    ids = [
        _item(db, source_id, "aware", published="2026-01-02T01:00:00+01:00"),
        _item(db, source_id, "naive", published="2026-01-02T00:00:00"),
        _item(db, source_id, "date-only", published="2026-01-01"),
        _item(db, source_id, "missing", published=None, created="2025-12-31T23:00:00Z"),
        _item(db, source_id, "malformed", published="bad", created="also-bad"),
    ]
    rows = [dict(row) for row in _page(db, limit=10).items]

    def python_key(row: dict) -> tuple[int, datetime, int]:
        date = effective_date(row)
        return (date is not None, date or datetime.min.replace(tzinfo=timezone.utc), row["id"])

    expected = [row["id"] for row in sorted(rows, key=python_key, reverse=True)]
    assert [row["id"] for row in rows] == expected
    assert _all_ids(db, limit=2) == expected
    assert set(expected) == set(ids)
