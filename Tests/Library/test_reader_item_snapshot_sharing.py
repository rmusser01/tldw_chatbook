"""Structural sharing regressions for ReaderItemSnapshot cached pages.

Task 13 of the nonconsole efficiency remediation: page turns must share
already-cached page objects (zero copies), cached rows must be frozen at
the construction boundary, and the committed-row patch lane used by the
Watchlists screen must rebuild rows without breaking the snapshot object
identity that in-flight page loads capture.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

import tldw_chatbook.UI.Watchlists_Modules.reader_item_snapshot as reader_snapshot
from tldw_chatbook.Subscriptions.watchlist_item_page import (
    WatchlistItemCursor,
    WatchlistItemPage,
)
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WatchlistsCollectionsScreen,
)
from tldw_chatbook.UI.Watchlists_Modules.reader_item_snapshot import (
    ReaderItemQuery,
    ReaderItemSnapshot,
)

PAGE_SIZE = 4


def _row(index: int, *, status: str = "new") -> dict[str, Any]:
    effective = f"2026-08-13 12:{index % 60:02d}:00"
    return {
        "id": f"local:watchlist_item:{index}",
        "item_id": index,
        "title": f"Item {index}",
        "source_name": "Sharing feed",
        "status": status,
        "created_at": effective,
        "effective_date": effective,
        "content": f"Body {index}",
    }


def _page(
    indexes: list[int], *, watermark: int, count: int | None = None, has_more: bool = True
) -> WatchlistItemPage:
    rows = tuple(_row(index) for index in indexes)
    last = rows[-1] if rows else None
    cursor = (
        WatchlistItemCursor(str(last["effective_date"]), int(last["item_id"]))
        if has_more and last is not None
        else None
    )
    return WatchlistItemPage(
        items=rows,
        has_more=has_more,
        snapshot_max_item_id=watermark,
        snapshot_count=count,
        next_cursor=cursor,
    )


def _browse(pages: list[WatchlistItemPage], *, page_size: int | None = None) -> ReaderItemSnapshot:
    query = ReaderItemQuery.freeze(("local", "all", "all", ""), {})
    snapshot = ReaderItemSnapshot.start(query, pages[0])
    for page in pages[1:]:
        snapshot, appended = snapshot.with_continuation(page, page_size=page_size)
        assert appended
    return snapshot


def _test_snapshot() -> ReaderItemSnapshot:
    return _browse(
        [
            _page([8, 7, 6, 5], watermark=8, count=8),
            _page([4, 3, 2, 1], watermark=8),
        ]
    )


def test_page_turns_share_cached_page_and_row_objects() -> None:
    pages = [
        _page([40, 39, 38, 37], watermark=40, count=40),
        _page([36, 35, 34, 33], watermark=40),
        _page([32, 31, 30, 29], watermark=40),
        _page([28, 27, 26, 25], watermark=40),
    ]
    query = ReaderItemQuery.freeze(("local", "all", "all", ""), {})
    snapshot = ReaderItemSnapshot.start(query, pages[0])
    first_page_tuple = snapshot.pages[0]
    first_row_ids = [id(row) for row in snapshot.pages[0]]
    for page in pages[1:]:
        snapshot, appended = snapshot.with_continuation(page)
        assert appended
    assert snapshot.page_count == 4
    # The page-1 tuple and every row on it are the SAME objects that were
    # admitted when page 1 was first cached.
    assert snapshot.pages[0] is first_page_tuple
    assert [id(row) for row in snapshot.pages[0]] == first_row_ids
    # Earlier turns keep sharing too.
    assert snapshot.pages[1] is not first_page_tuple


def test_paging_performs_zero_deepcopies(monkeypatch: pytest.MonkeyPatch) -> None:
    # Build every service page BEFORE counting so the counter only sees the
    # snapshot module's own copying behaviour.
    pages = [
        _page([80 - 4 * i + k for k in range(4)], watermark=80, count=80)
        for i in range(6)
    ]
    original_deepcopy = copy.deepcopy
    calls = {"module": 0, "copy": 0}

    def counting(value: Any, memo: Any = None) -> Any:
        if memo is None:
            calls["copy"] += 1
        return original_deepcopy(value) if memo is None else original_deepcopy(value, memo)

    monkeypatch.setattr(copy, "deepcopy", counting)
    monkeypatch.setattr(
        reader_snapshot, "deepcopy", counting, raising=False
    )
    snapshot = _browse(pages)
    assert snapshot.page_count == 6
    assert calls["module"] == 0
    assert calls["copy"] == 0


def test_cached_rows_reject_in_place_writes() -> None:
    snapshot = _test_snapshot()
    with pytest.raises(TypeError):
        snapshot.pages[0][0]["status"] = "reviewed"
    # mappingproxy exposes no mutating methods at all.
    with pytest.raises(AttributeError):
        snapshot.page(1)[0].update({"status": "reviewed"})
    pending_source = ReaderItemSnapshot.start(
        ReaderItemQuery.freeze(("local",), {}),
        _page([9, 8], watermark=9, count=2),
    )
    staged = pending_source.with_pending_items((_row(7),))
    with pytest.raises(TypeError):
        staged.pending_items[0]["status"] = "reviewed"


def test_paging_content_parity_across_turns() -> None:
    # Golden multi-page browse: cached-page navigation returns exactly the
    # service-ordered rows each turn admitted, duplicates are dropped, and
    # the final pending flush publishes the displaced tail in order.
    pages = [
        _page([12, 11, 10, 9], watermark=12, count=12),
        _page([9, 8, 7, 6], watermark=12),  # 9 is a duplicate of page 1
        _page([5, 4, 3, 2], watermark=12, has_more=False),
    ]
    query = ReaderItemQuery.freeze(("local", "all", "all", ""), {})
    snapshot = ReaderItemSnapshot.start(query, pages[0])
    snapshot, appended = snapshot.with_continuation(pages[1], page_size=3)
    assert appended
    assert list(snapshot.page(1)) == [_row(8), _row(7), _row(6)]
    # The displaced duplicate-free tail is staged, not lost.
    assert tuple(snapshot.pending_items) == ()
    snapshot, appended = snapshot.with_continuation(pages[2], page_size=3)
    assert appended
    assert list(snapshot.page(2)) == [_row(5), _row(4), _row(3)]
    assert tuple(snapshot.pending_items) == (_row(2),)
    snapshot, appended = snapshot.with_pending_page(3)
    assert appended
    assert list(snapshot.page(3)) == [_row(2)]
    assert snapshot.pending_items == ()
    # Navigating back to cached pages returns the same content.
    assert list(snapshot.page(0)) == [_row(12), _row(11), _row(10), _row(9)]
    assert list(snapshot.page(1)) == [_row(8), _row(7), _row(6)]
    assert snapshot.seen_ids == frozenset({2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
    assert snapshot.page_count == 4


def test_pending_rows_are_shared_into_visible_pages() -> None:
    snapshot = ReaderItemSnapshot.start(
        ReaderItemQuery.freeze(("local",), {}),
        _page([6, 5, 4, 3], watermark=6, count=4),
    )
    staged = snapshot.with_pending_items((_row(2), _row(1)))
    staged_row_ids = [id(row) for row in staged.pending_items]
    published, appended = staged.with_pending_page(2)
    assert appended
    assert [id(row) for row in published.page(1)] == staged_row_ids
    assert published.pending_items == ()


def test_close_to_cached_pages_shares_page_objects() -> None:
    snapshot = _test_snapshot()
    closed = snapshot.close_to_cached_pages()
    assert closed.snapshot_count == 8
    assert closed.pages is snapshot.pages
    assert closed.pending_items == ()
    assert closed.cursor is None
    assert not closed.has_more


def test_patch_cached_rows_rebuilds_matches_and_keeps_snapshot_identity() -> None:
    snapshot = _test_snapshot()
    untouched_page = snapshot.pages[0]
    untouched_row = snapshot.pages[0][0]
    snapshot_id = id(snapshot)

    row_key = snapshot.patch_cached_rows(
        lambda row: row.get("item_id") == 3, {"status": "reviewed"}
    )

    assert row_key == "local:watchlist_item:3"
    assert id(snapshot) == snapshot_id
    assert snapshot.pages[0] is untouched_page
    assert snapshot.pages[0][0] is untouched_row
    # Page 1 is [4, 3, 2, 1]; item 3 sits at index 1.
    assert snapshot.page(1)[1]["status"] == "reviewed"
    assert snapshot.page(1)[1]["item_id"] == 3
    with pytest.raises(TypeError):
        snapshot.page(1)[1]["status"] = "new"
    # Idempotent re-patch keeps working and stays a no-op on other rows.
    assert (
        snapshot.patch_cached_rows(
            lambda row: row.get("item_id") == 3, {"status": "new"}
        )
        == "local:watchlist_item:3"
    )
    assert snapshot.page(1)[1]["status"] == "new"


def test_patch_cached_rows_without_match_returns_none_and_keeps_pages() -> None:
    snapshot = _test_snapshot()
    pages_before = snapshot.pages
    assert snapshot.patch_cached_rows(lambda row: False, {"status": "gone"}) is None
    assert snapshot.pages is pages_before


def _bare_screen() -> WatchlistsCollectionsScreen:
    from unittest.mock import Mock

    return WatchlistsCollectionsScreen(Mock())


def test_status_patch_updates_every_projection_and_keeps_snapshot_identity() -> None:
    screen = _bare_screen()
    snapshot = _test_snapshot()
    screen._items_snapshot = snapshot
    screen._loaded_items = [dict(row) for row in snapshot.page(1)]
    screen._selected_content_item = screen._loaded_items[1]
    # Row 3 sits on cached page 1 (index 1), mirroring the pagination
    # regression that requires one mutation path to patch every projection.
    screen._patch_committed_items_after_mutation(3, status="reviewed")

    assert screen._items_snapshot is snapshot
    assert screen._items_snapshot.page(1)[1]["status"] == "reviewed"
    assert screen._loaded_items[1]["status"] == "reviewed"
    assert screen._selected_content_item["status"] == "reviewed"
    # Non-matching rows on the same page are untouched.
    assert screen._items_snapshot.page(1)[0]["status"] == "new"


def test_queue_flag_patch_reaches_cached_snapshot_rows() -> None:
    screen = _bare_screen()
    snapshot = _test_snapshot()
    screen._items_snapshot = snapshot
    screen._loaded_items = [dict(row) for row in snapshot.page(0)]

    screen._patch_item_queued_flag(7, True)

    assert screen._loaded_items[1]["queued_for_briefing"] is True
    assert screen._items_snapshot.page(0)[1]["queued_for_briefing"] is True
    assert "queued_for_briefing" not in screen._items_snapshot.page(0)[2]


def test_published_rows_are_mutable_copies_detached_from_the_cache() -> None:
    screen = _bare_screen()
    snapshot = _test_snapshot()
    screen._items_snapshot = snapshot
    # The publish boundary hands the pane plain dict copies (the
    # TASK-15464 content backfill mutates the selected row in place).
    screen._loaded_items = [dict(row) for row in snapshot.page(0)]
    screen._loaded_items[0]["content"] = "Fetched stale body"

    assert screen._loaded_items[0]["content"] == "Fetched stale body"
    assert snapshot.page(0)[0]["content"] == "Body 8"
