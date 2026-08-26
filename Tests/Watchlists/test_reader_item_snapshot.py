"""Contract tests for the pure reader item snapshot values."""

import pytest

from tldw_chatbook.Subscriptions.watchlist_item_page import (
    WatchlistItemCursor,
    WatchlistItemPage,
)
from tldw_chatbook.UI.Watchlists_Modules.reader_item_snapshot import (
    ReaderItemQuery,
    ReaderItemSnapshot,
)


def page(items, *, watermark=10, count=4, cursor=None, has_more=True):
    return WatchlistItemPage(tuple(items), has_more, watermark, count, cursor)


def test_page_values_are_frozen_and_constructible():
    cursor = WatchlistItemCursor("2026-01-01", 3)
    value = WatchlistItemPage(({"item_id": 3},), True, 10, 1, cursor)
    assert value.items == ({"item_id": 3},)
    assert value.next_cursor == cursor
    with pytest.raises(AttributeError):
        value.has_more = False


def test_query_freeze_commits_statuses_as_tuple_and_as_kwargs_detaches_list():
    statuses = ["new", "reviewed"]
    query = ReaderItemQuery.freeze(("local", "all", "all", ""), {"statuses": statuses, "limit": 20})
    statuses.append("ingested")
    assert query.kwargs == (("limit", 20), ("statuses", ("new", "reviewed")))
    kwargs = query.as_kwargs()
    assert kwargs == {"limit": 20, "statuses": ["new", "reviewed"]}
    kwargs["statuses"].append("changed")
    assert query.as_kwargs()["statuses"] == ["new", "reviewed"]


def test_query_freeze_rejects_mutable_context_and_non_status_values():
    with pytest.raises(TypeError):
        ReaderItemQuery.freeze((["local"],), {})
    with pytest.raises(TypeError):
        ReaderItemQuery.freeze(("local",), {"limit": []})
    with pytest.raises(TypeError):
        ReaderItemQuery.freeze(("local",), {"statuses": ["new", 1]})


def test_start_captures_first_page_and_keeps_empty_page():
    query = ReaderItemQuery.freeze(("local", "all", "all", ""), {"statuses": []})
    snapshot = ReaderItemSnapshot.start(query, page([], watermark=9, count=0, cursor=None, has_more=False))
    assert snapshot.query == query
    assert snapshot.watermark == 9
    assert snapshot.snapshot_count == 0
    assert snapshot.pages == ((),)
    assert snapshot.page_count == 1
    assert snapshot.seen_ids == frozenset()
    assert snapshot.cursor is None
    assert not snapshot.has_more
    assert snapshot.pending_arrivals == 0


def test_page_and_snapshot_detach_source_rows_but_keep_cached_rows_mutable():
    row = {"item_id": 1, "title": "before"}
    first = page([row])
    row["title"] = "after"
    assert first.items[0]["title"] == "before"
    snapshot = ReaderItemSnapshot.start(ReaderItemQuery.freeze(("local",), {}), first)
    first.items[0]["title"] = "page mutation"
    assert snapshot.pages[0][0]["title"] == "before"


def test_start_requires_first_page_count_and_seeds_seen_ids():
    query = ReaderItemQuery.freeze(("local", "all", "all", ""), {})
    with pytest.raises(ValueError):
        ReaderItemSnapshot.start(query, page([], count=None))
    snapshot = ReaderItemSnapshot.start(query, page([{"item_id": "2"}, {"id": 1}], cursor=WatchlistItemCursor(None, 1)))
    assert snapshot.seen_ids == frozenset({1, 2})
    assert snapshot.cursor == WatchlistItemCursor(None, 1)


def test_continuation_stages_copy_and_deduplicates_items():
    query = ReaderItemQuery.freeze(("local", "all", "all", ""), {})
    original = ReaderItemSnapshot.start(query, page([{"item_id": 2}, {"item_id": 1}], cursor=WatchlistItemCursor(None, 1)))
    candidate, appended = original.with_continuation(
        page([{"item_id": 1}, {"id": "0"}], cursor=WatchlistItemCursor(None, 0), has_more=False)
    )
    assert appended
    assert original.page_count == 1
    assert candidate.pages == (({"item_id": 2}, {"item_id": 1}), ({"id": "0"},))
    assert candidate.seen_ids == frozenset({0, 1, 2})
    assert not candidate.has_more
    candidate.pages[0][0]["changed"] = True
    assert "changed" not in original.pages[0][0]
    assert original.cursor == WatchlistItemCursor(None, 1)
    assert original.has_more
    assert original.seen_ids == frozenset({1, 2})


def test_unhashable_identity_is_skipped():
    query = ReaderItemQuery.freeze(("local",), {})
    snapshot = ReaderItemSnapshot.start(query, page([{"item_id": []}, {"id": {"bad": True}}, {"id": "ok"}]))
    assert snapshot.seen_ids == frozenset({"ok"})


def test_identity_falls_back_from_empty_item_id_and_preserves_string_ids():
    query = ReaderItemQuery.freeze(("local", "all", "all", ""), {})
    snapshot = ReaderItemSnapshot.start(query, page([{"item_id": "", "id": "external-a"}, {"id": "external-b"}]))
    assert snapshot.seen_ids == frozenset({"external-a", "external-b"})
    assert snapshot.pages[0] == ({"item_id": "", "id": "external-a"}, {"id": "external-b"})


def test_numeric_string_identity_is_canonical_across_item_id_and_id_fields():
    query = ReaderItemQuery.freeze(("local", "all", "all", ""), {})
    snapshot = ReaderItemSnapshot.start(query, page([{"item_id": "2"}]))
    candidate, appended = snapshot.with_continuation(page([{"item_id": "", "id": "2"}]))
    assert not appended
    assert candidate.page_count == 1
    assert candidate.seen_ids == frozenset({2})


def test_duplicate_only_continuation_advances_traversal_without_blank_page():
    query = ReaderItemQuery.freeze(("local", "all", "all", ""), {})
    original = ReaderItemSnapshot.start(query, page([{"item_id": 2}], cursor=WatchlistItemCursor(None, 2)))
    candidate, appended = original.with_continuation(
        page([{"id": 2}], cursor=WatchlistItemCursor(None, 1), has_more=True)
    )
    assert not appended
    assert candidate.page_count == 1
    assert candidate.cursor == WatchlistItemCursor(None, 1)
    assert candidate.has_more


def test_continuation_rejects_different_watermark_and_empty_identity():
    query = ReaderItemQuery.freeze(("local", "all", "all", ""), {})
    snapshot = ReaderItemSnapshot.start(query, page([{"item_id": 1}], watermark=10))
    with pytest.raises(ValueError):
        snapshot.with_continuation(page([], watermark=11))
    snapshot = ReaderItemSnapshot.start(query, page([{"title": "missing id"}]))
    assert snapshot.pages == ((),)


def test_page_accessors_are_bounds_checked_and_has_next_uses_cache_or_traversal():
    query = ReaderItemQuery.freeze(("local", "all", "all", ""), {})
    snapshot = ReaderItemSnapshot.start(query, page([{"id": 1}], cursor=WatchlistItemCursor(None, 1), has_more=True))
    with pytest.raises(IndexError):
        snapshot.page(1)
    with pytest.raises(IndexError):
        snapshot.page(-1)
    with pytest.raises(IndexError):
        snapshot.page(snapshot.page_count)
    with pytest.raises(IndexError):
        snapshot.has_next(-1)
    assert snapshot.page(0) == ({"id": 1},)
    assert snapshot.has_next(0)
    candidate, _ = snapshot.with_continuation(page([{"id": 2}], cursor=None, has_more=False))
    assert candidate.has_next(0)
    assert not candidate.has_next(1)


def test_empty_identity_is_not_admitted_on_continuation():
    query = ReaderItemQuery.freeze(("local", "all", "all", ""), {})
    snapshot = ReaderItemSnapshot.start(query, page([{"id": 1}]))
    candidate, appended = snapshot.with_continuation(page([{"item_id": ""}]))
    assert not appended
    assert candidate.pages == snapshot.pages
