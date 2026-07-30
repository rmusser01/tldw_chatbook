"""Tests for the briefing DB foundation (spec #2 phase 1, task 1).

Covers: the `briefings` / `briefing_items` tables and the two new
`watchlists` columns exist; the coverage-window watermark ignores `failed`
briefings (never advances the window on failure); and the global
`queued_for_briefing` flag survives the write path -> `get_new_items` ->
`normalize_watchlist_item` round trip.
"""

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService
from tldw_chatbook.Subscriptions.watchlist_normalizers import normalize_watchlist_item

pytestmark = pytest.mark.unit


def test_briefings_tables_exist_with_watermark_column():
    db = SubscriptionsDB(":memory:", "test")
    cols = {r[1] for r in db.conn.execute("PRAGMA table_info(briefings)")}
    assert {"watchlist_id", "status", "covers_through_item_id", "body_markdown"} <= cols
    jcols = {r[1] for r in db.conn.execute("PRAGMA table_info(briefing_items)")}
    assert {"briefing_id", "item_id", "featured"} <= jcols
    wcols = {r[1] for r in db.conn.execute("PRAGMA table_info(watchlists)")}
    assert {"briefing_selection_mode", "default_briefing_preset_id"} <= wcols


def test_latest_completed_watermark_ignores_failed_and_interrupted():
    """THE coverage invariant's DB half: failure never advances the window."""
    db = SubscriptionsDB(":memory:", "test")
    # Real watchlist-creation API: watchlists are created through
    # WatchlistBundleService.create(), not a SubscriptionsDB method --
    # SubscriptionsDB has no `create_watchlist`. `.create()` returns a dict;
    # its `id` key is the watchlist id.
    w = WatchlistBundleService(db).create(name="w")["id"]
    b1 = db.insert_briefing(w)
    db.update_briefing(b1, status="complete", covers_through_item_id=40)
    b2 = db.insert_briefing(w)
    db.update_briefing(b2, status="failed", covers_through_item_id=99, error="boom")
    b3 = db.insert_briefing(w)
    db.update_briefing(b3, status="empty", covers_through_item_id=55)
    assert db.latest_completed_watermark(w) == 55  # empty advances; failed never


def test_queue_flag_round_trips_through_the_normalizer():
    """Phase D's read-path lesson: the DB returns the flag; the normalizer
    must carry it, or every downstream consumer sees un-queued items."""
    db = SubscriptionsDB(":memory:", "test")
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    with db.transaction() as conn:
        cursor = conn.execute(
            "INSERT INTO subscription_items (subscription_id, url, title) "
            "VALUES (?, ?, ?)",
            (source_id, "https://a.example/1", "RAG Evaluation"),
        )
        item_id = cursor.lastrowid

    def _fetch_normalized():
        rows = db.get_new_items(subscription_id=source_id, status="new")
        assert len(rows) == 1
        return normalize_watchlist_item("local", rows[0])

    # Default: not queued.
    assert _fetch_normalized()["queued_for_briefing"] is False

    db.set_item_briefing_queued(item_id, True)
    assert _fetch_normalized()["queued_for_briefing"] is True

    db.set_item_briefing_queued(item_id, False)
    assert _fetch_normalized()["queued_for_briefing"] is False
