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


def test_update_briefing_rejects_unknown_field_but_accepts_a_valid_one():
    """Matches the sibling `update_subscription`'s allowlist pattern.

    `update_briefing` builds its SET clause from `**fields`; without an
    allowlist a typo'd or renamed keyword would silently build a query
    against a column that was never meant to be settable this way (or,
    worse, become attacker-reachable). A valid field must still work.
    """
    db = SubscriptionsDB(":memory:", "test")
    w = WatchlistBundleService(db).create(name="w")["id"]
    b = db.insert_briefing(w)

    with pytest.raises(ValueError, match="not_a_real_column"):
        db.update_briefing(b, not_a_real_column="oops")

    db.update_briefing(b, status="complete", body_markdown="hello")
    row = db.get_briefing(b)
    assert row["status"] == "complete"
    assert row["body_markdown"] == "hello"


def test_latest_completed_watermark_is_scoped_per_watchlist():
    """A busy watchlist's completions must never leak into a quiet one's
    watermark -- `latest_completed_watermark` is filtered by watchlist_id,
    not read from every `briefings` row regardless of owner."""
    db = SubscriptionsDB(":memory:", "test")
    busy = WatchlistBundleService(db).create(name="busy")["id"]
    quiet = WatchlistBundleService(db).create(name="quiet")["id"]

    busy_briefing = db.insert_briefing(busy)
    db.update_briefing(busy_briefing, status="complete", covers_through_item_id=500)

    # The quiet watchlist has never had a briefing at all yet.
    assert db.latest_completed_watermark(quiet) is None
    assert db.latest_completed_watermark(busy) == 500

    quiet_briefing = db.insert_briefing(quiet)
    db.update_briefing(quiet_briefing, status="complete", covers_through_item_id=3)

    # Each watchlist reads back only its own watermark.
    assert db.latest_completed_watermark(quiet) == 3
    assert db.latest_completed_watermark(busy) == 500


def test_ensure_watchlists_schema_restores_briefing_columns_on_a_pre_existing_db():
    """Re-arm idiom from `test_watchlist_noise_not_volume.py`'s migration
    tests: an in-memory connection can't be "reopened" to re-trigger
    `BaseDB.__init__`'s migration call, so drop the columns to simulate a
    database that predates this change and invoke the real migration
    method directly."""
    db = SubscriptionsDB(":memory:", "test")
    with db.transaction() as conn:
        conn.execute("ALTER TABLE watchlists DROP COLUMN briefing_selection_mode")
        conn.execute("ALTER TABLE watchlists DROP COLUMN default_briefing_preset_id")

    cols_before = {r[1] for r in db.conn.execute("PRAGMA table_info(watchlists)")}
    assert "briefing_selection_mode" not in cols_before
    assert "default_briefing_preset_id" not in cols_before

    db._ensure_watchlists_schema()

    cols_after = {r[1]: r for r in db.conn.execute("PRAGMA table_info(watchlists)")}
    assert "briefing_selection_mode" in cols_after
    assert "default_briefing_preset_id" in cols_after

    w = WatchlistBundleService(db).create(name="w")["id"]
    row = db.conn.execute(
        "SELECT briefing_selection_mode, default_briefing_preset_id "
        "FROM watchlists WHERE id = ?",
        (w,),
    ).fetchone()
    assert row["briefing_selection_mode"] == "auto_featured"
    assert row["default_briefing_preset_id"] is None


def test_list_briefings_returns_newest_first_by_identity():
    """Insert three out of any timestamp-collision-prone order and assert
    the exact id sequence -- identities, not just a count -- so a query
    that merely returns "three rows" without honoring recency cannot
    pass this by accident."""
    db = SubscriptionsDB(":memory:", "test")
    w = WatchlistBundleService(db).create(name="w")["id"]

    first = db.insert_briefing(w)
    second = db.insert_briefing(w)
    third = db.insert_briefing(w)

    listed = db.list_briefings(w)
    assert [row["id"] for row in listed] == [third, second, first]


def test_get_briefing_returns_none_for_a_missing_id():
    db = SubscriptionsDB(":memory:", "test")
    assert db.get_briefing(999999) is None
