"""Tests for the briefing DB foundation and item selection (spec #2 phase 1).

Task 1 (DB): the `briefings` / `briefing_items` tables and the two new
`watchlists` columns exist; the coverage-window watermark ignores `failed`
briefings (never advances the window on failure); and the global
`queued_for_briefing` flag survives the write path -> `get_new_items` ->
`normalize_watchlist_item` round trip.

Task 2 (selection): `select_briefing_items` -- the id-watermark window, the
three modes, the junction exclusion scoped to one watchlist, and the item
cap's overflow accounting. Everything is seeded through the real database
(`WatchlistBundleService.create`, `add_subscription`, `watchlist_sources`,
`persist_subscription_item`) rather than through fakes, because the rules
under test are *about* what the SQL sees.
"""

from datetime import datetime, timedelta, timezone

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions import briefing_selection
from tldw_chatbook.Subscriptions.briefing_selection import (
    BriefingSelection,
    select_briefing_items,
)
from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item
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


# --- Task 2: selection ------------------------------------------------------
#
# Seeding helpers. Item ids come from the real AUTOINCREMENT sequence, so the
# ORDER in which `_add_item` is called is the id order -- which is exactly the
# property the watermark relies on and several tests below deliberately set
# against `created_at`.


def _new_source(db, watchlist_id, name):
    """Add a subscription and attach it to a watchlist."""
    source_id = db.add_subscription(
        name=name, type="rss", source=f"https://{name}.example/feed.xml"
    )
    WatchlistBundleService(db).add_source(watchlist_id, source_id)
    return source_id


def _detached_source(db, name):
    """Add a subscription that is not (yet) in any watchlist."""
    return db.add_subscription(
        name=name, type="rss", source=f"https://{name}.example/feed.xml"
    )


def _add_item(db, source_id, title, created_at, *, queued=False):
    """Insert one item through the real persist path; return its id."""
    slug = title.lower().replace(" ", "-")
    with db.transaction() as conn:
        item_id = persist_subscription_item(
            conn,
            source_id,
            {
                "url": f"https://items.example/{source_id}/{slug}",
                "title": title,
                "content": f"body of {title}",
                "content_hash": f"hash-{source_id}-{slug}",
                "content_kind": "article",
                "content_format": "text",
            },
            run_id=None,
            now=created_at,
        )
    if queued:
        db.set_item_briefing_queued(item_id, True)
    return item_id


def _complete_briefing(db, watchlist_id, covers_through_item_id, *, item_ids=()):
    """Record a completed briefing (and, optionally, its junction rows).

    The junction write lives in the test rather than in the module under
    test on purpose: `briefing_selection` is read-only -- the service (task
    3) owns every write -- so a test that needs coverage to already exist
    has to create it the way the service will.
    """
    briefing_id = db.insert_briefing(watchlist_id)
    db.update_briefing(
        briefing_id, status="complete", covers_through_item_id=covers_through_item_id
    )
    with db.transaction() as conn:
        for item_id in item_ids:
            conn.execute(
                "INSERT INTO briefing_items (briefing_id, item_id, featured) "
                "VALUES (?, ?, 0)",
                (briefing_id, item_id),
            )
    return briefing_id


def _ids(selection):
    """Item ids of a selection, in selection order."""
    return [item["item_id"] for item in selection.items]


def test_watermark_window_excludes_a_late_added_sources_backlog():
    """The id watermark's free flood-fix: a source added after briefing 1 has
    historical items with ids below the watermark -- auto-excluded.

    The `created_at` values are set deliberately AGAINST the id order: the
    backlog rows are the newest by timestamp and the oldest by id. A window
    built on timestamps would drag the whole backlog into briefing 2; the id
    watermark cannot, which is the entire reason the spec chose ids.
    """
    db = SubscriptionsDB(":memory:", "test")
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]

    # The late-added source's backlog exists FIRST (low ids) but carries the
    # NEWEST timestamps.
    backlog_source = _detached_source(db, "backlog")
    backlog = [
        _add_item(db, backlog_source, f"Backlog {n}", "2026-07-29T23:00:00+00:00")
        for n in range(3)
    ]

    # The watchlist's original source has older timestamps and higher ids.
    original = _new_source(db, watchlist, "original")
    covered = _add_item(db, original, "Covered item", "2026-07-20T09:00:00+00:00")
    _complete_briefing(db, watchlist, covered, item_ids=[covered])

    # Now the backlog source joins the watchlist, and one genuinely new item
    # arrives on the original source.
    WatchlistBundleService(db).add_source(watchlist, backlog_source)
    fresh = _add_item(db, original, "Fresh item", "2026-07-21T09:00:00+00:00")

    selection = select_briefing_items(db, watchlist, mode="auto")

    assert _ids(selection) == [fresh]
    assert not set(backlog) & set(_ids(selection))
    assert covered not in _ids(selection)
    assert selection.covers_through_item_id == fresh
    assert selection.overflow_count == 0


def test_failed_briefing_does_not_advance_selection():
    """Generate window A -> complete at watermark X. Insert items. A failed
    briefing row with a HIGHER covers_through_item_id must not move the next
    selection: it still starts at X.
    """
    db = SubscriptionsDB(":memory:", "test")
    watchlist = WatchlistBundleService(db).create(name="Infra")["id"]
    source = _new_source(db, watchlist, "infra")

    window_a = [
        _add_item(db, source, f"A{n}", "2026-07-20T09:00:00+00:00") for n in range(2)
    ]
    watermark = max(window_a)
    _complete_briefing(db, watchlist, watermark, item_ids=window_a)

    after = [
        _add_item(db, source, f"B{n}", "2026-07-21T09:00:00+00:00") for n in range(3)
    ]

    # A failed attempt claims a watermark past everything -- and must not be
    # believed. `latest_completed_watermark` excludes 'failed'; this test
    # pins that the SELECTION honours that exclusion end to end.
    failed = db.insert_briefing(watchlist)
    db.update_briefing(
        failed, status="failed", covers_through_item_id=max(after), error="boom"
    )

    selection = select_briefing_items(db, watchlist, mode="auto")

    assert sorted(_ids(selection)) == sorted(after)
    assert selection.covers_through_item_id == max(after)
    # The retry re-covers the same window: nothing from window A leaks back
    # in, and nothing after the watermark was lost by the failure.
    assert not set(window_a) & set(_ids(selection))


def test_queued_items_bypass_the_window_in_both_modes():
    """A queued item OLDER than the watermark appears in curated AND
    auto_featured selections, marked featured in the latter; it does not
    appear in plain auto.
    """
    db = SubscriptionsDB(":memory:", "test")
    watchlist = WatchlistBundleService(db).create(name="Papers")["id"]
    source = _new_source(db, watchlist, "papers")

    # Queued three weeks ago, and already below the watermark -- the spec's
    # exact scenario ("a user who queues a three-week-old item wants it in
    # the next briefing").
    old_queued = _add_item(
        db, source, "Old but queued", "2026-07-08T09:00:00+00:00", queued=True
    )
    _complete_briefing(db, watchlist, old_queued)
    in_window = _add_item(db, source, "In window", "2026-07-29T09:00:00+00:00")

    auto = select_briefing_items(db, watchlist, mode="auto")
    assert _ids(auto) == [in_window]
    assert old_queued not in _ids(auto)
    assert auto.featured_ids == set()

    curated = select_briefing_items(db, watchlist, mode="curated")
    assert _ids(curated) == [old_queued]
    assert in_window not in _ids(curated)
    # Curated is not "featured": every item in the briefing is curated, so
    # there is nothing to give top billing over.
    assert curated.featured_ids == set()

    union = select_briefing_items(db, watchlist, mode="auto_featured")
    assert _ids(union) == [old_queued, in_window]  # featured first
    assert union.featured_ids == {old_queued}
    assert union.overflow_count == 0


def test_curated_excludes_items_this_watchlist_already_covered():
    """Junction rows for watchlist W exclude re-selection in W; the same item
    still selects for watchlist V (the global-queue-never-cleared rule).

    The flag is global and generation never clears it, so the ONLY thing that
    can stop a queued item from being selected again is a junction row of
    *this* watchlist's own briefing.
    """
    db = SubscriptionsDB(":memory:", "test")
    bundles = WatchlistBundleService(db)
    w = bundles.create(name="W")["id"]
    v = bundles.create(name="V")["id"]

    source = _new_source(db, w, "shared")
    bundles.add_source(v, source)  # the same source sits in both watchlists

    covered = _add_item(
        db, source, "Already briefed", "2026-07-20T09:00:00+00:00", queued=True
    )
    still_open = _add_item(
        db, source, "Not yet briefed", "2026-07-21T09:00:00+00:00", queued=True
    )

    # W briefed `covered`; V has never briefed anything.
    _complete_briefing(db, w, covered, item_ids=[covered])

    w_curated = select_briefing_items(db, w, mode="curated")
    assert _ids(w_curated) == [still_open]
    assert covered not in _ids(w_curated)

    v_curated = select_briefing_items(db, v, mode="curated")
    assert sorted(_ids(v_curated)) == sorted([still_open, covered])

    # The flag itself was never cleared by W's briefing -- V sees it set.
    assert all(item["queued_for_briefing"] for item in v_curated.items)


def test_overflow_counts_dropped_items_and_features_survive_the_cap():
    """cap=3, five window items + two queued: both queued kept + newest auto
    item; overflow_count == 4 -- exact identities, not just counts.
    """
    db = SubscriptionsDB(":memory:", "test")
    watchlist = WatchlistBundleService(db).create(name="Firehose")["id"]
    source = _new_source(db, watchlist, "firehose")

    # Two queued items below the watermark (window-exempt, hence featured).
    queued_old = _add_item(db, source, "Queued old", "2026-07-10T09:00:00+00:00", queued=True)
    queued_new = _add_item(db, source, "Queued new", "2026-07-11T09:00:00+00:00", queued=True)
    _complete_briefing(db, watchlist, queued_new)

    window = [
        _add_item(db, source, f"Window {n}", "2026-07-29T09:00:00+00:00")
        for n in range(5)
    ]

    selection = select_briefing_items(db, watchlist, mode="auto_featured", item_cap=3)

    # 2 featured + 5 auto = 7 considered; cap 3 keeps both featured (never
    # dropped) plus the single newest auto item. 7 - 3 = 4 dropped.
    assert _ids(selection) == [queued_new, queued_old, window[-1]]
    assert selection.featured_ids == {queued_old, queued_new}
    assert selection.overflow_count == 4
    assert len(selection.items) == 3
    # Every dropped item was an auto item, and all four were seen.
    assert not set(window[:-1]) & set(_ids(selection))
    assert selection.covers_through_item_id == window[-1]


def test_overflow_and_watermark_stay_exact_over_a_backlog_larger_than_the_cap():
    """Whole-branch review fix 2: the window is bounded in SQL so a large
    backlog is not materialised in full, but `overflow_count` and the
    watermark must still be exact over the FULL window, not merely over
    whatever got materialised.

    Seeds `cap + 30` window items -- meaningfully more than any single call
    could need to pull into Python.
    """
    db = SubscriptionsDB(":memory:", "test")
    watchlist = WatchlistBundleService(db).create(name="Torrent")["id"]
    source = _new_source(db, watchlist, "torrent")

    cap = 5
    backlog = [
        _add_item(db, source, f"Item {n}", "2026-07-29T09:00:00+00:00")
        for n in range(cap + 30)
    ]

    selection = select_briefing_items(db, watchlist, mode="auto", item_cap=cap)

    assert len(selection.items) == cap
    assert _ids(selection) == list(reversed(backlog[-cap:]))  # newest `cap`, id DESC
    assert selection.overflow_count == len(backlog) - cap  # exact, not an estimate
    assert selection.covers_through_item_id == backlog[-1]  # the TRUE max, not the max kept


def test_the_window_materialisation_is_bounded_not_the_full_backlog(monkeypatch):
    """The property fix 2 actually asked for: a large window backlog must
    not be pulled into Python row by row just to compute a count and a cap.

    Spies on the row-fetch seam (`_window_rows`) and asserts the number of
    rows it actually fetched -- not the row count `select_briefing_items`
    returns -- stays bounded by cap + featured count, never the full window.
    Featured (queued) items are seeded below the watermark, mirroring
    `test_overflow_counts_dropped_items_and_features_survive_the_cap`, so
    the bound is exercised with a non-zero featured count.
    """
    db = SubscriptionsDB(":memory:", "test")
    watchlist = WatchlistBundleService(db).create(name="Torrent")["id"]
    source = _new_source(db, watchlist, "torrent")

    queued = [
        _add_item(db, source, f"Queued {n}", "2026-07-11T09:00:00+00:00", queued=True)
        for n in range(2)
    ]
    _complete_briefing(db, watchlist, max(queued))
    cap = 5
    for n in range(cap + 30):
        _add_item(db, source, f"Item {n}", "2026-07-29T09:00:00+00:00")

    fetched_counts: list[int] = []
    real_window_rows = briefing_selection._window_rows

    def _spy(*args, **kwargs):
        rows = real_window_rows(*args, **kwargs)
        fetched_counts.append(len(rows))
        return rows

    monkeypatch.setattr(briefing_selection, "_window_rows", _spy)

    selection = select_briefing_items(db, watchlist, mode="auto_featured", item_cap=cap)

    assert len(selection.items) == cap
    featured_count = len(selection.featured_ids)
    assert fetched_counts, "the row-fetch seam must have been called"
    assert all(count <= cap + featured_count for count in fetched_counts), fetched_counts
    # Well short of the full backlog -- the whole property fix 2 exists for.
    assert sum(fetched_counts) < (cap + 30)


def test_overflowed_items_still_advance_the_watermark():
    """The honest-reporting half of the cap rule: items dropped by the cap
    were CONSIDERED and are reported in `overflow_count`, so the recorded
    watermark covers them.

    Constructed so the highest-id item is one the cap dropped: three queued
    items fill a cap of 2, leaving no room for the (newer, higher-id) window
    items at all. `covers_through_item_id` must be the max id considered --
    not the max id kept -- or the next briefing would re-select items this
    one already told the user about, duplicating coverage.
    """
    db = SubscriptionsDB(":memory:", "test")
    watchlist = WatchlistBundleService(db).create(name="Backlog")["id"]
    source = _new_source(db, watchlist, "backlog")

    queued = [
        _add_item(db, source, f"Queued {n}", "2026-07-10T09:00:00+00:00", queued=True)
        for n in range(3)
    ]
    _complete_briefing(db, watchlist, max(queued))
    window = [
        _add_item(db, source, f"Window {n}", "2026-07-29T09:00:00+00:00")
        for n in range(2)
    ]

    selection = select_briefing_items(db, watchlist, mode="auto_featured", item_cap=2)

    # Featured fill the cap: the two newest queued items, nothing else.
    assert _ids(selection) == [queued[2], queued[1]]
    assert selection.featured_ids == {queued[1], queued[2]}
    # 1 dropped featured + 2 dropped window items.
    assert selection.overflow_count == 3
    # Max KEPT id is queued[2]; max CONSIDERED id is the newest window item.
    assert max(_ids(selection)) == queued[2]
    assert selection.covers_through_item_id == window[-1]
    assert selection.covers_through_item_id > max(_ids(selection))


def test_first_window_is_the_last_seven_days_by_created_at():
    """No watermark (the first briefing ever) falls back to a 7-day window
    measured from the INJECTED `now` -- never `datetime.now()` inline, or
    this test could only be written against the wall clock.
    """
    db = SubscriptionsDB(":memory:", "test")
    watchlist = WatchlistBundleService(db).create(name="Fresh")["id"]
    source = _new_source(db, watchlist, "fresh")

    now = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)
    too_old = _add_item(db, source, "Eight days old", (now - timedelta(days=8)).isoformat())
    recent = _add_item(db, source, "Six days old", (now - timedelta(days=6)).isoformat())

    assert db.latest_completed_watermark(watchlist) is None
    selection = select_briefing_items(db, watchlist, mode="auto", now=now)

    assert _ids(selection) == [recent]
    assert too_old not in _ids(selection)
    assert selection.covers_from_ts == (now - timedelta(days=7)).isoformat()
    assert selection.covers_through_item_id == recent


def test_first_window_orders_same_second_ties_by_id():
    """`created_at` has one-second resolution (the TASK-1361 lesson), so a
    burst of items written in the same second is a genuine tie. Ordering by
    id breaks it deterministically: the cap keeps the two newest ROWS, and
    no item is either duplicated or silently dropped from the count.
    """
    db = SubscriptionsDB(":memory:", "test")
    watchlist = WatchlistBundleService(db).create(name="Burst")["id"]
    source = _new_source(db, watchlist, "burst")

    now = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)
    same_second = (now - timedelta(days=1)).isoformat()
    burst = [_add_item(db, source, f"Tie {n}", same_second) for n in range(3)]

    selection = select_briefing_items(db, watchlist, mode="auto", item_cap=2, now=now)

    assert _ids(selection) == [burst[2], burst[1]]
    assert len(_ids(selection)) == len(set(_ids(selection)))  # no duplicates
    assert selection.overflow_count == 1  # exactly one missed, not zero or two
    assert selection.covers_through_item_id == burst[2]


def test_empty_window_returns_none_watermark_and_does_not_advance():
    """`None` means "do not advance": a briefing over an empty window must
    not record a watermark at all, or an `empty` briefing would move the
    coverage line past items that never existed to be covered.
    """
    db = SubscriptionsDB(":memory:", "test")
    watchlist = WatchlistBundleService(db).create(name="Quiet")["id"]
    source = _new_source(db, watchlist, "quiet")
    only_item = _add_item(db, source, "Only item", "2026-07-20T09:00:00+00:00")
    _complete_briefing(db, watchlist, only_item, item_ids=[only_item])

    selection = select_briefing_items(db, watchlist, mode="auto_featured")

    assert isinstance(selection, BriefingSelection)
    assert selection.items == []
    assert selection.featured_ids == set()
    assert selection.overflow_count == 0
    assert selection.covers_through_item_id is None


def test_curated_selection_echoes_the_prior_watermark():
    """Curated mode never reads the coverage window, so it must not move it.

    The watermark is the *window's* line. Advancing it to the newest queued
    id would step it past window items no briefing ever covered -- silently:
    no `overflow_count`, no body text, no status.
    """
    db = SubscriptionsDB(":memory:", "test")
    watchlist = WatchlistBundleService(db).create(name="Curated")["id"]
    source = _new_source(db, watchlist, "curated")

    covered = _add_item(db, source, "Covered", "2026-07-20T09:00:00+00:00")
    _complete_briefing(db, watchlist, covered, item_ids=[covered])
    # Everything below arrives AFTER the watermark, so the naive "max id
    # considered" rule would drag the line forward to `queued`.
    _add_item(db, source, "Uncovered window item", "2026-07-21T09:00:00+00:00")
    queued = _add_item(db, source, "Queued", "2026-07-22T09:00:00+00:00", queued=True)

    selection = select_briefing_items(db, watchlist, mode="curated")

    assert _ids(selection) == [queued]
    assert selection.covers_through_item_id == covered  # the prior line, echoed
    assert selection.covers_through_item_id != queued


def test_switching_off_curated_still_delivers_the_accumulated_window():
    """The scenario the echo protects: a run of curated briefings must not
    quietly consume the window items that piled up beside them.

    Recorded end to end -- the curated briefing is written back with the
    watermark selection returned, exactly as the service will -- so this
    fails if either half of the contract slips.
    """
    db = SubscriptionsDB(":memory:", "test")
    watchlist = WatchlistBundleService(db).create(name="Switcher")["id"]
    source = _new_source(db, watchlist, "switcher")

    first = _add_item(db, source, "First", "2026-07-01T09:00:00+00:00")
    _complete_briefing(db, watchlist, first, item_ids=[first])

    window = [
        _add_item(db, source, f"Window {n}", "2026-07-20T09:00:00+00:00")
        for n in range(3)
    ]
    queued = _add_item(db, source, "Queued", "2026-07-21T09:00:00+00:00", queued=True)

    # A month of curated briefings, each recorded the way the service will.
    for _ in range(3):
        curated = select_briefing_items(db, watchlist, mode="curated")
        _complete_briefing(
            db,
            watchlist,
            curated.covers_through_item_id,
            item_ids=[item["item_id"] for item in curated.items],
        )

    assert db.latest_completed_watermark(watchlist) == first

    # Now the user switches the watchlist to auto_featured.
    after_switch = select_briefing_items(db, watchlist, mode="auto_featured")

    assert sorted(_ids(after_switch)) == sorted(window + [queued])
    assert after_switch.overflow_count == 0
    # The queued item was covered by the curated briefings, so it is no
    # longer featured -- it arrives once more through the auto leg
    # (redundant, never lossy).
    assert after_switch.featured_ids == set()
    assert after_switch.covers_through_item_id == queued


def test_a_failed_briefings_junction_rows_do_not_bury_a_queued_item():
    """Only a briefing that reached the user excludes an item from curation.

    Writing junction rows before the LLM call is a natural implementation, so
    a `failed` briefing plausibly leaves junction rows behind. If those rows
    counted, one failure would bury the queued item forever -- the user's
    curation destroyed by an error they were already shown.
    """
    db = SubscriptionsDB(":memory:", "test")
    watchlist = WatchlistBundleService(db).create(name="Flaky")["id"]
    source = _new_source(db, watchlist, "flaky")
    queued = _add_item(db, source, "Queued", "2026-07-20T09:00:00+00:00", queued=True)

    for status in ("failed", "generating"):
        briefing = db.insert_briefing(watchlist)
        db.update_briefing(briefing, status=status, error="boom")
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO briefing_items (briefing_id, item_id, featured) "
                "VALUES (?, ?, 1)",
                (briefing, queued),
            )

    # A failure and a zombie `generating` row (crashed worker) later, the
    # item is still curated.
    assert _ids(select_briefing_items(db, watchlist, mode="curated")) == [queued]

    # ... and a briefing that DID reach the user still excludes it.
    _complete_briefing(db, watchlist, queued, item_ids=[queued])
    assert select_briefing_items(db, watchlist, mode="curated").items == []


def test_covers_from_ts_compares_normalized_timestamps():
    """`created_at` is mixed-format in this table: `CURRENT_TIMESTAMP` writes
    `'2026-07-25 09:00:00'` and the persist path writes
    `'2026-07-25T08:00:00+00:00'`. A raw string `min()` sorts on the space
    versus the `T` and picks the LATER instant as the earliest.
    """
    db = SubscriptionsDB(":memory:", "test")
    watchlist = WatchlistBundleService(db).create(name="Formats")["id"]
    source = _new_source(db, watchlist, "formats")

    anchor = _add_item(db, source, "Anchor", "2026-07-01T09:00:00+00:00")
    _complete_briefing(db, watchlist, anchor, item_ids=[anchor])

    # Same day; the ISO one is an hour EARLIER but sorts LATER as a string.
    _add_item(db, source, "Iso format", "2026-07-25T08:00:00+00:00")
    _add_item(db, source, "Space format", "2026-07-25 09:00:00")

    selection = select_briefing_items(db, watchlist, mode="auto")

    assert len(selection.items) == 2
    assert selection.covers_from_ts == "2026-07-25 08:00:00"
    assert selection.covers_from_ts != "2026-07-25 09:00:00"


def test_unknown_mode_is_rejected_by_name():
    db = SubscriptionsDB(":memory:", "test")
    watchlist = WatchlistBundleService(db).create(name="Modes")["id"]
    with pytest.raises(ValueError, match="auto_featureed"):
        select_briefing_items(db, watchlist, mode="auto_featureed")
