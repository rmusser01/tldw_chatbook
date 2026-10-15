# Tests/Dreams/test_dreams_db_track.py
"""DreamsDB schema v2 behavior: tracked items, track runs, v1->v2 upgrade.

Also pins the query plans of the three track-loop indexes (the
``scripts/check_index_plan_pins.py`` census rows point here): every plan
assertion first proves ``sqlite_stat1`` is absent, because Dreams_DB.py
runs no ANALYZE and a plan captured with stats is not the plan a user's
database produces.
"""
import sqlite3

import pytest

from Tests.Dreams.test_schema_lineage import replace_with_historical_fixture
from tldw_chatbook.DB.Dreams_DB import DreamsDB


@pytest.fixture()
def db(tmp_path):
    database = DreamsDB(tmp_path / "dreams.sqlite", "test-client")
    yield database
    database.close()


def _make_item(db, **overrides):
    kwargs: dict = {
        "mechanism": "question",
        "intent": "event",
        "cadence_seconds": 3600,
    }
    kwargs.update(overrides)
    return db.create_tracked_item(**kwargs)


def test_create_and_get_tracked_item_roundtrip_with_defaults(db):
    item_id = _make_item(
        db,
        origin_story_id=42,
        subscription_id=7,
        query_template="flights to {place}",
        event_date="2026-10-01",
    )
    item = db.get_tracked_item(item_id)
    assert item is not None
    assert item["id"] == item_id
    assert item["origin_story_id"] == 42
    assert item["mechanism"] == "question"
    assert item["intent"] == "event"
    assert item["subscription_id"] == 7
    assert item["query_template"] == "flights to {place}"
    assert item["event_date"] == "2026-10-01"
    assert item["cadence_seconds"] == 3600
    # Column defaults from the DDL
    assert item["quiet_retire_count"] == 0
    assert item["status"] == "active"
    assert item["retired_reason"] is None
    assert item["created_by_dreams"] == 0
    assert item["last_checked"] is None
    assert item["created_at"]
    assert item["updated_at"]


def test_get_tracked_item_missing_returns_none(db):
    assert db.get_tracked_item(999) is None


def test_create_tracked_item_rejects_mechanism_outside_vocabulary(db):
    with pytest.raises(sqlite3.IntegrityError):
        _make_item(db, mechanism="bogus")


def test_create_tracked_item_rejects_intent_outside_vocabulary(db):
    with pytest.raises(sqlite3.IntegrityError):
        _make_item(db, intent="bogus")


def test_list_tracked_items_orders_newest_first_and_filters_status(db):
    first = _make_item(db, query_template="q1")
    second = _make_item(db, query_template="q2")
    third = _make_item(db, query_template="q3")
    db.set_tracked_status(second, "paused")
    active = db.list_tracked_items()
    assert [i["id"] for i in active] == [third, first]
    assert [i["id"] for i in db.list_tracked_items(status="paused")] == [second]
    assert [i["id"] for i in db.list_tracked_items(status="retired")] == []


def test_find_tracked_by_story(db):
    assert db.find_tracked_by_story(42) is None
    item_id = _make_item(db, origin_story_id=42)
    found = db.find_tracked_by_story(42)
    assert found is not None
    assert found["id"] == item_id


def test_find_tracked_by_story_returns_only_active_rows(db):
    """Final review, ruling P10: a retired/paused wrapper is not untrackable.

    Without the status filter the story modal's ``u`` resolves a
    sweep-retired row, re-retires it (``retired_reason`` COALESCE-rewrites
    the sweep's ``event_passed``/``quiet`` audit data), re-disables the
    already-disabled subscription, and posts "Stopped tracking this page."
    for a watch the user never manually stopped.
    """
    retired = _make_item(db, origin_story_id=42)
    db.set_tracked_status(retired, "retired", retired_reason="event_passed")
    assert db.find_tracked_by_story(42) is None, (
        "a retired row must not surface for untrack"
    )

    paused = _make_item(db, origin_story_id=42)
    db.set_tracked_status(paused, "paused")
    assert db.find_tracked_by_story(42) is None, (
        "a paused row must not surface for untrack either"
    )

    active = _make_item(db, origin_story_id=42)
    assert db.find_tracked_by_story(42)["id"] == active


def test_find_tracked_by_story_can_be_narrowed_to_one_mechanism(db):
    """Qodo #10 (PR #2921): one story may own a page watch AND a question
    watch. "The newest active wrapper" then names whichever was added
    last, so a caller that means one of them must be able to ask for it."""
    page = _make_item(db, origin_story_id=42, mechanism="page")
    question = _make_item(db, origin_story_id=42, mechanism="question")

    assert db.find_tracked_by_story(42, mechanism="page")["id"] == page
    assert db.find_tracked_by_story(42, mechanism="question")["id"] == question
    assert db.find_tracked_by_story(42)["id"] == question, (
        "unfiltered stays the newest active wrapper of any mechanism"
    )
    db.set_tracked_status(page, "retired", retired_reason="manual")
    assert db.find_tracked_by_story(42, mechanism="page") is None


def test_set_tracked_status_stamps_retired_reason_and_updated_at(db):
    item_id = _make_item(db)
    # Backdate the stamp so the update must land in a strictly later
    # millisecond (creation and update can tie within one ms).
    with db.transaction() as conn:
        conn.execute(
            "UPDATE dream_tracked_items SET updated_at = '2020-01-01T00:00:00.000Z'"
            " WHERE id = ?",
            (item_id,),
        )
    db.set_tracked_status(item_id, "retired", retired_reason="quiet: too many duds")
    after = db.get_tracked_item(item_id)
    assert after["status"] == "retired"
    assert after["retired_reason"] == "quiet: too many duds"
    assert after["updated_at"] > "2020-01-01T00:00:00.000Z"


def test_set_tracked_status_rejects_status_outside_vocabulary(db):
    item_id = _make_item(db)
    with pytest.raises(sqlite3.IntegrityError):
        db.set_tracked_status(item_id, "bogus")


def test_touch_tracked_checked_stamps_last_checked(db):
    item_id = _make_item(db)
    db.touch_tracked_checked(item_id, "2026-09-28T12:00:00.000Z")
    item = db.get_tracked_item(item_id)
    assert item["last_checked"] == "2026-09-28T12:00:00.000Z"
    assert item["updated_at"] >= "2026-09-28T12:00:00.000Z"


def test_count_active_tracked_counts_only_active(db):
    a = _make_item(db)
    b = _make_item(db)
    c = _make_item(db)
    assert db.count_active_tracked() == 3
    db.set_tracked_status(b, "paused")
    db.set_tracked_status(c, "retired")
    assert db.count_active_tracked() == 1
    assert db.list_tracked_items()[0]["id"] == a


def test_count_active_tracked_for_subscription_counts_only_live_wrappers(db):
    """Qodo #4 (PR #2921): several wrappers may ride one subscription; the
    count of those still ACTIVE is what decides whether retiring one of
    them may disable the shared source."""
    assert db.count_active_tracked_for_subscription(7) == 0
    first = _make_item(db, mechanism="page", subscription_id=7)
    _make_item(db, mechanism="page", subscription_id=7)
    _make_item(db, mechanism="page", subscription_id=8)
    paused = _make_item(db, mechanism="page", subscription_id=7)
    db.set_tracked_status(paused, "paused")
    assert db.count_active_tracked_for_subscription(7) == 2
    db.set_tracked_status(first, "retired", retired_reason="manual")
    assert db.count_active_tracked_for_subscription(7) == 1
    assert db.count_active_tracked_for_subscription(8) == 1


def test_insert_and_list_recent_track_runs_newest_first(db):
    item_id = _make_item(db)
    db.insert_track_run(item_id, status="baseline", digest_hash="h0")
    db.insert_track_run(item_id, status="unchanged", digest_hash="h1")
    db.insert_track_run(item_id, status="changed", digest_hash="h2",
                        verdict_note="price dropped", notified=1)
    runs = db.list_recent_track_runs(item_id)
    assert [r["status"] for r in runs] == ["changed", "unchanged", "baseline"]
    newest = runs[0]
    assert newest["digest_hash"] == "h2"
    assert newest["verdict_note"] == "price dropped"
    assert newest["notified"] == 1
    # Limit slices the newest end, not the oldest
    assert [r["status"] for r in db.list_recent_track_runs(item_id, limit=2)] == \
        ["changed", "unchanged"]
    assert db.list_recent_track_runs(999) == []


def test_latest_anchor_digest_skips_non_anchoring_runs_without_a_window(db):
    """Qodo #12 (PR #2921): the anchor is the newest run whose status may
    anchor a comparison, however many error/skipped/withheld runs came
    after it -- read directly, not searched for in a bounded slice."""
    item_id = _make_item(db)
    assert db.latest_anchor_digest(item_id) is None, "no runs yet"
    db.insert_track_run(item_id, status="skipped", digest_hash=None)
    assert db.latest_anchor_digest(item_id) is None, "nothing anchors yet"
    db.insert_track_run(item_id, status="baseline", digest_hash="h0")
    db.insert_track_run(item_id, status="changed", digest_hash="h1")
    for status in ("error", "skipped", "withheld") * 100:
        db.insert_track_run(item_id, status=status, digest_hash=None)
    assert db.latest_anchor_digest(item_id) == "h1"
    db.insert_track_run(item_id, status="rebaselined", digest_hash="h2")
    assert db.latest_anchor_digest(item_id) == "h2"
    other = _make_item(db)
    assert db.latest_anchor_digest(other) is None, "anchors are per item"


def test_insert_track_run_rejects_status_outside_vocabulary(db):
    item_id = _make_item(db)
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_track_run(item_id, status="bogus", digest_hash="h")


def test_consecutive_track_dispositions_counts_trailing_run(db):
    item_id = _make_item(db)
    # Insert order doubles as age order: baseline, unchanged, unchanged, changed
    for status in ("baseline", "unchanged", "unchanged", "changed"):
        db.insert_track_run(item_id, status=status, digest_hash="h")
    # Newest run is 'changed', so only 'changed' has a trailing run
    assert db.consecutive_track_dispositions(item_id, "changed") == 1
    assert db.consecutive_track_dispositions(item_id, "unchanged") == 0
    assert db.consecutive_track_dispositions(item_id, "baseline") == 0

    other_id = _make_item(db)
    for _ in range(3):
        db.insert_track_run(other_id, status="unchanged", digest_hash="h")
    assert db.consecutive_track_dispositions(other_id, "unchanged") == 3
    # No runs at all -> zero for any status
    empty_id = _make_item(db)
    assert db.consecutive_track_dispositions(empty_id, "unchanged") == 0


def _rewind_to_v1(db) -> None:
    replace_with_historical_fixture(db, 1)


def test_v1_file_upgrades_in_place_to_v2(tmp_path):
    path = tmp_path / "dreams-v1.sqlite"
    db = DreamsDB(path, "test-client")
    db.create_collection("2026-09-22", "scheduled", "digest")
    _rewind_to_v1(db)
    with db.connection() as conn:
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
            " AND name = 'dream_tracked_items'"
        ).fetchone() is None
        assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 1
    db.close()

    # Reopening the v1 file must upgrade it in place via the additive DDL
    # (through every later additive bump -- v3 adds the 'guardian' profile
    # source CHECK, so a v1 file lands on version 3 as well).
    reopened = DreamsDB(path, "test-client")
    with reopened.connection() as conn:
        tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        indexes = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_dream_%'"
            ).fetchall()
        }
        assert "dream_tracked_items" in tables
        assert "dream_track_runs" in tables
        assert "idx_dream_tracked_status" in indexes
        assert "idx_dream_track_runs_item" in indexes
        # Task 6 ledger item: the per-story reads Task 5 added
        # (``find_tracked_by_story`` + the badge) get their pinned index.
        assert "idx_dream_tracked_origin_story" in indexes
        assert conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()[0] == DreamsDB._CURRENT_SCHEMA_VERSION
    # The upgraded file is fully usable: track CRUD works and v1 data survived.
    item_id = reopened.create_tracked_item(
        mechanism="page", intent="deal", cadence_seconds=60
    )
    reopened.insert_track_run(item_id, status="baseline", digest_hash="h")
    assert reopened.count_active_tracked() == 1
    assert reopened.get_collection_by_date("2026-09-22")["status"] == "generating"
    reopened.close()


# --- Schema v3 (Guardian x Dreams, ADR-204 contract 3) -------------------------


def _rewind_to_v2(db) -> None:
    replace_with_historical_fixture(db, 2)


def test_v2_file_upgrades_in_place_to_v3_and_accepts_guardian_source(tmp_path):
    path = tmp_path / "dreams-v2.sqlite"
    db = DreamsDB(path, "test-client")
    db.upsert_profile_entry(
        "topic", "rust tui", weight=0.8, searchable=1, source="notes"
    )
    _rewind_to_v2(db)
    with db.connection() as conn:
        assert conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()[0] == 2
        # The v2 CHECK refuses the guardian source on the rewound file.
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO dream_interest_profile"
                " (facet, text, weight, searchable, source, created_at,"
                "  updated_at) VALUES ('topic', 'x', 0.5, 1, 'guardian',"
                " '2026-09-29T00:00:00Z', '2026-09-29T00:00:00Z')"
            )
    db.close()

    # Reopening the v2 file must upgrade it in place: the version stamps
    # forward and the rebuilt CHECK accepts 'guardian' through the public
    # writer.
    reopened = DreamsDB(path, "test-client")
    reopened.upsert_profile_entry(
        "topic", "late_night_work", weight=0.75, searchable=1, source="guardian"
    )
    with reopened.connection() as conn:
        assert conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()[0] == DreamsDB._CURRENT_SCHEMA_VERSION
    by_text = {row["text"]: row for row in reopened.list_profile()}
    assert by_text["rust tui"]["source"] == "notes", "v2 data survived"
    assert by_text["late_night_work"]["source"] == "guardian"
    reopened.close()


def test_fresh_v3_build_never_runs_the_profile_rebuild_migration(tmp_path):
    """A fresh build creates the new-CHECK table directly; the v2 -> v3
    rebuild migration must be a no-op there (pinned by table id continuity:
    the rebuild would have replaced the table object)."""
    db = DreamsDB(tmp_path / "dreams-fresh.sqlite", "test-client")
    db.upsert_profile_entry(
        "topic", "seeded", weight=1.0, searchable=1, source="seed"
    )
    with db.connection() as conn:
        (profile_id,) = conn.execute(
            "SELECT id FROM dream_interest_profile"
        ).fetchone()
        assert conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()[0] == DreamsDB._CURRENT_SCHEMA_VERSION
    assert profile_id == 1, "the seeded row keeps the first autoincrement id"
    db.close()


# --- Schema v4 (review of #2921, item 1): story goal provenance ------------------


def _story_columns(db) -> list[str]:
    with db.connection() as conn:
        return [row[1] for row in conn.execute(
            "PRAGMA table_info(dream_stories)")]


def _insert_plain_story(db, collection_id, **overrides):
    kwargs = dict(
        title="t", url="https://dreams.test/a", snippet="s", body="b",
        status="complete", source="web", kind="content", event_date=None,
        location=None, matched_topics=["rust tui"], query="q")
    kwargs.update(overrides)
    return db.insert_story(collection_id, **kwargs)


def test_schema_version_is_six():
    assert DreamsDB._CURRENT_SCHEMA_VERSION == 6


def test_story_goal_provenance_round_trips_and_defaults_to_none(db):
    """A story row records the goal whose query surfaced it, apart from
    the topics it matched; a story no goal's query surfaced records none."""
    collection = db.create_collection("2026-09-22", "scheduled", "digest")
    plain = _insert_plain_story(db, collection)
    attributed = _insert_plain_story(
        db, collection, url="https://dreams.test/b", goal="visit Japan")

    assert db.get_story(plain)["goal"] is None
    assert db.get_story(attributed)["goal"] == "visit Japan"
    assert db.get_story(attributed)["matched_topics"] == ["rust tui"]


def test_v3_file_upgrades_in_place_to_v4_keeping_its_stories(tmp_path):
    """v4 is additive: one nullable ``goal`` column on ``dream_stories``.
    A v3 file gains it on open, at the end of the row like a fresh build,
    and every story it already held survives with no goal recorded."""
    path = tmp_path / "dreams-v3.sqlite"
    db = DreamsDB(path, "test-client")
    fresh_columns = _story_columns(db)
    assert fresh_columns[-1] == "goal"
    collection = db.create_collection("2026-09-22", "scheduled", "digest")
    story_id = _insert_plain_story(db, collection)
    replace_with_historical_fixture(db, 3)
    assert "goal" not in _story_columns(db)
    db.close()

    reopened = DreamsDB(path, "test-client")
    try:
        assert _story_columns(reopened) == fresh_columns
        with reopened.connection() as conn:
            assert conn.execute(
                "SELECT MAX(version) FROM schema_version"
            ).fetchone()[0] == DreamsDB._CURRENT_SCHEMA_VERSION
        story = reopened.get_story(story_id)
        assert story["goal"] is None
        assert story["matched_topics"] == ["rust tui"], "v3 data survived"
        second = _insert_plain_story(
            reopened, collection, url="https://dreams.test/b", goal="g")
        assert reopened.get_story(second)["goal"] == "g"
    finally:
        reopened.close()


# --- Schema v5 (review of #2921, item 4): what the master switch stopped ----------


def _tracked_columns(db) -> list[str]:
    with db.connection() as conn:
        return [row[1] for row in conn.execute(
            "PRAGMA table_info(dream_tracked_items)")]


def test_tracked_suspension_record_round_trips_and_clears(db):
    """A wrapper carries the record of what the master switch stopped for
    it (opaque text to the DB), or NULL when it is not suspended."""
    item_id = _make_item(db)
    assert db.get_tracked_item(item_id)["suspension"] is None

    db.set_tracked_suspension(item_id, '{"subscription": true}')
    assert db.get_tracked_item(item_id)["suspension"] == \
        '{"subscription": true}'
    assert db.get_tracked_item(item_id)["status"] == "active", (
        "suspension is not a lifecycle state"
    )

    db.set_tracked_suspension(item_id, None)
    assert db.get_tracked_item(item_id)["suspension"] is None


def test_live_tracked_suspensions_lists_live_wrappers_records_only(db):
    """The startup reconcile's one read: the suspension record of every
    live (active or paused) wrapper, as stored -- ``None`` where there is
    none. Retired wrappers never appear, whatever record they carry."""
    assert db.live_tracked_suspensions() == []

    running = _make_item(db)
    suspended = _make_item(db)
    db.set_tracked_suspension(suspended, '{"settled": "off"}')
    assert sorted(db.live_tracked_suspensions(), key=str) == [
        None, '{"settled": "off"}']

    db.set_tracked_status(suspended, "paused")
    assert '{"settled": "off"}' in db.live_tracked_suspensions(), (
        "a paused watch is live too"
    )
    db.set_tracked_status(suspended, "retired", retired_reason="manual")
    assert db.live_tracked_suspensions() == [None]
    db.set_tracked_status(running, "retired", retired_reason="manual")
    assert db.live_tracked_suspensions() == []


#: The columns v6 adds to ``dream_tracked_items``, in DDL order.
_V6_COLUMNS = ("egress_marked", "alert_rule_id", "reminder_event_date")


def _rewind_tracked_items(db, *, to_version: int) -> None:
    replace_with_historical_fixture(db, to_version)


def _stored_version(db) -> int:
    with db.connection() as conn:
        return conn.execute(
            "SELECT MAX(version) FROM schema_version").fetchone()[0]


def test_v4_file_upgrades_in_place_keeping_its_tracked_items(tmp_path):
    """v5 is additive: one nullable ``suspension`` column on
    ``dream_tracked_items``. A v4 file runs that step and the v6 one, and
    ends with the columns in the order a fresh build declares them."""
    path = tmp_path / "dreams-v4.sqlite"
    db = DreamsDB(path, "test-client")
    fresh_columns = _tracked_columns(db)
    item_id = _make_item(db, origin_story_id=42, subscription_id=7)
    _rewind_tracked_items(db, to_version=4)
    assert "suspension" not in _tracked_columns(db)
    db.close()

    reopened = DreamsDB(path, "test-client")
    try:
        assert _tracked_columns(reopened) == fresh_columns
        assert _stored_version(reopened) == 6
        item = reopened.get_tracked_item(item_id)
        assert item["suspension"] is None
        assert (item["origin_story_id"], item["subscription_id"]) == (42, 7), (
            "v4 data survived"
        )
    finally:
        reopened.close()


def test_v5_file_upgrades_in_place_to_v6_keeping_its_tracked_items(tmp_path):
    """v6 (review round 3) adds three columns to ``dream_tracked_items``,
    last in the row like a fresh build: ``egress_marked`` (T1),
    ``alert_rule_id`` (T11) and ``reminder_event_date`` (T5). Existing
    wrappers come out UNMARKED -- that is the pending marker the one-time
    egress backfill works from -- with no rule and no reminder date on
    record."""
    path = tmp_path / "dreams-v5.sqlite"
    db = DreamsDB(path, "test-client")
    fresh_columns = _tracked_columns(db)
    assert tuple(fresh_columns[-3:]) == _V6_COLUMNS
    item_id = _make_item(
        db, origin_story_id=42, subscription_id=7, created_by_dreams=1)
    db.set_tracked_suspension(item_id, '{"settled": "off"}')
    _rewind_tracked_items(db, to_version=5)
    assert not set(_V6_COLUMNS) & set(_tracked_columns(db))
    db.close()

    reopened = DreamsDB(path, "test-client")
    try:
        assert _tracked_columns(reopened) == fresh_columns
        assert _stored_version(reopened) == 6
        item = reopened.get_tracked_item(item_id)
        assert (item["egress_marked"], item["alert_rule_id"],
                item["reminder_event_date"]) == (0, None, None)
        assert (item["origin_story_id"], item["subscription_id"],
                item["suspension"]) == (42, 7, '{"settled": "off"}'), (
            "v5 data survived"
        )
        assert reopened.subscriptions_awaiting_egress_mark() == [7]
    finally:
        reopened.close()
    # Re-opening a migrated file is a no-op.
    again = DreamsDB(path, "test-client")
    try:
        assert _tracked_columns(again) == fresh_columns
    finally:
        again.close()


def test_egress_mark_backlog_lists_dreams_own_unmarked_subscriptions(db):
    """T1: every status counts (a retired watch's source still exists and
    can be run from Watchlists); a source Dreams merely attached to never
    appears; marking is per subscription and final."""
    _make_item(db, mechanism="page", subscription_id=7, created_by_dreams=1)
    retired = _make_item(
        db, mechanism="page", subscription_id=8, created_by_dreams=1)
    db.set_tracked_status(retired, "retired", retired_reason="manual")
    _make_item(db, mechanism="page", subscription_id=7, created_by_dreams=1)
    _make_item(db, mechanism="page", subscription_id=9, created_by_dreams=0)
    _make_item(db, created_by_dreams=1)  # a question watch: no subscription
    _make_item(db, mechanism="page", subscription_id=10, created_by_dreams=1,
               egress_marked=1)

    assert db.subscriptions_awaiting_egress_mark() == [7, 8]
    assert db.tracking_boot_state()[1] is True

    db.set_egress_marked(7)
    assert db.subscriptions_awaiting_egress_mark() == [8]
    db.set_egress_marked(8)
    assert db.subscriptions_awaiting_egress_mark() == []
    assert db.tracking_boot_state()[1] is False


def test_tracking_boot_state_is_the_live_records_and_the_backlog_flag(db):
    live = _make_item(db)
    db.set_tracked_suspension(live, '{"settled": "off"}')

    assert db.tracking_boot_state() == (['{"settled": "off"}'], False)


# --- Qodo review on PR #2890 -----------------------------------------------------


def test_list_tracked_items_is_bounded(db):
    """Qodo #9: retired rows accumulate forever, so the status read is
    capped (TRACKED_ITEMS_LIST_LIMIT, mirroring ``clamp_limit``'s spirit);
    the default cap is respected, callers can page below it, and garbage
    limits clamp instead of unbounding (SQLite reads negative LIMIT as
    no-limit)."""
    from tldw_chatbook.DB.Dreams_DB import TRACKED_ITEMS_LIST_LIMIT

    seeded = TRACKED_ITEMS_LIST_LIMIT + 5
    with db.transaction() as conn:
        conn.executemany(
            "INSERT INTO dream_tracked_items"
            " (mechanism, intent, cadence_seconds, status, query_template,"
            "  created_at, updated_at)"
            " VALUES ('question', 'topic', 3600, 'retired', ?, ?, ?)",
            [(f"q {i}", "2026-09-01T00:00:00+00:00",
              "2026-09-01T00:00:00+00:00") for i in range(seeded)],
        )

    assert len(db.list_tracked_items("retired")) == TRACKED_ITEMS_LIST_LIMIT
    assert len(db.list_tracked_items("retired", limit=10)) == 10
    assert len(db.list_tracked_items("retired", limit=-5)) == 1, (
        "a negative limit clamps to 1, the way ``clamp_limit`` does"
    )


def test_set_tracked_plan_rewrites_cadence_and_event_date(db):
    """Qodo #2 support: the reuse path refreshes the caller's plan on the
    existing active wrapper -- both fields, including clearing the date."""
    item = db.create_tracked_item(
        mechanism="page", intent="event", cadence_seconds=43200,
        event_date="2026-10-01", subscription_id=9, created_by_dreams=1)

    db.set_tracked_plan(item, cadence_seconds=86400, event_date=None)

    row = db.get_tracked_item(item)
    assert row["cadence_seconds"] == 86400
    assert row["event_date"] is None
    assert row["status"] == "active"


# --- Query-plan pins (scripts/check_index_plan_pins.py census rows) -------------
#
# TASK-21126's rule: with no sqlite_stat1 the planner may ignore a perfect
# index, so "the index exists" proves nothing. Each pin below captures
# EXPLAIN QUERY PLAN for the ACTUAL production SQL (kept as literals on
# purpose -- importing the reader's string would hide a reader/index
# mismatch) on a seeded database with sqlite_stat1 ABSENT, and asserts the
# census-recorded index is the one the planner seeks.
#
# Honest scope, measured rather than assumed: unlike the media v9 set, these
# indexes buy the equality SEEK, not the sort. ``created_at DESC, id DESC``
# still goes through a temp B-tree (the indexes do not carry the ordering
# columns), and the ``id DESC`` tiebreak term keeps one even where the
# leading ordering column is indexed. The pins below assert the SEEK and the
# flip from SCAN (see the negative control); they deliberately claim
# nothing about sort elimination.

#: The three census rows in scripts/index_plan_pin_census.tsv.
RUNS_ITEM = "idx_dream_track_runs_item"
TRACKED_STATUS = "idx_dream_tracked_status"
TRACKED_ORIGIN_STORY = "idx_dream_tracked_origin_story"

#: Production SQL exactly as the readers spell it, with the binds each
#: caller actually passes.
RUNS_LIST_SQL = (
    "SELECT * FROM dream_track_runs WHERE tracked_item_id = ?"
    " ORDER BY created_at DESC, id DESC LIMIT ?"
)
TRACKED_LIST_SQL = (
    "SELECT * FROM dream_tracked_items WHERE status = ?"
    " ORDER BY created_at DESC, id DESC LIMIT ?"
)
ACTIVE_COUNT_SQL = (
    "SELECT COUNT(*) FROM dream_tracked_items WHERE status = 'active'"
)
#: Qodo #4 (PR #2921): "does another active wrapper ride this source?"
ACTIVE_FOR_SUBSCRIPTION_SQL = (
    "SELECT COUNT(*) FROM dream_tracked_items"
    " WHERE status = 'active' AND subscription_id = ?"
)
LIVE_SUSPENSIONS_SQL = (
    "SELECT suspension FROM dream_tracked_items"
    " WHERE status IN ('active', 'paused')"
)
#: TASK-34924: the start check's read of the live wrappers' rows.
LIVE_BOOT_ROWS_SQL = (
    "SELECT id, subscription_id, created_by_dreams, alert_rule_id,"
    " event_date, suspension FROM dream_tracked_items"
    " WHERE status IN ('active', 'paused')"
)
RETIRED_BOOT_ROWS_SQL = (
    "SELECT id, subscription_id, created_by_dreams, alert_rule_id,"
    " event_date, suspension FROM dream_tracked_items"
    " WHERE status = 'retired' AND suspension IS NOT NULL"
)
STORY_LOOKUP_SQL = (
    "SELECT * FROM dream_tracked_items WHERE origin_story_id = ?"
    " AND (status = 'active' OR (? AND status = 'paused'))"
    " AND (? IS NULL OR mechanism = ?)"
    " ORDER BY (status = 'active') DESC, created_at DESC,"
    " id DESC LIMIT 1"
)
#: Round 3, T5: every wrapper a story ever had (the reminder decision).
STORY_WRAPPERS_SQL = (
    "SELECT * FROM dream_tracked_items WHERE origin_story_id = ?"
    " ORDER BY id"
)

#: Qodo #12 (PR #2921): the comparison anchor, read without a row window.
ANCHOR_DIGEST_SQL = (
    "SELECT digest_hash FROM dream_track_runs"
    " WHERE tracked_item_id = ?"
    " AND status NOT IN ('error', 'skipped', 'withheld')"
    " ORDER BY created_at DESC, id DESC LIMIT 1"
)

PLAN_PINNED_QUERIES = {
    "list_recent_track_runs": (RUNS_LIST_SQL, (5, 5), RUNS_ITEM),
    "latest_anchor_digest": (ANCHOR_DIGEST_SQL, (5,), RUNS_ITEM),
    "list_tracked_items": (TRACKED_LIST_SQL, ("active", 500), TRACKED_STATUS),
    "count_active_tracked": (ACTIVE_COUNT_SQL, (), TRACKED_STATUS),
    "count_active_tracked_for_subscription": (
        ACTIVE_FOR_SUBSCRIPTION_SQL, (7,), TRACKED_STATUS),
    # Review of #2921, item 3c / round 3: the boot reconcile's one read.
    "live_tracked_suspensions": (LIVE_SUSPENSIONS_SQL, (), TRACKED_STATUS),
    "tracking_boot_state[records]": (
        LIVE_SUSPENSIONS_SQL, (), TRACKED_STATUS),
    "tracking_boot_rows[rows]": (LIVE_BOOT_ROWS_SQL, (), TRACKED_STATUS),
    "tracking_boot_rows[retired]": (RETIRED_BOOT_ROWS_SQL, (), TRACKED_STATUS),
    "find_tracked_by_story": (
        STORY_LOOKUP_SQL, (7, 0, None, None), TRACKED_ORIGIN_STORY),
    # Qodo #10 (PR #2921): the mechanism-narrowed form is the same
    # statement with the binds the page untrack and both track paths pass.
    "find_tracked_by_story[mechanism]": (
        STORY_LOOKUP_SQL, (7, 0, "page", "page"), TRACKED_ORIGIN_STORY),
    # Round 3, T5: the reuse lookup of both track paths also takes a
    # paused wrapper; same statement, same index.
    "find_tracked_by_story[include_paused]": (
        STORY_LOOKUP_SQL, (7, 1, "page", "page"), TRACKED_ORIGIN_STORY),
    "list_tracked_for_story": (
        STORY_WRAPPERS_SQL, (7,), TRACKED_ORIGIN_STORY),
}

CONSECUTIVE_DISPOSITIONS_SQL = (
    "SELECT COUNT(*) FROM dream_track_runs AS r"
    " WHERE r.tracked_item_id = ? AND r.status = ?"
    " AND r.id > COALESCE(("
    "     SELECT b.id FROM dream_track_runs AS b"
    "     WHERE b.tracked_item_id = ? AND b.status <> ?"
    "     ORDER BY b.created_at DESC, b.id DESC LIMIT 1"
    " ), 0)"
)


def _assert_no_stats(conn) -> None:
    """Prove the fixture reproduces the no-stats production state.

    Dreams_DB.py runs no ANALYZE, so no user's database carries
    ``sqlite_stat1`` and a plan captured with one present is not the plan
    they run.
    """
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_stat1'"
    ).fetchone() is None


def _plan(conn, sql: str, params: tuple = ()) -> str:
    """Capture EXPLAIN QUERY PLAN, stats-absent, as one detail string."""
    _assert_no_stats(conn)
    detail = " | ".join(
        row["detail"] for row in conn.execute("EXPLAIN QUERY PLAN " + sql, params)
    )
    assert detail, "an empty plan satisfies every negative assertion"
    return detail


def _seed_plan_corpus(db) -> None:
    """Insert tracked items and runs directly (fast, shape-exact).

    Goes around the writers deliberately: this section cares about the
    physical row shape the planner sees, and needs all three lifecycle
    statuses and NULL/duplicate origin stories in one table.
    300 items (100 per status, half with an origin story in 1..50) and
    600 runs over 12 items -- enough rows that the stats-free planner's
    choices are the ones it makes on a real database, not a toy.
    """
    items = []
    for i in range(1, 301):
        status = "active" if i % 3 == 0 else ("paused" if i % 3 == 1 else "retired")
        items.append(
            (
                i,
                (i % 50) + 1 if i % 2 else None,
                "question",
                "event",
                3600,
                status,
                f"2026-08-{1 + i % 28:02d}T00:00:00Z",
                "2026-08-01T00:00:00Z",
            )
        )
    runs = [
        (
            item,
            "unchanged" if j % 5 else "changed",
            f"h{j}",
            f"2026-09-{1 + j % 28:02d}T00:00:{j % 60:02d}Z",
        )
        for item in range(1, 13)
        for j in range(50)
    ]
    with db.transaction() as conn:
        conn.executemany(
            "INSERT INTO dream_tracked_items (id, origin_story_id, mechanism,"
            " intent, cadence_seconds, status, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            items,
        )
        conn.executemany(
            "INSERT INTO dream_track_runs (tracked_item_id, status, digest_hash,"
            " created_at) VALUES (?,?,?,?)",
            runs,
        )


@pytest.mark.parametrize("label", sorted(PLAN_PINNED_QUERIES))
def test_production_query_plan_seeks_its_census_index(db, label):
    sql, params, expected = PLAN_PINNED_QUERIES[label]
    _seed_plan_corpus(db)
    with db.connection() as conn:
        plan = _plan(conn, sql, params)
    assert expected in plan, f"{label}: {plan}"
    # count_active_tracked additionally reads the index alone (COVERING):
    # no table row is touched for the cap guard's COUNT.
    if label == "count_active_tracked":
        assert f"COVERING INDEX {TRACKED_STATUS}" in plan, plan


def test_consecutive_track_dispositions_plan_seeks_the_runs_index_both_halves(db):
    """The quiet-retire gate's COUNT and its scalar subquery both seek.

    The outer ``r`` scan and the inner ``b`` lookup each filter on
    ``tracked_item_id`` alone, so both must land on
    ``idx_dream_track_runs_item`` -- the subquery is the one that runs per
    retirement decision, and a SCAN there is the slow path this index
    exists to remove.
    """
    _seed_plan_corpus(db)
    with db.connection() as conn:
        plan = _plan(
            conn,
            CONSECUTIVE_DISPOSITIONS_SQL,
            (5, "unchanged", 5, "unchanged"),
        )
    assert f"SEARCH r USING INDEX {RUNS_ITEM}" in plan, plan
    assert f"SEARCH b USING INDEX {RUNS_ITEM}" in plan, plan


def test_dropping_the_indexes_flips_every_pinned_query_to_a_scan(db):
    """The negative control: each pin means the planner COULD have refused.

    Without the three indexes every pinned read loses its seek (captured
    with sqlite_stat1 absent, like the pins): the four single-table reads
    become full SCANs of their table, and the quiet-retire query's
    per-retirement subquery walks every run while the outer COUNT falls
    back to the rowid PK (``r.id > ?`` stays a rowid range). That is the
    plan family the indexes remove, and what makes "the index name is in
    the plan" a load-bearing assertion rather than a formality.
    """
    _seed_plan_corpus(db)
    with db.connection() as conn:
        for name in (RUNS_ITEM, TRACKED_STATUS, TRACKED_ORIGIN_STORY):
            conn.execute(f"DROP INDEX {name}")
        conn.commit()
        for label, (sql, params, _expected) in PLAN_PINNED_QUERIES.items():
            table = (
                "dream_track_runs"
                if "dream_track_runs" in sql
                else "dream_tracked_items"
            )
            assert f"SCAN {table}" in _plan(conn, sql, params), (
                f"{label}: expected a full scan without the indexes"
            )
        consecutive = _plan(
            conn, CONSECUTIVE_DISPOSITIONS_SQL, (5, "unchanged", 5, "unchanged")
        )
        assert "SCAN b" in consecutive, consecutive
        assert RUNS_ITEM not in consecutive, consecutive


# --- Qodo review on PR #2921 -----------------------------------------------------


def test_subscription_dream_owned_reads_any_status_and_only_dream_rows(db):
    """Qodo #6: one targeted read answers ownership for a subscription --
    a retired dream-created row pins it, an attached row never does."""
    assert db.subscription_dream_owned(7) is False
    _make_item(db, mechanism="page", subscription_id=7, created_by_dreams=0)
    assert db.subscription_dream_owned(7) is False, "attached is not owned"
    owned = _make_item(
        db, mechanism="page", subscription_id=7, created_by_dreams=1)
    db.set_tracked_status(owned, "retired", retired_reason="manual")
    assert db.subscription_dream_owned(7) is True
    assert db.subscription_dream_owned(8) is False


def test_find_tracked_by_story_takes_a_paused_wrapper_only_when_asked(db):
    """Round 3, T5: the two track paths look for the wrapper to REUSE, and
    a watch the sweep paused is still the story's watch. Untrack and the
    tracked badge keep seeing active wrappers only."""
    paused = _make_item(db, origin_story_id=7)
    db.set_tracked_status(paused, "paused")
    retired = _make_item(db, origin_story_id=7)
    db.set_tracked_status(retired, "retired", retired_reason="manual")

    assert db.find_tracked_by_story(7) is None
    assert db.find_tracked_by_story(7, include_paused=True)["id"] == paused

    active = _make_item(db, origin_story_id=7)
    assert db.find_tracked_by_story(7, include_paused=True)["id"] == active, (
        "an active wrapper is preferred over a paused one"
    )
    assert [row["id"] for row in db.list_tracked_for_story(7)] == [
        paused, retired, active]


def test_reminder_event_date_round_trips(db):
    item = _make_item(db)
    assert db.get_tracked_item(item)["reminder_event_date"] is None
    db.set_tracked_reminder_date(item, "2026-10-01")
    assert db.get_tracked_item(item)["reminder_event_date"] == "2026-10-01"


def test_the_pinned_boot_rows_statement_is_the_production_one():
    """The pin above runs this literal; it must be what the store runs."""
    assert DreamsDB._LIVE_BOOT_ROWS_SQL == LIVE_BOOT_ROWS_SQL
    assert DreamsDB._RETIRED_BOOT_ROWS_SQL == RETIRED_BOOT_ROWS_SQL


def test_tracking_boot_rows_are_the_live_wrappers_and_the_backlog_flag(db):
    live = db.create_tracked_item(
        mechanism="page", intent="topic", cadence_seconds=3600,
        subscription_id=7, created_by_dreams=1, egress_marked=0,
        alert_rule_id=3)
    retired = db.create_tracked_item(
        mechanism="page", intent="topic", cadence_seconds=3600,
        subscription_id=8, created_by_dreams=1)
    db.set_tracked_status(retired, "retired", retired_reason="manual")

    rows, gone, backlog = db.tracking_boot_rows()

    assert [(r["id"], r["subscription_id"], r["created_by_dreams"],
             r["alert_rule_id"], r["event_date"], r["suspension"])
            for r in rows] == [(live, 7, 1, 3, None, None)]
    assert gone == [] and backlog is True
    assert db.tracking_boot_state()[1] is True

    db.set_tracked_suspension(retired, '{"subscription": true}')
    _rows, gone, _backlog = db.tracking_boot_rows(retired=True)
    assert [(r["id"], r["suspension"]) for r in gone] == [
        (retired, '{"subscription": true}')]
