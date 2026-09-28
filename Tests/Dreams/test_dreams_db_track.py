# Tests/Dreams/test_dreams_db_track.py
"""DreamsDB schema v2 behavior: tracked items, track runs, v1->v2 upgrade."""
import sqlite3

import pytest

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
    """Rewind an open v2 database file to a faithful v1 state.

    v2 is additive only, so dropping the two track tables (and their indexes)
    and re-stamping the version row leaves exactly what a Phase 1 writer
    would have on disk.
    """
    with db.transaction() as conn:
        conn.execute("DROP INDEX IF EXISTS idx_dream_track_runs_item")
        conn.execute("DROP INDEX IF EXISTS idx_dream_tracked_status")
        conn.execute("DROP INDEX IF EXISTS idx_dream_tracked_origin_story")
        conn.execute("DROP TABLE IF EXISTS dream_track_runs")
        conn.execute("DROP TABLE IF EXISTS dream_tracked_items")
        # A v1-created file holds exactly one stamp: version 1 (a fresh v2
        # file only ever writes version 2, so rewriting the table, not
        # deleting one row, is the faithful old-file shape).
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (1)")


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

    # Reopening the v1 file must upgrade it in place via the additive DDL.
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
        assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 2
    # The upgraded file is fully usable: track CRUD works and v1 data survived.
    item_id = reopened.create_tracked_item(
        mechanism="page", intent="deal", cadence_seconds=60
    )
    reopened.insert_track_run(item_id, status="baseline", digest_hash="h")
    assert reopened.count_active_tracked() == 1
    assert reopened.get_collection_by_date("2026-09-22")["status"] == "generating"
    reopened.close()
