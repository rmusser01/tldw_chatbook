# Tests/Dreams/test_dreams_db.py
"""DreamsDB schema v1 behavior: date bucketing, story rows, seen ledger, usage."""
import sqlite3
from datetime import datetime, timezone

import pytest

from tldw_chatbook.DB.Dreams_DB import DreamsDB


@pytest.fixture()
def db(tmp_path):
    database = DreamsDB(tmp_path / "dreams.sqlite", "test-client")
    yield database
    database.close()


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def test_create_collection_enforces_one_row_per_local_date(db):
    first = db.create_collection("2026-09-22", "scheduled", "digest-1")
    assert first is not None
    assert db.create_collection("2026-09-22", "manual", "digest-2") is None
    assert db.get_collection_by_date("2026-09-22")["id"] == first


def test_insert_and_list_stories_roundtrip_with_metadata(db):
    cid = db.create_collection("2026-09-22", "scheduled", "d")
    sid = db.insert_story(
        cid, title="Cheap flights to Japan", url="https://example.com/f",
        snippet="Fares from $89", body="A story about fares.", status="complete",
        source="web", kind="deal", event_date=None, location="Japan",
        matched_topics=["visit japan"], query="cheap flights japan",
    )
    stories = db.list_stories(cid)
    assert [s["id"] for s in stories] == [sid]
    assert stories[0]["kind"] == "deal"
    assert stories[0]["kept"] == 0


def test_get_story_returns_full_row_and_none_for_missing(db):
    cid = db.create_collection("2026-09-22", "scheduled", "d")
    sid = db.insert_story(
        cid, title="Cheap flights to Japan", url="https://example.com/f",
        snippet="Fares from $89", body="A story about fares.", status="complete",
        source="web", kind="deal", event_date=None, location="Japan",
        matched_topics=["visit japan"], query="cheap flights japan",
    )
    story = db.get_story(sid)
    assert story["title"] == "Cheap flights to Japan"
    assert story["query"] == "cheap flights japan"
    assert story["matched_topics"] == ["visit japan"], (
        "the JSON column must come back parsed"
    )
    assert story["local_date"] == "2026-09-22", (
        "the owning collection's local date rides the row"
    )
    assert db.get_story(99999) is None


def test_insert_story_rejects_duplicate_url_within_collection(db):
    cid = db.create_collection("2026-09-22", "scheduled", "d")
    db.insert_story(cid, title="a", url="https://x/1", snippet="", body="",
                    status="complete", source="web", kind="content",
                    event_date=None, location=None, matched_topics=[], query="q")
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_story(cid, title="a2", url="https://x/1", snippet="", body="",
                        status="complete", source="web", kind="content",
                        event_date=None, location=None, matched_topics=[], query="q")


def test_seen_ledger_filters_and_prunes(db):
    db.seen_upsert([("https://a", "t-a"), ("https://b", "t-b")])
    assert db.seen_filter_unseen(["https://a", "https://c"]) == {"https://c"}
    assert db.prune_seen("2999-01-01T00:00:00+00:00") == 2


def test_usage_bump_accumulates_per_local_date(db):
    db.usage_bump("2026-09-22", searches=3)
    db.usage_bump("2026-09-22", llm_calls=5)
    assert db.usage_get("2026-09-22") == {"searches": 3, "llm_calls": 5}
    assert db.usage_get("2026-09-23") == {"searches": 0, "llm_calls": 0}


def test_fail_stale_generating_marks_only_old_generating_rows(db):
    db.create_collection("2026-09-22", "scheduled", "d")
    cutoff = _iso_now()  # reclaim horizon, taken before the fresh row exists
    db.create_collection("2026-09-23", "scheduled", "d2")  # stays untouched
    # Backdate the first row via direct SQL so only it predates the horizon
    # (the 15-minute stale-reclaim semantics the cycle relies on).
    with db.transaction() as conn:
        conn.execute(
            "UPDATE dreams_collections SET created_at = '2020-01-01T00:00:00+00:00'"
            " WHERE local_date = '2026-09-22'"
        )
    assert db.fail_stale_generating(cutoff) == 1
    assert db.get_collection_by_date("2026-09-22")["status"] == "failed"
    assert db.get_collection_by_date("2026-09-23")["status"] == "generating"


# --- Goal query angle (task-33165: feedback steers angles, never weights) -----


def _profile_row(db, facet, text):
    with db.connection() as conn:
        row = conn.execute(
            "SELECT * FROM dream_interest_profile WHERE facet = ? AND text = ?",
            (facet, text),
        ).fetchone()
    return dict(row) if row is not None else None


def test_set_goal_query_angle_writes_goal_rows_only_and_never_weight(db):
    db.upsert_profile_entry("goal", "visit japan", weight=1.0, searchable=1,
                            source="user")
    db.upsert_profile_entry("goal", "private wish", weight=0.9, searchable=0,
                            source="user")
    db.upsert_profile_entry("topic", "visit japan", weight=0.5, searchable=1,
                            source="user")

    db.set_goal_query_angle("goal", "visit japan", angle="avoid: event")
    assert _profile_row(db, "goal", "visit japan")["query_angle"] == \
        "avoid: event"
    # The same text as a TOPIC stays untouched: the write is pinned to goal
    # rows, and so is every other goal.
    assert _profile_row(db, "topic", "visit japan")["query_angle"] is None
    assert _profile_row(db, "goal", "private wish")["query_angle"] is None
    # The angle write never carries weight math with it (goals stay immune).
    assert _profile_row(db, "goal", "visit japan")["weight"] == 1.0


def test_set_goal_query_angle_replaces_and_stamps_updated_at(db):
    db.upsert_profile_entry("goal", "see a show", weight=1.0, searchable=1,
                            source="user")
    before = _profile_row(db, "goal", "see a show")["updated_at"]
    db.set_goal_query_angle("goal", "see a show", angle="avoid: event")
    db.set_goal_query_angle("goal", "see a show", angle="prefer: deal")
    row = _profile_row(db, "goal", "see a show")
    assert row["query_angle"] == "prefer: deal", "one note REPLACES, not adds"
    assert row["updated_at"] >= before


def test_set_goal_query_angle_none_clears_the_note(db):
    """``angle=None`` is a real CLEAR, not a skipped update."""
    db.upsert_profile_entry("goal", "visit japan", weight=1.0, searchable=1,
                            source="user")
    db.set_goal_query_angle("goal", "visit japan", angle="prefer: deal")
    db.set_goal_query_angle("goal", "visit japan", angle=None)
    assert _profile_row(db, "goal", "visit japan")["query_angle"] is None


def test_set_goal_query_angle_rejects_non_goal_facet(db):
    with pytest.raises(ValueError):
        db.set_goal_query_angle("topic", "rust tui", angle="prefer: deal")


def test_set_goal_query_angle_missing_row_is_benign_noop(db):
    db.set_goal_query_angle("goal", "no such goal", angle="prefer: deal")
    assert _profile_row(db, "goal", "no such goal") is None
