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
    # (the 15-minute stale-generating reclaim semantics the cycle relies on).
    with db.transaction() as conn:
        conn.execute(
            "UPDATE dreams_collections SET created_at = '2020-01-01T00:00:00+00:00'"
            " WHERE local_date = '2026-09-22'"
        )
    assert db.fail_stale_generating(cutoff) == 1
    assert db.get_collection_by_date("2026-09-22")["status"] == "failed"
    assert db.get_collection_by_date("2026-09-23")["status"] == "generating"
