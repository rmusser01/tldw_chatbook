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


def test_usage_reserve_takes_a_slot_only_while_under_the_limit(db):
    """Qodo #11 (PR #2921): the budget check and the spend are a read and
    a write under one ``BEGIN IMMEDIATE``, so a slot that is gone cannot be
    spent twice. The return value is how many slots were granted."""
    assert db.usage_reserve("2026-09-22", counter="searches", limit=2) == 1
    assert db.usage_reserve("2026-09-22", counter="searches", limit=2) == 1
    assert db.usage_reserve("2026-09-22", counter="searches", limit=2) == 0
    assert db.usage_reserve("2026-09-22", counter="llm_calls", limit=1) == 1
    assert db.usage_reserve("2026-09-22", counter="llm_calls", limit=1) == 0
    assert db.usage_reserve("2026-09-22", counter="llm_calls", limit=0) == 0
    assert db.usage_get("2026-09-22") == {"searches": 2, "llm_calls": 1}
    # A refused reservation on an untouched day spends nothing.
    assert db.usage_reserve("2026-09-23", counter="searches", limit=0) == 0
    assert db.usage_get("2026-09-23") == {"searches": 0, "llm_calls": 0}
    with pytest.raises(ValueError):
        db.usage_reserve("2026-09-22", counter="stories", limit=5)


def test_usage_reserve_grants_as_many_of_several_slots_as_remain(db):
    """A request for several units is granted the part of it the day still
    has. (No production caller asks for more than one any more: since
    round 3, item 2, the cycle reserves ONE slot before each search, as it
    does before each story. The ``count`` parameter is kept and pinned.)"""
    assert db.usage_reserve(
        "2026-09-22", counter="searches", limit=5, count=3) == 3
    assert db.usage_reserve(
        "2026-09-22", counter="searches", limit=5, count=3) == 2
    assert db.usage_reserve(
        "2026-09-22", counter="searches", limit=5, count=3) == 0
    assert db.usage_reserve(
        "2026-09-22", counter="searches", limit=5, count=0) == 0
    assert db.usage_get("2026-09-22")["searches"] == 5
    # A counter already past its limit (a lowered cap) grants nothing and
    # is never pushed backwards.
    assert db.usage_reserve(
        "2026-09-22", counter="searches", limit=2, count=1) == 0
    assert db.usage_get("2026-09-22")["searches"] == 5


def test_usage_reserve_holds_the_write_lock_from_its_read_to_its_write(
        db, tmp_path):
    """Review round 3, item 3b -- the deterministic form of "never
    oversells". The reservation reads the counter and then writes it; what
    makes that atomic is that both happen under ``BEGIN IMMEDIATE``. So a
    second connection that tries to write at the exact moment the read
    runs must be refused.

    What a reservation WITHOUT the transaction does here (run, round 5,
    N3): the trace callback fires as the read starts, so the other
    connection's +3 is written first and the read already sees it -- 1 +
    3 = 4 of 5 spent; the reservation then grants 1 instead of 4 and the
    counter ends at 5. It does not overspend in this sequence; it lets a
    second writer in between, and that is what the first two assertions
    catch (``outcome`` is "the other connection wrote", ``granted`` is 1).
    The last assertion holds either way. The threaded test further down
    cannot show the race reliably; it is a smoke test and says so."""
    day = "2026-09-22"
    assert db.usage_reserve(day, counter="searches", limit=5) == 1
    other = sqlite3.connect(
        tmp_path / "dreams.sqlite", timeout=0, isolation_level=None)
    outcome: list[str] = []

    def at_the_read(statement: str) -> None:
        if not statement.startswith("SELECT searches"):
            return
        try:
            other.execute(
                "UPDATE dream_daily_usage SET searches = searches + 3"
                " WHERE local_date = ?", (day,))
            outcome.append("the other connection wrote")
        except sqlite3.OperationalError as exc:
            outcome.append(str(exc))

    reserving = db._held_connection()
    reserving.set_trace_callback(at_the_read)
    try:
        granted = db.usage_reserve(day, counter="searches", limit=5, count=4)
    finally:
        reserving.set_trace_callback(None)
        other.close()

    assert outcome == ["database is locked"]
    assert granted == 4
    assert db.usage_get(day)["searches"] == 5, "never past the limit"


def test_usage_reserve_never_oversells_across_threads(db):
    """Each worker thread holds its own connection, exactly as concurrent
    track checks do; however they interleave, the limit holds. A smoke
    test only: it is timing-dependent (the deterministic proof is the
    write-lock test above)."""
    import threading

    start = threading.Barrier(8)
    granted: list[bool] = []

    def worker():
        start.wait(timeout=10)
        granted.append(
            db.usage_reserve("2026-09-22", counter="searches", limit=3))
        db.close()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert sorted(granted) == [0] * 5 + [1] * 3
    assert db.usage_get("2026-09-22")["searches"] == 3


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


# --- Review of #2921, round 4, I5: a collection id can come back ------------------
#
# ``dreams_collections.id`` is a plain INTEGER PRIMARY KEY, so the id of a
# deleted row is handed out again. A halted cycle deletes its own empty
# collection; a second process that was still working with that id then
# wrote into whatever collection got the id next.


def _story(db, collection_id, **kwargs):
    return db.insert_story(
        collection_id, title="t", url="https://dreams.test/s", snippet="",
        body="b", status="complete", source="web", kind="content",
        event_date=None, location=None, matched_topics=[], query="q",
        **kwargs)


def _claim_of(db, local_date):
    row = db.get_collection_by_date(local_date)
    return int(row["id"]), (local_date, row["created_at"])


def test_a_story_is_refused_when_its_collection_id_now_names_another_collection(
        db):
    stale_id, stale_claim = _claim_of(
        db, "2026-09-22") if db.create_collection(
            "2026-09-22", "scheduled", "d") else (None, None)
    assert db.delete_empty_collection(stale_id) is True
    with db.transaction() as conn:  # the next collection is made later
        pass
    new_id = db.create_collection("2026-09-23", "scheduled", "d2")
    assert new_id == stale_id, "fixture: the id was handed out again"

    with pytest.raises(sqlite3.IntegrityError):
        _story(db, stale_id, claim=stale_claim)

    assert db.list_stories(new_id) == [], "nothing landed in the new one"
    # The collection that IS the one claimed takes the story.
    _, new_claim = _claim_of(db, "2026-09-23")
    assert _story(db, new_id, claim=new_claim)


def test_a_story_is_refused_for_a_reused_id_on_the_same_date(db):
    """The date alone does not tell the two apart: the halted cycle hands
    its date back, and the next cycle of the same day takes date AND id."""
    db.create_collection("2026-09-22", "scheduled", "d")
    stale_id, stale_claim = _claim_of(db, "2026-09-22")
    db.delete_empty_collection(stale_id)
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO dreams_collections"
            " (id, local_date, status, profile_digest, trigger, created_at)"
            " VALUES (?, '2026-09-22', 'generating', 'd2', 'catchup',"
            " '2026-09-22T09:00:00.000Z')", (stale_id,))

    with pytest.raises(sqlite3.IntegrityError):
        _story(db, stale_id, claim=stale_claim)

    assert db.list_stories(stale_id) == []


def test_a_final_status_is_not_written_to_another_collection(db):
    db.create_collection("2026-09-22", "scheduled", "d")
    stale_id, stale_claim = _claim_of(db, "2026-09-22")
    db.delete_empty_collection(stale_id)
    new_id = db.create_collection("2026-09-23", "scheduled", "d2")
    assert new_id == stale_id, "fixture"

    assert db.set_collection_status(
        stale_id, "failed", claim=stale_claim) is False

    assert db.get_collection_by_date("2026-09-23")["status"] == "generating"
    _, new_claim = _claim_of(db, "2026-09-23")
    assert db.set_collection_status(
        new_id, "complete", claim=new_claim) is True
