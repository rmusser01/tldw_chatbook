"""Daily Report rows: the cross-watchlist DB join and the view derivation.

Real `SubscriptionsDB` under `tmp_path`, real write paths for seeding (the
persist seams are the ones production uses); the only faked collaborator is
the path-safety guard, and only in the test that pins its effect.
"""

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions import daily_reports_view
from tldw_chatbook.Subscriptions.daily_reports_view import list_recent_reports
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService

pytestmark = pytest.mark.unit


def _db(tmp_path) -> SubscriptionsDB:
    """File-backed DB -- thread-local connections make `:memory:` unusable."""
    return SubscriptionsDB(tmp_path / "subs.db", "test")


def _watchlist(db, name: str) -> int:
    return int(WatchlistBundleService(db).create(name)["id"])


def _briefing(db, watchlist_id: int, *, status: str = "complete") -> int:
    briefing_id = db.insert_briefing(watchlist_id)
    db.update_briefing(
        briefing_id,
        status=status,
        body_markdown="## Daily Brief\n\nOne story [item 1].",
        item_count=1,
    )
    return briefing_id


def _complete_script_with_audio(db, briefing_id: int, file_path: str) -> None:
    script_id = db.insert_briefing_script(
        briefing_id, preset_id=None, preset_name="Daily Brief",
        roster_snapshot_json="[]",
    )
    db.update_briefing_script(script_id, status="complete", turns_json="[]")
    audio_id = db.create_briefing_audio(script_id, voice_snapshot_json="[]")
    db.update_briefing_audio(
        audio_id, status="complete", file_path=file_path,
        duration_seconds=1.0, turn_count=1,
    )


def test_list_recent_briefings_orders_newest_first_across_watchlists(tmp_path):
    db = _db(tmp_path)
    w1, w2 = _watchlist(db, "Tech"), _watchlist(db, "World")
    b1 = _briefing(db, w1)
    b2 = _briefing(db, w2, status="empty")
    b3 = _briefing(db, w1)

    rows = db.list_recent_briefings(limit=10)

    assert [r["briefing_id"] for r in rows] == [b3, b2, b1]  # same-second ties break on id DESC
    by_id = {r["briefing_id"]: r for r in rows}
    assert by_id[b2]["watchlist_name"] == "World"
    assert by_id[b2]["status"] == "empty"
    assert by_id[b1]["item_count"] == 1


def test_list_recent_briefings_rejects_bad_limit(tmp_path):
    db = _db(tmp_path)
    with pytest.raises(ValueError):
        db.list_recent_briefings(limit=0)
    with pytest.raises(ValueError):
        db.list_recent_briefings(limit=True)


def test_list_recent_briefings_honors_limit(tmp_path):
    db = _db(tmp_path)
    w = _watchlist(db, "Tech")
    for _ in range(3):
        _briefing(db, w)
    assert len(db.list_recent_briefings(limit=2)) == 2


def test_report_rows_surface_audio_only_through_the_safety_guard(tmp_path, monkeypatch):
    db = _db(tmp_path)
    w = _watchlist(db, "Tech")
    b1 = _briefing(db, w)
    _complete_script_with_audio(db, b1, "/armored/briefing_audio/script-1-audio-1.wav")
    b2 = _briefing(db, w)  # text-only report

    monkeypatch.setattr(daily_reports_view, "audio_file_path_is_safe", lambda p: True)
    rows = list_recent_reports(db, limit=10)
    by_id = {r["id"]: r for r in rows}
    assert by_id[b1]["has_audio"] is True
    assert by_id[b1]["audio_file_path"] == "/armored/briefing_audio/script-1-audio-1.wav"
    assert by_id[b2]["has_audio"] is False
    assert by_id[b2]["audio_file_path"] is None

    monkeypatch.setattr(daily_reports_view, "audio_file_path_is_safe", lambda p: False)
    rows = list_recent_reports(db, limit=10)
    by_id = {r["id"]: r for r in rows}
    assert by_id[b1]["has_audio"] is False
    assert by_id[b1]["audio_file_path"] is None  # unsafe path never reaches the UI


def test_report_rows_label_watchlist_status_and_audio(tmp_path, monkeypatch):
    db = _db(tmp_path)
    w = _watchlist(db, "Daily Brief")
    b1 = _briefing(db, w)
    _complete_script_with_audio(db, b1, "/x/y.wav")
    _briefing(db, w, status="failed")

    monkeypatch.setattr(daily_reports_view, "audio_file_path_is_safe", lambda p: True)
    rows = list_recent_reports(db, limit=10)

    assert rows[1]["label"].startswith("Daily Brief — ")
    assert "audio" in rows[1]["label"]
    assert "(failed)" in rows[0]["label"]
