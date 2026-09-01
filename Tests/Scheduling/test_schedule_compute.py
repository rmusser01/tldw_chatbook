from datetime import datetime, timedelta, timezone

from tldw_chatbook.Scheduling.schedule_compute import (
    compute_next_run_at,
    schedule_slot_for,
)

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


def test_one_time_future_passes_through_and_past_returns_none():
    future = "2026-09-02T09:00:00+00:00"
    assert compute_next_run_at({"kind": "one_time", "run_at": future}, now=NOW) \
        == datetime(2026, 9, 2, 9, 0, tzinfo=timezone.utc)
    assert compute_next_run_at(
        {"kind": "one_time", "run_at": "2026-08-01T09:00:00+00:00"}, now=NOW
    ) is None


def test_interval_advances_from_now_never_replays():
    nxt = compute_next_run_at({"kind": "interval", "every_seconds": 900}, now=NOW)
    assert nxt == NOW + timedelta(seconds=900)


def test_interval_below_floor_is_invalid():
    assert compute_next_run_at({"kind": "interval", "every_seconds": 30}, now=NOW) is None


def test_daily_respects_timezone():
    nxt = compute_next_run_at(
        {"kind": "daily", "time_of_day": "09:00", "timezone": "America/New_York"},
        now=NOW,  # 12:00 UTC = 08:00 New York -> today's 09:00 NY is next
    )
    assert nxt == datetime(2026, 9, 1, 13, 0, tzinfo=timezone.utc)


def test_weekly_picks_next_weekday_occurrence():
    # 2026-09-01 is a Tuesday (weekday 1). Ask for Monday (0) 09:00 UTC.
    nxt = compute_next_run_at(
        {"kind": "weekly", "weekday": 0, "time_of_day": "09:00", "timezone": "UTC"},
        now=NOW,
    )
    assert nxt == datetime(2026, 9, 7, 9, 0, tzinfo=timezone.utc)


def test_cron_and_junk():
    nxt = compute_next_run_at({"kind": "cron", "cron": "0 9 * * *"}, now=NOW)
    assert nxt == datetime(2026, 9, 2, 9, 0, tzinfo=timezone.utc)
    assert compute_next_run_at({"kind": "cron", "cron": "not cron"}, now=NOW) is None
    assert compute_next_run_at({"kind": "nope"}, now=NOW) is None
    assert compute_next_run_at("junk", now=NOW) is None


def test_slot_string_is_canonical_utc_iso():
    assert schedule_slot_for(datetime(2026, 9, 2, 9, 0, tzinfo=timezone.utc)) \
        == "2026-09-02T09:00:00+00:00"
