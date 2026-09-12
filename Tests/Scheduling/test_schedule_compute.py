import os
import time
from datetime import datetime, timedelta, timezone

import pytest

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


# --- Finding E: no-explicit-timezone path must not use a fixed-offset snapshot ---


def test_daily_no_explicit_timezone_localizes_per_candidate_date_not_a_fixed_snapshot():
    """Finding E: `_machine_timezone()` used to memoize
    `datetime.now().astimezone().tzinfo` -- a FIXED UTC-offset snapshot of
    whatever the real wall-clock offset happens to be when the function
    runs -- and apply that SAME offset to every computed candidate
    regardless of the candidate's own date. That is wrong across a DST
    boundary. The fix instead resolves each candidate's offset per its own
    date via a naive `datetime.astimezone()` call (which, per Python's
    documented behavior, applies the platform's local DST rule for
    whatever date the naive value actually carries).

    Proof: force the process's local zone to America/New_York (a DST
    zone) for the duration of this test, then compute the SAME
    no-explicit-timezone daily schedule for a `now` inside DST (July) and
    one outside DST (January). A fixed-offset snapshot would use one
    offset for both regardless of which `now` was passed; per-date
    localization must pick EDT for July and EST for January -- a
    different UTC hour for the identical local wall-clock target.
    """
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset() is POSIX-only; cannot force the local zone here")

    original_tz = os.environ.get("TZ")
    os.environ["TZ"] = "America/New_York"
    time.tzset()
    try:
        schedule = {"kind": "daily", "time_of_day": "09:00"}  # no explicit timezone
        winter_next = compute_next_run_at(
            schedule, now=datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        )
        summer_next = compute_next_run_at(
            schedule, now=datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
        )
        assert winter_next is not None
        assert summer_next is not None
        # EST is UTC-5, EDT is UTC-4: the same 09:00 local wall-clock
        # target lands at a different UTC hour depending on which one is
        # in effect for that candidate's own date -- proof the offset was
        # resolved per-date, not from one snapshot shared by both calls.
        assert winter_next.hour != summer_next.hour
    finally:
        if original_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original_tz
        time.tzset()


# --- Finding F: DST fall-back must never compute a past instant ---------------


def test_weekly_dst_fallback_final_guard_never_returns_a_past_instant():
    """Finding F: during the America/New_York 2026-11-01 fall-back, wall
    clock 01:00-01:59 occurs twice (EDT then EST). `_compute_weekly` adds
    `candidate += timedelta(days=days_ahead)` unconditionally -- even when
    `days_ahead == 0` -- and Python's documented arithmetic behavior on an
    aware datetime always resets the result's `fold` to 0. That can flip a
    correctly-resolved EST (fold=1) candidate to an EDT (fold=0)
    interpretation, producing a UTC instant up to an hour earlier than
    intended. Regression-tested here with a `now` that lands in the
    second (EST, fold=1) occurrence of 01:00, asking for that same
    Sunday's 01:05 -- the final guard must detect a result <= now and
    advance one more period rather than ever handing back a past instant.
    """
    now = datetime(2026, 11, 1, 6, 0, tzinfo=timezone.utc)  # 01:00 EST, post-fallback
    nxt = compute_next_run_at(
        {
            "kind": "weekly",
            "weekday": 6,  # Sunday -- 2026-11-01 is a Sunday
            "time_of_day": "01:05",
            "timezone": "America/New_York",
        },
        now=now,
    )
    assert nxt is not None
    assert nxt > now
