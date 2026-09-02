"""Tests for schedule_vocabulary.py -- client/server schedule field-name
translation at the ownership boundary (schedules-handoff PR-5, task 3).
"""

from __future__ import annotations

from hypothesis import given, strategies as st

from tldw_chatbook.Scheduling.schedule_vocabulary import (
    to_local_schedule,
    to_server_schedule,
)


# ----------------------------------------------------------------------
# Per-kind rename tables, both directions (example-based)
# ----------------------------------------------------------------------


def test_one_time_schedule_is_unchanged_both_directions():
    schedule = {"kind": "one_time", "run_at": "2026-06-01T12:00:00+00:00"}
    assert to_server_schedule(schedule) == schedule
    assert to_local_schedule(schedule) == schedule


def test_interval_schedule_renames_every_seconds_to_seconds():
    client = {"kind": "interval", "every_seconds": 3600, "timezone": "UTC"}
    server = {"kind": "interval", "seconds": 3600, "timezone": "UTC"}
    assert to_server_schedule(client) == server
    assert to_local_schedule(server) == client


def test_interval_schedule_start_at_is_server_only_and_passes_through():
    # `start_at` has no client-vocab equivalent; it must survive a
    # server -> local translation untouched (unknown-key passthrough).
    server = {
        "kind": "interval",
        "seconds": 3600,
        "start_at": "2026-06-01T00:00:00+00:00",
    }
    assert to_local_schedule(server) == {
        "kind": "interval",
        "every_seconds": 3600,
        "start_at": "2026-06-01T00:00:00+00:00",
    }


def test_daily_schedule_renames_time_of_day_to_at():
    client = {"kind": "daily", "time_of_day": "09:00", "timezone": "America/New_York"}
    server = {"kind": "daily", "at": "09:00", "timezone": "America/New_York"}
    assert to_server_schedule(client) == server
    assert to_local_schedule(server) == client


def test_weekly_schedule_renames_time_of_day_and_passes_weekday_through():
    client = {"kind": "weekly", "time_of_day": "09:00", "weekday": 2, "timezone": "UTC"}
    server = {"kind": "weekly", "at": "09:00", "weekday": 2, "timezone": "UTC"}
    assert to_server_schedule(client) == server
    assert to_local_schedule(server) == client


def test_weekly_schedule_accepts_server_weekday_name_string_untouched():
    # Server accepts a weekday NAME too (build_trigger passes it straight
    # to APScheduler's day_of_week); no client equivalent exists, so it
    # must pass through as an ordinary value, not be coerced.
    server = {"kind": "weekly", "at": "09:00", "weekday": "mon"}
    assert to_local_schedule(server) == {
        "kind": "weekly",
        "time_of_day": "09:00",
        "weekday": "mon",
    }


def test_cron_schedule_is_unchanged_both_directions():
    schedule = {"kind": "cron", "cron": "0 9 * * 1-5", "timezone": "UTC"}
    assert to_server_schedule(schedule) == schedule
    assert to_local_schedule(schedule) == schedule


# ----------------------------------------------------------------------
# Unknown-kind / unknown-key passthrough and non-mutation
# ----------------------------------------------------------------------


def test_unknown_kind_returns_dict_unchanged():
    schedule = {"kind": "monthly", "day_of_month": 1, "timezone": "UTC"}
    assert to_server_schedule(schedule) == schedule
    assert to_local_schedule(schedule) == schedule


def test_extra_unknown_keys_pass_through_on_a_recognized_kind():
    client = {"kind": "interval", "every_seconds": 120, "notes": "custom field"}
    assert to_server_schedule(client) == {
        "kind": "interval",
        "seconds": 120,
        "notes": "custom field",
    }


def test_to_server_schedule_does_not_mutate_input():
    original = {"kind": "interval", "every_seconds": 90}
    snapshot = dict(original)
    to_server_schedule(original)
    assert original == snapshot


def test_to_local_schedule_does_not_mutate_input():
    original = {"kind": "daily", "at": "08:30"}
    snapshot = dict(original)
    to_local_schedule(original)
    assert original == snapshot


def test_to_server_schedule_returns_a_new_dict_object():
    original = {"kind": "cron", "cron": "0 9 * * 1-5"}
    assert to_server_schedule(original) is not original


# ----------------------------------------------------------------------
# Idempotence and round-trip properties (Hypothesis, seeded random fields)
# ----------------------------------------------------------------------

_TIMEZONES = st.sampled_from(["UTC", "America/New_York", "Europe/Warsaw", "Asia/Tokyo"])
_TIME_OF_DAY = st.builds(
    lambda h, m: f"{h:02d}:{m:02d}", st.integers(0, 23), st.integers(0, 59)
)

_one_time_schedules = st.fixed_dictionaries(
    {
        "kind": st.just("one_time"),
        # `run_at` is opaque to the translator (passthrough both ways) --
        # a realistic ISO datetime string, matching schedule_compute.py's
        # documented "ISO datetime string" convention.
        "run_at": st.datetimes().map(lambda dt: dt.isoformat()),
    }
)
_interval_schedules = st.fixed_dictionaries(
    {
        "kind": st.just("interval"),
        "every_seconds": st.integers(min_value=60, max_value=604800),
    },
    optional={"timezone": _TIMEZONES},
)
_daily_schedules = st.fixed_dictionaries(
    {"kind": st.just("daily"), "time_of_day": _TIME_OF_DAY},
    optional={"timezone": _TIMEZONES},
)
_weekly_schedules = st.fixed_dictionaries(
    {
        "kind": st.just("weekly"),
        "time_of_day": _TIME_OF_DAY,
        "weekday": st.integers(min_value=0, max_value=6),
    },
    optional={"timezone": _TIMEZONES},
)
_cron_schedules = st.fixed_dictionaries(
    {
        "kind": st.just("cron"),
        "cron": st.sampled_from(["0 9 * * 1-5", "*/5 * * * *", "0 0 1 * *"]),
    },
    optional={"timezone": _TIMEZONES},
)

_any_client_schedule = st.one_of(
    _one_time_schedules,
    _interval_schedules,
    _daily_schedules,
    _weekly_schedules,
    _cron_schedules,
)


@given(schedule=_any_client_schedule)
def test_round_trip_to_local_of_to_server_is_identity(schedule):
    """``to_local(to_server(s)) == s`` for every kind ``schedule_compute``
    supports, with seeded-random field values per kind."""
    assert to_local_schedule(to_server_schedule(schedule)) == schedule


@given(schedule=_any_client_schedule)
def test_to_server_schedule_is_idempotent(schedule):
    """A double-call at the push boundary must be harmless: Task 4's
    transfer action reuses this same push path, so an accidental second
    call to ``to_server_schedule`` must not double-translate."""
    once = to_server_schedule(schedule)
    twice = to_server_schedule(once)
    assert twice == once
