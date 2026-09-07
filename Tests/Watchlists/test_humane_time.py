"""Unit tests for `humane_time.py` (TASK-2308).

One formatter, one house style, for every Watchlists table timestamp.
`now=` is threaded through every relative-format test so nothing here reads
the real clock -- the whole point of the module's own `now` parameter.

`humane_timestamp` converts to the machine's LOCAL zone (`.astimezone()`)
before formatting, so the exact `HH:MM` in its relative-format branches
(Today/Yesterday/same-year) depends on wherever the test happens to run.
`_pin_local_zone_to_utc` pins the process's local zone to UTC for the
duration of this module's tests, the same way `now=` pins the clock --
without it these tests are only correct on a machine whose local zone
happens to already be UTC, which is exactly the kind of "passed on my
machine" gap this batch's UAT was born from.
"""

import time
from datetime import datetime, timezone

import pytest
from tldw_chatbook.UI.Watchlists_Modules.humane_time import (
    humane_timestamp,
    parse_timestamp,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _pin_local_zone_to_utc(monkeypatch):
    """Force `datetime.astimezone()` to be a no-op for this module's tests.

    `time.tzset()` is POSIX-only (absent on Windows); this repo's dev/CI
    environment is macOS/Linux (see the venv-only pytest convention), so the
    fixture applies unconditionally rather than silently degrading to an
    unpinned, machine-dependent run.
    """
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    yield
    time.tzset()


# --- parse_timestamp ------------------------------------------------------


def test_parses_iso_with_microseconds_and_utc_offset():
    parsed = parse_timestamp("2026-08-04T18:15:22.123456+00:00")
    assert parsed == datetime(2026, 8, 4, 18, 15, 22, 123456, tzinfo=timezone.utc)


def test_parses_iso_without_microseconds():
    parsed = parse_timestamp("2026-08-04T18:15:22+00:00")
    assert parsed == datetime(2026, 8, 4, 18, 15, 22, tzinfo=timezone.utc)


def test_parses_a_z_suffix_as_utc():
    parsed = parse_timestamp("2026-08-04T18:15:22Z")
    assert parsed == datetime(2026, 8, 4, 18, 15, 22, tzinfo=timezone.utc)


def test_parses_sqlites_space_separated_current_timestamp_shape():
    """SQLite's `CURRENT_TIMESTAMP` writes a space, not a `T` -- the exact
    shape the Artifacts pane's "humane-looking" column actually stored."""
    parsed = parse_timestamp("2026-08-04 18:22:44")
    assert parsed == datetime(2026, 8, 4, 18, 22, 44, tzinfo=timezone.utc)


def test_a_naive_value_is_treated_as_utc_not_local():
    """Every writer behind these columns stores UTC -- assuming local would
    silently shift every pre-existing row by the viewer's own offset."""
    parsed = parse_timestamp("2026-08-04T18:15:22")
    assert parsed is not None
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0


def test_a_naive_python_datetime_is_also_treated_as_utc():
    parsed = parse_timestamp(datetime(2026, 8, 4, 18, 15, 22))
    assert parsed == datetime(2026, 8, 4, 18, 15, 22, tzinfo=timezone.utc)


def test_an_aware_python_datetime_is_returned_unchanged():
    aware = datetime(2026, 8, 4, 18, 15, 22, tzinfo=timezone.utc)
    assert parse_timestamp(aware) is aware


def test_a_bare_date_becomes_midnight_utc():
    from datetime import date

    parsed = parse_timestamp(date(2026, 7, 18))
    assert parsed == datetime(2026, 7, 18, 0, 0, 0, tzinfo=timezone.utc)


def test_none_and_empty_string_are_unparseable():
    assert parse_timestamp(None) is None
    assert parse_timestamp("") is None
    assert parse_timestamp("   ") is None


def test_garbage_text_is_unparseable():
    assert parse_timestamp("not a timestamp at all") is None


# --- humane_timestamp: empty / unparseable --------------------------------


def test_empty_value_renders_the_dash():
    assert humane_timestamp(None) == "-"
    assert humane_timestamp("") == "-"
    assert humane_timestamp("   ") == "-"


def test_unparseable_value_passes_through_unchanged():
    """Rendering a dash over a value that exists would be a lie; raising
    inside a `compose()` exits the whole application."""
    assert humane_timestamp("not-a-real-timestamp") == "not-a-real-timestamp"


# --- humane_timestamp: relative formatting (clock-controlled) -------------


def test_today_renders_as_today_hh_mm():
    now = datetime(2026, 8, 4, 20, 0, 0, tzinfo=timezone.utc)
    out = humane_timestamp("2026-08-04T18:15:00+00:00", now=now)
    assert out == "Today 18:15"


def test_yesterday_renders_as_yesterday_hh_mm():
    now = datetime(2026, 8, 4, 9, 0, 0, tzinfo=timezone.utc)
    out = humane_timestamp("2026-08-03T18:15:00+00:00", now=now)
    assert out == "Yesterday 18:15"


def test_an_earlier_day_this_year_renders_month_day_and_time():
    now = datetime(2026, 8, 4, 9, 0, 0, tzinfo=timezone.utc)
    out = humane_timestamp("2026-07-18T09:30:00+00:00", now=now)
    assert out == "Jul 18 09:30"


def test_a_prior_year_renders_as_a_bare_date():
    now = datetime(2026, 8, 4, 9, 0, 0, tzinfo=timezone.utc)
    out = humane_timestamp("2025-12-31T23:59:00+00:00", now=now)
    assert out == "2025-12-31"


def test_a_date_only_value_renders_as_that_calendar_day_not_shifted():
    """A date-only value must never be converted to local time: doing so
    could render midnight UTC as the day before in a western timezone."""
    now = datetime(2026, 8, 4, 9, 0, 0, tzinfo=timezone.utc)
    out = humane_timestamp("2026-07-18", now=now)
    assert out == "2026-07-18"


def test_more_than_two_days_ago_but_still_this_year_is_not_today_or_yesterday():
    now = datetime(2026, 8, 4, 9, 0, 0, tzinfo=timezone.utc)
    out = humane_timestamp("2026-08-01T09:00:00+00:00", now=now)
    assert out not in ("Today 09:00", "Yesterday 09:00")
    assert out == "Aug 01 09:00"


def test_a_naive_now_is_treated_as_utc_too():
    """`now=` documents that it defaults to the current UTC instant; a naive
    `now` passed explicitly must be handled the same way `parse_timestamp`
    handles a naive stored value, not raise on the mixed-awareness compare."""
    now = datetime(2026, 8, 4, 20, 0, 0)  # naive
    out = humane_timestamp("2026-08-04T18:15:00+00:00", now=now)
    assert out == "Today 18:15"
