"""Tests for `Subscriptions/item_dates.py` (task-3072, plan Task 2).

The reader-row date foundation: one parser rule (naive = UTC), the
effective date, relative rendering, and local-day buckets for group
headers.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tldw_chatbook.Subscriptions.item_dates import (
    day_bucket,
    effective_date,
    parse_stored_datetime,
    relative_time,
)

NOW = datetime(2026, 8, 7, 10, 30, tzinfo=timezone.utc)


class TestParseStoredDatetime:
    def test_naive_iso_is_attached_to_utc(self):
        parsed = parse_stored_datetime("2026-08-07T09:41:00")
        assert parsed == datetime(2026, 8, 7, 9, 41, tzinfo=timezone.utc)

    def test_aware_iso_keeps_its_offset(self):
        parsed = parse_stored_datetime("2026-08-07T05:41:00-04:00")
        assert parsed == datetime(2026, 8, 7, 9, 41, tzinfo=timezone.utc)

    def test_z_suffix(self):
        parsed = parse_stored_datetime("2026-08-07T09:41:00Z")
        assert parsed == datetime(2026, 8, 7, 9, 41, tzinfo=timezone.utc)

    def test_naive_and_aware_of_the_same_instant_agree(self):
        naive = parse_stored_datetime("2026-08-07T09:41:00")
        aware = parse_stored_datetime("2026-08-07T09:41:00+00:00")
        assert naive == aware

    def test_unparseable_returns_none(self):
        assert parse_stored_datetime("not a date") is None
        assert parse_stored_datetime("") is None
        assert parse_stored_datetime(None) is None

    def test_datetime_passthrough(self):
        naive = datetime(2026, 8, 7, 9, 41)
        assert parse_stored_datetime(naive) == naive.replace(tzinfo=timezone.utc)


class TestEffectiveDate:
    def test_published_date_wins(self):
        item = {
            "published_date": "2026-08-06T15:00:00+00:00",
            "created_at": "2026-08-07T09:00:00+00:00",
        }
        assert effective_date(item) == datetime(2026, 8, 6, 15, 0, tzinfo=timezone.utc)

    def test_falls_back_to_created_at(self):
        item = {"published_date": None, "created_at": "2026-08-07T09:00:00+00:00"}
        assert effective_date(item) == datetime(2026, 8, 7, 9, 0, tzinfo=timezone.utc)

    def test_missing_published_key_falls_back(self):
        item = {"created_at": "2026-08-07T09:00:00+00:00"}
        assert effective_date(item) == datetime(2026, 8, 7, 9, 0, tzinfo=timezone.utc)

    def test_unparseable_published_falls_back_to_created(self):
        item = {"published_date": "garbage", "created_at": "2026-08-07T09:00:00+00:00"}
        assert effective_date(item) == datetime(2026, 8, 7, 9, 0, tzinfo=timezone.utc)

    def test_neither_present_returns_none(self):
        assert effective_date({}) is None
        assert effective_date({"published_date": None, "created_at": None}) is None


class TestRelativeTime:
    def test_same_day_renders_clock_time(self):
        # Built from NOW.astimezone() so the case is "later the same LOCAL
        # day" in whatever zone the suite runs in.
        dt = NOW.astimezone().replace(hour=9, minute=41, second=0, microsecond=0)
        assert relative_time(dt, now=NOW) == "9:41 AM"

    def test_yesterday(self):
        dt = NOW.astimezone() - timedelta(days=1)
        assert relative_time(dt, now=NOW) == "Yesterday"

    def test_same_year_older_day(self):
        dt = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
        assert relative_time(dt, now=NOW) == dt.astimezone().strftime("%b %d")

    def test_other_year(self):
        dt = datetime(2025, 12, 31, 12, 0, tzinfo=timezone.utc)
        assert relative_time(dt, now=NOW) == dt.astimezone().strftime("%Y-%m-%d")

    def test_none_renders_dash(self):
        assert relative_time(None, now=NOW) == "-"


class TestDayBucket:
    def test_today(self):
        dt = NOW.astimezone().replace(hour=1, minute=0, second=0, microsecond=0)
        assert day_bucket(dt, now=NOW) == "Today"

    def test_yesterday(self):
        dt = NOW.astimezone() - timedelta(days=1)
        assert day_bucket(dt, now=NOW) == "Yesterday"

    def test_older_day_is_a_date_label(self):
        dt = NOW.astimezone() - timedelta(days=6)
        assert day_bucket(dt, now=NOW) == dt.strftime("%B %d, %Y")

    def test_future_dated_buckets_into_today(self):
        # Bad feed clocks: an item "published" tomorrow must not float above
        # the list under a header of its own (the spec's Dates section).
        dt = NOW + timedelta(days=3)
        assert day_bucket(dt, now=NOW) == "Today"

    def test_none_is_unknown_date(self):
        assert day_bucket(None, now=NOW) == "Unknown date"

    def test_buckets_are_local_day_not_utc_day(self):
        # 01:30 UTC is still yesterday anywhere west of UTC+1:30; the bucket
        # must be computed in the viewer's zone, matching humane_time.
        dt = datetime(2026, 8, 7, 1, 30, tzinfo=timezone.utc)
        local_date = dt.astimezone().date()
        today = NOW.astimezone().date()
        expected = "Today" if local_date == today else (
            "Yesterday" if local_date == today - timedelta(days=1) else "other"
        )
        assert day_bucket(dt, now=NOW) == expected
