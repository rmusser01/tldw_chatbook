"""Stored-date reading for the Watchlists reader (task-3072).

The article list renders two things from an item's dates: a relative time
on the row ("9:41 AM", "Yesterday") and a local-day group header ("Today",
"Yesterday", "August 01, 2026"). Both need the same three decisions made
in exactly one place:

**One parser rule: naive means UTC.** `published_date` is stored ISO-8601
but mixed naive/aware -- `monitoring_engine._parse_date` returns
timezone-less strings for several feed formats while URL monitors store
aware UTC. Attaching naive values to the local zone would silently shift
every such row by the user's offset. This parser is MOVED OUT of
`UI/Watchlists_Modules/humane_time.py` (TASK-2308), where it was born:
parsing a stored value is a data-layer concern, and the layer direction is
UI -> Subscriptions, never the reverse -- a helper in `Subscriptions/`
cannot import from `UI/`. `humane_time` re-exports this function as
`parse_timestamp`; its public API is unchanged.

**The effective date.** `created_at` is INGEST time -- every item one
check produces carries the same value to the microsecond. TASK-2308
already showed that presenting it under a "Published" heading is how a
reader loses trust in the list, so a row sorts and renders by
`published_date`, falling back to `created_at` only when the feed itself
omitted a date (feed parsing returns `None` rather than defaulting to
"now").

**Bucketing is Python-side, not SQL.** Group headers and "Today" semantics
are computed over displayed rows in the viewer's local zone. The stored
tz strings cannot be trusted to SQL date functions (the mixed
naive/aware problem above), and the row set is page-bounded, so the Python
pass costs nothing.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

__all__ = [
    "parse_stored_datetime",
    "effective_date",
    "relative_time",
    "day_bucket",
]

#: Group header for rows whose effective date could not be determined at
#: all. Rendering a labelled group over a real item is honest; dropping the
#: row or inventing a date would not be.
UNKNOWN_DATE_HEADER = "Unknown date"


def parse_stored_datetime(value: Any) -> datetime | None:
    """Parse the timestamp shapes this app actually stores.

    Args:
        value: A `datetime`, a `date`, or a string. ISO-8601 with or without
            microseconds, with a `T` or a space separator, with a `Z` suffix,
            a numeric offset, or nothing at all.

    Returns:
        A timezone-aware `datetime` in UTC, or `None` when the value is empty
        or cannot be read. A date with no time component is returned as
        midnight UTC -- callers that care about the distinction should check
        the input, not the result.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    # `fromisoformat` accepts `Z` from 3.11, but normalizing it here keeps this
    # working if the minimum ever moves back down, and costs one comparison.
    if text.endswith(("Z", "z")):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def effective_date(item: dict[str, Any]) -> datetime | None:
    """The instant an item is sorted and rendered by.

    `published_date` when the feed supplied one (and it parses), otherwise
    `created_at` -- ingest time, used but never *presented* as a publish
    date (see the module docstring). `None` when the item carries neither
    or neither parses.
    """
    published = parse_stored_datetime(item.get("published_date"))
    if published is not None:
        return published
    return parse_stored_datetime(item.get("created_at"))


def _localize(dt: datetime) -> datetime:
    """`dt` in the viewer's local zone, naive attached to UTC first."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone()


def _reference(now: datetime | None) -> datetime:
    """The injected or current instant, as a local-zone datetime."""
    reference = now if now is not None else datetime.now(timezone.utc)
    return _localize(reference)


def relative_time(dt: datetime | None, *, now: datetime | None = None) -> str:
    """Render an effective date for line 1 of an article row.

    Clock time for today ("9:41 AM"), "Yesterday", a month-day for another
    day this year ("Aug 04"), ISO for older ("2025-12-31"), "-" for a
    missing date. Distinct from `humane_time.humane_timestamp`, which is the
    house style for table CELLS ("Today 18:15"): the row already sits under
    a day-group header, so repeating "Today" on every row is noise, and a
    24-hour clock is not how the reference readers render times.
    """
    if dt is None:
        return "-"
    local = _localize(dt)
    reference_local = _reference(now)
    today = reference_local.date()
    if local.date() >= today:
        # `>=` deliberately: future-dated items (bad feed clocks) render as
        # now-ish rather than a negative countdown.
        return local.strftime("%I:%M %p").lstrip("0")
    if local.date() == today - timedelta(days=1):
        return "Yesterday"
    if local.year == reference_local.year:
        return local.strftime("%b %d")
    return local.strftime("%Y-%m-%d")


def day_bucket(dt: datetime | None, *, now: datetime | None = None) -> str:
    """The group-header label for an effective date, in the viewer's zone.

    "Today" / "Yesterday" / a full date ("August 01, 2026"). Future-dated
    items (bad feed clocks) bucket into Today rather than floating above
    the list under a header of their own. `None` buckets into
    `UNKNOWN_DATE_HEADER` so a dateless item still has a group to live in.
    """
    if dt is None:
        return UNKNOWN_DATE_HEADER
    local = _localize(dt)
    today = _reference(now).date()
    if local.date() >= today:
        return "Today"
    if local.date() == today - timedelta(days=1):
        return "Yesterday"
    return local.strftime("%B %d, %Y")
