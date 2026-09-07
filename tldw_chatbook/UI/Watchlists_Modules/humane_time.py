"""One timestamp format for every Watchlists table.

TASK-2308. Sources ("Last scraped"), Items ("Created") and Runs ("Started")
each rendered `str(row.get(...))` straight from the database: 32 characters of
`2026-08-04T18:15:22.123456+00:00` per cell, in UTC, in a table whose other
columns are a title and a status. The column was the widest thing on screen
and the least readable.

**There was no house style to reuse.** The Artifacts pane looked humane
("2026-08-04 18:22:44") purely because SQLite's `CURRENT_TIMESTAMP` has that
shape -- it is still UTC, and still a raw column value. So the style is
written down here, once, and every table calls it.

**Naive means UTC.** Every writer behind these columns stores UTC:
`LocalWatchlistsService._utc_now`, SQLite `CURRENT_TIMESTAMP` defaults, and
the scrapers' `datetime.now(timezone.utc).isoformat()`. A naive value is
therefore attached to UTC rather than to the local zone -- assuming local
would silently shift every pre-existing row by the user's offset.

**Unparseable input passes through.** A backend this app does not control can
put anything in these fields. Rendering a dash over a value that exists would
be a lie, and raising inside a `compose()` exits the application, so anything
this module cannot read is returned unchanged.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

# task-3072: the parser LIVES in `Subscriptions/item_dates.py` now, moved
# there so the Subscriptions layer can parse stored dates without importing
# from `UI/` (the layer direction is UI -> Subscriptions, never the
# reverse). Re-exported here under its original name: this module's public
# API (`__all__`) and every caller are unchanged. See item_dates' module
# docstring for the full rationale.
from ...Subscriptions.item_dates import parse_stored_datetime as parse_timestamp

__all__ = ["humane_timestamp", "parse_timestamp"]

#: What an empty cell says. The dash every Watchlists table already used for
#: "no value", kept so this change cannot be mistaken for a new empty state.
EMPTY_TIMESTAMP = "-"


def _is_date_only(value: Any) -> bool:
    """Whether the stored value carries a date but no clock time.

    A date-only value must NOT be shifted into the local zone: `2026-07-18`
    means that calendar day, and converting midnight UTC westwards would
    render it as the 17th.
    """
    if isinstance(value, datetime):
        return False
    if isinstance(value, date):
        return True
    return ":" not in str(value)


def humane_timestamp(value: Any, *, now: datetime | None = None) -> str:
    """Render a stored timestamp for a Watchlists table cell.

    Args:
        value: The stored value; see `parse_timestamp` for the shapes read.
        now: The instant to measure "today"/"yesterday" against. Injected so
            the format is testable without freezing the clock; defaults to
            the current time.

    Returns:
        One of, in the viewer's local zone:

        * ``-`` for an empty value;
        * ``Today 18:15`` / ``Yesterday 18:15`` for the last two days;
        * ``Aug 04 18:15`` for another day in the same year;
        * ``2025-12-31`` for an older one;
        * ``2026-07-18`` for a value that carried no clock time at all;
        * the original text, unchanged, for anything unparseable.
    """
    if value is None:
        return EMPTY_TIMESTAMP
    if isinstance(value, str) and not value.strip():
        return EMPTY_TIMESTAMP
    parsed = parse_timestamp(value)
    if parsed is None:
        return str(value)
    if _is_date_only(value):
        return parsed.strftime("%Y-%m-%d")

    local = parsed.astimezone()
    reference = now if now is not None else datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    reference_local = reference.astimezone()

    today = reference_local.date()
    if local.date() == today:
        return f"Today {local:%H:%M}"
    if local.date() == today - timedelta(days=1):
        return f"Yesterday {local:%H:%M}"
    if local.year == reference_local.year:
        return f"{local:%b %d %H:%M}"
    return f"{local:%Y-%m-%d}"
