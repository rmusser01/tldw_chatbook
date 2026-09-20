"""One shared UTC timestamp contract for the package (TASK-32803.1).

The core-runtime review found twelve distinct UTC string shapes produced by ~55
helper copies, a thirteenth from SQLite's ``CURRENT_TIMESTAMP``, and ~100 sites
comparing them lexically -- three of which were wrong for users because a
space-separated cutoff sorts before a ``T``-separated stored value (``' '`` 0x20
< ``'T'`` 0x54) on the same calendar date.

This module is the one home for producing and reading UTC timestamps.

Canonical stored shape (see ADR-127): millisecond-precision ISO-8601 in UTC with
a ``Z`` suffix, e.g. ``2026-09-20T12:34:56.789Z``. It is fixed-width (24 chars),
so lexical (``TEXT``) ordering in SQLite equals chronological ordering, and it
round-trips through :func:`datetime.datetime.fromisoformat`.

Reading tolerates every shape already on disk (``parse_utc``); writing always
produces the canonical shape (``utc_now_iso`` / ``to_utc_iso``). A naive
timestamp (no offset) is interpreted as UTC -- that is the assumption every
existing lexical comparison already made; writers that produced naive *local*
time via ``datetime.now()`` are latent bugs the format guard (AC#2) flags.
"""

from __future__ import annotations

from datetime import datetime, timezone

#: The canonical stored UTC shape, e.g. ``2026-09-20T12:34:56.789Z``.
#: Millisecond precision, ``Z`` suffix, fixed width so TEXT sort == time sort.
CANONICAL_UTC_EXAMPLE = "2026-09-20T12:34:56.789Z"

#: The shape SQLite's ``CURRENT_TIMESTAMP`` default writes (space separator, no
#: offset, second precision), e.g. ``2026-09-20 12:34:56``. Documented here
#: because 31 tables default to it; ``parse_utc`` reads it, but new writes
#: should use the canonical shape so ordering stays lexical-safe.
SQLITE_DEFAULT_TIMESTAMP_EXAMPLE = "2026-09-20 12:34:56"


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC ``datetime``."""
    return datetime.now(timezone.utc)


def to_utc_iso(moment: datetime) -> str:
    """Format a ``datetime`` as the canonical millisecond-``Z`` UTC string.

    A naive ``datetime`` is interpreted as UTC (the package's storage
    assumption); an aware one is converted to UTC first.

    Args:
        moment: The instant to format.

    Returns:
        The canonical ``YYYY-MM-DDTHH:MM:SS.mmmZ`` string.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    else:
        moment = moment.astimezone(timezone.utc)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def utc_now_iso() -> str:
    """Return the current UTC time in the canonical millisecond-``Z`` shape."""
    return to_utc_iso(utc_now())


def parse_utc(value: str) -> datetime:
    """Parse any UTC timestamp shape on disk into an aware UTC ``datetime``.

    Accepts the canonical shape, ``+00:00``-offset ISO, ``T``- or
    space-separated forms, second/millisecond/microsecond precision, date-only,
    and SQLite's ``CURRENT_TIMESTAMP`` output. A parsed value with no offset is
    interpreted as UTC.

    Args:
        value: A timestamp string produced by any writer in the package.

    Returns:
        A timezone-aware ``datetime`` in UTC.

    Raises:
        ValueError: If ``value`` is not a recognizable timestamp.
    """
    text = value.strip()
    if not text:
        raise ValueError("empty timestamp")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
