"""One tz-normalizing datetime codec for the MCP/runtime-policy stores.

TASK-32862: six copies of ``_datetime_to_iso`` (plus ``_iso_to_datetime``
and now-iso variants) existed across MCP and runtime_policy, one of them
(``UX_Interop/server_parity_contracts.py``) silently emitting NAIVE-LOCAL
ISO strings for naive datetimes while every sibling normalizes to UTC.
This module is the single owner; the copies delegate to it.

Contract (the majority behavior the five MCP/runtime_policy copies already
shared, now with the UX_Interop gap closed):
- ``None`` in, ``None`` out.
- Naive datetimes are ASSUMED UTC (not local time) and normalized.
- Output is UTC, ``isoformat()``, with the ``+00:00`` suffix spelled ``Z``.
"""

from __future__ import annotations

from datetime import datetime, timezone


def datetime_to_iso(value: datetime | None) -> str | None:
    """Normalize a datetime to a UTC ISO-8601 string ending in ``Z``."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def iso_to_datetime(value: object) -> datetime | None:
    """Parse an ISO string (or pass a datetime through) to aware; None on bad input.

    Naive results (possible only for inputs without an offset) are assumed
    UTC, mirroring the encode side. Unparseable strings return None (the
    store copies' contract), not raise.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None
