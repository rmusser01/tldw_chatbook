"""Forgiving reminder-schedule text parsing: datetime + IANA timezone.

Pure module (no I/O, no imports of other Scheduling submodules, no
Textual) -- same discipline as ``schedule_compute.py``. Hoisted out of
``UI/Screens/scheduling/forms/reminder_form.py`` (redesign PR-3, task 3's
folded-in refactor, closing task-2-review.md finding 1: `SchedulingService.
edit_reminder_fields` was reaching UP into a UI-layer module for two
functions that were always pure -- a leading-underscore, module-private
name at that). ``reminder_form.py`` imports both back under their
original call shape; ``is_valid_zone`` is the public spelling of what was
``reminder_form.py``'s ``_is_valid_zone`` -- publicized because a
service-layer caller reaching across modules for a private name was part
of the smell.

Why these parsers live here and not in ``Utils/input_validation.py``: that
module is the *security* boundary ("Input validation utilities for
secure user input handling") -- boolean gatekeepers for traversal,
injection, SSRF and size-class risks. The helpers below are domain
format parsers: they normalize text into schedule values (an aware
datetime, a validated IANA zone name) and hand back presentation signals
callers render as live hints, such as ``parse_forgiving_datetime``'s
``assumed_local`` flag. Nothing here guards a trust boundary -- the parsed
values reach SQLite only through parameterized queries -- so hoisting them
into the security module would import zoneinfo into a module the
security-critical paths depend on, for no safety gain.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

#: Forgiving local datetime formats (naive -> system local zone).
_FORGIVING_DATETIME_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
)


def is_valid_zone(name: str) -> bool:
    """Return True when ``name`` resolves to an IANA timezone.

    Args:
        name: Candidate IANA zone name, e.g. ``"Europe/Berlin"``.

    Returns:
        True when the local tzdata can resolve ``name``, False otherwise.
    """
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, TypeError):
        return False
    return True


def parse_forgiving_datetime(raw: str) -> tuple[datetime | None, bool]:
    """Parse a run-at datetime, accepting forgiving local forms.

    Returns ``(parsed, assumed_local)``: full ISO-8601 keeps its offset
    (``assumed_local`` False); a naive form such as ``2026-08-28 09:00``
    is interpreted in the system's local timezone (``assumed_local``
    True). ``(None, False)`` when nothing parses.

    Args:
        raw: The user-entered run-at text; surrounding whitespace is
            ignored and an empty string is treated as "not a date".

    Returns:
        A ``(datetime | None, bool)`` pair. The datetime is always
        timezone-aware when parsing succeeds. The bool is True only when
        a naive input was assumed to be local time, so the caller can say
        so in the UI.
    """
    text = raw.strip()
    if not text:
        return None, False
    parsed: datetime | None = None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        for fmt in _FORGIVING_DATETIME_FORMATS:
            try:
                parsed = datetime.strptime(text, fmt)
            except ValueError:
                continue
            break
    if parsed is None:
        return None, False
    if parsed.tzinfo is None:
        return parsed.astimezone(), True
    return parsed, False
