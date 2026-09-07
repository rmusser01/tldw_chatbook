"""Forgiving reminder-schedule text parsing: datetime + IANA timezone.

Pure module (no I/O, no imports of other Scheduling submodules, no
Textual) -- same discipline as ``schedule_compute.py``. Hoisted out of
``UI/Screens/scheduling/forms/reminder_form.py`` (redesign PR-3, task 3's
folded-in refactor, closing task-2-review.md finding 1: `SchedulingService.
edit_reminder_fields` was reaching UP into a UI-layer module for two
functions that were always pure -- a leading-underscore, module-private
name at that). ``detect_system_timezone``/``system_timezone_name`` joined
them the same way (task-31711 fix round): `SchedulingService.
update_reminder`'s local-path branch needed the reminder form's own
detected-or-UTC zone to stop re-nulling a one-time reminder's timezone on
every edit, and that pair was still living in the UI-layer form module.
``reminder_form.py`` imports all four back under their original call
shape; ``is_valid_zone`` is the public spelling of what was
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

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

#: Fallback IANA zone when the machine's own zone can't be detected.
_DEFAULT_TIMEZONE = "UTC"

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
    (``assumed_local`` False); a naive form such as ``YYYY-MM-DD HH:MM``
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


def example_run_at_text(*, days_ahead: int = 7) -> str:
    """A never-in-the-past example run-at string, for placeholder/hint/
    error copy across the reminder and automation forms and detail panes.

    task-31711 AC#4: a hard-coded example date (12 sites) drifts into the
    past as real time passes -- "type a date like <a date that already
    happened>" reads as broken advice, not help.
    Computed fresh relative to "now" each call instead, in the same
    forgiving ``YYYY-MM-DD HH:MM`` shape :func:`parse_forgiving_datetime`
    accepts, so every caller's placeholder/hint/error text stays valid.

    Args:
        days_ahead: How many days past today the example date lands.
            Callers share the default so the shown example is identical
            everywhere; a few days out is comfortably future regardless
            of what hour "now" happens to be.

    Returns:
        A ``"YYYY-MM-DD 09:00"`` string, ``days_ahead`` days from today.
    """
    example_date = (datetime.now() + timedelta(days=days_ahead)).date()
    return f"{example_date.isoformat()} 09:00"


def detect_system_timezone() -> str | None:
    """Best-effort IANA name for the machine's local timezone, or None.

    Checks ``TZ`` first, then the ``/etc/localtime`` symlink (macOS and
    Linux both point it into a ``zoneinfo`` tree). Returns None where
    neither yields a valid zone (copied-file distros, containers,
    Windows) so callers can label the UTC fallback honestly instead of
    claiming it is the machine's zone (review F7).

    Hoisted from ``reminder_form.py`` (task-31711 fix round): the service
    layer's ``update_reminder`` needed the exact same detected-or-UTC
    zone the reminder form's own Select default uses, and reaching UP
    into a UI-layer module for it would repeat the smell this module's
    own docstring already describes fixing once. ``reminder_form.py``
    imports both this and ``system_timezone_name`` back under their
    original names.

    Returns:
        The detected IANA zone name, or None when detection fails.
    """
    tz_env = os.environ.get("TZ", "").strip()
    if tz_env and is_valid_zone(tz_env):
        return tz_env
    try:
        localtime = os.path.realpath("/etc/localtime")
    except OSError:
        localtime = ""
    if "/zoneinfo/" in localtime:
        name = localtime.split("/zoneinfo/", 1)[1]
        if is_valid_zone(name):
            return name
    return None


def system_timezone_name() -> str:
    """The detected machine zone, or UTC when detection fails.

    Returns:
        The IANA zone name from :func:`detect_system_timezone`, falling
        back to ``"UTC"``.
    """
    return detect_system_timezone() or _DEFAULT_TIMEZONE
