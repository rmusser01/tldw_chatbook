"""Scheduling module behavioral constants and config coercion.

Single source of truth for the numeric defaults shared by the scheduler
loop, the app wiring, and the generated configuration (task-18937/18939).
The config file and app wiring reference THESE constants; the only literal
lives here.
"""

from __future__ import annotations

from typing import Any

#: Default scheduler poll interval, seconds ([scheduling]
#: scheduler_poll_interval_seconds).
SCHEDULER_POLL_INTERVAL_SECONDS = 30.0

#: A dispatch more than this many seconds after its scheduled time counts as
#: "missed while away" (task-18937). 2x the poll interval: while the app
#: runs a dispatch lands within one poll, so beyond 2x the scheduler was
#: not running at the scheduled time.
MISSED_FIRE_GRACE_SECONDS = 60.0

#: Default handler execution timeout, seconds (task-18939): a handler still
#: running after this is cancelled and its dispatch records ``timed_out``.
#: <=0 disables the bound.
HANDLER_TIMEOUT_SECONDS = 300.0


def coerce_positive_float(
    value: Any, fallback: float, *, allow_zero: bool = False
) -> float:
    """Coerce a config value to a positive float, tolerating junk.

    User-editable TOML can carry strings, bools, or negatives; the scheduler
    must degrade to the documented default rather than crash or classify
    every dispatch as late (review finding: grace setting bypasses
    validation). ``bool`` is rejected explicitly -- ``True`` is an ``int``
    in Python and would silently coerce to ``1.0``.

    Args:
        value: The raw configured value.
        fallback: The default returned when ``value`` is unusable.
        allow_zero: When True, 0 is returned as-is (an explicit opt-out);
            when False (the default), 0 and negatives fall back.

    Returns:
        A usable positive float (or exactly ``0.0`` when allowed).
    """
    if isinstance(value, bool) or value is None:
        return fallback
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value.strip())
        except ValueError:
            return fallback
    else:
        return fallback
    if result < 0:
        return fallback
    if result == 0 and not allow_zero:
        return fallback
    return result
