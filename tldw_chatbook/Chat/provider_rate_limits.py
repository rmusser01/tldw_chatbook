"""Remaining provider rate limit, read from the last response's headers (TASK-28229).

``Utils.egress`` keeps the raw rate-limit headers of the last response per
provider (only while the Console gateway marks a provider call). This module
turns them into the one line the Console context/cost tooltip shows. It is
imported only once such an entry exists, so it adds nothing to the modules
loaded before the UI is ready.

Two header families are read:

* ``x-ratelimit-{limit,remaining,reset}[-{requests,tokens}][-{window}]``
  (OpenAI, Groq, Together, Cerebras, Mistral and most compatible APIs;
  a bare ``x-ratelimit-remaining`` counts requests);
* ``anthropic-ratelimit-{requests,tokens,input-tokens,output-tokens}-{limit,
  remaining,reset}``.

Values are parsed as numbers or times, never echoed, so header text cannot
reach the tooltip. A metric without a ``remaining`` value is not shown.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

_X_RATELIMIT = re.compile(
    r"x-ratelimit-(limit|remaining|reset)(?:-(requests|tokens))?"
    r"(?:-(second|minute|hour|day|week|month))?"
)
_ANTHROPIC_RATELIMIT = re.compile(
    r"anthropic-ratelimit-(requests|tokens|input-tokens|output-tokens)-"
    r"(limit|remaining|reset)"
)
_DURATION = re.compile(
    r"(?:(?P<h>\d+(?:\.\d+)?)h)?(?:(?P<m>\d+(?:\.\d+)?)m)?"
    r"(?:(?P<s>\d+(?:\.\d+)?)s)?(?:(?P<ms>\d+(?:\.\d+)?)ms)?"
)
_METRIC_ORDER = ("requests", "tokens", "input-tokens", "output-tokens")
_MAX_COUNT = 10**12
#: A reset further out than this is not believed.
_MAX_RESET_SECONDS = 40 * 24 * 3600


@dataclass(frozen=True)
class RateLimitWindow:
    """One metric's budget, as the provider reported it.

    Attributes:
        metric: ``requests``, ``tokens``, ``input-tokens`` or ``output-tokens``.
        window: The period a windowed header names (``day``), else ``None``.
        remaining: What is left, when reported.
        limit: The ceiling, when reported.
        reset_at: Wall-clock time the budget resets, when reported.
    """

    metric: str
    window: str | None
    remaining: int | None = None
    limit: int | None = None
    reset_at: float | None = None


def _count(value: str) -> int | None:
    try:
        number = int(value.strip())
    except ValueError:
        return None
    return number if 0 <= number <= _MAX_COUNT else None


def _reset_at(value: str, captured_at: float) -> float | None:
    """Read a reset as a duration, seconds, epoch seconds/ms or RFC 3339 time."""
    text = value.strip()
    seconds: float | None = None
    duration = _DURATION.fullmatch(text)
    if text and duration and any(duration.groupdict().values()):
        parts = {k: float(v) for k, v in duration.groupdict().items() if v}
        seconds = (
            parts.get("h", 0) * 3600
            + parts.get("m", 0) * 60
            + parts.get("s", 0)
            + parts.get("ms", 0) / 1000
        )
    else:
        try:
            number = float(text)
        except ValueError:
            number = None
        if number is not None:
            if number >= 1e12:  # epoch milliseconds
                seconds = number / 1000 - captured_at
            elif number >= 1e9:  # epoch seconds
                seconds = number - captured_at
            else:  # seconds from now
                seconds = number
        else:
            try:
                seconds = datetime.fromisoformat(text).timestamp() - captured_at
            except (ValueError, OverflowError):
                return None
    if seconds is None or not 0 <= seconds <= _MAX_RESET_SECONDS:
        return None
    return captured_at + seconds


def parse_rate_limit_headers(
    headers: Mapping[str, str], captured_at: float
) -> tuple[RateLimitWindow, ...]:
    """Group rate-limit headers into one window per metric and period.

    Args:
        headers: Header names (any case) to values.
        captured_at: Wall-clock time the response arrived.

    Returns:
        The windows that report a remaining value, requests first.
    """
    fields: dict[tuple[str, str | None], dict[str, object]] = {}
    for raw_name, value in headers.items():
        name = str(raw_name).lower()
        match = _X_RATELIMIT.fullmatch(name)
        if match:
            kind, metric, window = match.group(1), match.group(2) or "requests", match.group(3)
        else:
            match = _ANTHROPIC_RATELIMIT.fullmatch(name)
            if not match:
                continue
            metric, kind, window = match.group(1), match.group(2), None
        parsed = (
            _reset_at(str(value), captured_at) if kind == "reset" else _count(str(value))
        )
        if parsed is not None:
            fields.setdefault((metric, window), {})[kind] = parsed
    windows = [
        RateLimitWindow(
            metric=metric,
            window=window,
            remaining=values.get("remaining"),
            limit=values.get("limit"),
            reset_at=values.get("reset"),
        )
        for (metric, window), values in fields.items()
        if "remaining" in values
    ]
    return tuple(
        sorted(windows, key=lambda w: (_METRIC_ORDER.index(w.metric), w.window or ""))
    )


def _clock(at: float, captured_at: float) -> str:
    """Local clock time; a reset more than 20 hours out also names the day."""
    stamp = time.localtime(at)
    if at - captured_at > 20 * 3600:
        return time.strftime("%a %H:%M", stamp)
    return time.strftime("%H:%M:%S", stamp)


def format_rate_limit_line(headers: Mapping[str, str], captured_at: float) -> str | None:
    """Build the tooltip line, or ``None`` when no remaining budget is known.

    Absolute clock times, not "in 42s": the tooltip is rebuilt when Console
    refreshes, not when it is shown, so a relative time would go stale.

    Args:
        headers: The recorded rate-limit headers.
        captured_at: Wall-clock time the response arrived.

    Returns:
        E.g. ``"Rate limit at 14:32:05: 4,999/5,000 requests left (resets
        14:32:53) · 39,200/40,000 tokens left (resets 14:32:06)"``.
    """
    parts = []
    for window in parse_rate_limit_headers(headers, captured_at):
        unit = window.metric.replace("-", " ")
        if window.window:
            unit = f"{unit} per {window.window}"
        amount = (
            f"{window.remaining:,}/{window.limit:,}"
            if window.limit is not None
            else f"{window.remaining:,}"
        )
        reset = (
            f" (resets {_clock(window.reset_at, captured_at)})"
            if window.reset_at is not None
            else ""
        )
        parts.append(f"{amount} {unit} left{reset}")
    if not parts:
        return None
    return f"Rate limit at {_clock(captured_at, captured_at)}: " + " · ".join(parts)


__all__ = ["RateLimitWindow", "format_rate_limit_line", "parse_rate_limit_headers"]
