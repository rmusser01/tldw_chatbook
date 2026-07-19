"""Fail-closed regex validation + a never-raising matcher for Lore entries.

World-info matching runs on the UI event loop and Python ``re`` cannot be
portably time-bounded, so a catastrophic pattern could freeze the app. Patterns
are validated fail-closed at save and import; the send-path matcher additionally
never raises. The catastrophic-pattern heuristic is best-effort: it flags nested
unbounded-quantifier shapes and trivial identical alternation, but not general
alternation-overlap ReDoS (documented residual risk).
"""
from __future__ import annotations

import re

MAX_REGEX_PATTERN_LENGTH = 500

# An UNBOUNDED quantifier only: +, *, or {n,}. Never ? or bounded {n}/{n,m}.
_UNBOUNDED = r"(?:[*+]|\{\d+,\})"
# (a) a (flat) group whose body contains an unbounded quantifier, itself
#     immediately followed by an unbounded quantifier: (…+…)+ , (…*…)* , …
_NESTED_QUANT_RE = re.compile(r"\([^()]*" + _UNBOUNDED + r"[^()]*\)" + _UNBOUNDED)
# (b) a trivial identical two-way alternation (x|x) followed by an unbounded
#     quantifier: (a|a)* .
_IDENTICAL_ALT_RE = re.compile(r"\(([^()|]+)\|\1\)" + _UNBOUNDED)


def _looks_catastrophic(pattern: str) -> bool:
    return bool(_NESTED_QUANT_RE.search(pattern) or _IDENTICAL_ALT_RE.search(pattern))


def validate_regex_pattern(pattern: str) -> None:
    """Raise ValueError (user-facing) if a pattern is unusable or dangerous.

    Args:
        pattern: The regex pattern string to validate.

    Raises:
        ValueError: If the pattern is too long, has invalid syntax, or matches
            the catastrophic-pattern heuristic.
    """
    if len(pattern) > MAX_REGEX_PATTERN_LENGTH:
        raise ValueError(
            f"Regex pattern is too long (max {MAX_REGEX_PATTERN_LENGTH} characters)."
        )
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Invalid regex: {exc}") from exc
    if _looks_catastrophic(pattern):
        raise ValueError(
            "Regex pattern is too complex (nested quantifiers can hang matching)."
        )


def regex_search(pattern: str, text: str, ignore_case: bool) -> bool:
    """Search ``text`` for ``pattern``; never raises (bad pattern → False).

    Args:
        pattern: The regex pattern.
        text: The text to search.
        ignore_case: Whether to match case-insensitively.

    Returns:
        True if the pattern matches anywhere in the text, else False (also False
        on any error — a bad pattern simply does not fire).
    """
    try:
        return bool(re.search(pattern, text, re.IGNORECASE if ignore_case else 0))
    except Exception:
        return False
