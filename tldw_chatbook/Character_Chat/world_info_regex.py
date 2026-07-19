"""Fail-closed regex validation + a never-raising matcher for Lore entries.

World-info matching runs on the UI event loop and Python ``re`` cannot be
portably time-bounded, so a catastrophic pattern could freeze the app. Patterns
are validated fail-closed at save and import; the send-path matcher additionally
never raises. The catastrophic-pattern heuristic is best-effort: it flags
nested unbounded-quantifier shapes (at any grouping depth) and textually
duplicated alternation branches in a flat quantified group. Residual (accepted)
gaps it does NOT catch: general alternation-overlap ReDoS (``(a|ab)*``) and
identical-language branches that differ only by redundant grouping/escaping
(``(a|(a))*``). These are rare in hand-authored lore; the send-path never-raise
guard does not help against a hang, so the residual risk is documented, not
eliminated.
"""
from __future__ import annotations

import re

MAX_REGEX_PATTERN_LENGTH = 500

# An UNBOUNDED quantifier only: +, *, or {n,}. Never ? or bounded {n}/{n,m}.
_UNBOUNDED = r"(?:[*+]|\{\d+,\})"
# A flat group (no inner parens) that is immediately followed by an unbounded
# quantifier — its captured body is checked for a duplicated alternation branch.
_FLAT_QUANT_GROUP_RE = re.compile(r"\(([^()]*)\)(?=" + _UNBOUNDED + r")")
# Strip a leading group-modifier so the body is just the alternation content.
_GROUP_PREFIX_RE = re.compile(r"^\?(?::|P<\w+>|P=\w+|<[=!]|[=!#]|[aiLmsux]*:?)")


def _next_is_unbounded(pattern: str, i: int) -> bool:
    """Is there an unbounded quantifier (``*``/``+``/``{n,}``) at ``pattern[i:]``?"""
    if i < len(pattern) and pattern[i] in "*+":
        return True
    if i < len(pattern) and pattern[i] == "{":
        return bool(re.match(r"\{\d+,\}", pattern[i:]))
    return False


def _has_nested_unbounded_quantifier(pattern: str) -> bool:
    """True if some unbounded-quantified group's body (at any nesting depth)
    itself contains an unbounded quantifier — the ``(…+…)+`` ReDoS family,
    including through extra grouping like ``((a+))+`` / ``(?:(a+))+``.

    Walks the pattern with a group stack, honoring escapes and character
    classes, so nesting cannot evade detection (unlike a flat regex).
    """
    stack: list[bool] = []  # per open group: does its body hold an unbounded quant?
    i, n = 0, len(pattern)
    in_class = False
    while i < n:
        c = pattern[i]
        if c == "\\":
            i += 2
            continue
        if in_class:
            if c == "]":
                in_class = False
            i += 1
            continue
        if c == "[":
            in_class = True
            i += 1
            continue
        if c == "(":
            stack.append(False)
            i += 1
            continue
        if c == ")":
            body_has = stack.pop() if stack else False
            outer_unbounded = _next_is_unbounded(pattern, i + 1)
            if body_has and outer_unbounded:
                return True
            if stack and (body_has or outer_unbounded):
                stack[-1] = True  # bubble up to the enclosing group
            i += 1
            continue
        if c in "*+":
            if stack:
                stack[-1] = True
            i += 1
            continue
        if c == "{":
            m = re.match(r"\{\d+,\}", pattern[i:])
            if m:
                if stack:
                    stack[-1] = True
                i += m.end()
                continue
            i += 1
            continue
        i += 1
    return False


def _split_top_level_alternation(body: str) -> list[str]:
    """Split a (flat, paren-free) group body on top-level ``|`` only — NOT on
    escaped pipes (``\\|``) or pipes inside a character class (``[a|b]``). A naive
    ``body.split("|")`` would false-positive on those (e.g. reject ``(\\|\\|\\|)+``)."""
    parts: list[str] = []
    cur: list[str] = []
    i, n = 0, len(body)
    in_class = False
    while i < n:
        c = body[i]
        if c == "\\":
            cur.append(body[i:i + 2])
            i += 2
            continue
        if c == "[":
            in_class = True
        elif c == "]":
            in_class = False
        if c == "|" and not in_class:
            parts.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    parts.append("".join(cur))
    return parts


def _has_duplicate_alternation(pattern: str) -> bool:
    """True if a flat unbounded-quantified group has a repeated alternation
    branch — the ``(a|a)*`` / ``(a|a|a)+`` family (any number of branches).
    Splits only on real top-level alternation (see _split_top_level_alternation),
    so escaped/char-class pipes don't cause false positives."""
    for m in _FLAT_QUANT_GROUP_RE.finditer(pattern):
        body = _GROUP_PREFIX_RE.sub("", m.group(1))
        alts = _split_top_level_alternation(body)
        if len(alts) >= 2 and len(set(alts)) < len(alts):
            return True
    return False


def _looks_catastrophic(pattern: str) -> bool:
    return _has_nested_unbounded_quantifier(pattern) or _has_duplicate_alternation(pattern)


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
