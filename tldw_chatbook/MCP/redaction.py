"""Secret redaction applied at every MCP display and log boundary."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

REDACTED = "***"

_SECRET_KEY_RE = re.compile(
    r"(?i)(token|secret|passwd|password|api[-_]?key|authorization|bearer|credential)"
)
_INLINE_ARG_RE = re.compile(r"^(?P<key>[A-Za-z0-9_-]+)=(?P<value>.+)$")


def is_secret_key(key: str) -> bool:
    """Whether a mapping key / arg name looks like it holds a secret."""
    return bool(_SECRET_KEY_RE.search(str(key)))


def _is_present(value: Any) -> bool:
    """Whether a value is non-empty/non-None and thus worth redacting."""
    return value not in (None, "", {}, [], ())


def _redact_sequence(seq: Sequence[Any]) -> Sequence[Any]:
    """Deep-copy a list/tuple, redacting any nested Mappings it contains."""
    items = [
        redact_mapping(item)
        if isinstance(item, Mapping)
        else _redact_sequence(item)
        if isinstance(item, (list, tuple))
        else item
        for item in seq
    ]
    return tuple(items) if isinstance(seq, tuple) else items


def redact_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy with values under secret-looking keys replaced."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if is_secret_key(key) and _is_present(value):
            # A secret-keyed value must never survive, regardless of its
            # type (str, dict, list, ...) - this check must win over the
            # recursion branches below.
            result[key] = REDACTED
        elif isinstance(value, Mapping):
            result[key] = redact_mapping(value)
        elif isinstance(value, (list, tuple)):
            # Rebuild the sequence (copy) and redact nested Mappings inside
            # it. Strings/bytes are also Sequences but are handled by the
            # isinstance(value, (list, tuple)) check excluding them.
            result[key] = _redact_sequence(value)
        else:
            result[key] = value
    return result


def redact_args(args: Sequence[str]) -> list[str]:
    """Redact CLI arg values: `--api-key VALUE` pairs and `key=value` forms.

    Known residual leak: if the VALUE following a secret flag itself starts
    with "-" (e.g. `--api-key -9f...`, or any secret token that happens to
    start with a hyphen), it is treated as a new flag rather than the
    previous flag's value, and is re-evaluated and appended unredacted
    instead of being replaced with REDACTED. Only genuine `--flag value`
    pairs where the value does not start with "-" are covered by the
    flag-then-value branch below.

    Args:
        args: Raw CLI argument tokens, e.g. as launched for a local MCP
            server profile (`command` + `args`).

    Returns:
        A new list of the same length with secret-looking values replaced
        by `REDACTED`; non-secret tokens are returned unchanged.
    """
    redacted: list[str] = []
    previous_was_secret_flag = False
    for arg in args:
        text = str(arg)
        if previous_was_secret_flag:
            previous_was_secret_flag = False
            if not text.startswith("-"):
                redacted.append(REDACTED)
                continue
            # The previous flag was never given a value (this arg is itself
            # a new flag) - fall through and re-evaluate `text` normally so
            # it isn't silently swallowed as a secret value.
        inline = _INLINE_ARG_RE.match(text)
        if inline and is_secret_key(inline.group("key")):
            redacted.append(f"{inline.group('key')}={REDACTED}")
            continue
        redacted.append(text)
        if text.startswith("-") and is_secret_key(text.lstrip("-")):
            previous_was_secret_flag = True
    return redacted


def redact_url(url: str) -> str:
    """Redact secret-named query parameter values in a URL."""
    parts = urlsplit(str(url))
    if not parts.query:
        return str(url)
    query = [
        (key, REDACTED if is_secret_key(key) else value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
    ]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))
