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


def redact_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy with values under secret-looking keys replaced."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            result[key] = redact_mapping(value)
        elif is_secret_key(key) and value not in (None, ""):
            result[key] = REDACTED
        else:
            result[key] = value
    return result


def redact_args(args: Sequence[str]) -> list[str]:
    """Redact CLI arg values: `--api-key VALUE` pairs and `key=value` forms."""
    redacted: list[str] = []
    previous_was_secret_flag = False
    for arg in args:
        text = str(arg)
        if previous_was_secret_flag:
            redacted.append(REDACTED)
            previous_was_secret_flag = False
            continue
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
