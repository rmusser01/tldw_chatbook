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

#: A token that plausibly names a CLI flag rather than being a flag-shaped
#: secret. Used to tell `--api-key --verbose` (flag never given a value) from
#: `--api-key -9f3a...` (a secret that happens to start with a dash).
_PLAUSIBLE_FLAG_RE = re.compile(r"^--?[A-Za-z][A-Za-z0-9_-]*$")

#: Values that are secrets whatever key they arrive under. Key-name matching
#: alone let `{"note": "sk-live-..."}` through to the approval card.
#:
#: Deliberately anchored and length-bounded: a 40-char git SHA, a file path and
#: ordinary prose must NOT match. Over-redaction makes an approval card useless
#: for deciding whether to approve, so these patterns describe specific
#: credential formats rather than "looks random".
_SECRET_VALUE_PATTERNS = (
    # PEM private key block (any algorithm).
    re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----"),
    # JWT: three base64url segments, and the header segment starts "eyJ".
    re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}"),
    # OpenAI / Anthropic / Stripe-style prefixed keys.
    re.compile(r"\b[sprk]k-(?:[A-Za-z0-9]+-)?[A-Za-z0-9_-]{16,}"),
    # GitHub classic and fine-grained tokens.
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    # Slack.
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
    # AWS access key id (fixed 20-char form).
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    # Google API key (fixed 39-char form).
    re.compile(r"\bAIza[0-9A-Za-z_-]{35,}"),
    # An Authorization header value, however it is labelled.
    re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"),
    # Credentials embedded in a connection URI: scheme://user:password@host
    re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s/@]+@"),
)


def is_secret_key(key: str) -> bool:
    """Whether a mapping key / arg name looks like it holds a secret."""
    return bool(_SECRET_KEY_RE.search(str(key)))


def looks_like_secret_value(value: Any) -> bool:
    """Whether a value is shaped like a credential, whatever key it is under.

    Complements `is_secret_key`: that catches `{"api_key": "anything"}`, this
    catches `{"note": "sk-live-..."}`. Non-strings are never matched -- the
    caller redacts by key for those.
    """
    if not isinstance(value, str) or not value:
        return False
    return any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS)


def _redact_value(value: Any) -> Any:
    """Redact a scalar when its shape betrays it; otherwise return it as-is."""
    return REDACTED if looks_like_secret_value(value) else value


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
        else _redact_value(item)
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
            result[key] = _redact_value(value)
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
            if not _PLAUSIBLE_FLAG_RE.match(text):
                # Covers the ordinary `--api-key VALUE` case AND a secret that
                # happens to start with "-" (`--api-key -9f3a...`), which the
                # old startswith("-") test let through unredacted.
                redacted.append(REDACTED)
                continue
            # The previous flag was never given a value (this arg really does
            # name another flag) - fall through and re-evaluate `text` so it
            # isn't silently swallowed as a secret value.
        inline = _INLINE_ARG_RE.match(text)
        if inline and is_secret_key(inline.group("key")):
            redacted.append(f"{inline.group('key')}={REDACTED}")
            continue
        # A secret can also arrive as a bare positional token with no flag.
        redacted.append(_redact_value(text))
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
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )
