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
#: `--api-key -Xy9_abc...` (a secret that happens to start with a dash).
#:
#: Length separates them, and the dash count sets the bound. Long-form flags
#: use `--` and can be wordy (`--no-verify-ssl`); single-dash flags are short
#: (`-v`, `-xvf`). So a single dash followed by twenty-odd characters is a
#: credential, not a flag -- which is exactly the shape that survived
#: unredacted before (`--api-key -Xy9_abcdefghijklmnopqrs`).
#: An inline value (`--out=file.txt`) is still a flag, so the optional `=` tail
#: is matched and excluded from the length.
_MAX_SHORT_FLAG_NAME = 4
_MAX_LONG_FLAG_NAME = 40
_PLAUSIBLE_FLAG_RE = re.compile(
    r"^(?P<dashes>--?)(?P<name>[A-Za-z][A-Za-z0-9_-]*)(?P<value>=.*)?$"
)


def _is_plausible_flag(text: str) -> bool:
    """Whether a dash-prefixed token names a flag rather than being a secret."""
    match = _PLAUSIBLE_FLAG_RE.match(text)
    if match is None:
        return False
    limit = (
        _MAX_LONG_FLAG_NAME
        if match.group("dashes") == "--"
        else _MAX_SHORT_FLAG_NAME
    )
    return len(match.group("name")) <= limit

#: Values that are secrets whatever key they arrive under. Key-name matching
#: alone let `{"note": "sk-live-..."}` through to the approval card.
#:
#: Deliberately anchored and length-bounded: a 40-char git SHA, a file path and
#: ordinary prose must NOT match. Over-redaction makes an approval card useless
#: for deciding whether to approve, so these patterns describe specific
#: credential formats rather than "looks random".
_SECRET_VALUE_PATTERNS = (
    # PEM/PGP private key block. The header ends `-----` for PEM but
    # `KEY BLOCK-----` for PGP, so the anchor stops at PRIVATE KEY.
    re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY"),
    # JWT: three base64url segments, and the header segment starts "eyJ".
    re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}"),
    # OpenAI / Anthropic hyphenated keys. The long unhyphenated tail is what
    # separates a real key from a kebab-case path component: `sk-core-runtime-
    # helpers` is a directory, `sk-ant-api03-<32 chars>` is a credential.
    re.compile(r"\bsk-(?:[A-Za-z0-9]+-)*[A-Za-z0-9_]{20,}"),
    # Stripe-style underscore keys, which the hyphenated pattern never matched.
    re.compile(r"\b[sprk]k_(?:live|test)_[A-Za-z0-9]{16,}"),
    # GitHub classic, fine-grained, and GitLab.
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{16,}"),
    # Slack bot/user tokens and app-level tokens.
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\bxapp-[0-9]-[A-Za-z0-9-]{10,}"),
    # AWS access key id (fixed 20-char form).
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    # Google API key (fixed 39-char form).
    re.compile(r"\bAIza[0-9A-Za-z_-]{35,}"),
    # An Authorization header value, however it is labelled. The digit
    # lookahead and length floor are what stop this matching the English
    # sentence "supports basic authentication" -- credentials contain digits
    # and are long; prose after the word "basic" is neither.
    re.compile(
        r"(?i)\b(?:bearer|basic)\s+(?=[A-Za-z0-9._~+/=-]*[0-9])"
        r"[A-Za-z0-9._~+/=-]{16,}"
    ),
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
    if isinstance(value, (bytes, bytearray)):
        # Decoded only to test the shape; the original is never returned once
        # it matches, and undecodable bytes are left alone.
        try:
            probe = value.decode("utf-8", "strict")
        except (UnicodeDecodeError, AttributeError):
            return value
        return REDACTED if looks_like_secret_value(probe) else value
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

    A value following a secret flag is redacted even when it starts with "-"
    (`--api-key -9f3a...`): the token is only treated as a new flag when it
    looks like a flag NAME -- short, and not a credential-length token. That
    keeps `--api-key --verbose` from swallowing `--verbose` while closing the
    leak where a secret merely began with a hyphen (TASK-26011 review, I5).

    Residual: a credential shorter than the flag-name bound and shaped exactly
    like a flag name is still read as a flag. Bare tokens are shape-checked
    separately, which covers the realistic forms.

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
            if not _is_plausible_flag(text):
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


def _redact_netloc(netloc: str) -> str:
    """Strip `user:password@` credentials from a URL authority."""
    if "@" not in netloc:
        return netloc
    _userinfo, _, host = netloc.rpartition("@")
    return f"{REDACTED}@{host}"


def redact_url(url: str) -> str:
    """Redact credentials in a URL: query values, and `user:pass@` userinfo.

    Query values are judged by name AND by shape, so a secret under an
    innocuous parameter name does not survive (TASK-26011 review, I6).
    """
    parts = urlsplit(str(url))
    if not parts.query and "@" not in parts.netloc:
        return str(url)
    query = [
        (
            key,
            REDACTED
            if is_secret_key(key) or looks_like_secret_value(value)
            else value,
        )
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
    ]
    return urlunsplit(
        (
            parts.scheme,
            _redact_netloc(parts.netloc),
            parts.path,
            urlencode(query),
            parts.fragment,
        )
    )
