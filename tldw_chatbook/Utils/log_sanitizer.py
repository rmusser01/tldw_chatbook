"""
Log sanitizer utilities to prevent sensitive data from being logged.

This module provides functions to scrub API keys, passwords, and other
sensitive information from log messages.
"""

import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from tldw_chatbook.Utils.sensitive_config_keys import is_sensitive_config_key


REDACTION_MARKER = "***REDACTED***"
#: Hex characters kept from a content fingerprint. Twelve gives ~2e-8 collision
#: probability across a thousand distinct values in one debugging session --
#: far beyond what a maintainer reading a log needs -- while staying short
#: enough to scan by eye when correlating two lines.
CONTENT_FINGERPRINT_CHARS = 12
#: What ``content_fingerprint`` returns for an empty or missing value. A
#: fingerprint of "" is a valid digest, and printing it would make "the user
#: searched for nothing" indistinguishable from a real query at a glance;
#: an explicit token keeps that difference visible.
EMPTY_FINGERPRINT = "empty"
#: Labels that name a secret in a LOG LINE but are not config keys, so they
#: are deliberately not added to ``sensitive_config_keys`` (that predicate
#: also drives config encryption and the Privacy & Security protected-field
#: count, and widening it would start encrypting fields for a
#: logging-only reason).
_LOG_ONLY_SENSITIVE_FIELDS = frozenset(
    {
        "authorization",
        "proxy_authorization",
        "cookie",
        "set_cookie",
        "credential",
        "credentials",
        "database_url",
        "connection_string",
        "dsn",
        # TASK-19558. A bare ``key`` matches none of
        # ``is_sensitive_config_key``'s rules (its ``_key`` rule is an
        # underscore-SUFFIX match and ``api-key`` is a containment one), and
        # bare ``key`` is exactly how Google's Custom Search credential
        # travels: ``.../customsearch/v1?key=<API key>&cx=...``. Probed
        # before the fix: that whole URL passed through ``sanitize_string``
        # unchanged. The cost is stated rather than hidden -- a benign
        # ``key=<value>`` label loses its value in diagnostics too (the
        # measured population is small: ``config.py``'s "Failed to encrypt
        # config value (key=...)" is the only shipped log line of that
        # shape, and the name it prints is a secret-bearing config key
        # anyway).
        "key",
    }
)


def _is_sensitive_log_key(key: object) -> bool:
    """Return whether a ``key=value`` label in a log line names a secret.

    The hyphen normalization is applied to BOTH checks (TASK-19555; the defect
    itself is filed as TASK-19558). It used to be computed and then passed
    only to the ``_LOG_ONLY_SENSITIVE_FIELDS`` membership test, while
    ``is_sensitive_config_key`` received the raw key -- and that predicate's
    ``_key``/``_token``/``_secret``/``_password`` rules are suffix matches on
    underscore forms. So every hyphenated HTTP header name whose sensitivity
    comes from a suffix rather than the ``api-key`` containment rule --
    ``x-auth-token``, ``x-session-key``, ``x-client-secret`` -- was classified
    as harmless and its value was written out verbatim. Those are exactly the
    names provider request logging produces.

    Normalizing cannot create a false positive here: ``max-tokens`` normalizes
    to ``max_tokens``, which still does not end in ``_token``.
    """
    normalized = str(key).strip().lower().replace("-", "_")
    return (
        is_sensitive_config_key(key)
        or is_sensitive_config_key(normalized)
        or normalized in _LOG_ONLY_SENSITIVE_FIELDS
    )


_ASSIGNMENT_PREFIX = re.compile(
    r"""
    (?<![A-Za-z0-9_.-])
    (?:
        (?P<quote>["'])(?P<quoted_key>[A-Za-z0-9_.-]+)(?P=quote)
        |
        (?P<plain_key>[A-Za-z0-9_.-]+)
    )
    [ \t]*[:=][ \t]*
    """,
    re.IGNORECASE | re.VERBOSE,
)
_URL_USERINFO = re.compile(r"(https?://)[^/?#\s\r\n]*@", re.IGNORECASE)
_BEARER = re.compile(
    r"(?<![A-Za-z0-9_-])(Bearer\s+)(\S+)",
    re.IGNORECASE,
)
_STANDALONE_CREDENTIALS = (
    re.compile(r"(?<![A-Za-z0-9_-])sk-proj-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])"),
    re.compile(r"(?<![A-Za-z0-9_-])sk-ant-api03-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])"),
    # TASK-26022: Claude subscription OAuth access/refresh tokens (borrowed
    # read-only from Claude Code's credential file) — same envelope as API
    # keys, different prefix family.
    re.compile(
        r"(?<![A-Za-z0-9_-])sk-ant-o[ar]t\d{2}-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])"
    ),
    # TASK-19555 review: OpenRouter keys carry hyphens inside the body, so the
    # generic `sk-[A-Za-z0-9]{20,}` rule below never matched one.
    re.compile(r"(?<![A-Za-z0-9_-])sk-or-v1-[A-Za-z0-9]{20,}(?![A-Za-z0-9_-])"),
    re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9]{20,}(?![A-Za-z0-9_-])"),
    re.compile(r"(?<![A-Za-z0-9_-])AIza[A-Za-z0-9_-]{35}(?![A-Za-z0-9_-])"),
    # GitHub personal access / OAuth / server / refresh tokens, and the
    # fine-grained `github_pat_` form.
    re.compile(r"(?<![A-Za-z0-9_-])gh[pousr]_[A-Za-z0-9]{30,}(?![A-Za-z0-9_-])"),
    re.compile(r"(?<![A-Za-z0-9_-])github_pat_[A-Za-z0-9_]{30,}(?![A-Za-z0-9_-])"),
    # Hugging Face user access tokens.
    re.compile(r"(?<![A-Za-z0-9_-])hf_[A-Za-z0-9]{30,}(?![A-Za-z0-9_-])"),
    # AWS access key ids (the id alone identifies an account and is paired
    # with a secret often logged on the same line).
    re.compile(
        r"(?<![A-Za-z0-9_-])(?:AKIA|ASIA|AGPA|AIDA|AROA)[0-9A-Z]{16}(?![A-Za-z0-9_-])"
    ),
    # Slack bot/user/app tokens.
    re.compile(r"(?<![A-Za-z0-9_-])xox[baprs]-[A-Za-z0-9-]{10,}(?![A-Za-z0-9_-])"),
    # JSON Web Tokens: three base64url segments, header first. Bearer-prefixed
    # ones are already covered; these are the bare ones in bodies and errors.
    re.compile(
        r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
        r"\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"
    ),
)


_LINE_BREAK = re.compile(r"[\r\n]")


def _line_end(text: str, start: int) -> int:
    """Return the first CR/LF index at or after start, or len(text). One
    search: ``find("\\r")`` scanned to the end of text with no CR in it,
    once per line (TASK-41510 review r16tr-rev1)."""
    line_break = _LINE_BREAK.search(text, start)
    return line_break.start() if line_break is not None else len(text)


def _after_line_break(text: str, index: int) -> int:
    """Advance over LF, CR, or one CRLF pair; end stays at len(text)."""
    if index >= len(text):
        return len(text)
    if text[index] == "\r" and index + 1 < len(text) and text[index + 1] == "\n":
        return index + 2
    if text[index] in "\r\n":
        return index + 1
    return index


def _find_quoted_end(text: str, value_start: int, quote: str) -> tuple[int, bool]:
    """Return the closing-quote/line-end index and whether it closed."""
    index = value_start
    while index < len(text):
        character = text[index]
        if character in "\r\n":
            return index, False
        if character == quote:
            return index, True
        if (
            character == "\\"
            and index + 1 < len(text)
            and text[index + 1] not in "\r\n"
        ):
            index += 2
            continue
        index += 1
    return len(text), False


def _apply_replacements(text: str, spans: list[tuple[int, int]]) -> str:
    """Build one output from sorted non-overlapping spans and REDACTION_MARKER."""
    if not spans:
        return text

    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        parts.extend((text[cursor:start], REDACTION_MARKER))
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


#: Labels that name a credential in a diagnostic line. TASK-41133 split them
#: from the PII labels below so a shared Trace bundle masks these alone.
_DIAGNOSTIC_CREDENTIAL_LABELS = frozenset(
    {
        "password",
        "passwd",
        "pwd",
        "passphrase",
        "secret",
        "token",
        "api_key",
        "apikey",
        "authorization",
        "proxy_authorization",
        "cookie",
        "set_cookie",
        "credential",
        "credentials",
        "private_key",
        "access_token",
        "refresh_token",
        "auth_token",
        "api_token",
        "session_token",
        "client_secret",
        "ocp_apim_subscription_key",
        "aws_secret_access_key",
        "secret_access_key",
        "secret_key",
        "bearer_token",
    }
)
_DIAGNOSTIC_CREDENTIAL_SUFFIXES = (
    "_api_key",
    "_api_key_fallback",
    "_apikey",
    "_password",
    "_passphrase",
    "_secret",
    "_auth_token",
    "_api_token",
    "_access_token",
    "_refresh_token",
    "_session_token",
)
_DIAGNOSTIC_PII_LABELS = frozenset(
    {
        "email",
        "email_address",
        "phone",
        "phone_number",
        "ssn",
        "user",
        "username",
        "postal_address",
        "street_address",
        "passport_number",
        "credit_card",
        "card_number",
    }
)
#: Labels that describe a value rather than hold it (``api_key_env_var``).
_DIAGNOSTIC_METADATA_SUFFIXES = (
    "_env_var",
    "_configured",
    "_present",
    "_source",
    "_name",
)


def _credential_label_text(label: str) -> bool:
    """Whether a shared bundle judges the value of a label ``_LABEL``
    matched: a compound credential label (``_is_credential_label``), or one
    the older simple-label rule covered (``token``, ``db.password``; TASK-41510).
    Other single words (``Authorization``, ``cookie``) are not. A cached
    credential label answers from ``_LABEL_FACTS``; any other label is
    checked directly and never cached (review r15tr-rev12)."""
    if _SIMPLE_CREDENTIAL_LABEL.search(label):
        return True
    if label.isalnum():
        return False
    facts = _LABEL_FACTS.get(label)
    return facts[2] if facts else _is_credential_label(label)


def _is_credential_label(key: object) -> bool:
    """Whether a diagnostic label names a credential (not PII)."""
    normalized = str(key).casefold().replace("-", "_")
    if normalized.endswith(_DIAGNOSTIC_METADATA_SUFFIXES):
        return False
    return (
        normalized in _DIAGNOSTIC_CREDENTIAL_LABELS
        or normalized.endswith(_DIAGNOSTIC_CREDENTIAL_SUFFIXES)
        or normalized.startswith("x_")
        and normalized.endswith(("_token", "_key"))
    )


def _is_sensitive_diagnostic_key(key: object) -> bool:
    """Recognize credential/PII labels without hiding every key or token."""
    normalized = str(key).casefold().replace("-", "_")
    if normalized.endswith(_DIAGNOSTIC_METADATA_SUFFIXES):
        return normalized in {"full_name", "first_name", "last_name", "user_name"}
    return normalized in _DIAGNOSTIC_PII_LABELS or _is_credential_label(key)


def _protocol_value_end(text: str, start: int, end: int) -> int:
    """Stop at a separate log field, never inside a header parameter or quote."""
    index = start + 7 if text[start : start + 7].casefold() == "digest " else start
    parameter_start = index
    while index < end:
        character = text[index]
        if character in "\"'":
            quoted_end, closed = _find_quoted_end(text, index + 1, character)
            if not closed:
                return end
            index = quoted_end + 1
            continue
        if character in ",;":
            parameter_start = index + 1
        if character in " \t":
            boundary = index
            while index < end and text[index] in " \t":
                index += 1
            if text[index : index + 1] == "|":
                index += 1
                while index < end and text[index] in " \t":
                    index += 1
            # Cookie parameters follow semicolons; Digest parameters follow
            # commas. A new assignment separated only by whitespace (or a log
            # pipe) after a complete parameter belongs to the surrounding log.
            if (
                index < end
                and not text[index].isdigit()
                # An Expires timestamp such as 10:18:14 is header content.
                and _ASSIGNMENT_PREFIX.match(text, index, end)
                and text[parameter_start:boundary].strip()
            ):
                return boundary
            continue
        index += 1
    return end


def _redact_assignments(text: str, *, diagnostic: bool = False) -> str:
    """Classify label prefixes first, collect replacement spans, and always advance."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    cached_line_end = -1
    authorities = iter(_DIAGNOSTIC_URL_USERINFO.finditer(text) if diagnostic else ())
    authority = next(authorities, None)
    while match := _ASSIGNMENT_PREFIX.search(text, cursor):
        while authority is not None and authority.end() <= match.start():
            authority = next(authorities, None)
        key = match.group("quoted_key") or match.group("plain_key")
        sensitive = (
            _is_sensitive_diagnostic_key(key)
            if diagnostic
            else _is_sensitive_log_key(key)
        )
        # A colon inside URI userinfo is not a User: label, even in a username
        # such as alice+user. Quoted keys and key=value fields remain labels,
        # so an adjacent JSON/password field cannot be consumed as userinfo.
        if not sensitive or (
            authority is not None
            and authority.start() < match.start() < authority.end()
            and match.group("plain_key") is not None
            and match.group().rstrip().endswith(":")
        ):
            cursor = match.end()
            continue

        value_start = match.end()
        if value_start == len(text):
            cursor = value_start
            continue
        if text[value_start] in "\r\n":
            cursor = _after_line_break(text, value_start)
            continue

        if text[value_start] in "\"'":
            quote = text[value_start]
            value_end, closed = _find_quoted_end(text, value_start + 1, quote)
            if value_end > value_start + 1:
                spans.append((value_start + 1, value_end))
            cursor = value_end + 1 if closed else _after_line_break(text, value_end)
            continue

        if not diagnostic or value_start > cached_line_end:
            cached_line_end = _line_end(text, value_start)
        line_end = cached_line_end
        value_end = line_end
        normalized = key.casefold().replace("-", "_")
        protocol_value = diagnostic and (
            normalized in {"cookie", "set_cookie"}
            or normalized in {"authorization", "proxy_authorization"}
            and text[value_start : value_start + 7].casefold() == "digest "
        )
        if protocol_value:
            value_end = _protocol_value_end(text, value_start, line_end)
        elif diagnostic:
            for following in _ASSIGNMENT_PREFIX.finditer(text, value_start, line_end):
                if (
                    following.start() > value_start
                    and text[following.start() - 1] in " \t,;&"
                ):
                    value_end = following.start()
                    while value_end > value_start and text[value_end - 1] in " \t,;&":
                        value_end -= 1
                    break
        spans.append((value_start, value_end))
        cursor = (
            value_end if value_end < line_end else _after_line_break(text, line_end)
        )

    return _apply_replacements(text, spans)


def sanitize_trace_credentials_v1(text: str) -> str:
    """Preserve the original string projection used by credentials-v1 traces.

    Stored artifacts are compared with this projection when extending a trace.
    Changing it would make unchanged history appear modified. Application logs
    use ``sanitize_string`` instead.

    Args:
        text: The string to sanitize

    Returns:
        Sanitized string with sensitive data redacted
    """
    if not isinstance(text, str):
        return str(text)

    result = _redact_assignments(text)
    result = _URL_USERINFO.sub(r"\1" + REDACTION_MARKER + "@", result)
    result = _BEARER.sub(r"\1" + REDACTION_MARKER, result)
    for pattern in _STANDALONE_CREDENTIALS:
        result = pattern.sub(REDACTION_MARKER, result)
    return result


#: `sk_`/`rk_` secret and restricted keys, live or test (TASK-39061), masked
#: by diagnostics and by the credentials-v2 trace projection (TASK-39123).
#: They are not in ``_STANDALONE_CREDENTIALS`` because
#: ``sanitize_trace_credentials_v1`` reads that tuple, and its stored
#: projection must not change. The 24-character floor and the body's lack of
#: `_` keep identifiers such as ``sk_live_mode_enabled`` readable.
_UNDERSCORE_KEY_CREDENTIALS = (
    re.compile(r"(?<![A-Za-z0-9_-])[rs]k_(?:live|test)_[A-Za-z0-9]{24,}(?![A-Za-z0-9_-])"),
)
#: Stripe/Svix webhook signing secrets and Groq keys (TASK-40301): masked in
#: logs, shared Trace bundles and, from credentials-v3 on, stored traces
#: (TASK-41133). Not in v1 or v2, whose stored output must not change.
_V3_KEY_CREDENTIALS = (
    re.compile(r"(?<![A-Za-z0-9_-])whsec_[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9_-])"),
    re.compile(r"(?<![A-Za-z0-9_-])gsk_[A-Za-z0-9]{20,}(?![A-Za-z0-9_-])"),
)


def trace_credential_key_spans(text: str) -> list[tuple[int, int]]:
    """Code-point spans of the keys a newer projection masks beyond v1.

    A stored v1 or v2 policy's frozen PII spans are applied to its own
    projection; a span that cuts one of these keys is widened over it first,
    so no key fragment survives the current masks (TASK-39123, TASK-41133).
    A key the stored projection already masked is no longer in its text.
    """
    return [
        m.span()
        for p in (*_UNDERSCORE_KEY_CREDENTIALS, *_V3_KEY_CREDENTIALS)
        for m in p.finditer(text)
    ]


def sanitize_trace_credentials_v2(text: str) -> str:
    """The stored credentials-v2 trace projection: v1 plus bare `sk_`/`rk_` keys.

    Traces stored between TASK-39123 and TASK-41133 carry it. Readers that
    compare with or index into such a trace ask for it, then re-apply the
    current projection.

    Args:
        text: The string to sanitize.

    Returns:
        ``sanitize_trace_credentials_v1(text)`` with underscore keys masked.
    """
    result = sanitize_trace_credentials_v1(text)
    for pattern in _UNDERSCORE_KEY_CREDENTIALS:
        result = pattern.sub(REDACTION_MARKER, result)
    return result


def sanitize_trace_credentials_v3(text: str) -> str:
    """The credentials-v3 trace projection: v2 plus `whsec_` and `gsk_` keys.

    New traces are stored under it, and every read, export and share of a
    stored v1 or v2 trace re-applies it (TASK-41133).

    Args:
        text: The string to sanitize.

    Returns:
        ``sanitize_trace_credentials_v2(text)`` with those keys masked.
    """
    result = sanitize_trace_credentials_v2(text)
    for pattern in _V3_KEY_CREDENTIALS:
        result = pattern.sub(REDACTION_MARKER, result)
    return result


_DIAGNOSTIC_PII_PATTERNS = (
    re.compile(r"(?i)(?<![\w.%+-])[a-z0-9._%+-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+"),
    re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    re.compile(
        r"(?<!\w)(?:\+\d{1,3}[ -]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}(?!\w)"
    ),
)
_DIAGNOSTIC_URL_USERINFO = re.compile(
    r"((?<![a-z0-9+.-])[a-z][a-z0-9+.-]*://)[^/?#\s]*@", re.IGNORECASE
)
_DIAGNOSTIC_URL = re.compile(
    r"(?<![a-z0-9+.-])[a-z][a-z0-9+.-]*://[^\s\"'<>]+", re.IGNORECASE
)
_URL_QUERY_KEY = re.compile(r"([?&]key=)[^&#\s]*", re.IGNORECASE)
_CONNECTION_LABEL = re.compile(
    r"\b(?:connection[-_]string|dsn)[\"']?\s*[:=]", re.IGNORECASE
)
_CONNECTION_PRIVATE_FIELD = re.compile(
    r"(?<![\w.-])((?:uid|user[ _]id|user|username|pwd|password)\s*=\s*)"
    r"""(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\{(?:\}\}|[^}])*\}|[^;\s]+)""",
    re.IGNORECASE,
)
_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----[\s\S]*?"
    r"(?:-----END (?:[A-Z0-9]+ )?PRIVATE KEY-----|$)"
)
_TOKEN_CREDENTIALS = (
    *_STANDALONE_CREDENTIALS,
    *_UNDERSCORE_KEY_CREDENTIALS,
    *_V3_KEY_CREDENTIALS,
)
#: Credential-shaped forms of three log rules (TASK-40301 review I2). Logs
#: and the stored trace projections keep the broader rules, which also match
#: code and docs a shared bundle must keep: ``f"Bearer {api_key}"``,
#: ``Bearer <token>``, ``ssh://git@github.com/...``, prose naming a PEM
#: header. Here a Bearer token starts with a token character and runs 8
#: non-space characters (review r14ep-rev2 C1: Sanctum's ``5|<secret>``),
#: user info has a password or a 16-character user part (M-f: ``jdoe1234@``
#: stays), and an unterminated private key has a base64 body, after any
#: ``Proc-Type``/``DEK-Info`` header lines (M-g).
_BEARER_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_-])(Bearer\s+)([A-Za-z0-9_.~+/=-]\S{7,})", re.IGNORECASE
)
_URL_CREDENTIAL_USERINFO = re.compile(
    r"(?<![a-z0-9+.-])[a-z][a-z0-9+.-]*://"
    r"(?:[^/?#\s@:]*:[^/?#\s@]*|[^/?#\s@:]{16,})@",
    re.IGNORECASE,
)
_PRIVATE_KEY_CREDENTIAL = re.compile(
    r"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----"
    r"(?:[\s\S]*?-----END (?:[A-Z0-9]+ )?PRIVATE KEY-----"
    r"|\s*(?:[A-Za-z-]+:[^\n]*\n)*\s*[A-Za-z0-9+/]{16,}[\s\S]*)"
)
#: Every credential value shape logs mask, in its credential-shaped form. A
#: shared Trace bundle masks with this same tuple (TASK-40301), so the export
#: cannot drift from logs again; logs apply the token rules as they are and
#: broader forms of the other three.
CREDENTIAL_VALUE_PATTERNS = (
    _PRIVATE_KEY_CREDENTIAL,
    _URL_CREDENTIAL_USERINFO,
    _BEARER_TOKEN,
    *_TOKEN_CREDENTIALS,
)


#: A label: a word, ``api key`` (after any dotted name: ``self.API key``,
#: review r16tr-rev2), or a compound name (TASK-41133), either after any
#: leading hyphens (``--client-secret=``, TASK-41510). It may follow a ``.``
#: that does not end a name (``users[0].password``, ``$.token``,
#: ``...password``; reviews r16tr-rev1, rev2); ``db.password`` is one label.
#: Labels may repeat separators (pydantic's ``APP__X``) and may be quoted, or
#: escape-quoted inside a JSON string (``\"client_secret\":``). A separator
#: holds at most one ``.``, so a label after ``..`` cannot run on through
#: every later ``..`` (quadratic; review r16tr-rev3). A label may also follow
#: a run of ``-``, ``_`` and ``.`` that starts after a non-word character
#: (``_-password:``, ``-.password:``, as base's ``\b`` read them); the run is
#: matched once, from its start, so a run of them stays linear (review
#: r16tr-rev5). Any
#: non-line-break whitespace may surround the separator (NBSP, ideographic
#: space; review r16tr-rev2). ``==`` is a comparison, never a separator
#: (review r15tr-rev8).
_LABEL = re.compile(
    r"""
    (?:(?<![A-Za-z0-9_-])(?<![A-Za-z0-9_-]\.)|(?<![A-Za-z0-9_.-])[-_.]++(?=[A-Za-z0-9]))
    (?:\\?(?P<label_quote>["']))?
    (?P<label>(?i:-*(?:[A-Za-z0-9_]+[.-])*api[ ]key)|-*[A-Za-z0-9]+(?:(?:[_-]++\.?|\.)[_-]*+[A-Za-z0-9]+)*)
    (?(label_quote)\\?(?P=label_quote))
    [^\S\r\n]*(?::|=(?!=))[^\S\r\n]*
    """,
    re.VERBOSE,
)
#: The labels the export's older simple-label rule masked whole (TASK-41510):
#: one of these words after any non-word character (its ``\b``). Within a
#: label that is only a ``.`` or ``-`` (``db.password``, ``page-token``).
_SIMPLE_CREDENTIAL_LABEL = re.compile(
    r"(?:^|[.-])(?:api[ _-]?key|token|passwd|password|secret)\Z", re.IGNORECASE
)
#: A bare value runs to whitespace, a quote, or a ``&``/``;``/``,`` followed
#: by whitespace, a quote, the end or another ``name=`` field. Any other
#: ``&``/``;``/``,`` belongs to the value (``Tr0ub4dor&3``).
_BARE_VALUE = re.compile(
    r"""(?:[^\s"'&;,]|[&;,](?![\s"']|$|[A-Za-z0-9_.-]+[ \t]*[:=]))*+"""
)
#: Placeholders and templates that may hold spaces, read whole: ``<your
#: token>``, ``{{ secrets.X }}``, ``${{ secrets.X }}``, ``%(secret)s``, and the
#: bracketed markers this code emits (a shared bundle keeps the ``.`` before
#: ``[local path withheld]`` when it replaces ``./path``).
_TEMPLATE_VALUE = re.compile(
    r"<[^<>\r\n]{1,256}>|\$?\{\{[^{}\r\n]{0,256}\}\}|%\([A-Za-z_]\w*\)[sdr]"
    r"|(?:\.{1,2}|~)?\[[A-Za-z][A-Za-z ]{0,62}\]"
)
#: A code expression read whole, quotes included: a name chain with at least
#: one call or index (``resp.json()['access_token']``, ``os.environ["X"]``,
#: ``os.getenv("DB_PASSWORD", "...")``). Its arguments are names or literals.
_CODE_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
_CODE_ARGUMENT = (
    rf"""(?:{_CODE_NAME}[ \t]*=[ \t]*)?"""
    rf"""(?:{_CODE_NAME}(?:\.{_CODE_NAME})*|"[^"\r\n]*"|'[^'\r\n]*')"""
)
_CODE_ARGUMENTS = rf"""(?:{_CODE_ARGUMENT}(?:[ \t]*,[ \t]*{_CODE_ARGUMENT})*)?"""
_CODE_VALUE = re.compile(
    rf"""{_CODE_NAME}(?:\.{_CODE_NAME})*"""
    rf"""(?:\({_CODE_ARGUMENTS}\)|\[{_CODE_ARGUMENTS}\])"""
    rf"""(?:\.{_CODE_NAME}|\({_CODE_ARGUMENTS}\)|\[{_CODE_ARGUMENTS}\])*"""
)
#: A literal inside code; group 1 is the ``(`` or ``[`` (and any space or
#: line break) before a first argument or an index.
_CODE_LITERAL = re.compile(
    r"""([(\[][ \t\r\n]*)?(?:\"\"\"([\s\S]*?)\"\"\"|'''([\s\S]*?)'''|"([^"\r\n]*)"|'([^'\r\n]*)')"""
)
#: Names a first argument or index may hold (review r15tr-rev7): an
#: environment variable name (``"MY_APP_CS"``), a compound key in a call
#: (``cfg.get("client_secret")``, not ``SecretStr("secret")``) and any
#: lowercase key in an index (``cfg["token"]``).
_ENV_NAME = re.compile(r"[A-Z]+|[A-Z]+[0-9]*(?:_[A-Z0-9]+)+")
_COMPOUND_KEY = re.compile(r"[a-z]+(?:_[a-z0-9]+)+")
_INDEX_KEY = re.compile(r"[a-z]+(?:_[a-z0-9]+)*")
_PATH_KEY = re.compile(r"[a-z][a-z0-9_.-]*(?:/[a-z][a-z0-9_.-]*)+")
_CODE_VALUE_END = re.compile(r"""[\s,;&)\]}"']|\.(?:\s|$)|$""")
#: A YAML block scalar indicator; its value is the indented lines after it.
_YAML_BLOCK = re.compile(r"[|>][-+]?[ \t]*(?:\r\n|\r|\n)")
#: The value below a label that ends its line, past blank lines and any
#: non-line-break indentation (``password:\n  hunter2``, ``password =\n
#: hunter2``, NBSP too; review r16tr-rev3): not a nested ``name:`` or
#: ``"name":``, an env or config ``NAME=`` field (an empty ``KEY=`` before
#: another; base64 such as ``YWJj...=`` is a value), a ``- `` list item or a
#: ``# `` comment (``-hunter2`` and ``#hunter2`` are values; review r16tr-rev2).
_VALUE_BELOW = re.compile(
    r"""(?:\r\n|\r|\n)(?:[^\S\r\n]*+(?:\r\n|\r|\n))*+[^\S\r\n]*+"""
    r"""(?!\s|[-#](?=\s|\Z)|[A-Za-z0-9_.-]+[^\S\r\n]*:(?:\s|\Z)"""
    r"""|(?:[A-Z][A-Z0-9_]*|[a-z][a-z0-9_]*)[^\S\r\n]*=(?!=)"""
    r"""|(?P<q>["'])[^"'\r\n]*(?P=q)[^\S\r\n]*:)"""
)
#: The start of a code header whose subject is the label after it, matched
#: up to the label: ``if``/``elif``/``while`` with an optional ``not`` and
#: comparison (``if token:``, ``if not password:``, ``if x in token:``),
#: bare ``match``/``case``/``class``, ``for <name> in``, and
#: ``with``/``except <expr> as`` (``with open(p) as token:``, ``except
#: KeyError as secret:``). Prose that starts with the same word (``try this
#: password:``, ``for prod the password:``), a bare ``for password:``,
#: ``with password:`` or ``except token:``, ``def`` and a list item are not
#: (reviews r16tr-rev1..rev3; coordinator rulings). Known remainder:
#: lowercase prose shaped like a comparison (``if it is password:``).
_EXPR = r"[A-Za-z_][\w.]*(?:\([^()\r\n]*\))?"
_CODE_HEADER = re.compile(
    r"[ \t]*(?:"
    r"(?:(?:el)?if|while)[ \t]+(?:not[ \t]+)?"
    rf"(?:{_EXPR}[ \t]+(?:in|is|and|or|==|!=)[ \t]+(?:not[ \t]+)?)?"
    r"|(?:match|case|class)[ \t]+"
    r"|(?:async[ \t]+)?for[ \t]+[A-Za-z_]\w*[ \t]+in[ \t]+"
    rf"|(?:(?:async[ \t]+)?with|except)[ \t]+{_EXPR}[ \t]+as[ \t]+"
    r")"
)
_INDENTED_LINES = re.compile(r"(?:[ \t]+[^\r\n]*(?:\r\n|\r|\n|$))+")
#: Longer values are masked whole, never examined (review r15tr-rev3).
_MAX_EXAMINED_VALUE = 2048

# The look-alike allowlist (review r15tr-rev3 ruling, tightened in rev4): a
# labelled value is masked unless it is one of these.
_EMITTED_MARKERS = (
    "[credential redacted]",
    "[credential key redacted]",
    "[identifier redacted]",
    "[local path withheld]",
    "[redacted]",
    REDACTION_MARKER,
)
_PLACEHOLDER = re.compile(
    r"""(?x)
    your[-_][a-z_-]+ | YOUR[-_][A-Z_-]+ | Your(?:[A-Z][a-z]+)+Here
    | (?i:example|placeholder|redacted|none|null|nil|undefined|true|false)
    | [xX*]{3,} | [xX*]+(?:[-_.][xX*]+)+
    """
)
#: A word holding a letter and a digit; anchored at word starts so a long
#: run is scanned once, not once per position.
_LETTER_DIGIT_WORD = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*[0-9])[A-Za-z0-9]+"
)
_UPPER_SNAKE = r"_*[A-Z]+[0-9]*(?:_(?:[A-Z]+[0-9]*|[0-9]+))*"
_LOWER_SNAKE = r"_*[a-z]+[0-9]*(?:_(?:[a-z]+[0-9]*|[0-9]+))*"
_LOWER_KEBAB = r"[a-z]+[0-9]*(?:-(?:[a-z]+[0-9]*|[0-9]+))+"
_CAMEL = r"_*[A-Za-z][a-z]+(?:[A-Z][a-z]+)*[0-9]*"
_SNAKE_NAME = re.compile(rf"{_LOWER_SNAKE}|{_UPPER_SNAKE}")
_CONVENTIONAL_NAME = re.compile(rf"{_LOWER_SNAKE}|{_UPPER_SNAKE}|{_LOWER_KEBAB}|{_CAMEL}")
_DOTTED_NAME = re.compile(
    rf"(?:{_LOWER_SNAKE}|{_UPPER_SNAKE}|{_CAMEL})(?:\.(?:{_LOWER_SNAKE}|{_UPPER_SNAKE}|{_CAMEL}))+"
)
_TEMPLATE_NAME = re.compile(rf"{_LOWER_SNAKE}|{_DOTTED_NAME.pattern}")
_REFERENCE = re.compile(
    rf"""(?x)
    \$(?:{_UPPER_SNAKE}) | %(?:{_UPPER_SNAKE})% | \$\{{(?:{_UPPER_SNAKE}|{_LOWER_SNAKE})\}}
    """
)
_PATH_ROOTS = (
    "/Users/", "/home/", "/etc/", "/var/", "/opt/", "/run/", "/tmp/", "/usr/",
    "/srv/", "/root/", "/mnt/", "/private/", "/Volumes/", "./", "../", "~/",
)
_DRIVE_PATH = re.compile(r"[A-Za-z]:\\")
#: A random-looking path segment: upper, lower and digits, or a long run of
#: letters and digits (``~/Xk9mP2vL7wQ8`` is a secret, ``id_rsa2`` is not).
_RANDOM_SEGMENT = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"(?=[A-Za-z0-9]*[a-z])(?=[A-Za-z0-9]*[A-Z])(?=[A-Za-z0-9]*[0-9])[A-Za-z0-9]{6,}"
    r"|(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*[0-9])[A-Za-z0-9]{12,})"
)
_URL_SCHEME = re.compile(r"[a-z][a-z0-9+.-]*://|file:/")
_TIME = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?")
_ISO_DATE = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"
)
#: Words a code identifier for a credential may hold besides its label's.
_CREDENTIAL_WORDS = frozenset(
    {"secret", "key", "token", "password", "passphrase", "credential", "api",
     "auth", "access", "refresh", "client", "session"}
)
#: Words a reference or template must name (``$CLIENT_SECRET``, ``{api_key}``).
_REFERENCE_WORDS = _CREDENTIAL_WORDS | {"secrets", "keys", "tokens", "credentials", "passwords"}
_PASSWORD_LABEL_WORDS = frozenset({"password", "passphrase", "passwd", "pwd"})
#: A label ending in ``secret`` or ``secret_key`` names a value a person
#: chooses and is read as strictly as a password, except these (review
#: r15tr-rev5).
_ISSUED_SECRET_LABELS = (
    "client_secret", "consumer_secret", "api_secret", "aws_secret_access_key",
    "secret_access_key",
)
#: The only bare words that a sentence continuing after them makes prose
#: (``client_secret: see section 4.2``).
_PROSE_FUNCTION_WORDS = frozenset(
    {"a", "an", "the", "it", "is", "see", "set", "use", "in", "on", "at", "to",
     "from", "for", "of", "by", "via", "as", "and", "or", "not", "this", "that",
     "here", "below", "above"}
)
_IDENTIFIER_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")
_ANGLE_NAME = re.compile(r"[A-Za-z][A-Za-z_-]*")
#: Mixed case beyond a camelCase word (``XkmPvLwQ``, not ``JavaScript``).
_MIXED_CASE = re.compile(r"[a-z][A-Z][A-Za-z]*[a-z][A-Z]")
#: Five consonants in a row (``y`` counts as a vowel): rare in an English
#: word, common in random letters (``XKQZVBNMPLWRTDXK``, ``...jklmn``).
_CONSONANT_RUN = re.compile(r"[b-df-hj-np-tv-xzB-DF-HJ-NP-TV-XZ]{5}")
_LETTER_RUN = re.compile(r"[A-Za-z]{12,}")


def _random_letters(word: str) -> bool:
    """A run of 12 or more letters with five consonants in a row
    (``XKQZVBNMPLWRTDXK``, ``abcdefghijklmn``); ``strengths`` is too short
    (review r15tr-rev8)."""
    return any(_CONSONANT_RUN.search(run) for run in _LETTER_RUN.findall(word))


def _words(name: str) -> tuple[str, ...]:
    """Lowercase words of a name, split on separators and capitals, digits and
    ``v`` version tags dropped (``ClientSecret2`` -> client, secret). Not
    cached: it also reads values, which must not linger in a cache."""
    return tuple(
        word.lower()
        for word in _IDENTIFIER_WORD.findall(name)
        if not word.isdigit() and word.lower() != "v"
    )


#: Facts about label text the masker asks for on every line: its words,
#: whether it is strict, whether it names a credential. Only a plain label
#: that names a credential is kept here: no digit, no random-looking letters,
#: at most 64 characters. A secret can sit in a label position
#: (``"GOCSPX-...": "x"``, ``"letmein_now": "x"``), and no cache may hold
#: text that could be a secret (reviews r15tr-rev11, rev12), so anything else
#: is worked out each time. Bounded: it stops admitting at 1024 labels.
_LABEL_FACTS: dict[str, tuple[tuple[str, ...], bool, bool]] = {}
_PLAIN_LABEL = re.compile(r"[A-Za-z_.-]{1,64}")


def _label_facts(label: str) -> tuple[tuple[str, ...], bool, bool]:
    """``(words, strict, credential)`` for a label, cached only when it is a
    plain label that names a credential."""
    facts = _LABEL_FACTS.get(label)
    if facts is None:
        words = _words(label)
        normalized = label.casefold().replace("-", "_").strip("_")
        strict = bool(set(words) & _PASSWORD_LABEL_WORDS) or (
            normalized.endswith(("secret", "secret_key"))
            and not normalized.endswith(_ISSUED_SECRET_LABELS)
        )
        facts = (words, strict, _is_credential_label(label))
        if (
            facts[2]
            and len(_LABEL_FACTS) < 1024
            and _PLAIN_LABEL.fullmatch(label)
            and not _MIXED_CASE.search(label)
            and not _random_letters(label)
        ):
            _LABEL_FACTS[label] = facts
    return facts


def _label_words(label: str) -> tuple[str, ...]:
    """``_words`` of a label, through the plain-label cache."""
    return (_LABEL_FACTS.get(label) or _label_facts(label))[0]


def _label_name(name: str, label: str) -> bool:
    """The label's own name, versioned by at most two digits: ``client_secret``,
    ``clientSecret``, ``client_secret_2``, ``refresh_token2``, ``ClientSecret2``."""
    tokens = tuple(token.lower() for token in _IDENTIFIER_WORD.findall(name))
    label_words = _label_words(label)
    if tokens[: len(label_words)] != label_words:
        return False
    rest = tokens[len(label_words) :]
    if rest[:1] == ("v",):
        rest = rest[1:]
    return not rest or (len(rest) == 1 and rest[0].isdigit() and len(rest[0]) <= 2)


def _code_identifier(value: str, label: str, label_words: set[str]) -> bool:
    """A conventional name for the label's value (reviews r15tr-rev4, rev5).

    The last dotted segment counts and must be lower_snake, UPPER_SNAKE,
    lower-kebab or camelCase (so ``Client_Secret_2024`` is not a name). It is
    the label's own name, versioned (``client_secret_2``), or a compound
    snake name made only of credential words (``token_v2``, ``SECRET_KEY_V2``).
    """
    if not _CONVENTIONAL_NAME.fullmatch(value.lstrip(":")) and not _DOTTED_NAME.fullmatch(value):
        return False
    last = value.lstrip(":").rsplit(".", 1)[-1]
    if _label_name(last.strip("_"), label):
        return True
    if not (_SNAKE_NAME.fullmatch(last) and "_" in last.strip("_")):
        return False
    words = _words(last)
    return bool(words) and set(words) <= label_words | _CREDENTIAL_WORDS


def _names_a_credential(name: str) -> bool:
    """Whether a reference or template names a credential (``CLIENT_SECRET``)."""
    return bool(set(_words(name)) & _REFERENCE_WORDS)


def _template_or_marker(value: str) -> bool:
    """A template naming a credential, a name or phrase in angle brackets, or a
    marker this code emits."""
    inner = None
    if value.startswith("<") and value.endswith(">"):
        # A digit-free name or a phrase (``<your-client-secret>``, review r15tr-rev6).
        return bool(_ANGLE_NAME.fullmatch(value, 1, len(value) - 1)) or (
            " " in value and not _LETTER_DIGIT_WORD.search(value)
        )
    if value.startswith(("{{", "${{")) and value.endswith("}}"):
        inner = value[value.index("{{") + 2 : -2].strip()
    elif value.startswith("%(") and len(value) > 3:
        inner = value[2 : value.rfind(")")]
    elif value.startswith("{") and value.endswith("}"):
        inner = value[1:-1]
    if inner is not None:
        return bool(_TEMPLATE_NAME.fullmatch(inner)) and _names_a_credential(inner)
    return value.lstrip(".~") in _EMITTED_MARKERS


def _code_literals_look_alike(value: str, label: str, strict: bool) -> bool:
    """Every quoted literal inside code is a look-alike (review r15tr-rev6).

    A first argument or index may be a name (``os.environ["MY_APP_CS"]``,
    ``['access_token']``); any other literal (a default, a keyword argument,
    a fallback after ``or``) is judged as the label's value, so a placeholder
    is kept and ``os.getenv("DB_PASSWORD", "postgres")`` is masked.
    """
    for match in _CODE_LITERAL.finditer(value):
        literal = next(part for part in match.groups()[1:] if part is not None)
        opener = (match.group(1) or "")[:1]
        if opener and (
            (_ENV_NAME.fullmatch(literal) and not _random_letters(literal))
            or (_INDEX_KEY if opener == "[" else _COMPOUND_KEY).fullmatch(literal)
            or (_PATH_KEY.fullmatch(literal) and not _LETTER_DIGIT_WORD.search(literal))
            # the label's own name (``request.form.get("password")``, r16tr-rev1)
            or (_INDEX_KEY.fullmatch(literal) and _label_name(literal, label))
        ):
            continue
        if not _kept_look_alike(literal, label, quoted=True, prose_follows=False, strict=strict):
            return False
    return True


def _known_path(value: str) -> bool:
    """A path under a known root with no random-looking segment."""
    return (value.startswith(_PATH_ROOTS) or bool(_DRIVE_PATH.match(value))) and not (
        _RANDOM_SEGMENT.search(value)
    )


def _strict_label(label: str) -> bool:
    """A password label, or one whose value a person chooses: a label ending
    in ``secret`` or ``secret_key`` bar issued ones (review r15tr-rev5).
    Worked out in ``_label_facts``."""
    return (_LABEL_FACTS.get(label) or _label_facts(label))[1]


def _kept_look_alike(
    value: str, label: str, *, quoted: bool, prose_follows: bool, strict: bool = False
) -> bool:
    """Whether a labelled value is on the look-alike allowlist (kept).

    Everything else is masked (review r15tr-rev3: under-masking leaks a
    secret; over-masking only makes a shared bundle less readable). Under a
    password label or a label whose value a person chooses (one ending in
    ``secret`` or ``secret_key``, bar issued ones such as ``client_secret``),
    only placeholders other than ``changeme``, references, known-root paths,
    dotted code and an UPPER_SNAKE constant holding every label word are kept
    (reviews r15tr-rev4, rev5). Code is kept only when its literals are
    look-alikes too.
    """
    if not value:
        return True
    if len(value) > _MAX_EXAMINED_VALUE:
        return False
    label_words = set(_label_words(label))
    password_label = strict or _strict_label(label)
    code = _CODE_VALUE.fullmatch(value)
    if code is not None and not _code_literals_look_alike(value, label, strict):
        return False
    if (
        _PLACEHOLDER.fullmatch(value)
        or _template_or_marker(value)
        or (_REFERENCE.fullmatch(value) and _names_a_credential(value))
        or _known_path(value)
        or (code is not None and "." in value.split("(", 1)[0].split("[", 1)[0])
        or _DOTTED_NAME.fullmatch(value)
        or (
            re.fullmatch(_UPPER_SNAKE, value) is not None
            and label_words <= set(_words(value))
        )
    ):
        return True
    if password_label:
        return False
    if (
        value.casefold() in {"changeme", "change_me", "change-me"}
        or code is not None
        or _URL_SCHEME.match(value)
        or _TIME.fullmatch(value)
        or _ISO_DATE.fullmatch(value)
        or _code_identifier(value, label, label_words)
    ):
        return True
    # Prose under a secret, token or key label: a quoted sentence with no
    # token-shaped word, or a function word that words continue (the caller
    # sets ``prose_follows`` only for those).
    return ((quoted and " " in value) and not _LETTER_DIGIT_WORD.search(value)) or prose_follows


def _trimmed_end(text: str, start: int, end: int) -> int:
    """End of a bare value without a trailing full stop or unbalanced closer."""
    pairs = {")": "(", "]": "[", "}": "{"}
    balance = {
        closer: text.count(closer, start, end) - text.count(opener, start, end)
        for closer, opener in pairs.items()
    }
    while end > start:
        last = text[end - 1]
        if last == "." or balance.get(last, 0) > 0:
            if last in balance:
                balance[last] -= 1
            end -= 1
            continue
        break
    return end


def _value_span(text: str, start: int, line_end: int) -> tuple[int, int, bool]:
    """``(start, end, quoted)`` of the value that begins at ``start``.

    ``line_end`` is the end of ``start``'s line. A quoted value honours
    ``\\`` escapes (as ``_find_quoted_end`` does); an escape-quoted one
    (``\\"...\\"``, JSON inside a string) ends at the next escape-quote on the
    line. An unclosed quote leaves a bare value after it. A YAML block scalar
    (``|`` or ``>``) is its indented lines.
    """
    if text.startswith(('\\"', "\\'"), start):
        close = text.find(text[start : start + 2], start + 2, line_end)
        if close != -1:
            return start + 2, close, True
        start += 2
    elif start < len(text) and text[start] in "\"'":
        close, closed = _find_quoted_end(text, start + 1, text[start])
        if closed:
            return start + 1, close, True
        start += 1
    block = _YAML_BLOCK.match(text, start)
    if block is not None:
        lines = _INDENTED_LINES.match(text, block.end())
        if lines is not None:
            body = lines.group()
            return (
                lines.start() + len(body) - len(body.lstrip(" \t")),
                lines.start() + len(body.rstrip("\r\n")),
                False,
            )
    for template in (_TEMPLATE_VALUE.match(text, start), _CODE_VALUE.match(text, start)):
        if template is not None and _CODE_VALUE_END.match(text, template.end()):
            return start, template.end(), False
    end = _BARE_VALUE.match(text, start).end()
    return start, _trimmed_end(text, start, end), False


#: Words after a masked value under a strict label, up to the next field.
_MORE_WORDS = re.compile(r"[ \t]+(?![A-Za-z0-9_.-]+[ \t]*[:=])[^\s]")
_NEXT_FIELD = re.compile(r"[^\S\r\n]+(?=[A-Za-z0-9_.-]+[^\S\r\n]*[:=])")
_NEXT_WORD = re.compile(rf"[ \t]+(\S{{1,{_MAX_EXAMINED_VALUE}}})")


#: Tokens of a code expression: a quoted string (skipped whole, triple
#: quotes included), a line continuation, the space before a field
#: (``user=``, ``status:``, ``"refresh_token":``; ``==`` is no field), a ``?``
#: that a ternary's ``:`` answers, brackets, separators and line breaks.
_EXPRESSION_TOKEN = re.compile(
    r"""\"(?:\"\"[\s\S]*?\"\"\"|(?:[^"\\\r\n]|\\.)*+\")"""
    r"""|'(?:''[\s\S]*?'''|(?:[^'\\\r\n]|\\.)*+')|\\(?:\r\n|\r|\n)"""
    r"""|(?<![^\S\r\n])[^\S\r\n]++(?=(?:["'](?P<quoted_key>[A-Za-z0-9_.-]+)["']|[A-Za-z0-9_.-]+)[^\S\r\n]*(?P<separator>=(?!=)|:))"""
    r"""|\?(?![?.])|[()\[\]{},;\r\n]"""
)
#: What may follow a top-level comma for the comma to end the expression: a
#: field, a closer or the line end (``CLIENT_SECRET, "..."`` is a tuple).
_AFTER_COMMA = r"""[ \t]*(?:["']?[A-Za-z0-9_.-]+["']?[ \t]*(?:=(?!=)|:)|[)\]}\r\n]|$)"""
_FIELD_AFTER_COMMA = re.compile(_AFTER_COMMA)


#: What may follow a value that ends its expression: a ``;`` or closer, a
#: comma before a field, the line end, or the next field (a quoted key only
#: when it is a credential label: ``"note": "..."`` belongs to the value). A
#: value with no bracket of its own followed by one of these is a lone value,
#: judged as it is.
_LONE_VALUE_END = re.compile(
    rf"""[^\S\r\n]*(?:[;)\]}}\r\n]|$|,(?={_AFTER_COMMA}))"""
    r"""|(?<![^\S\r\n])[^\S\r\n]++(?=(?:["'](?P<quoted_key>[A-Za-z0-9_.-]+)["']|[A-Za-z0-9_.-]+)[^\S\r\n]*(?:=(?!=)|:))"""
)
_OPENER = re.compile(r"[(\[{]")


def _lone_value(text: str, value: str, end: int, quoted: bool, label: str) -> bool:
    """A value with no bracket of its own that its expression ends right
    after (``_LONE_VALUE_END``): judged as it is, with no expression scan."""
    if not quoted and _OPENER.search(value):
        return False
    lone = _LONE_VALUE_END.match(text, end + quoted)
    if lone is None:
        return False
    key = lone.group("quoted_key")
    # A quoted key can be any text (a secret too), so it is not cached; the
    # same label again (a repeated field) needs no check.
    return not key or key == label or _is_credential_label(key)  # '"note": "..."' belongs to the value
#: A call, read whole by its brackets when the grammar does not parse it.
_CALL_HEAD = re.compile(rf"{_CODE_NAME}(?:\.{_CODE_NAME})*\(")
#: A comparison with space before its operator (``a == b``, not ``abc==``):
#: a label right after one is an operand when a statement follows its colon
#: (``if client_secret == other_secret: return``, review r15tr-rev9).
_SPACED_COMPARISON = re.compile(r"[ \t](?:==|!=|<=|>=)[ \t]*\Z")
_STATEMENT = re.compile(r"(?:return|pass|break|continue|raise)\b")
#: A type annotation under a credential label (``client_secret: str``,
#: ``Optional[str]``, ``str | None``): kept, and a ``= default`` after it is
#: the value (review r15tr-rev9).
_TYPE_NAME = (
    r"(?:str|bytes|bytearray|int|float|bool|complex|dict|list|tuple|set|frozenset"
    r"|object|Any|None|Optional|Union|List|Dict|Tuple|Set|FrozenSet|Sequence"
    r"|Mapping|Iterable|(?:pydantic\.)?Secret(?:Str|Bytes))"
)
_TYPE_ONE = rf"{_TYPE_NAME}(?:\[{_TYPE_NAME}(?:[ \t]*,[ \t]*{_TYPE_NAME})*\])?"
_TYPE_TWO = rf"{_TYPE_NAME}(?:\[{_TYPE_ONE}(?:[ \t]*,[ \t]*{_TYPE_ONE})*\])?"
_ANNOTATION = re.compile(
    rf"{_TYPE_TWO}(?:[ \t]*\|[ \t]*{_TYPE_TWO})*(?=[ \t]*(?:=(?!=)|[,;)\]}}#\r\n]|$))"
)
_DEFAULT = re.compile(r"[ \t]*=(?!=)[ \t]*")
#: Words a statement or a type annotation above can start with.
_STATEMENT_WORDS = ("return", "pass", "break", "continue", "raise")
_CODE_WORDS = _STATEMENT_WORDS + (
    "str", "bytes", "int", "float", "bool", "complex", "dict", "list", "tuple", "set",
    "frozenset", "object", "Any", "None", "Optional", "Union", "List", "Dict", "Tuple",
    "Set", "FrozenSet", "Sequence", "Mapping", "Iterable", "Secret", "pydantic.",
)
#: An unquoted word after a label inside a kept call (``f(db_password=x)``).
_INNER_WORD = re.compile(r"""[^\s,()\[\]{}"']+""")


def _inner_values_look_alike(span: str) -> bool:
    """Every credential label inside a kept call holds a look-alike, judged
    under its own label: a quoted literal or the word after it. The call is
    then kept whole and not scanned again, so nested calls stay linear
    (review r15tr-rev9)."""
    for inner in _LABEL.finditer(span, 1):
        name = inner.group("label")
        if not _credential_label_text(name):
            continue
        literal = _CODE_LITERAL.match(span, inner.end())
        if literal is not None and literal.group(1) is None:
            value = next(part for part in literal.groups()[1:] if part is not None)
            if not _kept_look_alike(value, name, quoted=True, prose_follows=False):
                return False
            continue
        word = _INNER_WORD.match(span, inner.end())
        if word is not None and not _kept_look_alike(word.group(), name, quoted=False, prose_follows=False):
            return False
    return True


def _expression_end(text: str, start: int) -> tuple[int, bool]:
    """Where the expression at ``start`` ends, and whether its brackets
    balance there (reviews r15tr-rev7 to rev9).

    That is before a ``;``, a comma followed by a field, an unbalanced closer,
    the next field (a quoted key only when it is a credential label) or the
    line end; quoted strings are skipped whole, so a fallback after ``or``,
    ``||``, ``??`` or ``if ... else`` is inside it. While a bracket is open,
    or after a ``\\`` continuation, it runs on over line ends. One longer
    than ``_MAX_EXAMINED_VALUE`` runs to the end of the line that limit falls
    on, unbalanced.
    """
    limit = min(len(text), start + _MAX_EXAMINED_VALUE)
    depth = ternaries = 0
    for token in _EXPRESSION_TOKEN.finditer(text, start, limit):
        mark = token.group()[0]
        if mark in "\"'\\":
            continue  # a string, or a line continuation
        if mark in "\r\n":
            if not depth:
                return token.start(), True
        elif mark in "([{":
            depth += 1
        elif mark in ")]}":
            if not depth:
                return token.start(), True
            depth -= 1
        elif depth:
            continue
        elif mark == "?":
            ternaries += 1
        elif mark == ";" or (mark == "," and _FIELD_AFTER_COMMA.match(text, token.end(), limit)):
            return token.start(), True
        elif mark == ",":
            continue  # a tuple or argument list runs on
        elif ternaries and token.group("separator") == ":":
            ternaries -= 1  # "a ? b : c": the colon answers the "?"
        elif token.group("quoted_key") and not _is_credential_label(token.group("quoted_key")):
            continue  # '"note": "..."' after a value belongs to it
        else:
            return token.start(), True
    if limit == len(text):
        return limit, not depth
    return _line_end(text, limit), False


def _field_end(text: str, start: int, end: int, line_end: int) -> int:
    """Where a value that runs on past ``end`` stops: before the next field
    (``user=bob``) or at the line end, a trailing separator left visible."""
    field = _NEXT_FIELD.search(text, end, line_end)
    stop = field.start() if field else line_end
    return start + len(text[start:stop].rstrip(" \t,;|"))


def _owns_value_below(text: str, label: re.Match) -> bool:
    """Whether the value below a label that ends its line is the label's.

    After ``=`` it always is (``const password =\n  "..."``). After ``:`` it
    is, for a YAML key, prose (``my password:\nhunter2``) and a list item
    (``1. Password:\n   hunter2``), unless the label is the subject of a code
    header (``if token:\n    return token``; ``_CODE_HEADER``). It reads at
    most ``_MAX_EXAMINED_VALUE`` characters back for the line's start (an
    unbounded ``rfind`` was quadratic on CR-only text, review r16tr-rev2); a
    label further than that from its line's start owns the value below.
    """
    if label.group().rstrip().endswith("="):
        return True
    start = label.start()
    window = max(0, start - _MAX_EXAMINED_VALUE)
    newline = text.rfind("\n", window, start)
    line_break = max(newline, text.rfind("\r", max(newline + 1, window), start))
    if line_break < 0 and window:
        return True
    return _CODE_HEADER.fullmatch(text, line_break + 1, start) is None


def mask_credential_labels(text: str, marker: str = REDACTION_MARKER) -> str:
    """Mask every value under a credential label but look-alikes.

    The labels are the compound ones diagnostics mask (``client_secret=``,
    ``refresh_token:``, ``X-Auth-Token:``) and the simple ones the export's
    older rule masked whole (``token=``, ``password:``, ``api key:``;
    ``_credential_label_text``); PII labels such as ``user_name=`` are left
    alone. A value is masked whole unless it is on the look-alike
    allowlist (``_kept_look_alike``). A value that is kept, or that sits under
    any other label, is scanned again, so a label nested in it (a URL, JSON
    text, a query string) is found too (TASK-41133).

    Args:
        text: Text that may hold ``label=value`` or ``label: value`` pairs.
        marker: What replaces each masked value.

    Returns:
        The text with those values replaced and everything else unchanged.
    """
    parts: list[str] = []
    position = 0
    line_end = -1
    # Labels seen in this text, for this call only: freed on return, so it
    # holds no text beyond the call that passed it in (review r15tr-rev12).
    seen: dict[str, bool] = {}
    while (label := _LABEL.search(text, position)) is not None:
        name = label.group("label")
        credential = seen.get(name)
        if credential is None:
            credential = seen[name] = _credential_label_text(name)
        if credential:
            if label.end() > line_end:  # one scan per line, not per label
                line_end = _line_end(text, label.end())
            value_start = label.end()
            if value_start == line_end:
                # A value on a later line is this label's, as the older
                # simple-label rule read it (TASK-41510, review r16tr-rev1).
                below = _VALUE_BELOW.match(text, value_start)
                if below is not None and _owns_value_below(text, label):
                    value_start = below.end()
                    line_end = _line_end(text, value_start)
            if text.startswith(_CODE_WORDS, value_start) and label.group().rstrip().endswith(":"):
                if (
                    text.startswith(_STATEMENT_WORDS, value_start)
                    and _STATEMENT.match(text, value_start)
                    and _SPACED_COMPARISON.search(text, max(0, label.start() - 8), label.start())
                ):
                    parts.append(text[position:value_start])  # an operand, not a field
                    position = value_start
                    continue
                annotation = _ANNOTATION.match(text, value_start)
                if annotation is not None:
                    default = _DEFAULT.match(text, annotation.end())
                    if default is None:
                        parts.append(text[position : annotation.end()])  # a type, no value
                        position = annotation.end()
                        continue
                    value_start = default.end()
            start, end, quoted = _value_span(text, value_start, line_end)
            prose_follows = False
            if not quoted and text[start:end] in _PROSE_FUNCTION_WORDS:
                # A function word is prose before a word, never before a token
                # ("the <token>"); then the value runs on to the next field.
                word = _NEXT_WORD.match(text, end, line_end)
                if word is not None:
                    following = word.group(1)
                    prose_follows = not (
                        _LETTER_DIGIT_WORD.search(following)
                        or _MIXED_CASE.search(following)
                        or _random_letters(following)
                    )
                    if not prose_follows:
                        end = _field_end(text, start, word.end(1), line_end)
            value = text[start:end]
            kept = _kept_look_alike(value, name, quoted=quoted, prose_follows=prose_follows)
            if not (
                prose_follows
                or text.startswith(('\\"', "\\'"), value_start)
                or _lone_value(text, value, end, quoted, name)
            ):
                # Every value is judged with the whole expression it starts
                # (reviews r15tr-rev7, rev8): kept only when every literal in
                # it is a look-alike and no random-looking word sits outside
                # its quotes, else masked whole ("x or 'GOCSPX-...'",
                # '"{}".format("...")', a dict, a lambda, a call over lines).
                # A lone value is a span of one and is judged as before.
                span_start = value_start if quoted else start
                stop, balanced = _expression_end(text, span_start)
                span_end = span_start + len(text[span_start:stop].rstrip(" \t\r\n,;|\\"))
                while span_end > end and text[span_end - 1] == ".":  # a full stop
                    span_end -= 1
                call = None if kept or quoted else _CALL_HEAD.match(value)
                kept_call = False
                if span_end > end + quoted or call is not None:
                    if (
                        call is not None
                        and balanced
                        and not _CODE_VALUE.fullmatch(value)
                        and not _strict_label(name)
                    ):
                        # A call the grammar does not parse (nested, a number
                        # or keyword argument, over lines) is judged by its
                        # literals and words, as a tail is (review r15tr-rev9).
                        # Not under a strict label.
                        kept, end = True, start + len(call.group())
                        kept_call = True
                    # Judge what follows the value as it was judged above:
                    # its literals and the words outside them.
                    tail = text[end + quoted : span_end]
                    has_literal = '"' in tail or "'" in tail
                    kept = kept and (
                        span_end - span_start <= _MAX_EXAMINED_VALUE
                        and (
                            not has_literal
                            # an opener just before the tail marks a first argument
                            or _code_literals_look_alike(tail if quoted else text[end - 1 : span_end], name, False)
                        )
                        and not _RANDOM_SEGMENT.search(_CODE_LITERAL.sub(" ", tail) if has_literal else tail)
                    )
                    if kept and kept_call:
                        kept = _inner_values_look_alike(text[span_start:span_end])
                        if kept:  # judged whole, labels inside included
                            parts.append(text[position:span_end])
                            position = span_end
                            continue
                    start, end = span_start, span_end
            # A masked value reads as an emitted marker, so it is kept on a
            # second pass.
            if not kept:
                if not quoted and _strict_label(name) and _MORE_WORDS.match(text, end, line_end):
                    # An unquoted passphrase runs on to the next field or the line end.
                    end = _field_end(text, start, end, line_end)
                parts.extend((text[position:start], marker))
                position = end
                continue
        parts.append(text[position : label.end()])
        position = label.end()
    parts.append(text[position:])
    return "".join(parts)


def mask_url_query_keys(text: str, marker: str = REDACTION_MARKER) -> str:
    """Mask a URL's ``?key=``/``&key=`` value whatever its shape, as logs
    mask it (TASK-41510): ``?key=YOUR_API_KEY`` and ``?key=python`` too. A
    trailing full stop or unbalanced closer stays visible; an empty value and
    a marker this code emitted are left alone, so a second pass changes
    nothing.

    Args:
        text: Text that may hold URLs.
        marker: What replaces each value.

    Returns:
        The text with those values replaced and everything else unchanged.
    """
    parts: list[str] = []
    position = 0
    for url in _DIAGNOSTIC_URL.finditer(text):
        for key in _URL_QUERY_KEY.finditer(url.group()):
            start = url.start() + key.end(1)
            end = _trimmed_end(text, start, url.start() + key.end())
            if start == end or text.startswith(_EMITTED_MARKERS, start):
                continue
            parts.extend((text[position:start], marker))
            position = end
    parts.append(text[position:])
    return "".join(parts)


def _sanitize_diagnostic_text(text: str) -> str:
    """Mask private spans while retaining surrounding application diagnostics.

    The ``sanitize_trace_credentials_v1`` projection stays unchanged: stored trace
    artifacts use its versioned credential policy during equality checks.
    """
    result = _PRIVATE_KEY_BLOCK.sub(REDACTION_MARKER, text)
    result = "".join(
        _CONNECTION_PRIVATE_FIELD.sub(r"\1" + REDACTION_MARKER, line)
        if _CONNECTION_LABEL.search(line)
        else line
        for line in result.splitlines(keepends=True)
    )
    result = _redact_assignments(result, diagnostic=True)
    result = _DIAGNOSTIC_URL.sub(
        lambda match: _URL_QUERY_KEY.sub(r"\1" + REDACTION_MARKER, match.group()),
        result,
    )
    result = _DIAGNOSTIC_URL_USERINFO.sub(r"\1" + REDACTION_MARKER + "@", result)
    result = _BEARER.sub(r"\1" + REDACTION_MARKER, result)
    for pattern in (*_TOKEN_CREDENTIALS, *_DIAGNOSTIC_PII_PATTERNS):
        result = pattern.sub(REDACTION_MARKER, result)
    return result


def sanitize_string(text: str) -> str:
    """Mask recognized credentials and PII while retaining diagnostic text.

    Args:
        text: Log text; other values retain the historical string fallback.

    Returns:
        Text with private values masked and other fields preserved.
    """
    if not isinstance(text, str):
        return str(text)
    return redact_user_paths(_sanitize_diagnostic_text(text))


#: POSIX home roots whose next path segment is an operating-system account
#: name. Deliberately case-SENSITIVE: a case-insensitive ``/users/`` would also
#: rewrite REST URLs such as ``https://api.example.com/users/alice``, which is
#: not a home directory and whose shape a maintainer may need.
_HOME_ROOTS_POSIX = re.compile(
    r"(?:(?<=^)|(?<=[^A-Za-z0-9_.-]))(?:/Users/|/home/)[^/\\\s:'\"<>|]+"
)

#: Windows home roots -- a drive form (``C:\Users\name``) and a UNC form
#: (``\\SERVER\Users\name``). Matched case-INSENSITIVELY in full: Windows paths
#: are case-insensitive end to end, so ``c:\users\name`` and ``C:\Users\name``
#: are the same path and both must redact. An earlier revision applied
#: ``re.IGNORECASE`` to the drive letter only in intent but matched ``Users``
#: as a literal, so the lowercase spelling -- and every UNC path -- kept the
#: account name (caught in TASK-19555 review). There is no URL-collision risk
#: here because these forms are backslash-delimited.
_HOME_ROOTS_WINDOWS = re.compile(
    r"(?:(?<=^)|(?<=[^A-Za-z0-9_.-]))"
    r"(?:[A-Za-z]:\\Users\\|\\\\[^\\/\s:'\"<>|]+\\Users\\)"
    r"[^/\\\s:'\"<>|]+",
    re.IGNORECASE,
)

#: Cap applied to one line before redaction (TASK-19555 review). Two reasons,
#: and neither is a bypass -- truncation keeps strictly LESS data than the
#: uncapped line:
#:
#: * cost. Redaction is linear in line length, so an uncapped kv-dense line
#:   costs proportionally: ~21 us at a normal ~140 chars, ~553 us at 3.4 KB,
#:   ~7.9 ms at 100 KB -- paid on whichever thread emitted the record, the UI
#:   thread included.
#: * disclosure. The buffer bounds the number of lines but not their size, so
#:   a single dumped provider response body used to be retained whole.
#:
#: 2000 chars is far past the point a log line is readable in a terminal, so
#: nothing legible is lost, and it bounds the worst case at roughly 0.3 ms.
MAX_REDACTED_LINE_CHARS = 2000

#: Characters that may continue a path segment. Used as both lookbehind and
#: lookahead when substituting a literal home directory, so the match has to
#: cover a WHOLE segment.
_PATH_SEGMENT_CHARS = r"A-Za-z0-9_.\-~"

#: Whitespace treated as a token boundary when truncating (see
#: ``redact_log_line``). Credentials never contain whitespace, so cutting on
#: one cannot split a credential in half.
_TOKEN_BOUNDARY_CHARS = " \t\n\r\v\f"


@lru_cache(maxsize=8)
def _home_literal_pattern(candidates: tuple[str, ...]) -> "re.Pattern | None":
    """Compile a segment-anchored alternation over literal home directories.

    TASK-19555 Qodo round. This used to be a bare ``str.replace(home, "~")``,
    which is wrong in both directions when one home path is a prefix of
    another:

    * with ``$HOME=/Users/jan``, the line ``/Users/janedoe/Notes/x.pdf``
      became ``~edoe/Notes/x.pdf`` -- half of a DIFFERENT account's name left
      in place, still identifying, and with the ``/Users/`` prefix destroyed
      so ``_HOME_ROOTS_POSIX`` could no longer clean up after it;
    * with ``$HOME=/srv/appdata``, the unrelated ``/srv/appdata-backup/db``
      became ``~-backup/db``.

    Anchoring on ``_PATH_SEGMENT_CHARS`` at both ends makes a home match only
    a complete final segment. Longest candidate first, so when ``$HOME`` and
    ``Path.home()`` disagree by a prefix the more specific one wins.

    Args:
        candidates: Home-directory strings; empty entries are ignored.

    Returns:
        A compiled pattern, or None when no usable candidate was supplied.
    """
    usable = sorted(
        {candidate for candidate in candidates if candidate and len(candidate) > 1},
        key=len,
        reverse=True,
    )
    if not usable:
        return None
    alternation = "|".join(re.escape(candidate) for candidate in usable)
    return re.compile(
        rf"(?<![{_PATH_SEGMENT_CHARS}])(?:{alternation})(?![{_PATH_SEGMENT_CHARS}])"
    )


def redact_user_paths(text: str) -> str:
    """Replace home-directory prefixes with ``~`` so no account name survives.

    ``/Users/alice/Notes/Q3.pdf`` becomes ``~/Notes/Q3.pdf``: the operating
    system account name -- a real-name identifier on most desktops, and the
    single most repeated identity token in this application's path logging --
    is gone, while everything a maintainer reads the path for is intact.

    The running user's own home is substituted first and literally, so the
    rule still holds for accounts that live outside ``/Users`` or ``/home``
    (``/root``, or any ``$HOME`` override). That substitution is anchored on
    path-segment boundaries -- see ``_home_literal_pattern``.

    Args:
        text: One log line, or any free text that may embed filesystem paths.

    Returns:
        ``text`` with home-directory roots collapsed to ``~``.
    """
    if not isinstance(text, str) or not text:
        return text

    # os.environ first: Path.home() falls back to the password database, which
    # would rewrite paths the user's own $HOME no longer points at.
    try:
        resolved_home = str(Path.home())
    except (OSError, RuntimeError):  # pragma: no cover - no resolvable home
        resolved_home = ""
    pattern = _home_literal_pattern(
        (
            os.environ.get("HOME") or "",
            os.environ.get("USERPROFILE") or "",
            resolved_home,
        )
    )
    result = pattern.sub("~", text) if pattern is not None else text
    result = _HOME_ROOTS_POSIX.sub("~", result)
    return _HOME_ROOTS_WINDOWS.sub("~", result)


def content_fingerprint(
    value: object, *, chars: int = CONTENT_FINGERPRINT_CHARS
) -> str:
    """Return a stable correlation handle when a caller needs content identity.

    A fingerprint distinguishes different values that share a text prefix and
    remains stable across process restarts. It does not replace diagnostic
    text or credential/PII redaction. An unsalted digest of a short, guessable
    value can be recovered by trying candidates, so this is not a secrecy
    mechanism.

    Args:
        value: Any value; non-strings are rendered with ``str`` first.
        chars: Hex characters to keep. Defaults to
            ``CONTENT_FINGERPRINT_CHARS``.

    Returns:
        A lowercase hex digest prefix, or ``EMPTY_FINGERPRINT`` when the value
        is ``None`` or renders empty.
    """
    if value is None:
        return EMPTY_FINGERPRINT
    text = value if isinstance(value, str) else str(value)
    if not text:
        return EMPTY_FINGERPRINT
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return digest[: max(1, chars)]


def redact_log_line(text: str, max_length: int = MAX_REDACTED_LINE_CHARS) -> str:
    """Mask recognized credentials and PII in one formatted diagnostic line.

    Credential labels, Bearer tokens, URL credentials and private key blocks
    are masked, as are recognizable email addresses, phone numbers, SSNs,
    labelled personal fields and account names in home paths. Ordinary fields
    such as model names, versions, phases and non-secret keys stay readable.
    This recognizes explicit formats; it cannot identify every opaque secret
    or infer personal information from arbitrary prose.

    Oversized lines are cut on a TOKEN boundary, and redaction then runs over
    everything that survives the cut (TASK-19555 Qodo round). The first
    revision truncated at a fixed offset and redacted afterwards, which
    manufactured secrets: every pattern in ``_STANDALONE_CREDENTIALS`` carries
    a minimum length, so a key straddling the cap was sliced into a fragment
    too short to match and the fragment stayed in the Logs view and in "Copy
    visible logs".

    Cutting on whitespace fixes that without giving up the cost bound, because
    a credential never contains whitespace: it is either wholly inside the cut
    and redacted, or wholly outside it and discarded. Redaction over the full
    line would also be correct, but it is linear in length -- 14-20 ms for a
    100 KB line, on whichever thread emitted the record -- and that bound is
    the reason the cap exists.

    Args:
        text: One fully formatted log line.
        max_length: Characters kept before the token-aligned cut, or 0 to
            redact the whole line however long it is. See
            ``MAX_REDACTED_LINE_CHARS``.

    Returns:
        The line, cut on a token boundary if oversized, with recognised
        credentials and home-directory account names redacted.
    """
    if not isinstance(text, str):
        text = str(text)
    original_length = len(text)
    suffix = ""
    if max_length > 0 and original_length > max_length:
        head = text[:max_length]
        boundary = max(
            (head.rfind(character) for character in _TOKEN_BOUNDARY_CHARS),
            default=-1,
        )
        if boundary <= 0:
            # One unbroken token longer than the cap. There is no cut that
            # cannot split it, and a 2,000-character prefix of an opaque token
            # is most of a secret, so the body is withheld rather than sliced.
            # Formatter output always has whitespace in its timestamp prefix,
            # so this is reached only by raw, unformatted input.
            return (
                f"{REDACTION_MARKER} [oversized unbroken log line withheld, "
                f"{original_length} chars]"
            )
        text = head[:boundary]
        suffix = f"… [truncated, {original_length} chars]"
    # The suffix is generated text, so it is appended after redaction rather
    # than being fed through it.
    return redact_user_paths(_sanitize_diagnostic_text(text)) + suffix


def sanitize_dict(data: Dict[str, Any], deep: bool = True) -> Dict[str, Any]:
    """
    Sanitize a dictionary by redacting sensitive fields.

    Args:
        data: The dictionary to sanitize
        deep: Whether to recursively sanitize nested structures

    Returns:
        New dictionary with sensitive fields redacted
    """
    if not isinstance(data, dict):
        return data

    result = {}
    for key, value in data.items():
        # Check if key is sensitive
        if _is_sensitive_diagnostic_key(key):
            result[key] = REDACTION_MARKER
        elif deep and isinstance(value, dict):
            result[key] = sanitize_dict(value, deep=True)
        elif deep and isinstance(value, list):
            result[key] = sanitize_list(value, deep=True)
        elif isinstance(value, str):
            # Preserve dictionary context for values containing ODBC fields.
            if str(key).casefold().replace("-", "_") in {"dsn", "connection_string"}:
                value = _CONNECTION_PRIVATE_FIELD.sub(r"\1" + REDACTION_MARKER, value)
            result[key] = sanitize_string(value)
        else:
            result[key] = value

    return result


def sanitize_list(data: List[Any], deep: bool = True) -> List[Any]:
    """
    Sanitize a list by processing each element.

    Args:
        data: The list to sanitize
        deep: Whether to recursively sanitize nested structures

    Returns:
        New list with sensitive data redacted
    """
    if not isinstance(data, list):
        return data

    result = []
    for item in data:
        if isinstance(item, dict) and deep:
            result.append(sanitize_dict(item, deep=True))
        elif isinstance(item, list) and deep:
            result.append(sanitize_list(item, deep=True))
        elif isinstance(item, str):
            result.append(sanitize_string(item))
        else:
            result.append(item)

    return result


def sanitize_log_params(*args, **kwargs) -> tuple:
    """
    Sanitize both positional and keyword arguments for logging.

    Returns:
        Tuple of (sanitized_args, sanitized_kwargs)
    """
    clean_args = []
    for arg in args:
        if isinstance(arg, dict):
            clean_args.append(sanitize_dict(arg))
        elif isinstance(arg, list):
            clean_args.append(sanitize_list(arg))
        elif isinstance(arg, str):
            clean_args.append(sanitize_string(arg))
        else:
            clean_args.append(arg)

    clean_kwargs = sanitize_dict(kwargs) if kwargs else {}

    return tuple(clean_args), clean_kwargs


def create_safe_log_message(template: str, *args, **kwargs) -> str:
    """
    Create a safe log message by sanitizing all parameters.

    Args:
        template: The log message template
        *args: Positional arguments for the template
        **kwargs: Keyword arguments for the template

    Returns:
        Formatted log message with sensitive data redacted
    """
    # First sanitize all arguments
    clean_args, clean_kwargs = sanitize_log_params(*args, **kwargs)

    # Format with sanitized arguments first
    try:
        if clean_args and clean_kwargs:
            formatted = template.format(*clean_args, **clean_kwargs)
        elif clean_args:
            formatted = template.format(*clean_args)
        elif clean_kwargs:
            formatted = template.format(**clean_kwargs)
        else:
            formatted = template
    except Exception:
        # If formatting fails, just use template
        formatted = template

    # Then sanitize the final result to catch any embedded secrets
    return sanitize_string(formatted)


# Convenience function for common use case
def safe_log(logger_func, message: str, *args, **kwargs):
    """
    Safely log a message by sanitizing all parameters first.

    Args:
        logger_func: The logger function to use (e.g., logger.info)
        message: The log message
        *args: Additional arguments
        **kwargs: Additional keyword arguments
    """
    safe_message = create_safe_log_message(message, *args, **kwargs)
    logger_func(safe_message)
