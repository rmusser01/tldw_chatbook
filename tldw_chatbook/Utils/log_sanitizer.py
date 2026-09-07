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
    re.compile(r"(?<![A-Za-z0-9_-])sk-ant-o[ar]t\d{2}-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])"),
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
    re.compile(r"(?<![A-Za-z0-9_-])(?:AKIA|ASIA|AGPA|AIDA|AROA)[0-9A-Z]{16}(?![A-Za-z0-9_-])"),
    # Slack bot/user/app tokens.
    re.compile(r"(?<![A-Za-z0-9_-])xox[baprs]-[A-Za-z0-9-]{10,}(?![A-Za-z0-9_-])"),
    # JSON Web Tokens: three base64url segments, header first. Bearer-prefixed
    # ones are already covered; these are the bare ones in bodies and errors.
    re.compile(
        r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
        r"\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"
    ),
)


def _line_end(text: str, start: int) -> int:
    """Return the first CR/LF index at or after start, or len(text)."""
    line_feed = text.find("\n", start)
    carriage_return = text.find("\r", start)
    endings = (index for index in (line_feed, carriage_return) if index != -1)
    return min(endings, default=len(text))


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


def _redact_assignments(text: str) -> str:
    """Classify label prefixes first, collect replacement spans, and always advance."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    while match := _ASSIGNMENT_PREFIX.search(text, cursor):
        key = match.group("quoted_key") or match.group("plain_key")
        if not _is_sensitive_log_key(key):
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

        line_end = _line_end(text, value_start)
        spans.append((value_start, line_end))
        cursor = _after_line_break(text, line_end)

    return _apply_replacements(text, spans)


def sanitize_string(text: str) -> str:
    """
    Sanitize a string by removing sensitive data patterns.

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
    """Return a stable, plaintext-free handle for a value too sensitive to log.

    A search query, a prompt, or a model response is user content, so a
    diagnostic must not carry its words. But the thing a maintainer actually
    reads such a diagnostic *for* is identity -- "is this the same query that
    failed a minute ago?", "did every result come back with the same malformed
    body?" -- and identity survives hashing. Truncating the value instead (the
    ``value[:50]`` idiom this function replaces, TASK-21700) keeps the words
    and loses the identity: two different long queries that share a prefix
    print identically, so the one property the line was read for is the one
    truncation destroys.

    Scope of the guarantee, stated plainly so it is not over-read: this
    removes plaintext from the line. It is **not** a secrecy mechanism against
    an adversary who already holds the log -- an unsalted digest of a short,
    guessable string can be recovered by trying candidates. It is not salted
    per process on purpose: a per-run salt would break exactly the
    across-restart correlation the fingerprint exists to provide, and the
    values it covers here never reach a persistent sink in the first place
    (``PersistentDiagnosticFilter`` admits only schema-validated metadata
    records). The honest claim is "a maintainer reading this log cannot read
    the user's words", not "this value is protected".

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
    """Redact credentials and user identity from one formatted log line.

    This is the sink-side redaction applied to every record entering the
    in-app log collector (TASK-19555). It is deliberately narrower than
    ADR-029's metadata-only admission filter, which is an all-or-nothing DROP
    and would empty the Logs screen of the very content it exists to show.
    The bar here is *what is never wanted*: secrets and the operating-system
    account name have no debugging value, so removing them costs a maintainer
    nothing.

    Two honest limits, both disclosed to users in the Logs screen copy:

    * It removes credentials in RECOGNISED formats -- labelled ``key=``-style
      assignments, ``Bearer`` prefixes, URL userinfo, and the standalone
      shapes in ``_STANDALONE_CREDENTIALS``. A bare opaque token in a format
      none of those match survives. This is a denylist and denylists are never
      complete; the claim is "recognised formats", not "all credentials".
    * It does NOT remove free-form user content -- a note title, a search
      query, a prompt, a tool argument -- because nothing at a sink can tell
      which substring of a message was interpolated from user data. That
      exposure is handled by bounding what the bulk share action exports.

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
    return redact_user_paths(sanitize_string(text)) + suffix


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
        if _is_sensitive_log_key(key):
            result[key] = REDACTION_MARKER
        elif deep and isinstance(value, dict):
            result[key] = sanitize_dict(value, deep=True)
        elif deep and isinstance(value, list):
            result[key] = sanitize_list(value, deep=True)
        elif isinstance(value, str):
            # Still sanitize string values for embedded secrets
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
