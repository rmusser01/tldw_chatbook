"""
Log sanitizer utilities to prevent sensitive data from being logged.

This module provides functions to scrub API keys, passwords, and other
sensitive information from log messages.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List

from tldw_chatbook.Utils.sensitive_config_keys import is_sensitive_config_key


REDACTION_MARKER = "***REDACTED***"
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


def redact_user_paths(text: str) -> str:
    """Replace home-directory prefixes with ``~`` so no account name survives.

    ``/Users/alice/Notes/Q3.pdf`` becomes ``~/Notes/Q3.pdf``: the operating
    system account name -- a real-name identifier on most desktops, and the
    single most repeated identity token in this application's path logging --
    is gone, while everything a maintainer reads the path for is intact.

    The running user's own home is substituted first and literally, so the
    rule still holds for accounts that live outside ``/Users`` or ``/home``
    (``/root``, or any ``$HOME`` override).

    Args:
        text: One log line, or any free text that may embed filesystem paths.

    Returns:
        ``text`` with home-directory roots collapsed to ``~``.
    """
    if not isinstance(text, str) or not text:
        return text

    result = text
    # os.environ first: Path.home() falls back to the password database, which
    # would rewrite paths the user's own $HOME no longer points at.
    for candidate in (os.environ.get("HOME"), os.environ.get("USERPROFILE")):
        if candidate and len(candidate) > 1:
            result = result.replace(candidate, "~")
    try:
        home = str(Path.home())
    except (OSError, RuntimeError):  # pragma: no cover - no resolvable home
        home = ""
    if len(home) > 1:
        result = result.replace(home, "~")
    result = _HOME_ROOTS_POSIX.sub("~", result)
    return _HOME_ROOTS_WINDOWS.sub("~", result)


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

    Args:
        text: One fully formatted log line.
        max_length: Characters kept before truncation. See
            ``MAX_REDACTED_LINE_CHARS`` for why truncation happens first.

    Returns:
        The line, truncated if oversized, with recognised credentials and
        home-directory account names redacted.
    """
    if not isinstance(text, str):
        text = str(text)
    if max_length > 0 and len(text) > max_length:
        # Truncate BEFORE redacting: redaction is linear in length, and the
        # tail is being discarded either way.
        text = f"{text[:max_length]}… [truncated, {len(text)} chars]"
    return redact_user_paths(sanitize_string(text))


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
