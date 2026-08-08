"""
Log sanitizer utilities to prevent sensitive data from being logged.

This module provides functions to scrub API keys, passwords, and other
sensitive information from log messages.
"""

import re
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
    normalized = str(key).strip().lower().replace("-", "_")
    return is_sensitive_config_key(key) or normalized in _LOG_ONLY_SENSITIVE_FIELDS


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
    re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9]{20,}(?![A-Za-z0-9_-])"),
    re.compile(r"(?<![A-Za-z0-9_-])AIza[A-Za-z0-9_-]{35}(?![A-Za-z0-9_-])"),
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
        line_end = _line_end(text, value_start)
        if value_start == line_end:
            cursor = _after_line_break(text, line_end)
            continue

        if text[value_start] in "\"'":
            quote = text[value_start]
            value_end, closed = _find_quoted_end(text, value_start + 1, quote)
            if value_end > value_start + 1:
                spans.append((value_start + 1, value_end))
            cursor = value_end + 1 if closed else _after_line_break(text, value_end)
            continue

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
