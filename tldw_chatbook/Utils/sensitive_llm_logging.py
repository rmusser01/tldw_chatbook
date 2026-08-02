"""Request-scoped logging policy for sensitive LLM operations.

The policy is deliberately carried by a :class:`contextvars.ContextVar` so a
single auxiliary request can redact its final adapter diagnostics without
changing process-wide logger configuration or affecting concurrent chat work.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from urllib.parse import urlsplit


_SENSITIVE_LLM_REQUEST: ContextVar[bool] = ContextVar(
    "sensitive_llm_request", default=False
)
SENSITIVE_CONTENT_REDACTION = "<sensitive-content-redacted>"
SENSITIVE_ERROR_REDACTION = "<sensitive-error-detail-redacted>"


def is_sensitive_llm_request() -> bool:
    """Return whether the current execution context handles sensitive LLM data."""

    return _SENSITIVE_LLM_REQUEST.get()


@contextmanager
def sensitive_llm_request() -> Iterator[None]:
    """Mark the current request sensitive and restore the prior policy on exit."""

    token = _SENSITIVE_LLM_REQUEST.set(True)
    try:
        yield
    finally:
        _SENSITIVE_LLM_REQUEST.reset(token)


def safe_llm_log_value(value: object) -> object:
    """Return a value safe to interpolate into a diagnostic message.

    Call this before slicing, formatting, previewing, or serializing a value
    that may contain request or response content.
    """

    if is_sensitive_llm_request():
        return SENSITIVE_CONTENT_REDACTION
    return value


def safe_llm_error_detail(value: object) -> object:
    """Return error detail or a stable redaction for a sensitive request."""

    if is_sensitive_llm_request():
        return SENSITIVE_ERROR_REDACTION
    return value


def safe_llm_url_host(value: object) -> str:
    """Return only a URL host while sensitive; preserve ordinary diagnostics."""

    raw = str(value or "")
    if not is_sensitive_llm_request():
        return raw
    try:
        parsed = urlsplit(raw if "://" in raw else f"//{raw}")
        return parsed.hostname or "unknown"
    except ValueError:
        return "unknown"


def llm_content_byte_count(value: object) -> int:
    """Count UTF-8 content bytes without building a body preview or JSON string."""

    if value is None:
        return 0
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, Mapping):
        return sum(
            llm_content_byte_count(key) + llm_content_byte_count(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return sum(llm_content_byte_count(item) for item in value)
    return len(str(value).encode("utf-8"))


def llm_retry_count(configured: int) -> int:
    """Disable transport retries for a sensitive one-shot LLM request."""

    return 0 if is_sensitive_llm_request() else configured


def safe_llm_exception_message(exc: BaseException) -> str:
    """Return an exception description safe for logs and surfaced errors."""

    if is_sensitive_llm_request():
        return type(exc).__name__
    return str(exc)
