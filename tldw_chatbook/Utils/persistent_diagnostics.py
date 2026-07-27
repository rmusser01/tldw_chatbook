"""Strict admission boundary for metadata written to persistent diagnostics.

Chatbook's normal UI and terminal diagnostics may remain descriptive. The
persistent application sink is different: records originating in Chatbook code
must be emitted through :func:`log_persistent_metadata`, which accepts a small
schema and serializes values without arbitrary ``repr`` or exception text.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_PERSISTENT_METADATA_MARKER = "_tldw_metadata_only_record"
_SOURCE_PATH_FIELD = "_tldw_source_path"
_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,127}$")
_EVENT_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_PRIVATE_TOKEN_MARKERS = (
    "api-key",
    "api_key",
    "apikey",
    "bearer",
    "password",
    "secret",
    "sentinel",
    "sk-",
)

_TOKEN_FIELDS = frozenset(
    {
        "provider",
        "model",
        "operation",
        "status",
        "tool_name",
        "result_type",
        "exception_type",
        "error_category",
        "server_key",
        "initiator",
        "decision",
        "phase",
        "transport",
    }
)
_INTEGER_FIELDS = frozenset(
    {
        "status_code",
        "payload_length",
        "duration_ms",
        "retry_count",
        "unknown_argument_count",
        "result_size",
        "item_count",
        "batch_size",
        "chunk_count",
    }
)
_BOOLEAN_FIELDS = frozenset({"cache_hit", "streaming", "cancelled"})
_LIST_FIELDS = frozenset({"argument_names"})
_ALLOWED_FIELDS = _TOKEN_FIELDS | _INTEGER_FIELDS | _BOOLEAN_FIELDS | _LIST_FIELDS

def _is_chatbook_record(record: logging.LogRecord) -> bool:
    """Return whether a record originated in installed Chatbook code."""

    if record.name == "tldw_chatbook" or record.name.startswith("tldw_chatbook."):
        return True
    source_path = Path(
        getattr(record, _SOURCE_PATH_FIELD, None) or getattr(record, "pathname", "")
    )
    try:
        source_path.resolve().relative_to(_PACKAGE_ROOT)
    except (OSError, ValueError):
        return False
    return True


def _safe_token(value: Any) -> str:
    """Serialize an approved identity value without retaining arbitrary text."""

    if isinstance(value, str):
        normalized = value.strip()
        lowered = normalized.casefold()
        if _TOKEN_RE.fullmatch(normalized) and not any(
            marker in lowered for marker in _PRIVATE_TOKEN_MARKERS
        ):
            return normalized
    return "invalid"


def safe_metadata_token(value: Any) -> str:
    """Return the same bounded token serialization used by persistent logs."""

    return _safe_token(value)


def _safe_integer(value: Any) -> str:
    if isinstance(value, bool):
        return "invalid"
    if isinstance(value, int):
        return str(max(0, value))
    if isinstance(value, float) and math.isfinite(value):
        return str(max(0, int(value)))
    return "invalid"


def _safe_list(value: Any) -> str:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return "invalid"
    names = sorted({_safe_token(item) for item in value})
    if "invalid" in names:
        return "invalid"
    return ",".join(names) if names else "-"


def _format_metadata_value(field: str, value: Any) -> str:
    if field in _TOKEN_FIELDS:
        return _safe_token(value)
    if field in _INTEGER_FIELDS:
        return _safe_integer(value)
    if field in _BOOLEAN_FIELDS:
        return "true" if value is True else "false" if value is False else "invalid"
    if field in _LIST_FIELDS:
        return _safe_list(value)
    raise ValueError(f"unsupported persistent metadata field: {field}")


def log_persistent_metadata(
    target_logger: logging.Logger,
    level: int,
    event: str,
    **metadata: Any,
) -> None:
    """Emit one schema-validated record eligible for the persistent file sink.

    Arbitrary values, paths, exception strings, and object representations are
    deliberately unsupported. Invalid values for an approved field are
    represented as ``invalid`` instead of being echoed.
    """

    if not _EVENT_RE.fullmatch(event):
        raise ValueError("persistent diagnostic event must be a lowercase token")
    unsupported = sorted(set(metadata) - _ALLOWED_FIELDS)
    if unsupported:
        raise ValueError(
            "unsupported persistent metadata field(s): " + ", ".join(unsupported)
        )
    fields = [f"event={event}"]
    fields.extend(
        f"{field}={_format_metadata_value(field, metadata[field])}"
        for field in sorted(metadata)
    )
    target_logger.log(
        level,
        " ".join(fields),
        extra={_PERSISTENT_METADATA_MARKER: True},
    )


class PersistentDiagnosticFilter(logging.Filter):
    """Admit only metadata-only Chatbook records to a persistent handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        if _is_chatbook_record(record):
            return getattr(record, _PERSISTENT_METADATA_MARKER, False) is True
        # Third-party diagnostics often include URLs, bodies, exception text,
        # or object reprs. They remain available to UI/terminal handlers but
        # have no trusted schema for persistent admission.
        return False


__all__ = [
    "PersistentDiagnosticFilter",
    "log_persistent_metadata",
    "safe_metadata_token",
]
