# task_incidents.py
"""TASK-26027: durable failure incidents for scheduled tasks.

Groups repeated failures of one task with the same normalized error
signature into a single incident so a task failing hourly for a week is one
acknowledgeable incident, not a week of identical notifications. Pure
signature normalization here; the durable store + state machine live on
ScheduledTasksDB.
"""

from __future__ import annotations

import hashlib
import re

#: Max length of a human-readable normalized signature before hashing-tail.
_MAX_SIGNATURE_CHARS = 200

# Volatile substrings that vary run to run and must not defeat grouping
# (AC#5). Order matters: paths before bare numbers so a path's digits are
# not stripped first into an unrecognizable fragment.
_VOLATILE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # ISO-8601 timestamps
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?"), "<ts>"),
    # hex blobs / uuids (8+ hex, or uuid shape)
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F-]{4,}\b"), "<uuid>"),
    (re.compile(r"\b[0-9a-fA-F]{8,}\b"), "<hex>"),
    # filesystem paths (posix + windows)
    (re.compile(r"(?:/[\w.\-]+)+/?"), "<path>"),
    (re.compile(r"[A-Za-z]:\\[\\\w.\-]+"), "<path>"),
    # durations / sizes / bare numbers
    (re.compile(r"\b\d+(?:\.\d+)?(?:ms|s|m|h|kb|mb|gb|b)?\b", re.IGNORECASE), "<n>"),
)


def normalize_error_signature(error: str | None) -> str:
    """A stable signature for one error, volatile details removed (AC#5).

    Two failures of the same shape (same error class, same message modulo
    timestamps/ids/paths/numbers) produce the same signature; a different
    error class or message shape produces a different one.
    """
    text = " ".join(str(error or "no-error-text").split())
    for pattern, placeholder in _VOLATILE_PATTERNS:
        text = pattern.sub(placeholder, text)
    text = " ".join(text.split())
    if len(text) > _MAX_SIGNATURE_CHARS:
        # keep a readable head + a stable hash tail so over-long messages
        # still group deterministically without unbounded storage.
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        text = text[: _MAX_SIGNATURE_CHARS - 13] + " " + digest
    return text
