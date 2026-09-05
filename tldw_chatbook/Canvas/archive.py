"""Pure validation helpers for inert Chatbook Canvas archive records.

This module defines only archive metadata boundaries. It does not read source
entries, compile Canvas documents, or mutate the Canvas repository.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import PurePosixPath
from uuid import UUID

from .limits import (
    MAX_CANVAS_ORIGIN_TURN_ID_BYTES,
    MAX_CANVAS_TITLE_BYTES,
    MAX_CANVASES_PER_CONVERSATION,
    MAX_DURABLE_SOURCE_BYTES_PER_CONVERSATION,
    MAX_DURABLE_SOURCE_BYTES_PER_REVISION,
    MAX_REVISIONS_PER_CANVAS,
    validate_utf8_text,
)

CANVAS_ARCHIVE_EXTENSION_VERSION = "1.0"
CANVAS_ARCHIVE_SUPPORTED_RUNTIME_PROFILE = "canvas-v1"
MAX_CANVAS_ARCHIVE_DOCUMENTS = 1_000
MAX_CANVAS_ARCHIVE_REVISIONS = MAX_CANVAS_ARCHIVE_DOCUMENTS * MAX_REVISIONS_PER_CANVAS
MAX_CANVAS_ARCHIVE_REOPEN_HINTS = 10_000
MAX_CANVAS_ARCHIVE_SOURCE_BYTES = 512 * 1024 * 1024
MAX_CANVAS_ARCHIVE_CONVERSATION_ID_BYTES = 256
MAX_CANVAS_ARCHIVE_MESSAGE_ID_BYTES = 256
MAX_CANVAS_ARCHIVE_RUNTIME_PROFILE_BYTES = 64

_RUNTIME_PROFILE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)+$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ACTOR_KINDS = frozenset({"assistant", "user_rename", "user_import"})


class CanvasArchiveValidationError(ValueError):
    """A bounded Canvas archive-metadata validation failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"invalid Canvas archive: {code}")


def canvas_revision_source_path(canvas_id: str, revision_id: str) -> str:
    """Return the sole canonical, inert source entry path for a revision."""

    safe_canvas_id = validate_archive_uuid(canvas_id, field_name="canvas_id")
    safe_revision_id = validate_archive_uuid(revision_id, field_name="revision_id")
    return f"canvas/{safe_canvas_id}/{safe_revision_id}.html.txt"


def validate_revision_source_path(
    value: str, *, canvas_id: str, revision_id: str
) -> str:
    """Reject every path except the exact canonical inert revision path."""

    validate_inert_source_path_shape(value, revision_id=revision_id)
    expected = canvas_revision_source_path(canvas_id, revision_id)
    if value != expected or PurePosixPath(value).as_posix() != expected:
        raise CanvasArchiveValidationError("invalid_source_path")
    return value


def validate_inert_source_path_shape(value: object, *, revision_id: str) -> str:
    """Validate an inert canonical path before its containing owner is known."""

    if not isinstance(value, str) or len(value) != 89:
        raise CanvasArchiveValidationError("invalid_source_path")
    parts = value.split("/")
    suffix = ".html.txt"
    if len(parts) != 3 or parts[0] != "canvas" or not parts[2].endswith(suffix):
        raise CanvasArchiveValidationError("invalid_source_path")
    validate_archive_uuid(parts[1], field_name="source_canvas_id")
    path_revision_id = parts[2][: -len(suffix)]
    validate_archive_uuid(path_revision_id, field_name="source_revision_id")
    if path_revision_id != revision_id:
        raise CanvasArchiveValidationError("invalid_source_path")
    return value


def validate_archive_uuid(value: object, *, field_name: str) -> str:
    """Require a canonical lowercase UUID suitable for an archive path."""

    if not isinstance(value, str) or len(value) != 36:
        raise CanvasArchiveValidationError(f"invalid_{field_name}")
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError, TypeError):
        raise CanvasArchiveValidationError(f"invalid_{field_name}") from None
    if str(parsed) != value:
        raise CanvasArchiveValidationError(f"invalid_{field_name}")
    return value


def validate_bounded_identifier(
    value: object, *, field_name: str, byte_limit: int = 256
) -> str:
    """Validate an opaque non-empty identifier that never becomes a path."""

    if not isinstance(value, str) or not value or "\x00" in value:
        raise CanvasArchiveValidationError(f"invalid_{field_name}")
    try:
        validate_utf8_text(value, limit=byte_limit, field_name=field_name)
    except ValueError:
        raise CanvasArchiveValidationError(f"invalid_{field_name}") from None
    return value


def validate_runtime_profile(value: object) -> str:
    """Validate profile syntax while retaining well-formed unknown profiles."""

    if not isinstance(value, str):
        raise CanvasArchiveValidationError("invalid_runtime_profile")
    try:
        validate_utf8_text(
            value,
            limit=MAX_CANVAS_ARCHIVE_RUNTIME_PROFILE_BYTES,
            field_name="runtime_profile",
        )
    except ValueError:
        raise CanvasArchiveValidationError("invalid_runtime_profile") from None
    if _RUNTIME_PROFILE.fullmatch(value) is None:
        raise CanvasArchiveValidationError("invalid_runtime_profile")
    return value


def validate_digest(value: object) -> str:
    """Require the canonical lowercase SHA-256 spelling."""

    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise CanvasArchiveValidationError("invalid_content_sha256")
    return value


def validate_non_negative_int(value: object, *, field_name: str, maximum: int) -> int:
    """Validate an archive integer without accepting booleans or coercion."""

    if type(value) is not int or value < 0 or value > maximum:
        raise CanvasArchiveValidationError(f"invalid_{field_name}")
    return value


def validate_positive_int(value: object, *, field_name: str, maximum: int) -> int:
    """Validate a positive bounded archive integer."""

    result = validate_non_negative_int(value, field_name=field_name, maximum=maximum)
    if result == 0:
        raise CanvasArchiveValidationError(f"invalid_{field_name}")
    return result


def validate_timestamp(
    value: object, *, field_name: str, optional: bool = False
) -> str | None:
    """Require a bounded timezone-aware ISO-8601 timestamp."""

    if value is None and optional:
        return None
    if not isinstance(value, str) or not value or len(value) > 64:
        raise CanvasArchiveValidationError(f"invalid_{field_name}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise CanvasArchiveValidationError(f"invalid_{field_name}") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CanvasArchiveValidationError(f"invalid_{field_name}")
    return value


def validate_title(value: object) -> str:
    """Validate a revisioned Canvas title."""

    if not isinstance(value, str) or not value:
        raise CanvasArchiveValidationError("invalid_title")
    try:
        validate_utf8_text(value, limit=MAX_CANVAS_TITLE_BYTES, field_name="title")
    except ValueError:
        raise CanvasArchiveValidationError("invalid_title") from None
    return value


def validate_actor_kind(value: object) -> str:
    """Require one actor kind supported by the persisted Canvas schema."""

    if not isinstance(value, str) or value not in _ACTOR_KINDS:
        raise CanvasArchiveValidationError("invalid_actor_kind")
    return value


def require_exact_fields(
    value: object, *, required: frozenset[str], optional: frozenset[str] = frozenset()
) -> Mapping[str, object]:
    """Require a mapping with exactly the known fields for extension 1.0."""

    if not isinstance(value, Mapping):
        raise CanvasArchiveValidationError("invalid_record")
    keys = set(value)
    if not required.issubset(keys) or not keys.issubset(required | optional):
        raise CanvasArchiveValidationError("invalid_record_fields")
    if not all(isinstance(key, str) for key in keys):
        raise CanvasArchiveValidationError("invalid_record_fields")
    return value


__all__ = [
    "CANVAS_ARCHIVE_EXTENSION_VERSION",
    "CANVAS_ARCHIVE_SUPPORTED_RUNTIME_PROFILE",
    "MAX_CANVASES_PER_CONVERSATION",
    "MAX_CANVAS_ARCHIVE_CONVERSATION_ID_BYTES",
    "MAX_CANVAS_ARCHIVE_DOCUMENTS",
    "MAX_CANVAS_ARCHIVE_MESSAGE_ID_BYTES",
    "MAX_CANVAS_ARCHIVE_REOPEN_HINTS",
    "MAX_CANVAS_ARCHIVE_REVISIONS",
    "MAX_CANVAS_ARCHIVE_SOURCE_BYTES",
    "MAX_CANVAS_ORIGIN_TURN_ID_BYTES",
    "MAX_DURABLE_SOURCE_BYTES_PER_CONVERSATION",
    "MAX_DURABLE_SOURCE_BYTES_PER_REVISION",
    "MAX_REVISIONS_PER_CANVAS",
    "CanvasArchiveValidationError",
    "canvas_revision_source_path",
    "require_exact_fields",
    "validate_actor_kind",
    "validate_archive_uuid",
    "validate_bounded_identifier",
    "validate_digest",
    "validate_inert_source_path_shape",
    "validate_non_negative_int",
    "validate_positive_int",
    "validate_revision_source_path",
    "validate_runtime_profile",
    "validate_timestamp",
    "validate_title",
]
