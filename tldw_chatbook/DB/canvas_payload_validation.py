"""Pure Canvas SQLite payload checks shared by runtime and offline recovery."""

import hashlib
import sqlite3

CANVAS_REVISION_PAYLOAD_VALIDATION_FUNCTION = "canvas_revision_payload_valid"


def canvas_revision_payload_valid(
    source_bytes: object,
    content_sha256: object,
    declared_bytes: object,
) -> int:
    """Validate one Canvas payload without exposing its source bytes."""

    if (
        type(source_bytes) is not bytes
        or type(content_sha256) is not str
        or type(declared_bytes) is not int
    ):
        return 0
    try:
        source_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return 0
    return int(
        declared_bytes == len(source_bytes)
        and content_sha256 == hashlib.sha256(source_bytes).hexdigest()
    )


def install_canvas_revision_payload_validator(
    connection: sqlite3.Connection,
) -> None:
    """Install the pure Canvas payload validator required by the schema."""

    connection.create_function(
        CANVAS_REVISION_PAYLOAD_VALIDATION_FUNCTION,
        3,
        canvas_revision_payload_valid,
        deterministic=True,
    )
