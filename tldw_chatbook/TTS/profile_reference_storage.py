"""Metadata projections and bounded BLOB I/O for private clone references."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping
from typing import Literal, TypeAlias, cast

from tldw_chatbook.TTS.migrations.v2_to_v3 import (
    REFERENCE_ID_INDEX,
    REFERENCE_TABLE,
)
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_reference_audio import validate_canonical_reference_wav
from tldw_chatbook.TTS.profile_reference_types import (
    MAX_REFERENCE_COUNT,
    MAX_REFERENCE_TOTAL_BYTES,
    TTSCloneReference,
    TTSCloneRecipeRequirement,
    TTSCloneReferenceSummary,
)
from tldw_chatbook.TTS.profile_schema import decode_utc_datetime, decode_uuid

RowLike: TypeAlias = sqlite3.Row | Mapping[str, object]
REFERENCE_BLOB_CHUNK_BYTES = 256 * 1024
REFERENCE_VALIDATION_PROGRESS_OPCODE_INTERVAL = 1_000

REFERENCE_SUMMARY_ALIASES = (
    "reference_reference_id",
    "reference_byte_length",
    "reference_duration_ms",
    "reference_sample_rate_hz",
    "reference_channels",
    "reference_sample_encoding",
    "reference_created_at",
    "reference_updated_at",
    "reference_recipe_id",
    "reference_recipe_revision",
)

PROFILE_WITH_REFERENCE_SELECT = f"""
SELECT
    p.profile_id,
    p.display_name,
    p.normalized_name,
    p.provider_id,
    p.model_id,
    p.voice_id,
    p.response_format,
    p.speed,
    p.options_json,
    p.revision,
    p.created_at,
    p.updated_at,
    r.reference_id AS reference_reference_id,
    r.byte_length AS reference_byte_length,
    r.duration_ms AS reference_duration_ms,
    r.sample_rate_hz AS reference_sample_rate_hz,
    r.channels AS reference_channels,
    r.sample_encoding AS reference_sample_encoding,
    r.created_at AS reference_created_at,
    r.updated_at AS reference_updated_at,
    r.recipe_id AS reference_recipe_id,
    r.recipe_revision AS reference_recipe_revision,
    p.model_id AS reference_model_id
FROM tts_generation_profiles AS p
LEFT JOIN {REFERENCE_TABLE} AS r ON r.profile_id = p.profile_id
"""

ASSIGNED_PROFILE_WITH_REFERENCE_JOIN_SELECT = f"""
SELECT
    a.source AS assignment_source,
    a.authority_id AS assignment_authority_id,
    a.character_id AS assignment_character_id,
    a.profile_id AS assignment_profile_id,
    a.created_at AS assignment_created_at,
    a.updated_at AS assignment_updated_at,
    p.profile_id AS profile_profile_id,
    p.display_name AS profile_display_name,
    p.normalized_name AS profile_normalized_name,
    p.provider_id AS profile_provider_id,
    p.model_id AS profile_model_id,
    p.voice_id AS profile_voice_id,
    p.response_format AS profile_response_format,
    p.speed AS profile_speed,
    p.options_json AS profile_options_json,
    p.revision AS profile_revision,
    p.created_at AS profile_created_at,
    p.updated_at AS profile_updated_at,
    r.reference_id AS reference_reference_id,
    r.byte_length AS reference_byte_length,
    r.duration_ms AS reference_duration_ms,
    r.sample_rate_hz AS reference_sample_rate_hz,
    r.channels AS reference_channels,
    r.sample_encoding AS reference_sample_encoding,
    r.created_at AS reference_created_at,
    r.updated_at AS reference_updated_at,
    r.recipe_id AS reference_recipe_id,
    r.recipe_revision AS reference_recipe_revision,
    p.model_id AS reference_model_id
FROM character_tts_assignments AS a
LEFT JOIN tts_generation_profiles AS p ON p.profile_id = a.profile_id
LEFT JOIN {REFERENCE_TABLE} AS r ON r.profile_id = p.profile_id
"""

REFERENCE_PAYLOAD_SELECT = f"""
SELECT
    rowid AS reference_rowid,
    reference_id AS reference_reference_id,
    reference_text,
    sha256,
    byte_length AS reference_byte_length,
    duration_ms AS reference_duration_ms,
    sample_rate_hz AS reference_sample_rate_hz,
    channels AS reference_channels,
    sample_encoding AS reference_sample_encoding,
    created_at AS reference_created_at,
    updated_at AS reference_updated_at,
    recipe_id AS reference_recipe_id,
    recipe_revision AS reference_recipe_revision,
    (SELECT p.model_id FROM tts_generation_profiles AS p
     WHERE p.profile_id = {REFERENCE_TABLE}.profile_id) AS reference_model_id
FROM {REFERENCE_TABLE}
"""


def _repository_error() -> ProfileRepositoryError:
    return ProfileRepositoryError("corrupt_data")


def decode_reference_summary(row: RowLike) -> TTSCloneReferenceSummary | None:
    """Decode one optional LEFT JOIN summary without reading private payloads."""

    summary: TTSCloneReferenceSummary | None = None
    failed = False
    try:
        values = [row[column] for column in REFERENCE_SUMMARY_ALIASES]
        persisted_summary = values[:8]
        if all(value is None for value in persisted_summary):
            if any(value is not None for value in values[8:]):
                raise ValueError
            return None
        if any(value is None for value in persisted_summary):
            raise ValueError
        byte_length, duration_ms, sample_rate_hz, channels = values[1:5]
        sample_encoding = values[5]
        if (
            type(byte_length) is not int
            or type(duration_ms) is not int
            or type(sample_rate_hz) is not int
            or type(channels) is not int
            or type(sample_encoding) is not str
        ):
            raise ValueError
        recipe_id, recipe_revision = values[8:10]
        if (recipe_id is None) != (recipe_revision is None):
            raise ValueError
        requirement = None
        if recipe_id is not None:
            requirement = TTSCloneRecipeRequirement(
                recipe_id=cast(str, recipe_id),
                recipe_revision=cast(int, recipe_revision),
                model_id=cast(str, row["reference_model_id"]),
            )
        summary = TTSCloneReferenceSummary(
            reference_id=decode_uuid(values[0]),
            byte_length=byte_length,
            duration_ms=duration_ms,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            sample_encoding=cast(Literal["pcm_s16le"], sample_encoding),
            created_at=decode_utc_datetime(values[6]),
            updated_at=decode_utc_datetime(values[7]),
            recipe_requirement=requirement,
        )
    except Exception:
        failed = True
    if failed or summary is None:
        raise _repository_error() from None
    return summary


def decode_reference_payload(row: RowLike, wav_bytes: bytes) -> TTSCloneReference:
    """Decode one exact reference after its BLOB has been streamed separately."""

    try:
        summary = decode_reference_summary(row)
        reference_text = row["reference_text"]
        digest = row["sha256"]
        if (
            summary is None
            or type(reference_text) is not str
            or type(digest) is not str
        ):
            raise ValueError
        return TTSCloneReference(
            summary=summary,
            reference_text=reference_text,
            sha256=digest,
            wav_bytes=wav_bytes,
            recipe_requirement=summary.recipe_requirement,
        )
    except Exception:
        raise ProfileRepositoryError("reference_unavailable") from None


def _finish_blob_operation(
    body_error: BaseException | None,
    close_error: BaseException | None,
    *,
    error_code: str,
    database_error_code: str | None = None,
) -> None:
    for error in (body_error, close_error):
        if error is not None and not isinstance(error, Exception):
            raise error
    if database_error_code is not None and any(
        isinstance(error, sqlite3.DatabaseError) for error in (body_error, close_error)
    ):
        raise ProfileRepositoryError(database_error_code) from None
    if body_error is not None or close_error is not None:
        raise ProfileRepositoryError(error_code) from None


def write_reference_blob(
    connection: sqlite3.Connection,
    rowid: int,
    payload: bytes,
) -> None:
    """Write an allocated reference BLOB in bounded chunks and close it.

    Args:
        connection: Caller-owned SQLite connection containing the allocated row.
        rowid: Exact positive row identifier of the allocated reference BLOB.
        payload: Canonical bounded WAV bytes to write.

    Raises:
        ProfileRepositoryError: If allocation size differs, BLOB access fails,
            a write is incomplete, or close fails.
        BaseException: A caller control-flow signal raised by SQLite or cleanup.
    """

    blob: sqlite3.Blob | None = None
    body_error: BaseException | None = None
    close_error: BaseException | None = None
    try:
        blob = connection.blobopen(REFERENCE_TABLE, "wav_bytes", rowid)
        if len(blob) != len(payload):
            raise ValueError
        for offset in range(0, len(payload), REFERENCE_BLOB_CHUNK_BYTES):
            blob.write(payload[offset : offset + REFERENCE_BLOB_CHUNK_BYTES])
        if blob.tell() != len(payload):
            raise ValueError
    except BaseException as error:
        body_error = error
    if blob is not None:
        try:
            blob.close()
        except BaseException as error:
            close_error = error
    _finish_blob_operation(body_error, close_error, error_code="operation_failed")


def read_reference_blob(
    connection: sqlite3.Connection,
    rowid: int,
    byte_length: int,
    *,
    progress_guard: Callable[[], None] | None = None,
) -> bytes:
    """Read one exact reference BLOB in bounded chunks and close it."""

    blob: sqlite3.Blob | None = None
    body_error: BaseException | None = None
    close_error: BaseException | None = None
    payload: bytes | None = None
    try:
        if progress_guard is not None:
            progress_guard()
        blob = connection.blobopen(
            REFERENCE_TABLE,
            "wav_bytes",
            rowid,
            readonly=True,
        )
        if len(blob) != byte_length:
            raise ValueError
        parts: list[bytes] = []
        remaining = byte_length
        while remaining:
            if progress_guard is not None:
                progress_guard()
            chunk = blob.read(min(REFERENCE_BLOB_CHUNK_BYTES, remaining))
            if type(chunk) is not bytes or not chunk:
                raise ValueError
            parts.append(chunk)
            remaining -= len(chunk)
        if blob.read(1) != b"":
            raise ValueError
        if progress_guard is not None:
            progress_guard()
        payload = b"".join(parts)
    except BaseException as error:
        body_error = error
    if blob is not None:
        try:
            blob.close()
        except BaseException as error:
            close_error = error
    _finish_blob_operation(
        body_error,
        close_error,
        error_code="reference_unavailable",
        database_error_code="schema_corrupt",
    )
    if payload is None:
        raise ProfileRepositoryError("reference_unavailable")
    return payload


def validate_reference_rows(
    connection: sqlite3.Connection,
    *,
    check_deadline: Callable[[], None] | None = None,
) -> None:
    """Fully qualify every reference payload, metadata row, and total quota."""

    callback_error: BaseException | None = None
    body_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    progress_installed = False

    def interrupt_after_deadline() -> int:
        nonlocal callback_error
        assert check_deadline is not None
        try:
            check_deadline()
        except BaseException as error:
            callback_error = error
            return 1
        return 0

    try:
        if check_deadline is not None:
            check_deadline()
            connection.set_progress_handler(
                interrupt_after_deadline,
                REFERENCE_VALIDATION_PROGRESS_OPCODE_INTERVAL,
            )
            progress_installed = True
        quota = connection.execute(
            f"SELECT COUNT(*), COALESCE(SUM(byte_length), 0) FROM {REFERENCE_TABLE}"
        ).fetchone()
        if (
            quota is None
            or len(quota) != 2
            or type(quota[0]) is not int
            or type(quota[1]) is not int
            or not 0 <= quota[0] <= MAX_REFERENCE_COUNT
            or not 0 <= quota[1] <= MAX_REFERENCE_TOTAL_BYTES
        ):
            raise ValueError
        seen = 0
        for row in connection.execute(
            f"{REFERENCE_PAYLOAD_SELECT} ORDER BY profile_id"
        ):
            if check_deadline is not None:
                check_deadline()
            rowid = row["reference_rowid"]
            byte_length = row["reference_byte_length"]
            if (
                type(rowid) is not int
                or rowid <= 0
                or type(byte_length) is not int
                or byte_length <= 0
            ):
                raise ValueError
            payload = read_reference_blob(
                connection,
                rowid,
                byte_length,
                progress_guard=check_deadline,
            )
            reference = decode_reference_payload(row, payload)
            metadata = validate_canonical_reference_wav(payload)
            if (
                metadata.byte_length != reference.summary.byte_length
                or metadata.duration_ms != reference.summary.duration_ms
                or metadata.sample_rate_hz != reference.summary.sample_rate_hz
                or metadata.channels != reference.summary.channels
                or metadata.sample_encoding != reference.summary.sample_encoding
            ):
                raise ValueError
            seen += 1
        if seen != quota[0]:
            raise ValueError
        if check_deadline is not None:
            check_deadline()
    except BaseException as error:
        body_error = error
    if progress_installed:
        try:
            connection.set_progress_handler(None, 0)
        except BaseException as error:
            cleanup_error = error
    if callback_error is not None:
        body_error = callback_error
    for candidate_error in (body_error, cleanup_error):
        if candidate_error is not None and not isinstance(candidate_error, Exception):
            raise candidate_error
    if cleanup_error is not None:
        raise ProfileRepositoryError("reference_unavailable") from None
    if isinstance(body_error, ProfileRepositoryError):
        raise ProfileRepositoryError(body_error.code) from None
    if body_error is not None:
        raise ProfileRepositoryError("reference_unavailable") from None


__all__ = [
    "PROFILE_WITH_REFERENCE_SELECT",
    "ASSIGNED_PROFILE_WITH_REFERENCE_JOIN_SELECT",
    "REFERENCE_BLOB_CHUNK_BYTES",
    "REFERENCE_VALIDATION_PROGRESS_OPCODE_INTERVAL",
    "REFERENCE_ID_INDEX",
    "REFERENCE_PAYLOAD_SELECT",
    "REFERENCE_SUMMARY_ALIASES",
    "REFERENCE_TABLE",
    "decode_reference_payload",
    "decode_reference_summary",
    "read_reference_blob",
    "validate_reference_rows",
    "write_reference_blob",
]
