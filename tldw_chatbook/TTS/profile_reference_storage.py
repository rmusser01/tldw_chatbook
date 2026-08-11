"""Metadata-only persistence projections for private clone references."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from typing import Literal, TypeAlias, cast

from tldw_chatbook.TTS.migrations.v2_to_v3 import (
    REFERENCE_ID_INDEX,
    REFERENCE_TABLE,
)
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_reference_types import TTSCloneReferenceSummary
from tldw_chatbook.TTS.profile_schema import decode_utc_datetime, decode_uuid

RowLike: TypeAlias = sqlite3.Row | Mapping[str, object]

REFERENCE_SUMMARY_ALIASES = (
    "reference_reference_id",
    "reference_byte_length",
    "reference_duration_ms",
    "reference_sample_rate_hz",
    "reference_channels",
    "reference_sample_encoding",
    "reference_created_at",
    "reference_updated_at",
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
    r.updated_at AS reference_updated_at
FROM tts_generation_profiles AS p
LEFT JOIN {REFERENCE_TABLE} AS r ON r.profile_id = p.profile_id
"""


def _repository_error() -> ProfileRepositoryError:
    return ProfileRepositoryError("corrupt_data")


def decode_reference_summary(row: RowLike) -> TTSCloneReferenceSummary | None:
    """Decode one optional LEFT JOIN summary without reading private payloads."""

    summary: TTSCloneReferenceSummary | None = None
    failed = False
    try:
        values = [row[column] for column in REFERENCE_SUMMARY_ALIASES]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
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
        summary = TTSCloneReferenceSummary(
            reference_id=decode_uuid(values[0]),
            byte_length=byte_length,
            duration_ms=duration_ms,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            sample_encoding=cast(Literal["pcm_s16le"], sample_encoding),
            created_at=decode_utc_datetime(values[6]),
            updated_at=decode_utc_datetime(values[7]),
        )
    except Exception:
        failed = True
    if failed or summary is None:
        raise _repository_error() from None
    return summary


__all__ = [
    "PROFILE_WITH_REFERENCE_SELECT",
    "REFERENCE_ID_INDEX",
    "REFERENCE_SUMMARY_ALIASES",
    "REFERENCE_TABLE",
    "decode_reference_summary",
]
