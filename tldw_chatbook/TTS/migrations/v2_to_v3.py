"""Create schema version 3 for private profile-owned clone references."""

from __future__ import annotations

import sqlite3

from tldw_chatbook.TTS.profile_reference_types import (
    MAX_REFERENCE_CANONICAL_BYTES,
    MAX_REFERENCE_DURATION_MS,
    MAX_REFERENCE_SAMPLE_RATE_HZ,
    MAX_REFERENCE_TEXT_CHARACTERS,
    MAX_REFERENCE_TEXT_UTF8_BYTES,
    MIN_REFERENCE_SAMPLE_RATE_HZ,
    REFERENCE_SAMPLE_ENCODING,
)

TARGET_VERSION = 3
REFERENCE_TABLE = "tts_profile_clone_references"
REFERENCE_ID_INDEX = "idx_tts_profile_clone_references_reference_id"

REFERENCE_TABLE_DDL = f"""
CREATE TABLE {REFERENCE_TABLE} (
    profile_id TEXT PRIMARY KEY,
    reference_id TEXT NOT NULL,
    wav_bytes BLOB NOT NULL
        CHECK(typeof(wav_bytes) = 'blob'
            AND length(wav_bytes) BETWEEN 1 AND {MAX_REFERENCE_CANONICAL_BYTES}),
    reference_text TEXT NOT NULL
        CHECK(typeof(reference_text) = 'text'
            AND length(reference_text) BETWEEN 1 AND {MAX_REFERENCE_TEXT_CHARACTERS}
            AND length(CAST(reference_text AS BLOB)) <= {MAX_REFERENCE_TEXT_UTF8_BYTES}),
    sha256 TEXT NOT NULL
        CHECK(typeof(sha256) = 'text'
            AND length(sha256) = 64
            AND sha256 = lower(sha256)
            AND sha256 NOT GLOB '*[^0-9a-f]*'),
    byte_length INTEGER NOT NULL
        CHECK(typeof(byte_length) = 'integer'
            AND byte_length BETWEEN 1 AND {MAX_REFERENCE_CANONICAL_BYTES}
            AND byte_length = length(wav_bytes)),
    duration_ms INTEGER NOT NULL
        CHECK(typeof(duration_ms) = 'integer'
            AND duration_ms BETWEEN 1 AND {MAX_REFERENCE_DURATION_MS}),
    sample_rate_hz INTEGER NOT NULL
        CHECK(typeof(sample_rate_hz) = 'integer'
            AND sample_rate_hz BETWEEN {MIN_REFERENCE_SAMPLE_RATE_HZ}
                AND {MAX_REFERENCE_SAMPLE_RATE_HZ}),
    channels INTEGER NOT NULL
        CHECK(typeof(channels) = 'integer' AND channels IN (1, 2)),
    sample_encoding TEXT NOT NULL
        CHECK(typeof(sample_encoding) = 'text'
            AND sample_encoding = '{REFERENCE_SAMPLE_ENCODING}'),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(profile_id)
        REFERENCES tts_generation_profiles(profile_id)
        ON DELETE CASCADE
)
"""
REFERENCE_ID_INDEX_DDL = f"""
CREATE UNIQUE INDEX {REFERENCE_ID_INDEX}
ON {REFERENCE_TABLE}(reference_id)
"""


def migrate(connection: sqlite3.Connection) -> None:
    """Add the one-to-one reference table and advance the version fence."""

    connection.execute(REFERENCE_TABLE_DDL)
    connection.execute(REFERENCE_ID_INDEX_DDL)
    connection.execute("PRAGMA user_version = 3")


__all__ = [
    "REFERENCE_ID_INDEX",
    "REFERENCE_ID_INDEX_DDL",
    "REFERENCE_TABLE",
    "REFERENCE_TABLE_DDL",
    "TARGET_VERSION",
    "migrate",
]
