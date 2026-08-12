"""Create schema version 4 with exact clone-recipe provenance."""

from __future__ import annotations

import sqlite3

from tldw_chatbook.TTS.migrations.v2_to_v3 import (
    REFERENCE_ID_INDEX,
    REFERENCE_TABLE,
)
from tldw_chatbook.TTS.profile_reference_types import (
    MAX_REFERENCE_CANONICAL_BYTES,
    MAX_REFERENCE_DURATION_MS,
    MAX_REFERENCE_SAMPLE_RATE_HZ,
    MAX_REFERENCE_TEXT_CHARACTERS,
    MAX_REFERENCE_TEXT_UTF8_BYTES,
    MIN_REFERENCE_SAMPLE_RATE_HZ,
    REFERENCE_SAMPLE_ENCODING,
)

TARGET_VERSION = 4
_V3_REFERENCE_TABLE = f"_{REFERENCE_TABLE}_v3"

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
    recipe_id TEXT NULL,
    recipe_revision INTEGER NULL,
    CHECK(
        (recipe_id IS NULL AND recipe_revision IS NULL)
        OR (
            typeof(recipe_id) = 'text'
            AND length(CAST(recipe_id AS BLOB)) BETWEEN 1 AND 128
            AND instr(CAST(recipe_id AS BLOB), x'00') = 0
            AND recipe_id GLOB '[a-z0-9]*'
            AND recipe_id NOT GLOB '*[^a-z0-9._-]*'
            AND typeof(recipe_revision) = 'integer'
            AND recipe_revision BETWEEN 1 AND 2147483647
        )
    ),
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
    """Rebuild the reference table and leave historical provenance unknown."""

    connection.execute(f"ALTER TABLE {REFERENCE_TABLE} RENAME TO {_V3_REFERENCE_TABLE}")
    connection.execute(REFERENCE_TABLE_DDL)
    connection.execute(
        f"""
        INSERT INTO {REFERENCE_TABLE} (
            profile_id, reference_id, wav_bytes, reference_text, sha256,
            byte_length, duration_ms, sample_rate_hz, channels, sample_encoding,
            created_at, updated_at, recipe_id, recipe_revision
        )
        SELECT profile_id, reference_id, wav_bytes, reference_text, sha256,
               byte_length, duration_ms, sample_rate_hz, channels, sample_encoding,
               created_at, updated_at, NULL, NULL
        FROM {_V3_REFERENCE_TABLE}
        """
    )
    connection.execute(f"DROP TABLE {_V3_REFERENCE_TABLE}")
    connection.execute(REFERENCE_ID_INDEX_DDL)
    connection.execute("PRAGMA user_version = 4")


__all__ = [
    "REFERENCE_ID_INDEX_DDL",
    "REFERENCE_TABLE_DDL",
    "TARGET_VERSION",
    "migrate",
]
