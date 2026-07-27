"""Create schema version 1 for the TTS generation-profile store."""

from __future__ import annotations

import sqlite3

TARGET_VERSION = 1

PROFILE_TABLE_DDL = """
CREATE TABLE tts_generation_profiles (
    profile_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE,
    provider_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    voice_id TEXT NULL,
    response_format TEXT NOT NULL,
    speed REAL NOT NULL,
    options_json TEXT NOT NULL,
    revision INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""
ASSIGNMENT_TABLE_DDL = """
CREATE TABLE character_tts_assignments (
    source TEXT NOT NULL,
    authority_id TEXT NOT NULL,
    character_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(source, authority_id, character_id),
    FOREIGN KEY(profile_id)
        REFERENCES tts_generation_profiles(profile_id)
        ON DELETE RESTRICT
)
"""
ASSIGNMENT_PROFILE_INDEX_DDL = """
CREATE INDEX idx_character_tts_assignments_profile_id
ON character_tts_assignments(profile_id)
"""


def migrate(connection: sqlite3.Connection) -> None:
    """Apply the version-zero to version-one migration transactionally."""

    connection.execute(PROFILE_TABLE_DDL)
    connection.execute(ASSIGNMENT_TABLE_DDL)
    connection.execute(ASSIGNMENT_PROFILE_INDEX_DDL)
    connection.execute("PRAGMA user_version = 1")
