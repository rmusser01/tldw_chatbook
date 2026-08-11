"""Schema-v3 and metadata-projection tests for private clone references."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

import tldw_chatbook.TTS.profile_schema as profile_schema
from tldw_chatbook.TTS.migrations import v0_to_v1, v1_to_v2, v2_to_v3
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_reference_storage import (
    PROFILE_WITH_REFERENCE_SELECT,
    REFERENCE_ID_INDEX,
    REFERENCE_TABLE,
    decode_reference_summary,
)
from tldw_chatbook.TTS.profile_schema import open_profile_store
from tldw_chatbook.TTS.profile_reference_types import TTSCloneReferenceSummary

PROFILE_ID = "01234567-89ab-cdef-8123-456789abcdef"
REFERENCE_ID = "fedcba98-7654-4321-8123-456789abcdef"
NOW = "2026-08-10T12:34:56.123456Z"


def _safe_error(code: str) -> pytest.RaisesExc[ProfileRepositoryError]:
    return pytest.raises(
        ProfileRepositoryError,
        match=rf"^TTS profile repository failed: {code}$",
    )


def _create_populated_v2(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    v0_to_v1.migrate(connection)
    v1_to_v2.migrate(connection)
    connection.execute(
        """
        INSERT INTO tts_generation_profiles (
            profile_id, display_name, normalized_name, provider_id, model_id,
            voice_id, response_format, speed, options_json, revision,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            PROFILE_ID,
            "Private Voice",
            "private voice",
            "audio_cpp",
            "model-a",
            None,
            "wav",
            1.0,
            "{}",
            4,
            NOW,
            NOW,
        ),
    )
    connection.execute(
        """
        INSERT INTO character_tts_assignments (
            source, authority_id, character_id, profile_id, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("local", "local", "character-a", PROFILE_ID, NOW, NOW),
    )
    connection.commit()
    connection.close()


def _domain_rows(
    path: Path,
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    connection = sqlite3.connect(path)
    try:
        profiles = list(
            connection.execute(
                "SELECT * FROM tts_generation_profiles ORDER BY profile_id"
            )
        )
        assignments = list(
            connection.execute(
                """
                SELECT * FROM character_tts_assignments
                ORDER BY source, authority_id, character_id
                """
            )
        )
        return profiles, assignments
    finally:
        connection.close()


def test_v3_reference_schema_has_exact_owned_shape(tmp_path: Path) -> None:
    connection = open_profile_store(tmp_path / "profiles.sqlite3")
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert [
            tuple(row)
            for row in connection.execute(f"PRAGMA table_xinfo({REFERENCE_TABLE})")
        ] == [
            (0, "profile_id", "TEXT", 0, None, 1, 0),
            (1, "reference_id", "TEXT", 1, None, 0, 0),
            (2, "wav_bytes", "BLOB", 1, None, 0, 0),
            (3, "reference_text", "TEXT", 1, None, 0, 0),
            (4, "sha256", "TEXT", 1, None, 0, 0),
            (5, "byte_length", "INTEGER", 1, None, 0, 0),
            (6, "duration_ms", "INTEGER", 1, None, 0, 0),
            (7, "sample_rate_hz", "INTEGER", 1, None, 0, 0),
            (8, "channels", "INTEGER", 1, None, 0, 0),
            (9, "sample_encoding", "TEXT", 1, None, 0, 0),
            (10, "created_at", "TEXT", 1, None, 0, 0),
            (11, "updated_at", "TEXT", 1, None, 0, 0),
        ]
        indexes = {
            row["name"]: row
            for row in connection.execute(f"PRAGMA index_list({REFERENCE_TABLE})")
        }
        assert set(indexes) == {
            REFERENCE_ID_INDEX,
            f"sqlite_autoindex_{REFERENCE_TABLE}_1",
        }
        assert indexes[REFERENCE_ID_INDEX]["unique"] == 1
        assert [
            (row["name"], row["desc"], row["coll"])
            for row in connection.execute(f"PRAGMA index_xinfo({REFERENCE_ID_INDEX})")
            if row["key"] == 1
        ] == [("reference_id", 0, "BINARY")]
        foreign_keys = list(
            connection.execute(f"PRAGMA foreign_key_list({REFERENCE_TABLE})")
        )
        assert len(foreign_keys) == 1
        assert (
            foreign_keys[0]["table"],
            foreign_keys[0]["from"],
            foreign_keys[0]["to"],
            foreign_keys[0]["on_delete"],
        ) == ("tts_generation_profiles", "profile_id", "profile_id", "CASCADE")
    finally:
        connection.close()


def test_reference_projection_is_metadata_only_and_decodes_summary() -> None:
    lowered = PROFILE_WITH_REFERENCE_SELECT.casefold()
    assert "wav_bytes" not in lowered
    assert "reference_text" not in lowered
    assert "sha256" not in lowered
    row = {
        "reference_reference_id": REFERENCE_ID,
        "reference_byte_length": 4_844,
        "reference_duration_ms": 100,
        "reference_sample_rate_hz": 24_000,
        "reference_channels": 1,
        "reference_sample_encoding": "pcm_s16le",
        "reference_created_at": NOW,
        "reference_updated_at": NOW,
    }

    summary = decode_reference_summary(row)

    assert summary == TTSCloneReferenceSummary(
        reference_id=UUID(REFERENCE_ID),
        byte_length=4_844,
        duration_ms=100,
        sample_rate_hz=24_000,
        channels=1,
        sample_encoding="pcm_s16le",
        created_at=datetime(2026, 8, 10, 12, 34, 56, 123456, tzinfo=UTC),
        updated_at=datetime(2026, 8, 10, 12, 34, 56, 123456, tzinfo=UTC),
    )


def test_reference_projection_decodes_missing_left_join_as_none() -> None:
    row = {
        "reference_reference_id": None,
        "reference_byte_length": None,
        "reference_duration_ms": None,
        "reference_sample_rate_hz": None,
        "reference_channels": None,
        "reference_sample_encoding": None,
        "reference_created_at": None,
        "reference_updated_at": None,
    }

    assert decode_reference_summary(row) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("reference_reference_id", "not-a-uuid"),
        ("reference_byte_length", True),
        ("reference_sample_encoding", "wav"),
        ("reference_updated_at", "PRIVATE invalid time"),
    ],
)
def test_reference_projection_rejects_corrupt_metadata_context_free(
    field: str, value: object
) -> None:
    row: dict[str, object] = {
        "reference_reference_id": REFERENCE_ID,
        "reference_byte_length": 4_844,
        "reference_duration_ms": 100,
        "reference_sample_rate_hz": 24_000,
        "reference_channels": 1,
        "reference_sample_encoding": "pcm_s16le",
        "reference_created_at": NOW,
        "reference_updated_at": NOW,
    }
    row[field] = value

    with _safe_error("corrupt_data") as caught:
        decode_reference_summary(row)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "PRIVATE" not in repr(caught.value)


def test_populated_v2_migration_preserves_domain_and_adds_no_reference(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    _create_populated_v2(path)
    before = _domain_rows(path)

    connection = open_profile_store(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert (
            connection.execute(f"SELECT count(*) FROM {REFERENCE_TABLE}").fetchone()[0]
            == 0
        )
    finally:
        connection.close()

    assert _domain_rows(path) == before


def test_v2_migration_rolls_back_domain_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "profiles.sqlite3"
    _create_populated_v2(path)
    before = _domain_rows(path)
    real_migration = profile_schema.MIGRATIONS[2]

    def mutate_domain(connection: sqlite3.Connection) -> None:
        real_migration(connection)
        connection.execute(
            "UPDATE tts_generation_profiles SET display_name = 'PRIVATE changed'"
        )

    monkeypatch.setitem(profile_schema.MIGRATIONS, 2, mutate_domain)

    with _safe_error("migration_failed"):
        open_profile_store(path)

    raw = sqlite3.connect(path)
    try:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == 2
        assert (
            raw.execute(
                "SELECT count(*) FROM sqlite_schema WHERE name = ?", (REFERENCE_TABLE,)
            ).fetchone()[0]
            == 0
        )
    finally:
        raw.close()
    assert _domain_rows(path) == before


def test_v2_migration_runs_full_integrity_inside_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "profiles.sqlite3"
    _create_populated_v2(path)
    real_validate = profile_schema._validate_full_integrity
    observations: list[bool] = []

    def tracked(connection: sqlite3.Connection) -> None:
        observations.append(connection.in_transaction)
        real_validate(connection)

    monkeypatch.setattr(profile_schema, "_validate_full_integrity", tracked)

    connection = open_profile_store(path)
    connection.close()

    assert observations == [True]


def test_v2_migration_rolls_back_post_migration_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "profiles.sqlite3"
    _create_populated_v2(path)

    def fail_validation(_connection: sqlite3.Connection) -> None:
        raise RuntimeError("PRIVATE validation detail")

    monkeypatch.setattr(profile_schema, "_validate_full_integrity", fail_validation)

    with _safe_error("migration_failed") as caught:
        open_profile_store(path)

    raw = sqlite3.connect(path)
    try:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == 2
        assert (
            raw.execute(
                "SELECT count(*) FROM sqlite_schema WHERE name = ?", (REFERENCE_TABLE,)
            ).fetchone()[0]
            == 0
        )
    finally:
        raw.close()
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_schema_v3_migration_registration_is_exact() -> None:
    assert v2_to_v3.TARGET_VERSION == profile_schema.CURRENT_PROFILE_SCHEMA_VERSION == 3
    assert set(profile_schema.MIGRATIONS) == {0, 1, 2}
    assert profile_schema.MIGRATIONS[2] is v2_to_v3.migrate
