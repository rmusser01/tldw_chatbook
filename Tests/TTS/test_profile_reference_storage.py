"""Schema-v4 and metadata-projection tests for private clone references."""

from __future__ import annotations

import hashlib
import io
import sqlite3
import wave
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

import tldw_chatbook.TTS.profile_schema as profile_schema
from tldw_chatbook.TTS.migrations import v0_to_v1, v1_to_v2, v2_to_v3
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_migration_candidate import (
    ProfileMigrationBoundary,
    ProfileMigrationBoundaryRequest,
    ProfileMigrationBoundarySnapshot,
    step_profile_migration_candidate,
)
from tldw_chatbook.TTS.profile_reference_storage import (
    PROFILE_WITH_REFERENCE_SELECT,
    REFERENCE_ID_INDEX,
    REFERENCE_PAYLOAD_SELECT,
    REFERENCE_TABLE,
    decode_reference_payload,
    decode_reference_summary,
)
from tldw_chatbook.TTS.profile_schema import open_profile_store
from tldw_chatbook.TTS.profile_reference_types import (
    TTSCloneRecipeRequirement,
    TTSCloneReferenceSummary,
)

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


def _create_populated_v3_reference(path: Path) -> tuple[object, ...]:
    _create_populated_v2(path)
    frames = b"\x00\x00" * 2_400
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24_000)
        writer.writeframes(frames)
    wav_bytes = output.getvalue()
    connection = sqlite3.connect(path)
    v2_to_v3.migrate(connection)
    row = (
        PROFILE_ID,
        REFERENCE_ID,
        wav_bytes,
        "Private transcript",
        hashlib.sha256(wav_bytes).hexdigest(),
        len(wav_bytes),
        100,
        24_000,
        1,
        "pcm_s16le",
        NOW,
        NOW,
    )
    connection.execute(
        f"INSERT INTO {REFERENCE_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        row,
    )
    connection.commit()
    connection.close()
    return row


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


def test_v3_candidate_boundary_and_final_preserve_exact_private_reference(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidate.sqlite3"
    reference_before = _create_populated_v3_reference(path)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    boundary_rows: list[tuple[object, ...]] = []

    result = step_profile_migration_candidate(
        connection,
        boundary_sink=lambda snapshot, request: boundary_rows.append(
            _candidate_reference_row(snapshot, request)
        ),
    )

    assert tuple(request.kind for request in result.boundaries) == (
        ProfileMigrationBoundary.PRE_V4,
    )
    assert boundary_rows == [reference_before]
    final = sqlite3.connect(path)
    try:
        final_row = final.execute(
            f"SELECT * FROM {REFERENCE_TABLE} WHERE profile_id = ?",
            (PROFILE_ID,),
        ).fetchone()
        assert tuple(final_row[:12]) == reference_before
        assert final_row[12:] == (None, None)
    finally:
        final.close()


def _candidate_reference_row(
    snapshot: ProfileMigrationBoundarySnapshot,
    request: ProfileMigrationBoundaryRequest,
) -> tuple[object, ...]:
    assert request == ProfileMigrationBoundaryRequest(
        ProfileMigrationBoundary.PRE_V4,
        3,
    )
    destination = sqlite3.connect(":memory:")
    try:
        snapshot.backup_to(destination)
        return tuple(
            destination.execute(
                f"SELECT * FROM {REFERENCE_TABLE} WHERE profile_id = ?",
                (PROFILE_ID,),
            ).fetchone()
        )
    finally:
        destination.close()


def test_v3_candidate_rejects_corrupt_reference_before_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "PRIVATE-candidate.sqlite3"
    _create_populated_v3_reference(path)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA ignore_check_constraints = ON")
    connection.execute(f"UPDATE {REFERENCE_TABLE} SET sha256 = ?", ("0" * 64,))
    connection.commit()
    calls: list[ProfileMigrationBoundary] = []

    with _safe_error("reference_unavailable") as caught:
        step_profile_migration_candidate(
            connection,
            boundary_sink=lambda _borrowed, request: calls.append(request.kind),
        )

    assert calls == []
    assert str(path) not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_candidate_step_rejects_private_reference_mutation_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "candidate.sqlite3"
    before = _create_populated_v3_reference(path)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    real_migration = profile_schema.MIGRATIONS[3]

    def mutate_reference(candidate: sqlite3.Connection) -> None:
        real_migration(candidate)
        candidate.execute(
            f"UPDATE {REFERENCE_TABLE} SET reference_text = 'Changed transcript'"
        )

    monkeypatch.setitem(profile_schema.MIGRATIONS, 3, mutate_reference)

    with _safe_error("migration_failed"):
        step_profile_migration_candidate(connection)

    reopened = sqlite3.connect(path)
    try:
        assert reopened.execute("PRAGMA user_version").fetchone()[0] == 3
        assert (
            tuple(reopened.execute(f"SELECT * FROM {REFERENCE_TABLE}").fetchone())
            == before
        )
    finally:
        reopened.close()


def test_v4_reference_schema_has_exact_owned_shape(tmp_path: Path) -> None:
    connection = open_profile_store(tmp_path / "profiles.sqlite3")
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
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
            (12, "recipe_id", "TEXT", 0, None, 0, 0),
            (13, "recipe_revision", "INTEGER", 0, None, 0, 0),
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
    assert "recipe_id" in lowered
    assert "recipe_revision" in lowered
    row = {
        "reference_reference_id": REFERENCE_ID,
        "reference_byte_length": 4_844,
        "reference_duration_ms": 100,
        "reference_sample_rate_hz": 24_000,
        "reference_channels": 1,
        "reference_sample_encoding": "pcm_s16le",
        "reference_created_at": NOW,
        "reference_updated_at": NOW,
        "reference_recipe_id": "voice.recipe-1",
        "reference_recipe_revision": 7,
        "reference_model_id": "model-a",
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
        recipe_requirement=TTSCloneRecipeRequirement(
            recipe_id="voice.recipe-1",
            recipe_revision=7,
            model_id="model-a",
        ),
    )


def test_v4_reference_columns_are_nullable_and_pair_constrained(tmp_path: Path) -> None:
    path = tmp_path / "profiles.sqlite3"
    _create_populated_v3_reference(path)
    connection = open_profile_store(path)
    try:
        columns = {
            row["name"]: row
            for row in connection.execute(f"PRAGMA table_xinfo({REFERENCE_TABLE})")
        }
        assert columns["recipe_id"]["notnull"] == 0
        assert columns["recipe_revision"]["notnull"] == 0
        connection.execute(
            f"UPDATE {REFERENCE_TABLE} SET recipe_id = ?, recipe_revision = ?",
            ("voice.recipe-1", 2_147_483_647),
        )
        for recipe_id, recipe_revision in (
            (None, 1),
            ("voice.recipe-1", None),
            ("-leading", 1),
            ("UPPER", 1),
            ("voice.recipe-1", 0),
            ("voice.recipe-1", 2_147_483_648),
        ):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    f"UPDATE {REFERENCE_TABLE} SET recipe_id = ?, recipe_revision = ?",
                    (recipe_id, recipe_revision),
                )
    finally:
        connection.close()


@pytest.mark.parametrize(
    "recipe_id",
    ["a\x00UPPER", "a\x00", "a\x00._-", f"{'a' * 128}\x00hidden"],
)
def test_v4_sql_recipe_grammar_rejects_embedded_nul(
    tmp_path: Path,
    recipe_id: str,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    before = _create_populated_v3_reference(path)
    connection = open_profile_store(path)
    try:
        connection.execute(f"DELETE FROM {REFERENCE_TABLE}")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                f"INSERT INTO {REFERENCE_TABLE} VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*before, recipe_id, 1),
            )
    finally:
        connection.close()


def test_v3_to_v4_preserves_reference_and_leaves_recipe_unknown(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    before = _create_populated_v3_reference(path)

    migrated = open_profile_store(path)
    try:
        row = migrated.execute(
            f"SELECT * FROM {REFERENCE_TABLE} WHERE profile_id = ?", (PROFILE_ID,)
        ).fetchone()
        assert tuple(row[:12]) == before
        assert row["recipe_id"] is None
        assert row["recipe_revision"] is None
    finally:
        migrated.close()


def test_v4_reference_payload_decodes_exact_recipe_requirement(tmp_path: Path) -> None:
    path = tmp_path / "profiles.sqlite3"
    before = _create_populated_v3_reference(path)
    connection = open_profile_store(path)
    requirement = TTSCloneRecipeRequirement(
        recipe_id="voice.recipe-1",
        recipe_revision=7,
        model_id="model-a",
    )
    try:
        connection.execute(
            f"UPDATE {REFERENCE_TABLE} SET recipe_id = ?, recipe_revision = ?",
            (requirement.recipe_id, requirement.recipe_revision),
        )
        row = connection.execute(REFERENCE_PAYLOAD_SELECT).fetchone()
        reference = decode_reference_payload(row, before[2])  # type: ignore[arg-type]
    finally:
        connection.close()

    assert reference.recipe_requirement == requirement
    assert reference.summary.recipe_requirement == requirement


def test_v3_to_v4_validation_failure_rolls_back_candidate_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    _create_populated_v3_reference(path)
    real_validate = profile_schema._validate_migration_reference_rows

    def fail_post_migration(
        connection: sqlite3.Connection,
        *,
        schema_version: int,
    ) -> None:
        real_validate(connection, schema_version=schema_version)
        if schema_version == 4:
            raise RuntimeError("PRIVATE post-migration detail")

    monkeypatch.setattr(
        profile_schema,
        "_validate_migration_reference_rows",
        fail_post_migration,
    )

    with _safe_error("migration_failed") as caught:
        open_profile_store(path)

    candidate = sqlite3.connect(path)
    try:
        assert candidate.execute("PRAGMA user_version").fetchone()[0] == 3
        assert [
            row[1]
            for row in candidate.execute(f"PRAGMA table_xinfo({REFERENCE_TABLE})")
        ][-2:] == ["created_at", "updated_at"]
        assert (
            candidate.execute(f"SELECT COUNT(*) FROM {REFERENCE_TABLE}").fetchone()[0]
            == 1
        )
    finally:
        candidate.close()
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


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
        "reference_recipe_id": None,
        "reference_recipe_revision": None,
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
        "reference_recipe_id": None,
        "reference_recipe_revision": None,
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
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
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


def test_v2_migration_rejects_invalid_existing_domain_before_commit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    _create_populated_v2(path)
    raw = sqlite3.connect(path)
    raw.execute("UPDATE tts_generation_profiles SET revision = 0")
    raw.commit()
    raw.close()

    with _safe_error("migration_failed") as caught:
        open_profile_store(path)

    check = sqlite3.connect(path)
    try:
        assert check.execute("PRAGMA user_version").fetchone()[0] == 2
        assert (
            check.execute(
                "SELECT count(*) FROM sqlite_schema WHERE name = ?", (REFERENCE_TABLE,)
            ).fetchone()[0]
            == 0
        )
        assert (
            check.execute("SELECT revision FROM tts_generation_profiles").fetchone()[0]
            == 0
        )
    finally:
        check.close()
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


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


def test_schema_v4_migration_registration_is_exact() -> None:
    from tldw_chatbook.TTS.migrations import v3_to_v4

    assert v2_to_v3.TARGET_VERSION == 3
    assert v3_to_v4.TARGET_VERSION == profile_schema.CURRENT_PROFILE_SCHEMA_VERSION == 4
    assert set(profile_schema.MIGRATIONS) == {0, 1, 2, 3}
    assert profile_schema.MIGRATIONS[2] is v2_to_v3.migrate
    assert profile_schema.MIGRATIONS[3] is v3_to_v4.migrate
