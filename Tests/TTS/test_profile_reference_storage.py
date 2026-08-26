"""Schema-v4 and metadata-projection tests for private clone references."""

from __future__ import annotations

import gc
import hashlib
import io
import os
import signal
import sqlite3
import subprocess
import sys
import time
import tracemalloc
import wave
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

import tldw_chatbook.DB.private_sqlite as private_sqlite
import tldw_chatbook.TTS.profile_migration_candidate as migration_candidate
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
            _candidate_reference_row(snapshot, request, tmp_path / "pre-v4.sqlite3")
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
    destination_path: Path,
) -> tuple[object, ...]:
    assert request == ProfileMigrationBoundaryRequest(
        ProfileMigrationBoundary.PRE_V4,
        3,
    )
    opaque = private_sqlite.open_profile_migration_boundary_destination(
        destination_path,
        schema_version=request.schema_version,
    )
    snapshot.backup_to(opaque)
    destination = sqlite3.connect(destination_path)
    try:
        return tuple(
            destination.execute(
                f"SELECT * FROM {REFERENCE_TABLE} WHERE profile_id = ?",
                (PROFILE_ID,),
            ).fetchone()
        )
    finally:
        destination.close()


def test_boundary_evidence_retains_no_private_blob_bytes(tmp_path: Path) -> None:
    path = tmp_path / "candidate.sqlite3"
    reference_before = _create_populated_v3_reference(path)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    captured: list[object] = []
    original_capture = migration_candidate._capture_boundary_evidence

    def capture(candidate: sqlite3.Connection, version: int):
        evidence = original_capture(candidate, version)
        captured.append(evidence)
        return evidence

    migration_candidate._capture_boundary_evidence = capture
    try:
        step_profile_migration_candidate(
            connection,
            boundary_sink=lambda snapshot, request: _candidate_reference_row(
                snapshot,
                request,
                tmp_path / "compact.sqlite3",
            ),
        )
    finally:
        migration_candidate._capture_boundary_evidence = original_capture

    assert captured
    assert not _nested_contains_bytes(captured[0])
    assert repr(reference_before[2]) not in repr(captured[0])
    assert reference_before[3] not in repr(captured[0])


def test_boundary_compact_evidence_rejects_isolated_transcript_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidate.sqlite3"
    reference_before = _create_populated_v3_reference(path)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    destination_path = tmp_path / "mutated-boundary.sqlite3"

    def mutate_isolated(
        snapshot: ProfileMigrationBoundarySnapshot,
        request: ProfileMigrationBoundaryRequest,
    ) -> None:
        isolated = object.__getattribute__(
            snapshot,
            "_ProfileMigrationBoundarySnapshot__snapshot",
        )
        assert isinstance(isolated, sqlite3.Connection)
        isolated.execute(
            f"UPDATE {REFERENCE_TABLE} SET reference_text = 'Different valid transcript'"
        )
        isolated.commit()
        destination = private_sqlite.open_profile_migration_boundary_destination(
            destination_path,
            schema_version=request.schema_version,
        )
        with destination:
            snapshot.backup_to(destination)

    with _safe_error("migration_failed"):
        step_profile_migration_candidate(connection, boundary_sink=mutate_isolated)

    reopened = sqlite3.connect(destination_path)
    try:
        assert reopened.execute("PRAGMA user_version").fetchone()[0] == 0
    finally:
        reopened.close()
    live = sqlite3.connect(path)
    try:
        assert tuple(live.execute(f"SELECT * FROM {REFERENCE_TABLE}").fetchone()) == (
            reference_before
        )
    finally:
        live.close()


def _nested_contains_bytes(value: object) -> bool:
    if type(value) is bytes:
        return True
    if hasattr(value, "__dataclass_fields__"):
        return any(
            _nested_contains_bytes(getattr(value, name))
            for name in value.__dataclass_fields__  # type: ignore[attr-defined]
        )
    if isinstance(value, tuple):
        return any(_nested_contains_bytes(item) for item in value)
    return False


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
        step_profile_migration_candidate(
            connection,
            boundary_sink=lambda snapshot, request: _candidate_reference_row(
                snapshot,
                request,
                tmp_path / "mutated.sqlite3",
            ),
        )

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


# --- TASK-21130: the v3->v4 migration must not snapshot the BLOB table -------


def _canonical_wav(seconds: float, rate: int, channels: int, seed: int) -> bytes:
    """One valid canonical reference WAV, deterministic in ``seed``."""

    pattern = bytes((seed * 7 + index) % 251 for index in range(2 * channels))
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(pattern * int(seconds * rate))
    return output.getvalue()


def _create_populated_v3_references(
    path: Path,
    *,
    count: int,
    seconds: float = 1.0,
    rate: int = 96_000,
    channels: int = 2,
) -> int:
    """Build a schema-v3 store with ``count`` references; return total bytes.

    Every payload goes through the production canonical-WAV validator, and
    every metadata column is taken from what that validator reports, so the
    fixture is real data on the production path rather than hand-rolled rows.
    """

    from tldw_chatbook.TTS.profile_reference_audio import (
        validate_canonical_reference_wav,
    )

    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    v0_to_v1.migrate(connection)
    v1_to_v2.migrate(connection)
    for index in range(count):
        connection.execute(
            """
            INSERT INTO tts_generation_profiles (
                profile_id, display_name, normalized_name, provider_id, model_id,
                voice_id, response_format, speed, options_json, revision,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(UUID(int=(index + 1) | (0x8 << 60))),
                f"Private Voice {index}",
                f"private voice {index}",
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
    v2_to_v3.migrate(connection)
    total = 0
    for index in range(count):
        wav_bytes = _canonical_wav(seconds, rate, channels, index)
        metadata = validate_canonical_reference_wav(wav_bytes)
        total += len(wav_bytes)
        connection.execute(
            f"INSERT INTO {REFERENCE_TABLE} "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(UUID(int=(index + 1) | (0x8 << 60))),
                str(UUID(int=(index + 1) | (0x4 << 60) | (0x8 << 76))),
                wav_bytes,
                f"Private transcript {index}",
                hashlib.sha256(wav_bytes).hexdigest(),
                len(wav_bytes),
                metadata.duration_ms,
                metadata.sample_rate_hz,
                metadata.channels,
                metadata.sample_encoding,
                NOW,
                NOW,
            ),
        )
    connection.commit()
    connection.close()
    return total


def _reference_content_digest(path: Path) -> str:
    """Length-framed hash of every reference column, WAV payload included."""

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    digest = hashlib.sha256()
    try:
        for row in connection.execute(
            f"SELECT profile_id, reference_id, wav_bytes, reference_text, sha256,"
            f" byte_length, duration_ms, sample_rate_hz, channels, sample_encoding,"
            f" created_at, updated_at FROM {REFERENCE_TABLE} ORDER BY profile_id"
        ):
            for value in row:
                raw = value if type(value) is bytes else repr(value).encode("utf-8")
                digest.update(len(raw).to_bytes(8, "big"))
                digest.update(raw)
    finally:
        connection.close()
    return digest.hexdigest()


def _migrate_under_tracemalloc(path: Path) -> int:
    gc.collect()
    tracemalloc.start()
    try:
        connection = open_profile_store(path)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
    finally:
        connection.close()
    return peak


def test_v3_to_v4_migration_peak_allocation_does_not_scale_with_the_table(
    tmp_path: Path,
) -> None:
    """TASK-21130: peak allocation must track one reference, not the table.

    Two stores of the same per-reference size and different row counts are
    migrated. Before the fix the migration held two full projections of
    ``wav_bytes``, so the peak tracked the TOTAL payload and the difference
    between the two arms was ~2x the extra bytes; it must now be a rounding
    error against them.
    """
    small_path = tmp_path / "small.sqlite3"
    large_path = tmp_path / "large.sqlite3"
    small_bytes = _create_populated_v3_references(small_path, count=4)
    large_bytes = _create_populated_v3_references(large_path, count=32)
    extra_bytes = large_bytes - small_bytes
    assert extra_bytes > 8 * 1024 * 1024

    small_peak = _migrate_under_tracemalloc(small_path)
    large_peak = _migrate_under_tracemalloc(large_path)

    assert large_peak - small_peak < extra_bytes // 4, (
        f"peak grew {large_peak - small_peak} bytes for {extra_bytes} extra "
        "reference bytes -- the migration is still retaining the table"
    )
    assert large_peak < large_bytes, (
        f"peak {large_peak} exceeds the {large_bytes}-byte reference table"
    )


def test_v3_to_v4_migration_evidence_never_holds_a_reference_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The captured evidence itself must contain no WAV bytes and no transcript."""

    path = tmp_path / "profiles.sqlite3"
    _create_populated_v3_references(path, count=3)
    real_evidence = profile_schema._migration_reference_evidence
    captured: list[object] = []

    def tracked(connection: sqlite3.Connection):
        evidence = real_evidence(connection)
        captured.append(evidence)
        return evidence

    monkeypatch.setattr(profile_schema, "_migration_reference_evidence", tracked)

    connection = open_profile_store(path)
    connection.close()

    assert len(captured) == 2
    for evidence in captured:
        assert len(evidence) == 3
        assert not _nested_contains_bytes(evidence)
        assert "Private transcript" not in repr(evidence)
    assert captured[0] == captured[1]


def test_v3_to_v4_migration_result_is_byte_identical_to_the_v3_payloads(
    tmp_path: Path,
) -> None:
    """Identity is asserted with a content hash over the BLOBs, not a row count."""

    path = tmp_path / "profiles.sqlite3"
    _create_populated_v3_references(path, count=5)
    before = _reference_content_digest(path)
    before_rows = sqlite3.connect(path)
    try:
        raw_before = [
            tuple(row)
            for row in before_rows.execute(
                f"SELECT * FROM {REFERENCE_TABLE} ORDER BY profile_id"
            )
        ]
    finally:
        before_rows.close()

    connection = open_profile_store(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
        assert [row[0] for row in connection.execute("PRAGMA integrity_check")] == [
            "ok"
        ]
        raw_after = [
            tuple(row)
            for row in connection.execute(
                f"SELECT * FROM {REFERENCE_TABLE} ORDER BY profile_id"
            )
        ]
    finally:
        connection.close()

    assert _reference_content_digest(path) == before
    assert [row[:12] for row in raw_after] == raw_before
    assert {row[12:] for row in raw_after} == {(None, None)}


def test_v3_to_v4_migration_rejects_a_payload_only_mutation_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The payload-free evidence must still fail closed on a BLOB-only change.

    The injected mutation swaps in a different but equally valid canonical WAV
    of the SAME length, leaving ``sha256``, ``byte_length`` and every other
    projected column untouched -- the one mutation class the removed
    ``wav_bytes`` projection used to be the only guard against. It is caught
    because ``_validate_migration_reference_rows`` re-derives the digest from
    the stored BLOB on both sides of the climb.
    """
    path = tmp_path / "profiles.sqlite3"
    _create_populated_v3_references(path, count=2)
    before = _reference_content_digest(path)
    replacement = _canonical_wav(1.0, 96_000, 2, seed=99)
    real_migration = profile_schema.MIGRATIONS[3]

    def mutate_payload(connection: sqlite3.Connection) -> None:
        real_migration(connection)
        target = connection.execute(
            f"SELECT profile_id, byte_length FROM {REFERENCE_TABLE} "
            "ORDER BY profile_id LIMIT 1"
        ).fetchone()
        assert len(replacement) == target[1]
        connection.execute(
            f"UPDATE {REFERENCE_TABLE} SET wav_bytes = ? WHERE profile_id = ?",
            (replacement, target[0]),
        )

    monkeypatch.setitem(profile_schema.MIGRATIONS, 3, mutate_payload)

    with _safe_error("migration_failed") as caught:
        open_profile_store(path)

    raw = sqlite3.connect(path)
    try:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == 3
        assert [row[0] for row in raw.execute("PRAGMA integrity_check")] == ["ok"]
    finally:
        raw.close()
    assert _reference_content_digest(path) == before
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("updated_at", "2026-08-11T12:34:56.123456Z"),
        # Same UTF-8 length as "Private transcript 0", so only the digest half
        # of the projection can catch it.
        ("reference_text", "Private transcript X"),
        ("reference_id", "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"),
        ("duration_ms", 4_321),
    ],
)
def test_v3_to_v4_migration_rejects_a_metadata_only_mutation_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    column: str,
    value: object,
) -> None:
    """Every projected column is still compared across the climb.

    ``reference_text`` matters most here: nothing else in the migration
    compares transcripts across the boundary, so the UTF-8 length + digest
    that replaced the raw text in the evidence is the only guard on it.
    """

    path = tmp_path / "profiles.sqlite3"
    _create_populated_v3_references(path, count=2)
    before = _reference_content_digest(path)
    real_migration = profile_schema.MIGRATIONS[3]

    def mutate_metadata(connection: sqlite3.Connection) -> None:
        real_migration(connection)
        target = connection.execute(
            f"SELECT profile_id FROM {REFERENCE_TABLE} ORDER BY profile_id LIMIT 1"
        ).fetchone()[0]
        connection.execute(
            f"UPDATE {REFERENCE_TABLE} SET {column} = ? WHERE profile_id = ?",
            (value, target),
        )

    monkeypatch.setitem(profile_schema.MIGRATIONS, 3, mutate_metadata)

    with _safe_error("migration_failed"):
        open_profile_store(path)

    raw = sqlite3.connect(path)
    try:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == 3
    finally:
        raw.close()
    assert _reference_content_digest(path) == before


def test_v3_store_whose_digest_column_disagrees_with_its_blob_never_migrates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First link of the payload-identity chain, isolated from the closing one.

    The evidence no longer carries ``wav_bytes``, so it stands on the stored
    ``sha256`` column being a true digest of the payload BEFORE the climb
    starts. Asserting only that such a store fails to reach v4 would prove
    nothing about that link -- the post-migration validation catches the same
    poisoned digest on its own (verified by mutation). So this also requires
    that the v3->v4 migration is never *entered*: the pre-migration
    validation has to reject the store first.
    """

    path = tmp_path / "profiles.sqlite3"
    _create_populated_v3_references(path, count=2)
    before = _reference_content_digest(path)
    raw = sqlite3.connect(path)
    try:
        raw.execute("PRAGMA ignore_check_constraints = ON")
        raw.execute(
            f"UPDATE {REFERENCE_TABLE} SET sha256 = ? WHERE profile_id = "
            f"(SELECT MIN(profile_id) FROM {REFERENCE_TABLE})",
            ("0" * 64,),
        )
        raw.commit()
    finally:
        raw.close()
    poisoned = _reference_content_digest(path)
    assert poisoned != before

    attempts: list[int] = []
    real_migration = profile_schema.MIGRATIONS[3]

    def spy(connection: sqlite3.Connection) -> None:
        attempts.append(1)
        real_migration(connection)

    monkeypatch.setitem(profile_schema.MIGRATIONS, 3, spy)

    with _safe_error("migration_failed"):
        open_profile_store(path)

    assert attempts == [], "the climb started before the payload was validated"
    check = sqlite3.connect(path)
    try:
        assert check.execute("PRAGMA user_version").fetchone()[0] == 3
    finally:
        check.close()
    assert _reference_content_digest(path) == poisoned


def test_v3_store_with_a_structurally_corrupt_payload_never_migrates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The BLOB-corrupt error path: rejected before the climb, left at v3.

    Here the digest column agrees with the stored bytes -- they are simply not
    a canonical WAV any more. The payload is still streamed and decoded on the
    way in, so the migration must refuse to start.
    """

    path = tmp_path / "profiles.sqlite3"
    _create_populated_v3_references(path, count=2)
    raw = sqlite3.connect(path)
    try:
        target, length = raw.execute(
            f"SELECT profile_id, byte_length FROM {REFERENCE_TABLE} "
            "ORDER BY profile_id LIMIT 1"
        ).fetchone()
        rubbish = bytes((index * 31) % 256 for index in range(length))
        raw.execute(
            f"UPDATE {REFERENCE_TABLE} SET wav_bytes = ?, sha256 = ? "
            "WHERE profile_id = ?",
            (rubbish, hashlib.sha256(rubbish).hexdigest(), target),
        )
        raw.commit()
    finally:
        raw.close()
    corrupted = _reference_content_digest(path)

    attempts: list[int] = []
    real_migration = profile_schema.MIGRATIONS[3]

    def spy(connection: sqlite3.Connection) -> None:
        attempts.append(1)
        real_migration(connection)

    monkeypatch.setitem(profile_schema.MIGRATIONS, 3, spy)

    with _safe_error("migration_failed"):
        open_profile_store(path)

    assert attempts == []
    check = sqlite3.connect(path)
    try:
        assert check.execute("PRAGMA user_version").fetchone()[0] == 3
    finally:
        check.close()
    assert _reference_content_digest(path) == corrupted


def test_v3_to_v4_migration_is_reenterable_after_a_failed_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed attempt leaves v3 intact; the next open completes the climb."""

    path = tmp_path / "profiles.sqlite3"
    _create_populated_v3_references(path, count=3)
    before = _reference_content_digest(path)
    real_migration = profile_schema.MIGRATIONS[3]

    def fail_after_migrating(connection: sqlite3.Connection) -> None:
        real_migration(connection)
        raise RuntimeError("PRIVATE injected failure")

    monkeypatch.setitem(profile_schema.MIGRATIONS, 3, fail_after_migrating)
    with _safe_error("migration_failed"):
        open_profile_store(path)
    raw = sqlite3.connect(path)
    try:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == 3
    finally:
        raw.close()
    assert _reference_content_digest(path) == before

    monkeypatch.setitem(profile_schema.MIGRATIONS, 3, real_migration)
    connection = open_profile_store(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
        assert [row[0] for row in connection.execute("PRAGMA integrity_check")] == [
            "ok"
        ]
    finally:
        connection.close()
    assert _reference_content_digest(path) == before


_KILL_CHILD_TEMPLATE = """
import sys
import time
from pathlib import Path
sys.path.insert(0, {repo!r})
import tldw_chatbook.TTS.profile_schema as profile_schema

real = profile_schema.MIGRATIONS[3]


def park(connection):
    real(connection)
    with open({marker!r}, "w") as handle:
        handle.write("in-transaction")
    while True:
        time.sleep(0.05)


profile_schema.MIGRATIONS[3] = park
profile_schema.open_profile_store(Path({path!r}))
"""


def test_v3_to_v4_migration_survives_sigkill_mid_transaction(tmp_path: Path) -> None:
    """A real SIGKILL inside the climb must leave v3, then re-enter cleanly.

    The child parks INSIDE the migration transaction, after the v4 table has
    been rebuilt and ``PRAGMA user_version = 4`` executed but before commit,
    so the kill lands at the worst possible instant. Recovery must roll the
    whole climb back to v3 with byte-identical payloads, and the next open
    must finish the migration.
    """
    path = tmp_path / "profiles.sqlite3"
    _create_populated_v3_references(path, count=3)
    before = _reference_content_digest(path)
    marker = tmp_path / "parked.marker"
    repo_root = str(Path(__file__).resolve().parents[2])

    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _KILL_CHILD_TEMPLATE.format(
                repo=repo_root,
                marker=str(marker),
                path=str(path),
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            if marker.exists():
                break
            if child.poll() is not None:
                break
            time.sleep(0.02)
        assert child.poll() is None, (
            "child exited before it reached the migration "
            f"(stdout={child.stdout.read()!r} stderr={child.stderr.read()!r})"
        )
        assert marker.exists(), "child never parked inside the migration"
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=60)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=60)

    recovered = sqlite3.connect(path)
    try:
        assert recovered.execute("PRAGMA user_version").fetchone()[0] == 3
        assert [row[0] for row in recovered.execute("PRAGMA integrity_check")] == ["ok"]
    finally:
        recovered.close()
    assert _reference_content_digest(path) == before

    connection = open_profile_store(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
        assert [row[0] for row in connection.execute("PRAGMA integrity_check")] == [
            "ok"
        ]
    finally:
        connection.close()
    assert _reference_content_digest(path) == before


def test_already_migrated_v4_store_captures_no_migration_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reopening a current store must not run the evidence projection at all."""

    path = tmp_path / "profiles.sqlite3"
    _create_populated_v3_references(path, count=2)
    open_profile_store(path).close()
    digest = _reference_content_digest(path)

    calls: list[int] = []
    real_evidence = profile_schema._migration_reference_evidence

    def tracked(connection: sqlite3.Connection):
        calls.append(1)
        return real_evidence(connection)

    monkeypatch.setattr(profile_schema, "_migration_reference_evidence", tracked)

    connection = open_profile_store(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
    finally:
        connection.close()

    assert calls == []
    assert _reference_content_digest(path) == digest


def test_fresh_and_empty_stores_migrate_with_empty_reference_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A brand-new store and a populated v2 store both end with no references."""

    captured: list[object] = []
    real_evidence = profile_schema._migration_reference_evidence

    def tracked(connection: sqlite3.Connection):
        evidence = real_evidence(connection)
        captured.append(evidence)
        return evidence

    monkeypatch.setattr(profile_schema, "_migration_reference_evidence", tracked)

    fresh = open_profile_store(tmp_path / "fresh.sqlite3")
    try:
        assert fresh.execute("PRAGMA user_version").fetchone()[0] == 4
        assert (
            fresh.execute(f"SELECT count(*) FROM {REFERENCE_TABLE}").fetchone()[0] == 0
        )
    finally:
        fresh.close()

    v2_path = tmp_path / "v2.sqlite3"
    _create_populated_v2(v2_path)
    upgraded = open_profile_store(v2_path)
    try:
        assert upgraded.execute("PRAGMA user_version").fetchone()[0] == 4
        assert (
            upgraded.execute(f"SELECT count(*) FROM {REFERENCE_TABLE}").fetchone()[0]
            == 0
        )
    finally:
        upgraded.close()

    assert captured == [(), ()]


def test_candidate_and_live_migration_share_one_reference_evidence_shape(
    tmp_path: Path,
) -> None:
    """The candidate stepper and the live opener must project identically."""

    path = tmp_path / "shared.sqlite3"
    _create_populated_v3_references(path, count=2)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        assert migration_candidate._compact_reference_evidence(
            connection
        ) == profile_schema._migration_reference_evidence(connection)
    finally:
        connection.close()
