"""Tests for the dedicated versioned TTS profile SQLite schema."""

from __future__ import annotations

import inspect
import sqlite3
import tempfile
import threading
from collections.abc import Callable, Mapping
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

import tldw_chatbook.DB.private_sqlite as private_sqlite
import tldw_chatbook.TTS.profile_schema as profile_schema
from tldw_chatbook.TTS.migrations.v0_to_v1 import migrate as _raw_migrate_v0_to_v1
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_migration_candidate import (
    ProfileMigrationBoundary,
    ProfileMigrationBoundaryRequest,
    ProfileMigrationBoundarySnapshot,
    step_profile_migration_candidate,
)
from tldw_chatbook.TTS.profile_schema import (
    ASSIGNMENT_PROFILE_INDEX,
    BUSY_TIMEOUT_MS,
    CURRENT_PROFILE_SCHEMA_VERSION,
    MIGRATIONS,
    decode_assignment,
    decode_assigned_snapshot,
    decode_options,
    decode_profile,
    decode_utc_datetime,
    decode_uuid,
    encode_assignment,
    encode_options,
    encode_profile,
    encode_utc_datetime,
    encode_uuid,
    open_profile_store,
    validate_profile_candidate,
)
from tldw_chatbook.TTS.profile_types import (
    AssignedTTSProfileSnapshot,
    CharacterRef,
    CharacterTTSAssignment,
    TTSGenerationProfile,
    canonical_json_options,
)

NOW = datetime(2026, 7, 26, 12, 34, 56, 123456, tzinfo=UTC)
PROFILE_ID = UUID("01234567-89ab-cdef-8123-456789abcdef")


def _profile(**overrides: object) -> TTSGenerationProfile:
    values: dict[str, object] = {
        "profile_id": PROFILE_ID,
        "display_name": "Straße 音声",
        "normalized_name": "strasse 音声",
        "provider_id": "audio_cpp",
        "model_id": "model",
        "voice_id": None,
        "response_format": "wav",
        "speed": 1.0,
        "options": {},
        "revision": 7,
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(overrides)
    return TTSGenerationProfile(**values)  # type: ignore[arg-type]


def _assignment() -> CharacterTTSAssignment:
    return CharacterTTSAssignment(
        character_ref=CharacterRef(
            source="server", authority_id="srv-01", character_id="char-α"
        ),
        profile_id=PROFILE_ID,
    )


def _insert_profile(
    connection: sqlite3.Connection, profile: TTSGenerationProfile
) -> None:
    connection.execute(
        """
        INSERT INTO tts_generation_profiles (
            profile_id, display_name, normalized_name, provider_id, model_id,
            voice_id, response_format, speed, options_json, revision,
            created_at, updated_at
        ) VALUES (
            :profile_id, :display_name, :normalized_name, :provider_id, :model_id,
            :voice_id, :response_format, :speed, :options_json, :revision,
            :created_at, :updated_at
        )
        """,
        encode_profile(profile),
    )


def _insert_assignment(
    connection: sqlite3.Connection, assignment: CharacterTTSAssignment
) -> None:
    connection.execute(
        """
        INSERT INTO character_tts_assignments (
            source, authority_id, character_id, profile_id, created_at, updated_at
        ) VALUES (
            :source, :authority_id, :character_id, :profile_id, :created_at,
            :updated_at
        )
        """,
        encode_assignment(assignment, created_at=NOW, updated_at=NOW),
    )


def _safe_error(code: str) -> pytest.RaisesExc[ProfileRepositoryError]:
    return pytest.raises(
        ProfileRepositoryError,
        match=rf"^TTS profile repository failed: {code}$",
    )


def _directory_snapshot(path: Path) -> dict[str, bytes | None]:
    return {
        item.name: item.read_bytes() if item.is_file() else None
        for item in path.iterdir()
    }


_STANDARD_PROFILE_DDL = """
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
_STANDARD_ASSIGNMENT_DDL = """
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


def _create_custom_v1(
    path: Path,
    *,
    profile_ddl: str = _STANDARD_PROFILE_DDL,
    assignment_ddl: str = _STANDARD_ASSIGNMENT_DDL,
    extra_statements: tuple[str, ...] = (),
) -> None:
    connection = sqlite3.connect(path)
    connection.execute(profile_ddl)
    connection.execute(assignment_ddl)
    connection.execute(
        "CREATE INDEX idx_character_tts_assignments_profile_id "
        "ON character_tts_assignments(profile_id)"
    )
    for statement in extra_statements:
        connection.execute(statement)
    connection.execute("PRAGMA user_version = 1")
    connection.close()


def _build_populated_v1_store(db_path: Path) -> None:
    """Build an honest, populated v1 store by running the real v0->v1 migration.

    This deliberately does not fabricate a v1 store by monkeypatching
    ``CURRENT_PROFILE_SCHEMA_VERSION`` back to 1 -- it runs the module's own
    v0->v1 migration function on a raw connection, then inserts one profile
    row through the file's existing ``_insert_profile`` helper, so the
    resulting store is byte-for-byte what a pre-slice user's store would be.
    """

    connection = sqlite3.connect(db_path)
    try:
        _raw_migrate_v0_to_v1(connection)
        _insert_profile(connection, _profile())
        connection.commit()
    finally:
        connection.close()


def _build_candidate_version(db_path: Path, version: int) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    for source_version in range(version):
        MIGRATIONS[source_version](connection)
    if version > 0:
        _insert_profile(connection, _profile())
    connection.commit()
    return connection


def _backup_boundary(
    snapshot: ProfileMigrationBoundarySnapshot,
    request: ProfileMigrationBoundaryRequest,
    destination_path: Path,
) -> Path:
    destination = private_sqlite.open_profile_migration_boundary_destination(
        destination_path,
        schema_version=request.schema_version,
    )
    snapshot.backup_to(destination)
    return destination_path


@pytest.mark.parametrize(
    ("source_version", "expected_boundaries"),
    [
        (0, (ProfileMigrationBoundary.PRE_V3, ProfileMigrationBoundary.PRE_V4)),
        (1, (ProfileMigrationBoundary.PRE_V3, ProfileMigrationBoundary.PRE_V4)),
        (2, (ProfileMigrationBoundary.PRE_V3, ProfileMigrationBoundary.PRE_V4)),
        (3, (ProfileMigrationBoundary.PRE_V4,)),
    ],
)
def test_candidate_stepper_emits_exact_boundaries_in_version_order(
    tmp_path: Path,
    source_version: int,
    expected_boundaries: tuple[ProfileMigrationBoundary, ...],
) -> None:
    connection = _build_candidate_version(
        tmp_path / f"candidate-v{source_version}.sqlite3",
        source_version,
    )
    observed: list[tuple[ProfileMigrationBoundaryRequest, int]] = []

    def consume(
        snapshot: ProfileMigrationBoundarySnapshot,
        request: ProfileMigrationBoundaryRequest,
    ) -> None:
        observed.append((request, request.schema_version))
        _backup_boundary(
            snapshot,
            request,
            tmp_path / f"v{source_version}-{request.kind.value}.sqlite3",
        )

    result = step_profile_migration_candidate(connection, boundary_sink=consume)

    assert result.source_version == source_version
    assert result.final_version == CURRENT_PROFILE_SCHEMA_VERSION == 4
    assert result.boundaries == tuple(request for request, _version in observed)
    assert tuple(request.kind for request, _version in observed) == expected_boundaries
    assert (
        tuple(version for _request, version in observed)
        == tuple(request.schema_version for request, _version in observed)
        == tuple(
            2 if kind is ProfileMigrationBoundary.PRE_V3 else 3
            for kind in expected_boundaries
        )
    )
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")


def test_candidate_boundaries_preserve_exact_source_and_intermediate_domain(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidate.sqlite3"
    connection = _build_candidate_version(path, 2)
    source_domain = profile_schema._migration_domain_snapshot(connection)
    observations: list[
        tuple[
            ProfileMigrationBoundary,
            tuple[tuple[tuple[object, ...], ...], tuple[tuple[object, ...], ...]],
        ]
    ] = []

    def observe(
        snapshot: ProfileMigrationBoundarySnapshot,
        request: ProfileMigrationBoundaryRequest,
    ) -> None:
        observations.append(
            (
                request.kind,
                _snapshot_candidate_domain(
                    snapshot,
                    request,
                    tmp_path / f"domain-{request.kind.value}.sqlite3",
                ),
            )
        )

    step_profile_migration_candidate(connection, boundary_sink=observe)

    assert observations == [
        (ProfileMigrationBoundary.PRE_V3, source_domain),
        (ProfileMigrationBoundary.PRE_V4, source_domain),
    ]


def _snapshot_candidate_domain(
    snapshot: ProfileMigrationBoundarySnapshot,
    request: ProfileMigrationBoundaryRequest,
    destination_path: Path,
) -> tuple[tuple[tuple[object, ...], ...], tuple[tuple[object, ...], ...]]:
    _backup_boundary(snapshot, request, destination_path)
    destination = sqlite3.connect(destination_path)
    try:
        return profile_schema._migration_domain_snapshot(destination)
    finally:
        destination.close()


def test_candidate_boundary_failure_is_bounded_closes_and_does_not_continue(
    tmp_path: Path,
) -> None:
    path = tmp_path / "PRIVATE-candidate.sqlite3"
    connection = _build_candidate_version(path, 2)
    calls: list[ProfileMigrationBoundary] = []

    def fail_boundary(
        _snapshot: ProfileMigrationBoundarySnapshot,
        request: ProfileMigrationBoundaryRequest,
    ) -> None:
        calls.append(request.kind)
        raise RuntimeError(f"PRIVATE boundary failure at {path}")

    with _safe_error("migration_failed") as caught:
        step_profile_migration_candidate(connection, boundary_sink=fail_boundary)

    assert calls == [ProfileMigrationBoundary.PRE_V3]
    assert str(path) not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")
    reopened = sqlite3.connect(path)
    try:
        assert reopened.execute("PRAGMA user_version").fetchone()[0] == 2
    finally:
        reopened.close()


def test_candidate_boundary_capability_exposes_no_sql_mutation_api(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidate.sqlite3"
    connection = _build_candidate_version(path, 2)

    def mutate_boundary(candidate_access: object, _request: object) -> None:
        getattr(candidate_access, "execute")(
            "UPDATE tts_generation_profiles SET revision = revision + 1"
        )

    with _safe_error("migration_failed"):
        step_profile_migration_candidate(connection, boundary_sink=mutate_boundary)

    reopened = sqlite3.connect(path)
    try:
        assert reopened.execute("PRAGMA user_version").fetchone()[0] == 2
        assert (
            reopened.execute("SELECT revision FROM tts_generation_profiles").fetchone()[
                0
            ]
            == _profile().revision
        )
    finally:
        reopened.close()


def test_candidate_boundary_rejects_raw_permissive_destination(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidate.sqlite3"
    destination_path = tmp_path / "raw-destination.sqlite3"
    connection = _build_candidate_version(path, 2)
    destination = sqlite3.connect(destination_path)
    destination.close()
    destination_path.chmod(0o666)
    destination = sqlite3.connect(destination_path)

    with _safe_error("migration_failed"):
        step_profile_migration_candidate(
            connection,
            boundary_sink=lambda snapshot, _request: snapshot.backup_to(destination),
        )

    destination.close()
    assert destination_path.stat().st_mode & 0o777 == 0o666
    reopened = sqlite3.connect(destination_path)
    try:
        assert reopened.execute("PRAGMA user_version").fetchone()[0] == 0
        assert (
            reopened.execute(
                "SELECT COUNT(*) FROM sqlite_schema WHERE name NOT GLOB 'sqlite_*'"
            ).fetchone()[0]
            == 0
        )
    finally:
        reopened.close()


@pytest.mark.parametrize("sink_kind", ["missing", "noop"])
def test_candidate_boundary_requires_sink_to_consume_one_backup(
    tmp_path: Path,
    sink_kind: str,
) -> None:
    connection = _build_candidate_version(tmp_path / "candidate.sqlite3", 2)
    sink = None if sink_kind == "missing" else lambda _snapshot, _request: None

    with _safe_error("migration_failed"):
        step_profile_migration_candidate(
            connection,
            boundary_sink=sink,
        )


def test_candidate_boundary_rejects_destination_for_wrong_boundary(
    tmp_path: Path,
) -> None:
    connection = _build_candidate_version(tmp_path / "candidate.sqlite3", 2)
    destination_path = tmp_path / "wrong-boundary.sqlite3"

    def wrong_boundary(
        snapshot: ProfileMigrationBoundarySnapshot,
        _request: ProfileMigrationBoundaryRequest,
    ) -> None:
        destination = private_sqlite.open_profile_migration_boundary_destination(
            destination_path,
            schema_version=3,
        )
        with destination:
            snapshot.backup_to(destination)

    with _safe_error("migration_failed"):
        step_profile_migration_candidate(connection, boundary_sink=wrong_boundary)

    reopened = sqlite3.connect(destination_path)
    try:
        assert reopened.execute("PRAGMA user_version").fetchone()[0] == 0
    finally:
        reopened.close()


def test_candidate_boundary_requires_exact_destination_type(tmp_path: Path) -> None:
    class ForgedDestination(private_sqlite.ProfileMigrationBoundaryDestination):
        pass

    path = tmp_path / "candidate.sqlite3"
    destination_path = tmp_path / "forged.sqlite3"
    connection = _build_candidate_version(path, 2)
    real = private_sqlite.open_profile_migration_boundary_destination(
        destination_path,
        schema_version=2,
    )
    forged = object.__new__(ForgedDestination)
    for name in private_sqlite.ProfileMigrationBoundaryDestination.__slots__:
        private_name = f"_ProfileMigrationBoundaryDestination{name}"
        object.__setattr__(
            forged,
            private_name,
            object.__getattribute__(real, private_name),
        )

    with real, _safe_error("migration_failed"):
        step_profile_migration_candidate(
            connection,
            boundary_sink=lambda snapshot, _request: snapshot.backup_to(forged),
        )

    reopened = sqlite3.connect(destination_path)
    try:
        assert reopened.execute("PRAGMA user_version").fetchone()[0] == 0
    finally:
        reopened.close()


def test_candidate_boundary_requires_success_after_caught_invalid_attempt(
    tmp_path: Path,
) -> None:
    connection = _build_candidate_version(tmp_path / "candidate.sqlite3", 3)
    raw = sqlite3.connect(":memory:")
    destination_path = tmp_path / "second-attempt.sqlite3"
    errors: list[str] = []

    def catch_invalid_attempt(
        snapshot: ProfileMigrationBoundarySnapshot,
        request: ProfileMigrationBoundaryRequest,
    ) -> None:
        try:
            snapshot.backup_to(raw)  # type: ignore[arg-type]
        except ProfileRepositoryError as error:
            errors.append(error.code)
        destination = private_sqlite.open_profile_migration_boundary_destination(
            destination_path,
            schema_version=request.schema_version,
        )
        with destination:
            try:
                snapshot.backup_to(destination)
            except ProfileRepositoryError as error:
                errors.append(error.code)

    with _safe_error("migration_failed"):
        step_profile_migration_candidate(
            connection,
            boundary_sink=catch_invalid_attempt,
        )

    raw.close()
    assert errors == ["migration_failed", "migration_failed"]
    reopened = sqlite3.connect(destination_path)
    try:
        assert reopened.execute("PRAGMA user_version").fetchone() == (0,)
    finally:
        reopened.close()


def test_candidate_boundary_cannot_disable_guard_and_commit_domain_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidate.sqlite3"
    connection = _build_candidate_version(path, 2)

    def mutate_boundary(
        candidate_access: object,
        _request: ProfileMigrationBoundaryRequest,
    ) -> None:
        execute = getattr(candidate_access, "execute")
        execute("PRAGMA query_only = OFF")
        execute("UPDATE tts_generation_profiles SET revision = revision + 1")
        getattr(candidate_access, "commit")()

    with _safe_error("migration_failed"):
        step_profile_migration_candidate(connection, boundary_sink=mutate_boundary)

    reopened = sqlite3.connect(path)
    try:
        assert (
            reopened.execute("SELECT revision FROM tts_generation_profiles").fetchone()[
                0
            ]
            == _profile().revision
        )
        assert reopened.execute("PRAGMA user_version").fetchone()[0] == 2
    finally:
        reopened.close()


def test_candidate_boundary_registry_introspection_can_mutate_only_isolated_copy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidate.sqlite3"
    connection = _build_candidate_version(path, 2)
    destination_path = tmp_path / "isolated-copy.sqlite3"
    destination = private_sqlite.open_profile_migration_boundary_destination(
        destination_path,
        schema_version=2,
    )
    exposed_connections: list[sqlite3.Connection] = []

    def mutate_registry_snapshot(
        snapshot: ProfileMigrationBoundarySnapshot,
        _request: ProfileMigrationBoundaryRequest,
    ) -> None:
        exposed = object.__getattribute__(
            snapshot,
            "_ProfileMigrationBoundarySnapshot__snapshot",
        )
        assert isinstance(exposed, sqlite3.Connection)
        exposed_connections.append(exposed)
        exposed.execute("UPDATE tts_generation_profiles SET revision = revision + 1")
        exposed.commit()
        snapshot.backup_to(destination)

    with destination, _safe_error("migration_failed") as caught:
        step_profile_migration_candidate(
            connection,
            boundary_sink=mutate_registry_snapshot,
        )

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    copied = sqlite3.connect(destination_path)
    try:
        assert copied.execute("PRAGMA user_version").fetchone()[0] == 0
        assert (
            copied.execute(
                "SELECT COUNT(*) FROM sqlite_schema WHERE name NOT GLOB 'sqlite_*'"
            ).fetchone()[0]
            == 0
        )
    finally:
        copied.close()
    assert len(exposed_connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        exposed_connections[0].execute("SELECT 1")

    reopened = sqlite3.connect(path)
    try:
        assert reopened.execute("PRAGMA user_version").fetchone()[0] == 2
        assert (
            reopened.execute("SELECT revision FROM tts_generation_profiles").fetchone()[
                0
            ]
            == _profile().revision
        )
    finally:
        reopened.close()


def test_candidate_boundary_revalidates_live_evidence_after_callback(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidate.sqlite3"
    connection = _build_candidate_version(path, 2)

    def mutate_retained_live_alias(*_args: object) -> None:
        connection.execute("UPDATE tts_generation_profiles SET revision = revision + 1")
        connection.commit()

    with _safe_error("migration_failed"):
        step_profile_migration_candidate(
            connection,
            boundary_sink=mutate_retained_live_alias,
        )

    reopened = sqlite3.connect(path)
    try:
        assert reopened.execute("PRAGMA user_version").fetchone()[0] == 2
        assert (
            reopened.execute("SELECT revision FROM tts_generation_profiles").fetchone()[
                0
            ]
            == _profile().revision + 1
        )
    finally:
        reopened.close()


def test_candidate_boundary_callback_failure_revokes_and_closes_isolated_snapshot(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidate.sqlite3"
    connection = _build_candidate_version(path, 2)
    retained_capabilities: list[ProfileMigrationBoundarySnapshot] = []
    retained_snapshots: list[sqlite3.Connection] = []

    def fail_callback(
        snapshot: ProfileMigrationBoundarySnapshot,
        _request: ProfileMigrationBoundaryRequest,
    ) -> None:
        isolated = object.__getattribute__(
            snapshot,
            "_ProfileMigrationBoundarySnapshot__snapshot",
        )
        assert isinstance(isolated, sqlite3.Connection)
        retained_capabilities.append(snapshot)
        retained_snapshots.append(isolated)
        raise RuntimeError(f"PRIVATE callback failure at {path}")

    with _safe_error("migration_failed") as caught:
        step_profile_migration_candidate(connection, boundary_sink=fail_callback)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert str(path) not in repr(caught.value)
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        retained_snapshots[0].execute("SELECT 1")
    destination = private_sqlite.open_profile_migration_boundary_destination(
        tmp_path / "revoked.sqlite3",
        schema_version=2,
    )
    with destination:
        with _safe_error("migration_failed"):
            retained_capabilities[0].backup_to(destination)
    reopened = sqlite3.connect(path)
    try:
        assert reopened.execute("PRAGMA user_version").fetchone()[0] == 2
        assert (
            reopened.execute("SELECT revision FROM tts_generation_profiles").fetchone()[
                0
            ]
            == _profile().revision
        )
    finally:
        reopened.close()


def test_candidate_boundary_control_flow_wins_after_isolated_snapshot_cleanup(
    tmp_path: Path,
) -> None:
    class StopCandidate(BaseException):
        pass

    path = tmp_path / "candidate.sqlite3"
    connection = _build_candidate_version(path, 2)
    signal = StopCandidate()
    retained_snapshots: list[sqlite3.Connection] = []

    def stop_callback(
        snapshot: ProfileMigrationBoundarySnapshot,
        _request: ProfileMigrationBoundaryRequest,
    ) -> None:
        isolated = object.__getattribute__(
            snapshot,
            "_ProfileMigrationBoundarySnapshot__snapshot",
        )
        assert isinstance(isolated, sqlite3.Connection)
        retained_snapshots.append(isolated)
        raise signal

    with pytest.raises(StopCandidate) as caught:
        step_profile_migration_candidate(connection, boundary_sink=stop_callback)

    assert caught.value is signal
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        retained_snapshots[0].execute("SELECT 1")
    reopened = sqlite3.connect(path)
    try:
        assert reopened.execute("PRAGMA user_version").fetchone()[0] == 2
    finally:
        reopened.close()


@pytest.mark.parametrize("primary_kind", ["control_flow", "ordinary"])
def test_candidate_snapshot_backup_cleanup_preserves_error_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary_kind: str,
) -> None:
    class StopSnapshot(BaseException):
        pass

    class FailingCloseConnection(sqlite3.Connection):
        def close(self) -> None:
            super().close()
            raise RuntimeError("PRIVATE snapshot close failure")

    path = tmp_path / "candidate.sqlite3"
    connection = _build_candidate_version(path, 2)
    signal: BaseException = (
        StopSnapshot()
        if primary_kind == "control_flow"
        else RuntimeError("PRIVATE snapshot backup failure")
    )

    monkeypatch.setattr(
        private_sqlite,
        "_connect_registered_sqlite",
        lambda *_args, **_kwargs: sqlite3.connect(
            ":memory:",
            isolation_level=None,
            factory=FailingCloseConnection,
        ),
    )

    def fail_backup(*_args: object, **_kwargs: object) -> None:
        raise signal

    monkeypatch.setattr(private_sqlite, "_backup_pages", fail_backup)

    if primary_kind == "control_flow":
        with pytest.raises(StopSnapshot) as caught:
            step_profile_migration_candidate(
                connection,
                boundary_sink=lambda *_args: pytest.fail("sink must not run"),
            )
        assert caught.value is signal
    else:
        with _safe_error("migration_failed") as caught:
            step_profile_migration_candidate(
                connection,
                boundary_sink=lambda *_args: pytest.fail("sink must not run"),
            )
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "callback_error",
    [ProfileRepositoryError("schema_unsupported"), RuntimeError("PRIVATE callback")],
)
def test_candidate_boundary_normalizes_every_ordinary_collaborator_error(
    tmp_path: Path,
    callback_error: Exception,
) -> None:
    path = tmp_path / "PRIVATE-candidate.sqlite3"
    connection = _build_candidate_version(path, 2)

    def fail_callback(*_args: object) -> None:
        raise callback_error

    with _safe_error("migration_failed") as caught:
        step_profile_migration_candidate(connection, boundary_sink=fail_callback)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert str(path) not in repr(caught.value)
    assert str(callback_error) not in repr(caught.value)


def test_candidate_post_step_validation_failure_rolls_back_and_stops_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "candidate.sqlite3"
    connection = _build_candidate_version(path, 2)
    calls: list[ProfileMigrationBoundary] = []
    real_validate = profile_schema.validate_profile_store_version

    def fail_v3(candidate: sqlite3.Connection, version: int) -> None:
        real_validate(candidate, version)
        if version == 3:
            raise RuntimeError("PRIVATE post-step validation")

    monkeypatch.setattr(profile_schema, "validate_profile_store_version", fail_v3)

    def consume(
        snapshot: ProfileMigrationBoundarySnapshot,
        request: ProfileMigrationBoundaryRequest,
    ) -> None:
        calls.append(request.kind)
        _backup_boundary(snapshot, request, tmp_path / "pre-v3.sqlite3")

    with _safe_error("migration_failed"):
        step_profile_migration_candidate(
            connection,
            boundary_sink=consume,
        )

    assert calls == [ProfileMigrationBoundary.PRE_V3]
    reopened = sqlite3.connect(path)
    try:
        assert reopened.execute("PRAGMA user_version").fetchone()[0] == 2
    finally:
        reopened.close()


@pytest.mark.parametrize("source_kind", ["newer", "partial", "malformed"])
def test_candidate_uncertain_source_refuses_before_boundary_callback(
    tmp_path: Path,
    source_kind: str,
) -> None:
    path = tmp_path / f"PRIVATE-{source_kind}.sqlite3"
    if source_kind == "newer":
        connection = sqlite3.connect(path)
        connection.execute("PRAGMA user_version = 5")
    elif source_kind == "partial":
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE unexpected(value TEXT)")
    else:
        path.write_bytes(b"PRIVATE malformed sqlite")
        connection = sqlite3.connect(path)
    connection.commit()
    calls: list[ProfileMigrationBoundary] = []

    with pytest.raises(ProfileRepositoryError) as caught:
        step_profile_migration_candidate(
            connection,
            boundary_sink=lambda _borrowed, request: calls.append(request.kind),
        )

    assert caught.value.code in {
        "schema_unsupported",
        "schema_partial",
        "schema_corrupt",
    }
    assert calls == []
    assert str(path) not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_candidate_boundary_borrow_cannot_outlive_candidate_ownership(
    tmp_path: Path,
) -> None:
    connection = _build_candidate_version(tmp_path / "candidate.sqlite3", 3)
    retained: list[ProfileMigrationBoundarySnapshot] = []

    def consume_and_retain(
        snapshot: ProfileMigrationBoundarySnapshot,
        request: ProfileMigrationBoundaryRequest,
    ) -> None:
        retained.append(snapshot)
        _backup_boundary(snapshot, request, tmp_path / "retained.sqlite3")

    step_profile_migration_candidate(connection, boundary_sink=consume_and_retain)

    assert len(retained) == 1
    destination = private_sqlite.open_profile_migration_boundary_destination(
        tmp_path / "expired.sqlite3",
        schema_version=3,
    )
    with destination:
        with _safe_error("migration_failed"):
            retained[0].backup_to(destination)


def test_candidate_boundary_snapshot_is_same_thread_only(tmp_path: Path) -> None:
    connection = _build_candidate_version(tmp_path / "candidate.sqlite3", 3)
    destination = private_sqlite.open_profile_migration_boundary_destination(
        tmp_path / "foreign-thread.sqlite3",
        schema_version=3,
    )
    observed: list[str] = []

    def cross_thread(
        snapshot: ProfileMigrationBoundarySnapshot,
        _request: ProfileMigrationBoundaryRequest,
    ) -> None:
        def use_in_foreign_thread() -> None:
            try:
                snapshot.backup_to(destination)
            except ProfileRepositoryError as error:
                observed.append(error.code)

        worker = threading.Thread(target=use_in_foreign_thread)
        worker.start()
        worker.join()

    with destination, _safe_error("migration_failed"):
        step_profile_migration_candidate(connection, boundary_sink=cross_thread)

    assert observed == ["migration_failed"]


def test_candidate_boundary_value_is_immutable_and_payload_free() -> None:
    request = ProfileMigrationBoundaryRequest(ProfileMigrationBoundary.PRE_V4, 3)

    with pytest.raises(FrozenInstanceError):
        request.schema_version = 4  # type: ignore[misc]

    assert "sqlite" not in repr(request).casefold()
    assert not hasattr(request, "connection")
    assert not hasattr(request, "path")

    with _safe_error("migration_failed"):
        ProfileMigrationBoundarySnapshot(object(), object(), object())  # type: ignore[arg-type]


def test_candidate_boundary_snapshot_is_single_use_and_exact(
    tmp_path: Path,
) -> None:
    connection = _build_candidate_version(tmp_path / "candidate.sqlite3", 2)
    expected_domain = profile_schema._migration_domain_snapshot(connection)
    copied_domains: list[object] = []

    def copy_boundary(
        snapshot: ProfileMigrationBoundarySnapshot,
        request: ProfileMigrationBoundaryRequest,
    ) -> None:
        assert not hasattr(snapshot, "execute")
        assert not hasattr(snapshot, "commit")
        assert not hasattr(snapshot, "source")
        assert "sqlite" not in repr(snapshot).casefold()
        destination_path = tmp_path / f"copy-{request.kind.value}.sqlite3"
        destination = private_sqlite.open_profile_migration_boundary_destination(
            destination_path,
            schema_version=request.schema_version,
        )
        replay = private_sqlite.open_profile_migration_boundary_destination(
            tmp_path / f"replay-{request.kind.value}.sqlite3",
            schema_version=request.schema_version,
        )
        with replay:
            snapshot.backup_to(destination)
            copied = sqlite3.connect(destination_path)
            try:
                copied_domains.append(
                    (request.kind, profile_schema._migration_domain_snapshot(copied))
                )
            finally:
                copied.close()
            with _safe_error("migration_failed"):
                snapshot.backup_to(replay)

    step_profile_migration_candidate(connection, boundary_sink=copy_boundary)

    assert copied_domains == [
        (ProfileMigrationBoundary.PRE_V3, expected_domain),
        (ProfileMigrationBoundary.PRE_V4, expected_domain),
    ]


def test_candidate_boundary_snapshot_rejects_populated_foreign_destination(
    tmp_path: Path,
) -> None:
    candidate_path = tmp_path / "candidate.sqlite3"
    foreign_path = tmp_path / "foreign.sqlite3"
    connection = _build_candidate_version(candidate_path, 2)
    foreign = open_profile_store(foreign_path)
    foreign_before = profile_schema._migration_domain_snapshot(foreign)

    with _safe_error("migration_failed"):
        step_profile_migration_candidate(
            connection,
            boundary_sink=lambda snapshot, _request: snapshot.backup_to(foreign),
        )

    try:
        assert foreign.execute("PRAGMA user_version").fetchone()[0] == 4
        assert profile_schema._migration_domain_snapshot(foreign) == foreign_before
    finally:
        foreign.close()


def test_candidate_validates_source_boundaries_and_each_post_step_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _build_candidate_version(tmp_path / "candidate.sqlite3", 2)
    real_validate = profile_schema.validate_profile_store_version
    validated_versions: list[tuple[str, int]] = []

    def tracked(candidate: sqlite3.Connection, version: int) -> None:
        validated_versions.append(
            ("live" if candidate is connection else "snapshot", version)
        )
        real_validate(candidate, version)

    monkeypatch.setattr(profile_schema, "validate_profile_store_version", tracked)

    step_profile_migration_candidate(
        connection,
        boundary_sink=lambda snapshot, request: _backup_boundary(
            snapshot,
            request,
            tmp_path / f"validated-{request.kind.value}.sqlite3",
        ),
    )

    assert [version for owner, version in validated_versions if owner == "live"] == [
        2,
        2,
        2,
        3,
        3,
        3,
        4,
    ]
    assert [
        version for owner, version in validated_versions if owner == "snapshot"
    ] == [
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
    ]


def test_candidate_step_rejects_valid_domain_mutation_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "candidate.sqlite3"
    connection = _build_candidate_version(path, 2)
    real_migration = MIGRATIONS[2]

    def mutate_domain(candidate: sqlite3.Connection) -> None:
        real_migration(candidate)
        candidate.execute(
            "UPDATE tts_generation_profiles "
            "SET display_name = 'Changed', normalized_name = 'changed'"
        )

    monkeypatch.setitem(MIGRATIONS, 2, mutate_domain)

    with _safe_error("migration_failed"):
        step_profile_migration_candidate(
            connection,
            boundary_sink=lambda snapshot, request: _backup_boundary(
                snapshot,
                request,
                tmp_path / f"mutated-{request.kind.value}.sqlite3",
            ),
        )

    reopened = sqlite3.connect(path)
    try:
        assert reopened.execute("PRAGMA user_version").fetchone()[0] == 2
        assert (
            reopened.execute(
                "SELECT display_name FROM tts_generation_profiles"
            ).fetchone()[0]
            == _profile().display_name
        )
    finally:
        reopened.close()


@pytest.mark.parametrize("primary_kind", ["control_flow", "ordinary"])
def test_candidate_migration_error_preserves_control_flow_when_rollback_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary_kind: str,
) -> None:
    class StopMigration(BaseException):
        pass

    path = tmp_path / "candidate.sqlite3"
    connection = _build_candidate_version(path, 2)
    signal: BaseException = (
        StopMigration() if primary_kind == "control_flow" else RuntimeError("PRIVATE")
    )

    def fail_migration(candidate: sqlite3.Connection) -> None:
        candidate.execute("UPDATE tts_generation_profiles SET revision = revision + 1")

        def reject_rollback(
            action: int,
            arg1: str | None,
            _arg2: str | None,
            _database: str | None,
            _trigger: str | None,
        ) -> int:
            if action == sqlite3.SQLITE_TRANSACTION and arg1 == "ROLLBACK":
                raise RuntimeError("PRIVATE rollback failure")
            return sqlite3.SQLITE_OK

        candidate.set_authorizer(reject_rollback)
        raise signal

    monkeypatch.setitem(MIGRATIONS, 2, fail_migration)

    if primary_kind == "control_flow":
        with pytest.raises(StopMigration) as caught:
            step_profile_migration_candidate(
                connection,
                boundary_sink=lambda snapshot, request: _backup_boundary(
                    snapshot,
                    request,
                    tmp_path / "rollback-boundary.sqlite3",
                ),
            )
        assert caught.value is signal
    else:
        with _safe_error("migration_failed") as caught:
            step_profile_migration_candidate(
                connection,
                boundary_sink=lambda snapshot, request: _backup_boundary(
                    snapshot,
                    request,
                    tmp_path / "rollback-boundary.sqlite3",
                ),
            )
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None


def test_candidate_callback_control_flow_is_preserved_after_close(
    tmp_path: Path,
) -> None:
    class StopCandidate(BaseException):
        pass

    connection = _build_candidate_version(tmp_path / "candidate.sqlite3", 2)
    signal = StopCandidate()

    with pytest.raises(StopCandidate) as caught:
        step_profile_migration_candidate(
            connection,
            boundary_sink=lambda *_args: (_ for _ in ()).throw(signal),
        )

    assert caught.value is signal
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")


def test_empty_store_migrates_transactionally_and_is_configured(tmp_path: Path) -> None:
    path = tmp_path / "profiles.sqlite3"

    connection = open_profile_store(path)
    try:
        assert connection.row_factory is sqlite3.Row
        assert (
            connection.execute("PRAGMA user_version").fetchone()[0]
            == CURRENT_PROFILE_SCHEMA_VERSION
            == 4
        )
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert (
            connection.execute("PRAGMA busy_timeout").fetchone()[0] == BUSY_TIMEOUT_MS
        )
        assert set(MIGRATIONS) == {0, 1, 2, 3}
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table'"
            )
        }
        assert {
            "tts_generation_profiles",
            "character_tts_assignments",
        } <= tables
    finally:
        connection.close()


def test_profile_schema_ddl_lives_in_versioned_migration_module() -> None:
    schema_source = inspect.getsource(profile_schema)

    assert "CREATE TABLE tts_generation_profiles" not in schema_source

    from tldw_chatbook.TTS.migrations import v0_to_v1, v1_to_v2, v2_to_v3, v3_to_v4

    assert v0_to_v1.TARGET_VERSION == 1
    assert v1_to_v2.TARGET_VERSION == 2
    assert v2_to_v3.TARGET_VERSION == 3
    assert v3_to_v4.TARGET_VERSION == CURRENT_PROFILE_SCHEMA_VERSION == 4
    assert MIGRATIONS[0] is v0_to_v1.migrate
    assert MIGRATIONS[1] is v1_to_v2.migrate
    assert MIGRATIONS[2] is v2_to_v3.migrate
    assert MIGRATIONS[3] is v3_to_v4.migrate


@pytest.mark.parametrize(
    "operation",
    [
        lambda connection: profile_schema._table_xinfo_manifest(
            connection,
            "profiles); SELECT 1; --",
        ),
        lambda connection: profile_schema._has_exact_binary_index_keys(
            connection,
            "index); SELECT 1; --",
            ("profile_id",),
        ),
        lambda connection: profile_schema._has_exact_primary_key_index(
            connection,
            "profiles); SELECT 1; --",
            ("profile_id",),
        ),
    ],
)
def test_schema_pragma_helpers_reject_invalid_identifiers_before_sqlite(
    operation: Callable[[sqlite3.Connection], object],
) -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        with pytest.raises(ValueError):
            operation(connection)
    finally:
        connection.close()


@pytest.mark.parametrize("must_exist", [None, 0, 1, "true", object()])
def test_live_store_opener_requires_exact_bool_for_must_exist(
    tmp_path: Path,
    must_exist: object,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    before = _directory_snapshot(tmp_path)

    with _safe_error("operation_failed") as caught:
        open_profile_store(path, must_exist=must_exist)  # type: ignore[arg-type]

    assert _directory_snapshot(tmp_path) == before
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_live_store_no_create_mode_rejects_missing_without_creating(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing # profile?.sqlite3"
    before = _directory_snapshot(tmp_path)

    with _safe_error("missing") as caught:
        open_profile_store(path, must_exist=True)

    assert _directory_snapshot(tmp_path) == before
    assert path.exists() is False
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert str(path) not in str(caught.value)


def test_live_store_no_create_mode_rejects_non_file_without_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    path.mkdir()
    before = _directory_snapshot(tmp_path)

    with _safe_error("missing") as caught:
        open_profile_store(path, must_exist=True)

    assert _directory_snapshot(tmp_path) == before
    assert path.is_dir()
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_live_store_no_create_mode_rejects_existing_empty_file_without_migration(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty.sqlite3"
    path.touch()
    before = path.read_bytes()

    with _safe_error("schema_partial") as caught:
        open_profile_store(path, must_exist=True)

    assert path.read_bytes() == before == b""
    assert not tuple(tmp_path.glob("empty.sqlite3-*"))
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_live_store_no_create_mode_uses_quoted_rw_uri_and_validates_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "profiles # ready?.sqlite3"
    created = open_profile_store(path)
    created.close()
    real_connect = sqlite3.connect
    calls: list[tuple[object, dict[str, object]]] = []

    def tracked_connect(
        database: object,
        *args: object,
        **kwargs: object,
    ) -> sqlite3.Connection:
        calls.append((database, dict(kwargs)))
        return real_connect(database, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)

    connection = open_profile_store(path, must_exist=True)
    try:
        assert (
            connection.execute("PRAGMA user_version").fetchone()[0]
            == CURRENT_PROFILE_SCHEMA_VERSION
        )
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        connection.close()

    assert calls == [
        (
            f"{path.resolve().as_uri()}?mode=rw",
            {"uri": True, "isolation_level": None},
        )
    ]


def test_live_store_no_create_race_to_missing_is_bounded_and_context_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    created = open_profile_store(path)
    created.close()
    real_connect = sqlite3.connect

    def remove_before_connect(
        database: object,
        *args: object,
        **kwargs: object,
    ) -> sqlite3.Connection:
        path.unlink()
        return real_connect(database, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sqlite3, "connect", remove_before_connect)

    with _safe_error("missing") as caught:
        open_profile_store(path, must_exist=True)

    assert path.exists() is False
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert str(path) not in str(caught.value)


def test_v1_schema_has_required_constraints_and_index(tmp_path: Path) -> None:
    connection = open_profile_store(tmp_path / "profiles.sqlite3")
    try:
        profile_columns = {
            row["name"]: row
            for row in connection.execute("PRAGMA table_info(tts_generation_profiles)")
        }
        assert list(profile_columns) == [
            "profile_id",
            "display_name",
            "normalized_name",
            "provider_id",
            "model_id",
            "voice_id",
            "response_format",
            "speed",
            "options_json",
            "revision",
            "created_at",
            "updated_at",
        ]
        assert profile_columns["profile_id"]["pk"] == 1
        assert profile_columns["voice_id"]["notnull"] == 0
        assignment_columns = list(
            connection.execute("PRAGMA table_info(character_tts_assignments)")
        )
        assert [
            row["name"]
            for row in sorted(assignment_columns, key=lambda row: row["pk"])
            if row["pk"]
        ] == ["source", "authority_id", "character_id"]
        indexes = {
            row["name"]: row
            for row in connection.execute(
                "PRAGMA index_list(character_tts_assignments)"
            )
        }
        assert ASSIGNMENT_PROFILE_INDEX in indexes
        assert indexes[ASSIGNMENT_PROFILE_INDEX]["origin"] == "c"
        foreign_key = connection.execute(
            "PRAGMA foreign_key_list(character_tts_assignments)"
        ).fetchone()
        assert (
            foreign_key["table"],
            foreign_key["from"],
            foreign_key["to"],
            foreign_key["on_delete"],
        ) == ("tts_generation_profiles", "profile_id", "profile_id", "RESTRICT")
    finally:
        connection.close()


@pytest.mark.parametrize(
    "ddl",
    [
        "CREATE TABLE tts_generation_profiles (profile_id TEXT)",
        "CREATE TABLE unrelated (value TEXT)",
        "CREATE TABLE sqliteXlooks_internal (value TEXT)",
        "CREATE VIEW unrelated_view AS SELECT 1 AS value",
    ],
)
def test_version_zero_nonempty_store_is_rejected_without_replacement(
    tmp_path: Path, ddl: str
) -> None:
    path = tmp_path / "profiles.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(ddl)
    connection.commit()
    connection.close()
    before = path.read_bytes()

    with _safe_error("schema_partial"):
        open_profile_store(path)

    assert path.read_bytes() == before
    check = sqlite3.connect(path)
    try:
        assert check.execute("PRAGMA user_version").fetchone()[0] == 0
    finally:
        check.close()


def test_newer_schema_is_rejected_without_mutation(tmp_path: Path) -> None:
    path = tmp_path / "profiles.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(f"PRAGMA user_version = {CURRENT_PROFILE_SCHEMA_VERSION + 1}")
    connection.close()
    before = path.read_bytes()

    with _safe_error("schema_unsupported"):
        open_profile_store(path)

    assert path.read_bytes() == before


def test_populated_v1_store_upgrades_in_place_to_v4(tmp_path: Path) -> None:
    db_path = tmp_path / "profiles.sqlite3"
    _build_populated_v1_store(db_path)

    connection = open_profile_store(db_path)
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        assert version == CURRENT_PROFILE_SCHEMA_VERSION == 4
        rows = connection.execute(
            "SELECT COUNT(*) FROM tts_generation_profiles"
        ).fetchone()[0]
        assert rows == 1
        assert (
            decode_profile(
                connection.execute("SELECT * FROM tts_generation_profiles").fetchone()
            )
            == _profile()
        )
    finally:
        connection.close()


def test_future_version_store_still_fails_closed(tmp_path: Path) -> None:
    db_path = tmp_path / "profiles.sqlite3"
    _build_populated_v1_store(db_path)
    raw = sqlite3.connect(db_path)
    raw.execute(f"PRAGMA user_version = {CURRENT_PROFILE_SCHEMA_VERSION + 1}")
    raw.close()
    before = db_path.read_bytes()

    with _safe_error("schema_unsupported"):
        open_profile_store(db_path)

    assert db_path.read_bytes() == before


@pytest.mark.parametrize(
    "schema",
    [
        # Missing required profile column.
        """
        CREATE TABLE tts_generation_profiles (
            profile_id TEXT PRIMARY KEY, display_name TEXT NOT NULL,
            normalized_name TEXT NOT NULL UNIQUE, provider_id TEXT NOT NULL,
            model_id TEXT NOT NULL, voice_id TEXT, response_format TEXT NOT NULL,
            speed REAL NOT NULL, options_json TEXT NOT NULL, revision INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        # Missing normalized-name uniqueness.
        """
        CREATE TABLE tts_generation_profiles (
            profile_id TEXT PRIMARY KEY, display_name TEXT NOT NULL,
            normalized_name TEXT NOT NULL, provider_id TEXT NOT NULL,
            model_id TEXT NOT NULL, voice_id TEXT, response_format TEXT NOT NULL,
            speed REAL NOT NULL, options_json TEXT NOT NULL, revision INTEGER NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        )
        """,
        # Profile UUID identity must retain canonical BINARY equality.
        """
        CREATE TABLE tts_generation_profiles (
            profile_id TEXT PRIMARY KEY COLLATE NOCASE, display_name TEXT NOT NULL,
            normalized_name TEXT NOT NULL UNIQUE, provider_id TEXT NOT NULL,
            model_id TEXT NOT NULL, voice_id TEXT, response_format TEXT NOT NULL,
            speed REAL NOT NULL, options_json TEXT NOT NULL, revision INTEGER NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        )
        """,
        # Profile UUID identity must use the approved ascending key direction.
        """
        CREATE TABLE tts_generation_profiles (
            profile_id TEXT PRIMARY KEY DESC, display_name TEXT NOT NULL,
            normalized_name TEXT NOT NULL UNIQUE, provider_id TEXT NOT NULL,
            model_id TEXT NOT NULL, voice_id TEXT, response_format TEXT NOT NULL,
            speed REAL NOT NULL, options_json TEXT NOT NULL, revision INTEGER NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        )
        """,
    ],
)
def test_malformed_v1_profile_schema_is_rejected(tmp_path: Path, schema: str) -> None:
    path = tmp_path / "profiles.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(schema)
    connection.execute(
        """
        CREATE TABLE character_tts_assignments (
            source TEXT NOT NULL, authority_id TEXT NOT NULL, character_id TEXT NOT NULL,
            profile_id TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            PRIMARY KEY(source, authority_id, character_id),
            FOREIGN KEY(profile_id) REFERENCES tts_generation_profiles(profile_id)
                ON DELETE RESTRICT
        )
        """
    )
    connection.execute(
        "CREATE INDEX idx_character_tts_assignments_profile_id "
        "ON character_tts_assignments(profile_id)"
    )
    connection.execute("PRAGMA user_version = 1")
    connection.close()

    with _safe_error("schema_corrupt"):
        open_profile_store(path)


def test_unexpected_trigger_with_write_semantics_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "profiles.sqlite3"
    _create_custom_v1(
        path,
        extra_statements=(
            """
            CREATE TRIGGER delete_inserted_profile
            AFTER INSERT ON tts_generation_profiles
            BEGIN
                DELETE FROM tts_generation_profiles
                WHERE profile_id = NEW.profile_id;
            END
            """,
        ),
    )
    connection = sqlite3.connect(path)
    _insert_profile(connection, _profile())
    connection.commit()
    assert (
        connection.execute("SELECT COUNT(*) FROM tts_generation_profiles").fetchone()[0]
        == 0
    )
    connection.close()

    with _safe_error("schema_corrupt"):
        open_profile_store(path)


def test_unexpected_unique_index_with_write_semantics_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    _create_custom_v1(
        path,
        extra_statements=(
            "CREATE UNIQUE INDEX unexpected_provider_uniqueness "
            "ON tts_generation_profiles(provider_id)",
        ),
    )
    connection = sqlite3.connect(path)
    _insert_profile(connection, _profile())
    with pytest.raises(sqlite3.IntegrityError):
        _insert_profile(
            connection,
            _profile(
                profile_id=UUID("11234567-89ab-cdef-8123-456789abcdef"),
                display_name="Second",
                normalized_name="second",
            ),
        )
    connection.rollback()
    connection.close()

    with _safe_error("schema_corrupt"):
        open_profile_store(path)


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE TABLE unexpected_user_table (value TEXT)",
        (
            "CREATE VIEW unexpected_user_view AS "
            "SELECT profile_id FROM tts_generation_profiles"
        ),
    ],
)
def test_unexpected_user_schema_object_is_rejected(
    tmp_path: Path, statement: str
) -> None:
    path = tmp_path / "profiles.sqlite3"
    _create_custom_v1(path, extra_statements=(statement,))

    with _safe_error("schema_corrupt"):
        open_profile_store(path)


@pytest.mark.parametrize(
    "column_definition",
    [
        "extra TEXT",
        "extra TEXT NOT NULL DEFAULT 'value'",
        "extra TEXT GENERATED ALWAYS AS (display_name) VIRTUAL",
    ],
)
def test_extra_profile_column_is_rejected(
    tmp_path: Path, column_definition: str
) -> None:
    path = tmp_path / "profiles.sqlite3"
    _create_custom_v1(path)
    connection = sqlite3.connect(path)
    connection.execute(
        f"ALTER TABLE tts_generation_profiles ADD COLUMN {column_definition}"
    )
    connection.close()

    with _safe_error("schema_corrupt"):
        open_profile_store(path)


@pytest.mark.parametrize(
    "profile_ddl",
    [
        _STANDARD_PROFILE_DDL.replace(
            "provider_id TEXT NOT NULL",
            "provider_id TEXT NOT NULL COLLATE NOCASE",
        ),
        _STANDARD_PROFILE_DDL.replace(
            "speed REAL NOT NULL",
            "speed REAL NOT NULL CHECK (speed > 0)",
        ),
        _STANDARD_PROFILE_DDL.replace(
            "response_format TEXT NOT NULL",
            "response_format TEXT NOT NULL DEFAULT 'mp3'",
        ),
    ],
)
def test_unexpected_profile_column_semantics_are_rejected(
    tmp_path: Path, profile_ddl: str
) -> None:
    path = tmp_path / "profiles.sqlite3"
    _create_custom_v1(path, profile_ddl=profile_ddl)

    with _safe_error("schema_corrupt"):
        open_profile_store(path)


def test_unexpected_profile_foreign_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "profiles.sqlite3"
    profile_ddl = _STANDARD_PROFILE_DDL.replace(
        "provider_id TEXT NOT NULL",
        "provider_id TEXT NOT NULL REFERENCES provider_parent(provider_id)",
    )
    _create_custom_v1(
        path,
        profile_ddl=profile_ddl,
        extra_statements=(
            "CREATE TABLE provider_parent (provider_id TEXT PRIMARY KEY)",
        ),
    )

    with _safe_error("schema_corrupt"):
        open_profile_store(path)


@pytest.mark.parametrize("defect", ["partial", "nocase"])
def test_incompatible_unique_index_does_not_satisfy_normalized_name_uniqueness(
    tmp_path: Path, defect: str
) -> None:
    path = tmp_path / "profiles.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE tts_generation_profiles (
            profile_id TEXT PRIMARY KEY, display_name TEXT NOT NULL,
            normalized_name TEXT NOT NULL, provider_id TEXT NOT NULL,
            model_id TEXT NOT NULL, voice_id TEXT, response_format TEXT NOT NULL,
            speed REAL NOT NULL, options_json TEXT NOT NULL, revision INTEGER NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE UNIQUE INDEX incompatible_normalized_name "
        "ON tts_generation_profiles("
        + (
            "normalized_name COLLATE NOCASE"
            if defect == "nocase"
            else "normalized_name"
        )
        + ")"
        + (" WHERE revision > 1" if defect == "partial" else "")
    )
    connection.execute(
        """
        CREATE TABLE character_tts_assignments (
            source TEXT NOT NULL, authority_id TEXT NOT NULL, character_id TEXT NOT NULL,
            profile_id TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            PRIMARY KEY(source, authority_id, character_id),
            FOREIGN KEY(profile_id) REFERENCES tts_generation_profiles(profile_id)
                ON DELETE RESTRICT
        )
        """
    )
    connection.execute(
        "CREATE INDEX idx_character_tts_assignments_profile_id "
        "ON character_tts_assignments(profile_id)"
    )
    connection.execute("PRAGMA user_version = 1")
    connection.close()

    with _safe_error("schema_corrupt"):
        open_profile_store(path)


@pytest.mark.parametrize(
    "defect",
    [
        "pk",
        "pk_nocase",
        "pk_desc",
        "extra_pk",
        "index",
        "partial_index",
        "unique_index",
        "nocase_index",
        "fk",
        "delete",
    ],
)
def test_malformed_v1_assignment_schema_is_rejected(
    tmp_path: Path, defect: str
) -> None:
    path = tmp_path / "profiles.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE tts_generation_profiles (
            profile_id TEXT PRIMARY KEY, display_name TEXT NOT NULL,
            normalized_name TEXT NOT NULL UNIQUE, provider_id TEXT NOT NULL,
            model_id TEXT NOT NULL, voice_id TEXT, response_format TEXT NOT NULL,
            speed REAL NOT NULL, options_json TEXT NOT NULL, revision INTEGER NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        )
        """
    )
    pk = {
        "pk": "PRIMARY KEY(authority_id, source, character_id)",
        "pk_nocase": ("PRIMARY KEY(source, authority_id COLLATE NOCASE, character_id)"),
        "pk_desc": "PRIMARY KEY(source, authority_id DESC, character_id)",
        "extra_pk": "PRIMARY KEY(source, authority_id, character_id, tenant)",
    }.get(defect, "PRIMARY KEY(source, authority_id, character_id)")
    fk = (
        ""
        if defect == "fk"
        else "FOREIGN KEY(profile_id) REFERENCES "
        f"tts_generation_profiles(profile_id) ON DELETE {'CASCADE' if defect == 'delete' else 'RESTRICT'}"
    )
    connection.execute(
        f"""
        CREATE TABLE character_tts_assignments (
            source TEXT NOT NULL, authority_id TEXT NOT NULL, character_id TEXT NOT NULL,
            profile_id TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            {"tenant TEXT NOT NULL," if defect == "extra_pk" else ""}
            {pk}{"," if fk else ""} {fk}
        )
        """
    )
    if defect != "index":
        connection.execute(
            ("CREATE UNIQUE INDEX " if defect == "unique_index" else "CREATE INDEX ")
            + "idx_character_tts_assignments_profile_id "
            "ON character_tts_assignments("
            + (
                "profile_id COLLATE NOCASE"
                if defect == "nocase_index"
                else "profile_id"
            )
            + ")"
            + (" WHERE source = 'local'" if defect == "partial_index" else "")
        )
    if defect == "nocase_index":
        query_plan = " ".join(
            row[3]
            for row in connection.execute(
                "EXPLAIN QUERY PLAN SELECT * FROM character_tts_assignments "
                "WHERE profile_id = ?",
                (str(PROFILE_ID),),
            )
        )
        assert ASSIGNMENT_PROFILE_INDEX not in query_plan
    if defect == "pk_nocase":
        assignment_values = (
            "local",
            "Authority",
            "character",
            str(PROFILE_ID),
            "2026-07-26T12:34:56.123456Z",
            "2026-07-26T12:34:56.123456Z",
        )
        connection.execute(
            "INSERT INTO character_tts_assignments VALUES (?, ?, ?, ?, ?, ?)",
            assignment_values,
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO character_tts_assignments VALUES (?, ?, ?, ?, ?, ?)",
                (
                    assignment_values[0],
                    "authority",
                    *assignment_values[2:],
                ),
            )
        connection.rollback()
    connection.execute("PRAGMA user_version = 1")
    connection.close()

    with _safe_error("schema_corrupt"):
        open_profile_store(path)


def test_corrupt_bytes_are_rejected_without_replacement(tmp_path: Path) -> None:
    path = tmp_path / "profiles.sqlite3"
    raw = b"not sqlite and must remain untouched"
    path.write_bytes(raw)

    with _safe_error("schema_corrupt"):
        open_profile_store(path)

    assert path.read_bytes() == raw


def test_migration_failure_rolls_back_schema_and_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "profiles.sqlite3"

    def fail_after_ddl(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE half_schema (value TEXT)")
        connection.execute("CREATE TABLE half_schema (value TEXT)")

    monkeypatch.setitem(MIGRATIONS, 0, fail_after_ddl)
    with _safe_error("migration_failed"):
        open_profile_store(path)

    check = sqlite3.connect(path)
    try:
        assert check.execute("PRAGMA user_version").fetchone()[0] == 0
        assert (
            check.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' "
                "AND name = 'half_schema'"
            ).fetchone()
            is None
        )
    finally:
        check.close()


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_migration_control_flow_exception_rolls_back_and_closes_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_type: type[BaseException],
) -> None:
    path = tmp_path / "profiles.sqlite3"
    signal = exception_type("control flow")
    real_connect = sqlite3.connect
    retained_connections: list[sqlite3.Connection] = []

    def tracked_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = real_connect(*args, **kwargs)  # type: ignore[arg-type]
        retained_connections.append(connection)
        return connection

    def interrupt_after_ddl(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE half_schema (value TEXT)")
        raise signal

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    monkeypatch.setitem(MIGRATIONS, 0, interrupt_after_ddl)

    try:
        with pytest.raises(exception_type) as caught:
            open_profile_store(path)
        assert caught.value is signal

        second_writer = real_connect(path, timeout=0.1, isolation_level=None)
        try:
            second_writer.execute("BEGIN IMMEDIATE")
            assert second_writer.execute("PRAGMA user_version").fetchone()[0] == 0
            assert (
                second_writer.execute(
                    "SELECT name FROM sqlite_schema "
                    "WHERE type = 'table' AND name = 'half_schema'"
                ).fetchone()
                is None
            )
            second_writer.execute("CREATE TABLE writer_probe (value TEXT)")
            second_writer.rollback()
        finally:
            second_writer.close()
    finally:
        for connection in retained_connections:
            try:
                connection.close()
            except sqlite3.ProgrammingError:
                pass


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_migration_rollback_control_flow_exception_wins_and_connection_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_type: type[BaseException],
) -> None:
    path = tmp_path / "profiles.sqlite3"
    signal = exception_type("rollback control flow")
    real_connect = sqlite3.connect
    rollback_attempts: list[sqlite3.Connection] = []
    opened_connections: list[sqlite3.Connection] = []

    class InterruptingRollback(sqlite3.Connection):
        def rollback(self) -> None:
            rollback_attempts.append(self)
            super().rollback()
            raise signal

    def tracked_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        kwargs["factory"] = InterruptingRollback
        connection = real_connect(*args, **kwargs)  # type: ignore[arg-type]
        opened_connections.append(connection)
        return connection

    def fail_after_ddl(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE half_schema (value TEXT)")
        raise RuntimeError("ordinary migration failure")

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    monkeypatch.setitem(MIGRATIONS, 0, fail_after_ddl)

    with pytest.raises(exception_type) as caught:
        open_profile_store(path)

    assert caught.value is signal
    assert len(rollback_attempts) == len(opened_connections) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        opened_connections[0].execute("SELECT 1")

    second_writer = real_connect(path, timeout=0.1, isolation_level=None)
    try:
        second_writer.execute("BEGIN IMMEDIATE")
        assert second_writer.execute("PRAGMA user_version").fetchone()[0] == 0
        assert (
            second_writer.execute(
                "SELECT name FROM sqlite_schema "
                "WHERE type = 'table' AND name = 'half_schema'"
            ).fetchone()
            is None
        )
        second_writer.rollback()
    finally:
        second_writer.close()


def test_primary_migration_control_flow_signal_survives_cleanup_signals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "profiles.sqlite3"
    primary = KeyboardInterrupt("primary control flow")
    rollback_signal = SystemExit("rollback control flow")
    close_signal = KeyboardInterrupt("close control flow")
    real_connect = sqlite3.connect
    rollback_attempted = False
    close_attempted = False

    class InterruptingCleanup(sqlite3.Connection):
        def rollback(self) -> None:
            nonlocal rollback_attempted
            rollback_attempted = True
            super().rollback()
            raise rollback_signal

        def close(self) -> None:
            nonlocal close_attempted
            close_attempted = True
            super().close()
            raise close_signal

    def tracked_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        kwargs["factory"] = InterruptingCleanup
        return real_connect(*args, **kwargs)  # type: ignore[arg-type]

    def interrupt_after_ddl(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE half_schema (value TEXT)")
        raise primary

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    monkeypatch.setitem(MIGRATIONS, 0, interrupt_after_ddl)

    with pytest.raises(KeyboardInterrupt) as caught:
        open_profile_store(path)

    assert caught.value is primary
    assert rollback_attempted
    assert close_attempted
    check = real_connect(path)
    try:
        assert check.execute("PRAGMA user_version").fetchone()[0] == 0
        assert (
            check.execute(
                "SELECT name FROM sqlite_schema "
                "WHERE type = 'table' AND name = 'half_schema'"
            ).fetchone()
            is None
        )
    finally:
        check.close()


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_live_store_cleanup_close_control_flow_signal_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_type: type[BaseException],
) -> None:
    path = tmp_path / "profiles.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA user_version = 2")
    connection.close()
    signal = exception_type("live close control flow")
    real_connect = sqlite3.connect
    close_attempts: list[sqlite3.Connection] = []

    class InterruptingClose(sqlite3.Connection):
        def close(self) -> None:
            close_attempts.append(self)
            super().close()
            raise signal

    def tracked_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        kwargs["factory"] = InterruptingClose
        return real_connect(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)

    with pytest.raises(exception_type) as caught:
        open_profile_store(path)

    assert caught.value is signal
    assert len(close_attempts) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        close_attempts[0].execute("SELECT 1")
    check = real_connect(path)
    try:
        assert check.execute("PRAGMA user_version").fetchone()[0] == 2
    finally:
        check.close()


def test_profile_and_assignment_codecs_round_trip_exact_values(tmp_path: Path) -> None:
    profile = _profile()
    assignment = _assignment()
    connection = open_profile_store(tmp_path / "profiles.sqlite3")
    try:
        _insert_profile(connection, profile)
        _insert_assignment(connection, assignment)
        connection.commit()
        profile_row = connection.execute(
            "SELECT * FROM tts_generation_profiles"
        ).fetchone()
        assignment_row = connection.execute(
            "SELECT * FROM character_tts_assignments"
        ).fetchone()
        joined_row = connection.execute(
            """
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
                p.updated_at AS profile_updated_at
            FROM character_tts_assignments AS a
            JOIN tts_generation_profiles AS p ON p.profile_id = a.profile_id
            """
        ).fetchone()
        assert decode_profile(profile_row) == profile
        assert decode_assignment(assignment_row) == assignment
        assert decode_assigned_snapshot(joined_row) == AssignedTTSProfileSnapshot(
            assignment=assignment, profile=profile
        )
    finally:
        connection.close()


@pytest.mark.skip(
    reason="options re-enabled in a later slice — no valid draft can carry options in slice 1"
)
def test_profile_options_round_trip_through_codec() -> None:
    """Verify frozen options round-trip through schema codec.

    When options support is re-enabled for non-audio_cpp providers, this test
    documents the expected round-trip behavior for complex nested JSON options.
    """
    profile = _profile(options={"nested": {"items": [True, 2, 3.5, None]}, "é": "声"})
    connection = open_profile_store(Path(tempfile.mkdtemp()) / "profiles.sqlite3")
    try:
        _insert_profile(connection, profile)
        connection.commit()
        profile_row = connection.execute(
            "SELECT * FROM tts_generation_profiles"
        ).fetchone()
        # Verify options are frozen and canonicalized through the round-trip
        assert canonical_json_options(decode_profile(profile_row).options) == (
            '{"é":"声","nested":{"items":[true,2,3.5,null]}}'
        )
    finally:
        connection.close()


def test_scalar_codecs_are_canonical_and_exact() -> None:
    assert encode_uuid(PROFILE_ID) == str(PROFILE_ID)
    assert decode_uuid(str(PROFILE_ID)) == PROFILE_ID
    encoded_time = "2026-07-26T12:34:56.123456Z"
    assert encode_utc_datetime(NOW) == encoded_time
    assert decode_utc_datetime(encoded_time) == NOW
    options = {"z": [1, None], "a": {"é": True}}
    assert encode_options(options) == '{"a":{"é":true},"z":[1,null]}'
    assert canonical_json_options(decode_options(encode_options(options))) == (
        '{"a":{"é":true},"z":[1,null]}'
    )


@pytest.mark.parametrize(
    ("decoder", "value"),
    [
        (decode_uuid, PROFILE_ID.bytes),
        (decode_uuid, str(PROFILE_ID).upper()),
        (decode_utc_datetime, 123),
        (decode_utc_datetime, "2026-07-26T12:34:56+00:00"),
        (decode_options, b"{}"),
        (decode_options, "[]"),
        (decode_options, '{"x": NaN}'),
        (decode_options, '{ "x": 1 }'),
    ],
)
def test_scalar_decoders_fail_closed_without_value_leaks(
    decoder: object, value: object
) -> None:
    with _safe_error("corrupt_data") as caught:
        decoder(value)  # type: ignore[operator]
    assert repr(value) not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("profile_id", 42),
        ("display_name", b"Profile"),
        ("speed", "1.0"),
        ("revision", 0),
        ("created_at", "bad timestamp"),
        ("options_json", '{"bad":NaN}'),
    ],
)
def test_profile_decoder_revalidates_every_persisted_value(
    column: str, bad_value: object
) -> None:
    row = encode_profile(_profile())
    row[column] = bad_value

    with _safe_error("corrupt_data"):
        decode_profile(row)


@pytest.mark.parametrize("validation_kind", ["candidate", "live"])
@pytest.mark.parametrize(
    ("column", "raw_value"),
    [
        ("display_name", "  Straße 音声  "),
        ("display_name", f"{' ' * 129}Straße 音声"),
        ("response_format", " MP3 "),
        ("response_format", f"{' ' * 33}MP3"),
    ],
)
def test_profile_row_validation_rejects_noncanonical_raw_profile_text(
    tmp_path: Path,
    validation_kind: str,
    column: str,
    raw_value: str,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    connection = open_profile_store(path)
    _insert_profile(connection, _profile())
    update_sql = {
        "display_name": ("UPDATE tts_generation_profiles SET display_name = ?"),
        "response_format": ("UPDATE tts_generation_profiles SET response_format = ?"),
    }[column]
    connection.execute(update_sql, (raw_value,))
    connection.commit()

    if validation_kind == "candidate":
        connection.close()
        with _safe_error("corrupt_data"):
            validate_profile_candidate(path)
    else:
        try:
            with _safe_error("corrupt_data"):
                profile_schema.validate_profile_store_rows(connection)
        finally:
            connection.close()


@pytest.mark.parametrize("validation_kind", ["candidate", "live"])
def test_profile_row_validation_rejects_oversized_raw_options_before_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validation_kind: str,
) -> None:
    raw_options = "{" + ",".join(['"é":"声"'] * 1_500) + "}"
    assert len(raw_options) < 16 * 1024
    assert len(raw_options.encode("utf-8")) > 16 * 1024
    canonical_options = canonical_json_options(profile_schema.json.loads(raw_options))
    assert canonical_options == '{"é":"声"}'
    assert len(canonical_options.encode("utf-8")) < 16 * 1024

    path = tmp_path / "profiles.sqlite3"
    connection = open_profile_store(path)
    _insert_profile(connection, _profile())
    connection.execute(
        "UPDATE tts_generation_profiles SET options_json = ?",
        (raw_options,),
    )
    connection.commit()

    parsed_values: list[object] = []
    real_json_loads = profile_schema.json.loads

    def tracked_loads(*args: object, **kwargs: object) -> object:
        parsed_values.append(args[0])
        return real_json_loads(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(profile_schema.json, "loads", tracked_loads)
    if validation_kind == "candidate":
        connection.close()
        with _safe_error("corrupt_data"):
            validate_profile_candidate(path)
    else:
        try:
            with _safe_error("corrupt_data"):
                profile_schema.validate_profile_store_rows(connection)
        finally:
            connection.close()

    assert parsed_values == []


def test_hostile_row_mapping_is_mapped_to_safe_corrupt_data() -> None:
    class HostileRow(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise RuntimeError(f"secret value for {key}")

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter(())

        def __len__(self) -> int:
            return 0

    with _safe_error("corrupt_data") as caught:
        decode_profile(HostileRow())
    assert "secret" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_candidate_validation_reads_all_domain_rows_and_preserves_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidate #1?.sqlite3"
    connection = open_profile_store(path)
    _insert_profile(connection, _profile())
    _insert_assignment(connection, _assignment())
    connection.commit()
    connection.close()
    before = _directory_snapshot(tmp_path)

    validate_profile_candidate(path)

    assert _directory_snapshot(tmp_path) == before


def test_candidate_private_snapshot_is_restrictive_and_removed_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "candidate.sqlite3"
    connection = open_profile_store(path)
    connection.close()
    private_directory = tmp_path / "private-snapshots"
    private_directory.mkdir()
    real_mkstemp = profile_schema.tempfile.mkstemp
    real_connect = sqlite3.connect
    snapshot_paths: list[Path] = []

    def tracked_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        kwargs["dir"] = private_directory
        fd, name = real_mkstemp(*args, **kwargs)  # type: ignore[arg-type]
        snapshot_paths.append(Path(name))
        return fd, name

    def checked_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        assert len(snapshot_paths) == 1
        assert snapshot_paths[0].is_file()
        if profile_schema.os.name == "posix" and callable(
            getattr(profile_schema.os, "fchmod", None)
        ):
            assert (
                profile_schema.stat.S_IMODE(snapshot_paths[0].stat().st_mode) == 0o600
            )
        return real_connect(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(profile_schema.tempfile, "mkstemp", tracked_mkstemp)
    monkeypatch.setattr(sqlite3, "connect", checked_connect)

    validate_profile_candidate(path)

    assert len(snapshot_paths) == 1
    assert not snapshot_paths[0].exists()
    assert list(private_directory.iterdir()) == []


def test_candidate_without_posix_fchmod_still_validates_and_cleans_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "candidate.sqlite3"
    connection = open_profile_store(path)
    connection.close()
    private_directory = tmp_path / "private-snapshots"
    private_directory.mkdir()
    real_mkstemp = profile_schema.tempfile.mkstemp
    snapshot_paths: list[Path] = []

    def tracked_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        kwargs["dir"] = private_directory
        fd, name = real_mkstemp(*args, **kwargs)  # type: ignore[arg-type]
        snapshot_paths.append(Path(name))
        return fd, name

    monkeypatch.setattr(profile_schema.tempfile, "mkstemp", tracked_mkstemp)
    monkeypatch.setattr(profile_schema.os, "fchmod", None, raising=False)

    validate_profile_candidate(path)

    assert len(snapshot_paths) == 1
    assert not snapshot_paths[0].exists()
    assert list(private_directory.iterdir()) == []


def test_candidate_v0_is_rejected_without_migration(tmp_path: Path) -> None:
    path = tmp_path / "candidate.sqlite3"
    sqlite3.connect(path).close()
    before = path.read_bytes()

    with _safe_error("schema_unsupported"):
        validate_profile_candidate(path)

    assert path.read_bytes() == before
    check = sqlite3.connect(path)
    try:
        assert check.execute("PRAGMA user_version").fetchone()[0] == 0
    finally:
        check.close()


def test_populated_v1_candidate_upgrades_disposable_snapshot_and_preserves_original(
    tmp_path: Path,
) -> None:
    """A pre-slice v1 backup must validate, without ever mutating the original.

    The disposable private snapshot copy is upgraded in place (mirroring
    the live open flow's fence) so a v1 candidate validates successfully;
    the caller-supplied candidate file itself is a completely different
    file on disk and must stay byte-for-byte untouched throughout.
    """

    path = tmp_path / "candidate.sqlite3"
    _build_populated_v1_store(path)
    before = path.read_bytes()

    validate_profile_candidate(path)

    assert path.read_bytes() == before
    check = sqlite3.connect(path)
    try:
        assert check.execute("PRAGMA user_version").fetchone()[0] == 1
    finally:
        check.close()


def test_corrupt_but_v1_candidate_fails_closed_before_any_version_stamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A structurally corrupt v1 candidate must fail before any migration write.

    The upgrade step validates the pre-upgrade schema before running any
    migration, exactly like the live open flow -- so a candidate that
    merely claims ``user_version = 1`` but is missing a required column
    must fail with ``schema_corrupt``. Because v1->v2 is a version-fence-only
    migration (no DDL change), running the migration on this same corrupt
    structure would *also* eventually surface ``schema_corrupt`` further
    downstream -- so the error code alone cannot prove the ordering. This
    directly tracks every ``_run_migrations`` call and asserts it is never
    invoked, which is what actually proves no version-stamping write was
    ever attempted.
    """

    migration_calls: list[int] = []
    real_run_migrations = profile_schema._run_migrations

    def tracked_run_migrations(
        connection: sqlite3.Connection, from_version: int
    ) -> None:
        migration_calls.append(from_version)
        real_run_migrations(connection, from_version)

    monkeypatch.setattr(profile_schema, "_run_migrations", tracked_run_migrations)

    path = tmp_path / "candidate.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE tts_generation_profiles (
            profile_id TEXT PRIMARY KEY, display_name TEXT NOT NULL,
            normalized_name TEXT NOT NULL UNIQUE, provider_id TEXT NOT NULL,
            model_id TEXT NOT NULL, voice_id TEXT, response_format TEXT NOT NULL,
            speed REAL NOT NULL, options_json TEXT NOT NULL, revision INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE character_tts_assignments (
            source TEXT NOT NULL, authority_id TEXT NOT NULL, character_id TEXT NOT NULL,
            profile_id TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            PRIMARY KEY(source, authority_id, character_id),
            FOREIGN KEY(profile_id) REFERENCES tts_generation_profiles(profile_id)
                ON DELETE RESTRICT
        )
        """
    )
    connection.execute(
        "CREATE INDEX idx_character_tts_assignments_profile_id "
        "ON character_tts_assignments(profile_id)"
    )
    connection.execute("PRAGMA user_version = 1")
    connection.close()
    before = path.read_bytes()

    with _safe_error("schema_corrupt"):
        validate_profile_candidate(path)

    assert migration_calls == []
    assert path.read_bytes() == before


def test_missing_candidate_is_not_created(tmp_path: Path) -> None:
    path = tmp_path / "missing.sqlite3"

    with _safe_error("missing"):
        validate_profile_candidate(path)

    assert not path.exists()


def test_corrupt_candidate_is_not_replaced(tmp_path: Path) -> None:
    path = tmp_path / "candidate.sqlite3"
    raw = b"malformed candidate"
    path.write_bytes(raw)

    with _safe_error("schema_corrupt"):
        validate_profile_candidate(path)

    assert path.read_bytes() == raw


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_candidate_with_live_sidecar_is_rejected_without_mutation(
    tmp_path: Path, suffix: str
) -> None:
    path = tmp_path / "candidate.sqlite3"
    connection = open_profile_store(path)
    connection.close()
    sidecar = path.with_name(f"{path.name}{suffix}")
    sidecar.write_bytes(b"preexisting sidecar state")
    before = _directory_snapshot(tmp_path)

    with _safe_error("schema_corrupt"):
        validate_profile_candidate(path)

    assert _directory_snapshot(tmp_path) == before


def test_symlink_candidate_cannot_bypass_target_sidecar_rejection(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.sqlite3"
    connection = open_profile_store(target)
    connection.close()
    candidate = tmp_path / "candidate.sqlite3"
    candidate.symlink_to(target)
    target.with_name(f"{target.name}-wal").write_bytes(b"target WAL state")
    before = _directory_snapshot(tmp_path)

    with _safe_error("schema_corrupt"):
        validate_profile_candidate(candidate)

    assert _directory_snapshot(tmp_path) == before


def test_candidate_source_open_is_nonblocking_across_fifo_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "candidate.sqlite3"
    connection = open_profile_store(path)
    connection.close()
    original_path = tmp_path / "original-candidate.sqlite3"
    real_candidate_open = profile_schema._open_candidate_source
    observed_flags: list[int] = []

    def racing_os_open(
        target: object, flags: int, *args: object, **kwargs: object
    ) -> int:
        if Path(target) == path and not observed_flags:
            observed_flags.append(flags)
            path.replace(original_path)
            profile_schema.os.mkfifo(path)
            flags |= profile_schema.os.O_NONBLOCK
        return real_candidate_open(Path(target), flags)

    monkeypatch.setattr(profile_schema, "_open_candidate_source", racing_os_open)
    try:
        with _safe_error("schema_corrupt"):
            validate_profile_candidate(path)
    finally:
        if path.exists():
            path.unlink()
        original_path.replace(path)

    assert len(observed_flags) == 1
    assert observed_flags[0] == profile_schema._candidate_source_open_flags()
    assert observed_flags[0] & profile_schema.os.O_NONBLOCK


def test_candidate_source_open_flags_include_every_available_required_flag() -> None:
    class AllFlags:
        O_RDONLY = 1
        O_CLOEXEC = 2
        O_NONBLOCK = 4
        O_NOFOLLOW = 8
        O_BINARY = 16

    class ReadOnlyFlag:
        O_RDONLY = 1

    assert profile_schema._candidate_source_open_flags(AllFlags()) == 31
    assert profile_schema._candidate_source_open_flags(ReadOnlyFlag()) == 1


def test_candidate_rejects_private_snapshot_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "candidate.sqlite3"
    connection = open_profile_store(path)
    connection.close()
    original = path.read_bytes()

    private_directory = tmp_path / "private-snapshots"
    private_directory.mkdir()
    replacement = private_directory / "replacement.sqlite3"
    connection = open_profile_store(replacement)
    _insert_profile(connection, _profile())
    connection.commit()
    connection.close()

    real_mkstemp = profile_schema.tempfile.mkstemp
    real_connect = sqlite3.connect
    snapshot_paths: list[Path] = []
    replaced = False

    def tracked_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        kwargs["dir"] = private_directory
        fd, name = real_mkstemp(*args, **kwargs)  # type: ignore[arg-type]
        snapshot_paths.append(Path(name))
        return fd, name

    def racing_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        nonlocal replaced
        if snapshot_paths and not replaced:
            replacement.replace(snapshot_paths[0])
            replaced = True
        return real_connect(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(profile_schema.tempfile, "mkstemp", tracked_mkstemp)
    monkeypatch.setattr(sqlite3, "connect", racing_connect)

    with _safe_error("schema_corrupt"):
        validate_profile_candidate(path)

    assert replaced
    assert path.read_bytes() == original
    assert list(private_directory.iterdir()) == []


def test_candidate_rejects_sidecar_created_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "candidate.sqlite3"
    connection = open_profile_store(path)
    connection.close()
    original = path.read_bytes()
    sidecar = path.with_name(f"{path.name}-wal")
    original_entries = set(_directory_snapshot(tmp_path))
    real_connect = sqlite3.connect

    def racing_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        sidecar.write_bytes(b"late WAL state")
        return real_connect(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sqlite3, "connect", racing_connect)

    with _safe_error("schema_corrupt"):
        validate_profile_candidate(path)

    assert path.read_bytes() == original
    assert sidecar.read_bytes() == b"late WAL state"
    assert set(_directory_snapshot(tmp_path)) == original_entries | {sidecar.name}


def test_candidate_ignores_unrelated_sibling_created_during_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "candidate.sqlite3"
    connection = open_profile_store(path)
    connection.close()
    original = path.read_bytes()
    unrelated = tmp_path / "unrelated.txt"
    real_connect = sqlite3.connect

    def racing_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        unrelated.write_text("unrelated directory churn", encoding="utf-8")
        return real_connect(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sqlite3, "connect", racing_connect)

    validate_profile_candidate(path)

    assert path.read_bytes() == original
    assert unrelated.read_text(encoding="utf-8") == "unrelated directory churn"


def test_candidate_rejects_source_modified_during_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "candidate.sqlite3"
    connection = open_profile_store(path)
    _insert_profile(connection, _profile())
    connection.commit()
    connection.close()
    original = path.read_bytes()
    real_decode_profile = profile_schema.decode_profile
    modified = False

    def racing_decode_profile(row: object) -> TTSGenerationProfile:
        nonlocal modified
        if not modified:
            with path.open("ab") as source:
                source.write(b"late source mutation")
            modified = True
        return real_decode_profile(row)  # type: ignore[arg-type]

    monkeypatch.setattr(profile_schema, "decode_profile", racing_decode_profile)

    with _safe_error("schema_corrupt"):
        validate_profile_candidate(path)

    assert path.read_bytes() == original + b"late source mutation"


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_candidate_control_flow_exception_closes_and_removes_private_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_type: type[BaseException],
) -> None:
    path = tmp_path / "candidate.sqlite3"
    connection = open_profile_store(path)
    _insert_profile(connection, _profile())
    connection.commit()
    connection.close()
    original = path.read_bytes()
    private_directory = tmp_path / "private-snapshots"
    private_directory.mkdir()
    signal = exception_type("control flow")
    real_mkstemp = profile_schema.tempfile.mkstemp
    real_candidate_open = profile_schema._open_candidate_source
    real_connect = sqlite3.connect
    snapshot_paths: list[Path] = []
    source_fds: list[int] = []
    snapshot_connections: list[sqlite3.Connection] = []

    def tracked_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        kwargs["dir"] = private_directory
        fd, name = real_mkstemp(*args, **kwargs)  # type: ignore[arg-type]
        snapshot_paths.append(Path(name))
        return fd, name

    def tracked_os_open(*args: object, **kwargs: object) -> int:
        fd = real_candidate_open(*args, **kwargs)  # type: ignore[arg-type]
        if Path(str(args[0])) == path:
            source_fds.append(fd)
        return fd

    def tracked_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        opened = real_connect(*args, **kwargs)  # type: ignore[arg-type]
        snapshot_connections.append(opened)
        return opened

    def interrupt_decode(_row: object) -> TTSGenerationProfile:
        raise signal

    monkeypatch.setattr(profile_schema.tempfile, "mkstemp", tracked_mkstemp)
    monkeypatch.setattr(profile_schema, "_open_candidate_source", tracked_os_open)
    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    monkeypatch.setattr(profile_schema, "decode_profile", interrupt_decode)

    with pytest.raises(exception_type) as caught:
        validate_profile_candidate(path)

    assert caught.value is signal
    assert len(snapshot_paths) == len(source_fds) == 1
    # Two SQLite connections open against the disposable snapshot: the
    # brief read-write reopen that checks/upgrades its schema version, and
    # the immutable read-only handle used for the actual row decode that
    # `interrupt_decode` raises inside.
    assert len(snapshot_connections) == 2
    assert not snapshot_paths[0].exists()
    assert list(private_directory.iterdir()) == []
    with pytest.raises(OSError):
        profile_schema.os.fstat(source_fds[0])
    for snapshot_connection in snapshot_connections:
        with pytest.raises(sqlite3.ProgrammingError):
            snapshot_connection.execute("SELECT 1")
    assert path.read_bytes() == original


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("fd_kind", ["snapshot", "source"])
def test_candidate_fd_cleanup_control_flow_signal_is_preserved_after_all_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_type: type[BaseException],
    fd_kind: str,
) -> None:
    path = tmp_path / "candidate.sqlite3"
    connection = open_profile_store(path)
    connection.close()
    signal = exception_type(f"{fd_kind} close control flow")
    private_directory = tmp_path / "private-snapshots"
    private_directory.mkdir()
    real_mkstemp = profile_schema.tempfile.mkstemp
    real_candidate_open = profile_schema._open_candidate_source
    real_candidate_close = profile_schema._close_candidate_fd
    snapshot_paths: list[Path] = []
    snapshot_fds: list[int] = []
    source_fds: list[int] = []
    close_attempts: list[int] = []

    def tracked_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        kwargs["dir"] = private_directory
        fd, name = real_mkstemp(*args, **kwargs)  # type: ignore[arg-type]
        snapshot_fds.append(fd)
        snapshot_paths.append(Path(name))
        return fd, name

    def tracked_os_open(*args: object, **kwargs: object) -> int:
        fd = real_candidate_open(*args, **kwargs)  # type: ignore[arg-type]
        if Path(str(args[0])) == path:
            source_fds.append(fd)
        return fd

    def interrupting_close(fd: int) -> None:
        close_attempts.append(fd)
        real_candidate_close(fd)
        target_fds = snapshot_fds if fd_kind == "snapshot" else source_fds
        if fd in target_fds:
            raise signal

    monkeypatch.setattr(profile_schema.tempfile, "mkstemp", tracked_mkstemp)
    monkeypatch.setattr(profile_schema, "_open_candidate_source", tracked_os_open)
    monkeypatch.setattr(profile_schema, "_close_candidate_fd", interrupting_close)

    with pytest.raises(exception_type) as caught:
        validate_profile_candidate(path)

    assert caught.value is signal
    assert len(snapshot_paths) == len(snapshot_fds) == len(source_fds) == 1
    assert close_attempts == [snapshot_fds[0], source_fds[0]]
    for fd in (*snapshot_fds, *source_fds):
        with pytest.raises(OSError):
            real_candidate_close(fd)
    assert not snapshot_paths[0].exists()
    assert list(private_directory.iterdir()) == []


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_candidate_unlink_cleanup_control_flow_signal_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_type: type[BaseException],
) -> None:
    path = tmp_path / "candidate.sqlite3"
    connection = open_profile_store(path)
    connection.close()
    signal = exception_type("unlink control flow")
    private_directory = tmp_path / "private-snapshots"
    private_directory.mkdir()
    real_mkstemp = profile_schema.tempfile.mkstemp
    real_unlink = profile_schema.os.unlink
    snapshot_paths: list[Path] = []
    unlink_attempts: list[Path] = []

    def tracked_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        kwargs["dir"] = private_directory
        fd, name = real_mkstemp(*args, **kwargs)  # type: ignore[arg-type]
        snapshot_paths.append(Path(name))
        return fd, name

    def interrupting_unlink(target: object) -> None:
        target_path = Path(str(target))
        unlink_attempts.append(target_path)
        real_unlink(target)
        if target_path in snapshot_paths:
            raise signal

    with monkeypatch.context() as context:
        context.setattr(profile_schema.tempfile, "mkstemp", tracked_mkstemp)
        context.setattr(profile_schema.os, "unlink", interrupting_unlink)
        with pytest.raises(exception_type) as caught:
            validate_profile_candidate(path)

    assert caught.value is signal
    assert unlink_attempts == snapshot_paths
    assert len(snapshot_paths) == 1
    assert not snapshot_paths[0].exists()
    assert list(private_directory.iterdir()) == []


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_candidate_connection_cleanup_control_flow_signal_wins_ordinary_body_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_type: type[BaseException],
) -> None:
    path = tmp_path / "candidate.sqlite3"
    connection = open_profile_store(path)
    connection.close()
    signal = exception_type("snapshot connection close control flow")
    private_directory = tmp_path / "private-snapshots"
    private_directory.mkdir()
    real_mkstemp = profile_schema.tempfile.mkstemp
    real_connect = sqlite3.connect
    snapshot_paths: list[Path] = []
    close_attempts: list[sqlite3.Connection] = []

    class InterruptingClose(sqlite3.Connection):
        def close(self) -> None:
            close_attempts.append(self)
            super().close()
            raise signal

    def tracked_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        kwargs["dir"] = private_directory
        fd, name = real_mkstemp(*args, **kwargs)  # type: ignore[arg-type]
        snapshot_paths.append(Path(name))
        return fd, name

    def tracked_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        kwargs["factory"] = InterruptingClose
        return real_connect(*args, **kwargs)  # type: ignore[arg-type]

    def fail_schema(_connection: sqlite3.Connection) -> None:
        raise ValueError("ordinary body failure")

    monkeypatch.setattr(profile_schema.tempfile, "mkstemp", tracked_mkstemp)
    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    monkeypatch.setattr(profile_schema, "_validate_schema", fail_schema)

    with pytest.raises(exception_type) as caught:
        validate_profile_candidate(path)

    assert caught.value is signal
    assert len(snapshot_paths) == 1
    # Two SQLite connections open against the disposable snapshot: the
    # brief read-write reopen (already current here, so it never migrates)
    # closes first, then the immutable read-only handle whose
    # `_validate_schema` call is the ordinary failure `fail_schema` raises.
    assert len(close_attempts) == 2
    for closed_connection in close_attempts:
        with pytest.raises(sqlite3.ProgrammingError):
            closed_connection.execute("SELECT 1")
    assert not snapshot_paths[0].exists()
    assert list(private_directory.iterdir()) == []


@pytest.mark.parametrize(
    ("body_mode", "expected_code"),
    [
        ("success", "schema_corrupt"),
        ("structured_error", "corrupt_data"),
        ("ordinary_error", "schema_corrupt"),
    ],
)
def test_candidate_ordinary_cleanup_failure_maps_without_detail_leaks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body_mode: str,
    expected_code: str,
) -> None:
    path = tmp_path / "candidate.sqlite3"
    connection = open_profile_store(path)
    if body_mode == "structured_error":
        row = encode_profile(_profile())
        row["revision"] = 0
        connection.execute(
            """
            INSERT INTO tts_generation_profiles VALUES (
                :profile_id, :display_name, :normalized_name, :provider_id,
                :model_id, :voice_id, :response_format, :speed, :options_json,
                :revision, :created_at, :updated_at
            )
            """,
            row,
        )
        connection.commit()
    connection.close()

    private_directory = tmp_path / "private-snapshots"
    private_directory.mkdir()
    real_mkstemp = profile_schema.tempfile.mkstemp
    real_unlink = profile_schema.os.unlink
    snapshot_paths: list[Path] = []

    def tracked_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        kwargs["dir"] = private_directory
        fd, name = real_mkstemp(*args, **kwargs)  # type: ignore[arg-type]
        snapshot_paths.append(Path(name))
        return fd, name

    def fail_after_unlink(target: object) -> None:
        real_unlink(target)
        raise RuntimeError("private cleanup detail")

    with monkeypatch.context() as context:
        context.setattr(profile_schema.tempfile, "mkstemp", tracked_mkstemp)
        context.setattr(profile_schema.os, "unlink", fail_after_unlink)
        if body_mode == "ordinary_error":

            def fail_schema(_connection: sqlite3.Connection) -> None:
                raise ValueError("private body detail")

            context.setattr(profile_schema, "_validate_schema", fail_schema)

        with _safe_error(expected_code) as caught:
            validate_profile_candidate(path)

    assert "private cleanup detail" not in str(caught.value)
    assert "private body detail" not in str(caught.value)
    assert len(snapshot_paths) == 1
    assert not snapshot_paths[0].exists()
    assert list(private_directory.iterdir()) == []


def test_candidate_rejects_invalid_domain_row(tmp_path: Path) -> None:
    path = tmp_path / "candidate.sqlite3"
    connection = open_profile_store(path)
    row = encode_profile(_profile())
    row["revision"] = 0
    connection.execute(
        """
        INSERT INTO tts_generation_profiles VALUES (
            :profile_id, :display_name, :normalized_name, :provider_id, :model_id,
            :voice_id, :response_format, :speed, :options_json, :revision,
            :created_at, :updated_at
        )
        """,
        row,
    )
    connection.commit()
    connection.close()
    before = _directory_snapshot(tmp_path)

    with _safe_error("corrupt_data"):
        validate_profile_candidate(path)

    assert _directory_snapshot(tmp_path) == before


def test_full_row_validator_rejects_domain_invalid_live_store_row(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    connection = open_profile_store(path)
    _insert_profile(connection, _profile())
    connection.execute("UPDATE tts_generation_profiles SET revision = 0")

    try:
        with _safe_error("corrupt_data"):
            profile_schema.validate_profile_store_rows(connection)
    finally:
        connection.close()


def test_full_row_validator_checks_deadline_before_each_profile_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    connection = open_profile_store(path)
    _insert_profile(connection, _profile())
    _insert_profile(
        connection,
        _profile(
            profile_id=UUID("11234567-89ab-4def-8123-456789abcdef"),
            display_name="Second",
            normalized_name="second",
        ),
    )
    checks = 0
    decoded = 0
    real_decode = profile_schema.decode_profile

    def check_deadline() -> None:
        nonlocal checks
        checks += 1
        if checks == 3:
            raise ProfileRepositoryError("restore_failed")

    def traced_decode(row: object) -> TTSGenerationProfile:
        nonlocal decoded
        decoded += 1
        return real_decode(row)  # type: ignore[arg-type]

    monkeypatch.setattr(profile_schema, "decode_profile", traced_decode)
    try:
        with _safe_error("restore_failed"):
            profile_schema.validate_profile_store_rows(
                connection,
                check_deadline=check_deadline,
            )
    finally:
        connection.close()

    assert checks == 3
    assert decoded == 1


def test_candidate_deadline_interrupts_private_snapshot_copy_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "candidate.sqlite3"
    connection = open_profile_store(path)
    _insert_profile(connection, _profile())
    connection.commit()
    connection.close()
    snapshot_paths: list[Path] = []
    inside_copy = False
    real_mkstemp = profile_schema.tempfile.mkstemp

    def recorded_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        descriptor, name = real_mkstemp(*args, **kwargs)  # type: ignore[arg-type]
        snapshot_paths.append(Path(name))
        return descriptor, name

    def check_deadline() -> None:
        if inside_copy:
            raise ProfileRepositoryError("restore_failed")

    def interrupted_copy(
        _source_fd: int,
        snapshot_fd: int,
        *,
        check_deadline: Callable[[], None] | None = None,
    ) -> None:
        nonlocal inside_copy
        del snapshot_fd
        assert check_deadline is not None
        inside_copy = True
        check_deadline()

    monkeypatch.setattr(profile_schema.tempfile, "mkstemp", recorded_mkstemp)
    monkeypatch.setattr(
        profile_schema,
        "_copy_source_to_snapshot",
        interrupted_copy,
    )

    with _safe_error("restore_failed"):
        validate_profile_candidate(path, check_deadline=check_deadline)

    assert len(snapshot_paths) == 1
    assert not snapshot_paths[0].exists()


def test_schema_quick_check_is_interrupted_by_deadline_progress(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    connection = open_profile_store(path)
    expired = False
    progress_handlers: list[tuple[object, int]] = []

    class ProgressProxy:
        progress_handler: Callable[[], int] | None = None

        def __getattr__(self, name: str) -> object:
            return getattr(connection, name)

        def set_progress_handler(
            self,
            handler: Callable[[], int] | None,
            opcode_interval: int,
        ) -> None:
            self.progress_handler = handler
            progress_handlers.append((handler, opcode_interval))

        def execute(
            self,
            statement: str,
            parameters: object = (),
        ) -> sqlite3.Cursor:
            nonlocal expired
            if statement.strip() == "PRAGMA quick_check":
                expired = True
                assert self.progress_handler is not None
                if self.progress_handler() != 0:
                    raise sqlite3.OperationalError
            return connection.execute(statement, parameters)

    def check_deadline() -> None:
        if expired:
            raise ProfileRepositoryError("restore_failed")

    try:
        with _safe_error("restore_failed"):
            profile_schema._validate_schema(
                ProgressProxy(),  # type: ignore[arg-type]
                check_deadline=check_deadline,
            )
    finally:
        connection.close()

    assert callable(progress_handlers[0][0])
    assert progress_handlers[0][1] > 0
    assert progress_handlers[-1] == (None, 0)


def test_candidate_ordinary_error_removes_private_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "candidate.sqlite3"
    connection = open_profile_store(path)
    row = encode_profile(_profile())
    row["revision"] = 0
    connection.execute(
        """
        INSERT INTO tts_generation_profiles VALUES (
            :profile_id, :display_name, :normalized_name, :provider_id, :model_id,
            :voice_id, :response_format, :speed, :options_json, :revision,
            :created_at, :updated_at
        )
        """,
        row,
    )
    connection.commit()
    connection.close()
    private_directory = tmp_path / "private-snapshots"
    private_directory.mkdir()
    real_mkstemp = profile_schema.tempfile.mkstemp
    snapshot_paths: list[Path] = []

    def tracked_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        kwargs["dir"] = private_directory
        fd, name = real_mkstemp(*args, **kwargs)  # type: ignore[arg-type]
        snapshot_paths.append(Path(name))
        return fd, name

    monkeypatch.setattr(profile_schema.tempfile, "mkstemp", tracked_mkstemp)

    with _safe_error("corrupt_data"):
        validate_profile_candidate(path)

    assert len(snapshot_paths) == 1
    assert not snapshot_paths[0].exists()
    assert list(private_directory.iterdir()) == []


def test_foreign_key_check_failure_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "candidate.sqlite3"
    connection = open_profile_store(path)
    connection.execute("PRAGMA foreign_keys = OFF")
    _insert_assignment(connection, _assignment())
    connection.commit()
    connection.close()

    with _safe_error("schema_corrupt"):
        validate_profile_candidate(path)


def test_quick_check_failure_maps_to_schema_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "candidate.sqlite3"
    connection = open_profile_store(path)
    connection.close()

    class QuickCheckFailure(sqlite3.Connection):
        def execute(self, sql: str, parameters=()):  # type: ignore[no-untyped-def]
            if sql == "PRAGMA quick_check":
                return super().execute("SELECT 'not ok'")
            return super().execute(sql, parameters)

    original_connect = sqlite3.connect

    def injected_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        kwargs["factory"] = QuickCheckFailure
        return original_connect(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sqlite3, "connect", injected_connect)
    with _safe_error("schema_corrupt"):
        validate_profile_candidate(path)
