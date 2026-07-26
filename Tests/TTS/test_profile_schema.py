"""Tests for the dedicated versioned TTS profile SQLite schema."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
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
        "provider_id": "openai",
        "model_id": "tts-1-hd",
        "voice_id": None,
        "response_format": "mp3",
        "speed": 1.25,
        "options": {"nested": {"items": [True, 2, 3.5, None]}, "é": "声"},
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


def test_empty_store_migrates_transactionally_and_is_configured(tmp_path: Path) -> None:
    path = tmp_path / "profiles.sqlite3"

    connection = open_profile_store(path)
    try:
        assert connection.row_factory is sqlite3.Row
        assert (
            connection.execute("PRAGMA user_version").fetchone()[0]
            == CURRENT_PROFILE_SCHEMA_VERSION
            == 1
        )
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert (
            connection.execute("PRAGMA busy_timeout").fetchone()[0] == BUSY_TIMEOUT_MS
        )
        assert set(MIGRATIONS) == {0}
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
    connection.execute("PRAGMA user_version = 2")
    connection.close()
    before = path.read_bytes()

    with _safe_error("schema_unsupported"):
        open_profile_store(path)

    assert path.read_bytes() == before


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


def test_partial_unique_index_does_not_satisfy_normalized_name_uniqueness(
    tmp_path: Path,
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
        "CREATE UNIQUE INDEX partial_normalized_name "
        "ON tts_generation_profiles(normalized_name) WHERE revision > 1"
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
    ["pk", "extra_pk", "index", "partial_index", "unique_index", "fk", "delete"],
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
    pk = (
        "PRIMARY KEY(authority_id, source, character_id)"
        if defect == "pk"
        else (
            "PRIMARY KEY(source, authority_id, character_id, tenant)"
            if defect == "extra_pk"
            else "PRIMARY KEY(source, authority_id, character_id)"
        )
    )
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
            "ON character_tts_assignments(profile_id)"
            + (" WHERE source = 'local'" if defect == "partial_index" else "")
        )
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
        assert canonical_json_options(decode_profile(profile_row).options) == (
            '{"nested":{"items":[true,2,3.5,null]},"é":"声"}'
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
    path = tmp_path / "candidate.sqlite3"
    connection = open_profile_store(path)
    _insert_profile(connection, _profile())
    _insert_assignment(connection, _assignment())
    connection.commit()
    connection.close()
    before = path.read_bytes()

    validate_profile_candidate(path)

    assert path.read_bytes() == before


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

    with _safe_error("corrupt_data"):
        validate_profile_candidate(path)


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
