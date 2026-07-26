"""SQLite schema, validation, and persistence codecs for TTS profiles.

Connections remain caller-owned.  The live opener configures and returns a
connection; candidate validation owns and always closes its read-only connection.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeAlias, cast
from uuid import UUID

from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_types import (
    AssignedTTSProfileSnapshot,
    CharacterRef,
    CharacterTTSAssignment,
    FrozenJsonOptions,
    JsonOptions,
    TTSGenerationProfile,
    TTSProfileDraft,
    canonical_json_options,
)

CURRENT_PROFILE_SCHEMA_VERSION = 1
BUSY_TIMEOUT_MS = 5_000
PROFILE_TABLE = "tts_generation_profiles"
ASSIGNMENT_TABLE = "character_tts_assignments"
ASSIGNMENT_PROFILE_INDEX = "idx_character_tts_assignments_profile_id"

PROFILE_COLUMNS = (
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
)
ASSIGNMENT_COLUMNS = (
    "source",
    "authority_id",
    "character_id",
    "profile_id",
    "created_at",
    "updated_at",
)

# These aliases are the persistence contract for joined assignment/profile rows.
# Every duplicate column name is qualified by its owning record.
JOINED_ASSIGNMENT_ALIASES = tuple(
    f"assignment_{column}" for column in ASSIGNMENT_COLUMNS
)
JOINED_PROFILE_ALIASES = tuple(f"profile_{column}" for column in PROFILE_COLUMNS)

ASSIGNED_PROFILE_JOIN_SELECT = """
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

RowLike: TypeAlias = sqlite3.Row | Mapping[str, object]


def _repository_error(code: str) -> ProfileRepositoryError:
    return ProfileRepositoryError(code)


def encode_uuid(value: UUID) -> str:
    """Encode an exact UUID domain value as canonical SQLite text."""

    if type(value) is not UUID:
        raise _repository_error("corrupt_data")
    return str(value)


def decode_uuid(value: object) -> UUID:
    """Decode canonical UUID text, failing closed for every other value."""

    try:
        if type(value) is not str:
            raise ValueError
        decoded = UUID(value)
        if str(decoded) != value:
            raise ValueError
        return decoded
    except Exception:
        raise _repository_error("corrupt_data") from None


def encode_utc_datetime(value: datetime) -> str:
    """Encode an exact UTC datetime using fixed-width ISO-8601 microseconds."""

    try:
        if type(value) is not datetime or value.tzinfo is None:
            raise ValueError
        offset = value.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError
        return (
            value.astimezone(UTC)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )
    except Exception:
        raise _repository_error("corrupt_data") from None


def decode_utc_datetime(value: object) -> datetime:
    """Decode only the canonical timestamp representation emitted above."""

    try:
        if type(value) is not str or not value.endswith("Z"):
            raise ValueError
        decoded = datetime.fromisoformat(f"{value[:-1]}+00:00")
        if encode_utc_datetime(decoded) != value:
            raise ValueError
        return decoded
    except Exception:
        raise _repository_error("corrupt_data") from None


def encode_options(options: JsonOptions) -> str:
    """Encode validated JSON options using the domain canonicalizer."""

    try:
        return canonical_json_options(options)
    except Exception:
        raise _repository_error("corrupt_data") from None


def decode_options(value: object) -> FrozenJsonOptions:
    """Decode, validate, freeze, and require canonical JSON object text."""

    try:
        if type(value) is not str:
            raise ValueError
        parsed = json.loads(
            value,
            parse_constant=lambda _constant: (_ for _ in ()).throw(ValueError()),
        )
        if type(parsed) is not dict:
            raise ValueError
        canonical = canonical_json_options(parsed)
        if canonical != value:
            raise ValueError
        # Reconstructing a draft/profile will freeze once more; returning the
        # domain-canonicalized mapping here also makes this helper independently safe.
        return _freeze_via_profile_options(parsed)
    except Exception:
        raise _repository_error("corrupt_data") from None


def _freeze_via_profile_options(options: Mapping[str, object]) -> FrozenJsonOptions:
    """Freeze options without duplicating Task 2's validation implementation."""

    # The public canonicalizer validates but returns text. JSON-decoding that text
    # gives fresh exact built-ins; the profile constructor below performs freezing.
    sentinel = TTSProfileDraft(
        display_name="Options",
        provider_id="openai",
        model_id="options",
        voice_id=None,
        response_format="mp3",
        speed=1.0,
        options=cast(JsonOptions, options),
    )
    return cast(FrozenJsonOptions, sentinel.options)


def encode_profile(profile: TTSGenerationProfile) -> dict[str, object]:
    """Encode an exact profile domain object to SQLite-bindable values."""

    if type(profile) is not TTSGenerationProfile:
        raise _repository_error("corrupt_data")
    return {
        "profile_id": encode_uuid(profile.profile_id),
        "display_name": profile.display_name,
        "normalized_name": profile.normalized_name,
        "provider_id": profile.provider_id,
        "model_id": profile.model_id,
        "voice_id": profile.voice_id,
        "response_format": profile.response_format,
        "speed": profile.speed,
        "options_json": encode_options(profile.options),
        "revision": profile.revision,
        "created_at": encode_utc_datetime(profile.created_at),
        "updated_at": encode_utc_datetime(profile.updated_at),
    }


def _row_value(row: RowLike, column: str) -> object:
    return row[column]


def _decode_profile(row: RowLike, prefix: str) -> TTSGenerationProfile:
    try:
        display_name = _row_value(row, f"{prefix}display_name")
        normalized_name = _row_value(row, f"{prefix}normalized_name")
        provider_id = _row_value(row, f"{prefix}provider_id")
        model_id = _row_value(row, f"{prefix}model_id")
        voice_id = _row_value(row, f"{prefix}voice_id")
        response_format = _row_value(row, f"{prefix}response_format")
        speed = _row_value(row, f"{prefix}speed")
        revision = _row_value(row, f"{prefix}revision")
        if not all(
            type(value) is str
            for value in (
                display_name,
                normalized_name,
                provider_id,
                model_id,
                response_format,
            )
        ):
            raise ValueError
        if voice_id is not None and type(voice_id) is not str:
            raise ValueError
        if type(speed) is not float or type(revision) is not int:
            raise ValueError
        return TTSGenerationProfile(
            profile_id=decode_uuid(_row_value(row, f"{prefix}profile_id")),
            display_name=cast(str, display_name),
            normalized_name=cast(str, normalized_name),
            provider_id=cast(str, provider_id),
            model_id=cast(str, model_id),
            voice_id=cast(str | None, voice_id),
            response_format=cast(str, response_format),
            speed=speed,
            options=decode_options(_row_value(row, f"{prefix}options_json")),
            revision=revision,
            created_at=decode_utc_datetime(_row_value(row, f"{prefix}created_at")),
            updated_at=decode_utc_datetime(_row_value(row, f"{prefix}updated_at")),
        )
    except Exception:
        raise _repository_error("corrupt_data") from None


def decode_profile(row: RowLike) -> TTSGenerationProfile:
    """Decode and fully revalidate one profile persistence row."""

    return _decode_profile(row, "")


def encode_assignment(
    assignment: CharacterTTSAssignment,
    *,
    created_at: datetime,
    updated_at: datetime,
) -> dict[str, object]:
    """Encode an assignment and its separate persistence timestamps."""

    if type(assignment) is not CharacterTTSAssignment:
        raise _repository_error("corrupt_data")
    created = encode_utc_datetime(created_at)
    updated = encode_utc_datetime(updated_at)
    if created_at > updated_at:
        raise _repository_error("corrupt_data")
    return {
        "source": assignment.character_ref.source,
        "authority_id": assignment.character_ref.authority_id,
        "character_id": assignment.character_ref.character_id,
        "profile_id": encode_uuid(assignment.profile_id),
        "created_at": created,
        "updated_at": updated,
    }


def _decode_assignment(row: RowLike, prefix: str) -> CharacterTTSAssignment:
    try:
        source = _row_value(row, f"{prefix}source")
        authority_id = _row_value(row, f"{prefix}authority_id")
        character_id = _row_value(row, f"{prefix}character_id")
        if not all(
            type(value) is str for value in (source, authority_id, character_id)
        ):
            raise ValueError
        created_at = decode_utc_datetime(_row_value(row, f"{prefix}created_at"))
        updated_at = decode_utc_datetime(_row_value(row, f"{prefix}updated_at"))
        if created_at > updated_at:
            raise ValueError
        return CharacterTTSAssignment(
            character_ref=CharacterRef(
                source=cast(Literal["local", "server"], source),
                authority_id=cast(str, authority_id),
                character_id=cast(str, character_id),
            ),
            profile_id=decode_uuid(_row_value(row, f"{prefix}profile_id")),
        )
    except Exception:
        raise _repository_error("corrupt_data") from None


def decode_assignment(row: RowLike) -> CharacterTTSAssignment:
    """Decode and fully revalidate one assignment persistence row."""

    return _decode_assignment(row, "")


def decode_assigned_snapshot(row: RowLike) -> AssignedTTSProfileSnapshot:
    """Decode a joined row using the documented deterministic aliases."""

    try:
        return AssignedTTSProfileSnapshot(
            assignment=_decode_assignment(row, "assignment_"),
            profile=_decode_profile(row, "profile_"),
        )
    except Exception:
        raise _repository_error("corrupt_data") from None


def _migrate_v0_to_v1(connection: sqlite3.Connection) -> None:
    """Create schema version 1 inside the caller's active transaction."""

    connection.execute(
        """
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
    )
    connection.execute(
        """
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
    )
    connection.execute(
        f"CREATE INDEX {ASSIGNMENT_PROFILE_INDEX} "
        "ON character_tts_assignments(profile_id)"
    )


MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {0: _migrate_v0_to_v1}


def _configure_connection(connection: sqlite3.Connection) -> None:
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        raise _repository_error("schema_corrupt")
    connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    if connection.execute("PRAGMA busy_timeout").fetchone()[0] != BUSY_TIMEOUT_MS:
        raise _repository_error("schema_corrupt")


def _user_tables(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            """
            SELECT name FROM sqlite_schema
            WHERE type = 'table' AND name NOT GLOB 'sqlite_*'
            """
        )
    }


def _table_info(connection: sqlite3.Connection, table: str) -> dict[str, sqlite3.Row]:
    return {
        row["name"]: row for row in connection.execute(f"PRAGMA table_info({table})")
    }


def _index_columns(connection: sqlite3.Connection, index: str) -> list[str]:
    return [row["name"] for row in connection.execute(f"PRAGMA index_info({index})")]


def _validate_schema(connection: sqlite3.Connection) -> None:
    """Validate every required structural and integrity invariant for v1."""

    try:
        if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
            raise ValueError
        if _user_tables(connection) < {PROFILE_TABLE, ASSIGNMENT_TABLE}:
            raise ValueError

        expected_profile = {
            "profile_id": ("TEXT", 0, 1),
            "display_name": ("TEXT", 1, 0),
            "normalized_name": ("TEXT", 1, 0),
            "provider_id": ("TEXT", 1, 0),
            "model_id": ("TEXT", 1, 0),
            "voice_id": ("TEXT", 0, 0),
            "response_format": ("TEXT", 1, 0),
            "speed": ("REAL", 1, 0),
            "options_json": ("TEXT", 1, 0),
            "revision": ("INTEGER", 1, 0),
            "created_at": ("TEXT", 1, 0),
            "updated_at": ("TEXT", 1, 0),
        }
        profile_info = _table_info(connection, PROFILE_TABLE)
        for name, (declared_type, not_null, pk) in expected_profile.items():
            row = profile_info.get(name)
            if row is None or (row["type"].upper(), row["notnull"], row["pk"]) != (
                declared_type,
                not_null,
                pk,
            ):
                raise ValueError
        if [row["name"] for row in profile_info.values() if row["pk"]] != [
            "profile_id"
        ]:
            raise ValueError

        unique_normalized = False
        for row in connection.execute(f"PRAGMA index_list({PROFILE_TABLE})"):
            if (
                row["unique"] == 1
                and row["partial"] == 0
                and _index_columns(connection, row["name"]) == ["normalized_name"]
            ):
                unique_normalized = True
        if not unique_normalized:
            raise ValueError

        expected_assignment = {
            "source": ("TEXT", 1, 1),
            "authority_id": ("TEXT", 1, 2),
            "character_id": ("TEXT", 1, 3),
            "profile_id": ("TEXT", 1, 0),
            "created_at": ("TEXT", 1, 0),
            "updated_at": ("TEXT", 1, 0),
        }
        assignment_info = _table_info(connection, ASSIGNMENT_TABLE)
        for name, (declared_type, not_null, pk) in expected_assignment.items():
            row = assignment_info.get(name)
            if row is None or (row["type"].upper(), row["notnull"], row["pk"]) != (
                declared_type,
                not_null,
                pk,
            ):
                raise ValueError
        assignment_pk = sorted(
            (row["pk"], row["name"]) for row in assignment_info.values() if row["pk"]
        )
        if [name for _position, name in assignment_pk] != [
            "source",
            "authority_id",
            "character_id",
        ]:
            raise ValueError

        assignment_indexes = {
            row["name"]: row
            for row in connection.execute(f"PRAGMA index_list({ASSIGNMENT_TABLE})")
        }
        profile_index = assignment_indexes.get(ASSIGNMENT_PROFILE_INDEX)
        if (
            profile_index is None
            or profile_index["origin"] != "c"
            or profile_index["partial"] != 0
            or profile_index["unique"] != 0
            or _index_columns(connection, ASSIGNMENT_PROFILE_INDEX) != ["profile_id"]
        ):
            raise ValueError

        foreign_keys = list(
            connection.execute(f"PRAGMA foreign_key_list({ASSIGNMENT_TABLE})")
        )
        if len(foreign_keys) != 1:
            raise ValueError
        foreign_key = foreign_keys[0]
        if (
            foreign_key["table"],
            foreign_key["from"],
            foreign_key["to"],
            foreign_key["on_delete"],
        ) != (PROFILE_TABLE, "profile_id", "profile_id", "RESTRICT"):
            raise ValueError

        quick_check = [row[0] for row in connection.execute("PRAGMA quick_check")]
        if quick_check != ["ok"]:
            raise ValueError
        if list(connection.execute("PRAGMA foreign_key_check")):
            raise ValueError
    except ProfileRepositoryError:
        raise
    except Exception:
        raise _repository_error("schema_corrupt") from None


def _migrate_empty_store(connection: sqlite3.Connection) -> None:
    try:
        connection.execute("BEGIN IMMEDIATE")
        version = 0
        while version < CURRENT_PROFILE_SCHEMA_VERSION:
            migration = MIGRATIONS.get(version)
            if migration is None:
                raise RuntimeError
            migration(connection)
            version += 1
            connection.execute(f"PRAGMA user_version = {version}")
        connection.commit()
    except Exception:
        try:
            connection.rollback()
        except Exception:
            pass
        raise _repository_error("migration_failed") from None


def open_profile_store(path: Path) -> sqlite3.Connection:
    """Open/configure a live store, migrating only a truly empty v0 database."""

    connection: sqlite3.Connection | None = None
    try:
        if not isinstance(path, Path):
            raise _repository_error("operation_failed")
        connection = sqlite3.connect(path, isolation_level=None)
        _configure_connection(connection)
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if type(version) is not int:
            raise _repository_error("schema_corrupt")
        if version > CURRENT_PROFILE_SCHEMA_VERSION:
            raise _repository_error("schema_unsupported")
        if version == 0:
            if _user_tables(connection):
                raise _repository_error("schema_partial")
            journal_mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if journal_mode != "wal":
                raise _repository_error("schema_corrupt")
            _migrate_empty_store(connection)
        elif version != CURRENT_PROFILE_SCHEMA_VERSION:
            raise _repository_error("schema_unsupported")
        else:
            _validate_schema(connection)
            journal_mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if journal_mode != "wal":
                raise _repository_error("schema_corrupt")
        _validate_schema(connection)
        return connection
    except ProfileRepositoryError:
        if connection is not None:
            connection.close()
        raise
    except Exception:
        if connection is not None:
            connection.close()
        raise _repository_error("schema_corrupt") from None


def _validate_all_rows(connection: sqlite3.Connection) -> None:
    try:
        for row in connection.execute(f"SELECT * FROM {PROFILE_TABLE}"):
            decode_profile(row)
        for row in connection.execute(f"SELECT * FROM {ASSIGNMENT_TABLE}"):
            decode_assignment(row)
        for row in connection.execute(ASSIGNED_PROFILE_JOIN_SELECT):
            decode_assigned_snapshot(row)
    except ProfileRepositoryError:
        raise
    except Exception:
        raise _repository_error("corrupt_data") from None


def validate_profile_candidate(path: Path) -> None:
    """Validate an existing v1 candidate without writing or migrating it."""

    if not isinstance(path, Path) or not path.is_file():
        raise _repository_error("missing")
    connection: sqlite3.Connection | None = None
    try:
        uri = f"{path.resolve().as_uri()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        _configure_connection(connection)
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version != CURRENT_PROFILE_SCHEMA_VERSION:
            raise _repository_error("schema_unsupported")
        _validate_schema(connection)
        _validate_all_rows(connection)
    except ProfileRepositoryError:
        raise
    except Exception:
        raise _repository_error("schema_corrupt") from None
    finally:
        if connection is not None:
            connection.close()
