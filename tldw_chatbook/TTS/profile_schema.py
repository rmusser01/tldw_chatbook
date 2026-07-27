"""SQLite schema, validation, and persistence codecs for TTS profiles.

Connections remain caller-owned.  The live opener configures and returns a
connection; candidate validation owns and always closes its read-only connection.
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import tempfile
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

_PROFILE_TABLE_DDL = """
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
_ASSIGNMENT_TABLE_DDL = """
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
_ASSIGNMENT_PROFILE_INDEX_DDL = (
    f"CREATE INDEX {ASSIGNMENT_PROFILE_INDEX} ON character_tts_assignments(profile_id)"
)

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

    connection.execute(_PROFILE_TABLE_DDL)
    connection.execute(_ASSIGNMENT_TABLE_DDL)
    connection.execute(_ASSIGNMENT_PROFILE_INDEX_DDL)


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


def _normalized_ddl(sql: str) -> str:
    return " ".join(sql.split())


def _validate_owned_schema_sql(connection: sqlite3.Connection) -> None:
    expected = {
        ("table", PROFILE_TABLE): _normalized_ddl(_PROFILE_TABLE_DDL),
        ("table", ASSIGNMENT_TABLE): _normalized_ddl(_ASSIGNMENT_TABLE_DDL),
        ("index", ASSIGNMENT_PROFILE_INDEX): _normalized_ddl(
            _ASSIGNMENT_PROFILE_INDEX_DDL
        ),
    }
    actual: dict[tuple[str, str], str] = {}
    for row in connection.execute(
        """
        SELECT type, name, sql
        FROM sqlite_schema
        WHERE name NOT GLOB 'sqlite_*'
        """
    ):
        if (
            type(row["type"]) is not str
            or type(row["name"]) is not str
            or type(row["sql"]) is not str
        ):
            raise ValueError
        actual[(row["type"], row["name"])] = _normalized_ddl(row["sql"])
    if actual != expected:
        raise ValueError


def _table_xinfo_manifest(
    connection: sqlite3.Connection, table: str
) -> list[tuple[int, str, str, int, object, int, int]]:
    return [
        (
            row["cid"],
            row["name"],
            row["type"],
            row["notnull"],
            row["dflt_value"],
            row["pk"],
            row["hidden"],
        )
        for row in connection.execute(f"PRAGMA table_xinfo({table})")
    ]


def _has_exact_binary_index_keys(
    connection: sqlite3.Connection, index: str, columns: tuple[str, ...]
) -> bool:
    key_rows = [
        row
        for row in connection.execute(f"PRAGMA index_xinfo({index})")
        if row["key"] == 1
    ]
    return [(row["name"], row["desc"], row["coll"]) for row in key_rows] == [
        (column, 0, "BINARY") for column in columns
    ]


def _has_exact_primary_key_index(
    connection: sqlite3.Connection, table: str, columns: tuple[str, ...]
) -> bool:
    primary_indexes = [
        row
        for row in connection.execute(f"PRAGMA index_list({table})")
        if row["origin"] == "pk"
    ]
    return (
        len(primary_indexes) == 1
        and primary_indexes[0]["unique"] == 1
        and primary_indexes[0]["partial"] == 0
        and _has_exact_binary_index_keys(
            connection, primary_indexes[0]["name"], columns
        )
    )


def _validate_schema(connection: sqlite3.Connection) -> None:
    """Validate every required structural and integrity invariant for v1."""

    try:
        if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
            raise ValueError
        if _user_tables(connection) != {PROFILE_TABLE, ASSIGNMENT_TABLE}:
            raise ValueError
        _validate_owned_schema_sql(connection)

        if _table_xinfo_manifest(connection, PROFILE_TABLE) != [
            (0, "profile_id", "TEXT", 0, None, 1, 0),
            (1, "display_name", "TEXT", 1, None, 0, 0),
            (2, "normalized_name", "TEXT", 1, None, 0, 0),
            (3, "provider_id", "TEXT", 1, None, 0, 0),
            (4, "model_id", "TEXT", 1, None, 0, 0),
            (5, "voice_id", "TEXT", 0, None, 0, 0),
            (6, "response_format", "TEXT", 1, None, 0, 0),
            (7, "speed", "REAL", 1, None, 0, 0),
            (8, "options_json", "TEXT", 1, None, 0, 0),
            (9, "revision", "INTEGER", 1, None, 0, 0),
            (10, "created_at", "TEXT", 1, None, 0, 0),
            (11, "updated_at", "TEXT", 1, None, 0, 0),
        ]:
            raise ValueError
        if not _has_exact_primary_key_index(connection, PROFILE_TABLE, ("profile_id",)):
            raise ValueError

        profile_indexes = list(
            connection.execute(f"PRAGMA index_list({PROFILE_TABLE})")
        )
        normalized_indexes = [row for row in profile_indexes if row["origin"] == "u"]
        if (
            len(profile_indexes) != 2
            or len(normalized_indexes) != 1
            or normalized_indexes[0]["unique"] != 1
            or normalized_indexes[0]["partial"] != 0
            or not _has_exact_binary_index_keys(
                connection, normalized_indexes[0]["name"], ("normalized_name",)
            )
        ):
            raise ValueError
        if list(connection.execute(f"PRAGMA foreign_key_list({PROFILE_TABLE})")):
            raise ValueError

        if _table_xinfo_manifest(connection, ASSIGNMENT_TABLE) != [
            (0, "source", "TEXT", 1, None, 1, 0),
            (1, "authority_id", "TEXT", 1, None, 2, 0),
            (2, "character_id", "TEXT", 1, None, 3, 0),
            (3, "profile_id", "TEXT", 1, None, 0, 0),
            (4, "created_at", "TEXT", 1, None, 0, 0),
            (5, "updated_at", "TEXT", 1, None, 0, 0),
        ]:
            raise ValueError
        if not _has_exact_primary_key_index(
            connection,
            ASSIGNMENT_TABLE,
            ("source", "authority_id", "character_id"),
        ):
            raise ValueError

        assignment_index_rows = list(
            connection.execute(f"PRAGMA index_list({ASSIGNMENT_TABLE})")
        )
        assignment_indexes = {row["name"]: row for row in assignment_index_rows}
        profile_index = assignment_indexes.get(ASSIGNMENT_PROFILE_INDEX)
        if (
            len(assignment_index_rows) != 2
            or profile_index is None
            or profile_index["origin"] != "c"
            or profile_index["partial"] != 0
            or profile_index["unique"] != 0
            or not _has_exact_binary_index_keys(
                connection, ASSIGNMENT_PROFILE_INDEX, ("profile_id",)
            )
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
    except BaseException as error:
        try:
            connection.rollback()
        except BaseException:
            pass
        if not isinstance(error, Exception):
            raise
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
    except BaseException as error:
        if connection is not None:
            try:
                connection.close()
            except BaseException:
                pass
        if isinstance(error, ProfileRepositoryError):
            raise
        if isinstance(error, Exception):
            raise _repository_error("schema_corrupt") from None
        raise


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


def _source_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _candidate_sidecars(resolved_path: Path) -> tuple[Path, ...]:
    return tuple(
        resolved_path.with_name(f"{resolved_path.name}{suffix}")
        for suffix in ("-wal", "-shm", "-journal")
    )


def _sidecars_absent(resolved_path: Path) -> bool:
    return not any(
        os.path.lexists(sidecar) for sidecar in _candidate_sidecars(resolved_path)
    )


def _source_is_unchanged(
    source_fd: int,
    resolved_path: Path,
    source_identity: tuple[int, ...],
    directory_identity: tuple[int, ...],
) -> bool:
    return (
        _source_identity(os.fstat(source_fd)) == source_identity
        and _source_identity(os.stat(resolved_path)) == source_identity
        and _directory_identity(os.stat(resolved_path.parent)) == directory_identity
        and _sidecars_absent(resolved_path)
    )


def _snapshot_is_unchanged(
    snapshot_fd: int,
    snapshot_path: str,
    snapshot_identity: tuple[int, ...],
) -> bool:
    return (
        _source_identity(os.fstat(snapshot_fd)) == snapshot_identity
        and _source_identity(os.lstat(snapshot_path)) == snapshot_identity
    )


def _copy_source_to_snapshot(source_fd: int, snapshot_fd: int) -> None:
    while chunk := os.read(source_fd, 1024 * 1024):
        offset = 0
        while offset < len(chunk):
            written = os.write(snapshot_fd, chunk[offset:])
            if written <= 0:
                raise OSError
            offset += written
    os.fsync(snapshot_fd)


def _close_suppressing_errors(resource: object) -> bool:
    try:
        close = getattr(resource, "close")
        close()
        return True
    except BaseException:
        return False


def validate_profile_candidate(path: Path) -> None:
    """Validate a point-in-time private snapshot of a standalone v1 backup.

    A later restore must validate its own repository-controlled staged snapshot;
    a successful path validation is never an authorization to trust future bytes.
    """

    if not isinstance(path, Path):
        raise _repository_error("missing")
    try:
        resolved_path = path.resolve(strict=True)
    except FileNotFoundError:
        raise _repository_error("missing") from None
    except Exception:
        raise _repository_error("schema_corrupt") from None
    if not resolved_path.is_file():
        raise _repository_error("missing")
    try:
        directory_state = _directory_identity(os.stat(resolved_path.parent))
        if not _sidecars_absent(resolved_path):
            raise _repository_error("schema_corrupt")
    except ProfileRepositoryError:
        raise
    except Exception:
        raise _repository_error("schema_corrupt") from None

    source_fd: int | None = None
    snapshot_fd: int | None = None
    snapshot_path: str | None = None
    connection: sqlite3.Connection | None = None
    completed = False
    try:
        path_state = _source_identity(os.stat(resolved_path))
        if not stat.S_ISREG(path_state[2]):
            raise ValueError

        source_flags = (
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        source_flags |= getattr(os, "O_NOFOLLOW", 0)
        source_fd = os.open(resolved_path, source_flags)
        source_state = _source_identity(os.fstat(source_fd))
        if source_state != path_state or not _source_is_unchanged(
            source_fd,
            resolved_path,
            source_state,
            directory_state,
        ):
            raise ValueError

        snapshot_fd, snapshot_path = tempfile.mkstemp(
            prefix="tldw-tts-profile-candidate-",
            suffix=".sqlite3",
        )
        os.fchmod(snapshot_fd, 0o600)
        _copy_source_to_snapshot(source_fd, snapshot_fd)
        snapshot_state = _source_identity(os.fstat(snapshot_fd))
        if (
            snapshot_state[3] != source_state[3]
            or not stat.S_ISREG(snapshot_state[2])
            or stat.S_IMODE(snapshot_state[2]) != 0o600
            or not _snapshot_is_unchanged(
                snapshot_fd,
                snapshot_path,
                snapshot_state,
            )
            or not _source_is_unchanged(
                source_fd,
                resolved_path,
                source_state,
                directory_state,
            )
        ):
            raise ValueError

        snapshot_uri = f"{Path(snapshot_path).resolve().as_uri()}?mode=ro&immutable=1"
        connection = sqlite3.connect(snapshot_uri, uri=True, isolation_level=None)
        if not _snapshot_is_unchanged(
            snapshot_fd,
            snapshot_path,
            snapshot_state,
        ):
            raise ValueError
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        _configure_connection(connection)
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version != CURRENT_PROFILE_SCHEMA_VERSION:
            raise _repository_error("schema_unsupported")
        _validate_schema(connection)
        _validate_all_rows(connection)
        connection.close()
        connection = None
        if not _snapshot_is_unchanged(
            snapshot_fd,
            snapshot_path,
            snapshot_state,
        ) or not _source_is_unchanged(
            source_fd,
            resolved_path,
            source_state,
            directory_state,
        ):
            raise ValueError
        completed = True
    except ProfileRepositoryError:
        raise
    except Exception:
        raise _repository_error("schema_corrupt") from None
    finally:
        cleanup_succeeded = True
        if connection is not None:
            cleanup_succeeded &= _close_suppressing_errors(connection)
        if snapshot_fd is not None:
            try:
                os.close(snapshot_fd)
            except BaseException:
                cleanup_succeeded = False
        if source_fd is not None:
            try:
                os.close(source_fd)
            except BaseException:
                cleanup_succeeded = False
        if snapshot_path is not None:
            try:
                os.unlink(snapshot_path)
            except FileNotFoundError:
                pass
            except BaseException:
                cleanup_succeeded = False
        if completed and not cleanup_succeeded:
            raise _repository_error("schema_corrupt") from None
