"""Shared stdlib-only schema, domain codecs, and metadata proof for TTS stores."""

from __future__ import annotations

import hashlib
import sqlite3
import struct
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from json import loads as _json_loads
from typing import Any, Literal, TypeAlias, cast
from uuid import UUID

from tldw_chatbook.DB.sql_identifier_core import escape_identifier, validate_identifier
from tldw_chatbook.TTS.migrations.v0_to_v1 import (
    ASSIGNMENT_PROFILE_INDEX_DDL as _ASSIGNMENT_PROFILE_INDEX_DDL,
)
from tldw_chatbook.TTS.migrations.v0_to_v1 import (
    ASSIGNMENT_TABLE_DDL as _ASSIGNMENT_TABLE_DDL,
)
from tldw_chatbook.TTS.migrations.v0_to_v1 import (
    PROFILE_TABLE_DDL as _PROFILE_TABLE_DDL,
)
from tldw_chatbook.TTS.migrations.v2_to_v3 import (
    REFERENCE_ID_INDEX as _REFERENCE_ID_INDEX,
)
from tldw_chatbook.TTS.migrations.v2_to_v3 import (
    REFERENCE_ID_INDEX_DDL as _REFERENCE_ID_INDEX_DDL,
)
from tldw_chatbook.TTS.migrations.v2_to_v3 import (
    REFERENCE_TABLE as _REFERENCE_TABLE,
)
from tldw_chatbook.TTS.migrations.v2_to_v3 import (
    REFERENCE_TABLE_DDL as _REFERENCE_TABLE_DDL,
)
from tldw_chatbook.TTS.migrations.v3_to_v4 import (
    REFERENCE_ID_INDEX_DDL as _V4_REFERENCE_ID_INDEX_DDL,
)
from tldw_chatbook.TTS.migrations.v3_to_v4 import (
    REFERENCE_TABLE_DDL as _V4_REFERENCE_TABLE_DDL,
)
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_types import (
    AssignedTTSProfileSnapshot,
    CharacterRef,
    CharacterTTSAssignment,
    FrozenJsonOptions,
    JsonOptions,
    TTSGenerationProfile,
    _freeze_options,
    canonical_json_options,
)

CURRENT_PROFILE_SCHEMA_VERSION = 4
BUSY_TIMEOUT_MS = 5_000
_DEADLINE_PROGRESS_OPCODE_INTERVAL = 1_000
_MAX_PERSISTED_DISPLAY_NAME_CHARACTERS = 128
_MAX_PERSISTED_RESPONSE_FORMAT_CHARACTERS = 32
_MAX_PERSISTED_OPTIONS_BYTES = 16 * 1024
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
LEFT JOIN tts_generation_profiles AS p ON p.profile_id = a.profile_id
"""

RowLike: TypeAlias = sqlite3.Row | Mapping[str, object]

_MAX_EXACT_METADATA_ROWS = 1_000_000


def _repository_error(code: str) -> ProfileRepositoryError:
    return ProfileRepositoryError(code)


def _update_metadata_digest(digest: Any, value: object) -> None:
    """Length-frame one SQLite scalar into an incremental digest."""

    if value is None:
        payload = b""
        tag = b"n"
    elif type(value) is int:
        payload = str(value).encode("ascii")
        tag = b"i"
    elif type(value) is float:
        payload = struct.pack(">d", value)
        tag = b"f"
    elif type(value) is str:
        payload = value.encode("utf-8")
        tag = b"s"
    else:
        raise _repository_error("corrupt_data")
    digest.update(tag)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _stream_exact_store_metadata_evidence(
    connection: sqlite3.Connection,
) -> tuple[bytes, tuple[int, int, int]]:
    """Stream bounded exact metadata evidence without retaining rows or blobs."""

    statements = (
        """
        SELECT profile_id, display_name, normalized_name, provider_id,
               model_id, voice_id, response_format, speed, options_json,
               revision, created_at, updated_at
        FROM tts_generation_profiles
        ORDER BY profile_id
        """,
        """
        SELECT source, authority_id, character_id, profile_id,
               created_at, updated_at
        FROM character_tts_assignments
        ORDER BY source, authority_id, character_id
        """,
        f"""
        SELECT profile_id, reference_id, sha256, byte_length,
               length(wav_bytes), length(CAST(reference_text AS BLOB)),
               duration_ms, sample_rate_hz, channels, sample_encoding,
               created_at, updated_at, recipe_id, recipe_revision
        FROM {_REFERENCE_TABLE}
        ORDER BY profile_id
        """,
    )
    digest = hashlib.sha256()
    counts: list[int] = []
    total = 0
    for table_index, statement in enumerate(statements):
        count = 0
        digest.update(b"t" + table_index.to_bytes(1, "big"))
        for row in connection.execute(statement):
            count += 1
            total += 1
            if total > _MAX_EXACT_METADATA_ROWS:
                raise _repository_error("corrupt_data")
            digest.update(b"r")
            for value in row:
                _update_metadata_digest(digest, value)
        counts.append(count)
    return digest.digest(), (counts[0], counts[1], counts[2])


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
    except Exception:  # noqa: BLE001 - normalize hostile persisted values without source text
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
    except Exception:  # noqa: BLE001 - normalize hostile persisted values without source text
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
    except Exception:  # noqa: BLE001 - normalize hostile persisted values without source text
        raise _repository_error("corrupt_data") from None


def decode_options(value: object) -> FrozenJsonOptions:
    """Decode, validate, freeze, and require canonical JSON object text."""

    try:
        if type(value) is not str:
            raise ValueError
        if len(value.encode("utf-8")) > _MAX_PERSISTED_OPTIONS_BYTES:
            raise ValueError
        parsed = _json_loads(
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
    except Exception:  # noqa: BLE001 - normalize hostile persisted values without source text
        raise _repository_error("corrupt_data") from None


def _freeze_via_profile_options(options: Mapping[str, object]) -> FrozenJsonOptions:
    """Freeze options without duplicating Task 2's validation implementation."""

    # Manually freeze the provided options using internal freezing logic
    return _freeze_options(cast(JsonOptions, options))


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
        display_name = cast(str, display_name)
        response_format = cast(str, response_format)
        if (
            len(display_name) > _MAX_PERSISTED_DISPLAY_NAME_CHARACTERS
            or len(response_format) > _MAX_PERSISTED_RESPONSE_FORMAT_CHARACTERS
        ):
            raise ValueError
        profile = TTSGenerationProfile(
            profile_id=decode_uuid(_row_value(row, f"{prefix}profile_id")),
            display_name=display_name,
            normalized_name=cast(str, normalized_name),
            provider_id=cast(str, provider_id),
            model_id=cast(str, model_id),
            voice_id=cast(str | None, voice_id),
            response_format=response_format,
            speed=speed,
            options=decode_options(_row_value(row, f"{prefix}options_json")),
            revision=revision,
            created_at=decode_utc_datetime(_row_value(row, f"{prefix}created_at")),
            updated_at=decode_utc_datetime(_row_value(row, f"{prefix}updated_at")),
        )
        if (
            profile.display_name != display_name
            or profile.response_format != response_format
        ):
            raise ValueError
        return profile
    except Exception:  # noqa: BLE001 - normalize hostile persisted values without source text
        raise _repository_error("corrupt_data") from None


def decode_profile(row: RowLike) -> TTSGenerationProfile:
    """Decode and fully revalidate one profile persistence row."""

    return _decode_profile(row, "")


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
    except Exception:  # noqa: BLE001 - normalize hostile persisted values without source text
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
    except Exception:  # noqa: BLE001 - normalize hostile persisted values without source text
        raise _repository_error("corrupt_data") from None


def _configure_connection(connection: sqlite3.Connection) -> None:
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        raise _repository_error("schema_corrupt")
    connection.execute("PRAGMA busy_timeout = 5000")
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


def _user_schema_objects(connection: sqlite3.Connection) -> set[tuple[str, str]]:
    return {
        (row[0], row[1])
        for row in connection.execute(
            """
            SELECT type, name FROM sqlite_schema
            WHERE lower(substr(name, 1, 7)) != 'sqlite_'
            """
        )
    }


def _normalized_ddl(sql: str) -> str:
    return " ".join(sql.split())


def _validated_quoted_identifier(identifier: object, identifier_kind: str) -> str:
    if type(identifier) is not str:
        raise ValueError
    exact_identifier = cast(str, identifier)
    if not validate_identifier(exact_identifier, identifier_kind):
        raise ValueError
    return escape_identifier(exact_identifier)


def _validate_owned_schema_sql(
    connection: sqlite3.Connection, *, schema_version: int
) -> None:
    expected = {
        ("table", PROFILE_TABLE): _normalized_ddl(_PROFILE_TABLE_DDL),
        ("table", ASSIGNMENT_TABLE): _normalized_ddl(_ASSIGNMENT_TABLE_DDL),
        ("index", ASSIGNMENT_PROFILE_INDEX): _normalized_ddl(
            _ASSIGNMENT_PROFILE_INDEX_DDL
        ),
    }
    if schema_version in (3, 4):
        expected[("table", _REFERENCE_TABLE)] = _normalized_ddl(
            _REFERENCE_TABLE_DDL if schema_version == 3 else _V4_REFERENCE_TABLE_DDL
        )
        expected[("index", _REFERENCE_ID_INDEX)] = _normalized_ddl(
            _REFERENCE_ID_INDEX_DDL
            if schema_version == 3
            else _V4_REFERENCE_ID_INDEX_DDL
        )
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
    quoted_table = _validated_quoted_identifier(table, "table name")
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
        for row in connection.execute(f"PRAGMA table_xinfo({quoted_table})")
    ]


def _has_exact_binary_index_keys(
    connection: sqlite3.Connection, index: str, columns: tuple[str, ...]
) -> bool:
    quoted_index = _validated_quoted_identifier(index, "index name")
    key_rows = [
        row
        for row in connection.execute(f"PRAGMA index_xinfo({quoted_index})")
        if row["key"] == 1
    ]
    return [(row["name"], row["desc"], row["coll"]) for row in key_rows] == [
        (column, 0, "BINARY") for column in columns
    ]


def _has_exact_primary_key_index(
    connection: sqlite3.Connection, table: str, columns: tuple[str, ...]
) -> bool:
    quoted_table = _validated_quoted_identifier(table, "table name")
    primary_indexes = [
        row
        for row in connection.execute(f"PRAGMA index_list({quoted_table})")
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


def _run_with_deadline_progress(
    connection: sqlite3.Connection,
    check_deadline: Callable[[], None] | None,
    operation: Callable[[], None],
) -> None:
    """Run SQLite work with cooperative deadline interruption when requested."""

    if check_deadline is None:
        operation()
        return

    callback_error: BaseException | None = None
    body_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    progress_installed = False

    def interrupt_after_deadline() -> int:
        nonlocal callback_error
        try:
            check_deadline()
        except BaseException as error:  # noqa: BLE001 - preserve control flow through progress-hook cleanup
            callback_error = error
            return 1
        return 0

    try:
        check_deadline()
        connection.set_progress_handler(
            interrupt_after_deadline,
            _DEADLINE_PROGRESS_OPCODE_INTERVAL,
        )
        progress_installed = True
        operation()
        check_deadline()
    except BaseException as error:  # noqa: BLE001 - preserve control flow through progress-hook cleanup
        body_error = error

    if progress_installed:
        try:
            connection.set_progress_handler(None, 0)
        except BaseException as error:  # noqa: BLE001 - preserve control flow through progress-hook cleanup
            cleanup_error = error

    if callback_error is not None:
        body_error = callback_error
    for candidate_error in (body_error, cleanup_error):
        if candidate_error is not None and not isinstance(candidate_error, Exception):
            raise candidate_error
    if cleanup_error is not None:
        raise cleanup_error
    if body_error is not None:
        raise body_error


def _validate_schema(
    connection: sqlite3.Connection,
    *,
    expected_version: int | None = None,
    check_deadline: Callable[[], None] | None = None,
) -> None:
    """Validate every required structural and integrity invariant.

    Versions one and two share the legacy manifest. Version three additionally
    owns the exact private clone-reference table and unique reference index.
    """

    try:
        version = (
            connection.execute("PRAGMA user_version").fetchone()[0]
            if expected_version is None
            else expected_version
        )
        if type(version) is not int or version not in (1, 2, 3, 4):
            raise ValueError
        _run_with_deadline_progress(
            connection,
            check_deadline,
            lambda: _validate_schema_body(connection, schema_version=version),
        )
    except ProfileRepositoryError:
        raise
    except BaseException as error:
        if not isinstance(error, Exception):
            raise
        raise _repository_error("schema_corrupt") from None


def _validate_schema_body(
    connection: sqlite3.Connection, *, schema_version: int
) -> None:
    """Validate schema invariants while any caller-owned progress hook is active."""

    try:
        if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
            raise ValueError
        expected_tables = {PROFILE_TABLE, ASSIGNMENT_TABLE}
        if schema_version in (3, 4):
            expected_tables.add(_REFERENCE_TABLE)
        if _user_tables(connection) != expected_tables:
            raise ValueError
        _validate_owned_schema_sql(connection, schema_version=schema_version)

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
            connection.execute("PRAGMA index_list(tts_generation_profiles)")
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
        if list(connection.execute("PRAGMA foreign_key_list(tts_generation_profiles)")):
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
            connection.execute("PRAGMA index_list(character_tts_assignments)")
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
            connection.execute("PRAGMA foreign_key_list(character_tts_assignments)")
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

        if schema_version in (3, 4):
            quoted_reference_table = _validated_quoted_identifier(
                _REFERENCE_TABLE,
                "table name",
            )
            expected_reference_manifest = [
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
            if schema_version == 4:
                expected_reference_manifest.extend(
                    [
                        (12, "recipe_id", "TEXT", 0, None, 0, 0),
                        (13, "recipe_revision", "INTEGER", 0, None, 0, 0),
                    ]
                )
            if (
                _table_xinfo_manifest(connection, _REFERENCE_TABLE)
                != expected_reference_manifest
            ):
                raise ValueError
            if not _has_exact_primary_key_index(
                connection, _REFERENCE_TABLE, ("profile_id",)
            ):
                raise ValueError
            reference_index_rows = list(
                connection.execute(f"PRAGMA index_list({quoted_reference_table})")
            )
            reference_indexes = {row["name"]: row for row in reference_index_rows}
            reference_id_index = reference_indexes.get(_REFERENCE_ID_INDEX)
            if (
                len(reference_index_rows) != 2
                or reference_id_index is None
                or reference_id_index["origin"] != "c"
                or reference_id_index["partial"] != 0
                or reference_id_index["unique"] != 1
                or not _has_exact_binary_index_keys(
                    connection, _REFERENCE_ID_INDEX, ("reference_id",)
                )
            ):
                raise ValueError
            reference_foreign_keys = list(
                connection.execute(f"PRAGMA foreign_key_list({quoted_reference_table})")
            )
            if len(reference_foreign_keys) != 1:
                raise ValueError
            reference_foreign_key = reference_foreign_keys[0]
            if (
                reference_foreign_key["table"],
                reference_foreign_key["from"],
                reference_foreign_key["to"],
                reference_foreign_key["on_delete"],
            ) != (PROFILE_TABLE, "profile_id", "profile_id", "CASCADE"):
                raise ValueError

        quick_check = [row[0] for row in connection.execute("PRAGMA quick_check")]
        if quick_check != ["ok"]:
            raise ValueError
        if list(connection.execute("PRAGMA foreign_key_check")):
            raise ValueError
    except ProfileRepositoryError:
        raise
    except Exception:  # noqa: BLE001 - normalize hostile persisted values without source text
        raise _repository_error("schema_corrupt") from None


def validate_profile_store_rows(
    connection: sqlite3.Connection,
    *,
    check_deadline: Callable[[], None] | None = None,
) -> None:
    """Decode every schema-owned profile, assignment, and joined snapshot row.

    Args:
        connection: Caller-owned connection to a validated profile-store schema.

    Raises:
        ProfileRepositoryError: If any persisted domain value fails closed.
        BaseException: A caller control-flow signal preserved unchanged.
    """

    try:
        if check_deadline is not None:
            check_deadline()
        for row in connection.execute("SELECT * FROM tts_generation_profiles"):
            if check_deadline is not None:
                check_deadline()
            decode_profile(row)
        for row in connection.execute("SELECT * FROM character_tts_assignments"):
            if check_deadline is not None:
                check_deadline()
            decode_assignment(row)
        for row in connection.execute(ASSIGNED_PROFILE_JOIN_SELECT):
            if check_deadline is not None:
                check_deadline()
            decode_assigned_snapshot(row)
        if check_deadline is not None:
            check_deadline()
    except ProfileRepositoryError:
        raise
    except Exception:  # noqa: BLE001 - normalize hostile persisted values without source text
        raise _repository_error("corrupt_data") from None
