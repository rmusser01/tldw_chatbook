"""SQLite schema, validation, and persistence codecs for TTS profiles.

Connections remain caller-owned. The live opener configures and returns a
connection. Candidate validation owns its disposable snapshot connections and
retains uncertain native cleanup: a brief read-write reopen upgrades only the
copy, followed by an immutable read-only validation handle.
"""

from __future__ import annotations

import json
import hashlib
import os
import sqlite3
import stat
import struct
from dataclasses import dataclass
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast
from uuid import UUID

if TYPE_CHECKING:
    from tldw_chatbook.TTS.profile_repository import _BackupNativeState

from tldw_chatbook.DB.private_sqlite import (
    _SQLiteAdmissionOutcome,
    connect_private_sqlite,
    connect_private_sqlite_descriptor,
)
from tldw_chatbook.DB.sql_validation import escape_identifier, validate_identifier
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_migration_journal import (
    MAX_PROFILE_MIGRATION_ARTIFACT_BYTES,
)
from tldw_chatbook.Utils import private_paths
from tldw_chatbook.TTS.migrations.v0_to_v1 import (
    ASSIGNMENT_PROFILE_INDEX_DDL as _ASSIGNMENT_PROFILE_INDEX_DDL,
)
from tldw_chatbook.TTS.migrations.v0_to_v1 import (
    ASSIGNMENT_TABLE_DDL as _ASSIGNMENT_TABLE_DDL,
)
from tldw_chatbook.TTS.migrations.v0_to_v1 import (
    PROFILE_TABLE_DDL as _PROFILE_TABLE_DDL,
)
from tldw_chatbook.TTS.migrations.v0_to_v1 import migrate as _migrate_v0_to_v1
from tldw_chatbook.TTS.migrations.v1_to_v2 import migrate as _migrate_v1_to_v2
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
from tldw_chatbook.TTS.migrations.v2_to_v3 import migrate as _migrate_v2_to_v3
from tldw_chatbook.TTS.migrations.v3_to_v4 import (
    REFERENCE_ID_INDEX_DDL as _V4_REFERENCE_ID_INDEX_DDL,
)
from tldw_chatbook.TTS.migrations.v3_to_v4 import (
    REFERENCE_TABLE_DDL as _V4_REFERENCE_TABLE_DDL,
)
from tldw_chatbook.TTS.migrations.v3_to_v4 import migrate as _migrate_v3_to_v4
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


@dataclass(frozen=True, slots=True, repr=False)
class PostInitProfileStoreAuthority:
    """Exact closed-store identity retained across exclusive/shared handoff."""

    parent_identity: os.stat_result
    file_identity: os.stat_result

    def __repr__(self) -> str:
        return "PostInitProfileStoreAuthority(<private>)"


def _repository_error(code: str) -> ProfileRepositoryError:
    return ProfileRepositoryError(code)


class _ExactCurrentProfileConnection:
    """Live SQLite handle retaining the descriptor authority that admitted it."""

    def __init__(
        self,
        connection: sqlite3.Connection | None,
        *,
        evidence_connection: sqlite3.Connection | None,
        selected: Path,
        parent_fd: int,
        file_fd: int,
        parent_identity: os.stat_result | None,
        file_identity: os.stat_result | None,
        sidecar_fds: dict[str, int],
        sidecar_identities: dict[str, os.stat_result],
    ) -> None:
        self._connection = connection
        self._evidence_connection = evidence_connection
        self.selected = selected
        self.parent_fd = parent_fd
        self.file_fd = file_fd
        self.parent_identity = parent_identity
        self.file_identity = file_identity
        self.sidecar_fds = sidecar_fds
        self.sidecar_identities = sidecar_identities

        self.leases: list[Any] = []
        self.native_descriptors: set[int] = set()
        self.attempted_descriptors: set[int] = set()
        self.attempted_connections: set[str] = set()
        self.pending: set[object] = set()
        self.uncertain = False
        self.body_error: BaseException | None = None
        self.cleanup_errors: list[BaseException] = []

    def admit(self) -> None:
        from tldw_chatbook.Backup_Recovery import storage_admission as storage

        lease = storage.acquire_storage(self.selected)
        # Existing strong native lease membership retains this concrete owner.
        lease.native_owner = self
        self.leases.append(lease)

    def open_parent_descriptor(self, *args: Any, **kwargs: Any) -> int:
        self.admit()
        outcome = private_paths._NativeOpenOutcome()
        attempt = object()
        self.pending.add(attempt)
        try:
            descriptor = private_paths._native_open(*args, _outcome=outcome, **kwargs)
        finally:
            if outcome.descriptor is not None:
                self.native_descriptors.add(outcome.descriptor)
                self.pending.discard(attempt)
            elif outcome.rejected:
                self.pending.discard(attempt)
        return descriptor

    def open_descriptor(self, *args: Any, **kwargs: Any) -> int:
        self.admit()
        attempt = object()
        self.pending.add(attempt)
        open = os.open
        try:
            descriptor = open(*args, **kwargs)
        except OSError:
            if open is private_paths._ORIGINAL_NATIVE_OPEN:
                self.pending.discard(attempt)
            raise
        self.native_descriptors.add(descriptor)
        self.pending.discard(attempt)
        return descriptor

    def close_parent_descriptor(self, descriptor: int) -> None:
        self.close_descriptor(descriptor, _parent_traversal=True)

    def close_descriptor(
        self, descriptor: int, *, _parent_traversal: bool = False
    ) -> None:
        if descriptor in self.attempted_descriptors:
            raise ExactProfileStoreCleanupError(self)
        self.attempted_descriptors.add(descriptor)
        try:
            if _parent_traversal:
                private_paths._native_close(descriptor)
            else:
                os.close(descriptor)
        except BaseException as error:
            self.uncertain = True
            self.cleanup_errors.append(error)
            raise
        self.native_descriptors.discard(descriptor)
        self.attempted_descriptors.discard(descriptor)

    def __getattr__(self, name: str) -> object:
        return getattr(self._connection, name)

    def execute(self, sql: str, parameters: object = ()) -> sqlite3.Cursor:
        return self._connection.execute(sql, parameters)  # type: ignore[arg-type]

    @property
    def in_transaction(self) -> bool:
        return self._connection.in_transaction

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

    @property
    def row_factory(self) -> object:
        return self._connection.row_factory

    @row_factory.setter
    def row_factory(self, value: object) -> None:
        self._connection.row_factory = value  # type: ignore[assignment]

    def close(self) -> None:
        if self.uncertain or self.pending:
            self.uncertain = True
            raise ExactProfileStoreCleanupError(self)
        try:
            for role in ("_connection", "_evidence_connection"):
                native = getattr(self, role)
                if native is not None and role not in self.attempted_connections:
                    if role == "_connection":
                        revalidate_exact_current_profile_store(
                            cast(sqlite3.Connection, self), self.selected
                        )
                    self.attempted_connections.add(role)
                    native.close()
            for suffix, descriptor in tuple(self.sidecar_fds.items()):
                self.close_descriptor(descriptor)
                del self.sidecar_fds[suffix]
            if self.file_fd >= 0:
                self.close_descriptor(self.file_fd)
                self.file_fd = -1
            if self.parent_fd >= 0:
                self.close_descriptor(self.parent_fd)
                self.parent_fd = -1
            for descriptor in tuple(self.native_descriptors):
                self.close_descriptor(descriptor)
            while self.leases:
                self.leases[-1].close()
                self.leases.pop()
        except BaseException as error:
            self.uncertain = True
            self.cleanup_errors.append(error)
            raise


class ExactProfileStoreCleanupError(ProfileRepositoryError):
    """Carry exact retained authority when an internal close cannot settle."""

    def __init__(self, connection: _ExactCurrentProfileConnection) -> None:
        super().__init__("operation_failed")
        self.connection = connection


class ExactProfileStoreNotCurrentError(ProfileRepositoryError):
    """Signal that shared proof must yield to exclusive initialization."""

    def __init__(self) -> None:
        super().__init__("schema_partial")


class ExactProfileStoreAuthorityError(ProfileRepositoryError):
    """Signal that retained live-store namespace authority no longer matches."""

    def __init__(self) -> None:
        super().__init__("operation_failed")


def _exact_store_namespace_safe(
    parent_fd: int,
    leaf: str,
) -> bool:
    publication = f".{leaf}.migration-publication.json"
    for suffix in ("", "-wal", "-shm", "-journal"):
        try:
            os.stat(
                f"{publication}{suffix}",
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            continue
        except OSError:
            return False
        return False
    for suffix in ("-wal", "-shm", "-journal"):
        try:
            observed = os.stat(
                f"{leaf}{suffix}",
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            continue
        except OSError:
            return False
        if suffix == "-journal":
            return False
        if (
            private_paths._classify_private_file_stat(
                observed,
                expected_uid=os.geteuid(),
            )
            is not None
            or stat.S_IMODE(observed.st_mode) != 0o600
        ):
            return False
    return True


def _open_exact_store_sidecars(
    parent_fd: int,
    leaf: str,
    *,
    _owner: _ExactCurrentProfileConnection | None = None,
) -> tuple[dict[str, int], dict[str, os.stat_result]] | None:
    open_fd = _owner.open_descriptor if _owner is not None else os.open
    close_fd = _owner.close_descriptor if _owner is not None else os.close
    descriptors: dict[str, int] = {}
    identities: dict[str, os.stat_result] = {}
    for suffix in ("-wal", "-shm"):
        try:
            descriptor = open_fd(
                f"{leaf}{suffix}",
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_NOCTTY", 0),
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            descriptor = -1
        except OSError:
            descriptor = -2
        if descriptor >= 0:
            observed = os.fstat(descriptor)
            try:
                named = os.stat(
                    f"{leaf}{suffix}",
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except OSError:
                close_fd(descriptor)
                descriptor = -2
            else:
                if (
                    not private_paths._same_identity(observed, named)
                    or private_paths._classify_private_file_stat(
                        observed,
                        expected_uid=os.geteuid(),
                    )
                    is not None
                    or stat.S_IMODE(observed.st_mode) != 0o600
                ):
                    close_fd(descriptor)
                    descriptor = -2
                else:
                    descriptors[suffix] = descriptor
                    identities[suffix] = observed
        if descriptor == -2:
            for opened in descriptors.values():
                close_fd(opened)
            raise _repository_error("operation_failed")
    if not descriptors:
        return None
    if set(descriptors) != {"-wal", "-shm"}:
        for opened in descriptors.values():
            close_fd(opened)
        raise ExactProfileStoreNotCurrentError()
    return descriptors, identities


def revalidate_exact_current_profile_store(
    connection: sqlite3.Connection,
    path: Path | None,
) -> None:
    """Recheck retained exact-current authority immediately before live use."""

    if not isinstance(connection, _ExactCurrentProfileConnection):
        return
    if (
        path is None
        or connection.selected != path
        or connection.file_fd < 0
        or connection.parent_fd < 0
    ):
        raise ExactProfileStoreAuthorityError()
    reopened_parent_fd = -1
    try:
        reopened_parent_fd, reopened_leaf = private_paths._open_verified_parent(
            path,
            missing_leaf_allowed=False,
        )
        reopened_parent = os.fstat(reopened_parent_fd)
        opened_parent = os.fstat(connection.parent_fd)
        opened_file = os.fstat(connection.file_fd)
        named = os.stat(
            path.name,
            dir_fd=connection.parent_fd,
            follow_symlinks=False,
        )
    except Exception:
        raise ExactProfileStoreAuthorityError() from None
    finally:
        if reopened_parent_fd >= 0:
            os.close(reopened_parent_fd)
    sidecars_match = set(connection.sidecar_fds) == {"-wal", "-shm"}
    if sidecars_match:
        for suffix, descriptor in connection.sidecar_fds.items():
            try:
                opened_sidecar = os.fstat(descriptor)
                named_sidecar = os.stat(
                    f"{path.name}{suffix}",
                    dir_fd=connection.parent_fd,
                    follow_symlinks=False,
                )
            except OSError:
                sidecars_match = False
                break
            expected_sidecar = connection.sidecar_identities[suffix]
            if (
                not private_paths._same_identity(opened_sidecar, expected_sidecar)
                or not private_paths._same_identity(named_sidecar, expected_sidecar)
                or private_paths._classify_private_file_stat(
                    named_sidecar,
                    expected_uid=os.geteuid(),
                )
                is not None
                or stat.S_IMODE(named_sidecar.st_mode) != 0o600
            ):
                sidecars_match = False
                break
    if (
        reopened_leaf != path.name
        or not _same_parent_authority(reopened_parent, connection.parent_identity)
        or not _same_parent_authority(opened_parent, connection.parent_identity)
        or not private_paths._same_identity(opened_file, connection.file_identity)
        or not private_paths._same_identity(named, connection.file_identity)
        or private_paths._classify_private_file_stat(
            named,
            expected_uid=os.geteuid(),
        )
        is not None
        or stat.S_IMODE(named.st_mode) != 0o600
        or not sidecars_match
        or not _exact_store_namespace_safe(
            connection.parent_fd,
            path.name,
        )
    ):
        raise ExactProfileStoreAuthorityError()


def _same_parent_authority(
    observed: os.stat_result,
    expected: os.stat_result,
) -> bool:
    """Compare stable identity and security metadata for one store parent."""

    return (
        observed.st_nlink > 0
        and expected.st_nlink > 0
        and (
            observed.st_dev,
            observed.st_ino,
            observed.st_mode,
            observed.st_uid,
            observed.st_gid,
        )
        == (
            expected.st_dev,
            expected.st_ino,
            expected.st_mode,
            expected.st_uid,
            expected.st_gid,
        )
    )


def _matches_post_init_authority(
    parent: os.stat_result,
    file: os.stat_result,
    expected: PostInitProfileStoreAuthority,
) -> bool:
    return (
        _same_parent_authority(parent, expected.parent_identity)
        and private_paths._same_identity(file, expected.file_identity)
        and file.st_size == expected.file_identity.st_size
        and file.st_nlink == 1
        and private_paths._classify_private_file_stat(
            file,
            expected_uid=os.geteuid(),
        )
        is None
        and stat.S_IMODE(file.st_mode) == 0o600
    )


def capture_post_init_profile_store_authority(
    path: Path,
) -> PostInitProfileStoreAuthority:
    """Pin one closed, sidecar-free store before releasing exclusive ownership."""

    parent_fd = -1
    file_fd = -1
    try:
        parent_fd, leaf = private_paths._open_verified_parent(
            path,
            missing_leaf_allowed=False,
        )
        parent = os.fstat(parent_fd)
        if not _exact_store_namespace_safe(parent_fd, leaf):
            raise _repository_error("operation_failed")
        file_fd = os.open(
            leaf,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOCTTY", 0),
            dir_fd=parent_fd,
        )
        opened = os.fstat(file_fd)
        named = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        provisional = PostInitProfileStoreAuthority(parent, opened)
        if (
            opened.st_size > MAX_PROFILE_MIGRATION_ARTIFACT_BYTES
            or not _matches_post_init_authority(parent, opened, provisional)
            or not _matches_post_init_authority(parent, named, provisional)
        ):
            raise _repository_error("operation_failed")
        os.fsync(file_fd)
        os.fsync(parent_fd)
        settled_parent = os.fstat(parent_fd)
        settled_file = os.fstat(file_fd)
        settled_named = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not _matches_post_init_authority(
                settled_parent,
                settled_file,
                provisional,
            )
            or not _matches_post_init_authority(
                settled_parent,
                settled_named,
                provisional,
            )
            or not _exact_store_namespace_safe(parent_fd, leaf)
        ):
            raise _repository_error("operation_failed")
        return provisional
    except ProfileRepositoryError:
        raise
    except Exception:
        raise _repository_error("operation_failed") from None
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


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
        if len(value.encode("utf-8")) > _MAX_PERSISTED_OPTIONS_BYTES:
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

    # Manually freeze the provided options using internal freezing logic
    return _freeze_options(cast(JsonOptions, options))


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


MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {
    0: _migrate_v0_to_v1,
    1: _migrate_v1_to_v2,
    2: _migrate_v2_to_v3,
    3: _migrate_v3_to_v4,
}


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
        except BaseException as error:
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
    except BaseException as error:
        body_error = error

    if progress_installed:
        try:
            connection.set_progress_handler(None, 0)
        except BaseException as error:
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
    except Exception:
        raise _repository_error("schema_corrupt") from None


def _validate_full_integrity(connection: sqlite3.Connection) -> None:
    """Run full integrity and foreign-key validation before migration commit."""

    if [row[0] for row in connection.execute("PRAGMA integrity_check")] != ["ok"]:
        raise ValueError
    if list(connection.execute("PRAGMA foreign_key_check")):
        raise ValueError


def _migration_domain_snapshot(
    connection: sqlite3.Connection,
) -> tuple[tuple[tuple[object, ...], ...], tuple[tuple[object, ...], ...]]:
    """Capture exact ordered v2 profile and assignment persistence domains."""

    profiles = tuple(
        tuple(row)
        for row in connection.execute(
            """
            SELECT profile_id, display_name, normalized_name, provider_id,
                   model_id, voice_id, response_format, speed, options_json,
                   revision, created_at, updated_at
            FROM tts_generation_profiles
            ORDER BY profile_id
            """
        )
    )
    assignments = tuple(
        tuple(row)
        for row in connection.execute(
            """
            SELECT source, authority_id, character_id, profile_id,
                   created_at, updated_at
            FROM character_tts_assignments
            ORDER BY source, authority_id, character_id
            """
        )
    )
    return profiles, assignments


def _migration_reference_snapshot(
    connection: sqlite3.Connection,
) -> tuple[tuple[object, ...], ...]:
    """Capture every schema-v3 reference field in deterministic order."""

    return tuple(
        tuple(row)
        for row in connection.execute(
            f"""
            SELECT profile_id, reference_id, wav_bytes, reference_text, sha256,
                   byte_length, duration_ms, sample_rate_hz, channels,
                   sample_encoding, created_at, updated_at
            FROM {_REFERENCE_TABLE}
            ORDER BY profile_id
            """
        )
    )


def _validate_migration_reference_rows(
    connection: sqlite3.Connection,
    *,
    schema_version: int,
) -> None:
    """Fully decode references at either side of the v3-to-v4 boundary."""

    from tldw_chatbook.TTS.profile_reference_audio import (
        validate_canonical_reference_wav,
    )
    from tldw_chatbook.TTS.profile_reference_storage import (
        decode_reference_payload,
        read_reference_blob,
        validate_reference_rows,
    )
    from tldw_chatbook.TTS.profile_reference_types import (
        MAX_REFERENCE_COUNT,
        MAX_REFERENCE_TOTAL_BYTES,
    )

    if schema_version == 4:
        validate_reference_rows(connection)
        return
    quota = connection.execute(
        f"SELECT COUNT(*), COALESCE(SUM(byte_length), 0) FROM {_REFERENCE_TABLE}"
    ).fetchone()
    if (
        quota is None
        or type(quota[0]) is not int
        or type(quota[1]) is not int
        or not 0 <= quota[0] <= MAX_REFERENCE_COUNT
        or not 0 <= quota[1] <= MAX_REFERENCE_TOTAL_BYTES
    ):
        raise ValueError
    seen = 0
    for row in connection.execute(
        f"""
        SELECT r.rowid AS reference_rowid,
               r.reference_id AS reference_reference_id,
               r.reference_text, r.sha256,
               r.byte_length AS reference_byte_length,
               r.duration_ms AS reference_duration_ms,
               r.sample_rate_hz AS reference_sample_rate_hz,
               r.channels AS reference_channels,
               r.sample_encoding AS reference_sample_encoding,
               r.created_at AS reference_created_at,
               r.updated_at AS reference_updated_at,
               NULL AS reference_recipe_id,
               NULL AS reference_recipe_revision,
               p.model_id AS reference_model_id
        FROM {_REFERENCE_TABLE} AS r
        JOIN {PROFILE_TABLE} AS p ON p.profile_id = r.profile_id
        ORDER BY r.profile_id
        """
    ):
        payload = read_reference_blob(
            connection,
            row["reference_rowid"],
            row["reference_byte_length"],
        )
        reference = decode_reference_payload(row, payload)
        metadata = validate_canonical_reference_wav(payload)
        if (
            metadata.byte_length != reference.summary.byte_length
            or metadata.duration_ms != reference.summary.duration_ms
            or metadata.sample_rate_hz != reference.summary.sample_rate_hz
            or metadata.channels != reference.summary.channels
            or metadata.sample_encoding != reference.summary.sample_encoding
        ):
            raise ValueError
        seen += 1
    if seen != quota[0]:
        raise ValueError


class _CleanupState:
    """Run all cleanup actions while preserving the first control-flow signal."""

    def __init__(self, primary_error: BaseException | None = None) -> None:
        self.control_flow: BaseException | None = (
            primary_error
            if primary_error is not None and not isinstance(primary_error, Exception)
            else None
        )
        self.ordinary_cleanup_failed = False

    def attempt(self, action: Callable[[], object]) -> None:
        try:
            action()
        except BaseException as error:
            if not isinstance(error, Exception):
                if self.control_flow is None:
                    self.control_flow = error
            else:
                self.ordinary_cleanup_failed = True

    def raise_control_flow(self) -> None:
        if self.control_flow is not None:
            raise self.control_flow


def _run_migrations(connection: sqlite3.Connection, from_version: int) -> None:
    """Run every registered migration from ``from_version`` up to current.

    Used both to build a brand-new store from version 0 and to upgrade an
    existing populated store from any older version in place. The whole
    climb runs inside one ``BEGIN IMMEDIATE`` transaction so a mid-flight
    failure leaves the store exactly as it was found.
    """

    body_error: BaseException | None = None
    try:
        connection.execute("BEGIN IMMEDIATE")
        version = from_version
        domain_snapshot = (
            ((), ()) if from_version == 0 else _migration_domain_snapshot(connection)
        )
        reference_snapshot: tuple[tuple[object, ...], ...] = ()
        if from_version >= 3:
            _validate_migration_reference_rows(connection, schema_version=from_version)
            reference_snapshot = _migration_reference_snapshot(connection)
        while version < CURRENT_PROFILE_SCHEMA_VERSION:
            if version == 2:
                validate_profile_store_rows(connection)
            migration = MIGRATIONS.get(version)
            if migration is None:
                raise RuntimeError
            migration(connection)
            version += 1
            version_row = connection.execute("PRAGMA user_version").fetchone()
            if (
                version_row is None
                or len(version_row) != 1
                or type(version_row[0]) is not int
                or version_row[0] != version
            ):
                raise RuntimeError
        _validate_full_integrity(connection)
        _validate_schema_body(connection, schema_version=CURRENT_PROFILE_SCHEMA_VERSION)
        if (
            connection.execute(
                f"SELECT count(*) FROM {_REFERENCE_TABLE} "
                "WHERE recipe_id IS NOT NULL OR recipe_revision IS NOT NULL"
            ).fetchone()[0]
            != 0
        ):
            raise RuntimeError
        if _migration_domain_snapshot(connection) != domain_snapshot:
            raise RuntimeError
        if _migration_reference_snapshot(connection) != reference_snapshot:
            raise RuntimeError
        validate_profile_store_rows(connection)
        _validate_migration_reference_rows(
            connection,
            schema_version=CURRENT_PROFILE_SCHEMA_VERSION,
        )
        connection.commit()
    except BaseException as error:
        body_error = error

    if body_error is None:
        return
    cleanup = _CleanupState(body_error)
    cleanup.attempt(connection.rollback)
    cleanup.raise_control_flow()
    raise _repository_error("migration_failed") from None


def _migrate_empty_store(connection: sqlite3.Connection) -> None:
    _run_migrations(connection, 0)


def peek_profile_store_schema_version(path: Path) -> int | None:
    """Read only the on-disk schema version, without validating or migrating.

    A cheap, side-effect-free hint the repository's lease orchestration uses
    to decide whether an already-existing store needs an exclusive-lease
    upgrade (see :data:`CURRENT_PROFILE_SCHEMA_VERSION`) before it is safe to
    open under a shared lease -- opening under shared is documented and
    relied on elsewhere as read-only, and the in-place upgrade in
    :func:`open_profile_store` is a write.

    Returns ``None`` whenever the version cannot be determined this way --
    missing file, unreadable, corrupt, or any other failure. Callers must
    treat ``None`` as "no opinion" and fall back to the normal open flow,
    which already handles every one of those cases correctly on its own.
    """

    if not isinstance(path, Path):
        return None
    connection: sqlite3.Connection | None = None
    try:
        connection = connect_private_sqlite(
            "tts.profile_store_version_peek",
            path,
            read_only=True,
            isolation_level=None,
        )
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        return version if type(version) is int else None
    except Exception:
        return None
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


def open_exact_current_profile_store(
    path: Path,
    *,
    expected_post_init_authority: PostInitProfileStoreAuthority | None = None,
) -> sqlite3.Connection:
    """Open exact current v4 without migration while retaining its proof pin."""

    if not isinstance(path, Path) or not path.is_absolute():
        raise _repository_error("operation_failed")
    parent_fd = -1
    file_fd = -1
    sidecar_fds: dict[str, int] = {}
    sidecar_identities: dict[str, os.stat_result] = {}
    descriptor: sqlite3.Connection | None = None
    live: sqlite3.Connection | None = None
    owned = _ExactCurrentProfileConnection(
        None,
        evidence_connection=None,
        selected=path,
        parent_fd=-1,
        file_fd=-1,
        parent_identity=None,
        file_identity=None,
        sidecar_fds={},
        sidecar_identities={},
    )
    owned.admit()
    body_error: BaseException | None = None
    try:
        parent_fd, leaf = private_paths._open_verified_parent(
            path,
            missing_leaf_allowed=False,
            _open=owned.open_parent_descriptor,
            _close=owned.close_parent_descriptor,
        )
        owned.parent_fd = parent_fd
        parent_identity = os.fstat(parent_fd)
        owned.parent_identity = parent_identity
        if expected_post_init_authority is not None and not _same_parent_authority(
            parent_identity,
            expected_post_init_authority.parent_identity,
        ):
            raise _repository_error("operation_failed")
        if not _exact_store_namespace_safe(parent_fd, leaf):
            raise ExactProfileStoreNotCurrentError()
        file_fd = owned.open_descriptor(
            leaf,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOCTTY", 0),
            dir_fd=parent_fd,
        )
        owned.file_fd = file_fd
        file_identity = os.fstat(file_fd)
        owned.file_identity = file_identity
        if (
            file_identity.st_size > MAX_PROFILE_MIGRATION_ARTIFACT_BYTES
            or private_paths._classify_private_file_stat(
                file_identity,
                expected_uid=os.geteuid(),
            )
            is not None
            or stat.S_IMODE(file_identity.st_mode) != 0o600
        ):
            raise ExactProfileStoreNotCurrentError()
        named = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if not private_paths._same_identity(named, file_identity) or (
            expected_post_init_authority is not None
            and (
                not _matches_post_init_authority(
                    parent_identity,
                    file_identity,
                    expected_post_init_authority,
                )
                or not _matches_post_init_authority(
                    parent_identity,
                    named,
                    expected_post_init_authority,
                )
            )
        ):
            raise _repository_error("operation_failed")
        pinned_sidecars = _open_exact_store_sidecars(parent_fd, leaf, _owner=owned)
        if pinned_sidecars is not None:
            sidecar_fds, sidecar_identities = pinned_sidecars
            owned.sidecar_fds = sidecar_fds
            owned.sidecar_identities = sidecar_identities

        owned.admit()
        owned.pending.add("evidence")
        admission_outcome = _SQLiteAdmissionOutcome()
        try:
            descriptor = connect_private_sqlite_descriptor(
                "tts.profile_store_descriptor",
                file_fd,
                isolation_level=None,
                _admission_outcome=admission_outcome,
            )
        finally:
            if admission_outcome.admission_refused:
                owned.pending.discard("evidence")
        owned._evidence_connection = descriptor
        owned.pending.discard("evidence")
        _configure_connection(descriptor)
        descriptor_version = descriptor.execute("PRAGMA user_version").fetchone()[0]
        if descriptor_version != CURRENT_PROFILE_SCHEMA_VERSION:
            raise ExactProfileStoreNotCurrentError()
        _validate_schema(descriptor)
        validate_profile_store_rows(descriptor)
        _stream_exact_store_metadata_evidence(descriptor)

        owned.admit()
        owned.pending.add("live")
        admission_outcome = _SQLiteAdmissionOutcome()
        try:
            live = connect_private_sqlite(
                "tts.profile_store",
                path,
                must_exist=True,
                expected_identity=file_identity,
                isolation_level=None,
                _admission_outcome=admission_outcome,
            )
        finally:
            if admission_outcome.admission_refused:
                owned.pending.discard("live")
        owned._connection = live
        owned.pending.discard("live")
        live.execute("PRAGMA query_only = ON")
        if live.execute("PRAGMA query_only").fetchone()[0] != 1:
            raise _repository_error("schema_corrupt")
        # Force SQLite to acquire its main database and WAL cohort before
        # retaining exact sidecar descriptors.
        live.execute("PRAGMA user_version").fetchone()
        if live.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
            raise ExactProfileStoreNotCurrentError()
        if not sidecar_fds:
            opened_sidecars = _open_exact_store_sidecars(parent_fd, leaf, _owner=owned)
            if opened_sidecars is None:
                raise _repository_error("operation_failed")
            sidecar_fds, sidecar_identities = opened_sidecars
        owned.sidecar_fds = sidecar_fds
        owned.sidecar_identities = sidecar_identities
        live = None
        descriptor = None
        parent_fd = -1
        file_fd = -1
        sidecar_fds = {}
        sidecar_identities = {}
        _configure_connection(cast(sqlite3.Connection, owned))
        live_version = owned.execute("PRAGMA user_version").fetchone()[0]
        if live_version != CURRENT_PROFILE_SCHEMA_VERSION:
            raise _repository_error("schema_partial")
        _validate_schema(cast(sqlite3.Connection, owned))
        validate_profile_store_rows(cast(sqlite3.Connection, owned))
        owned.execute("BEGIN")
        _stream_exact_store_metadata_evidence(cast(sqlite3.Connection, owned))
        revalidate_exact_current_profile_store(
            cast(sqlite3.Connection, owned),
            path,
        )
        if expected_post_init_authority is not None and (
            not _matches_post_init_authority(
                os.fstat(owned.parent_fd),
                os.fstat(owned.file_fd),
                expected_post_init_authority,
            )
        ):
            raise _repository_error("operation_failed")
        owned.rollback()
        revalidate_exact_current_profile_store(
            cast(sqlite3.Connection, owned),
            path,
        )
        owned.execute("PRAGMA query_only = OFF")
        if owned.execute("PRAGMA query_only").fetchone()[0] != 0:
            raise _repository_error("schema_corrupt")
        revalidate_exact_current_profile_store(
            cast(sqlite3.Connection, owned),
            path,
        )
    except FileNotFoundError:
        body_error = ExactProfileStoreNotCurrentError()
    except BaseException as error:
        body_error = error

    if body_error is None:
        assert owned is not None
        return cast(sqlite3.Connection, owned)

    owned.body_error = body_error
    try:
        owned.close()
    except BaseException:
        raise ExactProfileStoreCleanupError(owned) from None
    if not isinstance(body_error, Exception):
        raise body_error
    if isinstance(body_error, ProfileRepositoryError):
        raise body_error
    raise _repository_error("schema_corrupt") from None


def open_profile_store(
    path: Path,
    *,
    must_exist: bool = False,
    check_deadline: Callable[[], None] | None = None,
) -> sqlite3.Connection:
    """Open/configure a live store, optionally refusing to create a missing file.

    Args:
        path: Profile-store path.
        must_exist: When true, use SQLite ``mode=rw`` so no missing database can
            be created during restore validation or lifecycle rebind.
        check_deadline: Optional restore-time cooperative deadline callback.

    Returns:
        One fully configured and validated caller-owned connection.

    Raises:
        ProfileRepositoryError: If inputs or the store fail closed validation.
    """

    connection: sqlite3.Connection | None = None
    body_error: BaseException | None = None
    try:
        if (
            not isinstance(path, Path)
            or type(must_exist) is not bool
            or (check_deadline is not None and not callable(check_deadline))
        ):
            raise _repository_error("operation_failed")
        if check_deadline is not None:
            check_deadline()
        if must_exist:
            resolution_missing = False
            resolution_failed = False
            resolved_path: Path | None = None
            try:
                resolved_path = path.resolve(strict=True)
            except FileNotFoundError:
                resolution_missing = True
            except Exception:
                resolution_failed = True
            if resolution_missing:
                raise _repository_error("missing")
            if resolution_failed or resolved_path is None:
                raise _repository_error("operation_failed")
            if not resolved_path.is_file():
                raise _repository_error("missing")
        connect_error: BaseException | None = None
        try:
            connection = connect_private_sqlite(
                "tts.profile_store",
                path,
                must_exist=must_exist,
                isolation_level=None,
            )
        except BaseException as error:
            connect_error = error
        if connect_error is not None:
            missing_after_connect = False
            if must_exist:
                try:
                    missing_after_connect = not path.resolve(strict=True).is_file()
                except FileNotFoundError:
                    missing_after_connect = True
                except Exception:
                    pass
            if missing_after_connect:
                raise _repository_error("missing")
            raise connect_error
        assert connection is not None
        if check_deadline is not None:
            check_deadline()
        _configure_connection(connection)
        if check_deadline is not None:
            check_deadline()
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if type(version) is not int:
            raise _repository_error("schema_corrupt")
        if version > CURRENT_PROFILE_SCHEMA_VERSION:
            raise _repository_error("schema_unsupported")
        if version == 0:
            if must_exist:
                raise _repository_error("schema_partial")
            if _user_schema_objects(connection):
                raise _repository_error("schema_partial")
            journal_mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if journal_mode != "wal":
                raise _repository_error("schema_corrupt")
            # NORMAL is safe under WAL (app-crash-safe; only an OS/power
            # crash can lose the last commit, acceptable for this local TTS
            # profile store) and avoids an fsync per commit. This owner is
            # private-file only (no :memory: target), so no memory guard is
            # needed here (task-15465).
            connection.execute("PRAGMA synchronous = NORMAL")
            _migrate_empty_store(connection)
        elif version < CURRENT_PROFILE_SCHEMA_VERSION:
            _validate_schema(
                connection,
                expected_version=version,
                check_deadline=check_deadline,
            )
            journal_mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if journal_mode != "wal":
                raise _repository_error("schema_corrupt")
            # NORMAL is safe under WAL -- see the version==0 branch above for
            # the full rationale (task-15465).
            connection.execute("PRAGMA synchronous = NORMAL")
            _run_migrations(connection, version)
        else:
            _validate_schema(
                connection,
                check_deadline=check_deadline,
            )
            journal_mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if journal_mode != "wal":
                raise _repository_error("schema_corrupt")
            # NORMAL is safe under WAL -- see the version==0 branch above for
            # the full rationale (task-15465).
            connection.execute("PRAGMA synchronous = NORMAL")
        _validate_schema(
            connection,
            check_deadline=check_deadline,
        )
    except BaseException as error:
        body_error = error

    if body_error is None:
        assert connection is not None
        return connection

    cleanup = _CleanupState(body_error)
    if connection is not None:
        cleanup.attempt(connection.close)
    cleanup.raise_control_flow()
    if isinstance(body_error, ProfileRepositoryError):
        raise body_error
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
    except Exception:
        raise _repository_error("corrupt_data") from None


def validate_profile_store_version(
    connection: sqlite3.Connection,
    expected_version: int,
) -> None:
    """Fully validate one exact supported persisted schema and its domain.

    This validator is intentionally path-free so private migration and restore
    candidates can reuse the live store's exact schema/domain codecs without
    discovering or opening the configured repository file.

    Args:
        connection: Caller-owned connection to the candidate store.
        expected_version: Exact supported schema version required on disk.

    Raises:
        ProfileRepositoryError: If schema, integrity, foreign keys, or any
            decoded domain value fails closed validation.
        BaseException: A caller control-flow signal preserved unchanged.
    """

    try:
        if type(expected_version) is not int or expected_version not in (1, 2, 3, 4):
            raise ValueError
        version_row = connection.execute("PRAGMA user_version").fetchone()
        if (
            version_row is None
            or len(version_row) != 1
            or type(version_row[0]) is not int
            or version_row[0] != expected_version
        ):
            raise ValueError
        _validate_schema(connection, expected_version=expected_version)
        _validate_full_integrity(connection)
        validate_profile_store_rows(connection)
        if expected_version >= 3:
            _validate_migration_reference_rows(
                connection,
                schema_version=expected_version,
            )
    except ProfileRepositoryError:
        raise
    except BaseException as error:
        if not isinstance(error, Exception):
            raise
        raise _repository_error("schema_corrupt") from None


def _source_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _candidate_source_open_flags(flag_source: object = os) -> int:
    return (
        int(getattr(flag_source, "O_RDONLY", 0))
        | int(getattr(flag_source, "O_CLOEXEC", 0))
        | int(getattr(flag_source, "O_NONBLOCK", 0))
        | int(getattr(flag_source, "O_NOFOLLOW", 0))
        | int(getattr(flag_source, "O_BINARY", 0))
    )


def _open_candidate_source(path: Path, flags: int) -> int:
    return os.open(path, flags)


def _close_candidate_fd(descriptor: int) -> None:
    os.close(descriptor)


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
) -> bool:
    return (
        _source_identity(os.fstat(source_fd)) == source_identity
        and _source_identity(os.stat(resolved_path)) == source_identity
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


def _copy_source_to_snapshot(
    source_fd: int,
    snapshot_fd: int,
    *,
    check_deadline: Callable[[], None] | None = None,
) -> None:
    while True:
        if check_deadline is not None:
            check_deadline()
        chunk = os.read(source_fd, 1024 * 1024)
        if not chunk:
            break
        offset = 0
        while offset < len(chunk):
            if check_deadline is not None:
                check_deadline()
            written = os.write(snapshot_fd, chunk[offset:])
            if written <= 0:
                raise OSError
            offset += written
    if check_deadline is not None:
        check_deadline()
    os.fsync(snapshot_fd)
    if check_deadline is not None:
        check_deadline()


def _apply_posix_snapshot_mode(snapshot_fd: int) -> bool:
    if os.name != "posix":
        return False
    fchmod = getattr(os, "fchmod", None)
    if not callable(fchmod):
        return False
    fchmod(snapshot_fd, 0o600)
    return True


_CANDIDATE_DIR_FD_CALLS = frozenset((os.open, os.stat, os.unlink, os.rmdir))


class _CandidateValidationJob:
    """Own one ordinary synchronous candidate call, never installed authority."""

    def __init__(self, source: object) -> None:
        from tldw_chatbook.Backup_Recovery import storage_admission as storage

        self.storage = storage
        self.parent_pins_available = (
            os.name == "posix"
            and bool(getattr(os, "O_DIRECTORY", 0))
            and bool(getattr(os, "O_NOFOLLOW", 0))
            and callable(getattr(os, "fchmod", None))
            and _CANDIDATE_DIR_FD_CALLS.issubset(getattr(os, "supports_dir_fd", ()))
        )
        self.source = source
        self.pid = os.getpid()
        self.thread = storage.threading.current_thread()
        self.resolved_source: Path | None = None
        self.source_identity: tuple[int, ...] | None = None
        self.attempt = None
        self.leases = []
        self.descriptors: dict[str, int] = {}
        self.connections: dict[str, sqlite3.Connection] = {}
        self.attempted: set[str] = set()
        self.directory: Path | None = None
        self.directory_identity: tuple[int, int] | None = None
        self.directory_parent_identity: tuple[int, int] | None = None
        self.snapshot: Path | None = None
        self.snapshot_identity: tuple[int, int] | None = None
        self.snapshot_parent_identity: tuple[int, int] | None = None
        self.allocation_pending = False
        self.uncertain = False
        self.body_error: BaseException | None = None
        self.cleanup_errors: list[BaseException] = []
        with storage._changed:
            storage._raw_operations.add(self)
            storage._changed.notify_all()

    def admit(self, path: Path | None) -> None:
        # No token is installed or inherited as this job's authority. Even an
        # outer admitted repository operation cannot grant it work after pause.
        with self.storage._lock:
            if self.storage._pause is not None:
                from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

                raise RecoveryRequired("storage_locally_paused")
        if self.attempt is None:
            self.attempt = self.storage._Acquisition()
        self.attempt.check(path)
        lease = self.storage.acquire_storage(path)
        self.leases.append(lease)
        self.attempt.check(path)

    def pin_parent(self, role: str, path: Path) -> tuple[int, int]:
        self.admit(path)
        if not self.parent_pins_available:
            # Explicit ordinary-unqualified platform path; never a retry after
            # pin/identity/IO failure and never installed/capture authority.
            named = path.lstat()
            if not stat.S_ISDIR(named.st_mode):
                raise ValueError
            return named.st_dev, named.st_ino
        self.allocation_pending = True
        self.descriptors[role] = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        self.allocation_pending = False
        opened = os.fstat(self.descriptors[role])
        named = path.lstat()
        identity = (opened.st_dev, opened.st_ino)
        if not stat.S_ISDIR(opened.st_mode) or identity != (named.st_dev, named.st_ino):
            self.uncertain = True
            raise ValueError
        return identity

    def check_directory(self) -> None:
        assert self.directory is not None
        named = self.directory.lstat()
        pinned = (
            os.fstat(self.descriptors["directory"])
            if self.parent_pins_available
            else named
        )
        if (named.st_dev, named.st_ino) != self.directory_identity or (
            pinned.st_dev,
            pinned.st_ino,
        ) != self.directory_identity:
            raise ValueError
        parent = self.directory.parent.lstat()
        pinned_parent = (
            os.fstat(self.descriptors["directory_parent"])
            if self.parent_pins_available
            else parent
        )
        if (parent.st_dev, parent.st_ino) != self.directory_parent_identity or (
            pinned_parent.st_dev,
            pinned_parent.st_ino,
        ) != self.directory_parent_identity:
            raise ValueError

    def check_snapshot(self) -> None:
        assert self.snapshot is not None
        named = self.snapshot.lstat()
        parent = self.snapshot.parent.lstat()
        pinned = (
            os.fstat(self.descriptors["snapshot_parent"])
            if self.parent_pins_available
            else parent
        )
        if (
            not stat.S_ISREG(named.st_mode)
            or named.st_nlink != 1
            or (named.st_dev, named.st_ino) != self.snapshot_identity
            or (parent.st_dev, parent.st_ino) != self.snapshot_parent_identity
            or (pinned.st_dev, pinned.st_ino) != self.snapshot_parent_identity
            or not _sidecars_absent(self.snapshot)
        ):
            raise ValueError

    def close_descriptor(self, role: str) -> None:
        if role in self.attempted:
            raise ValueError
        self.attempted.add(role)
        _close_candidate_fd(self.descriptors[role])
        del self.descriptors[role]

    def cleanup(self, body_error: BaseException | None) -> _CleanupState:
        if self.body_error is None:
            self.body_error = body_error
        cleanup = _CleanupState(body_error)

        def attempt(action: Callable[[], object]) -> None:
            def recorded() -> None:
                try:
                    action()
                except BaseException as error:
                    self.uncertain = True
                    self.cleanup_errors.append(error)
                    raise

            cleanup.attempt(recorded)

        for role in ("upgrade", "read"):
            if role in self.connections:

                def close_connection(role: str = role) -> None:
                    self.check_snapshot()
                    if role in self.attempted:
                        raise ValueError
                    self.attempted.add(role)
                    self.connections[role].close()
                    del self.connections[role]

                attempt(close_connection)
        for role in ("snapshot", "source"):
            if role in self.descriptors:
                attempt(lambda role=role: self.close_descriptor(role))
        if self.allocation_pending:
            self.uncertain = True
        if not self.uncertain and self.snapshot is not None:

            def remove_snapshot() -> None:
                self.check_snapshot()
                if self.parent_pins_available:
                    os.unlink(
                        self.snapshot.name, dir_fd=self.descriptors["snapshot_parent"]
                    )
                else:
                    os.unlink(self.snapshot)
                self.snapshot = None

            attempt(remove_snapshot)
        if not self.uncertain and self.directory is not None:

            def remove_directory() -> None:
                self.check_directory()
                if self.parent_pins_available:
                    os.rmdir(
                        self.directory.name, dir_fd=self.descriptors["directory_parent"]
                    )
                else:
                    self.directory.rmdir()
                self.directory = None

            attempt(remove_directory)
        if not self.uncertain:
            for role in tuple(self.descriptors):
                attempt(lambda role=role: self.close_descriptor(role))
        if not self.uncertain:
            for lease in self.leases:
                attempt(lease.close)
                if self.uncertain:
                    break
        if self.attempt is not None:
            self.attempt.close()
        if not self.uncertain:
            with self.storage._changed:
                self.storage._raw_operations.discard(self)
                self.storage._changed.notify_all()
        return cleanup


def validate_profile_candidate(
    path: Path,
    *,
    check_deadline: Callable[[], None] | None = None,
    _outer_job: _BackupNativeState | None = None,
) -> None:
    """Validate a disposable private copy, retaining uncertain native cleanup."""
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    job = _CandidateValidationJob(path)
    if _outer_job is not None:
        _outer_job.candidate_job = job
    body_error: BaseException | None = None
    try:
        job.admit(path if isinstance(path, Path) else None)
        _validate_profile_candidate(path, check_deadline=check_deadline, job=job)
    except RecoveryRequired:
        body_error = _repository_error("schema_corrupt")
    except BaseException as error:
        body_error = error
    cleanup = job.cleanup(body_error)
    cleanup.raise_control_flow()
    if body_error is not None:
        raise body_error
    if cleanup.ordinary_cleanup_failed or job.uncertain:
        raise _repository_error("schema_corrupt") from None


def _validate_profile_candidate(
    path: Path,
    *,
    check_deadline: Callable[[], None] | None = None,
    job: _CandidateValidationJob,
) -> None:
    """Validate a point-in-time private snapshot of a standalone v1 backup.

    A later restore must validate its own repository-controlled staged snapshot;
    a successful path validation is never an authorization to trust future bytes.
    """

    if check_deadline is not None:
        check_deadline()
    if not isinstance(path, Path):
        raise _repository_error("missing")
    try:
        resolved_path = path.resolve(strict=True)
    except FileNotFoundError:
        raise _repository_error("missing") from None
    except Exception:
        raise _repository_error("schema_corrupt") from None
    if check_deadline is not None:
        check_deadline()
    if not resolved_path.is_file():
        raise _repository_error("missing")
    try:
        if not _sidecars_absent(resolved_path):
            raise _repository_error("schema_corrupt")
    except ProfileRepositoryError:
        raise
    except Exception:
        raise _repository_error("schema_corrupt") from None
    if check_deadline is not None:
        check_deadline()

    source_fd: int | None = None
    snapshot_fd: int | None = None
    snapshot_path: str | None = None
    snapshot_directory: Path | None = None
    connection: sqlite3.Connection | None = None
    upgrade_connection: sqlite3.Connection | None = None
    body_error: BaseException | None = None
    try:
        if check_deadline is not None:
            check_deadline()
        job.resolved_source = resolved_path
        path_state = _source_identity(os.stat(resolved_path))
        if not stat.S_ISREG(path_state[2]):
            raise ValueError

        job.admit(resolved_path)
        job.allocation_pending = True
        source_fd = _open_candidate_source(
            resolved_path,
            _candidate_source_open_flags(),
        )
        job.descriptors["source"] = source_fd
        job.allocation_pending = False
        if check_deadline is not None:
            check_deadline()
        source_state = _source_identity(os.fstat(source_fd))
        job.source_identity = source_state
        if source_state != path_state or not _source_is_unchanged(
            source_fd,
            resolved_path,
            source_state,
        ):
            raise ValueError

        job.admit(Path(tempfile.gettempdir()).resolve(strict=True))
        job.allocation_pending = True
        snapshot_directory = Path(
            tempfile.mkdtemp(
                prefix="tldw-tts-profile-candidate-",
                dir=Path(tempfile.gettempdir()).resolve(strict=True),
            )
        )
        job.directory = snapshot_directory
        job.allocation_pending = False
        job.directory_parent_identity = job.pin_parent(
            "directory_parent", snapshot_directory.parent
        )
        job.directory_identity = job.pin_parent("directory", snapshot_directory)
        job.check_directory()
        if job.parent_pins_available:
            os.fchmod(job.descriptors["directory"], 0o700)
        else:
            os.chmod(snapshot_directory, 0o700)
        job.admit(snapshot_directory)
        job.check_directory()
        job.allocation_pending = True
        snapshot_fd, snapshot_path = tempfile.mkstemp(
            prefix="snapshot-",
            suffix=".sqlite3",
            dir=snapshot_directory,
        )
        job.descriptors["snapshot"] = snapshot_fd
        job.snapshot = Path(snapshot_path)
        job.allocation_pending = False
        info = os.fstat(snapshot_fd)
        job.snapshot_identity = (info.st_dev, info.st_ino)
        job.snapshot_parent_identity = job.pin_parent(
            "snapshot_parent", job.snapshot.parent
        )
        posix_mode_enforced = _apply_posix_snapshot_mode(snapshot_fd)
        _copy_source_to_snapshot(
            source_fd,
            snapshot_fd,
            check_deadline=check_deadline,
        )
        snapshot_state = _source_identity(os.fstat(snapshot_fd))
        if (
            snapshot_state[3] != source_state[3]
            or not stat.S_ISREG(snapshot_state[2])
            or (posix_mode_enforced and stat.S_IMODE(snapshot_state[2]) != 0o600)
            or not _snapshot_is_unchanged(
                snapshot_fd,
                snapshot_path,
                snapshot_state,
            )
            or not _source_is_unchanged(
                source_fd,
                resolved_path,
                source_state,
            )
        ):
            raise ValueError

        if check_deadline is not None:
            check_deadline()
        job.admit(Path(snapshot_path))
        job.allocation_pending = True
        upgrade_connection = connect_private_sqlite(
            "tts.profile_candidate_upgrade",
            snapshot_path,
            must_exist=True,
            isolation_level=None,
        )
        job.connections["upgrade"] = upgrade_connection
        job.allocation_pending = False
        job.check_snapshot()
        _configure_connection(upgrade_connection)
        # Force the disposable snapshot out of WAL mode before touching it:
        # switching away from WAL always checkpoints and removes any -wal/
        # -shm sidecars, which keeps the private snapshot directory
        # deterministically single-file no matter whether the upgrade below
        # actually writes anything.
        upgrade_journal_mode = upgrade_connection.execute(
            "PRAGMA journal_mode = DELETE"
        ).fetchone()[0]
        if upgrade_journal_mode != "delete":
            raise _repository_error("schema_corrupt")
        candidate_version = upgrade_connection.execute(
            "PRAGMA user_version"
        ).fetchone()[0]
        if type(candidate_version) is not int:
            raise _repository_error("schema_corrupt")
        if 0 < candidate_version < CURRENT_PROFILE_SCHEMA_VERSION:
            # Mirror the live open flow's upgrade sequence exactly: validate
            # the schema at its current (pre-upgrade) shape first -- a
            # structurally corrupt v1 candidate must fail closed here,
            # before any version-stamping write -- then migrate in place.
            # The caller-supplied candidate at `resolved_path`/`source_fd`
            # is never opened for write; only this disposable copy is.
            _validate_schema(
                upgrade_connection,
                expected_version=candidate_version,
                check_deadline=check_deadline,
            )
            _run_migrations(upgrade_connection, candidate_version)
        # The upgrade step above (if it ran) is the only writer this
        # disposable snapshot ever has; recompute its identity so the
        # unchanged-checks below re-anchor to the post-upgrade bytes
        # instead of misreading our own write as tampering.
        snapshot_state = _source_identity(os.fstat(snapshot_fd))

        if check_deadline is not None:
            check_deadline()
        job.admit(Path(snapshot_path))
        job.allocation_pending = True
        connection = connect_private_sqlite(
            "tts.profile_candidate",
            snapshot_path,
            read_only=True,
            immutable=True,
            isolation_level=None,
        )
        job.connections["read"] = connection
        job.allocation_pending = False
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
        _validate_schema(
            connection,
            check_deadline=check_deadline,
        )
        validate_profile_store_rows(
            connection,
            check_deadline=check_deadline,
        )
        if not _snapshot_is_unchanged(
            snapshot_fd,
            snapshot_path,
            snapshot_state,
        ) or not _source_is_unchanged(
            source_fd,
            resolved_path,
            source_state,
        ):
            raise ValueError
    except BaseException as error:
        body_error = error
        job.body_error = error

    if body_error is not None:
        if not isinstance(body_error, Exception) or isinstance(
            body_error, ProfileRepositoryError
        ):
            raise body_error
        raise _repository_error("schema_corrupt") from None
