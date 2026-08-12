"""Version-specific stepping for caller-owned private profile candidates."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import tldw_chatbook.TTS.profile_schema as _profile_schema
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError


class ProfileMigrationBoundary(str, Enum):
    """A downgrade snapshot boundary on a validated candidate."""

    PRE_V3 = "pre_v3"
    PRE_V4 = "pre_v4"


@dataclass(frozen=True, slots=True)
class ProfileMigrationBoundaryRequest:
    """Public, payload-free request to snapshot one exact schema boundary."""

    kind: ProfileMigrationBoundary
    schema_version: int


@dataclass(frozen=True, slots=True)
class ProfileMigrationCandidateResult:
    """Payload-free result after a candidate reaches the current schema."""

    source_version: int
    final_version: int
    boundaries: tuple[ProfileMigrationBoundaryRequest, ...]


@dataclass(slots=True)
class _SnapshotCapabilityState:
    source: sqlite3.Connection
    schema_version: int
    used: bool = False


_CAPABILITY_FACTORY_TOKEN = object()
_CAPABILITY_STATES: dict[object, _SnapshotCapabilityState] = {}


class ProfileMigrationBoundarySnapshot:
    """Revocable single-use authority to copy one exact validated boundary."""

    __slots__ = ("__key",)

    def __init__(self, factory_token: object, key: object) -> None:
        if factory_token is not _CAPABILITY_FACTORY_TOKEN:
            raise ProfileRepositoryError("migration_failed")
        self.__key = key

    def __repr__(self) -> str:
        return "ProfileMigrationBoundarySnapshot()"

    def backup_to(self, destination: sqlite3.Connection) -> None:
        """Copy the boundary into one caller-owned empty destination.

        Args:
            destination: Already-open empty private SQLite destination. The
                caller retains ownership and must close and durably prepare it.

        Raises:
            ProfileRepositoryError: If authority is expired/used or the
                destination is not an exact empty foreign connection.
            BaseException: A control-flow signal preserved unchanged.
        """

        try:
            state = _CAPABILITY_STATES.get(self.__key)
            if state is None or state.used:
                raise ValueError
            state.used = True
            if (
                not isinstance(destination, sqlite3.Connection)
                or destination is state.source
                or destination.in_transaction
            ):
                raise ValueError
            destination.row_factory = sqlite3.Row
            destination.execute("PRAGMA foreign_keys = ON")
            database_rows = list(destination.execute("PRAGMA database_list"))
            version_row = destination.execute("PRAGMA user_version").fetchone()
            objects_row = destination.execute(
                "SELECT COUNT(*) FROM sqlite_schema WHERE name NOT GLOB 'sqlite_*'"
            ).fetchone()
            if (
                len(database_rows) != 1
                or database_rows[0][1] != "main"
                or version_row is None
                or len(version_row) != 1
                or type(version_row[0]) is not int
                or version_row[0] != 0
                or objects_row is None
                or len(objects_row) != 1
                or type(objects_row[0]) is not int
                or objects_row[0] != 0
            ):
                raise ValueError
            state.source.backup(destination)
            _profile_schema.validate_profile_store_version(
                destination,
                state.schema_version,
            )
        except ProfileRepositoryError:
            raise ProfileRepositoryError("migration_failed") from None
        except BaseException as error:
            if not isinstance(error, Exception):
                raise
            raise ProfileRepositoryError("migration_failed") from None


ProfileMigrationBoundarySink = Callable[
    [ProfileMigrationBoundarySnapshot, ProfileMigrationBoundaryRequest], None
]


def step_profile_migration_candidate(
    connection: sqlite3.Connection,
    *,
    boundary_sink: ProfileMigrationBoundarySink | None = None,
) -> ProfileMigrationCandidateResult:
    """Take ownership of and advance one already-open private candidate.

    A narrow snapshot capability is synchronously borrowed by ``boundary_sink``
    only after exact boundary validation. It can copy once into a caller-owned
    empty destination, exposes no source connection or SQL API, and is revoked
    when the callback returns or raises. The candidate is always closed before
    this function returns or raises.

    Args:
        connection: Already-open private candidate connection. Ownership is
            transferred to this call and the connection is always closed.
        boundary_sink: Optional synchronous owner for exact v2/v3 snapshots.

    Returns:
        Immutable, payload-free boundary and final-version facts.

    Raises:
        ProfileRepositoryError: If the source or any migration step fails.
        BaseException: A caller control-flow signal preserved after cleanup.
    """

    body_error: BaseException | None = None
    cleanup_errors: list[BaseException] = []
    result: ProfileMigrationCandidateResult | None = None
    migration_started = False
    try:
        if not isinstance(connection, sqlite3.Connection) or (
            boundary_sink is not None and not callable(boundary_sink)
        ):
            raise ProfileRepositoryError("operation_failed")
        if connection.in_transaction:
            raise ProfileRepositoryError("migration_failed")
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {_profile_schema.BUSY_TIMEOUT_MS}")
        source_version = _read_source_version(connection)
        boundaries: list[ProfileMigrationBoundaryRequest] = []
        version = source_version
        _validate_candidate_version(connection, version)
        while version < _profile_schema.CURRENT_PROFILE_SCHEMA_VERSION:
            request = _boundary_request(version)
            if request is not None:
                boundaries.append(request)
                if boundary_sink is not None:
                    _emit_boundary(connection, request, boundary_sink)
                _validate_candidate_version(connection, version)

            migration_started = True
            _step_candidate(connection, version)
            version += 1

        result = ProfileMigrationCandidateResult(
            source_version=source_version,
            final_version=version,
            boundaries=tuple(boundaries),
        )
    except BaseException as error:
        body_error = error

    connection_in_transaction = False
    if isinstance(connection, sqlite3.Connection):
        try:
            connection_in_transaction = connection.in_transaction
        except BaseException as error:
            cleanup_errors.append(error)
    if connection_in_transaction:
        try:
            connection.rollback()
        except BaseException as error:
            cleanup_errors.append(error)
    if isinstance(connection, sqlite3.Connection):
        try:
            connection.close()
        except BaseException as error:
            cleanup_errors.append(error)

    for pending_error in (body_error, *cleanup_errors):
        if pending_error is not None and not isinstance(pending_error, Exception):
            raise pending_error
    if body_error is not None:
        if isinstance(body_error, ProfileRepositoryError) and not migration_started:
            raise ProfileRepositoryError(body_error.code) from None
        raise ProfileRepositoryError("migration_failed") from None
    if cleanup_errors:
        raise ProfileRepositoryError("migration_failed") from None
    assert result is not None
    return result


def _read_source_version(connection: sqlite3.Connection) -> int:
    try:
        row = connection.execute("PRAGMA user_version").fetchone()
        if row is None or len(row) != 1 or type(row[0]) is not int:
            raise ValueError
        version = row[0]
        if version > _profile_schema.CURRENT_PROFILE_SCHEMA_VERSION:
            raise ProfileRepositoryError("schema_unsupported")
        if version < 0:
            raise ValueError
        return version
    except ProfileRepositoryError:
        raise
    except BaseException as error:
        if not isinstance(error, Exception):
            raise
        raise ProfileRepositoryError("schema_corrupt") from None


def _validate_candidate_version(
    connection: sqlite3.Connection,
    version: int,
) -> None:
    if version == 0:
        if _profile_schema._user_schema_objects(connection):
            raise ProfileRepositoryError("schema_partial")
        try:
            _profile_schema._validate_full_integrity(connection)
        except Exception:
            raise ProfileRepositoryError("schema_corrupt") from None
        return
    _profile_schema.validate_profile_store_version(connection, version)


def _boundary_request(version: int) -> ProfileMigrationBoundaryRequest | None:
    if version == 2:
        return ProfileMigrationBoundaryRequest(ProfileMigrationBoundary.PRE_V3, 2)
    if version == 3:
        return ProfileMigrationBoundaryRequest(ProfileMigrationBoundary.PRE_V4, 3)
    return None


def _emit_boundary(
    connection: sqlite3.Connection,
    request: ProfileMigrationBoundaryRequest,
    sink: ProfileMigrationBoundarySink,
) -> None:
    key = object()
    _CAPABILITY_STATES[key] = _SnapshotCapabilityState(
        source=connection,
        schema_version=request.schema_version,
    )
    capability = ProfileMigrationBoundarySnapshot(_CAPABILITY_FACTORY_TOKEN, key)
    body_error: BaseException | None = None
    try:
        sink(capability, request)
    except BaseException as error:
        body_error = error
    finally:
        _CAPABILITY_STATES.pop(key, None)
    if body_error is not None and not isinstance(body_error, Exception):
        raise body_error
    if body_error is not None:
        raise ProfileRepositoryError("migration_failed") from None


def _step_candidate(connection: sqlite3.Connection, version: int) -> None:
    profile_domain = (
        None if version == 0 else _profile_schema._migration_domain_snapshot(connection)
    )
    reference_domain = (
        _profile_schema._migration_reference_snapshot(connection)
        if version >= 3
        else ()
    )
    try:
        connection.execute("BEGIN IMMEDIATE")
        migration = _profile_schema.MIGRATIONS.get(version)
        if migration is None:
            raise ValueError
        migration(connection)
        next_version = version + 1
        _profile_schema.validate_profile_store_version(connection, next_version)
        if profile_domain is not None and (
            _profile_schema._migration_domain_snapshot(connection) != profile_domain
        ):
            raise ValueError
        if version >= 3 and (
            _profile_schema._migration_reference_snapshot(connection)
            != reference_domain
        ):
            raise ValueError
        if (
            next_version == 3
            and connection.execute(
                f"SELECT COUNT(*) FROM {_profile_schema._REFERENCE_TABLE}"
            ).fetchone()[0]
            != 0
        ):
            raise ValueError
        if (
            next_version == 4
            and connection.execute(
                f"SELECT COUNT(*) FROM {_profile_schema._REFERENCE_TABLE} "
                "WHERE recipe_id IS NOT NULL OR recipe_revision IS NOT NULL"
            ).fetchone()[0]
            != 0
        ):
            raise ValueError
        connection.commit()
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise


__all__ = [
    "ProfileMigrationBoundary",
    "ProfileMigrationBoundaryRequest",
    "ProfileMigrationBoundarySnapshot",
    "ProfileMigrationCandidateResult",
    "ProfileMigrationBoundarySink",
    "step_profile_migration_candidate",
]
