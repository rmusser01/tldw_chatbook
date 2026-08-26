"""Version-specific stepping for caller-owned private profile candidates."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

import tldw_chatbook.TTS.profile_schema as _profile_schema
from tldw_chatbook.DB import private_sqlite as _private_sqlite
from tldw_chatbook.DB.private_sqlite import (
    ProfileMigrationBoundaryDestination,
    backup_profile_migration_boundary,
)
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


_ProfileDomain: TypeAlias = tuple[
    tuple[tuple[object, ...], ...],
    tuple[tuple[object, ...], ...],
]
_ReferenceDomain: TypeAlias = tuple[tuple[object, ...], ...]


@dataclass(frozen=True, slots=True, repr=False)
class _BoundaryEvidence:
    schema_version: int
    profile_domain: _ProfileDomain
    reference_domain: _ReferenceDomain

    def __repr__(self) -> str:
        return "_BoundaryEvidence(<private>)"


_CAPABILITY_FACTORY_TOKEN = object()


class ProfileMigrationBoundarySnapshot:
    """Revocable single-use authority to copy one exact validated boundary."""

    __slots__ = (
        "__attempted",
        "__completed",
        "__evidence",
        "__snapshot",
        "__thread_id",
    )

    def __init__(
        self,
        factory_token: object,
        snapshot: sqlite3.Connection,
        evidence: _BoundaryEvidence,
    ) -> None:
        if factory_token is not _CAPABILITY_FACTORY_TOKEN:
            raise ProfileRepositoryError("migration_failed")
        from threading import get_ident

        self.__attempted = False
        self.__completed = False
        self.__evidence: _BoundaryEvidence | None = evidence
        self.__snapshot: sqlite3.Connection | None = snapshot
        self.__thread_id = get_ident()

    def __repr__(self) -> str:
        return "ProfileMigrationBoundarySnapshot()"

    def backup_to(self, destination: ProfileMigrationBoundaryDestination) -> None:
        """Consume and durably prepare one opaque boundary destination.

        Args:
            destination: Already-open empty private SQLite destination consumed
                through checkpoint, close, fsync, immutable validation, and
                readiness. The caller owns only later artifact publication or
                cleanup.

        Raises:
            ProfileRepositoryError: If authority is expired/used or the
                destination is not an exact empty foreign connection.
            BaseException: A control-flow signal preserved unchanged.
        """

        try:
            from threading import get_ident

            snapshot = self.__snapshot
            evidence = self.__evidence
            if (
                snapshot is None
                or evidence is None
                or self.__attempted
                or self.__thread_id != get_ident()
            ):
                raise ValueError
            self.__attempted = True
            if type(destination) is not ProfileMigrationBoundaryDestination:
                raise ValueError
            _require_boundary_evidence(snapshot, evidence)
            backup_profile_migration_boundary(
                snapshot,
                destination,
                schema_version=evidence.schema_version,
                validate=lambda connection: _require_boundary_evidence(
                    connection,
                    evidence,
                ),
            )
            self.__completed = True
        except ProfileRepositoryError:
            raise ProfileRepositoryError("migration_failed") from None
        except BaseException as error:
            if not isinstance(error, Exception):
                raise
            raise ProfileRepositoryError("migration_failed") from None

    def _revoke(self) -> None:
        self.__snapshot = None
        self.__evidence = None


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
                evidence = _capture_boundary_evidence(connection, version)
                _run_boundary_callback(
                    connection,
                    request,
                    evidence,
                    boundary_sink,
                )

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


def _capture_boundary_evidence(
    connection: sqlite3.Connection,
    version: int,
) -> _BoundaryEvidence:
    _validate_candidate_version(connection, version)
    return _BoundaryEvidence(
        schema_version=version,
        profile_domain=_profile_schema._migration_domain_snapshot(connection),
        reference_domain=_compact_reference_evidence(connection)
        if version >= 3
        else (),
    )


def _compact_reference_evidence(
    connection: sqlite3.Connection,
) -> _ReferenceDomain:
    """Capture exact reference identity/metadata without retaining WAV bytes.

    One shared definition with the live opener's migration evidence
    (TASK-21130) so both paths carry the same payload-free projection.
    """

    return _profile_schema._migration_reference_evidence(connection)


def _require_boundary_evidence(
    connection: sqlite3.Connection,
    evidence: _BoundaryEvidence,
) -> None:
    _validate_candidate_version(connection, evidence.schema_version)
    if (
        _profile_schema._migration_domain_snapshot(connection)
        != evidence.profile_domain
        or (
            _compact_reference_evidence(connection)
            if evidence.schema_version >= 3
            else ()
        )
        != evidence.reference_domain
    ):
        raise ProfileRepositoryError("migration_failed")


def _raise_boundary_errors(*errors: BaseException | None) -> None:
    for error in errors:
        if error is not None and not isinstance(error, Exception):
            raise error
    if any(error is not None for error in errors):
        raise ProfileRepositoryError("migration_failed") from None


def _run_boundary_callback(
    connection: sqlite3.Connection,
    request: ProfileMigrationBoundaryRequest,
    evidence: _BoundaryEvidence,
    sink: ProfileMigrationBoundarySink | None,
) -> None:
    callback_error: BaseException | None = None
    live_validation_error: BaseException | None = None
    if sink is not None:
        try:
            _emit_boundary(connection, evidence, request, sink)
        except BaseException as error:
            callback_error = error
    else:
        callback_error = ValueError()
    try:
        _require_boundary_evidence(connection, evidence)
    except BaseException as error:
        live_validation_error = error
    _raise_boundary_errors(callback_error, live_validation_error)


def _emit_boundary(
    connection: sqlite3.Connection,
    evidence: _BoundaryEvidence,
    request: ProfileMigrationBoundaryRequest,
    sink: ProfileMigrationBoundarySink,
) -> None:
    snapshot: sqlite3.Connection | None = None
    capability: ProfileMigrationBoundarySnapshot | None = None
    body_error: BaseException | None = None
    validation_error: BaseException | None = None
    close_error: BaseException | None = None
    try:
        snapshot = _private_sqlite._snapshot_connection_to_memory(connection)
        _require_boundary_evidence(snapshot, evidence)
        capability = ProfileMigrationBoundarySnapshot(
            _CAPABILITY_FACTORY_TOKEN,
            snapshot,
            evidence,
        )
        sink(capability, request)
        if not object.__getattribute__(
            capability,
            "_ProfileMigrationBoundarySnapshot__completed",
        ):
            raise ValueError
    except BaseException as error:
        body_error = error
    finally:
        if capability is not None:
            capability._revoke()
    if snapshot is not None:
        try:
            _require_boundary_evidence(snapshot, evidence)
        except BaseException as error:
            validation_error = error
        try:
            snapshot.close()
        except BaseException as error:
            close_error = error
    _raise_boundary_errors(body_error, validation_error, close_error)


def _step_candidate(connection: sqlite3.Connection, version: int) -> None:
    profile_domain = (
        None if version == 0 else _profile_schema._migration_domain_snapshot(connection)
    )
    # Payload-free evidence (TASK-21130). ``version >= 3`` implies
    # ``version == 3``, which always has a downgrade boundary, so
    # ``_capture_boundary_evidence`` -> ``_validate_candidate_version`` has
    # just re-derived sha256(wav_bytes) for every row; the post-step
    # ``validate_profile_store_version`` below re-derives it again. Those two
    # bracket this comparison of the sha256 column and give exact payload
    # identity without ever holding a second copy of the table.
    reference_domain = _compact_reference_evidence(connection) if version >= 3 else ()
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
            _compact_reference_evidence(connection) != reference_domain
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
    except BaseException as primary_error:
        rollback_error: BaseException | None = None
        if connection.in_transaction:
            try:
                connection.rollback()
            except BaseException as error:
                rollback_error = error
        for pending_error in (primary_error, rollback_error):
            if pending_error is not None and not isinstance(pending_error, Exception):
                raise pending_error
        raise primary_error


__all__ = [
    "ProfileMigrationBoundary",
    "ProfileMigrationBoundaryRequest",
    "ProfileMigrationBoundarySnapshot",
    "ProfileMigrationCandidateResult",
    "ProfileMigrationBoundarySink",
    "step_profile_migration_candidate",
]
