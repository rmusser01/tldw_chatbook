"""Single connection, migration, and transaction owner for Notes device state."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from weakref import WeakValueDictionary

from loguru import logger

from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
from tldw_chatbook.Notes import notes_device_state_schema
from tldw_chatbook.Notes.notes_sync_conflicts import linked_undo_operation_id
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncBindingState,
    NotesSyncDirection,
    NotesSyncOperationState,
    NotesSyncRecoveryAdmission,
    NotesSyncRootState,
    NotesSyncSerializationProfile,
    normalize_notes_sync_relative_path,
    validate_notes_sync_digest,
    validate_notes_sync_opaque_id,
    validate_notes_sync_reason_code,
)


_ROOT_TRANSITIONS: Mapping[NotesSyncRootState, frozenset[NotesSyncRootState]] = (
    MappingProxyType(
        {
            NotesSyncRootState.PENDING: frozenset(
                {
                    NotesSyncRootState.ACTIVE,
                    NotesSyncRootState.PAUSED,
                    NotesSyncRootState.DISCONNECTED,
                }
            ),
            NotesSyncRootState.ACTIVE: frozenset(
                {NotesSyncRootState.PAUSED, NotesSyncRootState.DISCONNECTED}
            ),
            NotesSyncRootState.PAUSED: frozenset(
                {NotesSyncRootState.ACTIVE, NotesSyncRootState.DISCONNECTED}
            ),
            NotesSyncRootState.DISCONNECTED: frozenset(),
        }
    )
)
_BINDING_TRANSITIONS: Mapping[
    NotesSyncBindingState, frozenset[NotesSyncBindingState]
] = MappingProxyType(
    {
        NotesSyncBindingState.CANDIDATE: frozenset(
            {NotesSyncBindingState.ACTIVE, NotesSyncBindingState.DISCONNECTED}
        ),
        NotesSyncBindingState.ACTIVE: frozenset(
            {
                NotesSyncBindingState.PAUSED,
                NotesSyncBindingState.NEEDS_ATTENTION,
                NotesSyncBindingState.DISCONNECTED,
            }
        ),
        NotesSyncBindingState.PAUSED: frozenset(
            {
                NotesSyncBindingState.ACTIVE,
                NotesSyncBindingState.NEEDS_ATTENTION,
                NotesSyncBindingState.DISCONNECTED,
            }
        ),
        NotesSyncBindingState.NEEDS_ATTENTION: frozenset(
            {
                NotesSyncBindingState.ACTIVE,
                NotesSyncBindingState.PAUSED,
                NotesSyncBindingState.DISCONNECTED,
            }
        ),
        NotesSyncBindingState.DISCONNECTED: frozenset(),
    }
)
_OPERATION_TRANSITIONS: Mapping[
    NotesSyncOperationState, frozenset[NotesSyncOperationState]
] = MappingProxyType(
    {
        NotesSyncOperationState.PENDING: frozenset(
            {
                NotesSyncOperationState.RECOVERY_ADMITTED,
                NotesSyncOperationState.NEEDS_ATTENTION,
            }
        ),
        NotesSyncOperationState.RECOVERY_ADMITTED: frozenset(
            {
                NotesSyncOperationState.FIRST_AUTHORITY_APPLIED,
                NotesSyncOperationState.NEEDS_ATTENTION,
            }
        ),
        NotesSyncOperationState.FIRST_AUTHORITY_APPLIED: frozenset(
            {
                NotesSyncOperationState.SECOND_AUTHORITY_APPLIED,
                NotesSyncOperationState.NEEDS_ATTENTION,
            }
        ),
        NotesSyncOperationState.SECOND_AUTHORITY_APPLIED: frozenset(
            {
                NotesSyncOperationState.BINDING_UPDATED,
                NotesSyncOperationState.NEEDS_ATTENTION,
            }
        ),
        NotesSyncOperationState.BINDING_UPDATED: frozenset(
            {
                NotesSyncOperationState.VERIFIED,
                NotesSyncOperationState.NEEDS_ATTENTION,
            }
        ),
        NotesSyncOperationState.VERIFIED: frozenset(
            {
                NotesSyncOperationState.COMPLETED,
                NotesSyncOperationState.NEEDS_ATTENTION,
            }
        ),
        NotesSyncOperationState.NEEDS_ATTENTION: frozenset(
            {NotesSyncOperationState.RECOVERY_ADMITTED}
        ),
        NotesSyncOperationState.COMPLETED: frozenset(),
    }
)
_SETTING_KEYS = frozenset({"cutover_marker", "recovery_capacity"})
_CONFLICT_SUBSTAGES = (
    "recovery_admitted",
    "folders_established",
    "copy_created",
    "placement_created",
    "copy_verified",
    "bound_note_updated",
    "file_reverified",
    "binding_updated",
    "verified",
)
_CONFLICT_SUBSTAGE_STATES = {
    "recovery_admitted": NotesSyncOperationState.RECOVERY_ADMITTED,
    "folders_established": NotesSyncOperationState.RECOVERY_ADMITTED,
    "copy_created": NotesSyncOperationState.RECOVERY_ADMITTED,
    "placement_created": NotesSyncOperationState.RECOVERY_ADMITTED,
    "copy_verified": NotesSyncOperationState.RECOVERY_ADMITTED,
    "bound_note_updated": NotesSyncOperationState.FIRST_AUTHORITY_APPLIED,
    "file_reverified": NotesSyncOperationState.SECOND_AUTHORITY_APPLIED,
    "binding_updated": NotesSyncOperationState.BINDING_UPDATED,
    "verified": NotesSyncOperationState.VERIFIED,
}
_CONFLICT_OPAQUE_ID_CAPACITY = 256
_CONFLICT_VERSION_CAPACITY = 20
_UNDO_SUBSTAGES = (
    "recovery_admitted",
    "authority_restored",
    "opposite_verified",
    "binding_updated",
    "copy_cleanup_complete",
    "verified",
)


def _now() -> int:
    return max(1, time.time_ns())


def _validate_optional_opaque_id(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return validate_notes_sync_opaque_id(value, field_name=field_name)


def _validate_optional_bounded_private_text(
    value: object,
    *,
    field_name: str,
    maximum: int,
) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value or len(value) > maximum or "\x00" in value:
        raise ValueError(f"{field_name} must be a bounded private string.")
    return value


@dataclass(frozen=True, slots=True, repr=False)
class NotesSyncRootRecord:
    """Private durable configuration for one lasting root."""

    root_id: str
    note_scope_id: str
    logical_folder_id: str | None
    canonical_path: str
    direction: NotesSyncDirection
    state: NotesSyncRootState
    remote_origin_id: str | None = None
    cursor: str | None = None
    last_status_code: str | None = None

    def __post_init__(self) -> None:
        validate_notes_sync_opaque_id(self.root_id, field_name="root_id")
        validate_notes_sync_opaque_id(self.note_scope_id, field_name="note_scope_id")
        _validate_optional_opaque_id(
            self.logical_folder_id,
            field_name="logical_folder_id",
        )
        _validate_optional_opaque_id(
            self.remote_origin_id, field_name="remote_origin_id"
        )
        if (
            type(self.canonical_path) is not str
            or not self.canonical_path
            or len(self.canonical_path) > 4096
            or "\x00" in self.canonical_path
        ):
            raise ValueError("canonical_path must be a bounded non-empty private path.")
        if type(self.direction) is not NotesSyncDirection:
            raise TypeError("direction must be a NotesSyncDirection.")
        if type(self.state) is not NotesSyncRootState:
            raise TypeError("state must be a NotesSyncRootState.")
        _validate_optional_bounded_private_text(
            self.cursor,
            field_name="cursor",
            maximum=4096,
        )
        validate_notes_sync_reason_code(self.last_status_code)

    def __repr__(self) -> str:
        return f"NotesSyncRootRecord(root_id={self.root_id!r}, state={self.state!r})"


@dataclass(frozen=True, slots=True, repr=False)
class NotesSyncBindingRecord:
    """Private durable ownership binding between one path and Database Note."""

    binding_id: str
    root_id: str
    note_scope_id: str
    note_id: str
    normalized_relative_path: str
    stable_identity_digest: str
    state: NotesSyncBindingState
    serialization: NotesSyncSerializationProfile
    content_digest: str
    note_version: int

    def __post_init__(self) -> None:
        for name, value in (
            ("binding_id", self.binding_id),
            ("root_id", self.root_id),
            ("note_scope_id", self.note_scope_id),
            ("note_id", self.note_id),
        ):
            validate_notes_sync_opaque_id(value, field_name=name)
        object.__setattr__(
            self,
            "normalized_relative_path",
            normalize_notes_sync_relative_path(self.normalized_relative_path),
        )
        validate_notes_sync_digest(
            self.stable_identity_digest,
            field_name="stable_identity_digest",
        )
        validate_notes_sync_digest(self.content_digest, field_name="content_digest")
        if type(self.state) is not NotesSyncBindingState:
            raise TypeError("state must be a NotesSyncBindingState.")
        if type(self.serialization) is not NotesSyncSerializationProfile:
            raise TypeError("serialization must be a NotesSyncSerializationProfile.")
        if type(self.note_version) is not int or self.note_version < 0:
            raise ValueError("note_version must be non-negative.")

    def __repr__(self) -> str:
        return (
            f"NotesSyncBindingRecord(binding_id={self.binding_id!r}, "
            f"state={self.state!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class NotesSyncOperationRecord:
    """Privacy-safe durable journal metadata; payload bytes live in recovery."""

    operation_id: str
    root_id: str
    binding_id: str | None
    kind: str
    state: NotesSyncOperationState
    reason_code: str | None
    observation_token: str
    expected_note_version: int | None
    expected_file_digest: str | None

    def __post_init__(self) -> None:
        validate_notes_sync_opaque_id(self.operation_id, field_name="operation_id")
        validate_notes_sync_opaque_id(self.root_id, field_name="root_id")
        _validate_optional_opaque_id(self.binding_id, field_name="binding_id")
        validate_notes_sync_reason_code(self.kind)
        if type(self.state) is not NotesSyncOperationState:
            raise TypeError("state must be a NotesSyncOperationState.")
        validate_notes_sync_reason_code(self.reason_code)
        validate_notes_sync_opaque_id(
            self.observation_token,
            field_name="observation_token",
        )
        if self.expected_note_version is not None and (
            type(self.expected_note_version) is not int
            or self.expected_note_version < 0
        ):
            raise ValueError("expected_note_version must be non-negative.")
        if self.expected_file_digest is not None:
            validate_notes_sync_digest(
                self.expected_file_digest,
                field_name="expected_file_digest",
            )

    def __repr__(self) -> str:
        return (
            f"NotesSyncOperationRecord(operation_id={self.operation_id!r}, "
            f"state={self.state!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class NotesSyncRecoveryRecord:
    """Private recovery material retained by one journal operation."""

    recovery_id: str
    operation_id: str
    payload: bytes
    metadata: bytes
    expires_at: int

    def __post_init__(self) -> None:
        validate_notes_sync_opaque_id(self.recovery_id, field_name="recovery_id")
        validate_notes_sync_opaque_id(self.operation_id, field_name="operation_id")
        if type(self.payload) is not bytes or type(self.metadata) is not bytes:
            raise TypeError("recovery payload and metadata must be bytes.")
        if type(self.expires_at) is not int or self.expires_at <= 0:
            raise ValueError("expires_at must be positive.")

    def __repr__(self) -> str:
        return f"NotesSyncRecoveryRecord(recovery_id={self.recovery_id!r})"


@dataclass(frozen=True, slots=True, repr=False)
class NotesSyncResolutionHistoryRecord:
    """Private durable facts for one conflict-resolution history row."""

    operation_id: str
    binding_id: str
    kind: str
    state: NotesSyncOperationState
    reason_code: str | None
    completed_at: int | None
    updated_at: int
    recovery_expires_at: int | None
    undo_state: NotesSyncOperationState | None
    undo_reason_code: str | None

    @property
    def undone(self) -> bool:
        """Return whether the deterministic linked Undo completed."""

        return self.undo_state is NotesSyncOperationState.COMPLETED

    def __repr__(self) -> str:
        return "NotesSyncResolutionHistoryRecord(<private>)"


@dataclass(frozen=True, slots=True, repr=False)
class NotesSyncLegacyMigrationRecord:
    """Bounded record for one later legacy-migration review."""

    migration_id: str
    source_fingerprint: str
    state: str
    reason_code: str | None

    def __post_init__(self) -> None:
        validate_notes_sync_opaque_id(self.migration_id, field_name="migration_id")
        validate_notes_sync_digest(
            self.source_fingerprint,
            field_name="source_fingerprint",
        )
        if self.state not in {"pending_review", "reviewed", "rejected"}:
            raise ValueError("state must be a supported legacy migration state.")
        validate_notes_sync_reason_code(self.reason_code)

    def __repr__(self) -> str:
        return (
            f"NotesSyncLegacyMigrationRecord(migration_id={self.migration_id!r}, "
            f"state={self.state!r})"
        )


@dataclass(frozen=True, slots=True)
class NotesSyncStoreSetting:
    """One allowlisted, bounded store setting."""

    key: str
    value: str

    def __post_init__(self) -> None:
        if self.key not in _SETTING_KEYS:
            raise ValueError("setting key is not supported.")
        if type(self.value) is not str or not self.value or len(self.value) > 256:
            raise ValueError("setting value must be between 1 and 256 characters.")
        if self.key == "recovery_capacity":
            if (
                not self.value.isascii()
                or not self.value.isdecimal()
                or self.value.startswith("0")
                or len(self.value) > 19
                or int(self.value) > 2**63 - 1
            ):
                raise ValueError(
                    "recovery_capacity setting must be a canonical positive decimal."
                )
        elif self.key == "cutover_marker":
            try:
                validate_notes_sync_opaque_id(
                    self.value,
                    field_name="cutover_marker setting",
                )
            except ValueError:
                raise ValueError(
                    "cutover_marker setting must be a bounded machine token."
                ) from None


@dataclass(frozen=True, slots=True)
class NotesSyncRootSummary:
    """Path-free public projection of one root."""

    root_id: str
    direction: NotesSyncDirection
    state: NotesSyncRootState
    last_status_code: str | None

    def __post_init__(self) -> None:
        validate_notes_sync_opaque_id(self.root_id, field_name="root_id")
        if type(self.direction) is not NotesSyncDirection:
            raise TypeError("direction must be a NotesSyncDirection.")
        if type(self.state) is not NotesSyncRootState:
            raise TypeError("state must be a NotesSyncRootState.")
        validate_notes_sync_reason_code(self.last_status_code)


@dataclass(frozen=True, slots=True)
class NotesSyncBindingSummary:
    """Path- and hash-free public projection of one binding."""

    binding_id: str
    root_id: str
    note_scope_id: str
    note_id: str
    state: NotesSyncBindingState
    note_version: int

    def __post_init__(self) -> None:
        for name, value in (
            ("binding_id", self.binding_id),
            ("root_id", self.root_id),
            ("note_scope_id", self.note_scope_id),
            ("note_id", self.note_id),
        ):
            validate_notes_sync_opaque_id(value, field_name=name)
        if type(self.state) is not NotesSyncBindingState:
            raise TypeError("state must be a NotesSyncBindingState.")
        if type(self.note_version) is not int or self.note_version < 0:
            raise ValueError("note_version must be non-negative.")


class NotesDeviceStateError(RuntimeError):
    """A device-state operation failed without disclosing private values."""


class NotesDeviceStateStore:
    """Own the sole private connection and transaction seam for notes.sync_state.

    Connection model (task-21101): each thread that uses the store holds ONE
    long-lived connection created on first use (``threading.local``), rather
    than opening a fresh connection per operation. The held connection runs
    ``journal_mode=WAL`` + ``synchronous=NORMAL`` with ``isolation_level=None``
    (true autocommit; transactions are the explicit ``BEGIN``/``BEGIN
    IMMEDIATE`` issued by :meth:`transaction`), and the schema census
    (``initialize_notes_device_schema``) runs once per connection lifetime --
    at open -- not per transaction. WAL is only adopted after the census has
    proven the database is this owner's: a refused foreign database is never
    modified.

    Thread affinity: the store is reached from the UI thread and from
    ``asyncio.to_thread`` worker pools. ``check_same_thread=False`` is safe
    here because thread-local storage guarantees each connection is used only
    by its creating thread; the one cross-thread touch is :meth:`close`, which
    sqlite3 permits. Call :meth:`close` only after in-flight operations have
    quiesced; a closed store transparently re-opens on next use.
    """

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)
        self._operation_locks: WeakValueDictionary[str, asyncio.Lock] = (
            WeakValueDictionary()
        )
        self._thread_local = threading.local()
        self._connections_guard = threading.Lock()
        self._connections: list[sqlite3.Connection] = []

    def __repr__(self) -> str:
        return "NotesDeviceStateStore(<private>)"

    def operation_lock(self, operation_id: str) -> asyncio.Lock:
        """Return the process-local lock shared by executors using this store."""

        validate_notes_sync_opaque_id(operation_id, field_name="operation_id")
        lock = self._operation_locks.get(operation_id)
        if lock is None:
            lock = asyncio.Lock()
            self._operation_locks[operation_id] = lock
        return lock

    def _connect(
        self,
        *,
        read_only: bool = False,
        must_exist: bool = False,
        **connection_options: object,
    ) -> sqlite3.Connection:
        connection = connect_private_sqlite(
            "notes.sync_state",
            self._database_path,
            read_only=read_only,
            must_exist=must_exist,
            **connection_options,
        )
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _open_schema_ready_connection(self) -> sqlite3.Connection:
        """Open, census-validate, and WAL-configure one held connection."""

        connection = self._connect(
            check_same_thread=False,
            isolation_level=None,
        )
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                notes_device_state_schema.initialize_notes_device_schema(connection)
            except BaseException:
                connection.rollback()
                raise
            connection.commit()
            # journal_mode is persisted in the database file, so WAL is
            # adopted only AFTER the census proved the database is this
            # owner's; a refused foreign database must stay byte-identical.
            # synchronous is per-connection and must be reapplied per open;
            # NORMAL is app-crash-safe under WAL and drops the per-commit
            # fsync that made receipt-heavy imports pay FULL's price.
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
        except BaseException:
            connection.close()
            raise
        return connection

    def _get_connection(self) -> sqlite3.Connection:
        connection = getattr(self._thread_local, "connection", None)
        if connection is not None:
            return connection
        connection = self._open_schema_ready_connection()
        self._thread_local.connection = connection
        with self._connections_guard:
            self._connections.append(connection)
        return connection

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        """Yield this thread's schema-ready held connection in one transaction."""

        connection = self._get_connection()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            try:
                connection.rollback()
            except Exception as rollback_error:
                # A rollback can itself fail (e.g. shutdown close()d the
                # held connection under us); that secondary error must not
                # mask the original one. Type name only: no private values.
                logger.debug(
                    "Notes device store rollback failed after a transaction "
                    "error: {}",
                    type(rollback_error).__name__,
                )
            raise

    def close(self) -> None:
        """Close every held connection; the store re-arms on next use.

        Best-effort and callable from any thread; callers must let in-flight
        operations quiesce first.
        """

        with self._connections_guard:
            connections = tuple(self._connections)
            self._connections.clear()
            self._thread_local = threading.local()
        for connection in connections:
            try:
                connection.close()
            except Exception:
                continue

    def initialize(self) -> None:
        """Atomically initialize or migrate this isolated private owner."""

        try:
            if getattr(self._thread_local, "connection", None) is None:
                # A fresh connection runs the schema census as it opens.
                self._get_connection()
            else:
                # A repeated initialize() keeps its tamper-detection
                # contract: re-run the census on the held connection.
                with self.transaction(immediate=True) as connection:
                    notes_device_state_schema.initialize_notes_device_schema(
                        connection
                    )
        except notes_device_state_schema.NotesDeviceSchemaError as error:
            message = str(error)
            if message.startswith("Unsupported"):
                raise NotesDeviceStateError(message) from None
            if "incompatible" in message:
                raise NotesDeviceStateError(message) from None
            raise NotesDeviceStateError(
                "The private Notes device schema could not be initialized safely."
            ) from None
        except Exception:
            raise NotesDeviceStateError(
                "The private Notes device schema could not be initialized safely."
            ) from None

    def create_root(self, record: NotesSyncRootRecord) -> NotesSyncRootRecord:
        if type(record) is not NotesSyncRootRecord:
            raise TypeError("record must be a NotesSyncRootRecord.")
        timestamp = _now()
        with self.transaction(immediate=True) as connection:
            connection.execute(
                """
                INSERT INTO notes_sync_roots (
                    root_id, note_scope_id, logical_folder_id, canonical_path,
                    remote_origin_id, direction, state, cursor, last_status_code,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.root_id,
                    record.note_scope_id,
                    record.logical_folder_id,
                    record.canonical_path,
                    record.remote_origin_id,
                    record.direction.value,
                    record.state.value,
                    record.cursor,
                    record.last_status_code,
                    timestamp,
                    timestamp,
                ),
            )
        return record

    def get_root(self, root_id: str) -> NotesSyncRootRecord:
        validate_notes_sync_opaque_id(root_id, field_name="root_id")
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT root_id, note_scope_id, logical_folder_id, canonical_path,
                       direction, state, remote_origin_id, cursor, last_status_code
                FROM notes_sync_roots WHERE root_id = ?
                """,
                (root_id,),
            ).fetchone()
        if row is None:
            raise NotesDeviceStateError("The requested sync root does not exist.")
        return NotesSyncRootRecord(
            root_id=row[0],
            note_scope_id=row[1],
            logical_folder_id=row[2],
            canonical_path=row[3],
            direction=NotesSyncDirection(row[4]),
            state=NotesSyncRootState(row[5]),
            remote_origin_id=row[6],
            cursor=row[7],
            last_status_code=row[8],
        )

    def assign_root_folder(self, root_id: str, logical_folder_id: str) -> None:
        validate_notes_sync_opaque_id(root_id, field_name="root_id")
        validate_notes_sync_opaque_id(
            logical_folder_id,
            field_name="logical_folder_id",
        )
        with self.transaction(immediate=True) as connection:
            changed = connection.execute(
                """
                UPDATE notes_sync_roots
                SET logical_folder_id = ?, updated_at = ?
                WHERE root_id = ? AND state != 'disconnected'
                """,
                (logical_folder_id, _now(), root_id),
            ).rowcount
        if changed != 1:
            raise NotesDeviceStateError("The sync root cannot accept a folder owner.")

    def record_root_activation_recovery(
        self, root_id: str, logical_folder_id: str
    ) -> NotesSyncRootRecord:
        """Durably retain an orphan-risk folder under a revisitable root."""

        validate_notes_sync_opaque_id(root_id, field_name="root_id")
        validate_notes_sync_opaque_id(logical_folder_id, field_name="logical_folder_id")
        with self.transaction(immediate=True) as connection:
            root = connection.execute(
                """
                SELECT root_id, note_scope_id, canonical_path, direction,
                       remote_origin_id, cursor
                FROM notes_sync_roots
                WHERE root_id = ? AND state IN ('pending', 'paused')
                """,
                (root_id,),
            ).fetchone()
            if root is None:
                raise NotesDeviceStateError(
                    "The activation recovery owner could not be retained."
                )
            changed = connection.execute(
                """
                UPDATE notes_sync_roots
                SET logical_folder_id = ?, state = 'paused',
                    last_status_code = 'activation_recovery_required', updated_at = ?
                WHERE root_id = ? AND state IN ('pending', 'paused')
                """,
                (logical_folder_id, _now(), root_id),
            ).rowcount
        if changed != 1:
            raise NotesDeviceStateError(
                "The activation recovery owner could not be retained."
            )
        return NotesSyncRootRecord(
            root_id=root[0],
            note_scope_id=root[1],
            logical_folder_id=logical_folder_id,
            canonical_path=root[2],
            direction=NotesSyncDirection(root[3]),
            state=NotesSyncRootState.PAUSED,
            remote_origin_id=root[4],
            cursor=root[5],
            last_status_code="activation_recovery_required",
        )

    def activate_migration_candidate(
        self,
        root_id: str,
        logical_folder_id: str,
        binding_ids: tuple[str, ...],
    ) -> NotesSyncRootRecord:
        """Atomically admit one reviewed migrated root and its exact candidates."""

        validate_notes_sync_opaque_id(root_id, field_name="root_id")
        validate_notes_sync_opaque_id(logical_folder_id, field_name="logical_folder_id")
        if type(binding_ids) is not tuple:
            raise TypeError("binding_ids must be a tuple.")
        for binding_id in binding_ids:
            validate_notes_sync_opaque_id(binding_id, field_name="binding_id")
        if len(set(binding_ids)) != len(binding_ids):
            raise ValueError("binding_ids must be unique.")
        with self.transaction(immediate=True) as connection:
            root = connection.execute(
                """
                SELECT root_id, note_scope_id, logical_folder_id, canonical_path,
                       direction, state, remote_origin_id, cursor, last_status_code
                FROM notes_sync_roots WHERE root_id = ?
                """,
                (root_id,),
            ).fetchone()
            if (
                root is None
                or root[5] != NotesSyncRootState.PAUSED.value
                or root[8] != "migration_review_required"
                or root[2] not in (None, logical_folder_id)
            ):
                raise NotesDeviceStateError(
                    "The migrated sync root is not awaiting activation review."
                )
            current = tuple(
                row[0]
                for row in connection.execute(
                    """
                    SELECT binding_id FROM notes_sync_bindings
                    WHERE root_id = ? AND state = 'candidate'
                    ORDER BY binding_id
                    """,
                    (root_id,),
                ).fetchall()
            )
            total = connection.execute(
                "SELECT COUNT(*) FROM notes_sync_bindings WHERE root_id = ?",
                (root_id,),
            ).fetchone()[0]
            if current != tuple(sorted(binding_ids)) or total != len(binding_ids):
                raise NotesDeviceStateError(
                    "The reviewed migration candidate set is stale."
                )
            timestamp = _now()
            connection.execute(
                """
                UPDATE notes_sync_roots
                SET logical_folder_id = ?, state = 'active',
                    last_status_code = 'activating', updated_at = ?
                WHERE root_id = ? AND state = 'paused'
                """,
                (logical_folder_id, timestamp, root_id),
            )
            connection.execute(
                """
                UPDATE notes_sync_bindings
                SET state = 'active', updated_at = ?
                WHERE root_id = ? AND state = 'candidate'
                """,
                (timestamp, root_id),
            )
            activated = NotesSyncRootRecord(
                root_id=root[0],
                note_scope_id=root[1],
                logical_folder_id=logical_folder_id,
                canonical_path=root[3],
                direction=NotesSyncDirection(root[4]),
                state=NotesSyncRootState.ACTIVE,
                remote_origin_id=root[6],
                cursor=root[7],
                last_status_code="activating",
            )
        return activated

    def transition_root(
        self,
        root_id: str,
        state: NotesSyncRootState,
    ) -> NotesSyncRootRecord:
        validate_notes_sync_opaque_id(root_id, field_name="root_id")
        if type(state) is not NotesSyncRootState:
            raise TypeError("state must be a NotesSyncRootState.")
        with self.transaction(immediate=True) as connection:
            row = connection.execute(
                """
                SELECT state, logical_folder_id
                FROM notes_sync_roots WHERE root_id = ?
                """,
                (root_id,),
            ).fetchone()
            if row is None:
                raise NotesDeviceStateError("The requested sync root does not exist.")
            current_state = NotesSyncRootState(row[0])
            if state not in _ROOT_TRANSITIONS[current_state]:
                raise NotesDeviceStateError(
                    "The requested root transition is not allowed."
                )
            if state is NotesSyncRootState.ACTIVE and row[1] is None:
                raise NotesDeviceStateError(
                    "The requested root transition requires a logical folder owner."
                )
            timestamp = _now()
            if state is NotesSyncRootState.PAUSED:
                connection.execute(
                    """
                    UPDATE notes_sync_bindings
                    SET state = 'paused', updated_at = ?
                    WHERE root_id = ? AND state = 'active'
                    """,
                    (timestamp, root_id),
                )
            elif state is NotesSyncRootState.ACTIVE:
                connection.execute(
                    """
                    UPDATE notes_sync_bindings
                    SET state = 'active', updated_at = ?
                    WHERE root_id = ? AND state = 'paused'
                    """,
                    (timestamp, root_id),
                )
            elif state is NotesSyncRootState.DISCONNECTED:
                connection.execute(
                    """
                    UPDATE notes_sync_bindings
                    SET state = 'disconnected', updated_at = ?
                    WHERE root_id = ? AND state != 'disconnected'
                    """,
                    (timestamp, root_id),
                )
            changed = connection.execute(
                """
                UPDATE notes_sync_roots SET state = ?, updated_at = ?
                WHERE root_id = ? AND state = ?
                """,
                (state.value, timestamp, root_id, current_state.value),
            ).rowcount
        if changed != 1:
            raise NotesDeviceStateError("The requested root transition is stale.")
        return self.get_root(root_id)

    def update_root_status(
        self,
        root_id: str,
        status_code: str,
    ) -> NotesSyncRootRecord:
        """Persist one bounded path-free runtime status for a known root."""

        validate_notes_sync_opaque_id(root_id, field_name="root_id")
        validate_notes_sync_reason_code(status_code)
        with self.transaction(immediate=True) as connection:
            changed = connection.execute(
                """
                UPDATE notes_sync_roots
                SET last_status_code = ?, updated_at = ?
                WHERE root_id = ?
                """,
                (status_code, _now(), root_id),
            ).rowcount
        if changed != 1:
            raise NotesDeviceStateError("The requested sync root does not exist.")
        return self.get_root(root_id)

    def list_root_summaries(self) -> tuple[NotesSyncRootSummary, ...]:
        with self.transaction() as connection:
            rows = connection.execute(
                """
                SELECT root_id, direction, state, last_status_code
                FROM notes_sync_roots ORDER BY root_id
                """
            ).fetchall()
        return tuple(
            NotesSyncRootSummary(
                root_id=row[0],
                direction=NotesSyncDirection(row[1]),
                state=NotesSyncRootState(row[2]),
                last_status_code=row[3],
            )
            for row in rows
        )

    def create_binding(self, record: NotesSyncBindingRecord) -> NotesSyncBindingRecord:
        if type(record) is not NotesSyncBindingRecord:
            raise TypeError("record must be a NotesSyncBindingRecord.")
        timestamp = _now()
        with self.transaction(immediate=True) as connection:
            root = connection.execute(
                """
                SELECT note_scope_id, state, logical_folder_id
                FROM notes_sync_roots WHERE root_id = ?
                """,
                (record.root_id,),
            ).fetchone()
            if root is None or root[0] != record.note_scope_id:
                raise NotesDeviceStateError(
                    "A binding note scope must match its parent root."
                )
            if record.state is NotesSyncBindingState.ACTIVE:
                if root[1] != NotesSyncRootState.ACTIVE.value or root[2] is None:
                    raise NotesDeviceStateError(
                        "An active binding requires an active root with a folder owner."
                    )
            connection.execute(
                """
                INSERT INTO notes_sync_bindings (
                    binding_id, root_id, note_scope_id, note_id,
                    normalized_relative_path, stable_identity_digest, state,
                    utf8_bom, newline, final_newline, file_mode,
                    content_digest, note_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.binding_id,
                    record.root_id,
                    record.note_scope_id,
                    record.note_id,
                    record.normalized_relative_path,
                    record.stable_identity_digest,
                    record.state.value,
                    int(record.serialization.utf8_bom),
                    record.serialization.newline,
                    int(record.serialization.final_newline),
                    record.serialization.mode,
                    record.content_digest,
                    record.note_version,
                    timestamp,
                    timestamp,
                ),
            )
        return record

    @staticmethod
    def _binding_from_row(row: tuple[object, ...]) -> NotesSyncBindingRecord:
        return NotesSyncBindingRecord(
            binding_id=str(row[0]),
            root_id=str(row[1]),
            note_scope_id=str(row[2]),
            note_id=str(row[3]),
            normalized_relative_path=str(row[4]),
            stable_identity_digest=str(row[5]),
            state=NotesSyncBindingState(str(row[6])),
            serialization=NotesSyncSerializationProfile(
                utf8_bom=bool(row[7]),
                newline=str(row[8]),
                final_newline=bool(row[9]),
                mode=int(row[10]),
            ),
            content_digest=str(row[11]),
            note_version=int(row[12]),
        )

    def get_binding(self, binding_id: str) -> NotesSyncBindingRecord:
        validate_notes_sync_opaque_id(binding_id, field_name="binding_id")
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT binding_id, root_id, note_scope_id, note_id,
                       normalized_relative_path, stable_identity_digest, state,
                       utf8_bom, newline, final_newline, file_mode,
                       content_digest, note_version
                FROM notes_sync_bindings WHERE binding_id = ?
                """,
                (binding_id,),
            ).fetchone()
        if row is None:
            raise NotesDeviceStateError("The requested sync binding does not exist.")
        return self._binding_from_row(row)

    def list_bindings(self, root_id: str) -> tuple[NotesSyncBindingRecord, ...]:
        validate_notes_sync_opaque_id(root_id, field_name="root_id")
        with self.transaction() as connection:
            rows = connection.execute(
                """
                SELECT binding_id, root_id, note_scope_id, note_id,
                       normalized_relative_path, stable_identity_digest, state,
                       utf8_bom, newline, final_newline, file_mode,
                       content_digest, note_version
                FROM notes_sync_bindings WHERE root_id = ? ORDER BY binding_id
                """,
                (root_id,),
            ).fetchall()
        return tuple(self._binding_from_row(row) for row in rows)

    def list_binding_summaries(
        self,
        root_id: str,
    ) -> tuple[NotesSyncBindingSummary, ...]:
        validate_notes_sync_opaque_id(root_id, field_name="root_id")
        with self.transaction() as connection:
            rows = connection.execute(
                """
                SELECT binding_id, root_id, note_scope_id, note_id,
                       state, note_version
                FROM notes_sync_bindings WHERE root_id = ? ORDER BY binding_id
                """,
                (root_id,),
            ).fetchall()
        return tuple(
            NotesSyncBindingSummary(
                binding_id=row[0],
                root_id=row[1],
                note_scope_id=row[2],
                note_id=row[3],
                state=NotesSyncBindingState(row[4]),
                note_version=row[5],
            )
            for row in rows
        )

    def transition_binding(
        self,
        binding_id: str,
        state: NotesSyncBindingState,
    ) -> NotesSyncBindingRecord:
        current = self.get_binding(binding_id)
        if (
            type(state) is not NotesSyncBindingState
            or state not in _BINDING_TRANSITIONS[current.state]
        ):
            raise NotesDeviceStateError(
                "The requested binding transition is not allowed."
            )
        with self.transaction(immediate=True) as connection:
            if state is NotesSyncBindingState.ACTIVE:
                active_root = connection.execute(
                    """
                    SELECT 1 FROM notes_sync_roots
                    WHERE root_id = ? AND state = 'active'
                          AND logical_folder_id IS NOT NULL
                    """,
                    (current.root_id,),
                ).fetchone()
                if active_root is None:
                    raise NotesDeviceStateError(
                        "An active binding requires an active root with a folder owner."
                    )
            changed = connection.execute(
                """
                UPDATE notes_sync_bindings SET state = ?, updated_at = ?
                WHERE binding_id = ? AND state = ?
                """,
                (state.value, _now(), binding_id, current.state.value),
            ).rowcount
        if changed != 1:
            raise NotesDeviceStateError("The requested binding transition is stale.")
        return self.get_binding(binding_id)

    def create_operation(
        self,
        record: NotesSyncOperationRecord,
    ) -> NotesSyncOperationRecord:
        if type(record) is not NotesSyncOperationRecord:
            raise TypeError("record must be a NotesSyncOperationRecord.")
        if record.state is not NotesSyncOperationState.PENDING:
            raise NotesDeviceStateError(
                "A new sync operation must enter through the pending stage."
            )
        timestamp = _now()
        with self.transaction(immediate=True) as connection:
            if record.binding_id is not None:
                binding_root = connection.execute(
                    "SELECT root_id FROM notes_sync_bindings WHERE binding_id = ?",
                    (record.binding_id,),
                ).fetchone()
                if binding_root is None or binding_root[0] != record.root_id:
                    raise NotesDeviceStateError(
                        "A journal operation and its binding must use the same root."
                    )
            connection.execute(
                """
                INSERT INTO notes_sync_operations (
                    operation_id, root_id, binding_id, kind, state, reason_code,
                    observation_token, expected_note_version,
                    expected_file_digest, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.operation_id,
                    record.root_id,
                    record.binding_id,
                    record.kind,
                    record.state.value,
                    record.reason_code,
                    record.observation_token,
                    record.expected_note_version,
                    record.expected_file_digest,
                    timestamp,
                    timestamp,
                ),
            )
        return record

    def admit_operation_recovery(
        self,
        operation: NotesSyncOperationRecord,
        recovery: NotesSyncRecoveryRecord,
        *,
        capacity_bytes: int,
        now: int | None = None,
        retention_ns: int | None = None,
    ) -> NotesSyncRecoveryAdmission:
        """Atomically reserve recovery capacity and persist mutation intent."""

        if type(operation) is not NotesSyncOperationRecord:
            raise TypeError("operation must be a NotesSyncOperationRecord.")
        if type(recovery) is not NotesSyncRecoveryRecord:
            raise TypeError("recovery must be a NotesSyncRecoveryRecord.")
        if operation.state is not NotesSyncOperationState.PENDING:
            raise NotesDeviceStateError(
                "A new sync operation must enter through the pending stage."
            )
        if recovery.operation_id != operation.operation_id:
            raise NotesDeviceStateError(
                "Recovery and journal operation identifiers must match."
            )
        if type(capacity_bytes) is not int or capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive.")
        if now is None:
            now = _now()
        elif type(now) is not int or now <= 0:
            raise ValueError("now must be positive.")
        if retention_ns is not None and (
            type(retention_ns) is not int or retention_ns <= 0
        ):
            raise ValueError("retention_ns must be positive.")

        required = len(recovery.payload) + len(recovery.metadata)
        with self.transaction(immediate=True) as connection:
            existing_operation = connection.execute(
                """
                SELECT root_id, binding_id, kind, state, reason_code,
                       observation_token, expected_note_version,
                       expected_file_digest
                FROM notes_sync_operations WHERE operation_id = ?
                """,
                (operation.operation_id,),
            ).fetchone()
            existing_recovery = connection.execute(
                """
                SELECT recovery_id, operation_id, payload, metadata,
                       expires_at, created_at
                FROM notes_sync_recovery WHERE operation_id = ?
                """,
                (operation.operation_id,),
            ).fetchone()
            if existing_operation is not None or existing_recovery is not None:
                exact_operation = existing_operation == (
                    operation.root_id,
                    operation.binding_id,
                    operation.kind,
                    NotesSyncOperationState.RECOVERY_ADMITTED.value,
                    None,
                    operation.observation_token,
                    operation.expected_note_version,
                    operation.expected_file_digest,
                )
                exact_recovery = existing_recovery[:4] == (
                    recovery.recovery_id,
                    recovery.operation_id,
                    recovery.payload,
                    recovery.metadata,
                ) and (
                    (
                        retention_ns is None
                        and existing_recovery[4] == recovery.expires_at
                    )
                    or (
                        retention_ns is not None
                        and type(existing_recovery[5]) is int
                        and existing_recovery[4] == existing_recovery[5] + retention_ns
                    )
                )
                if not exact_operation or not exact_recovery:
                    raise NotesDeviceStateError(
                        "The recovery admission conflicts with durable state."
                    )
                used = connection.execute(
                    """
                    SELECT COALESCE(SUM(length(payload) + length(metadata)), 0)
                    FROM notes_sync_recovery
                    """
                ).fetchone()[0]
                return NotesSyncRecoveryAdmission(
                    admitted=True,
                    reason_code=None,
                    required_bytes=required,
                    available_bytes=max(
                        0,
                        capacity_bytes - (int(used) - required),
                    ),
                )
            connection.execute(
                """
                DELETE FROM notes_sync_recovery
                WHERE expires_at <= ?
                  AND operation_id IN (
                      SELECT operation_id FROM notes_sync_operations
                      WHERE state = 'completed'
                  )
                """,
                (now,),
            )
            used = connection.execute(
                """
                SELECT COALESCE(SUM(length(payload) + length(metadata)), 0)
                FROM notes_sync_recovery
                """
            ).fetchone()[0]
            available = max(0, capacity_bytes - int(used))
            if required > available:
                return NotesSyncRecoveryAdmission(
                    admitted=False,
                    reason_code="recovery_capacity_exceeded",
                    required_bytes=required,
                    available_bytes=available,
                )
            if operation.binding_id is not None:
                binding_root = connection.execute(
                    "SELECT root_id FROM notes_sync_bindings WHERE binding_id = ?",
                    (operation.binding_id,),
                ).fetchone()
                if binding_root is None or binding_root[0] != operation.root_id:
                    raise NotesDeviceStateError(
                        "A journal operation and its binding must use the same root."
                    )
            timestamp = _now()
            connection.execute(
                """
                INSERT INTO notes_sync_operations (
                    operation_id, root_id, binding_id, kind, state, reason_code,
                    observation_token, expected_note_version,
                    expected_file_digest, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    operation.operation_id,
                    operation.root_id,
                    operation.binding_id,
                    operation.kind,
                    operation.state.value,
                    operation.reason_code,
                    operation.observation_token,
                    operation.expected_note_version,
                    operation.expected_file_digest,
                    timestamp,
                    timestamp,
                ),
            )
            connection.execute(
                """
                INSERT INTO notes_sync_recovery (
                    recovery_id, operation_id, payload, metadata,
                    expires_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    recovery.recovery_id,
                    recovery.operation_id,
                    recovery.payload,
                    recovery.metadata,
                    (
                        recovery.expires_at
                        if retention_ns is None
                        else timestamp + retention_ns
                    ),
                    timestamp,
                ),
            )
            connection.execute(
                """
                UPDATE notes_sync_operations
                SET state = 'recovery_admitted', updated_at = ?
                WHERE operation_id = ? AND state = 'pending'
                """,
                (timestamp, operation.operation_id),
            )
        return NotesSyncRecoveryAdmission(
            admitted=True,
            reason_code=None,
            required_bytes=required,
            available_bytes=available,
        )

    def find_operation(self, operation_id: str) -> NotesSyncOperationRecord | None:
        """Return one operation, distinguishing absence from store failure."""

        validate_notes_sync_opaque_id(operation_id, field_name="operation_id")
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT operation_id, root_id, binding_id, kind, state,
                       reason_code, observation_token, expected_note_version,
                       expected_file_digest
                FROM notes_sync_operations WHERE operation_id = ?
                """,
                (operation_id,),
            ).fetchone()
        if row is None:
            return None
        return NotesSyncOperationRecord(
            operation_id=row[0],
            root_id=row[1],
            binding_id=row[2],
            kind=row[3],
            state=NotesSyncOperationState(row[4]),
            reason_code=row[5],
            observation_token=row[6],
            expected_note_version=row[7],
            expected_file_digest=row[8],
        )

    def get_operation(self, operation_id: str) -> NotesSyncOperationRecord:
        operation = self.find_operation(operation_id)
        if operation is None:
            raise NotesDeviceStateError("The requested sync operation does not exist.")
        return operation

    def list_incomplete_operations(self) -> tuple[NotesSyncOperationRecord, ...]:
        """Return durable work that still requires completion or attention."""

        with self.transaction() as connection:
            rows = connection.execute(
                """
                SELECT operation_id, root_id, binding_id, kind, state,
                       reason_code, observation_token, expected_note_version,
                       expected_file_digest
                FROM notes_sync_operations
                WHERE state != 'completed'
                ORDER BY created_at, operation_id
                """
            ).fetchall()
        return tuple(
            NotesSyncOperationRecord(
                operation_id=row[0],
                root_id=row[1],
                binding_id=row[2],
                kind=row[3],
                state=NotesSyncOperationState(row[4]),
                reason_code=row[5],
                observation_token=row[6],
                expected_note_version=row[7],
                expected_file_digest=row[8],
            )
            for row in rows
        )

    def list_resolution_history(
        self,
        root_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        now: int | None = None,
    ) -> tuple[NotesSyncResolutionHistoryRecord, ...]:
        """Return one bounded newest-first page of reviewed resolutions."""

        validate_notes_sync_opaque_id(root_id, field_name="root_id")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100.")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be non-negative.")
        if now is not None and (type(now) is not int or now <= 0):
            raise ValueError("now must be positive.")
        with self.transaction() as connection:
            rows = connection.execute(
                """
                SELECT operation_id, binding_id, kind, state, reason_code,
                       updated_at
                FROM notes_sync_operations
                WHERE root_id = ?
                  AND kind IN (
                    'resolve_keep_file', 'resolve_keep_note', 'resolve_keep_both'
                  )
                ORDER BY updated_at DESC, operation_id DESC
                LIMIT ? OFFSET ?
                """,
                (root_id, limit, offset),
            ).fetchall()
            result: list[NotesSyncResolutionHistoryRecord] = []
            for row in rows:
                operation_id, binding_id, kind, state, reason_code, updated_at = row
                undo_id = linked_undo_operation_id(root_id, operation_id)
                undo = connection.execute(
                    "SELECT state, reason_code FROM notes_sync_operations "
                    "WHERE operation_id = ? AND root_id = ? "
                    "AND kind = 'undo_resolution'",
                    (undo_id, root_id),
                ).fetchone()
                recovery = connection.execute(
                    "SELECT expires_at FROM notes_sync_recovery WHERE operation_id = ?",
                    (operation_id,),
                ).fetchone()
                operation_state = NotesSyncOperationState(state)
                result.append(
                    NotesSyncResolutionHistoryRecord(
                        operation_id=operation_id,
                        binding_id=binding_id,
                        kind=kind,
                        state=operation_state,
                        reason_code=reason_code,
                        completed_at=(
                            updated_at
                            if operation_state is NotesSyncOperationState.COMPLETED
                            else None
                        ),
                        updated_at=updated_at,
                        recovery_expires_at=(
                            recovery[0] if recovery is not None else None
                        ),
                        undo_state=(
                            NotesSyncOperationState(undo[0])
                            if undo is not None
                            else None
                        ),
                        undo_reason_code=(undo[1] if undo is not None else None),
                    )
                )
        return tuple(result)

    def transition_operation(
        self,
        operation_id: str,
        state: NotesSyncOperationState,
    ) -> NotesSyncOperationRecord:
        current = self.get_operation(operation_id)
        if (
            type(state) is not NotesSyncOperationState
            or state not in _OPERATION_TRANSITIONS[current.state]
        ):
            raise NotesDeviceStateError(
                "The requested operation transition is not allowed."
            )
        with self.transaction(immediate=True) as connection:
            changed = connection.execute(
                """
                UPDATE notes_sync_operations
                SET state = ?, reason_code = NULL, updated_at = ?
                WHERE operation_id = ? AND state = ?
                """,
                (state.value, _now(), operation_id, current.state.value),
            ).rowcount
        if changed != 1:
            raise NotesDeviceStateError("The requested operation transition is stale.")
        return self.get_operation(operation_id)

    def advance_conflict_substage(
        self,
        *,
        operation_id: str,
        recovery_id: str,
        expected_operation_state: NotesSyncOperationState,
        expected_substage: str,
        next_substage: str,
        expected_payload_digest: str,
        expected_metadata_length: int,
        folder_authority: tuple[str, int, str, int] | None = None,
        copy_authority: tuple[str, int] | None = None,
        placement_authority: tuple[str, int] | None = None,
    ) -> NotesSyncOperationRecord:
        """Advance one exact Keep-both checkpoint without growing recovery."""

        validate_notes_sync_opaque_id(operation_id, field_name="operation_id")
        validate_notes_sync_opaque_id(recovery_id, field_name="recovery_id")
        validate_notes_sync_digest(
            expected_payload_digest, field_name="expected_payload_digest"
        )
        if type(expected_operation_state) is not NotesSyncOperationState:
            raise TypeError("expected_operation_state must be an operation state.")
        if type(expected_metadata_length) is not int or expected_metadata_length <= 0:
            raise ValueError("expected_metadata_length must be positive.")
        try:
            stage_index = _CONFLICT_SUBSTAGES.index(expected_substage)
        except ValueError:
            raise NotesDeviceStateError("The conflict substage is corrupt.") from None
        if (
            stage_index + 1 >= len(_CONFLICT_SUBSTAGES)
            or _CONFLICT_SUBSTAGES[stage_index + 1] != next_substage
        ):
            raise NotesDeviceStateError(
                "The requested conflict substage transition is not allowed."
            )
        if expected_substage == "recovery_admitted":
            if type(folder_authority) is not tuple or len(folder_authority) != 4:
                raise NotesDeviceStateError("The folder authority is corrupt.")
            parent_id, parent_version, child_id, child_version = folder_authority
            validate_notes_sync_opaque_id(parent_id, field_name="parent_folder_id")
            validate_notes_sync_opaque_id(child_id, field_name="child_folder_id")
            if any(
                type(version) is not int
                or version < 0
                or len(str(version)) > _CONFLICT_VERSION_CAPACITY
                for version in (parent_version, child_version)
            ):
                raise NotesDeviceStateError("The folder authority is corrupt.")
        elif folder_authority is not None:
            raise NotesDeviceStateError("The folder authority is corrupt.")
        if expected_substage == "folders_established":
            if type(copy_authority) is not tuple or len(copy_authority) != 2:
                raise NotesDeviceStateError("The copy authority is corrupt.")
            copy_note_id, copy_version = copy_authority
            validate_notes_sync_opaque_id(copy_note_id, field_name="copy_note_id")
            if (
                type(copy_version) is not int
                or copy_version < 0
                or len(str(copy_version)) > _CONFLICT_VERSION_CAPACITY
            ):
                raise NotesDeviceStateError("The copy authority is corrupt.")
        elif copy_authority is not None:
            raise NotesDeviceStateError("The copy authority is corrupt.")
        if expected_substage == "copy_created":
            if type(placement_authority) is not tuple or len(placement_authority) != 2:
                raise NotesDeviceStateError("The placement authority is corrupt.")
            placement_id, placement_version = placement_authority
            validate_notes_sync_opaque_id(placement_id, field_name="placement_id")
            if (
                type(placement_version) is not int
                or placement_version < 0
                or len(str(placement_version)) > _CONFLICT_VERSION_CAPACITY
            ):
                raise NotesDeviceStateError("The placement authority is corrupt.")
        elif placement_authority is not None:
            raise NotesDeviceStateError("The placement authority is corrupt.")
        expected_current_state = _CONFLICT_SUBSTAGE_STATES[expected_substage]
        if expected_substage == "file_reverified":
            expected_current_state = NotesSyncOperationState.BINDING_UPDATED
        if expected_operation_state is not expected_current_state:
            raise NotesDeviceStateError("The conflict substage state is corrupt.")
        next_state = _CONFLICT_SUBSTAGE_STATES[next_substage]
        longest = max(map(len, _CONFLICT_SUBSTAGES))
        with self.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT operation.kind, operation.state, recovery.payload, "
                "recovery.metadata FROM notes_sync_operations AS operation "
                "JOIN notes_sync_recovery AS recovery "
                "ON recovery.operation_id = operation.operation_id "
                "WHERE operation.operation_id = ? AND recovery.recovery_id = ?",
                (operation_id, recovery_id),
            ).fetchone()
            if row is None:
                raise NotesDeviceStateError(
                    "The requested conflict recovery does not exist."
                )
            kind, state, payload, metadata = row
            if (
                kind != "resolve_keep_both"
                or state != expected_operation_state.value
                or type(payload) is not bytes
                or type(metadata) is not bytes
                or len(metadata) != expected_metadata_length
                or hashlib.sha256(payload).hexdigest() != expected_payload_digest
            ):
                raise NotesDeviceStateError("The conflict recovery is corrupt.")
            try:
                decoded = json.loads(metadata.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise NotesDeviceStateError(
                    "The conflict recovery is corrupt."
                ) from None
            expected_padding = " " * (longest - len(expected_substage))
            if (
                not isinstance(decoded, dict)
                or decoded.get("conflict_substage") != expected_substage
                or decoded.get("conflict_substage_padding") != expected_padding
                or decoded.get("recovery_payload_digest") != expected_payload_digest
            ):
                raise NotesDeviceStateError("The conflict substage is corrupt.")
            for prefix in ("conflict_parent", "conflict_root"):
                folder_id = decoded.get(f"{prefix}_actual_folder_id")
                folder_id_padding = decoded.get(f"{prefix}_actual_folder_id_padding")
                version = decoded.get(f"{prefix}_actual_folder_version")
                version_padding = decoded.get(f"{prefix}_actual_folder_version_padding")
                if (
                    type(folder_id) is not str
                    or type(folder_id_padding) is not str
                    or type(version) is not str
                    or type(version_padding) is not str
                    or folder_id_padding
                    != " " * (_CONFLICT_OPAQUE_ID_CAPACITY - len(folder_id))
                    or version_padding
                    != " " * (_CONFLICT_VERSION_CAPACITY - len(version))
                ):
                    raise NotesDeviceStateError("The folder authority is corrupt.")
            copy_version_value = decoded.get("conflict_copy_note_version")
            copy_version_padding = decoded.get("conflict_copy_note_version_padding")
            placement_id_value = decoded.get("conflict_placement_membership_id")
            placement_id_padding = decoded.get(
                "conflict_placement_membership_id_padding"
            )
            placement_version_value = decoded.get("conflict_placement_version")
            placement_version_padding = decoded.get(
                "conflict_placement_version_padding"
            )
            if (
                type(copy_version_value) is not str
                or type(copy_version_padding) is not str
                or type(placement_id_value) is not str
                or type(placement_id_padding) is not str
                or type(placement_version_value) is not str
                or type(placement_version_padding) is not str
                or copy_version_padding
                != " " * (_CONFLICT_VERSION_CAPACITY - len(copy_version_value))
                or placement_id_padding
                != " " * (_CONFLICT_OPAQUE_ID_CAPACITY - len(placement_id_value))
                or placement_version_padding
                != " " * (_CONFLICT_VERSION_CAPACITY - len(placement_version_value))
            ):
                raise NotesDeviceStateError("The effect authority is corrupt.")
            copy_checkpointed = bool(copy_version_value)
            placement_checkpointed = bool(
                placement_id_value and placement_version_value
            )
            try:
                parsed_copy_version = int(copy_version_value or "0")
                parsed_placement_version = int(placement_version_value or "0")
                if placement_id_value:
                    validate_notes_sync_opaque_id(
                        placement_id_value,
                        field_name="placement_id",
                    )
            except (TypeError, ValueError):
                raise NotesDeviceStateError(
                    "The effect authority is corrupt."
                ) from None
            if (
                (copy_version_value and str(parsed_copy_version) != copy_version_value)
                or parsed_copy_version < 0
                or (placement_version_value and not placement_id_value)
                or (placement_id_value and not placement_version_value)
                or (
                    placement_version_value
                    and str(parsed_placement_version) != placement_version_value
                )
                or parsed_placement_version < 0
                or copy_checkpointed
                != (stage_index >= _CONFLICT_SUBSTAGES.index("copy_created"))
                or placement_checkpointed
                != (stage_index >= _CONFLICT_SUBSTAGES.index("placement_created"))
            ):
                raise NotesDeviceStateError("The effect authority is corrupt.")
            if folder_authority is not None:
                for prefix, folder_id, version in (
                    ("conflict_parent", parent_id, parent_version),
                    ("conflict_root", child_id, child_version),
                ):
                    decoded[f"{prefix}_actual_folder_id"] = folder_id
                    decoded[f"{prefix}_actual_folder_id_padding"] = " " * (
                        _CONFLICT_OPAQUE_ID_CAPACITY - len(folder_id)
                    )
                    encoded_version = str(version)
                    decoded[f"{prefix}_actual_folder_version"] = encoded_version
                    decoded[f"{prefix}_actual_folder_version_padding"] = " " * (
                        _CONFLICT_VERSION_CAPACITY - len(encoded_version)
                    )
            if copy_authority is not None:
                if decoded.get("conflict_copy_note_id") != copy_note_id:
                    raise NotesDeviceStateError("The copy authority is corrupt.")
                encoded_version = str(copy_version)
                decoded["conflict_copy_note_version"] = encoded_version
                decoded["conflict_copy_note_version_padding"] = " " * (
                    _CONFLICT_VERSION_CAPACITY - len(encoded_version)
                )
            if placement_authority is not None:
                decoded["conflict_placement_membership_id"] = placement_id
                decoded["conflict_placement_membership_id_padding"] = " " * (
                    _CONFLICT_OPAQUE_ID_CAPACITY - len(placement_id)
                )
                encoded_version = str(placement_version)
                decoded["conflict_placement_version"] = encoded_version
                decoded["conflict_placement_version_padding"] = " " * (
                    _CONFLICT_VERSION_CAPACITY - len(encoded_version)
                )
            decoded["conflict_substage"] = next_substage
            decoded["conflict_substage_padding"] = " " * (longest - len(next_substage))
            replacement = json.dumps(
                decoded,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            if len(replacement) != expected_metadata_length:
                raise NotesDeviceStateError("The conflict recovery length drifted.")
            updated = connection.execute(
                "UPDATE notes_sync_recovery SET metadata = ? "
                "WHERE recovery_id = ? AND operation_id = ? AND metadata = ?",
                (replacement, recovery_id, operation_id, metadata),
            ).rowcount
            advanced = connection.execute(
                "UPDATE notes_sync_operations SET state = ?, reason_code = NULL, "
                "updated_at = ? WHERE operation_id = ? AND state = ?",
                (
                    next_state.value,
                    _now(),
                    operation_id,
                    expected_operation_state.value,
                ),
            ).rowcount
            if updated != 1 or advanced != 1:
                raise NotesDeviceStateError(
                    "The conflict substage transition is stale."
                )
        return self.get_operation(operation_id)

    def mark_operation_attention(
        self,
        operation_id: str,
        reason_code: str,
    ) -> NotesSyncOperationRecord:
        """Durably fence an admitted operation from automatic replay."""

        validate_notes_sync_opaque_id(operation_id, field_name="operation_id")
        selected_reason = validate_notes_sync_reason_code(reason_code)
        if selected_reason is None:
            raise ValueError("reason_code is required.")
        with self.transaction(immediate=True) as connection:
            changed = connection.execute(
                """
                UPDATE notes_sync_operations
                SET state = 'needs_attention', reason_code = ?, updated_at = ?
                WHERE operation_id = ? AND state != 'completed'
                """,
                (selected_reason, _now(), operation_id),
            ).rowcount
        if changed != 1:
            raise NotesDeviceStateError(
                "The requested operation cannot enter attention."
            )
        return self.get_operation(operation_id)

    def advance_undo_substage(
        self,
        *,
        operation_id: str,
        recovery_id: str,
        expected_substage: str,
        next_substage: str,
        expected_metadata_length: int,
    ) -> None:
        """Replace one padded linked-Undo checkpoint without changing capacity."""

        validate_notes_sync_opaque_id(operation_id, field_name="operation_id")
        validate_notes_sync_opaque_id(recovery_id, field_name="recovery_id")
        if type(expected_metadata_length) is not int or expected_metadata_length <= 0:
            raise ValueError("expected_metadata_length must be positive.")
        try:
            index = _UNDO_SUBSTAGES.index(expected_substage)
        except ValueError:
            raise NotesDeviceStateError("The Undo substage is corrupt.") from None
        if (
            index + 1 >= len(_UNDO_SUBSTAGES)
            or _UNDO_SUBSTAGES[index + 1] != next_substage
        ):
            raise NotesDeviceStateError(
                "The requested Undo substage transition is not allowed."
            )
        longest = max(map(len, _UNDO_SUBSTAGES))
        with self.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT operation.kind, recovery.payload, recovery.metadata "
                "FROM notes_sync_operations AS operation "
                "JOIN notes_sync_recovery AS recovery "
                "ON recovery.operation_id = operation.operation_id "
                "WHERE operation.operation_id = ? AND recovery.recovery_id = ?",
                (operation_id, recovery_id),
            ).fetchone()
            if row is None:
                raise NotesDeviceStateError("The linked Undo recovery does not exist.")
            kind, payload, metadata = row
            if (
                kind != "undo_resolution"
                or type(payload) is not bytes
                or type(metadata) is not bytes
                or len(metadata) != expected_metadata_length
            ):
                raise NotesDeviceStateError("The linked Undo recovery is corrupt.")
            try:
                decoded = json.loads(metadata.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise NotesDeviceStateError(
                    "The linked Undo recovery is corrupt."
                ) from None
            if (
                not isinstance(decoded, dict)
                or decoded.get("undo_payload_digest")
                != hashlib.sha256(payload).hexdigest()
                or decoded.get("undo_substage") != expected_substage
                or decoded.get("undo_substage_padding")
                != " " * (longest - len(expected_substage))
            ):
                raise NotesDeviceStateError("The linked Undo substage is corrupt.")
            decoded["undo_substage"] = next_substage
            decoded["undo_substage_padding"] = " " * (longest - len(next_substage))
            replacement = json.dumps(
                decoded,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            if len(replacement) != expected_metadata_length:
                raise NotesDeviceStateError("The linked Undo recovery length drifted.")
            changed = connection.execute(
                "UPDATE notes_sync_recovery SET metadata = ? "
                "WHERE recovery_id = ? AND operation_id = ? AND metadata = ?",
                (replacement, recovery_id, operation_id, metadata),
            ).rowcount
            if changed != 1:
                raise NotesDeviceStateError("The linked Undo substage is stale.")

    def mark_operation_partial_attention(
        self,
        operation_id: str,
        recovery_id: str,
        reason_code: str,
        metadata: bytes,
        *,
        capacity_bytes: int,
    ) -> NotesSyncOperationRecord:
        """Atomically persist private cleanup authority and fence replay."""

        validate_notes_sync_opaque_id(operation_id, field_name="operation_id")
        validate_notes_sync_opaque_id(recovery_id, field_name="recovery_id")
        selected_reason = validate_notes_sync_reason_code(reason_code)
        if selected_reason is None:
            raise ValueError("reason_code is required.")
        if type(metadata) is not bytes:
            raise TypeError("metadata must be bytes.")
        if type(capacity_bytes) is not int or capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive.")
        with self.transaction(immediate=True) as connection:
            current = connection.execute(
                """
                SELECT length(metadata) FROM notes_sync_recovery
                WHERE recovery_id = ? AND operation_id = ?
                """,
                (recovery_id, operation_id),
            ).fetchone()
            if current is None:
                raise NotesDeviceStateError(
                    "The requested recovery record does not exist."
                )
            used = int(
                connection.execute(
                    """
                    SELECT COALESCE(SUM(length(payload) + length(metadata)), 0)
                    FROM notes_sync_recovery
                    """
                ).fetchone()[0]
            )
            if used - int(current[0]) + len(metadata) > capacity_bytes:
                raise NotesDeviceStateError(
                    "The private recovery capacity cannot admit cleanup authority."
                )
            updated = connection.execute(
                """
                UPDATE notes_sync_recovery SET metadata = ?
                WHERE recovery_id = ? AND operation_id = ?
                """,
                (metadata, recovery_id, operation_id),
            ).rowcount
            fenced = connection.execute(
                """
                UPDATE notes_sync_operations
                SET state = 'needs_attention', reason_code = ?, updated_at = ?
                WHERE operation_id = ? AND state != 'completed'
                """,
                (selected_reason, _now(), operation_id),
            ).rowcount
            if updated != 1 or fenced != 1:
                raise NotesDeviceStateError(
                    "The requested partial operation cannot enter attention."
                )
        return self.get_operation(operation_id)

    def commit_binding_stage(
        self,
        operation_id: str,
        *,
        expected: NotesSyncBindingRecord,
        replacement: NotesSyncBindingRecord,
    ) -> NotesSyncOperationRecord:
        """Atomically update one binding baseline and its journal stage."""

        validate_notes_sync_opaque_id(operation_id, field_name="operation_id")
        if type(expected) is not NotesSyncBindingRecord:
            raise TypeError("expected must be a NotesSyncBindingRecord.")
        if type(replacement) is not NotesSyncBindingRecord:
            raise TypeError("replacement must be a NotesSyncBindingRecord.")
        if (
            replacement.binding_id != expected.binding_id
            or replacement.root_id != expected.root_id
            or replacement.note_scope_id != expected.note_scope_id
            or replacement.note_id != expected.note_id
        ):
            raise NotesDeviceStateError("A binding baseline cannot change ownership.")
        timestamp = _now()
        with self.transaction(immediate=True) as connection:
            changed = connection.execute(
                """
                UPDATE notes_sync_bindings
                SET normalized_relative_path = ?, stable_identity_digest = ?,
                    state = ?, utf8_bom = ?, newline = ?, final_newline = ?,
                    file_mode = ?, content_digest = ?, note_version = ?,
                    updated_at = ?
                WHERE binding_id = ? AND root_id = ? AND note_scope_id = ?
                  AND note_id = ? AND normalized_relative_path = ?
                  AND stable_identity_digest = ? AND state = ?
                  AND utf8_bom = ? AND newline = ? AND final_newline = ?
                  AND file_mode = ? AND content_digest = ? AND note_version = ?
                """,
                (
                    replacement.normalized_relative_path,
                    replacement.stable_identity_digest,
                    replacement.state.value,
                    int(replacement.serialization.utf8_bom),
                    replacement.serialization.newline,
                    int(replacement.serialization.final_newline),
                    replacement.serialization.mode,
                    replacement.content_digest,
                    replacement.note_version,
                    timestamp,
                    expected.binding_id,
                    expected.root_id,
                    expected.note_scope_id,
                    expected.note_id,
                    expected.normalized_relative_path,
                    expected.stable_identity_digest,
                    expected.state.value,
                    int(expected.serialization.utf8_bom),
                    expected.serialization.newline,
                    int(expected.serialization.final_newline),
                    expected.serialization.mode,
                    expected.content_digest,
                    expected.note_version,
                ),
            ).rowcount
            if changed != 1:
                raise NotesDeviceStateError("The requested binding baseline is stale.")
            advanced = connection.execute(
                """
                UPDATE notes_sync_operations
                SET state = 'binding_updated', updated_at = ?
                WHERE operation_id = ? AND binding_id = ?
                  AND state = 'second_authority_applied'
                """,
                (timestamp, operation_id, expected.binding_id),
            ).rowcount
            if advanced != 1:
                raise NotesDeviceStateError("The requested journal stage is stale.")
        return self.get_operation(operation_id)

    def create_binding_stage(
        self,
        operation_id: str,
        record: NotesSyncBindingRecord,
    ) -> NotesSyncOperationRecord:
        """Atomically activate one reviewed binding and its journal stage."""

        validate_notes_sync_opaque_id(operation_id, field_name="operation_id")
        if type(record) is not NotesSyncBindingRecord:
            raise TypeError("record must be a NotesSyncBindingRecord.")
        if record.state is not NotesSyncBindingState.ACTIVE:
            raise NotesDeviceStateError("A new sync binding must be active.")
        timestamp = _now()
        with self.transaction(immediate=True) as connection:
            root = connection.execute(
                """
                SELECT note_scope_id, state, logical_folder_id
                FROM notes_sync_roots WHERE root_id = ?
                """,
                (record.root_id,),
            ).fetchone()
            if (
                root is None
                or root[0] != record.note_scope_id
                or root[1] != NotesSyncRootState.ACTIVE.value
                or root[2] is None
            ):
                raise NotesDeviceStateError(
                    "An active binding requires its reviewed active root."
                )
            connection.execute(
                """
                INSERT INTO notes_sync_bindings (
                    binding_id, root_id, note_scope_id, note_id,
                    normalized_relative_path, stable_identity_digest, state,
                    utf8_bom, newline, final_newline, file_mode,
                    content_digest, note_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.binding_id,
                    record.root_id,
                    record.note_scope_id,
                    record.note_id,
                    record.normalized_relative_path,
                    record.stable_identity_digest,
                    record.state.value,
                    int(record.serialization.utf8_bom),
                    record.serialization.newline,
                    int(record.serialization.final_newline),
                    record.serialization.mode,
                    record.content_digest,
                    record.note_version,
                    timestamp,
                    timestamp,
                ),
            )
            advanced = connection.execute(
                """
                UPDATE notes_sync_operations
                SET binding_id = ?, state = 'binding_updated', updated_at = ?
                WHERE operation_id = ? AND binding_id IS NULL
                  AND state = 'second_authority_applied'
                """,
                (record.binding_id, timestamp, operation_id),
            ).rowcount
            if advanced != 1:
                raise NotesDeviceStateError("The requested journal stage is stale.")
        return self.get_operation(operation_id)

    def resolve_operation_disconnect(
        self,
        operation_id: str,
        *,
        binding_id: str,
    ) -> NotesSyncOperationRecord:
        """Atomically relinquish one binding and settle its attention entry."""

        validate_notes_sync_opaque_id(operation_id, field_name="operation_id")
        validate_notes_sync_opaque_id(binding_id, field_name="binding_id")
        timestamp = _now()
        with self.transaction(immediate=True) as connection:
            disconnected = connection.execute(
                """
                UPDATE notes_sync_bindings
                SET state = 'disconnected', updated_at = ?
                WHERE binding_id = ? AND state != 'disconnected'
                """,
                (timestamp, binding_id),
            ).rowcount
            completed = connection.execute(
                """
                UPDATE notes_sync_operations
                SET state = 'completed', reason_code = NULL, updated_at = ?
                WHERE operation_id = ? AND binding_id = ?
                  AND state = 'needs_attention'
                """,
                (timestamp, operation_id, binding_id),
            ).rowcount
            if disconnected != 1 or completed != 1:
                raise NotesDeviceStateError(
                    "The requested disconnect resolution is stale."
                )
        return self.get_operation(operation_id)

    def resolve_unbound_operation_disconnect(
        self,
        operation_id: str,
    ) -> NotesSyncOperationRecord:
        """Settle one reviewed create without claiming either created authority."""

        validate_notes_sync_opaque_id(operation_id, field_name="operation_id")
        with self.transaction(immediate=True) as connection:
            completed = connection.execute(
                """
                UPDATE notes_sync_operations
                SET state = 'completed', reason_code = NULL, updated_at = ?
                WHERE operation_id = ? AND binding_id IS NULL
                  AND state = 'needs_attention'
                """,
                (_now(), operation_id),
            ).rowcount
            if completed != 1:
                raise NotesDeviceStateError(
                    "The requested unbound disconnect resolution is stale."
                )
        return self.get_operation(operation_id)

    def put_recovery(self, record: NotesSyncRecoveryRecord) -> None:
        if type(record) is not NotesSyncRecoveryRecord:
            raise TypeError("record must be a NotesSyncRecoveryRecord.")
        with self.transaction(immediate=True) as connection:
            connection.execute(
                """
                INSERT INTO notes_sync_recovery (
                    recovery_id, operation_id, payload, metadata,
                    expires_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.recovery_id,
                    record.operation_id,
                    record.payload,
                    record.metadata,
                    record.expires_at,
                    _now(),
                ),
            )

    def load_recovery(self, recovery_id: str) -> NotesSyncRecoveryRecord:
        validate_notes_sync_opaque_id(recovery_id, field_name="recovery_id")
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT recovery_id, operation_id, payload, metadata, expires_at
                FROM notes_sync_recovery WHERE recovery_id = ?
                """,
                (recovery_id,),
            ).fetchone()
        if row is None:
            raise NotesDeviceStateError("The requested recovery record does not exist.")
        if type(row[2]) is not bytes or type(row[3]) is not bytes:
            raise NotesDeviceStateError("The requested recovery record is corrupt.")
        return NotesSyncRecoveryRecord(
            recovery_id=row[0],
            operation_id=row[1],
            payload=row[2],
            metadata=row[3],
            expires_at=row[4],
        )

    def find_operation_recovery(
        self,
        operation_id: str,
    ) -> NotesSyncRecoveryRecord | None:
        """Find the private recovery owned by an operation, preserving absence."""

        validate_notes_sync_opaque_id(operation_id, field_name="operation_id")
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT recovery_id FROM notes_sync_recovery
                WHERE operation_id = ?
                """,
                (operation_id,),
            ).fetchone()
        if row is None:
            return None
        return self.load_recovery(row[0])

    def load_operation_recovery(
        self,
        operation_id: str,
    ) -> NotesSyncRecoveryRecord:
        """Load the sole private recovery record owned by one operation."""

        recovery = self.find_operation_recovery(operation_id)
        if recovery is None:
            raise NotesDeviceStateError("The requested recovery record does not exist.")
        return recovery

    def record_legacy_migration(
        self,
        record: NotesSyncLegacyMigrationRecord,
    ) -> None:
        if type(record) is not NotesSyncLegacyMigrationRecord:
            raise TypeError("record must be a NotesSyncLegacyMigrationRecord.")
        timestamp = _now()
        with self.transaction(immediate=True) as connection:
            connection.execute(
                """
                INSERT INTO notes_sync_legacy_migrations (
                    migration_id, source_fingerprint, state, reason_code,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.migration_id,
                    record.source_fingerprint,
                    record.state,
                    record.reason_code,
                    timestamp,
                    timestamp,
                ),
            )

    def set_setting(self, setting: NotesSyncStoreSetting) -> None:
        if type(setting) is not NotesSyncStoreSetting:
            raise TypeError("setting must be a NotesSyncStoreSetting.")
        with self.transaction(immediate=True) as connection:
            connection.execute(
                """
                INSERT INTO notes_sync_store_settings (
                    setting_key, setting_value, updated_at
                ) VALUES (?, ?, ?)
                ON CONFLICT(setting_key) DO UPDATE SET
                    setting_value = excluded.setting_value,
                    updated_at = excluded.updated_at
                """,
                (setting.key, setting.value, _now()),
            )

    def get_setting(self, key: str) -> NotesSyncStoreSetting | None:
        if key not in _SETTING_KEYS:
            raise ValueError("setting key is not supported.")
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT setting_key, setting_value
                FROM notes_sync_store_settings WHERE setting_key = ?
                """,
                (key,),
            ).fetchone()
        if row is None:
            return None
        return NotesSyncStoreSetting(key=row[0], value=row[1])


__all__ = [
    "NotesDeviceStateError",
    "NotesDeviceStateStore",
    "NotesSyncBindingRecord",
    "NotesSyncBindingSummary",
    "NotesSyncLegacyMigrationRecord",
    "NotesSyncOperationRecord",
    "NotesSyncRecoveryRecord",
    "NotesSyncRecoveryAdmission",
    "NotesSyncResolutionHistoryRecord",
    "NotesSyncRootRecord",
    "NotesSyncRootSummary",
    "NotesSyncStoreSetting",
]
