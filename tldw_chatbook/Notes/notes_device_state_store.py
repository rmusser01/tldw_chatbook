"""Single connection, migration, and transaction owner for Notes device state."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
from tldw_chatbook.Notes import notes_device_state_schema
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncBindingState,
    NotesSyncDirection,
    NotesSyncOperationState,
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
    """Own the sole private connection and transaction seam for notes.sync_state."""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)

    def __repr__(self) -> str:
        return "NotesDeviceStateStore(<private>)"

    def _connect(
        self,
        *,
        read_only: bool = False,
        must_exist: bool = False,
    ) -> sqlite3.Connection:
        connection = connect_private_sqlite(
            "notes.sync_state",
            self._database_path,
            read_only=read_only,
            must_exist=must_exist,
        )
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        """Yield one writable connection inside a schema-ready transaction."""

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            notes_device_state_schema.initialize_notes_device_schema(connection)
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        """Atomically initialize or migrate this isolated private owner."""

        try:
            with self.transaction(immediate=True):
                pass
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

    def get_operation(self, operation_id: str) -> NotesSyncOperationRecord:
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
            raise NotesDeviceStateError("The requested sync operation does not exist.")
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
                UPDATE notes_sync_operations SET state = ?, updated_at = ?
                WHERE operation_id = ? AND state = ?
                """,
                (state.value, _now(), operation_id, current.state.value),
            ).rowcount
        if changed != 1:
            raise NotesDeviceStateError("The requested operation transition is stale.")
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
    "NotesSyncRootRecord",
    "NotesSyncRootSummary",
    "NotesSyncStoreSetting",
]
