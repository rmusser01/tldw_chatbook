"""Redacted repository for paused, device-private Notes sync roots."""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import cast
from uuid import uuid4

from tldw_chatbook.Notes.notes_sync_state_schema import (
    NotesSyncStateSchemaError,
    notes_sync_state_transaction,
)


MAX_SYNC_ROOTS = 64
MAX_SYNC_BINDINGS = 100_000
_MAX_PATH_LENGTH = 32_768
_MAX_DISPLAY_NAME_LENGTH = 255
_MAX_ID_LENGTH = 256
_MIN_SQLITE_INTEGER = -(2**63)
_MAX_SQLITE_INTEGER = 2**63 - 1
_DIRECTIONS = frozenset({"folder_to_notes", "notes_to_folder", "bidirectional"})
_DURABLE_DIRECTIONS = _DIRECTIONS | {"unspecified"}
_REASON_CODE_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_LOWER_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_ROOT_COLUMNS = """root_id, lexical_root_path, display_name, direction, state,
                   row_version, needs_rescan, reason_code, source_kind,
                   source_locator_digest, source_migration_id, created_at, updated_at"""
_BINDING_COLUMNS = """binding_id, root_id, note_id, lexical_relative_path, path_key,
                      state, row_version, needs_rescan, reason_code, source_kind,
                      source_locator_digest, source_migration_id, created_at, updated_at"""


class NotesSyncStateError(RuntimeError):
    """Report a bounded failure without private sync-state details."""


class SyncStateConflictError(NotesSyncStateError):
    """Report a stale version or prohibited lifecycle transition."""


class SyncStateCapacityError(NotesSyncStateError):
    """Report exhaustion of a fixed sync-state safety ceiling."""


class SyncStateCorruptionError(NotesSyncStateError):
    """Report a malformed durable projection without disclosing its values."""


class SyncRootState(StrEnum):
    """Representable v2 lifecycle states for one lasting-sync root."""

    CANDIDATE = "candidate"
    PAUSED = "paused"
    DISCONNECTED = "disconnected"


class SyncBindingState(StrEnum):
    """Representable v2 lifecycle states for one provisional binding."""

    CANDIDATE = "candidate"
    NEEDS_ATTENTION = "needs_attention"
    DISCONNECTED = "disconnected"


@dataclass(frozen=True, slots=True, repr=False)
class SyncRootRecord:
    """Immutable exact projection of one private ``sync_roots`` row."""

    root_id: str = field(repr=False)
    lexical_root_path: str = field(repr=False)
    display_name: str
    direction: str
    state: SyncRootState
    row_version: int
    needs_rescan: bool
    reason_code: str | None
    source_kind: str | None = field(repr=False)
    source_locator_digest: str | None = field(repr=False)
    source_migration_id: str | None = field(repr=False)
    created_at: int
    updated_at: int

    def __repr__(self) -> str:
        """Return a diagnostic representation containing no private identity."""
        return (
            f"SyncRootRecord(state={self.state.value!r}, "
            f"row_version={self.row_version!r}, "
            f"needs_rescan={self.needs_rescan!r}, "
            f"reason_code={self.reason_code!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class SyncBindingRecord:
    """Immutable exact projection of one private ``sync_bindings`` row."""

    binding_id: str = field(repr=False)
    root_id: str = field(repr=False)
    note_id: str = field(repr=False)
    lexical_relative_path: str = field(repr=False)
    path_key: str | None = field(repr=False)
    state: SyncBindingState
    row_version: int
    needs_rescan: bool
    reason_code: str | None
    source_kind: str | None = field(repr=False)
    source_locator_digest: str | None = field(repr=False)
    source_migration_id: str | None = field(repr=False)
    created_at: int
    updated_at: int

    def __repr__(self) -> str:
        """Return a diagnostic representation containing no private identity."""
        return (
            f"SyncBindingRecord(state={self.state.value!r}, "
            f"row_version={self.row_version!r}, "
            f"needs_rescan={self.needs_rescan!r}, "
            f"reason_code={self.reason_code!r})"
        )


def _validate_text(
    value: object,
    *,
    field_name: str,
    maximum: int,
) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be text.")
    if not value or len(value) > maximum or "\x00" in value:
        raise ValueError(f"{field_name} violates its bounded text contract.")
    return value


def _validate_direction(direction: object) -> str:
    validated = _validate_text(direction, field_name="direction", maximum=64)
    if validated not in _DIRECTIONS:
        raise ValueError("direction must be a supported manual sync direction.")
    return validated


def _validate_root_id(root_id: object) -> str:
    return _validate_text(root_id, field_name="root_id", maximum=_MAX_ID_LENGTH)


def _validate_binding_id(binding_id: object) -> str:
    return _validate_text(
        binding_id,
        field_name="binding_id",
        maximum=_MAX_ID_LENGTH,
    )


def _validate_note_id(note_id: object) -> str:
    return _validate_text(note_id, field_name="note_id", maximum=_MAX_ID_LENGTH)


def _validate_expected_version(expected_version: object) -> int:
    if type(expected_version) is not int:
        raise TypeError("expected_version must be an integer.")
    if not 1 <= expected_version <= _MAX_SQLITE_INTEGER:
        raise ValueError("expected_version must be a bounded positive integer.")
    return expected_version


def _validate_reason_code(reason_code: object) -> str:
    if type(reason_code) is not str:
        raise TypeError("reason_code must be text.")
    if _REASON_CODE_PATTERN.fullmatch(reason_code) is None:
        raise ValueError("reason_code must be a bounded lowercase ASCII machine token.")
    return reason_code


def _timestamp_after(previous: int = 0) -> int:
    observed = time.time_ns()
    if type(observed) is not int or not (
        _MIN_SQLITE_INTEGER <= observed <= _MAX_SQLITE_INTEGER
    ):
        raise NotesSyncStateError(
            "The system clock is outside the durable sync-state timestamp range."
        )
    if type(previous) is not int or not 0 <= previous <= _MAX_SQLITE_INTEGER:
        raise SyncStateCorruptionError(
            "A private sync-state timestamp is outside the canonical range."
        )
    if previous == _MAX_SQLITE_INTEGER:
        raise NotesSyncStateError(
            "The durable sync-state timestamp cannot be advanced."
        )
    return max(1, observed, previous + 1)


def _root_record(row: tuple[object, ...]) -> SyncRootRecord:
    failure: SyncStateCorruptionError | None = None
    try:
        if len(row) != 13:
            raise ValueError
        typed_row = cast(
            tuple[
                str,
                str,
                str,
                str,
                str,
                int,
                int,
                str | None,
                str | None,
                str | None,
                str | None,
                int,
                int,
            ],
            row,
        )
        (
            root_id,
            lexical_root_path,
            display_name,
            direction,
            state,
            row_version,
            needs_rescan,
            reason_code,
            source_kind,
            source_locator_digest,
            source_migration_id,
            created_at,
            updated_at,
        ) = typed_row
        if not all(
            type(value) is str
            for value in (root_id, lexical_root_path, display_name, direction, state)
        ):
            raise TypeError
        if not all(
            type(value) is int
            for value in (row_version, needs_rescan, created_at, updated_at)
        ):
            raise TypeError
        if needs_rescan not in (0, 1):
            raise ValueError
        if not 1 <= len(root_id) <= _MAX_ID_LENGTH or "\x00" in root_id:
            raise ValueError
        if (
            not 1 <= len(lexical_root_path) <= _MAX_PATH_LENGTH
            or "\x00" in lexical_root_path
        ):
            raise ValueError
        if (
            not 1 <= len(display_name) <= _MAX_DISPLAY_NAME_LENGTH
            or "\x00" in display_name
        ):
            raise ValueError
        if direction not in _DURABLE_DIRECTIONS:
            raise ValueError
        if row_version <= 0 or created_at <= 0 or updated_at <= 0:
            raise ValueError
        for optional_text in (
            reason_code,
            source_kind,
            source_locator_digest,
            source_migration_id,
        ):
            if optional_text is not None and type(optional_text) is not str:
                raise TypeError
        if (
            reason_code is not None
            and _REASON_CODE_PATTERN.fullmatch(reason_code) is None
        ):
            raise ValueError
        source_fields = (
            source_kind,
            source_locator_digest,
            source_migration_id,
        )
        if any(value is None for value in source_fields) != all(
            value is None for value in source_fields
        ):
            raise ValueError
        if source_kind is not None and (
            source_kind != "legacy_notes_sync_v1"
            or source_locator_digest is None
            or _LOWER_DIGEST_PATTERN.fullmatch(source_locator_digest) is None
            or source_migration_id is None
            or len(source_migration_id) != 36
            or "\x00" in source_migration_id
        ):
            raise ValueError
        if direction == "unspecified" and not (
            source_kind == "legacy_notes_sync_v1"
            and needs_rescan == 1
            and reason_code == "legacy_direction_invalid"
        ):
            raise ValueError
        return SyncRootRecord(
            root_id=root_id,
            lexical_root_path=lexical_root_path,
            display_name=display_name,
            direction=direction,
            state=SyncRootState(state),
            row_version=row_version,
            needs_rescan=bool(needs_rescan),
            reason_code=reason_code,
            source_kind=source_kind,
            source_locator_digest=source_locator_digest,
            source_migration_id=source_migration_id,
            created_at=created_at,
            updated_at=updated_at,
        )
    except (IndexError, TypeError, ValueError):
        failure = SyncStateCorruptionError(
            "A private sync-root record is incompatible with canonical v2."
        )
    if failure is None:
        raise AssertionError("Sync-root projection did not return or fail.")
    raise failure from None


def _binding_record(row: tuple[object, ...]) -> SyncBindingRecord:
    failure: SyncStateCorruptionError | None = None
    try:
        if len(row) != 14:
            raise ValueError
        typed_row = cast(
            tuple[
                str,
                str,
                str,
                str,
                str | None,
                str,
                int,
                int,
                str | None,
                str | None,
                str | None,
                str | None,
                int,
                int,
            ],
            row,
        )
        (
            binding_id,
            root_id,
            note_id,
            lexical_relative_path,
            path_key,
            state,
            row_version,
            needs_rescan,
            reason_code,
            source_kind,
            source_locator_digest,
            source_migration_id,
            created_at,
            updated_at,
        ) = typed_row
        if not all(
            type(value) is str
            for value in (
                binding_id,
                root_id,
                note_id,
                lexical_relative_path,
                state,
            )
        ):
            raise TypeError
        if not all(
            type(value) is int
            for value in (row_version, needs_rescan, created_at, updated_at)
        ):
            raise TypeError
        if needs_rescan not in (0, 1):
            raise ValueError
        for identifier in (binding_id, root_id, note_id):
            if not 1 <= len(identifier) <= _MAX_ID_LENGTH or "\x00" in identifier:
                raise ValueError
        if (
            not 1 <= len(lexical_relative_path) <= _MAX_PATH_LENGTH
            or "\x00" in lexical_relative_path
        ):
            raise ValueError
        if path_key is not None and (
            type(path_key) is not str
            or not 1 <= len(path_key) <= _MAX_PATH_LENGTH
            or "\x00" in path_key
        ):
            raise ValueError
        if row_version <= 0 or created_at <= 0 or updated_at <= 0:
            raise ValueError
        if reason_code is not None and (
            type(reason_code) is not str
            or _REASON_CODE_PATTERN.fullmatch(reason_code) is None
        ):
            raise ValueError
        source_fields = (
            source_kind,
            source_locator_digest,
            source_migration_id,
        )
        if any(value is None for value in source_fields) != all(
            value is None for value in source_fields
        ):
            raise ValueError
        if source_kind is not None and (
            type(source_kind) is not str
            or source_kind != "legacy_notes_sync_v1"
            or type(source_locator_digest) is not str
            or _LOWER_DIGEST_PATTERN.fullmatch(source_locator_digest) is None
            or type(source_migration_id) is not str
            or len(source_migration_id) != 36
            or "\x00" in source_migration_id
        ):
            raise ValueError
        return SyncBindingRecord(
            binding_id=binding_id,
            root_id=root_id,
            note_id=note_id,
            lexical_relative_path=lexical_relative_path,
            path_key=path_key,
            state=SyncBindingState(state),
            row_version=row_version,
            needs_rescan=bool(needs_rescan),
            reason_code=reason_code,
            source_kind=source_kind,
            source_locator_digest=source_locator_digest,
            source_migration_id=source_migration_id,
            created_at=created_at,
            updated_at=updated_at,
        )
    except (IndexError, TypeError, ValueError):
        failure = SyncStateCorruptionError(
            "A private sync-binding record is incompatible with canonical v2."
        )
    if failure is None:
        raise AssertionError("Sync-binding projection did not return or fail.")
    raise failure from None


def _select_root(
    connection: sqlite3.Connection,
    root_id: str,
) -> SyncRootRecord | None:
    row = connection.execute(
        f"SELECT {_ROOT_COLUMNS} FROM sync_roots WHERE root_id = ?",
        (root_id,),
    ).fetchone()
    return None if row is None else _root_record(row)


def _require_root(
    connection: sqlite3.Connection,
    root_id: str,
) -> SyncRootRecord:
    record = _select_root(connection, root_id)
    if record is None:
        raise NotesSyncStateError("Sync root was not found.")
    return record


def _require_advanceable_root(record: SyncRootRecord) -> None:
    if record.row_version >= _MAX_SQLITE_INTEGER:
        raise NotesSyncStateError("A private sync-root version cannot be advanced.")


def _select_binding(
    connection: sqlite3.Connection,
    binding_id: str,
) -> SyncBindingRecord | None:
    row = connection.execute(
        f"SELECT {_BINDING_COLUMNS} FROM sync_bindings WHERE binding_id = ?",
        (binding_id,),
    ).fetchone()
    return None if row is None else _binding_record(row)


def _require_binding(
    connection: sqlite3.Connection,
    binding_id: str,
) -> SyncBindingRecord:
    record = _select_binding(connection, binding_id)
    if record is None:
        raise NotesSyncStateError("Sync binding was not found.")
    return record


def _require_advanceable_binding(record: SyncBindingRecord) -> None:
    if record.row_version >= _MAX_SQLITE_INTEGER:
        raise NotesSyncStateError("A private sync-binding version cannot be advanced.")


def _ensure_root_children_consistent(
    connection: sqlite3.Connection,
    root_id: str | None = None,
) -> None:
    predicate = "" if root_id is None else " AND root.root_id = ?"
    parameters: tuple[str, ...] = () if root_id is None else (root_id,)
    contradiction = connection.execute(
        """SELECT 1
           FROM sync_bindings AS binding
           JOIN sync_roots AS root ON root.root_id = binding.root_id
           WHERE root.state = 'disconnected'
             AND binding.state <> 'disconnected'"""
        + predicate
        + " LIMIT 1",
        parameters,
    ).fetchone()
    if contradiction is not None:
        raise SyncStateCorruptionError(
            "Private sync root and binding states contradict canonical v2."
        )


def _require_live_parent(
    connection: sqlite3.Connection,
    root_id: str,
    *,
    existing_child: bool = False,
) -> SyncRootRecord:
    root = _select_root(connection, root_id)
    if root is None:
        if existing_child:
            raise SyncStateCorruptionError(
                "A private sync binding has no canonical parent root."
            )
        raise NotesSyncStateError("Sync root was not found.")
    if root.state is SyncRootState.DISCONNECTED:
        if existing_child:
            raise SyncStateCorruptionError(
                "Private sync root and binding states contradict canonical v2."
            )
        raise SyncStateConflictError("A disconnected sync root is terminal.")
    return root


def _execute_mutation(
    connection: sqlite3.Connection,
    statement: str,
    parameters: tuple[object, ...],
) -> sqlite3.Cursor:
    failure: SyncStateConflictError | None = None
    try:
        return connection.execute(statement, parameters)
    except sqlite3.IntegrityError:
        failure = SyncStateConflictError(
            "Private sync-state ownership conflicts with this operation."
        )
    if failure is None:
        raise AssertionError("Private sync-state mutation did not return or fail.")
    raise failure from None


@contextmanager
def _repository_transaction(
    database_path: Path,
    *,
    immediate: bool = False,
) -> Iterator[sqlite3.Connection]:
    failure: NotesSyncStateError | None = None
    try:
        with notes_sync_state_transaction(
            database_path,
            immediate=immediate,
        ) as connection:
            yield connection
    except sqlite3.IntegrityError:
        failure = SyncStateConflictError(
            "Private sync-state ownership conflicts with this operation."
        )
    except (NotesSyncStateSchemaError, sqlite3.Error):
        failure = NotesSyncStateError(
            "The private sync-state operation could not be completed."
        )
    if failure is not None:
        raise failure from None


class NotesSyncStateRepository:
    """Persist bounded paused-root state through the shared schema owner."""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)

    def __repr__(self) -> str:
        return "NotesSyncStateRepository(<private>)"

    def create_candidate_root(
        self,
        lexical_root_path: str,
        display_name: str,
        direction: str,
    ) -> SyncRootRecord:
        """Create one manual candidate without inspecting or normalizing its path."""
        validated_path = _validate_text(
            lexical_root_path,
            field_name="lexical_root_path",
            maximum=_MAX_PATH_LENGTH,
        )
        validated_name = _validate_text(
            display_name,
            field_name="display_name",
            maximum=_MAX_DISPLAY_NAME_LENGTH,
        )
        validated_direction = _validate_direction(direction)
        root_id = str(uuid4())
        created_at = _timestamp_after()
        with _repository_transaction(self._database_path, immediate=True) as connection:
            live_count = connection.execute(
                "SELECT count(*) FROM sync_roots WHERE state <> 'disconnected'"
            ).fetchone()[0]
            if type(live_count) is not int:
                raise SyncStateCorruptionError(
                    "The private sync-root count is incompatible with canonical v2."
                )
            if live_count >= MAX_SYNC_ROOTS:
                raise SyncStateCapacityError(
                    f"Live sync-root capacity of {MAX_SYNC_ROOTS} is exhausted."
                )
            _execute_mutation(
                connection,
                """INSERT INTO sync_roots (
                       root_id, lexical_root_path, display_name, direction, state,
                       row_version, needs_rescan, reason_code, source_kind,
                       source_locator_digest, source_migration_id, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, 'candidate', 1, 1, NULL, NULL, NULL, NULL, ?, ?)""",
                (
                    root_id,
                    validated_path,
                    validated_name,
                    validated_direction,
                    created_at,
                    created_at,
                ),
            )
            return _require_root(connection, root_id)

    def get_root(self, root_id: str) -> SyncRootRecord:
        """Return one root or raise a redacted missing-root error."""
        validated_root_id = _validate_root_id(root_id)
        with _repository_transaction(self._database_path) as connection:
            record = _require_root(connection, validated_root_id)
            _ensure_root_children_consistent(connection, validated_root_id)
            return record

    def list_roots(self) -> tuple[SyncRootRecord, ...]:
        """Return every root in stable creation order without raw SQLite rows."""
        with _repository_transaction(self._database_path) as connection:
            _ensure_root_children_consistent(connection)
            rows = connection.execute(
                f"SELECT {_ROOT_COLUMNS} FROM sync_roots ORDER BY created_at, root_id"
            ).fetchall()
            return tuple(_root_record(row) for row in rows)

    def create_provisional_binding(
        self,
        root_id: str,
        note_id: str,
        lexical_relative_path: str,
    ) -> SyncBindingRecord:
        """Create one provisional binding under a live paused or candidate root."""
        validated_root_id = _validate_root_id(root_id)
        validated_note_id = _validate_note_id(note_id)
        validated_path = _validate_text(
            lexical_relative_path,
            field_name="lexical_relative_path",
            maximum=_MAX_PATH_LENGTH,
        )
        binding_id = str(uuid4())
        created_at = _timestamp_after()
        with _repository_transaction(self._database_path, immediate=True) as connection:
            _require_live_parent(connection, validated_root_id)
            live_count = connection.execute(
                "SELECT count(*) FROM sync_bindings WHERE state <> 'disconnected'"
            ).fetchone()[0]
            if type(live_count) is not int:
                raise SyncStateCorruptionError(
                    "The private sync-binding count is incompatible with canonical v2."
                )
            if live_count >= MAX_SYNC_BINDINGS:
                raise SyncStateCapacityError(
                    f"Live sync-binding capacity of {MAX_SYNC_BINDINGS} is exhausted."
                )
            owner = connection.execute(
                """SELECT 1 FROM sync_bindings
                   WHERE note_id = ? AND state <> 'disconnected' LIMIT 1""",
                (validated_note_id,),
            ).fetchone()
            if owner is not None:
                raise SyncStateConflictError(
                    "A live sync binding already owns this Database Note."
                )
            _execute_mutation(
                connection,
                """INSERT INTO sync_bindings (
                       binding_id, root_id, note_id, lexical_relative_path, path_key,
                       state, row_version, needs_rescan, reason_code, source_kind,
                       source_locator_digest, source_migration_id, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, NULL, 'candidate', 1, 1, NULL,
                             NULL, NULL, NULL, ?, ?)""",
                (
                    binding_id,
                    validated_root_id,
                    validated_note_id,
                    validated_path,
                    created_at,
                    created_at,
                ),
            )
            return _require_binding(connection, binding_id)

    def get_binding(self, binding_id: str) -> SyncBindingRecord:
        """Return one binding or raise a redacted missing-binding error."""
        validated_binding_id = _validate_binding_id(binding_id)
        with _repository_transaction(self._database_path) as connection:
            record = _require_binding(connection, validated_binding_id)
            root = _select_root(connection, record.root_id)
            if root is None:
                raise SyncStateCorruptionError(
                    "A private sync binding has no canonical parent root."
                )
            if (
                root.state is SyncRootState.DISCONNECTED
                and record.state is not SyncBindingState.DISCONNECTED
            ):
                raise SyncStateCorruptionError(
                    "Private sync root and binding states contradict canonical v2."
                )
            return record

    def list_bindings(self, *, root_id: str) -> tuple[SyncBindingRecord, ...]:
        """Return one root's bindings in stable creation order."""
        validated_root_id = _validate_root_id(root_id)
        with _repository_transaction(self._database_path) as connection:
            _require_root(connection, validated_root_id)
            _ensure_root_children_consistent(connection, validated_root_id)
            rows = connection.execute(
                f"""SELECT {_BINDING_COLUMNS} FROM sync_bindings
                    WHERE root_id = ? ORDER BY created_at, binding_id""",
                (validated_root_id,),
            ).fetchall()
            return tuple(_binding_record(row) for row in rows)

    def update_provisional_binding(
        self,
        binding_id: str,
        expected_version: int,
        *,
        lexical_relative_path: str | None = None,
        path_key: str | None = None,
    ) -> SyncBindingRecord:
        """Compare-and-set provisional path metadata without deriving identity."""
        validated_binding_id = _validate_binding_id(binding_id)
        validated_version = _validate_expected_version(expected_version)
        if lexical_relative_path is None and path_key is None:
            raise ValueError("At least one provisional-binding field must be updated.")
        validated_path = (
            None
            if lexical_relative_path is None
            else _validate_text(
                lexical_relative_path,
                field_name="lexical_relative_path",
                maximum=_MAX_PATH_LENGTH,
            )
        )
        validated_key = (
            None
            if path_key is None
            else _validate_text(
                path_key,
                field_name="path_key",
                maximum=_MAX_PATH_LENGTH,
            )
        )
        with _repository_transaction(self._database_path, immediate=True) as connection:
            current = _require_binding(connection, validated_binding_id)
            if current.state is SyncBindingState.DISCONNECTED:
                raise SyncStateConflictError("A disconnected sync binding is terminal.")
            _require_advanceable_binding(current)
            _require_live_parent(connection, current.root_id, existing_child=True)
            if validated_key is not None:
                owner = connection.execute(
                    """SELECT 1 FROM sync_bindings
                       WHERE root_id = ? AND path_key = ?
                         AND binding_id <> ? AND state <> 'disconnected'
                       LIMIT 1""",
                    (current.root_id, validated_key, validated_binding_id),
                ).fetchone()
                if owner is not None:
                    raise SyncStateConflictError(
                        "A live sync binding already owns this path key."
                    )
            changed = _execute_mutation(
                connection,
                """UPDATE sync_bindings
                   SET lexical_relative_path = COALESCE(?, lexical_relative_path),
                       path_key = COALESCE(?, path_key),
                       row_version = row_version + 1, updated_at = ?
                   WHERE binding_id = ? AND row_version = ?
                     AND state <> 'disconnected'""",
                (
                    validated_path,
                    validated_key,
                    _timestamp_after(current.updated_at),
                    validated_binding_id,
                    validated_version,
                ),
            ).rowcount
            if changed != 1:
                raise SyncStateConflictError(
                    "Sync binding version or lifecycle state conflicts with this update."
                )
            return _require_binding(connection, validated_binding_id)

    def mark_binding_needs_attention(
        self,
        binding_id: str,
        expected_version: int,
        reason_code: str,
    ) -> SyncBindingRecord:
        """Compare-and-set one live binding into its review-required state."""
        validated_binding_id = _validate_binding_id(binding_id)
        validated_version = _validate_expected_version(expected_version)
        validated_reason = _validate_reason_code(reason_code)
        with _repository_transaction(self._database_path, immediate=True) as connection:
            current = _require_binding(connection, validated_binding_id)
            if current.state is SyncBindingState.DISCONNECTED:
                raise SyncStateConflictError("A disconnected sync binding is terminal.")
            _require_advanceable_binding(current)
            _require_live_parent(connection, current.root_id, existing_child=True)
            changed = _execute_mutation(
                connection,
                """UPDATE sync_bindings
                   SET state = 'needs_attention', needs_rescan = 1, reason_code = ?,
                       row_version = row_version + 1, updated_at = ?
                   WHERE binding_id = ? AND row_version = ?
                     AND state <> 'disconnected'""",
                (
                    validated_reason,
                    _timestamp_after(current.updated_at),
                    validated_binding_id,
                    validated_version,
                ),
            ).rowcount
            if changed != 1:
                raise SyncStateConflictError(
                    "Sync binding version or lifecycle state conflicts with review."
                )
            return _require_binding(connection, validated_binding_id)

    def disconnect_binding(
        self,
        binding_id: str,
        expected_version: int,
    ) -> SyncBindingRecord:
        """Compare-and-set one binding into its terminal disconnected state."""
        validated_binding_id = _validate_binding_id(binding_id)
        validated_version = _validate_expected_version(expected_version)
        with _repository_transaction(self._database_path, immediate=True) as connection:
            current = _require_binding(connection, validated_binding_id)
            if current.state is SyncBindingState.DISCONNECTED:
                raise SyncStateConflictError("A disconnected sync binding is terminal.")
            _require_advanceable_binding(current)
            _require_live_parent(connection, current.root_id, existing_child=True)
            changed = _execute_mutation(
                connection,
                """UPDATE sync_bindings
                   SET state = 'disconnected', row_version = row_version + 1,
                       updated_at = ?
                   WHERE binding_id = ? AND row_version = ?
                     AND state <> 'disconnected'""",
                (
                    _timestamp_after(current.updated_at),
                    validated_binding_id,
                    validated_version,
                ),
            ).rowcount
            if changed != 1:
                raise SyncStateConflictError(
                    "Sync binding version or lifecycle state conflicts with disconnect."
                )
            return _require_binding(connection, validated_binding_id)

    def update_candidate_root(
        self,
        root_id: str,
        expected_version: int,
        *,
        display_name: str | None = None,
        direction: str | None = None,
    ) -> SyncRootRecord:
        """Compare-and-set mutable metadata on one candidate root."""
        validated_root_id = _validate_root_id(root_id)
        validated_version = _validate_expected_version(expected_version)
        if display_name is None and direction is None:
            raise ValueError("At least one candidate-root field must be updated.")
        validated_name = (
            None
            if display_name is None
            else _validate_text(
                display_name,
                field_name="display_name",
                maximum=_MAX_DISPLAY_NAME_LENGTH,
            )
        )
        validated_direction = (
            None if direction is None else _validate_direction(direction)
        )
        with _repository_transaction(self._database_path, immediate=True) as connection:
            current = _require_root(connection, validated_root_id)
            _require_advanceable_root(current)
            updated_at = _timestamp_after(current.updated_at)
            changed = _execute_mutation(
                connection,
                """UPDATE sync_roots
                   SET display_name = COALESCE(?, display_name),
                       direction = COALESCE(?, direction),
                       row_version = row_version + 1,
                       updated_at = ?
                   WHERE root_id = ? AND row_version = ? AND state = 'candidate'""",
                (
                    validated_name,
                    validated_direction,
                    updated_at,
                    validated_root_id,
                    validated_version,
                ),
            ).rowcount
            if changed != 1:
                raise SyncStateConflictError(
                    "Sync root version or lifecycle state conflicts with this update."
                )
            return _require_root(connection, validated_root_id)

    def pause_root(
        self,
        root_id: str,
        expected_version: int,
        reason_code: str,
    ) -> SyncRootRecord:
        """Compare-and-set one candidate root into its bounded paused state."""
        validated_root_id = _validate_root_id(root_id)
        validated_version = _validate_expected_version(expected_version)
        validated_reason = _validate_reason_code(reason_code)
        with _repository_transaction(self._database_path, immediate=True) as connection:
            current = _require_root(connection, validated_root_id)
            _require_advanceable_root(current)
            changed = _execute_mutation(
                connection,
                """UPDATE sync_roots
                   SET state = 'paused', reason_code = ?,
                       row_version = row_version + 1, updated_at = ?
                   WHERE root_id = ? AND row_version = ? AND state = 'candidate'""",
                (
                    validated_reason,
                    _timestamp_after(current.updated_at),
                    validated_root_id,
                    validated_version,
                ),
            ).rowcount
            if changed != 1:
                raise SyncStateConflictError(
                    "Sync root version or lifecycle state conflicts with pause."
                )
            return _require_root(connection, validated_root_id)

    def disconnect_root(
        self,
        root_id: str,
        expected_version: int,
    ) -> SyncRootRecord:
        """Atomically disconnect one root and every live child binding."""
        validated_root_id = _validate_root_id(root_id)
        validated_version = _validate_expected_version(expected_version)
        with _repository_transaction(self._database_path, immediate=True) as connection:
            current = _require_root(connection, validated_root_id)
            if current.state is SyncRootState.DISCONNECTED:
                raise SyncStateConflictError("A disconnected sync root is terminal.")
            _require_advanceable_root(current)
            invalid_child_version = connection.execute(
                """SELECT 1 FROM sync_bindings
                   WHERE root_id = ? AND state <> 'disconnected'
                     AND (typeof(row_version) <> 'integer' OR row_version <= 0)
                   LIMIT 1""",
                (validated_root_id,),
            ).fetchone()
            if invalid_child_version is not None:
                raise SyncStateCorruptionError(
                    "A private child version is incompatible with canonical v2."
                )
            unadvanceable_child = connection.execute(
                """SELECT 1 FROM sync_bindings
                   WHERE root_id = ? AND state <> 'disconnected'
                     AND row_version >= ? LIMIT 1""",
                (validated_root_id, _MAX_SQLITE_INTEGER),
            ).fetchone()
            if unadvanceable_child is not None:
                raise NotesSyncStateError("A private child version cannot be advanced.")
            child_updated_at = connection.execute(
                """SELECT COALESCE(MAX(updated_at), 0)
                   FROM sync_bindings
                   WHERE root_id = ? AND state <> 'disconnected'""",
                (validated_root_id,),
            ).fetchone()[0]
            if type(child_updated_at) is not int:
                raise SyncStateCorruptionError(
                    "A private child timestamp is incompatible with canonical v2."
                )
            updated_at = _timestamp_after(max(current.updated_at, child_updated_at))
            _execute_mutation(
                connection,
                """UPDATE sync_bindings
                   SET state = 'disconnected', row_version = row_version + 1,
                       updated_at = ?
                   WHERE root_id = ? AND state <> 'disconnected'""",
                (updated_at, validated_root_id),
            )
            changed = _execute_mutation(
                connection,
                """UPDATE sync_roots
                   SET state = 'disconnected', row_version = row_version + 1,
                       updated_at = ?
                   WHERE root_id = ? AND row_version = ?
                     AND state <> 'disconnected'""",
                (updated_at, validated_root_id, validated_version),
            ).rowcount
            if changed != 1:
                raise SyncStateConflictError(
                    "Sync root version or lifecycle state conflicts with disconnect."
                )
            return _require_root(connection, validated_root_id)


__all__ = (
    "MAX_SYNC_BINDINGS",
    "MAX_SYNC_ROOTS",
    "NotesSyncStateError",
    "NotesSyncStateRepository",
    "SyncBindingRecord",
    "SyncBindingState",
    "SyncRootRecord",
    "SyncRootState",
    "SyncStateCapacityError",
    "SyncStateConflictError",
    "SyncStateCorruptionError",
)
