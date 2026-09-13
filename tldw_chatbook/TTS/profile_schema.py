"""SQLite schema, validation, and persistence codecs for TTS profiles.

Connections remain caller-owned. The live opener configures and returns a
connection. Candidate validation owns its disposable snapshot connections and
retains uncertain native cleanup: a brief read-write reopen upgrades only the
copy, followed by an immutable read-only validation handle.
"""

from __future__ import annotations

import hashlib
import sqlite3
import stat
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast
from uuid import UUID

if TYPE_CHECKING:
    from tldw_chatbook.TTS.profile_repository import _BackupNativeState

from tldw_chatbook.DB.private_sqlite import (
    _SQLiteAdmissionOutcome,
    connect_private_sqlite_descriptor,

    _connect_registered_sqlite,
    connect_private_sqlite,
)
from tldw_chatbook.DB.sql_validation import escape_identifier, validate_identifier
from tldw_chatbook.Utils.platform_files import os
from tldw_chatbook.DB.private_sqlite_process import (
    HELPER_ADMISSION,
    HelperLease,
    HelperPreDispatchTimeoutError,
    HelperProtocolError,
    HelperTimeoutError,
    HelperUnavailableError,
    OperationDeadline,
)
from tldw_chatbook.DB.private_sqlite_protocol import (
    FileIdentity,
    PrepareRequest,
    TTSRestoreAuthority,
    is_tts_authority_refusal,
    validate_tts_identity,
)
from tldw_chatbook.TTS import profile_validation as _validation
from tldw_chatbook.TTS.migrations.v0_to_v1 import migrate as _migrate_v0_to_v1
from tldw_chatbook.TTS.migrations.v1_to_v2 import migrate as _migrate_v1_to_v2
from tldw_chatbook.TTS.migrations.v2_to_v3 import migrate as _migrate_v2_to_v3
from tldw_chatbook.TTS.migrations.v3_to_v4 import migrate as _migrate_v3_to_v4
from tldw_chatbook.TTS.profile_errors import (
    ProfileRepositoryError,
    _raise_migration_cleanup_failure,
)
from tldw_chatbook.TTS.profile_migration_journal import (
    MAX_PROFILE_MIGRATION_ARTIFACT_BYTES,
)
from tldw_chatbook.TTS.profile_sqlite_policy import configure_native_close_policy
from tldw_chatbook.TTS.profile_types import (
    CharacterTTSAssignment,
    JsonOptions,
    TTSGenerationProfile,
    canonical_json_options,
)
from tldw_chatbook.Utils import private_paths

# Compatibility exports retain one implementation in the isolated validator leaf.
_REFERENCE_TABLE = _validation._REFERENCE_TABLE
CURRENT_PROFILE_SCHEMA_VERSION = _validation.CURRENT_PROFILE_SCHEMA_VERSION
BUSY_TIMEOUT_MS = _validation.BUSY_TIMEOUT_MS
_DEADLINE_PROGRESS_OPCODE_INTERVAL = _validation._DEADLINE_PROGRESS_OPCODE_INTERVAL
_MAX_PERSISTED_DISPLAY_NAME_CHARACTERS = (
    _validation._MAX_PERSISTED_DISPLAY_NAME_CHARACTERS
)
_MAX_PERSISTED_RESPONSE_FORMAT_CHARACTERS = (
    _validation._MAX_PERSISTED_RESPONSE_FORMAT_CHARACTERS
)
_MAX_PERSISTED_OPTIONS_BYTES = _validation._MAX_PERSISTED_OPTIONS_BYTES
PROFILE_TABLE = _validation.PROFILE_TABLE
ASSIGNMENT_TABLE = _validation.ASSIGNMENT_TABLE
ASSIGNMENT_PROFILE_INDEX = _validation.ASSIGNMENT_PROFILE_INDEX
PROFILE_COLUMNS = _validation.PROFILE_COLUMNS
ASSIGNMENT_COLUMNS = _validation.ASSIGNMENT_COLUMNS
JOINED_ASSIGNMENT_ALIASES = _validation.JOINED_ASSIGNMENT_ALIASES
JOINED_PROFILE_ALIASES = _validation.JOINED_PROFILE_ALIASES
ASSIGNED_PROFILE_JOIN_SELECT = _validation.ASSIGNED_PROFILE_JOIN_SELECT
RowLike = _validation.RowLike
_MAX_EXACT_METADATA_ROWS = _validation._MAX_EXACT_METADATA_ROWS
_repository_error = _validation._repository_error
_update_metadata_digest = _validation._update_metadata_digest
_stream_exact_store_metadata_evidence = (
    _validation._stream_exact_store_metadata_evidence
)
encode_uuid = _validation.encode_uuid
decode_uuid = _validation.decode_uuid
encode_utc_datetime = _validation.encode_utc_datetime
decode_utc_datetime = _validation.decode_utc_datetime
decode_options = _validation.decode_options
_freeze_via_profile_options = _validation._freeze_via_profile_options
_row_value = _validation._row_value
_decode_profile = _validation._decode_profile
decode_profile = _validation.decode_profile
_decode_assignment = _validation._decode_assignment
decode_assignment = _validation.decode_assignment
decode_assigned_snapshot = _validation.decode_assigned_snapshot
_configure_connection = _validation._configure_connection
_user_tables = _validation._user_tables
_user_schema_objects = _validation._user_schema_objects
_normalized_ddl = _validation._normalized_ddl
_validated_quoted_identifier = _validation._validated_quoted_identifier
_validate_owned_schema_sql = _validation._validate_owned_schema_sql
_table_xinfo_manifest = _validation._table_xinfo_manifest
_has_exact_binary_index_keys = _validation._has_exact_binary_index_keys
_has_exact_primary_key_index = _validation._has_exact_primary_key_index
_run_with_deadline_progress = _validation._run_with_deadline_progress
_validate_schema = _validation._validate_schema
_validate_schema_body = _validation._validate_schema_body
validate_profile_store_rows = _validation.validate_profile_store_rows


@dataclass(frozen=True, slots=True, repr=False)
class PostInitProfileStoreAuthority:
    """Exact closed-store identity retained across exclusive/shared handoff."""

    parent_identity: os.stat_result
    file_identity: os.stat_result

    def __repr__(self) -> str:
        return "PostInitProfileStoreAuthority(<private>)"


class _ExactCurrentProfileConnection:
    """Own live SQLite, remote original-inode proof, and one local directory."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        selected: Path,
        parent_fd: int,
        helper: HelperLease,
        identity: dict[str, object],
    ) -> None:
        self._connection = connection
        self.selected = selected
        self._parent_fd = parent_fd
        self._helper = helper
        self._parent_identity = FileIdentity.from_payload(identity["parent"])
        self._file_identity = FileIdentity.from_payload(identity["main"])
        self._proof_lost = False
        self._sqlite_closed = False
        self._wal_acquired = False
        self._cohort_complete = identity["wal"] is not None
        self._backup_lease = None
        self._directory_fds = {parent_fd}
        self._directory_pending = set()
        self._directory_failed_closes = set()

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

    def _lose_proof(self) -> None:
        if not self._proof_lost:
            self._proof_lost = True
            self._helper.retain_terminal_owner()
            HELPER_ADMISSION.latch_tts_proof_loss()
            try:
                self._helper.close()
            except Exception:  # noqa: BLE001 - retain ownership and bound private cleanup failures
                raise ExactProfileStoreProofLostError(self) from None
        raise ExactProfileStoreProofLostError(self)

    def _request(
        self, operation: str, deadline: OperationDeadline
    ) -> dict[str, object]:
        if self._proof_lost:
            raise ExactProfileStoreProofLostError(self)
        try:
            response = self._helper.request(operation, deadline=deadline)
        except HelperPreDispatchTimeoutError:
            raise
        except (HelperUnavailableError, HelperProtocolError, HelperTimeoutError):
            self._lose_proof()
        if is_tts_authority_refusal(response):
            raise ExactProfileStoreAuthorityError()
        if response["status"] != "ok":
            self._lose_proof()
        return validate_tts_identity(response["identity"])

    def _verify_directory(self) -> None:
        if self._directory_pending or self._directory_failed_closes:
            raise ExactProfileStoreCleanupError(self)
        reopened = -1
        try:
            reopened, leaf = private_paths._open_verified_parent(
                self.selected, missing_leaf_allowed=False,
                _open=self._open_directory_component,
                _close=self._close_directory_component,
            )
            expected = self._parent_identity
            for descriptor in (reopened, self._parent_fd):
                observed = FileIdentity.from_stat(os.fstat(descriptor))
                if (
                    leaf != self.selected.name
                    or observed.nlink <= 0
                    or expected.nlink <= 0
                    or (
                        observed.dev,
                        observed.ino,
                        observed.mode,
                        observed.uid,
                        observed.gid,
                    )
                    != (
                        expected.dev,
                        expected.ino,
                        expected.mode,
                        expected.uid,
                        expected.gid,
                    )
                ):
                    raise ExactProfileStoreAuthorityError()
        except Exception:  # noqa: BLE001 - normalize private path/stat failures without their contents
            raise ExactProfileStoreAuthorityError() from None
        finally:
            if reopened >= 0:
                self._close_directory(reopened, os.close)

    def _open_directory_component(self, *args, **kwargs):
        outcome = private_paths._NativeOpenOutcome()
        attempt = object()
        self._directory_pending.add(attempt)
        try:
            return private_paths._native_open(*args, _outcome=outcome, **kwargs)
        finally:
            if outcome.descriptor is not None:
                self._directory_fds.add(outcome.descriptor)
                self._directory_pending.discard(attempt)
            elif outcome.rejected:
                self._directory_pending.discard(attempt)

    def _close_directory_component(self, descriptor):
        self._close_directory(descriptor, private_paths._native_close)

    def _close_directory(self, descriptor, close):
        if descriptor in self._directory_failed_closes:
            raise ExactProfileStoreCleanupError(self)
        try:
            close(descriptor)
        except BaseException:
            # An error can follow successful native close. Keep exclusion and
            # never retry this descriptor number, which could already be reused.
            self._directory_failed_closes.add(descriptor)
            raise
        self._directory_fds.discard(descriptor)

    def _revalidate(self, deadline: OperationDeadline) -> None:
        operation = (
            "tts_pin_sidecars"
            if self._wal_acquired and not self._cohort_complete
            else "tts_recheck"
        )
        identity = self._request(operation, deadline)
        self._file_identity = FileIdentity.from_payload(identity["main"])
        self._cohort_complete = identity["wal"] is not None
        self._verify_directory()

    def export_restore_authority(
        self, *, deadline: OperationDeadline
    ) -> TTSRestoreAuthority:
        identity = self._request("tts_export_restore_authority", deadline)
        self._verify_directory()
        remote = TTSRestoreAuthority.from_payload(identity)
        # The helper binds stable directory authority before SQLite can create
        # sidecars. Namespace mutation additionally needs the directory's exact
        # current link count (including those owned creations on macOS).
        return TTSRestoreAuthority(
            parent=FileIdentity.from_stat(os.fstat(self._parent_fd)),
            main=remote.main,
            wal=remote.wal,
            shm=remote.shm,
        )

    def verified_parent_fd(self, *, deadline: OperationDeadline) -> int:
        self._revalidate(deadline)
        return self._parent_fd

    def close(self) -> None:
        if self._directory_pending or self._directory_failed_closes:
            raise ExactProfileStoreCleanupError(self)
        # Keep the native flag enabled. Normal cleanup owns PASSIVE; restore
        # owns its stronger TRUNCATE and must not acquire a duplicate checkpoint.
        if not self._sqlite_closed:
            self._revalidate(OperationDeadline(None))
            self._connection.close()
            self._sqlite_closed = True
        self._helper.close()
        if self._parent_fd >= 0:
            self._close_directory(self._parent_fd, os.close)
            self._parent_fd = -1
        for descriptor in tuple(self._directory_fds):
            self._close_directory(descriptor, os.close)
        if self._backup_lease is not None:
            self._backup_lease.close()
            self._backup_lease = None


class _NativeExactCurrentProfileConnection(_ExactCurrentProfileConnection):
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
        self._delete_mode_partial_cleanup = False
        self._proof_lost = False
        self._sqlite_closed = False

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
        return self._observe_revalidation_parent_descriptor(*args, **kwargs)

    def _observe_revalidation_parent_descriptor(self, *args: Any, **kwargs: Any) -> int:
        """Observe the held owner's original bounded recheck without new admission."""
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
                            cast(sqlite3.Connection, self),
                            self.selected,
                            _delete_mode_partial_cleanup=self._delete_mode_partial_cleanup,
                        )
                    self.attempted_connections.add(role)
                    native.close()
                    if role == "_connection":
                        self._sqlite_closed = True
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

    def export_restore_authority(self, *, deadline: OperationDeadline) -> TTSRestoreAuthority:
        deadline.remaining(30.0)
        revalidate_exact_current_profile_store(self, self.selected)
        return TTSRestoreAuthority(
            parent=FileIdentity.from_stat(os.fstat(self.parent_fd)),
            main=FileIdentity.from_stat(os.fstat(self.file_fd)),
            wal=FileIdentity.from_stat(os.fstat(self.sidecar_fds["-wal"])),
            shm=FileIdentity.from_stat(os.fstat(self.sidecar_fds["-shm"])),
        )

    def verified_parent_fd(self, *, deadline: OperationDeadline) -> int:
        deadline.remaining(30.0)
        revalidate_exact_current_profile_store(self, self.selected)
        return self.parent_fd


class ExactProfileStoreCleanupError(ProfileRepositoryError):
    """Carry exact retained authority when an internal close cannot settle."""

    def __init__(self, connection: _ExactCurrentProfileConnection) -> None:
        super().__init__("operation_failed")
        self.connection = connection


def _exact_profile_store_cleanup_error(
    error: BaseException,
) -> ExactProfileStoreCleanupError | None:
    """Read only this live-open attempt's owner, bypassing signal hooks."""
    if isinstance(error, ExactProfileStoreCleanupError):
        return error
    metadata = BaseException.__dict__["__dict__"].__get__(error, BaseException)
    cleanup = metadata.get("_profile_exact_cleanup_error")
    return cleanup if isinstance(cleanup, ExactProfileStoreCleanupError) else None


def _carry_exact_profile_cleanup(
    error: BaseException, cleanup: ExactProfileStoreCleanupError | None
) -> None:
    """Keep prior live owners reachable without transferring them again."""
    metadata = BaseException.__dict__["__dict__"].__get__(error, BaseException)
    previous = _exact_profile_store_cleanup_error(error)
    if previous is not None and previous is not cleanup:
        history = metadata.get("_profile_exact_cleanup_history", ())
        if all(previous is not retained for retained in history):
            history = (*history, previous)
        metadata["_profile_exact_cleanup_history"] = history
    else:
        metadata.setdefault("_profile_exact_cleanup_history", ())
    # None is intentional: a reused signal may now leave before acquiring a
    # live handle, or after healthy close. Earlier owners are history only.
    metadata["_profile_exact_cleanup_error"] = cleanup


class ExactProfileStoreNotCurrentError(ProfileRepositoryError):
    """Signal that shared proof must yield to exclusive initialization."""

    def __init__(self) -> None:
        super().__init__("schema_partial")


class ExactProfileStoreAuthorityError(ProfileRepositoryError):
    """Signal that retained live-store namespace authority no longer matches."""

    def __init__(self) -> None:
        super().__init__("operation_failed")


class ExactProfileStoreProofLostError(ExactProfileStoreAuthorityError):
    """Retain the complete live owner after irreplaceable remote proof loss."""

    def __init__(self, connection: _ExactCurrentProfileConnection) -> None:
        ProfileRepositoryError.__init__(self, "restart_required")
        self.connection = connection


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


def _revalidate_native_exact_current_profile_store(
    connection: sqlite3.Connection,
    path: Path | None,
    *,
    _delete_mode_partial_cleanup: bool = False,
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
            _open=connection._observe_revalidation_parent_descriptor,
            _close=connection.close_parent_descriptor,
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
            connection.close_descriptor(reopened_parent_fd)
    sidecars_match = set(connection.sidecar_fds) == {"-wal", "-shm"}
    if _delete_mode_partial_cleanup and connection._delete_mode_partial_cleanup:
        sidecars_match = not connection.sidecar_fds
        for suffix in ("-wal", "-shm"):
            try:
                os.stat(
                    f"{path.name}{suffix}",
                    dir_fd=connection.parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                continue
            except OSError:
                sidecars_match = False
            else:
                sidecars_match = False
    elif sidecars_match:
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


def revalidate_exact_current_profile_store(
    connection: sqlite3.Connection,
    path: Path | None,
    *,
    deadline: OperationDeadline | None = None,
    _delete_mode_partial_cleanup: bool = False,
) -> None:
    """Recheck the original remote proof and local directory before live use."""
    if isinstance(connection, _NativeExactCurrentProfileConnection):
        return _revalidate_native_exact_current_profile_store(
            connection, path, _delete_mode_partial_cleanup=_delete_mode_partial_cleanup
        )
    if not isinstance(connection, _ExactCurrentProfileConnection):
        return
    if path is None or connection.selected != path:
        raise ExactProfileStoreAuthorityError()
    connection._revalidate(deadline or OperationDeadline(None))


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
    *,
    _native=None,
) -> PostInitProfileStoreAuthority:
    """Pin one closed, sidecar-free store before releasing exclusive ownership."""

    from .profile_migration_native import (
        _migration_native,
        _native_parent,
        _native_open,
        _native_close,
    )

    with _migration_native((path,), native=_native) as _native:
        parent_fd = -1
        file_fd = -1
        try:
            parent_fd, leaf = _native_parent(
                _native,
                private_paths,
                path,
                missing_leaf_allowed=False,
            )
            parent = os.fstat(parent_fd)
            if not _exact_store_namespace_safe(parent_fd, leaf):
                raise _repository_error("operation_failed")
            file_fd = _native_open(
                _native,
                os,
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
                _native_close(_native, os, file_fd)
            if parent_fd >= 0:
                _native_close(_native, os, parent_fd)


def encode_options(options: JsonOptions) -> str:
    """Encode validated JSON options using the domain canonicalizer."""

    try:
        return canonical_json_options(options)
    except Exception:
        raise _repository_error("corrupt_data") from None


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


MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {
    0: _migrate_v0_to_v1,
    1: _migrate_v1_to_v2,
    2: _migrate_v2_to_v3,
    3: _migrate_v3_to_v4,
}


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


def _migration_reference_evidence(
    connection: sqlite3.Connection,
) -> tuple[tuple[object, ...], ...]:
    """Project every reference field except the WAV payload, in row order.

    The projection is deliberately payload-free (TASK-21130): selecting
    ``wav_bytes`` here materialised the whole reference table in Python, and
    the migration held two such projections at once -- measured at 966 MiB of
    peak allocation for a store at the 512 MiB
    :data:`~tldw_chatbook.TTS.profile_reference_types.MAX_REFERENCE_TOTAL_BYTES`
    bound, against this subsystem's own 256 KiB streaming norm.

    Byte-for-byte payload identity is still proved, by transitivity rather
    than by retention: the stored ``sha256`` column travels in this projection
    verbatim, and :func:`_validate_migration_reference_rows` re-derives
    ``sha256(wav_bytes)`` from the streamed BLOB and requires it to equal that
    column on *both* sides of the migration (see
    ``TTSCloneReference.__post_init__``). So ``blob_before == sha_before``,
    ``sha_before == sha_after`` (this projection), ``sha_after == blob_after``
    together give ``blob_before == blob_after``, and any caller comparing two
    of these projections must run that validation at both boundaries.

    ``reference_text`` is replaced by its UTF-8 length and digest so the
    evidence retains no private transcript either; the same shape is used for
    the downgrade-boundary evidence in ``profile_migration_candidate``.
    """

    return tuple(
        (
            row[0],
            row[1],
            len(row[2].encode("utf-8")),
            hashlib.sha256(row[2].encode("utf-8")).hexdigest(),
            *tuple(row[3:]),
        )
        for row in connection.execute(
            f"""
            SELECT profile_id, reference_id, reference_text, sha256,
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
        reference_evidence: tuple[tuple[object, ...], ...] = ()
        if from_version >= 3:
            # First link of the payload-identity chain: this proves
            # sha256(wav_bytes) == the sha256 column for every row BEFORE the
            # migration. The evidence captured on the next line then carries
            # that column (never the payload) across the climb.
            _validate_migration_reference_rows(connection, schema_version=from_version)
            reference_evidence = _migration_reference_evidence(connection)
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
        if _migration_reference_evidence(connection) != reference_evidence:
            raise RuntimeError
        validate_profile_store_rows(connection)
        # Closing link of the payload-identity chain: re-derives
        # sha256(wav_bytes) from the streamed BLOB and requires it to equal
        # the sha256 column the evidence above just proved unchanged. Neither
        # side ever holds more than one payload at a time.
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


def _open_native_exact_current_profile_store(
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
    owned = _NativeExactCurrentProfileConnection(
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
        journal_mode = live.execute("PRAGMA journal_mode").fetchone()[0]
        if journal_mode != "wal":
            owned._delete_mode_partial_cleanup = (
                journal_mode == "delete" and not sidecar_fds
            )
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


def _open_helper_exact_current_profile_store(
    path: Path,
    *,
    expected_post_init_authority: PostInitProfileStoreAuthority | None = None,
    deadline: OperationDeadline | None = None,
) -> sqlite3.Connection:
    """Admit current live SQLite with remote proof and one reserved envelope."""
    cleanup_error: ExactProfileStoreCleanupError | None = None
    try:
        if not isinstance(path, Path) or not path.is_absolute():
            raise _repository_error("operation_failed")
        try:
            path.lstat()
        except FileNotFoundError:
            raise ExactProfileStoreNotCurrentError() from None
        deadline = deadline or OperationDeadline(time.monotonic() + 30.0)
        parent_fd = -1
        owned: _ExactCurrentProfileConnection | None = None
        body_error: BaseException | None = None
        with HELPER_ADMISSION.reserve(
            transient=1, retained=1, deadline=deadline
        ) as reservation:
            helper = HelperLease.start(
                PrepareRequest(str(path), False, False, False),
                operation="tts_exact_current",
                reservation=reservation,
                deadline=deadline,
            )
            try:
                response = helper.initial_response
                if response["status"] != "ok":
                    if response.get("reason") == "exact_not_current":
                        raise ExactProfileStoreNotCurrentError()
                    raise _repository_error(response.get("reason", "operation_failed"))
                identity = validate_tts_identity(response["identity"])
                parent_fd, _leaf = private_paths._open_verified_parent(
                    path, missing_leaf_allowed=False
                )
                expected = expected_post_init_authority
                main = FileIdentity.from_payload(identity["main"])
                parent = FileIdentity.from_payload(identity["parent"])
                if expected is not None:
                    expected_main = FileIdentity.from_stat(expected.file_identity)
                    expected_parent = FileIdentity.from_stat(expected.parent_identity)
                    if (
                        not main.same_inode(expected_main)
                        or main.size != expected_main.size
                        or (parent.dev, parent.ino, parent.mode, parent.uid, parent.gid)
                        != (
                            expected_parent.dev,
                            expected_parent.ino,
                            expected_parent.mode,
                            expected_parent.uid,
                            expected_parent.gid,
                        )
                    ):
                        raise ExactProfileStoreAuthorityError()
                live = _connect_registered_sqlite(
                    "tts.profile_store",
                    path,
                    must_exist=True,
                    expected_identity=main,
                    isolation_level=None,
                    operation_deadline=deadline.expires_at,
                    reservation=reservation,
                )
                # Ownership exists before even policy configuration can fail.
                owned = _ExactCurrentProfileConnection(
                    live,
                    selected=path,
                    parent_fd=parent_fd,
                    helper=helper,
                    identity=identity,
                )
                parent_fd = -1
                configure_native_close_policy(live)
                owned._revalidate(deadline)
                live.execute("PRAGMA query_only = ON")
                if live.execute("PRAGMA query_only").fetchone()[0] != 1:
                    raise _repository_error("schema_corrupt")
                live.execute("PRAGMA user_version").fetchone()
                if live.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
                    raise ExactProfileStoreNotCurrentError()
                owned._wal_acquired = True
                owned._revalidate(deadline)
                _configure_connection(cast(sqlite3.Connection, owned))
                if (
                    owned.execute("PRAGMA user_version").fetchone()[0]
                    != CURRENT_PROFILE_SCHEMA_VERSION
                ):
                    raise _repository_error("schema_partial")
                check = lambda: deadline.remaining(30.0)
                _validate_schema(cast(sqlite3.Connection, owned), check_deadline=check)
                validate_profile_store_rows(
                    cast(sqlite3.Connection, owned), check_deadline=check
                )
                owned.execute("BEGIN")
                _run_with_deadline_progress(
                    cast(sqlite3.Connection, owned),
                    check,
                    lambda: _stream_exact_store_metadata_evidence(
                        cast(sqlite3.Connection, owned)
                    ),
                )
                owned._revalidate(deadline)
                if (
                    expected is not None
                    and owned._file_identity.size != expected.file_identity.st_size
                ):
                    raise ExactProfileStoreAuthorityError()
                owned.rollback()
                owned._revalidate(deadline)
                owned.execute("PRAGMA query_only = OFF")
                if owned.execute("PRAGMA query_only").fetchone()[0] != 0:
                    raise _repository_error("schema_corrupt")
                owned._revalidate(deadline)
            except BaseException as error:  # noqa: BLE001 - settle complete ownership before redelivering control flow
                body_error = error
            if body_error is not None and owned is not None:
                try:
                    owned.close()
                except BaseException as error:  # noqa: BLE001 - preserve the earliest control signal and live owner
                    cleanup_error = ExactProfileStoreCleanupError(owned)
                    if isinstance(body_error, Exception):
                        body_error = (
                            error
                            if not isinstance(error, Exception)
                            or isinstance(error, ExactProfileStoreProofLostError)
                            else cleanup_error
                        )
            if owned is not None and not owned._sqlite_closed and not owned._proof_lost:
                # Settle the transient child before transferring this complete owner.
                # Do not raise inside the reservation: its exceptional exit would
                # otherwise reap healthy retained proof needed for close retry.
                reservation.handoff_retained(helper)
            if parent_fd >= 0:
                os.close(parent_fd)
        if body_error is not None:
            if not isinstance(body_error, Exception) or isinstance(
                body_error, ProfileRepositoryError
            ):
                raise body_error
            raise _repository_error("schema_corrupt") from None
        assert owned is not None
        return cast(sqlite3.Connection, owned)
    except BaseException as error:
        if not isinstance(error, Exception):
            _carry_exact_profile_cleanup(error, cleanup_error)
        raise


def open_exact_current_profile_store(
    path: Path,
    *,
    expected_post_init_authority: PostInitProfileStoreAuthority | None = None,
    deadline: OperationDeadline | None = None,
) -> sqlite3.Connection:
    """Admit the complete proof owner before opening helper or native resources."""
    if private_paths._WINDOWS_PLATFORM:
        return _open_native_exact_current_profile_store(
            path, expected_post_init_authority=expected_post_init_authority
        )
    from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage

    lease = acquire_storage(path)
    try:
        connection = _open_helper_exact_current_profile_store(
            path, expected_post_init_authority=expected_post_init_authority,
            deadline=deadline,
        )
    except BaseException as error:
        cleanup = _exact_profile_store_cleanup_error(error)
        retained = (
            error.connection if isinstance(error, ExactProfileStoreProofLostError)
            else cleanup.connection if cleanup is not None else None
        )
        if retained is not None:
            retained._backup_lease = lease
        else:
            lease.close()
        raise
    connection._backup_lease = lease
    return connection


def open_profile_store(
    path: Path,
    *,
    must_exist: bool = False,
    check_deadline: Callable[[], None] | None = None,
    _native=None,
    _source_outcome=None,
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

    from .profile_migration_native import (
        _source_allocation,
        _source_options,
        _source_observed,
        _source_close,
    )

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
            with _source_allocation(
                _native, path, _outcome=_source_outcome
            ) as _source_call_outcome:
                connection = connect_private_sqlite(
                    "tts.profile_store",
                    path,
                    must_exist=must_exist,
                    isolation_level=None,
                    **_source_options(_source_call_outcome),
                )
                _source_observed(_source_call_outcome, connection)
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
        cleanup.attempt(lambda: _source_close(_native, connection))
    cleanup.raise_control_flow()
    if isinstance(body_error, ProfileRepositoryError):
        raise body_error
    raise _repository_error("schema_corrupt") from None


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
        self.sqlite_close_failures: dict[str, BaseException] = {}
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

    def close(self) -> None:
        """Retry only retained SQLite closes; uncertain fd numbers are never reused."""
        retry_errors = tuple(self.sqlite_close_failures.values())
        if self.allocation_pending or any(
            all(error is not retry for retry in retry_errors)
            for error in self.cleanup_errors
        ):
            raise _repository_error("schema_corrupt")
        for role in self.sqlite_close_failures:
            self.attempted.discard(role)
        self.cleanup_errors.clear()
        self.sqlite_close_failures.clear()
        self.uncertain = False
        cleanup = self.cleanup(None)
        cleanup.raise_control_flow()
        if self.connections:
            _raise_migration_cleanup_failure(self, *self.cleanup_errors)
        if cleanup.ordinary_cleanup_failed or self.uncertain:
            raise _repository_error("schema_corrupt")

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
                    try:
                        self.connections[role].close()
                    except BaseException as error:
                        self.sqlite_close_failures[role] = error
                        raise
                    del self.connections[role]

                attempt(close_connection)
                if role in self.connections:
                    return cleanup
        if self.connections:
            # Every file/directory pin remains owned while any SQLite view can
            # still hold a native lock or escaped alias.
            return cleanup
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
    if job.connections:
        _raise_migration_cleanup_failure(
            job, body_error, *job.cleanup_errors,
            code=body_error.code if isinstance(body_error, ProfileRepositoryError) else "schema_corrupt",
        )
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
