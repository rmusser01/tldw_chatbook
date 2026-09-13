"""Explicit native outcomes for one ordinary migration/publication operation.

This record retains exclusion. It never authorizes IO, callbacks or capture.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tldw_chatbook.Utils import private_paths
from tldw_chatbook.Utils.platform_files import os


@dataclass(eq=False)
class _MigrationSourceOutcome:
    path: Path
    admission: Any = None
    connection: Any = None
    returned_connection: Any = None
    pending: bool = True
    close_attempted: bool = False
    closed: bool = False


class _MigrationNativeState:
    def __init__(
        self, sources: tuple[Path, ...], repository: Any = None, related=()
    ) -> None:
        self.sources = sources
        self.paths = frozenset((*sources, *related))
        self.parent_paths: dict[int, Path] = {}
        self.repository = repository
        self.thread = threading.current_thread()
        self.pid = os.getpid()
        self.leases: list[Any] = []
        self.pending: set[object] = set()
        self.readers: list[Any] = []
        self.source_connections: list[_MigrationSourceOutcome] = []
        self.descriptors: dict[int, Any] = {}
        self.failed_closes: set[int] = set()
        self.errors: list[BaseException] = []
        self.body_error: BaseException | None = None
        self.uncertain = False
        self.active = True

    def check(self) -> None:
        if (
            not self.active
            or self.pid != os.getpid()
            or self.thread is not threading.current_thread()
            or (
                self.repository is not None
                and self not in self.repository._migration_native_operations
            )
        ):
            raise ValueError("migration_native_operation_invalid")

    def check_paths(self, sources) -> None:
        self.check()
        if any(
            private_paths.lexical_path(source) not in self.paths for source in sources
        ):
            raise ValueError("migration_native_source_mismatch")

    def admit(self) -> None:
        from tldw_chatbook.Backup_Recovery import storage_admission as storage

        self.check()
        if self.uncertain or self.pending:
            raise ValueError("migration_native_outcome_unresolved")
        for source in self.sources:
            lease = storage.acquire_storage(source)
            lease.native_owner = self
            self.leases.append(lease)

    def open(self, primitive: Any, close: Any, *args: Any, **kwargs: Any) -> int:
        parent = self.parent_paths.get(kwargs.get("dir_fd"))
        if "dir_fd" in kwargs:
            if parent is None:
                raise ValueError("migration_native_parent_mismatch")
            self.check_paths((parent / args[0],))
        else:
            self.check_paths((args[0],))
        self.admit()
        attempt = object()
        self.pending.add(attempt)
        try:
            fd = primitive(*args, **kwargs)
        except OSError:
            if primitive is private_paths._ORIGINAL_NATIVE_OPEN:
                self.pending.discard(attempt)
            raise
        self.descriptors[fd] = close
        self.pending.discard(attempt)
        return fd

    def open_parent(self, module: Any, *args: Any, **kwargs: Any):
        """Hold one traversal admission while checking each component's authority."""
        from tldw_chatbook.Backup_Recovery import storage_admission as storage

        self.check_paths((args[0],))
        first_lease = len(self.leases)
        self.admit()
        leases = tuple(self.leases[first_lease:])

        def source_identity(source):
            try:
                info = os.stat(source, follow_symlinks=False)
            except FileNotFoundError:
                return None
            return (
                info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
                info.st_size, info.st_mtime_ns, info.st_ctime_ns,
            )

        admission = storage._Acquisition()
        identities = None

        def opening(*a: Any, **kw: Any) -> int:
            self.check()
            if self.uncertain or self.pending:
                raise ValueError("migration_native_outcome_unresolved")
            for source, lease, identity in zip(
                self.sources, leases, identities, strict=True
            ):
                with storage._lock:
                    admission.check(source)
                    lease.execution_context(source)
                if source_identity(source) != identity:
                    raise ValueError("migration_native_source_changed")
            attempt = object()
            self.pending.add(attempt)
            outcome = private_paths._NativeOpenOutcome()
            try:
                return module._native_open(*a, _outcome=outcome, **kw)
            finally:
                if outcome.descriptor is not None:
                    self.descriptors[outcome.descriptor] = module._native_close
                    self.pending.discard(attempt)
                elif outcome.rejected:
                    self.pending.discard(attempt)

        try:
            identities = tuple(source_identity(source) for source in self.sources)
            result = module._open_verified_parent(
                *args,
                _open=opening,
                _close=lambda fd: self.close(module._native_close, fd),
                **kwargs,
            )
        finally:
            admission.close()
        self.parent_paths[result[0]] = Path(args[0]).parent
        return result

    def close(self, primitive: Any, fd: int) -> None:
        self.check()
        if fd in self.failed_closes:
            raise ValueError("migration_native_close_unresolved")
        try:
            primitive(fd)
        except BaseException as error:
            self.failed_closes.add(fd)
            self.uncertain = True
            self.errors.append(error)
            raise
        self.descriptors.pop(fd, None)
        self.parent_paths.pop(fd, None)

    def new_reader(self):
        from tldw_chatbook.DB.private_sqlite import _SQLiteDescriptorOutcome

        self.admit()
        outcome = _SQLiteDescriptorOutcome()
        self.readers.append(outcome)
        return outcome

    def close_reader(self, connection, outcome) -> None:
        self.check()
        if outcome.connection_close_attempted:
            raise ValueError("migration_reader_close_unresolved")
        outcome.connection_close_attempted = True
        try:
            connection.close()
            # Match the ordinary privateSQLite native-retirement requirement.
            # A delegated successful close cannot leave its actual native open.
            if outcome.connection is not None:
                sqlite3.Connection.close(outcome.connection)
        except BaseException as error:
            self.uncertain = True
            outcome.errors.append(error)
            self.errors.append(error)
            raise
        outcome.connection_closed = True

    def retry_close_reader(self, connection, outcome) -> None:
        """Retry a retained SQLite object while its complete descriptor cohort is held."""
        self.check()
        if outcome not in self.readers or outcome.connection_closed:
            if outcome.connection_closed:
                return
            raise ValueError("migration_reader_not_retained")
        previous = tuple(outcome.errors)
        try:
            connection.close()
            if outcome.connection is not None:
                sqlite3.Connection.close(outcome.connection)
        except BaseException as error:
            outcome.errors.append(error)
            self.errors.append(error)
            self.uncertain = True
            raise
        outcome.connection_closed = True
        outcome.errors.clear()
        self.errors[:] = [
            error for error in self.errors
            if all(error is not old for old in previous)
        ]
        self.uncertain = bool(self.errors or self.failed_closes or self.pending)

    def close_source(self, connection):
        self.check()
        outcome = next(
            item
            for item in self.source_connections
            if item.connection is connection or item.returned_connection is connection
        )
        if outcome.close_attempted:
            raise ValueError("migration_source_close_unresolved")
        outcome.close_attempted = True
        try:
            connection.close()
        except BaseException as error:
            self.uncertain = True
            self.errors.append(error)
            raise
        outcome.closed = True

    def finish(self) -> None:
        for outcome in self.source_connections:
            if outcome.pending or (
                outcome.connection is not None and not outcome.closed
            ):
                self.uncertain = True
        for outcome in self.readers:
            if (
                outcome.connection is not None
                and not outcome.connection_close_attempted
            ):
                try:
                    self.close_reader(outcome.connection, outcome)
                except BaseException:  # noqa: BLE001, S110 - retained on this native owner
                    pass
            if (
                not outcome.entered
                or outcome.duplicate_pending
                or outcome.connector_pending
                or outcome.connection_pending
                or (outcome.duplicate is not None and not outcome.duplicate_closed)
                or (outcome.connection is not None and not outcome.connection_closed)
            ):
                self.uncertain = True
        if any(
            outcome.pending or (outcome.connection is not None and not outcome.closed)
            for outcome in self.source_connections
        ) or any(
            outcome.connection_pending or outcome.connector_pending
            or (outcome.connection is not None and not outcome.connection_closed)
            for outcome in self.readers
        ):
            # A live SQLite view keeps all original pins. Only a successful
            # object-close retry may allow the descriptor retirement below.
            return
        # Close independently retained parents skipped by an earlier finally
        # failure. Never replay an uncertain close or infer retirement from EBADF.
        for fd, close in tuple(self.descriptors.items()):
            if fd not in self.failed_closes:
                try:
                    self.close(close, fd)
                except BaseException:  # noqa: BLE001, S110 - retained on this native owner
                    pass
        if self.pending or self.descriptors or self.uncertain:
            return
        while self.leases:
            try:
                self.leases[-1].close()
            except BaseException as error:  # noqa: BLE001 - retained; caller preserves control flow
                self.uncertain = True
                self.errors.append(error)
                return
            self.leases.pop()
        self.active = False
        if self.repository is not None:
            self.repository._migration_native_operations.discard(self)


@contextmanager
def _migration_native(sources, native=None, *, repository=None, related=()):
    if native is not None:
        if type(native) is not _MigrationNativeState:
            raise TypeError("migration_native_operation_invalid")
        native.check_paths(sources)
        yield native
        return
    native = _MigrationNativeState(
        tuple(private_paths.lexical_path(p) for p in sources), repository, related
    )
    if repository is not None:
        repository._migration_native_operations.add(native)
    body_error = None
    try:
        # Independent outer holds survive uncertain later allocation-lease close.
        native.admit()
        native.admit()
        yield native
    except BaseException as error:
        body_error = error
        native.body_error = error
        raise
    finally:
        previous_errors = len(native.errors)
        native.finish()
        if body_error is None:
            for cleanup_error in native.errors[previous_errors:]:
                if not isinstance(cleanup_error, Exception):
                    raise cleanup_error
        if native.active and body_error is None:
            from .profile_errors import ProfileRepositoryError

            raise ProfileRepositoryError("migration_failed")


def _native_open(native, os_module, *args, **kwargs):
    if native is None:
        return os_module.open(*args, **kwargs)
    return native.open(os_module.open, os_module.close, *args, **kwargs)


def _native_close(native, os_module, fd):
    if native is None:
        return os_module.close(fd)
    return native.close(os_module.close, fd)


def _native_parent(native, module, *args, **kwargs):
    if native is None:
        return module._open_verified_parent(*args, **kwargs)
    return native.open_parent(module, *args, **kwargs)


def _migration_paths(active: Path) -> tuple[Path, ...]:
    """Exact installed active/boundary/rollback/journal/tombstone namespace."""
    from .profile_migration_journal import (
        PROFILE_MIGRATION_CANDIDATE_LEAVES,
        PROFILE_MIGRATION_ROLLBACK_LEAVES,
    )
    from .profile_migration_namespace import MigrationTombstoneKey

    leaves = (
        active.name,
        active.name + ".pre-v3.sqlite3",
        active.name + ".pre-v4.sqlite3",
        "." + active.name + ".migration-publication.json",
        *PROFILE_MIGRATION_CANDIDATE_LEAVES.values(),
        *PROFILE_MIGRATION_ROLLBACK_LEAVES.values(),
        *(
            ".profile-migration-" + key.value + ".tombstone"
            for key in MigrationTombstoneKey
        ),
    )
    return tuple(active.with_name(leaf) for leaf in leaves)


@contextmanager
def _native_reader(native):
    if native is None:
        yield None
        return
    outcome = native.new_reader()
    try:
        yield outcome
    finally:
        if (
            not outcome.entered
            or outcome.duplicate_pending
            or outcome.connector_pending
            or outcome.connection_pending
            or (outcome.duplicate is not None and not outcome.duplicate_closed)
        ):
            native.uncertain = True


def _close_reader(native, connection, outcome):
    if native is None:
        return connection.close()
    return native.close_reader(connection, outcome)


@contextmanager
def _source_allocation(native, path, *, _outcome=None):
    if native is None:
        yield None
        return
    from tldw_chatbook.DB.private_sqlite import _SQLiteAdmissionOutcome

    native.check_paths((path,))
    if _outcome is None:
        native.admit()
        outcome = _MigrationSourceOutcome(private_paths.lexical_path(path))
        native.source_connections.append(outcome)
    else:
        outcome = _outcome
        if (
            outcome not in native.source_connections
            or outcome.path != private_paths.lexical_path(path)
            or outcome.connection is not None
        ):
            raise ValueError("migration_source_outcome_mismatch")
        try:
            native.admit()
        except BaseException:
            outcome.pending = False
            raise
    outcome.admission = _SQLiteAdmissionOutcome()
    try:
        yield outcome
    finally:
        if outcome.admission.admission_refused:
            outcome.pending = False


def _source_options(outcome):
    return {} if outcome is None else {"_admission_outcome": outcome.admission}


def _source_observed(outcome, connection):
    if outcome is not None:
        outcome.connection = connection
        outcome.pending = False


def _source_close(native, connection):
    if native is None:
        return connection.close()
    return native.close_source(connection)


def _source_initialize(native, initialize, path, **kwargs):
    if native is None:
        return initialize(path, **kwargs)
    native.check_paths((path,))
    native.admit()
    outcome = _MigrationSourceOutcome(private_paths.lexical_path(path))
    native.source_connections.append(outcome)
    connection = initialize(path, _native=native, _source_outcome=outcome, **kwargs)
    outcome.returned_connection = connection
    return connection


def _source_cleanup(native, connection):
    """Observe first cleanup; preserve the pre-existing public retry separately."""
    if native is None:
        return connection.close()
    outcome = next(
        item
        for item in native.source_connections
        if item.connection is connection or item.returned_connection is connection
    )
    if outcome.close_attempted:
        # This existing compatibility retry cannot change the recorded outcome.
        return connection.close()
    return native.close_source(connection)
