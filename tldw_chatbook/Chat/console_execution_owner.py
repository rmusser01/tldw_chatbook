"""Fail-closed process ownership of one secured physical Console run store."""

from __future__ import annotations

import errno
import stat
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import ClassVar, Self, TextIO

from tldw_chatbook.Utils.platform_files import fcntl, os
from tldw_chatbook.Utils.private_paths import (
    PrivatePathStatus,
    lexical_path,
    open_private_binary,
    open_private_lock_stream,
    verify_trusted_directory,
)


class ConsoleExecutionBusy(RuntimeError):
    """Another process already owns execution for this run store."""


class ConsoleExecutionUnavailable(RuntimeError):
    """Safe execution ownership could not be established or has expired."""


def _identity(path: Path) -> tuple[int, int]:
    info = path.stat()
    return info.st_dev, info.st_ino


def _store_identity(path: Path) -> tuple[int, int] | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise ConsoleExecutionUnavailable("unsafe_run_store")
    return info.st_dev, info.st_ino


class ConsoleExecutionOwner:
    """An OS lease held by the actual executing process until physical cleanup.

    The lock leaf is stable and is never unlinked. A fork cannot use the parent's
    capability; an executing child must acquire its own lease after spawning.
    """

    _live: ClassVar[set[ConsoleExecutionOwner]] = set()

    def __init__(self, path: Path, stream: TextIO) -> None:
        self._lock = RLock()
        self._users = 0
        self._closing = False
        self._path = path
        self._stream = stream
        self._pid = os.getpid()
        self._parent_identity = _identity(path.parent)
        info = os.fstat(stream.fileno())
        self._lock_identity = info.st_dev, info.st_ino
        self._store_identity = _store_identity(path)

    @classmethod
    def acquire(cls, run_store: Path) -> ConsoleExecutionOwner:
        """Acquire the nonblocking lease, refusing unsafe paths and lock errors."""
        stream = None
        owner = None
        try:
            path = lexical_path(run_store)
            result = verify_trusted_directory(path.parent, allow_shared_sticky=False)
            if result.status != PrivatePathStatus.TRUSTED_DIRECTORY:
                raise ConsoleExecutionUnavailable("unverified_run_store_parent")
            stream = open_private_lock_stream(
                path.with_name(f".{path.name}.execution.lock")
            )
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    raise ConsoleExecutionBusy("console_execution_busy") from None
                raise
            try:
                path.lstat()
            except FileNotFoundError:
                pass
            else:
                # Preserve native legacy-file hardening, only after winning the lease.
                with open_private_binary(path):
                    pass
            owner = cls(path, stream)
            cls._live.add(owner)
            owner.require_current(path)
            return owner
        except BaseException as exc:
            if owner is not None:
                cls._live.discard(owner)
            if stream is not None:
                stream.close()
            if isinstance(exc, (ConsoleExecutionBusy, ConsoleExecutionUnavailable)):
                raise
            if not isinstance(exc, Exception):
                raise
            raise ConsoleExecutionUnavailable("console_execution_unavailable") from exc

    @classmethod
    def _acquire_consolidation(cls, scope) -> ConsoleExecutionOwner:
        """Acquire only the current native session's already-private run store."""
        from tldw_chatbook.Backup_Recovery.admission import Admission
        from tldw_chatbook.Backup_Recovery.native_files import pinned_directory
        from tldw_chatbook.Backup_Recovery.storage_admission import _MobileScope

        if cls is not ConsoleExecutionOwner or type(scope) is not _MobileScope:
            raise ConsoleExecutionUnavailable("consolidation_scope_required")
        _MobileScope.check(scope)
        if (
            scope.owner["owner_id"] != "db.agent_runs"
            or scope.execution_owner is not None
        ):
            raise ConsoleExecutionUnavailable("consolidation_run_scope_required")
        path = Path(scope.owner["path"])
        result = verify_trusted_directory(path.parent, allow_shared_sticky=False)
        if (
            result.status != PrivatePathStatus.TRUSTED_DIRECTORY
            or _store_identity(path) is None
        ):
            raise ConsoleExecutionUnavailable("unverified_run_store_parent")
        leaf = f".{path.name}.execution.lock"
        with pinned_directory(path.parent) as parent:
            try:
                Admission._create_lock(parent, leaf)
            except FileExistsError:
                pass
            fd = Admission._open(parent, leaf, os.O_RDWR)
            # The native scope owns the descriptor even if fdopen/locking fails.
            scope.execution_fd = fd
            stream = os.fdopen(fd, "r+")
            scope.execution_stream = stream
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    raise ConsoleExecutionBusy("console_execution_busy") from None
                raise
            owner = cls(path, stream)
            scope.execution_owner = owner
            cls._live.add(owner)
            owner.require_current(path)
            scope.execution_use = owner._use(path)
            scope.execution_use.__enter__()
            _MobileScope.check(scope)
            return owner

    def require_current(self, run_store: Path) -> None:
        """Require this process's retained lease and the admitted resource identity."""
        try:
            path = lexical_path(run_store)
            if (
                self not in self._live
                or self._pid != os.getpid()
                or self._stream.closed
            ):
                raise ConsoleExecutionUnavailable("execution_owner_expired")
            result = verify_trusted_directory(path.parent, allow_shared_sticky=False)
            if (
                result.status != PrivatePathStatus.TRUSTED_DIRECTORY
                or path.name != self._path.name
                or _identity(path.parent) != self._parent_identity
                or _identity(path.with_name(f".{path.name}.execution.lock"))
                != self._lock_identity
            ):
                raise ConsoleExecutionUnavailable("execution_resource_changed")
            current = _store_identity(path)
            if self._store_identity is not None and current != self._store_identity:
                raise ConsoleExecutionUnavailable("execution_store_changed")
            self._store_identity = current
        except ConsoleExecutionUnavailable:
            raise
        except Exception as exc:
            raise ConsoleExecutionUnavailable("execution_identity_unavailable") from exc

    @contextmanager
    def _use(self, run_store: Path):
        """Retain the lease through an admitted physical SQLite operation."""
        with self._lock:
            if self._closing:
                raise ConsoleExecutionUnavailable("execution_owner_retiring")
            self.require_current(run_store)
            self._users += 1
        try:
            yield
        finally:
            with self._lock:
                self._users -= 1
                if self._closing and not self._users:
                    self._stream.close()
                    self._live.discard(self)

    def close(self) -> None:
        """Release this process's lease only after its physical work has retired."""
        # Closing an inherited FD must not explicitly unlock the parent's lease.
        with self._lock:
            self._closing = True
            if not self._users:
                self._stream.close()
                self._live.discard(self)

    def __enter__(self) -> Self:
        self.require_current(self._path)
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
