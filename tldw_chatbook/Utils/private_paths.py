"""Private local-file lifecycle primitives.

This module is deliberately dependency-leaf: callers choose failure policy and
diagnostics while this module performs lexical selection and filesystem checks.
"""

from __future__ import annotations

import contextlib
import errno
import functools
import os
import secrets
import stat
import sys
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO, Callable, Iterator, TextIO, TypeAlias

PathInput: TypeAlias = str | os.PathLike[str]

_PRIVATE_FILE_MODE = 0o600
_PRIVATE_DIRECTORY_MODE = 0o700
_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_NOCTTY = getattr(os, "O_NOCTTY", 0)
_PRIVATE_FILE_OPEN_FLAGS = os.O_RDONLY | _NOFOLLOW | _NONBLOCK | _NOCTTY
_WINDOWS_PLATFORM = os.name == "nt"
# Matches the usual kernel ELOOP budget, so a symlink cycle terminates instead
# of walking forever.
_MAX_TRUSTED_SYMLINK_HOPS = 8


class PrivatePathStatus(StrEnum):
    CREATED_PRIVATE = "created_private"
    HARDENED_PRIVATE = "hardened_private"
    ALREADY_PRIVATE = "already_private"
    UNSAFE_PARENT = "unsafe_parent"
    WRONG_OWNER = "wrong_owner"
    LINK_OR_NON_REGULAR = "link_or_non_regular"
    OPERATION_FAILED = "operation_failed"
    TRUSTED_DIRECTORY = "trusted_directory"
    UNVERIFIED_PLATFORM = "unverified_platform"


@dataclass(frozen=True)
class PrivatePathResult:
    lexical_path: Path
    status: PrivatePathStatus
    reason: str | None = None

    @property
    def verified_private(self) -> bool:
        return self.status in {
            PrivatePathStatus.CREATED_PRIVATE,
            PrivatePathStatus.HARDENED_PRIVATE,
            PrivatePathStatus.ALREADY_PRIVATE,
        }

    @property
    def usable(self) -> bool:
        return self.verified_private or self.status in {
            PrivatePathStatus.TRUSTED_DIRECTORY,
            PrivatePathStatus.UNVERIFIED_PLATFORM,
        }


class PrivatePathError(OSError):
    def __init__(self, result: PrivatePathResult) -> None:
        self.result = result
        reason = f": {result.reason}" if result.reason else ""
        super().__init__(f"{result.status.value}{reason}")


@dataclass
class PrivateBinaryFile:
    """A pinned binary stream paired with its verified privacy posture."""

    stream: BinaryIO
    result: PrivatePathResult


def lexical_path(path: PathInput) -> Path:
    raw = os.fspath(path)
    if "\x00" in raw:
        raise ValueError("Path must not contain NUL")
    expanded = os.path.expanduser(raw)
    return Path(os.path.abspath(os.path.normpath(expanded)))


def _posix_guards_available() -> bool:
    # `os.readlink` joined this set when trusted-symlink traversal was added:
    # the walk now depends on it being descriptor-relative, and a platform
    # without that support must fall through to the unverified-platform
    # branch rather than raising TypeError out of the walk.
    required_dir_fd = {os.open, os.stat, os.mkdir, os.readlink}
    return (
        os.name == "posix"
        and _NOFOLLOW != 0
        and _NONBLOCK != 0
        and _NOCTTY != 0
        and getattr(os, "O_DIRECTORY", 0) != 0
        and required_dir_fd.issubset(os.supports_dir_fd)
        and os.stat in os.supports_follow_symlinks
        and hasattr(os, "geteuid")
        and hasattr(os, "fstat")
        and hasattr(os, "fchmod")
        and hasattr(os, "fsync")
    )


def _atomic_posix_guards_available() -> bool:
    return _posix_guards_available() and {
        os.rename,
        os.unlink,
    }.issubset(os.supports_dir_fd)


def _classify_private_file_stat(
    file_stat: os.stat_result,
    *,
    expected_uid: int,
) -> PrivatePathStatus | None:
    if not stat.S_ISREG(file_stat.st_mode):
        return PrivatePathStatus.LINK_OR_NON_REGULAR
    if file_stat.st_nlink != 1:
        return PrivatePathStatus.LINK_OR_NON_REGULAR
    if file_stat.st_uid != expected_uid:
        return PrivatePathStatus.WRONG_OWNER
    return None


def _trusted_directory_owner(directory_stat: os.stat_result, euid: int) -> bool:
    return directory_stat.st_uid in {0, euid}


def _open_directory_component(parent_fd: int, component: str) -> int:
    return _native_open(
        component,
        _DIRECTORY_OPEN_FLAGS | _NOFOLLOW,
        dir_fd=parent_fd,
    )


def _trusted_symlink(link_stat: os.stat_result) -> bool:
    """Report whether a symlink component may be traversed.

    Only *root-owned* symlinks qualify, and only when no other user could have
    written them. This is deliberately narrower than `_trusted_directory_owner`,
    which also accepts the effective uid: a directory owned by the caller is the
    caller's own storage, whereas following a caller-owned symlink would silently
    relocate the application's private files behind a lexical alias. Platform
    symlinks such as macOS `/var -> private/var` are root-owned, so this is the
    smallest rule that lets the walk reach the system temporary directory.

    Note that Linux reports every symlink as mode 0o777, so the write-bit gate
    rejects all symlinks there. That is intentional: Linux has no root-owned
    symlink on the paths this module walks, so no traversal is needed.
    """

    if link_stat.st_uid != 0:
        return False
    return not bool(stat.S_IMODE(link_stat.st_mode) & 0o022)


def _read_trusted_symlink(
    parent_fd: int,
    component: str,
    selected: Path,
    exc: OSError,
) -> str:
    """Return a trusted symlink component's target, or re-raise the open error.

    Both the `lstat` and the `readlink` go through `dir_fd`, so the component is
    never re-derived from a path string that an attacker could repoint between
    the two calls.
    """

    if exc.errno not in {errno.ELOOP, errno.ENOTDIR}:
        raise _private_path_error_from_oserror(selected, exc) from None
    try:
        link_stat = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        raise _private_path_error_from_oserror(selected, exc) from None
    if not stat.S_ISLNK(link_stat.st_mode) or not _trusted_symlink(link_stat):
        raise _private_path_error_from_oserror(selected, exc) from None
    # `TypeError` as well as `OSError`: `os.readlink` raises TypeError, not
    # OSError, on a build where it does not accept `dir_fd`. The probe above
    # should stop us reaching here on such a platform, but a guard that
    # crashes rather than refusing is the wrong failure mode -- fail closed.
    try:
        return os.readlink(component, dir_fd=parent_fd)
    except (OSError, TypeError):
        raise _private_path_error_from_oserror(selected, exc) from None


def _symlink_walk_components(target: str) -> tuple[bool, list[str]]:
    absolute = target.startswith(os.sep)
    components = [part for part in target.split(os.sep) if part not in {"", os.curdir}]
    # A target that names its own directory ("." or "/") must still produce one
    # walk step so the per-component trust checks run on the resulting directory.
    return absolute, components or [os.curdir]


def _follow_trusted_symlink(
    *,
    current_fd: int,
    component: str,
    pending: list[str],
    hops: int,
    selected: Path,
    exc: OSError,
    _close: Callable[[int], None] | None = None,
) -> tuple[int, int]:
    """Splice a trusted symlink's target into the pending walk.

    Returns the descriptor the walk should continue from and the new hop count.
    `current_fd` is only replaced once its successor is open, so the caller
    always has exactly one descriptor to close on failure.
    """

    # Classify first, so an unrelated open failure keeps reporting its own cause
    # even once symlinks have been followed.
    target = _read_trusted_symlink(current_fd, component, selected, exc)
    if hops >= _MAX_TRUSTED_SYMLINK_HOPS:
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.LINK_OR_NON_REGULAR,
                reason="symlink_hop_limit_exceeded",
            )
        )
    absolute, components = _symlink_walk_components(target)
    pending[:0] = components
    if not absolute:
        return current_fd, hops + 1
    root_fd = _native_open(os.sep, _DIRECTORY_OPEN_FLAGS | _NOFOLLOW)
    (_close or _native_close)(current_fd)
    return root_fd, hops + 1


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _private_file_postcondition_holds(
    file_fd: int,
    parent_fd: int,
    leaf: str,
    *,
    expected_identity: os.stat_result,
) -> bool:
    opened = os.fstat(file_fd)
    entry = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    return (
        _same_identity(opened, expected_identity)
        and _same_identity(entry, expected_identity)
        and stat.S_ISREG(opened.st_mode)
        and opened.st_nlink == 1
        and entry.st_nlink == 1
        and opened.st_uid == os.geteuid()
        and stat.S_IMODE(opened.st_mode) == _PRIVATE_FILE_MODE
    )


def _private_directory_postcondition_holds(
    directory_fd: int,
    parent_fd: int,
    component: str,
    *,
    expected_identity: os.stat_result,
) -> bool:
    opened = os.fstat(directory_fd)
    entry = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
    return (
        _same_identity(opened, expected_identity)
        and _same_identity(entry, expected_identity)
        and stat.S_ISDIR(opened.st_mode)
        and opened.st_uid == os.geteuid()
        and stat.S_IMODE(opened.st_mode) == _PRIVATE_DIRECTORY_MODE
    )


def _trusted_directory_postcondition_holds(
    directory_fd: int,
    parent_fd: int,
    component: str,
    *,
    expected_identity: os.stat_result,
) -> bool:
    opened = os.fstat(directory_fd)
    entry = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
    return (
        _same_identity(opened, expected_identity)
        and _same_identity(entry, expected_identity)
        and stat.S_ISDIR(opened.st_mode)
        and _trusted_directory_owner(opened, os.geteuid())
    )


def _open_verified_parent(
    selected: Path,
    *,
    missing_leaf_allowed: bool,
    _close: Callable[[int], None] | None = None,
) -> tuple[int, str]:
    close = _close or _native_close
    parts = selected.parts
    if len(parts) < 2 or parts[0] != os.sep:
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                reason="invalid_absolute_path",
            )
        )

    euid = os.geteuid()
    current_fd = _native_open(os.sep, _DIRECTORY_OPEN_FLAGS | _NOFOLLOW)
    try:
        current_stat = os.fstat(current_fd)
        pending = list(parts[1:-1])
        symlink_hops = 0
        while pending:
            component = pending.pop(0)
            if not _trusted_directory_owner(current_stat, euid):
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.UNSAFE_PARENT,
                        reason="untrusted_directory_owner",
                    )
                )
            current_mode = stat.S_IMODE(current_stat.st_mode)
            current_writable = bool(current_mode & 0o022)
            current_sticky = bool(current_mode & stat.S_ISVTX)
            if current_writable and not current_sticky:
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.UNSAFE_PARENT,
                        reason="shared_writable_parent",
                    )
                )
            try:
                next_fd = _open_directory_component(current_fd, component)
            except FileNotFoundError:
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.UNSAFE_PARENT,
                        reason="missing_parent",
                    )
                ) from None
            except OSError as exc:
                current_fd, symlink_hops = _follow_trusted_symlink(
                    current_fd=current_fd,
                    component=component,
                    pending=pending,
                    hops=symlink_hops,
                    selected=selected,
                    exc=exc,
                    _close=_close,
                )
                current_stat = os.fstat(current_fd)
                continue

            transferred = False
            try:
                next_stat = os.fstat(next_fd)
                if not stat.S_ISDIR(next_stat.st_mode):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.LINK_OR_NON_REGULAR,
                            reason="non_directory_parent",
                        )
                    )
                if not _trusted_directory_owner(next_stat, euid):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.UNSAFE_PARENT,
                            reason="untrusted_directory_owner",
                        )
                    )
                old_fd = current_fd
                current_fd = next_fd
                transferred = True
                close(old_fd)
                current_stat = next_stat
            finally:
                if not transferred:
                    close(next_fd)

        if not _trusted_directory_owner(current_stat, euid):
            raise PrivatePathError(
                PrivatePathResult(
                    selected,
                    PrivatePathStatus.UNSAFE_PARENT,
                    reason="untrusted_directory_owner",
                )
            )
        final_parent_mode = stat.S_IMODE(current_stat.st_mode)
        final_parent_writable = bool(final_parent_mode & 0o022)
        final_parent_sticky = bool(final_parent_mode & stat.S_ISVTX)
        if final_parent_writable and (not final_parent_sticky or missing_leaf_allowed):
            raise PrivatePathError(
                PrivatePathResult(
                    selected,
                    PrivatePathStatus.UNSAFE_PARENT,
                    reason=(
                        "missing_leaf_in_shared_sticky_parent"
                        if final_parent_sticky and missing_leaf_allowed
                        else "shared_writable_parent"
                    ),
                )
            )
        return current_fd, parts[-1]
    except BaseException:
        close(current_fd)
        raise


def _private_path_error_from_oserror(
    selected: Path,
    exc: OSError,
) -> PrivatePathError:
    status = (
        PrivatePathStatus.LINK_OR_NON_REGULAR
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}
        else PrivatePathStatus.OPERATION_FAILED
    )
    return PrivatePathError(
        PrivatePathResult(
            selected,
            status,
            reason=type(exc).__name__,
        )
    )


def _open_leaf_for_create(parent_fd: int, leaf: str) -> int:
    return _native_open(
        leaf,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
        _PRIVATE_FILE_MODE,
        dir_fd=parent_fd,
    )


def _runtime_operation(path=None):
    raw = sys.modules.get("tldw_chatbook.Backup_Recovery.raw_participants")
    return raw._runtime_operation(path) if raw is not None else None


def _native_open(*args, **kwargs):
    operation = _runtime_operation()
    fd = os.open(*args, **kwargs)
    if operation is not None:
        raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
        raw._states[operation].descriptors.add(fd)
    return fd


def _native_close(fd):
    operation = _runtime_operation()
    if operation is None:
        os.close(fd)
    else:
        raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
        raw._close_descriptor(raw._states[operation], fd)


def _close_runtime_stream(operation, stream):
    """Retire a source stream only after an explicitly successful close."""
    raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
    state = raw._states[operation]
    try:
        stream.close()
        if not stream.closed:
            raise RuntimeError("raw_resources_not_retired")
    except BaseException:
        state.uncertain = True
        raise
    state.files.remove(stream)


def _admitted_file(function):
    @functools.wraps(function)
    def admitted(path, *args, **kwargs):
        from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage
        operation = _runtime_operation(lexical_path(path))
        if operation is not None:
            raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
            state = raw._check(operation)
            config = sys.modules.get("tldw_chatbook.config")
            if config is not None and state.source is config:
                selected = lexical_path(path)
                allowed = (
                    function.__name__ == "secure_private_directory" and selected in state.directories
                    or function.__name__ == "create_private_text" and selected in (state.selected, state.selected.with_name(state.selected.name + ".lock"))
                    or function.__name__ == "atomic_private_write_bytes" and selected in state.temporaries
                )
                if not allowed:
                    raise RuntimeError("raw_source_helper_not_supported")
            else:
                expected = (state.source.application_owned_directory
                            if function.__name__ == "secure_private_directory" else state.selected)
                if function.__name__ not in {"atomic_private_write_bytes", "secure_private_directory"} or expected is None or lexical_path(path) != lexical_path(expected) or (function.__name__ == "atomic_private_write_bytes" and state.temporary is None):
                    raise RuntimeError("raw_source_helper_not_supported")
            return function(path, *args, **kwargs)
        with acquire_storage(lexical_path(path)):
            return function(path, *args, **kwargs)
    return admitted


class _AdmittedStream:
    def __init__(self, stream, lease):
        self._stream = stream
        self._lease = lease

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._stream.close()
        self._lease.close()

    def __iter__(self):
        return iter(self._stream)

    def __del__(self):
        try:
            self.close()
        except BaseException:
            pass


class _ConfigStream:
    """A config lock stream whose native lifetime has no GC retirement path."""

    def __init__(self, stream, operation):
        self._stream = stream
        self._operation = operation
        self._fd = stream.fileno()
        self._retired = False

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def close(self):
        if self._retired:
            return
        raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
        raw._check(self._operation)
        _close_runtime_stream(self._operation, self._stream)
        _native_close(self._fd)
        self._retired = True


def _operation_temporary(operation, selected):
    raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
    state = raw._check(operation, selected, writing=True)
    return state.temporaries.get(selected, state.temporary)


def _admitted_stream(function):
    @functools.wraps(function)
    def admitted(path, *args, **kwargs):
        from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage
        operation = _runtime_operation()
        if operation is not None:
            raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
            state = raw._check(operation, lexical_path(path), writing=True)
            config = sys.modules.get("tldw_chatbook.config")
            if state.source is not config or lexical_path(path) != state.selected.with_name(state.selected.name + ".lock"):
                raise RuntimeError("raw_source_helper_not_supported")
            return _ConfigStream(function(path, *args, **kwargs), operation)
        lease = acquire_storage(lexical_path(path))
        try:
            return _AdmittedStream(function(path, *args, **kwargs), lease)
        except BaseException:
            lease.close()
            raise
    return admitted


def _admitted_reader(function):
    @functools.wraps(function)
    def admitted(path, *args, **kwargs):
        from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage
        operation = _runtime_operation(lexical_path(path))
        if operation is not None:
            raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
            state = raw._check(operation)
            config = sys.modules.get("tldw_chatbook.config")
            allowed = (state.selected,)
            if config is not None and state.source is config:
                allowed += (config._advanced_backup_path(state.selected),)
            if lexical_path(path) not in allowed:
                raise RuntimeError("raw_source_helper_not_supported")
            yield from function(path, *args, **kwargs)
            return
        with acquire_storage(lexical_path(path)):
            yield from function(path, *args, **kwargs)
    return admitted


@_admitted_file
def create_private_text(
    path: PathInput,
    text: str,
    *,
    application_owned_directory: PathInput | None = None,
    encoding: str = "utf-8",
) -> PrivatePathResult:
    """Create a new private text file without replacing an existing entry."""

    selected = lexical_path(path)
    payload = text.encode(encoding)
    if application_owned_directory is not None:
        owned_dir = lexical_path(application_owned_directory)
        if selected.parent != owned_dir:
            raise ValueError("Application-owned directory must be the target parent")
        secure_private_directory(
            owned_dir,
            create=True,
            application_owned=True,
        )

    if not _posix_guards_available():
        if _WINDOWS_PLATFORM:
            with selected.open("xb") as handle:
                handle.write(payload)
                handle.flush()
            return PrivatePathResult(
                selected,
                PrivatePathStatus.UNVERIFIED_PLATFORM,
                reason="native_acl_not_verified",
            )
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                reason="required_posix_guards_unavailable",
            )
        )

    parent_fd, leaf = _open_verified_parent(
        selected,
        missing_leaf_allowed=True,
    )
    file_fd = -1
    try:
        file_fd = _open_leaf_for_create(parent_fd, leaf)
        created_stat = os.fstat(file_fd)
        os.fchmod(file_fd, _PRIVATE_FILE_MODE)
        view = memoryview(payload)
        while view:
            written = os.write(file_fd, view)
            if written == 0:
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.OPERATION_FAILED,
                        reason="zero_byte_write",
                    )
                )
            view = view[written:]
        os.fsync(file_fd)
        if not _private_file_postcondition_holds(
            file_fd,
            parent_fd,
            leaf,
            expected_identity=created_stat,
        ):
            raise PrivatePathError(
                PrivatePathResult(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    reason="private_file_postcondition_failed",
                )
            )
        return PrivatePathResult(selected, PrivatePathStatus.CREATED_PRIVATE)
    except FileExistsError:
        raise
    except PrivatePathError:
        raise
    except OSError as exc:
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                reason=type(exc).__name__,
            )
        ) from None
    finally:
        if file_fd >= 0:
            _native_close(file_fd)
        _native_close(parent_fd)


def _prepare_application_owned_parent(
    selected: Path,
    application_owned_directory: PathInput | None,
) -> None:
    if application_owned_directory is None:
        return
    owned_dir = lexical_path(application_owned_directory)
    if selected.parent != owned_dir:
        raise ValueError("Application-owned directory must be the target parent")
    secure_private_directory(
        owned_dir,
        create=True,
        application_owned=True,
    )


@_admitted_file
def atomic_private_write_bytes(
    path: PathInput,
    payload: bytes,
    *,
    application_owned_directory: PathInput | None = None,
) -> PrivatePathResult:
    """Atomically replace a private file without following its target."""

    selected = lexical_path(path)
    operation = _runtime_operation(selected)
    raw = sys.modules.get("tldw_chatbook.Backup_Recovery.raw_participants")
    _prepare_application_owned_parent(selected, application_owned_directory)

    if not _atomic_posix_guards_available():
        if _WINDOWS_PLATFORM:
            try:
                if operation is None:
                    fd, temporary = tempfile.mkstemp(
                        dir=selected.parent,
                        prefix=f".{selected.name}.",
                        suffix=".tmp",
                    )
                else:
                    temporary = _operation_temporary(operation, selected)
                    fd = _native_open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _PRIVATE_FILE_MODE)
                    info = os.fstat(fd)
                    raw._check(operation).created_files[temporary] = (info.st_dev, info.st_ino)
            except OSError as exc:
                raise _private_path_error_from_oserror(selected, exc) from None
            try:
                with os.fdopen(fd, "wb", closefd=operation is None) as stream:
                    if operation is not None:
                        raw._check(operation).files.append(stream)
                    stream.write(payload)
                    stream.flush()
                if operation is not None:
                    raw._check(operation).files.remove(stream)
                    _native_close(fd)
                    raw._replace(operation, temporary, selected)
                else:
                    os.replace(temporary, selected)
            except BaseException:
                if operation is not None:
                    state = raw._states[operation]
                    if state.files or fd in state.descriptors:
                        state.uncertain = True
                    if not state.uncertain:
                        raw._remove_temporary(operation, temporary)
                else:
                    try:
                        os.unlink(temporary)
                    except FileNotFoundError:
                        pass
                raise
            return PrivatePathResult(
                selected,
                PrivatePathStatus.UNVERIFIED_PLATFORM,
                reason="native_acl_not_verified",
            )
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                reason="required_posix_guards_unavailable",
            )
        )

    parent_fd, leaf = _open_verified_parent(
        selected,
        missing_leaf_allowed=True,
    )
    operation = _runtime_operation(selected)
    raw = sys.modules.get("tldw_chatbook.Backup_Recovery.raw_participants")
    temporary_leaf = (_operation_temporary(operation, selected).name if operation is not None
                      else f".{leaf}.{secrets.token_hex(8)}.tmp")
    temporary_fd = -1
    temporary_exists = False
    published = False
    try:
        try:
            existing_stat = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing_stat = None
            prior_mode = None
        else:
            rejected = _classify_private_file_stat(
                existing_stat,
                expected_uid=os.geteuid(),
            )
            if rejected is not None:
                raise PrivatePathError(PrivatePathResult(selected, rejected))
            prior_mode = stat.S_IMODE(existing_stat.st_mode)
            existing_fd = _native_open(
                leaf,
                _PRIVATE_FILE_OPEN_FLAGS,
                dir_fd=parent_fd,
            )
            try:
                opened_stat = os.fstat(existing_fd)
                if not _same_identity(opened_stat, existing_stat):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.OPERATION_FAILED,
                            reason="target_replaced",
                        )
                    )
                if prior_mode != _PRIVATE_FILE_MODE:
                    os.fchmod(existing_fd, _PRIVATE_FILE_MODE)
            finally:
                _native_close(existing_fd)

        temporary_fd = _open_leaf_for_create(parent_fd, temporary_leaf)
        temporary_exists = True
        temporary_stat = os.fstat(temporary_fd)
        if operation is not None:
            raw._check(operation).created_files[selected.parent / temporary_leaf] = (temporary_stat.st_dev, temporary_stat.st_ino)
        view = memoryview(payload)
        while view:
            written = os.write(temporary_fd, view)
            if written == 0:
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.OPERATION_FAILED,
                        reason="zero_byte_write",
                    )
                )
            view = view[written:]
        os.fchmod(temporary_fd, _PRIVATE_FILE_MODE)
        os.fsync(temporary_fd)

        try:
            current_stat = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            current_stat = None
        if existing_stat is None:
            if current_stat is not None:
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.OPERATION_FAILED,
                        reason="target_appeared",
                    )
                )
        elif current_stat is None or not _same_identity(current_stat, existing_stat):
            raise PrivatePathError(
                PrivatePathResult(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    reason="target_replaced",
                )
            )

        if operation is not None:
            raw._check_temporary_identity(raw._check(operation), selected.parent / temporary_leaf)
        try:
            os.rename(
                temporary_leaf,
                leaf,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        except BaseException:
            if operation is not None:
                raw._states[operation].uncertain = True
            raise
        temporary_exists = False
        published = True
        if operation is not None:
            raw._check(operation).created_files.pop(selected.parent / temporary_leaf)
        os.fsync(parent_fd)
        if not _private_file_postcondition_holds(
            temporary_fd,
            parent_fd,
            leaf,
            expected_identity=temporary_stat,
        ):
            raise PrivatePathError(
                PrivatePathResult(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    reason="private_file_postcondition_failed",
                )
            )
        if existing_stat is None:
            status = PrivatePathStatus.CREATED_PRIVATE
        elif prior_mode != _PRIVATE_FILE_MODE:
            status = PrivatePathStatus.HARDENED_PRIVATE
        else:
            status = PrivatePathStatus.ALREADY_PRIVATE
        return PrivatePathResult(selected, status)
    except BaseException as exc:
        if operation is not None and published:
            raw._states[operation].uncertain = True
        if isinstance(exc, OSError) and not isinstance(exc, PrivatePathError):
            raise _private_path_error_from_oserror(selected, exc) from None
        raise
    finally:
        if temporary_fd >= 0:
            _native_close(temporary_fd)
        try:
            if temporary_exists:
                try:
                    if operation is not None:
                        raw._remove_temporary(operation, selected.parent / temporary_leaf)
                    else:
                        os.unlink(temporary_leaf, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
        finally:
            _native_close(parent_fd)


def atomic_private_write_text(
    path: PathInput,
    text: str,
    *,
    application_owned_directory: PathInput | None = None,
    encoding: str = "utf-8",
) -> PrivatePathResult:
    """Atomically replace a private text file."""

    return atomic_private_write_bytes(
        path,
        text.encode(encoding),
        application_owned_directory=application_owned_directory,
    )


@contextlib.contextmanager
def open_private_text_append(
    path: PathInput,
    *,
    application_owned_directory: PathInput | None = None,
    encoding: str = "utf-8",
    errors: str | None = None,
) -> Iterator[TextIO]:
    """Open a private text file for append without following its target."""

    stream = open_private_text_append_stream(
        path,
        application_owned_directory=application_owned_directory,
        encoding=encoding,
        errors=errors,
    )
    with stream:
        yield stream
        stream.flush()
        os.fsync(stream.fileno())


@_admitted_stream
def open_private_text_append_stream(
    path: PathInput,
    *,
    application_owned_directory: PathInput | None = None,
    encoding: str = "utf-8",
    errors: str | None = None,
) -> TextIO:
    """Return a pinned private text stream opened for append."""

    selected = lexical_path(path)
    _prepare_application_owned_parent(selected, application_owned_directory)

    if not _posix_guards_available():
        if _WINDOWS_PLATFORM:
            selected.parent.mkdir(parents=True, exist_ok=True)
            operation = _runtime_operation(selected)
            if operation is None:
                return selected.open("a", encoding=encoding, errors=errors)
            fd = _native_open(selected, os.O_WRONLY | os.O_APPEND | os.O_CREAT, _PRIVATE_FILE_MODE)
            try:
                stream = os.fdopen(fd, "a", encoding=encoding, errors=errors, closefd=False)
            except BaseException:
                _native_close(fd)
                raise
            raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
            raw._check(operation).files.append(stream)
            return stream
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                reason="required_posix_guards_unavailable",
            )
        )

    parent_fd, leaf = _open_verified_parent(
        selected,
        missing_leaf_allowed=True,
    )
    file_fd = -1
    try:
        try:
            entry_stat = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            entry_stat = None
        else:
            rejected = _classify_private_file_stat(
                entry_stat,
                expected_uid=os.geteuid(),
            )
            if rejected is not None:
                raise PrivatePathError(PrivatePathResult(selected, rejected))

        file_fd = _native_open(
            leaf,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | _NOFOLLOW | _NONBLOCK | _NOCTTY,
            _PRIVATE_FILE_MODE,
            dir_fd=parent_fd,
        )
        file_stat = os.fstat(file_fd)
        rejected = _classify_private_file_stat(
            file_stat,
            expected_uid=os.geteuid(),
        )
        if rejected is not None:
            raise PrivatePathError(PrivatePathResult(selected, rejected))
        if entry_stat is not None and not _same_identity(file_stat, entry_stat):
            raise PrivatePathError(
                PrivatePathResult(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    reason="target_replaced",
                )
            )
        os.fchmod(file_fd, _PRIVATE_FILE_MODE)
        if not _private_file_postcondition_holds(
            file_fd,
            parent_fd,
            leaf,
            expected_identity=file_stat,
        ):
            raise PrivatePathError(
                PrivatePathResult(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    reason="private_file_postcondition_failed",
                )
            )
        operation = _runtime_operation(selected)
        stream = os.fdopen(
            file_fd,
            "a",
            encoding=encoding,
            errors=errors,
            closefd=operation is None,
        )
        if operation is not None:
            raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
            raw._check(operation).files.append(stream)
        file_fd = -1
        return stream
    except PrivatePathError:
        raise
    except OSError as exc:
        raise _private_path_error_from_oserror(selected, exc) from None
    finally:
        if file_fd >= 0:
            _native_close(file_fd)
        _native_close(parent_fd)


@contextlib.contextmanager
@_admitted_reader
def open_private_binary(path: PathInput) -> Iterator[PrivateBinaryFile]:
    """Open and harden an existing private file without following links."""

    selected = lexical_path(path)
    if not _posix_guards_available():
        if _WINDOWS_PLATFORM:
            operation = _runtime_operation(selected)
            if operation is None:
                with selected.open("rb") as stream:
                    yield PrivateBinaryFile(
                        stream=stream,
                        result=PrivatePathResult(selected, PrivatePathStatus.UNVERIFIED_PLATFORM, reason="native_acl_not_verified"),
                    )
            else:
                raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
                state = raw._check(operation)
                fd = _native_open(selected, os.O_RDONLY | getattr(os, "O_BINARY", 0))
                try:
                    stream = os.fdopen(fd, "rb", closefd=False)
                    state.files.append(stream)
                    try:
                        yield PrivateBinaryFile(
                            stream=stream,
                            result=PrivatePathResult(selected, PrivatePathStatus.UNVERIFIED_PLATFORM, reason="native_acl_not_verified"),
                        )
                    finally:
                        _close_runtime_stream(operation, stream)
                finally:
                    _native_close(fd)
            return
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                reason="required_posix_guards_unavailable",
            )
        )

    parent_fd, leaf = _open_verified_parent(
        selected,
        missing_leaf_allowed=False,
    )
    file_fd = -1
    try:
        try:
            entry_stat = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise _private_path_error_from_oserror(selected, exc) from None

        if not stat.S_ISREG(entry_stat.st_mode):
            raise PrivatePathError(
                PrivatePathResult(
                    selected,
                    PrivatePathStatus.LINK_OR_NON_REGULAR,
                )
            )

        try:
            file_fd = _native_open(
                leaf,
                _PRIVATE_FILE_OPEN_FLAGS,
                dir_fd=parent_fd,
            )
        except FileNotFoundError as exc:
            raise _private_path_error_from_oserror(selected, exc) from None
        except OSError as exc:
            raise _private_path_error_from_oserror(selected, exc) from None

        try:
            file_stat = os.fstat(file_fd)
            rejected = _classify_private_file_stat(
                file_stat,
                expected_uid=os.geteuid(),
            )
            if rejected is not None:
                raise PrivatePathError(PrivatePathResult(selected, rejected))
            prior_mode = stat.S_IMODE(file_stat.st_mode)
            if prior_mode != _PRIVATE_FILE_MODE:
                os.fchmod(file_fd, _PRIVATE_FILE_MODE)
                status = PrivatePathStatus.HARDENED_PRIVATE
            else:
                status = PrivatePathStatus.ALREADY_PRIVATE
            if not _private_file_postcondition_holds(
                file_fd,
                parent_fd,
                leaf,
                expected_identity=file_stat,
            ):
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.OPERATION_FAILED,
                        reason="private_file_postcondition_failed",
                    )
                )
            operation = _runtime_operation(selected)
            stream = os.fdopen(file_fd, "rb", closefd=operation is None)
            stream_fd = file_fd
            if operation is not None:
                raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
                raw._check(operation).files.append(stream)
            file_fd = -1
        except PrivatePathError:
            raise
        except OSError as exc:
            raise _private_path_error_from_oserror(selected, exc) from None
    finally:
        if file_fd >= 0:
            _native_close(file_fd)
        _native_close(parent_fd)

    if operation is None:
        with stream:
            yield PrivateBinaryFile(
                stream=stream,
                result=PrivatePathResult(selected, status),
            )
    else:
        try:
            yield PrivateBinaryFile(
                stream=stream,
                result=PrivatePathResult(selected, status),
            )
        finally:
            _close_runtime_stream(operation, stream)
            _native_close(stream_fd)


@_admitted_file
def secure_private_directory(
    path: PathInput,
    *,
    create: bool,
    application_owned: bool,
) -> PrivatePathResult:
    """Create or harden an application-owned directory."""

    selected = lexical_path(path)
    if not application_owned:
        raise ValueError("Only application-owned directories may be changed")
    if selected == Path(selected.anchor):
        raise ValueError("The filesystem root cannot be application-owned")
    if not _posix_guards_available():
        if _WINDOWS_PLATFORM:
            if create:
                selected.mkdir(parents=True, exist_ok=True)
            return PrivatePathResult(
                selected,
                PrivatePathStatus.UNVERIFIED_PLATFORM,
                reason="native_acl_not_verified",
            )
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                reason="required_posix_guards_unavailable",
            )
        )

    euid = os.geteuid()
    parts = selected.parts
    current_fd = _native_open(os.sep, _DIRECTORY_OPEN_FLAGS | _NOFOLLOW)
    created_final = False
    hardened_final = False
    try:
        current_stat = os.fstat(current_fd)
        pending = list(parts[1:])
        symlink_hops = 0
        while pending:
            component = pending.pop(0)
            is_final = not pending
            if not _trusted_directory_owner(current_stat, euid):
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.UNSAFE_PARENT,
                        reason="untrusted_directory_owner",
                    )
                )
            current_mode = stat.S_IMODE(current_stat.st_mode)
            shared_writable = bool(current_mode & 0o022)
            sticky = bool(current_mode & stat.S_ISVTX)
            if shared_writable and not sticky:
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.UNSAFE_PARENT,
                        reason="shared_writable_parent",
                    )
                )

            created_component = False
            try:
                next_fd = _open_directory_component(current_fd, component)
            except FileNotFoundError:
                if not create:
                    raise
                if shared_writable:
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.UNSAFE_PARENT,
                            reason="missing_component_in_shared_sticky_parent",
                        )
                    ) from None
                os.mkdir(
                    component,
                    mode=_PRIVATE_DIRECTORY_MODE,
                    dir_fd=current_fd,
                )
                next_fd = _open_directory_component(current_fd, component)
                created_component = True
            except OSError as exc:
                current_fd, symlink_hops = _follow_trusted_symlink(
                    current_fd=current_fd,
                    component=component,
                    pending=pending,
                    hops=symlink_hops,
                    selected=selected,
                    exc=exc,
                )
                current_stat = os.fstat(current_fd)
                continue

            transferred = False
            try:
                next_stat = os.fstat(next_fd)
                if not stat.S_ISDIR(next_stat.st_mode):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.LINK_OR_NON_REGULAR,
                            reason="non_directory_component",
                        )
                    )
                if next_stat.st_uid != euid and (created_component or is_final):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.WRONG_OWNER,
                            reason="application_directory_wrong_owner",
                        )
                    )
                if not _trusted_directory_owner(next_stat, euid):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.UNSAFE_PARENT,
                            reason="untrusted_directory_owner",
                        )
                    )
                if shared_writable and next_stat.st_uid not in {0, euid}:
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.UNSAFE_PARENT,
                            reason="sticky_child_wrong_owner",
                        )
                    )

                if created_component or is_final:
                    before = stat.S_IMODE(next_stat.st_mode)
                    if before != _PRIVATE_DIRECTORY_MODE:
                        os.fchmod(next_fd, _PRIVATE_DIRECTORY_MODE)
                        if is_final:
                            hardened_final = True
                    if not _private_directory_postcondition_holds(
                        next_fd,
                        current_fd,
                        component,
                        expected_identity=next_stat,
                    ):
                        raise PrivatePathError(
                            PrivatePathResult(
                                selected,
                                PrivatePathStatus.OPERATION_FAILED,
                                reason="private_directory_postcondition_failed",
                            )
                        )
                if is_final and created_component:
                    created_final = True

                old_fd = current_fd
                current_fd = next_fd
                transferred = True
                _native_close(old_fd)
                current_stat = os.fstat(current_fd)
            finally:
                if not transferred:
                    _native_close(next_fd)

        status = (
            PrivatePathStatus.CREATED_PRIVATE
            if created_final
            else (
                PrivatePathStatus.HARDENED_PRIVATE
                if hardened_final
                else PrivatePathStatus.ALREADY_PRIVATE
            )
        )
        return PrivatePathResult(selected, status)
    except PrivatePathError:
        raise
    except OSError as exc:
        raise _private_path_error_from_oserror(selected, exc) from None
    finally:
        _native_close(current_fd)


def verify_trusted_directory(
    path: PathInput,
    *,
    allow_shared_sticky: bool,
) -> PrivatePathResult:
    """Verify an existing lexical directory without creating or changing it.

    Security intent — do not weaken this to make a test pass:

    The walk starts at `/` and opens one component at a time through `dir_fd`,
    with `O_NOFOLLOW`, checking every directory it crosses is owned by root or
    the effective uid and is not group/world-writable without the sticky bit.
    The point is that no other local user can ever redirect the final directory,
    and that the answer cannot change between the check and the open, because
    the path string is never re-derived after the first component.

    A symlink component is crossed only when the symlink *itself* passes the
    same trust test (see `_trusted_symlink`: root-owned, no group/world write
    bits), and traversal then resumes from its target under the identical
    per-component rules, capped at `_MAX_TRUSTED_SYMLINK_HOPS`. That is what
    lets macOS reach `/var/folders/...` across `/var -> private/var`. Anything
    else — a caller-owned alias, a link in a shared directory, a non-directory —
    still fails with `LINK_OR_NON_REGULAR`.

    Resolving the path up front (`Path.resolve()`, `os.path.realpath`) would be
    a much simpler way to reach the same directories and is the wrong answer: it
    follows every symlink before anything has been vetted and re-opens a name
    that may have changed since. `lexical_path` deliberately does not resolve.
    """

    selected = lexical_path(path)
    operation = _runtime_operation()
    if operation is not None:
        raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
        state = raw._check(operation)
        if selected != state.selected.parent and selected not in state.directories:
            raise RuntimeError("raw_source_helper_not_supported")

    if not _posix_guards_available():
        if _WINDOWS_PLATFORM:
            if not selected.is_dir():
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.OPERATION_FAILED,
                        reason="missing_or_non_directory",
                    )
                )
            return PrivatePathResult(
                selected,
                PrivatePathStatus.UNVERIFIED_PLATFORM,
                reason="native_acl_not_verified",
            )
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                reason="required_posix_guards_unavailable",
            )
        )

    parts = selected.parts
    if len(parts) < 2 or parts[0] != os.sep:
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                reason="invalid_absolute_path",
            )
        )

    euid = os.geteuid()
    current_fd = _native_open(os.sep, _DIRECTORY_OPEN_FLAGS | _NOFOLLOW)
    try:
        current_stat = os.fstat(current_fd)
        pending = list(parts[1:])
        symlink_hops = 0
        while pending:
            component = pending.pop(0)
            is_final = not pending
            if not _trusted_directory_owner(current_stat, euid):
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.UNSAFE_PARENT,
                        reason="untrusted_directory_owner",
                    )
                )
            current_mode = stat.S_IMODE(current_stat.st_mode)
            current_writable = bool(current_mode & 0o022)
            current_sticky = bool(current_mode & stat.S_ISVTX)
            if current_writable and not current_sticky:
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.UNSAFE_PARENT,
                        reason="shared_writable_parent",
                    )
                )

            try:
                next_fd = _open_directory_component(current_fd, component)
            except FileNotFoundError:
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.UNSAFE_PARENT,
                        reason="missing_directory",
                    )
                ) from None
            except OSError as exc:
                current_fd, symlink_hops = _follow_trusted_symlink(
                    current_fd=current_fd,
                    component=component,
                    pending=pending,
                    hops=symlink_hops,
                    selected=selected,
                    exc=exc,
                )
                current_stat = os.fstat(current_fd)
                continue

            transferred = False
            try:
                next_stat = os.fstat(next_fd)
                if not stat.S_ISDIR(next_stat.st_mode):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.LINK_OR_NON_REGULAR,
                            reason="non_directory_component",
                        )
                    )
                if not _trusted_directory_owner(next_stat, euid):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            (
                                PrivatePathStatus.WRONG_OWNER
                                if is_final
                                else PrivatePathStatus.UNSAFE_PARENT
                            ),
                            reason="untrusted_directory_owner",
                        )
                    )
                if not _trusted_directory_postcondition_holds(
                    next_fd,
                    current_fd,
                    component,
                    expected_identity=next_stat,
                ):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.OPERATION_FAILED,
                            reason="trusted_directory_postcondition_failed",
                        )
                    )

                next_mode = stat.S_IMODE(next_stat.st_mode)
                next_writable = bool(next_mode & 0o022)
                next_sticky = bool(next_mode & stat.S_ISVTX)
                if next_writable and (
                    not next_sticky or (is_final and not allow_shared_sticky)
                ):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.UNSAFE_PARENT,
                            reason=(
                                "shared_sticky_directory_not_allowed"
                                if next_sticky and is_final
                                else "shared_writable_parent"
                            ),
                        )
                    )

                old_fd = current_fd
                current_fd = next_fd
                transferred = True
                _native_close(old_fd)
                current_stat = next_stat
            finally:
                if not transferred:
                    _native_close(next_fd)

        return PrivatePathResult(
            selected,
            PrivatePathStatus.TRUSTED_DIRECTORY,
            reason="trusted_existing_directory",
        )
    except PrivatePathError:
        raise
    except OSError as exc:
        raise _private_path_error_from_oserror(selected, exc) from None
    finally:
        _native_close(current_fd)
