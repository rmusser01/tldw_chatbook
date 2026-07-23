"""Private local-file lifecycle primitives.

This module is deliberately dependency-leaf: callers choose failure policy and
diagnostics while this module performs lexical selection and filesystem checks.
"""

from __future__ import annotations

import contextlib
import errno
import os
import stat
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO, Iterator, TypeAlias

PathInput: TypeAlias = str | os.PathLike[str]

_PRIVATE_FILE_MODE = 0o600
_PRIVATE_DIRECTORY_MODE = 0o700
_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_WINDOWS_PLATFORM = os.name == "nt"


class PrivatePathStatus(StrEnum):
    CREATED_PRIVATE = "created_private"
    HARDENED_PRIVATE = "hardened_private"
    ALREADY_PRIVATE = "already_private"
    UNSAFE_PARENT = "unsafe_parent"
    WRONG_OWNER = "wrong_owner"
    LINK_OR_NON_REGULAR = "link_or_non_regular"
    OPERATION_FAILED = "operation_failed"
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
        return self.verified_private or (
            self.status is PrivatePathStatus.UNVERIFIED_PLATFORM
        )


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
    required_dir_fd = {os.open, os.stat, os.mkdir, os.unlink}
    return (
        os.name == "posix"
        and _NOFOLLOW != 0
        and getattr(os, "O_DIRECTORY", 0) != 0
        and required_dir_fd.issubset(os.supports_dir_fd)
        and os.stat in os.supports_follow_symlinks
        and hasattr(os, "geteuid")
        and hasattr(os, "fstat")
        and hasattr(os, "fchmod")
        and hasattr(os, "fsync")
    )


def _classify_private_file_stat(
    file_stat: os.stat_result,
    *,
    expected_uid: int,
) -> PrivatePathStatus | None:
    if not stat.S_ISREG(file_stat.st_mode):
        return PrivatePathStatus.LINK_OR_NON_REGULAR
    if file_stat.st_uid != expected_uid:
        return PrivatePathStatus.WRONG_OWNER
    return None


def _trusted_directory_owner(directory_stat: os.stat_result, euid: int) -> bool:
    return directory_stat.st_uid in {0, euid}


def _open_directory_component(parent_fd: int, component: str) -> int:
    return os.open(
        component,
        _DIRECTORY_OPEN_FLAGS | _NOFOLLOW,
        dir_fd=parent_fd,
    )


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


def _open_verified_parent(
    selected: Path,
    *,
    missing_leaf_allowed: bool,
) -> tuple[int, str]:
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
    current_fd = os.open(os.sep, _DIRECTORY_OPEN_FLAGS | _NOFOLLOW)
    try:
        current_stat = os.fstat(current_fd)
        for component in parts[1:-1]:
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
                status = (
                    PrivatePathStatus.LINK_OR_NON_REGULAR
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}
                    else PrivatePathStatus.OPERATION_FAILED
                )
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        status,
                        reason=type(exc).__name__,
                    )
                ) from None

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
                os.close(old_fd)
                current_stat = next_stat
            finally:
                if not transferred:
                    os.close(next_fd)

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
        os.close(current_fd)
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


@contextlib.contextmanager
def open_private_binary(path: PathInput) -> Iterator[PrivateBinaryFile]:
    """Open and harden an existing private file without following links."""

    selected = lexical_path(path)
    if not _posix_guards_available():
        if _WINDOWS_PLATFORM:
            with selected.open("rb") as stream:
                yield PrivateBinaryFile(
                    stream=stream,
                    result=PrivatePathResult(
                        selected,
                        PrivatePathStatus.UNVERIFIED_PLATFORM,
                        reason="native_acl_not_verified",
                    ),
                )
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
            file_fd = os.open(leaf, os.O_RDONLY | _NOFOLLOW, dir_fd=parent_fd)
        except FileNotFoundError:
            raise
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
            stream = os.fdopen(file_fd, "rb", closefd=True)
            file_fd = -1
        except PrivatePathError:
            raise
        except OSError as exc:
            raise _private_path_error_from_oserror(selected, exc) from None
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(parent_fd)

    with stream:
        yield PrivateBinaryFile(
            stream=stream,
            result=PrivatePathResult(selected, status),
        )


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
    current_fd = os.open(os.sep, _DIRECTORY_OPEN_FLAGS | _NOFOLLOW)
    created_final = False
    hardened_final = False
    try:
        current_stat = os.fstat(current_fd)
        for index, component in enumerate(parts[1:], start=1):
            is_final = index == len(parts) - 1
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
                raise _private_path_error_from_oserror(selected, exc) from None

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
                os.close(old_fd)
                current_stat = os.fstat(current_fd)
            finally:
                if not transferred:
                    os.close(next_fd)

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
        os.close(current_fd)
