"""Shared raw SQLite artifact checks. Live callers must use the exec boundary.

The local seam temporarily delegates here during the staged migration. No SQL,
application startup, policy registry or transport lives in this dependency leaf.
"""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Callable
from pathlib import Path

from tldw_chatbook.DB.private_sqlite_protocol import (
    FileIdentity,
    PrepareRequest,
    PrepareResult,
)
from tldw_chatbook.Utils import private_paths
from tldw_chatbook.Utils.private_paths import (
    PrivatePathError,
    PrivatePathResult,
    PrivatePathStatus,
)

_PRIVATE_FILE_MODE = 0o600
_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


def _failure(
    selected: Path,
    status: PrivatePathStatus,
    reason: str,
) -> PrivatePathError:
    return PrivatePathError(PrivatePathResult(selected, status, reason=reason))


def _open_artifact_fd(
    parent_fd: int,
    leaf: str,
    *,
    writable: bool,
    create: bool,
) -> int:
    flags = os.O_RDWR if writable else os.O_RDONLY
    flags |= (
        getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOCTTY", 0)
    )
    if create:
        flags |= os.O_CREAT | os.O_EXCL
    return os.open(
        leaf,
        flags,
        _PRIVATE_FILE_MODE,
        dir_fd=parent_fd,
    )


def _artifact_postcondition_holds(
    file_fd: int,
    parent_fd: int,
    leaf: str,
    *,
    expected_identity: os.stat_result,
    selected: Path,
    enforce_private_mode: bool = True,
) -> bool:
    del selected
    opened = os.fstat(file_fd)
    entry = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    return (
        private_paths._same_identity(opened, expected_identity)
        and private_paths._same_identity(entry, expected_identity)
        and stat.S_ISREG(opened.st_mode)
        and opened.st_nlink == 1
        and entry.st_nlink == 1
        and opened.st_uid == os.geteuid()
        and (
            not enforce_private_mode
            or stat.S_IMODE(opened.st_mode) == _PRIVATE_FILE_MODE
        )
    )


def _path_error_from_oserror(selected: Path, exc: OSError) -> PrivatePathError:
    status = (
        PrivatePathStatus.LINK_OR_NON_REGULAR
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}
        else PrivatePathStatus.OPERATION_FAILED
    )
    return _failure(selected, status, type(exc).__name__)


class _OptionalSQLiteGenerationChanged(Exception):
    """Restart optional-sidecar validation against the current named inode."""


_OPTIONAL_SIDECAR_REVALIDATION_ATTEMPTS = 4


def _optional_sidecar_restart_or_absent(
    parent_fd: int,
    leaf: str,
    selected: Path,
) -> bool:
    try:
        current = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise _path_error_from_oserror(selected, exc) from None

    rejected = private_paths._classify_private_file_stat(
        current,
        expected_uid=os.geteuid(),
    )
    if rejected is not None:
        raise _failure(selected, rejected, "unsafe_sqlite_artifact")
    raise _OptionalSQLiteGenerationChanged


def _prepare_posix_artifact_generation(
    selected: Path,
    *,
    writable: bool,
    create_if_missing: bool,
    optional: bool,
    enforce_private_mode: bool,
    open_artifact_fd: Callable[..., int] | None = None,
    postcondition_holds: Callable[..., bool] | None = None,
    identity_out: list[os.stat_result] | None = None,
) -> bool:
    open_artifact_fd = open_artifact_fd or _open_artifact_fd
    postcondition = postcondition_holds or _artifact_postcondition_holds
    parent_fd, leaf = private_paths._open_verified_parent(
        selected,
        missing_leaf_allowed=create_if_missing,
    )
    file_fd = -1
    writable_fd = -1
    try:
        try:
            entry_stat = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            if optional:
                return False
            if not create_if_missing:
                raise _failure(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    "missing_sqlite_artifact",
                ) from None
            entry_stat = None
        except OSError as exc:
            raise _path_error_from_oserror(selected, exc) from None

        if entry_stat is not None and not stat.S_ISREG(entry_stat.st_mode):
            raise _failure(
                selected,
                PrivatePathStatus.LINK_OR_NON_REGULAR,
                "non_regular_sqlite_artifact",
            )
        if (
            optional
            and entry_stat is not None
            and entry_stat.st_nlink == 0
            and entry_stat.st_uid == os.geteuid()
        ):
            return _optional_sidecar_restart_or_absent(
                parent_fd,
                leaf,
                selected,
            )
        if entry_stat is not None:
            entry_rejected = private_paths._classify_private_file_stat(
                entry_stat,
                expected_uid=os.geteuid(),
            )
            if entry_rejected is not None:
                raise _failure(
                    selected,
                    entry_rejected,
                    "unsafe_sqlite_artifact",
                )

        created = entry_stat is None
        try:
            file_fd = open_artifact_fd(
                parent_fd,
                leaf,
                writable=created,
                create=created,
            )
        except FileNotFoundError:
            if optional:
                return _optional_sidecar_restart_or_absent(
                    parent_fd,
                    leaf,
                    selected,
                )
            raise _failure(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                "missing_sqlite_artifact",
            ) from None
        except OSError as exc:
            raise _path_error_from_oserror(selected, exc) from None

        opened_stat = os.fstat(file_fd)
        if optional and opened_stat.st_nlink == 0:
            if not stat.S_ISREG(opened_stat.st_mode):
                raise _failure(
                    selected,
                    PrivatePathStatus.LINK_OR_NON_REGULAR,
                    "unsafe_sqlite_artifact",
                )
            if opened_stat.st_uid != os.geteuid():
                raise _failure(
                    selected,
                    PrivatePathStatus.WRONG_OWNER,
                    "unsafe_sqlite_artifact",
                )
            return _optional_sidecar_restart_or_absent(
                parent_fd,
                leaf,
                selected,
            )
        rejected = private_paths._classify_private_file_stat(
            opened_stat,
            expected_uid=os.geteuid(),
        )
        if rejected is not None:
            raise _failure(selected, rejected, "unsafe_sqlite_artifact")
        if entry_stat is not None and not private_paths._same_identity(
            entry_stat,
            opened_stat,
        ):
            if optional:
                return _optional_sidecar_restart_or_absent(
                    parent_fd,
                    leaf,
                    selected,
                )
            raise _failure(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                "private_sqlite_identity_changed",
            )

        if (
            enforce_private_mode
            and stat.S_IMODE(opened_stat.st_mode) != _PRIVATE_FILE_MODE
        ):
            os.fchmod(file_fd, _PRIVATE_FILE_MODE)
        try:
            postcondition_holds = postcondition(
                file_fd,
                parent_fd,
                leaf,
                expected_identity=opened_stat,
                selected=selected,
                enforce_private_mode=enforce_private_mode,
            )
        except FileNotFoundError:
            if optional:
                return _optional_sidecar_restart_or_absent(
                    parent_fd,
                    leaf,
                    selected,
                )
            raise _failure(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                "private_sqlite_postcondition_failed",
            ) from None
        if not postcondition_holds:
            if optional:
                return _optional_sidecar_restart_or_absent(
                    parent_fd,
                    leaf,
                    selected,
                )
            raise _failure(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                "private_sqlite_postcondition_failed",
            )

        if writable and not created:
            try:
                writable_fd = open_artifact_fd(
                    parent_fd,
                    leaf,
                    writable=True,
                    create=False,
                )
            except FileNotFoundError:
                if optional:
                    return _optional_sidecar_restart_or_absent(
                        parent_fd,
                        leaf,
                        selected,
                    )
                raise _failure(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    "missing_sqlite_artifact",
                ) from None
            except OSError as exc:
                raise _path_error_from_oserror(selected, exc) from None

            writable_stat = os.fstat(writable_fd)
            if optional and writable_stat.st_nlink == 0:
                if not stat.S_ISREG(writable_stat.st_mode):
                    raise _failure(
                        selected,
                        PrivatePathStatus.LINK_OR_NON_REGULAR,
                        "unsafe_sqlite_artifact",
                    )
                if writable_stat.st_uid != os.geteuid():
                    raise _failure(
                        selected,
                        PrivatePathStatus.WRONG_OWNER,
                        "unsafe_sqlite_artifact",
                    )
                return _optional_sidecar_restart_or_absent(
                    parent_fd,
                    leaf,
                    selected,
                )
            rejected = private_paths._classify_private_file_stat(
                writable_stat,
                expected_uid=os.geteuid(),
            )
            if rejected is not None:
                raise _failure(selected, rejected, "unsafe_sqlite_artifact")
            if not private_paths._same_identity(opened_stat, writable_stat):
                if optional:
                    return _optional_sidecar_restart_or_absent(
                        parent_fd,
                        leaf,
                        selected,
                    )
                raise _failure(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    "private_sqlite_identity_changed",
                )
            try:
                writable_postcondition_holds = postcondition(
                    writable_fd,
                    parent_fd,
                    leaf,
                    expected_identity=opened_stat,
                    selected=selected,
                    enforce_private_mode=enforce_private_mode,
                )
            except FileNotFoundError:
                if optional:
                    return _optional_sidecar_restart_or_absent(
                        parent_fd,
                        leaf,
                        selected,
                    )
                raise _failure(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    "private_sqlite_postcondition_failed",
                ) from None
            if not writable_postcondition_holds:
                if optional:
                    return _optional_sidecar_restart_or_absent(
                        parent_fd,
                        leaf,
                        selected,
                    )
                raise _failure(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    "private_sqlite_postcondition_failed",
                )
        if identity_out is not None:
            identity_out.append(os.fstat(file_fd))
        return True
    except PrivatePathError:
        raise
    except OSError as exc:
        raise _path_error_from_oserror(selected, exc) from None
    finally:
        if writable_fd >= 0:
            os.close(writable_fd)
        if file_fd >= 0:
            os.close(file_fd)
        os.close(parent_fd)


def _prepare_posix_artifact(
    selected: Path,
    *,
    writable: bool,
    create_if_missing: bool,
    optional: bool = False,
    enforce_private_mode: bool = True,
    open_artifact_fd: Callable[..., int] | None = None,
    postcondition_holds: Callable[..., bool] | None = None,
    identity_out: list[os.stat_result] | None = None,
) -> bool:
    attempts = _OPTIONAL_SIDECAR_REVALIDATION_ATTEMPTS if optional else 1
    for attempt in range(attempts):
        try:
            return _prepare_posix_artifact_generation(
                selected,
                writable=writable,
                create_if_missing=create_if_missing,
                optional=optional,
                enforce_private_mode=enforce_private_mode,
                open_artifact_fd=open_artifact_fd,
                postcondition_holds=postcondition_holds,
                identity_out=identity_out,
            )
        except _OptionalSQLiteGenerationChanged:
            if attempt + 1 == attempts:
                raise _failure(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    "optional_sqlite_generation_churn",
                ) from None
    raise AssertionError("unreachable optional SQLite revalidation state")


def _prepare_windows_artifact(
    selected: Path,
    *,
    writable: bool,
    create_if_missing: bool,
    optional: bool = False,
) -> bool:
    del writable
    try:
        file_stat = selected.lstat()
    except FileNotFoundError:
        if optional:
            return False
        if not create_if_missing:
            raise _failure(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                "missing_sqlite_artifact",
            ) from None
        file_fd = os.open(
            selected,
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
            _PRIVATE_FILE_MODE,
        )
        os.close(file_fd)
        return True

    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
        raise _failure(
            selected,
            PrivatePathStatus.LINK_OR_NON_REGULAR,
            "unsafe_sqlite_artifact",
        )
    return True


def _prepare_artifact(
    selected: Path,
    *,
    writable: bool,
    create_if_missing: bool,
    optional: bool = False,
    enforce_private_mode: bool = True,
    open_artifact_fd: Callable[..., int] | None = None,
    postcondition_holds: Callable[..., bool] | None = None,
    identity_out: list[os.stat_result] | None = None,
) -> bool:
    if private_paths._posix_guards_available():
        return _prepare_posix_artifact(
            selected,
            writable=writable,
            create_if_missing=create_if_missing,
            optional=optional,
            enforce_private_mode=enforce_private_mode,
            open_artifact_fd=open_artifact_fd,
            postcondition_holds=postcondition_holds,
            identity_out=identity_out,
        )
    if private_paths._WINDOWS_PLATFORM:
        return _prepare_windows_artifact(
            selected,
            writable=writable,
            create_if_missing=create_if_missing,
            optional=optional,
        )
    raise _failure(
        selected,
        PrivatePathStatus.OPERATION_FAILED,
        "required_posix_guards_unavailable",
    )


def prepare_batch(request: PrepareRequest) -> PrepareResult:
    """Validate the fixed four-artifact cohort using the shared raw checks."""
    selected = Path(request.path)
    private_paths.verify_trusted_directory(selected.parent, allow_shared_sticky=False)
    statuses = []
    main_identity = []
    for suffix in ("", *_SIDECAR_SUFFIXES):
        artifact = Path(f"{selected}{suffix}")
        try:
            before = artifact.lstat()
        except FileNotFoundError:
            before = None
        exists = _prepare_artifact(
            artifact,
            writable=request.writable,
            create_if_missing=request.create_if_missing and not suffix,
            optional=bool(suffix),
            enforce_private_mode=not request.preserve_source_mode,
            identity_out=main_identity if not suffix else None,
        )
        if not exists:
            statuses.append("absent")
        elif private_paths._WINDOWS_PLATFORM:
            statuses.append("unverified_platform")
        elif request.preserve_source_mode:
            statuses.append("preserved_source_mode")
        elif before is None:
            statuses.append("created_private")
        elif stat.S_IMODE(before.st_mode) != _PRIVATE_FILE_MODE:
            statuses.append("hardened_private")
        else:
            statuses.append("already_private")
    identity = FileIdentity.from_stat(
        main_identity[0] if main_identity else selected.lstat()
    )
    return PrepareResult(identity, tuple(statuses))
