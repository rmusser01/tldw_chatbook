"""Pin one admitted workspace root inside a single-threaded helper process."""

from __future__ import annotations

import ctypes
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterator

from tldw_chatbook.Utils.filesystem_identity import (
    DirectoryChain,
    DirectoryIdentity,
    DirectoryIdentityError,
    directory_identity_from_stat,
)


class WorkspaceRootPinError(RuntimeError):
    """Raised when an admitted root cannot be retained and verified safely."""


@dataclass(slots=True)
class PinnedWorkspaceRoot:
    """A verified helper-local current directory retained by an open handle."""

    canonical_locator: Path = field(repr=False)
    identity: DirectoryIdentity
    root_fd: int | None = field(default=None, repr=False)
    _previous_fd: int | None = field(default=None, repr=False)
    _windows_handle: int | None = field(default=None, repr=False)
    _previous_directory: str | None = field(default=None, repr=False)
    _closed: bool = field(default=False, repr=False)

    def relative_path(self, value: str) -> Path:
        """Return one lexical path relative to the retained helper root."""
        if type(value) is not str or not value or "\x00" in value:
            raise WorkspaceRootPinError("workspace operation requires a relative path")
        for lexical in (PurePosixPath(value), PureWindowsPath(value)):
            if lexical.drive or lexical.root or lexical.anchor:
                raise WorkspaceRootPinError(
                    "workspace operation requires a relative path"
                )
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise WorkspaceRootPinError("workspace operation requires a relative path")
        return relative

    def close(self) -> None:
        """Restore the prior helper directory and release retained handles."""
        if self._closed:
            return
        self._closed = True
        if os.name == "posix":
            self._close_posix()
        elif os.name == "nt":
            self._close_windows()

    def _close_posix(self) -> None:
        restore_error = False
        try:
            if self._previous_fd is not None:
                os.fchdir(self._previous_fd)
        except OSError:
            restore_error = True
        finally:
            for descriptor in (self.root_fd, self._previous_fd):
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        restore_error = True
            self.root_fd = None
            self._previous_fd = None
        if restore_error:
            raise WorkspaceRootPinError("workspace root pin cleanup failed")

    def _close_windows(self) -> None:
        restore_error = False
        try:
            if self._previous_directory is not None:
                _windows_set_current_directory(self._previous_directory)
        except WorkspaceRootPinError:
            restore_error = True
        finally:
            if self._windows_handle is not None:
                try:
                    _windows_close_handle(self._windows_handle)
                except WorkspaceRootPinError:
                    restore_error = True
            self._windows_handle = None
        if restore_error:
            raise WorkspaceRootPinError("workspace root pin cleanup failed")


@contextmanager
def pin_workspace_root(
    canonical_locator: Path,
    chain: DirectoryChain,
) -> Iterator[PinnedWorkspaceRoot]:
    """Open, identity-check, chdir to, and retain one admitted root."""
    locator = Path(canonical_locator)
    if (
        type(chain) is not DirectoryChain
        or not chain.identities
        or not locator.is_absolute()
        or locator != chain.canonical_root
    ):
        raise WorkspaceRootPinError("invalid admitted workspace root")

    if os.name == "posix":
        pinned = _pin_posix(locator, chain.identities[0])
    elif os.name == "nt":
        pinned = _pin_windows(locator, chain.identities[0])
    else:
        raise WorkspaceRootPinError("workspace root pinning is unsupported")

    try:
        yield pinned
    finally:
        pinned.close()


def _pin_posix(locator: Path, expected: DirectoryIdentity) -> PinnedWorkspaceRoot:
    required = ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC")
    if not hasattr(os, "fchdir") or any(not hasattr(os, name) for name in required):
        raise WorkspaceRootPinError("workspace root pinning is unsupported")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    root_fd: int | None = None
    previous_fd: int | None = None
    try:
        previous_fd = os.open(".", flags)
        root_fd = os.open(locator, flags)
        if _posix_identity(root_fd) != expected:
            raise WorkspaceRootPinError("workspace root identity mismatch")
        os.fchdir(root_fd)
        if directory_identity_from_stat(os.stat(".", follow_symlinks=False)) != expected:
            raise WorkspaceRootPinError("workspace root identity mismatch")
        return PinnedWorkspaceRoot(
            canonical_locator=locator,
            identity=expected,
            root_fd=root_fd,
            _previous_fd=previous_fd,
        )
    except WorkspaceRootPinError:
        _restore_and_close_posix(previous_fd, root_fd)
        raise
    except (DirectoryIdentityError, OSError, ValueError) as error:
        _restore_and_close_posix(previous_fd, root_fd)
        raise WorkspaceRootPinError("workspace root pinning failed") from error


def _posix_identity(descriptor: int) -> DirectoryIdentity:
    try:
        return directory_identity_from_stat(os.fstat(descriptor))
    except (DirectoryIdentityError, OSError) as error:
        raise WorkspaceRootPinError("workspace root metadata unavailable") from error


def _restore_and_close_posix(previous_fd: int | None, root_fd: int | None) -> None:
    if previous_fd is not None:
        try:
            os.fchdir(previous_fd)
        except OSError:
            pass
    for descriptor in (root_fd, previous_fd):
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _pin_windows(locator: Path, expected: DirectoryIdentity) -> PinnedWorkspaceRoot:
    if expected.reparse:
        raise WorkspaceRootPinError("unsafe workspace root metadata")
    handle: int | None = None
    previous_directory: str | None = None
    try:
        previous_directory = os.getcwd()
        handle, identity, reparse = _windows_open_directory(locator)
        if reparse:
            raise WorkspaceRootPinError("unsafe workspace root metadata")
        if (identity.device, identity.inode) != (expected.device, expected.inode):
            raise WorkspaceRootPinError("workspace root identity mismatch")
        _windows_set_current_directory(str(locator))
        verification, verified_identity, verified_reparse = _windows_open_directory(Path("."))
        try:
            if verified_reparse or (
                verified_identity.device,
                verified_identity.inode,
            ) != (expected.device, expected.inode):
                raise WorkspaceRootPinError("workspace root identity mismatch")
        finally:
            _windows_close_handle(verification)
        return PinnedWorkspaceRoot(
            canonical_locator=locator,
            identity=expected,
            _windows_handle=handle,
            _previous_directory=previous_directory,
        )
    except WorkspaceRootPinError:
        if previous_directory is not None:
            try:
                _windows_set_current_directory(previous_directory)
            except WorkspaceRootPinError:
                pass
        if handle is not None:
            try:
                _windows_close_handle(handle)
            except WorkspaceRootPinError:
                pass
        raise
    except (OSError, ValueError) as error:
        if handle is not None:
            try:
                _windows_close_handle(handle)
            except WorkspaceRootPinError:
                pass
        raise WorkspaceRootPinError("workspace root pinning failed") from error


def _windows_open_directory(path: Path) -> tuple[int, DirectoryIdentity, bool]:
    if os.name != "nt":
        raise WorkspaceRootPinError("Windows root pinning is unavailable")
    from ctypes import wintypes

    class ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.GetFileInformationByHandle.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ByHandleFileInformation),
    ]
    kernel32.GetFileInformationByHandle.restype = wintypes.BOOL

    file_read_attributes = 0x0080
    share_all = 0x00000001 | 0x00000002 | 0x00000004
    open_existing = 3
    backup_semantics = 0x02000000
    open_reparse_point = 0x00200000
    raw_handle = kernel32.CreateFileW(
        str(path),
        file_read_attributes,
        share_all,
        None,
        open_existing,
        backup_semantics | open_reparse_point,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    handle = int(ctypes.cast(raw_handle, ctypes.c_void_p).value or 0)
    if not handle or handle == invalid_handle:
        raise WorkspaceRootPinError("workspace root open failed")
    information = ByHandleFileInformation()
    if not kernel32.GetFileInformationByHandle(handle, ctypes.byref(information)):
        _windows_close_handle(handle)
        raise WorkspaceRootPinError("workspace root metadata unavailable")
    inode = (int(information.nFileIndexHigh) << 32) | int(information.nFileIndexLow)
    reparse = bool(int(information.dwFileAttributes) & 0x00000400)
    return (
        handle,
        DirectoryIdentity(
            device=int(information.dwVolumeSerialNumber),
            inode=inode,
            mode=0,
            reparse=reparse,
        ),
        reparse,
    )


def _windows_set_current_directory(path: str) -> None:
    if os.name != "nt":
        raise WorkspaceRootPinError("Windows root pinning is unavailable")
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.SetCurrentDirectoryW.argtypes = [wintypes.LPCWSTR]
    kernel32.SetCurrentDirectoryW.restype = wintypes.BOOL
    if not kernel32.SetCurrentDirectoryW(path):
        raise WorkspaceRootPinError("workspace root current-directory pin failed")


def _windows_close_handle(handle: int) -> None:
    if os.name != "nt":
        raise WorkspaceRootPinError("Windows root pinning is unavailable")
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    if not kernel32.CloseHandle(handle):
        raise WorkspaceRootPinError("workspace root handle cleanup failed")


__all__ = ["PinnedWorkspaceRoot", "WorkspaceRootPinError", "pin_workspace_root"]
