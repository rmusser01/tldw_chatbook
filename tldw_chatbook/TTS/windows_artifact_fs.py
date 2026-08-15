"""Native Windows filesystem boundary for owned audio.cpp artifacts.

The module remains import-safe on non-Windows hosts. Concrete native
capabilities are added only behind the explicit Windows implementation.
"""

from __future__ import annotations

import ctypes
import os
import platform
import struct
import sys
from dataclasses import dataclass, field
from pathlib import PureWindowsPath
from typing import Any, Literal, Protocol

from ctypes import wintypes


WindowsProcessArchitecture = Literal["x86", "x86_64"]
WindowsArtifactKind = Literal["directory", "file"]
WindowsArtifactPrivacyPosture = Literal[
    "unverified",
    "windows_account_protected",
]
WindowsArtifactErrorCode = Literal[
    "unsupported",
    "unavailable",
    "changed",
    "privacy_unavailable",
    "busy",
    "cleanup_failed",
]

_PATH_ERROR = "Windows artifact path is unavailable"
_DEVICE_PREFIXES = ("\\\\?\\", "\\\\.\\", "\\device\\", "\\??\\")
_RESERVED_NAMES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)
_INVALID_COMPONENT_CHARACTERS = frozenset('<>:"/\\|?*')
_X86_MACHINES = frozenset(
    {"x86", "i386", "i486", "i586", "i686", "amd64", "x86_64", "x64"}
)
_X64_MACHINES = frozenset({"amd64", "x86_64", "x64"})
_ERROR_MESSAGES: dict[WindowsArtifactErrorCode, str] = {
    "unsupported": "Windows audio.cpp artifacts are unsupported",
    "unavailable": "Windows artifact storage is unavailable",
    "changed": "Windows artifact identity changed",
    "privacy_unavailable": "Windows artifact privacy is unavailable",
    "busy": "Windows artifact storage is busy",
    "cleanup_failed": "Windows artifact cleanup did not complete",
}
_CONTROL_CLEANUP_OWNER = "_windows_artifact_cleanup_owner"

_FILE_ATTRIBUTE_DIRECTORY = 0x00000010
_FILE_ATTRIBUTE_READONLY = 0x00000001
_FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_FILE_LIST_DIRECTORY = 0x00000001
_FILE_READ_ATTRIBUTES = 0x00000080
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_FILE_WRITE_ATTRIBUTES = 0x00000100
_GENERIC_READ = 0x80000000
_GENERIC_WRITE = 0x40000000
_DELETE = 0x00010000
_READ_CONTROL = 0x00020000
_WRITE_DAC = 0x00040000
_OPEN_EXISTING = 3
_CREATE_NEW = 1
_HANDLE_FLAG_INHERIT = 0x00000001
_FILE_BEGIN = 0
_LOCKFILE_FAIL_IMMEDIATELY = 0x00000001
_LOCKFILE_EXCLUSIVE_LOCK = 0x00000002
_FILE_BASIC_INFO_CLASS = 0
_FILE_DISPOSITION_INFO_CLASS = 4
_FILE_ATTRIBUTE_TAG_INFO_CLASS = 9
_FILE_ID_INFO_CLASS = 18
_SE_FILE_OBJECT = 1
_DACL_SECURITY_INFORMATION = 0x00000004
_PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000
_SE_DACL_PROTECTED = 0x1000
_TOKEN_QUERY = 0x0008
_TOKEN_USER_CLASS = 1
_SDDL_REVISION_1 = 1
_ACCESS_ALLOWED_ACE_TYPE = 0
_OBJECT_INHERIT_ACE = 0x01
_CONTAINER_INHERIT_ACE = 0x02
_FILE_ALL_ACCESS = 0x001F01FF
_SYSTEM_SID = "S-1-5-18"
_ADMINISTRATORS_SID = "S-1-5-32-544"


class _FileIdInfo(ctypes.Structure):
    _fields_ = [
        ("VolumeSerialNumber", ctypes.c_ulonglong),
        ("FileId", ctypes.c_ubyte * 16),
    ]


class _FileAttributeTagInfo(ctypes.Structure):
    _fields_ = [
        ("FileAttributes", wintypes.DWORD),
        ("ReparseTag", wintypes.DWORD),
    ]


class _FileBasicInfo(ctypes.Structure):
    _fields_ = [
        ("CreationTime", ctypes.c_longlong),
        ("LastAccessTime", ctypes.c_longlong),
        ("LastWriteTime", ctypes.c_longlong),
        ("ChangeTime", ctypes.c_longlong),
        ("FileAttributes", wintypes.DWORD),
    ]


class _FileDispositionInfo(ctypes.Structure):
    _fields_ = [("DeleteFile", wintypes.BOOLEAN)]


class _Overlapped(ctypes.Structure):
    _fields_ = [
        ("Internal", ctypes.c_size_t),
        ("InternalHigh", ctypes.c_size_t),
        ("Offset", wintypes.DWORD),
        ("OffsetHigh", wintypes.DWORD),
        ("hEvent", wintypes.HANDLE),
    ]


class _SecurityAttributes(ctypes.Structure):
    _fields_ = [
        ("nLength", wintypes.DWORD),
        ("lpSecurityDescriptor", wintypes.LPVOID),
        ("bInheritHandle", wintypes.BOOL),
    ]


class _SidAndAttributes(ctypes.Structure):
    _fields_ = [("Sid", wintypes.LPVOID), ("Attributes", wintypes.DWORD)]


class _TokenUser(ctypes.Structure):
    _fields_ = [("User", _SidAndAttributes)]


class _Acl(ctypes.Structure):
    _fields_ = [
        ("AclRevision", ctypes.c_ubyte),
        ("Sbz1", ctypes.c_ubyte),
        ("AclSize", wintypes.WORD),
        ("AceCount", wintypes.WORD),
        ("Sbz2", wintypes.WORD),
    ]


class _AceHeader(ctypes.Structure):
    _fields_ = [
        ("AceType", ctypes.c_ubyte),
        ("AceFlags", ctypes.c_ubyte),
        ("AceSize", wintypes.WORD),
    ]


class _AccessAllowedAce(ctypes.Structure):
    _fields_ = [
        ("Header", _AceHeader),
        ("Mask", wintypes.DWORD),
        ("SidStart", wintypes.DWORD),
    ]


@dataclass(frozen=True, slots=True)
class WindowsFileIdentity:
    """Stable native object identity with no public path or handle value."""

    volume_serial_number: int
    file_id: bytes = field(repr=False)
    kind: WindowsArtifactKind
    reparse_tag: int

    def __post_init__(self) -> None:
        if (
            type(self.volume_serial_number) is not int
            or type(self.file_id) is not bytes
            or len(self.file_id) != 16
            or self.kind not in ("directory", "file")
            or type(self.reparse_tag) is not int
            or self.reparse_tag < 0
        ):
            raise ValueError("Windows file identity is invalid")


class _WindowsKernel(Protocol):
    """Small injectable native-call boundary used by the artifact owner."""

    def open_handle(
        self,
        path: str,
        *,
        kind: WindowsArtifactKind,
        create_new: bool,
        writable: bool,
    ) -> int: ...

    def set_inheritable(self, handle: int, inheritable: bool) -> None: ...

    def identity(self, handle: int) -> WindowsFileIdentity: ...

    def current_user_sid(self) -> bytes: ...

    def set_private_acl(
        self,
        handle: int,
        user_sid: bytes,
        directory: bool,
    ) -> None: ...

    def verify_private_acl(
        self,
        handle: int,
        user_sid: bytes,
        directory: bool,
    ) -> bool: ...

    def write_all(self, handle: int, data: bytes) -> None: ...

    def read(self, handle: int, count: int, offset: int) -> bytes: ...

    def flush(self, handle: int) -> None: ...

    def set_read_only(self, handle: int) -> None: ...

    def lock_exclusive_nonblocking(self, handle: int) -> None: ...

    def unlock(self, handle: int) -> None: ...

    def delete_handle(self, handle: int) -> None: ...

    def close_handle(self, handle: int) -> None: ...


class WindowsArtifactError(RuntimeError):
    """Stable path-independent failure with optional cleanup ownership."""

    __slots__ = ("_cleanup_owner", "code")

    def __init__(
        self,
        code: WindowsArtifactErrorCode,
        *,
        cleanup_owner: WindowsPinnedHandle | None = None,
    ) -> None:
        self.code = code
        self._cleanup_owner = cleanup_owner
        super().__init__(_ERROR_MESSAGES[code])

    def take_cleanup_owner(self) -> WindowsPinnedHandle | None:
        """Transfer the exact retained handle once."""

        owner = self._cleanup_owner
        self._cleanup_owner = None
        return owner


def _sanitize_control(error: BaseException) -> BaseException:
    """Bound one synchronous control signal without changing its family."""

    if isinstance(error, BaseExceptionGroup):
        bounded = BaseExceptionGroup(
            "Windows artifact operation interrupted",
            tuple(_sanitize_control(child) for child in error.exceptions),
        )
        bounded.__suppress_context__ = True
        return bounded
    if isinstance(error, SystemExit):
        code = error.code
        error.code = code if code is None or type(code) is int else 1
        error.args = () if error.code is None else (error.code,)
    else:
        error.args = ()
    error.__cause__ = None
    error.__context__ = None
    error.__traceback__ = None
    error.__suppress_context__ = True
    return error


def _attach_control_cleanup_owner(
    error: BaseException,
    owner: WindowsPinnedHandle,
) -> None:
    setattr(error, _CONTROL_CLEANUP_OWNER, owner)


def take_windows_artifact_cleanup_owner(
    error: BaseException,
) -> WindowsPinnedHandle | None:
    """Take one exact cleanup owner from a bounded public failure."""

    if isinstance(error, WindowsArtifactError):
        return WindowsArtifactError.take_cleanup_owner(error)
    owner = getattr(error, _CONTROL_CLEANUP_OWNER, None)
    if not isinstance(owner, WindowsPinnedHandle):
        return None
    setattr(error, _CONTROL_CLEANUP_OWNER, None)
    return owner


def _raise_control(error: BaseException) -> None:
    """Raise a sanitized control without retaining an active private context."""

    try:
        raise error from None
    except BaseException as reraised:
        reraised.__cause__ = None
        reraised.__context__ = None
        reraised.__suppress_context__ = True
        raise


class WindowsPinnedHandle:
    """Opaque owner of one non-inheritable native file or directory handle."""

    __slots__ = (
        "_cleanup_delete_pending",
        "_cleanup_only",
        "_closed",
        "_deleted",
        "_handle",
        "_identity",
        "_kernel",
        "_private",
        "_user_sid",
    )

    def __init__(
        self,
        kernel: _WindowsKernel,
        handle: int,
        identity: WindowsFileIdentity | None,
        *,
        private: bool,
        user_sid: bytes,
    ) -> None:
        self._kernel = kernel
        self._handle = handle
        self._identity = identity
        self._private = private
        self._user_sid = user_sid
        self._closed = False
        self._deleted = False
        self._cleanup_only = identity is None
        self._cleanup_delete_pending = False

    @property
    def identity(self) -> WindowsFileIdentity:
        """Return the immutable identity captured at open."""

        if self._identity is None:
            raise WindowsArtifactError("unavailable")
        return self._identity

    @property
    def privacy_posture(self) -> WindowsArtifactPrivacyPosture:
        """Return only the posture actually verified on this open object."""

        return "windows_account_protected" if self._private else "unverified"

    def read(self, count: int, *, offset: int = 0) -> bytes:
        """Read bounded bytes through the exact retained handle."""

        if self._closed or self._cleanup_only or type(count) is not int or count < 0:
            raise WindowsArtifactError("unavailable")
        try:
            return self._kernel.read(self._handle, count, offset)
        except Exception:
            pass
        raise WindowsArtifactError("unavailable") from None

    def lock_exclusive_nonblocking(self) -> None:
        """Acquire one exact nonblocking owner lock."""

        if self._closed or self._cleanup_only:
            raise WindowsArtifactError("unavailable")
        busy = False
        failed = False
        try:
            self._kernel.lock_exclusive_nonblocking(self._handle)
        except BlockingIOError:
            busy = True
        except Exception:
            failed = True
        if busy:
            raise WindowsArtifactError("busy") from None
        if failed:
            raise WindowsArtifactError("unavailable") from None

    def unlock(self) -> None:
        """Release the exact owner lock."""

        if self._closed:
            return
        failed = False
        try:
            self._kernel.unlock(self._handle)
        except Exception:
            failed = True
        if failed:
            raise WindowsArtifactError("cleanup_failed") from None

    def verify_private_acl(self) -> bool:
        """Re-read the approved protected DACL on the exact handle."""

        if self._closed or not self._private:
            return False
        try:
            return self._kernel.verify_private_acl(
                self._handle,
                self._user_sid,
                self._identity is not None and self._identity.kind == "directory",
            )
        except Exception:
            return False

    def delete_exact(self) -> None:
        """Mark only the unchanged, private, exact object for deletion."""

        if self._closed or self._deleted or self._cleanup_only:
            return
        changed = False
        privacy_failed = False
        delete_failed = False
        try:
            changed = self._kernel.identity(self._handle) != self._identity
            if not changed:
                privacy_failed = not self.verify_private_acl()
            if not changed and not privacy_failed:
                self._kernel.delete_handle(self._handle)
        except Exception:
            delete_failed = True
        if changed:
            raise WindowsArtifactError("changed") from None
        if privacy_failed:
            raise WindowsArtifactError("privacy_unavailable") from None
        if delete_failed:
            raise WindowsArtifactError("cleanup_failed", cleanup_owner=self) from None
        self._deleted = True

    def close(self) -> None:
        """Close once, retaining this exact owner after an ordinary failure."""

        if self._closed:
            return
        if self._cleanup_delete_pending and not self._deleted:
            try:
                self._kernel.delete_handle(self._handle)
            except Exception:
                raise WindowsArtifactError(
                    "cleanup_failed", cleanup_owner=self
                ) from None
            self._deleted = True
        failed = False
        try:
            self._kernel.close_handle(self._handle)
        except BaseException as error:
            if isinstance(error, Exception):
                failed = True
            else:
                control = _sanitize_control(error)
                _attach_control_cleanup_owner(control, self)
                _raise_control(control)
        if failed:
            raise WindowsArtifactError("cleanup_failed", cleanup_owner=self) from None
        self._closed = True

    def __repr__(self) -> str:
        return "WindowsPinnedHandle(<opaque>)"


class WindowsArtifactFilesystem(Protocol):
    """Minimal injectable capability consumed by audio.cpp owners."""

    def pin_directory_no_reparse(
        self, path: str | os.PathLike[str]
    ) -> WindowsPinnedHandle: ...

    def open_file_no_reparse(
        self,
        path: str | os.PathLike[str],
        *,
        writable: bool = False,
    ) -> WindowsPinnedHandle: ...

    def create_private_directory(
        self, path: str | os.PathLike[str]
    ) -> WindowsPinnedHandle: ...

    def protect_private_directory(
        self, path: str | os.PathLike[str]
    ) -> WindowsPinnedHandle: ...

    def create_private_file(
        self,
        path: str | os.PathLike[str],
        data: bytes,
        *,
        read_only: bool = False,
    ) -> WindowsPinnedHandle: ...


class NativeWindowsArtifactFilesystem:
    """Handle-pinned Windows artifact operations over one native kernel."""

    def __init__(self, *, kernel: _WindowsKernel | None = None) -> None:
        self._kernel = _native_windows_kernel() if kernel is None else kernel
        try:
            self._user_sid = self._kernel.current_user_sid()
        except Exception:
            raise WindowsArtifactError("privacy_unavailable") from None

    def pin_directory_no_reparse(
        self, path: str | os.PathLike[str]
    ) -> WindowsPinnedHandle:
        """Pin one existing ordinary directory without following reparse data."""

        return self._open(path, kind="directory", create_new=False, writable=False)

    def open_file_no_reparse(
        self,
        path: str | os.PathLike[str],
        *,
        writable: bool = False,
    ) -> WindowsPinnedHandle:
        """Open one existing regular file without following reparse data."""

        return self._open(
            path,
            kind="file",
            create_new=False,
            writable=writable,
        )

    def create_private_directory(
        self, path: str | os.PathLike[str]
    ) -> WindowsPinnedHandle:
        """Create and verify one protected private directory."""

        owner = self._open(
            path,
            kind="directory",
            create_new=True,
            writable=True,
        )
        return self._make_private(owner, directory=True)

    def protect_private_directory(
        self, path: str | os.PathLike[str]
    ) -> WindowsPinnedHandle:
        """Protect and verify one existing ordinary directory."""

        owner = self._open(
            path,
            kind="directory",
            create_new=False,
            writable=True,
        )
        return self._make_private(owner, directory=True)

    def create_private_file(
        self,
        path: str | os.PathLike[str],
        data: bytes,
        *,
        read_only: bool = False,
    ) -> WindowsPinnedHandle:
        """Create, flush, protect, and verify one private file."""

        if type(data) is not bytes:
            raise WindowsArtifactError("unavailable")
        owner = self._open(path, kind="file", create_new=True, writable=True)
        failed = False
        control: BaseException | None = None
        try:
            self._kernel.write_all(owner._handle, data)
            self._kernel.flush(owner._handle)
        except BaseException as error:
            if isinstance(error, Exception):
                failed = True
            else:
                control = _sanitize_control(error)
        if control is not None:
            try:
                self._retire_failed_creation(owner)
            except WindowsArtifactError as cleanup_error:
                cleanup = cleanup_error.take_cleanup_owner()
                if cleanup is not None:
                    _attach_control_cleanup_owner(control, cleanup)
            _raise_control(control)
        if failed:
            self._retire_failed_creation(owner)
            raise WindowsArtifactError("unavailable") from None
        owner = self._make_private(owner, directory=False)
        if read_only:
            read_only_failed = False
            try:
                self._kernel.set_read_only(owner._handle)
            except Exception:
                read_only_failed = True
            if read_only_failed:
                self._retire_failed_creation(owner)
                raise WindowsArtifactError("privacy_unavailable") from None
        return owner

    def _open(
        self,
        path: str | os.PathLike[str],
        *,
        kind: WindowsArtifactKind,
        create_new: bool,
        writable: bool,
    ) -> WindowsPinnedHandle:
        try:
            normalized = normalize_windows_artifact_path(path)
        except ValueError:
            raise WindowsArtifactError("unavailable") from None
        handle: int | None = None
        identity: WindowsFileIdentity | None = None
        failed = False
        try:
            handle = self._kernel.open_handle(
                normalized,
                kind=kind,
                create_new=create_new,
                writable=writable,
            )
            self._kernel.set_inheritable(handle, False)
            identity = self._kernel.identity(handle)
        except Exception:
            failed = True
        if failed or handle is None or identity is None:
            if handle is not None:
                cleanup = WindowsPinnedHandle(
                    self._kernel,
                    handle,
                    identity,
                    private=False,
                    user_sid=self._user_sid,
                )
                cleanup._cleanup_only = True
                try:
                    cleanup.close()
                except WindowsArtifactError:
                    raise WindowsArtifactError(
                        "cleanup_failed", cleanup_owner=cleanup
                    ) from None
            raise WindowsArtifactError("unavailable") from None
        owner = WindowsPinnedHandle(
            self._kernel,
            handle,
            identity,
            private=False,
            user_sid=self._user_sid,
        )
        if identity.kind != kind or identity.reparse_tag:
            owner._cleanup_only = True
            try:
                owner.close()
            except WindowsArtifactError:
                raise WindowsArtifactError(
                    "cleanup_failed", cleanup_owner=owner
                ) from None
            raise WindowsArtifactError("changed") from None
        return owner

    def _make_private(
        self,
        owner: WindowsPinnedHandle,
        *,
        directory: bool,
    ) -> WindowsPinnedHandle:
        failed = False
        try:
            self._kernel.set_private_acl(owner._handle, self._user_sid, directory)
            failed = not self._kernel.verify_private_acl(
                owner._handle,
                self._user_sid,
                directory,
            )
        except Exception:
            failed = True
        if failed:
            self._retire_failed_creation(owner)
            raise WindowsArtifactError("privacy_unavailable") from None
        owner._private = True
        return owner

    @staticmethod
    def _retire_failed_creation(owner: WindowsPinnedHandle) -> None:
        owner._cleanup_only = True
        owner._cleanup_delete_pending = True
        owner.close()


class UnavailableWindowsArtifactFilesystem:
    """Import-safe placeholder for hosts outside the supported Windows tuple."""

    @staticmethod
    def _unsupported() -> WindowsPinnedHandle:
        raise WindowsArtifactError("unsupported")

    def pin_directory_no_reparse(
        self, path: str | os.PathLike[str]
    ) -> WindowsPinnedHandle:
        del path
        return self._unsupported()

    def open_file_no_reparse(
        self,
        path: str | os.PathLike[str],
        *,
        writable: bool = False,
    ) -> WindowsPinnedHandle:
        del path, writable
        return self._unsupported()

    def create_private_directory(
        self, path: str | os.PathLike[str]
    ) -> WindowsPinnedHandle:
        del path
        return self._unsupported()

    def protect_private_directory(
        self, path: str | os.PathLike[str]
    ) -> WindowsPinnedHandle:
        del path
        return self._unsupported()

    def create_private_file(
        self,
        path: str | os.PathLike[str],
        data: bytes,
        *,
        read_only: bool = False,
    ) -> WindowsPinnedHandle:
        del path, data, read_only
        return self._unsupported()


class _CtypesWindowsKernel:
    """Private, explicitly signed Win32 calls used by the public capability."""

    def __init__(self) -> None:
        if os.name != "nt" or not hasattr(ctypes, "WinDLL"):
            raise WindowsArtifactError("unsupported")
        win_dll = getattr(ctypes, "WinDLL")
        self._kernel32 = win_dll("kernel32", use_last_error=True)
        self._advapi32 = win_dll("advapi32", use_last_error=True)
        self._configure_signatures()

    @staticmethod
    def _handle(value: int) -> wintypes.HANDLE:
        return wintypes.HANDLE(value)

    @staticmethod
    def _native_failed() -> OSError:
        return OSError("native Windows artifact operation failed")

    @staticmethod
    def _valid_handle(handle: Any) -> int:
        value = ctypes.cast(handle, ctypes.c_void_p).value
        if value in {None, ctypes.c_void_p(-1).value}:
            raise _CtypesWindowsKernel._native_failed()
        return int(value)

    def _configure_signatures(self) -> None:
        kernel32 = self._kernel32
        advapi32 = self._advapi32

        kernel32.CreateFileW.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(_SecurityAttributes),
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.CreateDirectoryW.argtypes = (
            wintypes.LPCWSTR,
            ctypes.POINTER(_SecurityAttributes),
        )
        kernel32.CreateDirectoryW.restype = wintypes.BOOL
        kernel32.RemoveDirectoryW.argtypes = (wintypes.LPCWSTR,)
        kernel32.RemoveDirectoryW.restype = wintypes.BOOL
        kernel32.SetHandleInformation.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
        )
        kernel32.SetHandleInformation.restype = wintypes.BOOL
        kernel32.GetFileInformationByHandleEx.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        )
        kernel32.GetFileInformationByHandleEx.restype = wintypes.BOOL
        kernel32.SetFileInformationByHandle.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        )
        kernel32.SetFileInformationByHandle.restype = wintypes.BOOL
        kernel32.SetFilePointerEx.argtypes = (
            wintypes.HANDLE,
            ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_longlong),
            wintypes.DWORD,
        )
        kernel32.SetFilePointerEx.restype = wintypes.BOOL
        kernel32.ReadFile.argtypes = (
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(_Overlapped),
        )
        kernel32.ReadFile.restype = wintypes.BOOL
        kernel32.WriteFile.argtypes = kernel32.ReadFile.argtypes
        kernel32.WriteFile.restype = wintypes.BOOL
        kernel32.FlushFileBuffers.argtypes = (wintypes.HANDLE,)
        kernel32.FlushFileBuffers.restype = wintypes.BOOL
        kernel32.LockFileEx.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(_Overlapped),
        )
        kernel32.LockFileEx.restype = wintypes.BOOL
        kernel32.UnlockFileEx.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(_Overlapped),
        )
        kernel32.UnlockFileEx.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.GetCurrentProcess.argtypes = ()
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.LocalFree.argtypes = (wintypes.HLOCAL,)
        kernel32.LocalFree.restype = wintypes.HLOCAL

        advapi32.OpenProcessToken.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.HANDLE),
        )
        advapi32.OpenProcessToken.restype = wintypes.BOOL
        advapi32.GetTokenInformation.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        )
        advapi32.GetTokenInformation.restype = wintypes.BOOL
        advapi32.GetLengthSid.argtypes = (wintypes.LPVOID,)
        advapi32.GetLengthSid.restype = wintypes.DWORD
        advapi32.IsValidSid.argtypes = (wintypes.LPVOID,)
        advapi32.IsValidSid.restype = wintypes.BOOL
        advapi32.ConvertSidToStringSidW.argtypes = (
            wintypes.LPVOID,
            ctypes.POINTER(wintypes.LPWSTR),
        )
        advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
        advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.DWORD),
        )
        advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = (
            wintypes.BOOL
        )
        advapi32.GetSecurityDescriptorDacl.argtypes = (
            wintypes.LPVOID,
            ctypes.POINTER(wintypes.BOOL),
            ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.BOOL),
        )
        advapi32.GetSecurityDescriptorDacl.restype = wintypes.BOOL
        advapi32.GetSecurityDescriptorControl.argtypes = (
            wintypes.LPVOID,
            ctypes.POINTER(wintypes.WORD),
            ctypes.POINTER(wintypes.DWORD),
        )
        advapi32.GetSecurityDescriptorControl.restype = wintypes.BOOL
        advapi32.SetSecurityInfo.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.LPVOID,
            wintypes.LPVOID,
            wintypes.LPVOID,
        )
        advapi32.SetSecurityInfo.restype = wintypes.DWORD
        advapi32.GetSecurityInfo.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.LPVOID),
        )
        advapi32.GetSecurityInfo.restype = wintypes.DWORD
        advapi32.GetAce.argtypes = (
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.LPVOID),
        )
        advapi32.GetAce.restype = wintypes.BOOL

    def _sid_string(self, sid: bytes) -> str:
        buffer = ctypes.create_string_buffer(sid)
        pointer = ctypes.cast(buffer, wintypes.LPVOID)
        if not self._advapi32.IsValidSid(pointer):
            raise self._native_failed()
        rendered = wintypes.LPWSTR()
        if not self._advapi32.ConvertSidToStringSidW(pointer, ctypes.byref(rendered)):
            raise self._native_failed()
        try:
            return str(rendered.value)
        finally:
            self._kernel32.LocalFree(rendered)

    def _private_descriptor(self, user_sid: bytes, directory: bool) -> wintypes.LPVOID:
        sid = self._sid_string(user_sid)
        flags = "OICI" if directory else ""
        sddl = (
            f"D:P(A;{flags};FA;;;{sid})"
            f"(A;{flags};FA;;;{_SYSTEM_SID})"
            f"(A;{flags};FA;;;{_ADMINISTRATORS_SID})"
        )
        descriptor = wintypes.LPVOID()
        if not self._advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl,
            _SDDL_REVISION_1,
            ctypes.byref(descriptor),
            None,
        ):
            raise self._native_failed()
        return descriptor

    def _security_attributes(
        self, directory: bool
    ) -> tuple[_SecurityAttributes, wintypes.LPVOID]:
        descriptor = self._private_descriptor(self.current_user_sid(), directory)
        attributes = _SecurityAttributes(
            ctypes.sizeof(_SecurityAttributes), descriptor, False
        )
        return attributes, descriptor

    def open_handle(
        self,
        path: str,
        *,
        kind: WindowsArtifactKind,
        create_new: bool,
        writable: bool,
    ) -> int:
        descriptor: wintypes.LPVOID | None = None
        attributes: _SecurityAttributes | None = None
        created_directory = False
        try:
            if create_new:
                attributes, descriptor = self._security_attributes(kind == "directory")
            if kind == "directory" and create_new:
                if attributes is None:
                    raise self._native_failed()
                if not self._kernel32.CreateDirectoryW(path, ctypes.byref(attributes)):
                    raise self._native_failed()
                created_directory = True

            access = _FILE_READ_ATTRIBUTES | _READ_CONTROL
            if kind == "directory":
                access |= _FILE_LIST_DIRECTORY
            else:
                access |= _GENERIC_READ
            if writable:
                access |= _GENERIC_WRITE | _FILE_WRITE_ATTRIBUTES
            if create_new or (kind == "directory" and writable):
                access |= _DELETE | _WRITE_DAC
            disposition = (
                _CREATE_NEW if create_new and kind == "file" else _OPEN_EXISTING
            )
            flags = _FILE_FLAG_OPEN_REPARSE_POINT
            if kind == "directory":
                flags |= _FILE_FLAG_BACKUP_SEMANTICS
            handle = self._kernel32.CreateFileW(
                path,
                access,
                _FILE_SHARE_READ | _FILE_SHARE_WRITE,
                ctypes.byref(attributes) if attributes is not None else None,
                disposition,
                flags,
                None,
            )
            return self._valid_handle(handle)
        except BaseException:
            if created_directory:
                self._kernel32.RemoveDirectoryW(path)
            raise
        finally:
            if descriptor is not None:
                self._kernel32.LocalFree(descriptor)

    def set_inheritable(self, handle: int, inheritable: bool) -> None:
        value = _HANDLE_FLAG_INHERIT if inheritable else 0
        if not self._kernel32.SetHandleInformation(
            self._handle(handle), _HANDLE_FLAG_INHERIT, value
        ):
            raise self._native_failed()

    def identity(self, handle: int) -> WindowsFileIdentity:
        file_id = _FileIdInfo()
        attributes = _FileAttributeTagInfo()
        native = self._handle(handle)
        if not self._kernel32.GetFileInformationByHandleEx(
            native,
            _FILE_ID_INFO_CLASS,
            ctypes.byref(file_id),
            ctypes.sizeof(file_id),
        ) or not self._kernel32.GetFileInformationByHandleEx(
            native,
            _FILE_ATTRIBUTE_TAG_INFO_CLASS,
            ctypes.byref(attributes),
            ctypes.sizeof(attributes),
        ):
            raise self._native_failed()
        kind: WindowsArtifactKind = (
            "directory"
            if attributes.FileAttributes & _FILE_ATTRIBUTE_DIRECTORY
            else "file"
        )
        tag = (
            int(attributes.ReparseTag)
            if attributes.FileAttributes & _FILE_ATTRIBUTE_REPARSE_POINT
            else 0
        )
        return WindowsFileIdentity(
            volume_serial_number=int(file_id.VolumeSerialNumber),
            file_id=bytes(file_id.FileId),
            kind=kind,
            reparse_tag=tag,
        )

    def current_user_sid(self) -> bytes:
        token = wintypes.HANDLE()
        if not self._advapi32.OpenProcessToken(
            self._kernel32.GetCurrentProcess(), _TOKEN_QUERY, ctypes.byref(token)
        ):
            raise self._native_failed()
        try:
            required = wintypes.DWORD()
            self._advapi32.GetTokenInformation(
                token, _TOKEN_USER_CLASS, None, 0, ctypes.byref(required)
            )
            if not required.value:
                raise self._native_failed()
            buffer = ctypes.create_string_buffer(required.value)
            if not self._advapi32.GetTokenInformation(
                token,
                _TOKEN_USER_CLASS,
                buffer,
                required,
                ctypes.byref(required),
            ):
                raise self._native_failed()
            user = ctypes.cast(buffer, ctypes.POINTER(_TokenUser)).contents
            length = self._advapi32.GetLengthSid(user.User.Sid)
            if not length:
                raise self._native_failed()
            return ctypes.string_at(user.User.Sid, length)
        finally:
            self._kernel32.CloseHandle(token)

    def set_private_acl(self, handle: int, user_sid: bytes, directory: bool) -> None:
        descriptor = self._private_descriptor(user_sid, directory)
        try:
            present = wintypes.BOOL()
            defaulted = wintypes.BOOL()
            dacl = wintypes.LPVOID()
            if (
                not self._advapi32.GetSecurityDescriptorDacl(
                    descriptor,
                    ctypes.byref(present),
                    ctypes.byref(dacl),
                    ctypes.byref(defaulted),
                )
                or not present.value
                or not dacl
            ):
                raise self._native_failed()
            result = self._advapi32.SetSecurityInfo(
                self._handle(handle),
                _SE_FILE_OBJECT,
                _DACL_SECURITY_INFORMATION | _PROTECTED_DACL_SECURITY_INFORMATION,
                None,
                None,
                dacl,
                None,
            )
            if result:
                raise self._native_failed()
        finally:
            self._kernel32.LocalFree(descriptor)

    def verify_private_acl(self, handle: int, user_sid: bytes, directory: bool) -> bool:
        dacl = wintypes.LPVOID()
        descriptor = wintypes.LPVOID()
        result = self._advapi32.GetSecurityInfo(
            self._handle(handle),
            _SE_FILE_OBJECT,
            _DACL_SECURITY_INFORMATION,
            None,
            None,
            ctypes.byref(dacl),
            None,
            ctypes.byref(descriptor),
        )
        if result or not descriptor or not dacl:
            if descriptor:
                self._kernel32.LocalFree(descriptor)
            return False
        try:
            control = wintypes.WORD()
            revision = wintypes.DWORD()
            if (
                not self._advapi32.GetSecurityDescriptorControl(
                    descriptor, ctypes.byref(control), ctypes.byref(revision)
                )
                or not control.value & _SE_DACL_PROTECTED
            ):
                return False
            acl = ctypes.cast(dacl, ctypes.POINTER(_Acl)).contents
            if acl.AceCount != 3:
                return False
            expected = sorted(
                [self._sid_string(user_sid), _SYSTEM_SID, _ADMINISTRATORS_SID]
            )
            trustees: list[str] = []
            required_flags = _OBJECT_INHERIT_ACE | _CONTAINER_INHERIT_ACE
            for index in range(acl.AceCount):
                pointer = wintypes.LPVOID()
                if not self._advapi32.GetAce(dacl, index, ctypes.byref(pointer)):
                    return False
                ace = ctypes.cast(pointer, ctypes.POINTER(_AccessAllowedAce)).contents
                if (
                    ace.Header.AceType != _ACCESS_ALLOWED_ACE_TYPE
                    or ace.Mask != _FILE_ALL_ACCESS
                    or (
                        directory
                        and ace.Header.AceFlags & required_flags != required_flags
                    )
                    or (not directory and ace.Header.AceFlags & required_flags)
                ):
                    return False
                sid_pointer = ctypes.c_void_p(
                    int(ctypes.cast(pointer, ctypes.c_void_p).value or 0)
                    + _AccessAllowedAce.SidStart.offset
                )
                length = self._advapi32.GetLengthSid(sid_pointer)
                if not length:
                    return False
                trustees.append(self._sid_string(ctypes.string_at(sid_pointer, length)))
            return sorted(trustees) == expected
        finally:
            self._kernel32.LocalFree(descriptor)

    def write_all(self, handle: int, data: bytes) -> None:
        if len(data) > 0xFFFFFFFF:
            raise self._native_failed()
        written = wintypes.DWORD()
        buffer = ctypes.create_string_buffer(data)
        if not self._kernel32.WriteFile(
            self._handle(handle), buffer, len(data), ctypes.byref(written), None
        ) or written.value != len(data):
            raise self._native_failed()

    def read(self, handle: int, count: int, offset: int) -> bytes:
        if count > 0xFFFFFFFF or offset < 0:
            raise self._native_failed()
        if not self._kernel32.SetFilePointerEx(
            self._handle(handle), offset, None, _FILE_BEGIN
        ):
            raise self._native_failed()
        buffer = ctypes.create_string_buffer(count)
        read = wintypes.DWORD()
        if not self._kernel32.ReadFile(
            self._handle(handle), buffer, count, ctypes.byref(read), None
        ):
            raise self._native_failed()
        return bytes(buffer.raw[: read.value])

    def flush(self, handle: int) -> None:
        if not self._kernel32.FlushFileBuffers(self._handle(handle)):
            raise self._native_failed()

    def set_read_only(self, handle: int) -> None:
        information = _FileBasicInfo()
        native = self._handle(handle)
        if not self._kernel32.GetFileInformationByHandleEx(
            native,
            _FILE_BASIC_INFO_CLASS,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            raise self._native_failed()
        information.FileAttributes |= _FILE_ATTRIBUTE_READONLY
        if not self._kernel32.SetFileInformationByHandle(
            native,
            _FILE_BASIC_INFO_CLASS,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            raise self._native_failed()

    def lock_exclusive_nonblocking(self, handle: int) -> None:
        overlapped = _Overlapped()
        if not self._kernel32.LockFileEx(
            self._handle(handle),
            _LOCKFILE_EXCLUSIVE_LOCK | _LOCKFILE_FAIL_IMMEDIATELY,
            0,
            1,
            0,
            ctypes.byref(overlapped),
        ):
            get_last_error = getattr(ctypes, "get_last_error", lambda: 0)
            if get_last_error() in {32, 33}:
                raise BlockingIOError("native Windows artifact lock is busy")
            raise self._native_failed()

    def unlock(self, handle: int) -> None:
        overlapped = _Overlapped()
        if not self._kernel32.UnlockFileEx(
            self._handle(handle), 0, 1, 0, ctypes.byref(overlapped)
        ):
            raise self._native_failed()

    def delete_handle(self, handle: int) -> None:
        native = self._handle(handle)
        information = _FileBasicInfo()
        if not self._kernel32.GetFileInformationByHandleEx(
            native,
            _FILE_BASIC_INFO_CLASS,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            raise self._native_failed()
        if information.FileAttributes & _FILE_ATTRIBUTE_READONLY:
            information.FileAttributes &= ~_FILE_ATTRIBUTE_READONLY
            if not self._kernel32.SetFileInformationByHandle(
                native,
                _FILE_BASIC_INFO_CLASS,
                ctypes.byref(information),
                ctypes.sizeof(information),
            ):
                raise self._native_failed()
        disposition = _FileDispositionInfo(True)
        if not self._kernel32.SetFileInformationByHandle(
            native,
            _FILE_DISPOSITION_INFO_CLASS,
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        ):
            raise self._native_failed()

    def close_handle(self, handle: int) -> None:
        if not self._kernel32.CloseHandle(self._handle(handle)):
            raise self._native_failed()


def _native_windows_kernel() -> _WindowsKernel:
    """Construct the native kernel only for the admitted Windows tuple."""

    if not windows_audio_cpp_platform_supported():
        raise WindowsArtifactError("unsupported")
    return _CtypesWindowsKernel()


def _invalid_component(component: str) -> bool:
    if component in {"", ".."} or component.endswith((" ", ".")):
        return True
    if any(ord(character) < 32 for character in component):
        return True
    if any(character in _INVALID_COMPONENT_CHARACTERS for character in component):
        return True
    basename = component.split(".", 1)[0].casefold()
    return basename in _RESERVED_NAMES


def normalize_windows_artifact_path(value: str | os.PathLike[str]) -> str:
    """Return one validated extended-length drive or UNC path.

    User-provided device namespaces are rejected. The extended namespace is
    introduced only after the ordinary path has passed lexical validation.

    Args:
        value: Ordinary absolute Windows drive or UNC path.

    Returns:
        The equivalent extended-length path used by native calls.

    Raises:
        ValueError: If the value is relative, ambiguous, or unsafe.
    """

    try:
        raw = os.fspath(value)
    except (TypeError, ValueError):
        raise ValueError(_PATH_ERROR) from None
    if type(raw) is not str or not raw or "\x00" in raw:
        raise ValueError(_PATH_ERROR)
    normalized_separators = raw.replace("/", "\\")
    folded = normalized_separators.casefold()
    if folded.startswith(_DEVICE_PREFIXES):
        raise ValueError(_PATH_ERROR)

    selected = PureWindowsPath(normalized_separators)
    if not selected.is_absolute() or not selected.drive or selected.root != "\\":
        raise ValueError(_PATH_ERROR)
    if any(_invalid_component(component) for component in selected.parts[1:]):
        raise ValueError(_PATH_ERROR)

    ordinary = str(selected)
    if selected.drive.startswith("\\\\"):
        return "\\\\?\\UNC\\" + ordinary.removeprefix("\\\\")
    if len(selected.drive) != 2 or selected.drive[1] != ":":
        raise ValueError(_PATH_ERROR)
    return f"\\\\?\\{ordinary}"


def normalize_windows_process_architecture(
    machine: str,
    pointer_bits: int,
) -> WindowsProcessArchitecture | None:
    """Return the supported process architecture without projecting ARM."""

    if type(machine) is not str or type(pointer_bits) is not int:
        return None
    folded = machine.casefold()
    if folded not in _X86_MACHINES:
        return None
    if pointer_bits == 32:
        return "x86"
    if pointer_bits == 64 and folded in _X64_MACHINES:
        return "x86_64"
    return None


def windows_audio_cpp_platform_supported(
    *,
    system_name: str | None = None,
    windows_major: int | None = None,
    python_version: tuple[int, int] | None = None,
    machine: str | None = None,
    pointer_bits: int | None = None,
) -> bool:
    """Return whether the current or supplied tuple meets the Windows floor."""

    selected_system = platform.system() if system_name is None else system_name
    if selected_system.casefold() != "windows":
        return False
    if windows_major is None:
        get_windows_version = getattr(sys, "getwindowsversion", None)
        selected_windows_major = (
            int(get_windows_version().major) if callable(get_windows_version) else 0
        )
    else:
        selected_windows_major = windows_major
    selected_python = (
        (sys.version_info.major, sys.version_info.minor)
        if python_version is None
        else python_version
    )
    selected_machine = platform.machine() if machine is None else machine
    selected_pointer_bits = (
        struct.calcsize("P") * 8 if pointer_bits is None else pointer_bits
    )
    return bool(
        type(selected_windows_major) is int
        and selected_windows_major >= 10
        and type(selected_python) is tuple
        and len(selected_python) == 2
        and selected_python >= (3, 12)
        and normalize_windows_process_architecture(
            selected_machine,
            selected_pointer_bits,
        )
        is not None
    )


OS_WINDOWS_ARTIFACT_FILESYSTEM: WindowsArtifactFilesystem = (
    NativeWindowsArtifactFilesystem()
    if windows_audio_cpp_platform_supported()
    else UnavailableWindowsArtifactFilesystem()
)


__all__ = (
    "NativeWindowsArtifactFilesystem",
    "OS_WINDOWS_ARTIFACT_FILESYSTEM",
    "UnavailableWindowsArtifactFilesystem",
    "WindowsArtifactError",
    "WindowsArtifactFilesystem",
    "WindowsArtifactKind",
    "WindowsArtifactPrivacyPosture",
    "WindowsFileIdentity",
    "WindowsPinnedHandle",
    "WindowsProcessArchitecture",
    "normalize_windows_artifact_path",
    "normalize_windows_process_architecture",
    "take_windows_artifact_cleanup_owner",
    "windows_audio_cpp_platform_supported",
)
