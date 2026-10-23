"""Native shared-description lifetime for one private binding (no app imports).

The fixed-purpose scanner creates no descendants. Close, never unlock, the
parent reference: inherited references must continue excluding replacement.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import stat
import subprocess
import threading

_spawn_lock = threading.Lock()


class LifetimeBusy(RuntimeError):
    """A prior writer or inherited scanner still holds the native capability."""


class BindingLifetime:
    def __init__(self, path: Path, handle: int):
        self.path = path
        self.handle = handle
        self.closed = False
        self._receipt_descriptors: set[int] = set()
        self._receipt_handles: set[int] = set()

    def assert_live(self) -> None:
        """Check that this code-owned native capability still names a live file."""
        if self.closed:
            raise ValueError("lifetime_closed")
        if os.name == "nt":
            from ctypes import wintypes

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetFileType.argtypes = [wintypes.HANDLE]
            kernel.GetFileType.restype = wintypes.DWORD
            if kernel.GetFileType(wintypes.HANDLE(self.handle)) != 1:
                raise OSError("lifetime_handle_invalid")
        else:
            held = os.fstat(self.handle)
            named = self.path.stat(follow_symlinks=False)
            if (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino):
                raise OSError("lifetime_handle_replaced")

    @classmethod
    def acquire(cls, path: Path, *, existing_only: bool = False) -> BindingLifetime:
        """Acquire an existing-or-new stable object; caller admits private parent."""
        path = Path(path)
        if os.name == "nt":
            from ctypes import wintypes

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            create = kernel.CreateFileW
            create.argtypes = [
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.LPVOID,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.HANDLE,
            ]
            create.restype = wintypes.HANDLE
            handle = create(
                str(path),
                0x80000000 | 0x40000000,
                0,
                None,
                3 if existing_only else 4,
                0x00200000,
                None,
            )  # OPEN_ALWAYS, OPEN_REPARSE_POINT
            if handle == wintypes.HANDLE(-1).value:
                error = ctypes.get_last_error()
                if error in (32, 33):
                    raise LifetimeBusy("binding_lifetime_busy")
                raise ctypes.WinError(error)
            try:

                class AttributeTag(ctypes.Structure):
                    _fields_ = [("attributes", wintypes.DWORD), ("tag", wintypes.DWORD)]

                info = AttributeTag()
                query = kernel.GetFileInformationByHandleEx
                query.argtypes = [
                    wintypes.HANDLE,
                    ctypes.c_int,
                    wintypes.LPVOID,
                    wintypes.DWORD,
                ]
                query.restype = wintypes.BOOL
                if not query(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
                    raise ctypes.WinError(ctypes.get_last_error())

                class StandardInfo(ctypes.Structure):
                    _fields_ = [
                        ("allocation_size", ctypes.c_longlong),
                        ("end_of_file", ctypes.c_longlong),
                        ("links", wintypes.DWORD),
                        ("delete_pending", ctypes.c_ubyte),
                        ("directory", ctypes.c_ubyte),
                    ]

                standard = StandardInfo()
                if not query(
                    handle, 1, ctypes.byref(standard), ctypes.sizeof(standard)
                ):
                    raise ctypes.WinError(ctypes.get_last_error())
                if info.attributes & (0x400 | 0x10) or standard.links != 1:
                    raise ValueError("lifetime_alias_refused")
            except BaseException:
                kernel.CloseHandle(wintypes.HANDLE(handle))
                raise
        elif os.name == "posix":
            import fcntl

            handle = os.open(
                path,
                (0 if existing_only else os.O_CREAT)
                | os.O_RDWR
                | os.O_NOFOLLOW
                | os.O_CLOEXEC,
                0o600,
            )
            try:
                info = os.fstat(handle)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError("lifetime_alias_refused")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise LifetimeBusy("binding_lifetime_busy") from error
            except BaseException:
                os.close(handle)
                raise
        else:
            raise RuntimeError("lifetime_backend_unavailable")
        return cls(path, handle)

    @contextmanager
    def _cleanup_receipt_descriptor(self):
        """Duplicate only this pinned control file, never reopen a pathname."""
        if self.closed or self.path.name != "cleanup-owner-v1.lock":
            raise ValueError("invalid_cleanup_receipt_handle")
        if os.name == "nt":
            import msvcrt
            from ctypes import wintypes

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            duplicate = kernel.DuplicateHandle
            duplicate.argtypes = [
                wintypes.HANDLE,
                wintypes.HANDLE,
                wintypes.HANDLE,
                ctypes.POINTER(wintypes.HANDLE),
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            ]
            duplicate.restype = wintypes.BOOL
            copied = wintypes.HANDLE()
            process = wintypes.HANDLE(-1)
            if not duplicate(
                process, self.handle, process, ctypes.byref(copied), 0, False, 2
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            self._receipt_handles.add(copied.value)
            try:
                descriptor = msvcrt.open_osfhandle(
                    copied.value, os.O_RDWR | os.O_BINARY
                )
            except BaseException:
                if not kernel.CloseHandle(copied):
                    raise ctypes.WinError(ctypes.get_last_error())
                self._receipt_handles.remove(copied.value)
                raise
            self._receipt_handles.remove(copied.value)
        else:
            descriptor = os.dup(self.handle)
        self._receipt_descriptors.add(descriptor)
        try:
            yield descriptor
        finally:
            os.close(descriptor)
            self._receipt_descriptors.remove(descriptor)

    def read_cleanup_receipt(self) -> bytes:
        """Read at most 4096 control bytes through the exclusive native handle."""
        with self._cleanup_receipt_descriptor() as descriptor:
            if os.fstat(descriptor).st_size > 4096:
                raise ValueError("cleanup_receipt_too_large")
            os.lseek(descriptor, 0, os.SEEK_SET)
            chunks = bytearray()
            while len(chunks) <= 4096:
                block = os.read(descriptor, 4097 - len(chunks))
                if not block:
                    return bytes(chunks)
                chunks.extend(block)
            raise ValueError("cleanup_receipt_too_large")

    def write_cleanup_receipt(self, payload: bytes) -> None:
        """Enroll/repair bounded live-owned metadata, flush before publication."""
        if not isinstance(payload, bytes) or not 0 < len(payload) <= 4096:
            raise ValueError("invalid_cleanup_receipt_size")
        with self._cleanup_receipt_descriptor() as descriptor:
            os.lseek(descriptor, 0, os.SEEK_SET)
            remaining = memoryview(payload)
            while remaining:
                count = os.write(descriptor, remaining)
                if count <= 0:
                    raise OSError("cleanup_receipt_write_incomplete")
                remaining = remaining[count:]
            os.ftruncate(descriptor, len(payload))
            os.fsync(descriptor)

    @contextmanager
    def inheritance(self):
        """Supply a narrow creation-time capability list, restoring inheritability."""
        if self.closed:
            raise ValueError("lifetime_closed")
        if os.name == "nt":
            with _spawn_lock:
                previous = os.get_handle_inheritable(self.handle)
                os.set_handle_inheritable(self.handle, True)
                try:
                    startup = subprocess.STARTUPINFO()
                    startup.lpAttributeList = {"handle_list": [self.handle]}
                    yield {"startupinfo": startup, "close_fds": True}
                finally:
                    os.set_handle_inheritable(self.handle, previous)
        else:
            yield {"pass_fds": (self.handle,), "close_fds": True}

    def spawn(self, args, **kwargs):
        """Create the contained worker with the native reference already inherited."""
        with self.inheritance() as inherited:
            return subprocess.Popen(args, **kwargs, **inherited)

    def close(self) -> None:
        if self.closed:
            return
        if self._receipt_descriptors or self._receipt_handles:
            raise OSError("cleanup_receipt_descriptor_retirement_unproven")
        if os.name == "nt":
            from ctypes import wintypes

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            if not kernel.CloseHandle(wintypes.HANDLE(self.handle)):
                raise ctypes.WinError(ctypes.get_last_error())
        else:
            os.close(self.handle)
        self.closed = True


def retain_inherited(value: str) -> int:
    """Validate bootstrap capability metadata, retaining the raw reference to exit."""
    if not value.isascii() or not value.isdecimal():
        raise ValueError("invalid_lifetime_capability")
    handle = int(value)
    if handle <= 0:
        raise ValueError("invalid_lifetime_capability")
    if os.name == "nt":
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        query = kernel.GetFileType
        query.argtypes = [wintypes.HANDLE]
        query.restype = wintypes.DWORD
        if query(wintypes.HANDLE(handle)) != 1:
            raise ValueError("invalid_lifetime_capability")
    elif os.name == "posix":
        if not stat.S_ISREG(os.fstat(handle).st_mode):
            raise ValueError("invalid_lifetime_capability")
    else:
        raise ValueError("lifetime_backend_unavailable")
    return handle
