"""One local, token-user-only Windows named pipe for a profile writer."""

from __future__ import annotations

import _winapi
import ctypes as C
import io
import struct
import time
from collections.abc import Iterator
from contextlib import contextmanager
from ctypes import wintypes as W

from tldw_chatbook.Utils.windows_files import _native


# A stock ChaChaNotes SQLite row may contain a 1,000,000,000-byte blob,
# base64-encoded by the domain codec. Archive imports support 528 MiB before
# encoding. Enforce an independent aggregate cap without preallocating it.
MAX_MESSAGE_BYTES = 2**31 - 1
MAX_FRAME_BYTES = 1024 * 1024
_BUFFER_BYTES = 64 * 1024
_MAGIC = b"PWJ2"
_END = bytes(4)
_MIN_BYTES_PER_SECOND = 1024 * 1024
_FRAME_IDLE_SECONDS = 10.0
_PIPE_REJECT_REMOTE_CLIENTS = 8
_SECURITY_SQOS_PRESENT = 0x00100000
_KERNEL_OBJECT = 6
_OWNER_AND_DACL = 5


class _SecurityAttributes(C.Structure):
    _fields_ = [
        ("length", W.DWORD), ("descriptor", W.LPVOID), ("inherit", W.BOOL),
    ]


_kernel = C.WinDLL("kernel32", use_last_error=True)
_disconnect = _kernel.DisconnectNamedPipe
_disconnect.argtypes, _disconnect.restype = [W.HANDLE], W.BOOL
_final_path = _kernel.GetFinalPathNameByHandleW
_final_path.argtypes = [W.HANDLE, W.LPWSTR, W.DWORD, W.DWORD]
_final_path.restype = W.DWORD
_server_pid = _kernel.GetNamedPipeServerProcessId
_server_pid.argtypes, _server_pid.restype = [W.HANDLE, C.POINTER(W.ULONG)], W.BOOL
_client_pid = _kernel.GetNamedPipeClientProcessId
_client_pid.argtypes, _client_pid.restype = [W.HANDLE, C.POINTER(W.ULONG)], W.BOOL


def final_directory_name(fd: int) -> str:
    """Return one normalized volume-GUID spelling from the checked live handle."""
    handle = _native().handle(fd)
    result = C.create_unicode_buffer(32768)
    length = _final_path(handle, result, len(result), 1)
    if not length or length >= len(result):
        raise OSError("writer profile directory identity unavailable")
    return result.value.casefold()


def require_owner(handle: int) -> None:
    """Require a kernel object owned by and accessible to this token user alone."""
    native = _native()
    owner, acl, descriptor = C.c_void_p(), C.c_void_p(), C.c_void_p()
    status = native.advapi.GetSecurityInfo(
        handle, _KERNEL_OBJECT, _OWNER_AND_DACL,
        C.byref(owner), None, C.byref(acl), None, C.byref(descriptor),
    )
    if status:
        raise C.WinError(status)
    try:
        if (
            native.sid_string(owner) == native.user_sid and acl.value
            and struct.unpack("<H", C.string_at(acl.value + 4, 2))[0] == 1
        ):
            ace = C.c_void_p()
            native.check(native.advapi.GetAce(acl, 0, C.byref(ace)))
            # ACCESS_ALLOWED_ACE, no inheritance/condition, same TokenUser SID.
            if C.string_at(ace, 2) == b"\0\0" and native.sid_string(ace.value + 8) == native.user_sid:
                return
        raise PermissionError("profile writer endpoint owner or permissions changed")
    finally:
        native.kernel.LocalFree(descriptor)


@contextmanager
def private_security_attributes() -> Iterator[_SecurityAttributes]:
    """Supply a protected DACL for exactly this TokenUser on a new kernel object."""
    native = _native()
    descriptor = C.c_void_p()
    sddl = f"O:{native.user_sid}D:P(A;;GA;;;{native.user_sid})"
    native.check(native.advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        sddl, 1, C.byref(descriptor), None,
    ))
    try:
        yield _SecurityAttributes(C.sizeof(_SecurityAttributes), descriptor, False)
    finally:
        native.kernel.LocalFree(descriptor)


@contextmanager
def server(name: str) -> Iterator[int]:
    """Elect one owner. Keep this handle across disconnect and physical work."""
    with private_security_attributes() as attributes:
        handle = _winapi.CreateNamedPipe(
            name, _winapi.PIPE_ACCESS_DUPLEX | _winapi.FILE_FLAG_FIRST_PIPE_INSTANCE
            | _winapi.FILE_FLAG_OVERLAPPED, _PIPE_REJECT_REMOTE_CLIENTS, 1,
            _BUFFER_BYTES, _BUFFER_BYTES, 0, C.addressof(attributes),
        )
    try:
        yield handle
    finally:
        _winapi.CloseHandle(handle)


@contextmanager
def client(name: str) -> Iterator[int]:
    """Open the elected local owner. The peer receives no impersonation rights."""
    handle = _winapi.CreateFile(
        name, _winapi.GENERIC_READ | _winapi.GENERIC_WRITE, 0, 0,
        _winapi.OPEN_EXISTING, _winapi.FILE_FLAG_OVERLAPPED | _SECURITY_SQOS_PRESENT, 0,
    )
    try:
        require_owner(handle)
        yield handle
    finally:
        _winapi.CloseHandle(handle)


def server_process(handle: int) -> int:
    """Pin the actual process behind a connected endpoint for drained shutdown."""
    pid = W.ULONG()
    _native().check(_server_pid(handle, C.byref(pid)))
    return _winapi.OpenProcess(_winapi.SYNCHRONIZE, False, pid.value)


def client_pid(handle: int) -> int:
    """Read the host PID from the connected kernel pipe, never from JSON."""
    pid = W.ULONG()
    _native().check(_client_pid(handle, C.byref(pid)))
    return pid.value


def wait_available(name: str, timeout: float) -> None:
    _winapi.WaitNamedPipe(name, max(1, int(timeout * 1000)))


def _complete(overlapped, deadline: float | None) -> int:
    """A timeout cancels and joins native I/O before the handle can be reused."""
    try:
        milliseconds = (
            _winapi.INFINITE if deadline is None
            else max(0, int((deadline - time.monotonic()) * 1000))
        )
        if _winapi.WaitForSingleObject(overlapped.event, milliseconds) != _winapi.WAIT_OBJECT_0:
            raise TimeoutError("profile writer pipe timed out")
    except BaseException:
        overlapped.cancel()
        overlapped.GetOverlappedResult(True)
        raise
    transferred, error = overlapped.GetOverlappedResult(True)
    if error:
        raise C.WinError(error)
    return transferred


def accept(handle: int, timeout: float) -> None:
    _complete(_winapi.ConnectNamedPipe(handle, overlapped=True), time.monotonic() + timeout)


def disconnect(handle: int) -> None:
    _native().check(_disconnect(handle))


def _read_exact(handle: int, size: int, deadline: float | None) -> bytes:
    data = bytearray()
    while len(data) < size:
        read, _ = _winapi.ReadFile(handle, min(size - len(data), _BUFFER_BYTES), True)
        if not _complete(read, deadline):
            raise EOFError("profile writer pipe closed")
        data.extend(read.getbuffer())
    return bytes(data)


def read_frame(handle: int, timeout: float | None, *, maximum: int = MAX_MESSAGE_BYTES) -> bytes:
    """Receive a complete length-prefixed chunk sequence, including its terminator."""
    deadline = None if timeout is None else time.monotonic() + timeout
    if _read_exact(handle, 4, deadline) != _MAGIC:
        raise ValueError("invalid profile writer frame protocol")
    idle = _FRAME_IDLE_SECONDS if timeout is None else timeout
    total, = struct.unpack("!Q", _read_exact(handle, 8, time.monotonic() + idle))
    if not 0 < total <= min(maximum, MAX_MESSAGE_BYTES):
        raise ValueError("profile writer frame length out of bounds")
    deadline = time.monotonic() + idle + total / _MIN_BYTES_PER_SECOND
    body = io.BytesIO()
    while True:
        expires = min(deadline, time.monotonic() + idle)
        size, = struct.unpack("!I", _read_exact(handle, 4, expires))
        if size == 0 and body.tell() == total:
            return body.getvalue()
        if not 0 < size <= min(MAX_FRAME_BYTES, total - body.tell()):
            raise ValueError("invalid profile writer frame sequence")
        body.write(_read_exact(handle, size, expires))


def _write(handle: int, data: bytes, deadline: float) -> None:
    write, _ = _winapi.WriteFile(handle, data, True)
    if _complete(write, deadline) != len(data):
        raise OSError("profile writer frame was incomplete")


def write_frame(handle: int, body: bytes, timeout: float) -> None:
    total = len(body)
    if not 0 < total <= MAX_MESSAGE_BYTES:
        raise ValueError("profile writer frame length out of bounds")
    header = _MAGIC + struct.pack("!Q", total)
    if total <= MAX_FRAME_BYTES:
        _write(handle, header + struct.pack("!I", total) + body + _END, time.monotonic() + timeout)
        return
    deadline = time.monotonic() + timeout + total / _MIN_BYTES_PER_SECOND
    _write(handle, header, time.monotonic() + timeout)
    view = memoryview(body)
    for start in range(0, total, MAX_FRAME_BYTES):
        chunk = view[start:start + MAX_FRAME_BYTES]
        _write(handle, struct.pack("!I", len(chunk)) + chunk, min(deadline, time.monotonic() + timeout))
    _write(handle, _END, min(deadline, time.monotonic() + timeout))
