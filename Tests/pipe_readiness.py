"""Bounded readiness for actual subprocess pipes on their native platform."""

import os
import select
import time


def pipe_readable(stream, timeout: float) -> bool:
    """Wait for bytes or EOF without consuming the subprocess response."""
    if os.name != "nt":
        return bool(select.select([stream], [], [], timeout)[0])
    import ctypes
    from ctypes import wintypes
    import msvcrt

    peek = ctypes.WinDLL("kernel32", use_last_error=True).PeekNamedPipe
    peek.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    peek.restype = wintypes.BOOL
    handle = msvcrt.get_osfhandle(stream.fileno())
    deadline = time.monotonic() + timeout
    while True:
        available = wintypes.DWORD()
        if not peek(handle, None, 0, None, ctypes.byref(available), None):
            error = ctypes.get_last_error()
            if error == 109:  # ERROR_BROKEN_PIPE: EOF is readable too.
                return True
            raise ctypes.WinError(error)
        if available.value:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(remaining, 0.01))
