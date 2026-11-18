"""Windows run presence across profile-writer restarts, without a storage lease.

The process executing a run holds an unsignalled kernel event. Windows retires
it on process exit. Reconciliation tests only whether that exact run still has
an executor. No SQLite transaction or recovery handle spans the provider wait.
"""

from __future__ import annotations

import ctypes
import hashlib
import sqlite3
import threading
from functools import cache
from pathlib import Path

from tldw_chatbook.Utils.platform_files import os


@cache
def _event_api():
    from ._profile_writer_windows import _SecurityAttributes

    library = ctypes.WinDLL("kernel32", use_last_error=True)
    library.CreateEventW.argtypes = [
        ctypes.POINTER(_SecurityAttributes),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_wchar_p,
    ]
    library.CreateEventW.restype = ctypes.c_void_p
    library.OpenEventW.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_wchar_p]
    library.OpenEventW.restype = ctypes.c_void_p
    return library


def _event_name(profile: Path, run_id: str, sid: str) -> str:
    # The name identifies the actual profile directory, including case aliases.
    stamp = os.stat(profile, follow_symlinks=False)
    digest = hashlib.sha256(
        f"{sid}/{stamp.st_dev}/{stamp.st_ino}/{run_id}".encode()
    ).hexdigest()
    return "Global\\tldw-agent-run-" + digest


_handles = {}
_lock = threading.Lock()


def claim_run(profile: Path, run_id: str) -> None:
    """Publish liveness before submitting create_run, rejecting duplicate owners."""
    from tldw_chatbook.Utils.windows_files import _native
    from ._profile_writer_windows import private_security_attributes

    native = _native()
    name = _event_name(profile, run_id, native.user_sid)
    api = _event_api()
    with private_security_attributes() as attributes:
        ctypes.set_last_error(0)
        event = api.CreateEventW(ctypes.byref(attributes), True, False, name)
        error = ctypes.get_last_error()
        if not event:
            raise ctypes.WinError(error)
        if error == 183:  # ERROR_ALREADY_EXISTS
            native.check(native.kernel.CloseHandle(event))
            raise sqlite3.IntegrityError("Agent run already has an executor")
        with _lock:
            _handles[profile, run_id] = event


def run_is_live(profile: Path, run_id: str) -> bool:
    """Only a positively absent kernel object proves the run can be repaired."""
    from tldw_chatbook.Utils.windows_files import _native
    from ._profile_writer_windows import require_owner

    native = _native()
    name = _event_name(profile, run_id, native.user_sid)
    event = _event_api().OpenEventW(0x00020000, False, name)  # READ_CONTROL
    if not event:
        error = ctypes.get_last_error()
        if error == 2:  # ERROR_FILE_NOT_FOUND
            return False
        raise ctypes.WinError(error)
    try:
        require_owner(event)
    finally:
        native.check(native.kernel.CloseHandle(event))
    return True


def release_run(profile: Path, run_id: str) -> None:
    from tldw_chatbook.Utils.windows_files import _native

    with _lock:
        event = _handles.get((profile, run_id))
        if event is not None:
            native = _native()
            native.check(native.kernel.CloseHandle(event))
            del _handles[profile, run_id]
