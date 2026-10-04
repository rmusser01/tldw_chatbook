"""Bounded observations of real subprocess pipes without consuming readiness."""

from __future__ import annotations

import os
import select
import subprocess
import time
from pathlib import Path
from typing import IO, Any, Sequence

if os.name == "nt":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _peek_named_pipe = ctypes.WinDLL("kernel32", use_last_error=True).PeekNamedPipe
    _peek_named_pipe.argtypes = (
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
    )
    _peek_named_pipe.restype = wintypes.BOOL
    _ERROR_BROKEN_PIPE = 109

_POLL_INTERVAL = 0.005


def popen_with_captured_stderr(
    args: Sequence[str], stderr_path: Path, **options: Any
) -> subprocess.Popen:
    """Start a child with preserved stderr that cannot fill an unread pipe.

    Args:
        args: Child command and arguments.
        stderr_path: Owned test path for this child's diagnostic output.
        **options: Popen options other than stderr.

    Returns:
        The real child with a separate readable stderr file. The caller retains
        responsibility for killing/waiting and closing its stdin/stdout/stderr.
    """
    with stderr_path.open("wb") as captured:
        child = subprocess.Popen(args, stderr=captured, **options)
    try:
        if child.text_mode:
            child.stderr = stderr_path.open(
                "r", encoding=child.encoding, errors=child.errors
            )
        else:
            child.stderr = stderr_path.open("rb")
    except BaseException:
        child.kill()
        child.wait(timeout=5)
        if child.stdin is not None:
            child.stdin.close()
        if child.stdout is not None:
            child.stdout.close()
        raise
    return child


def pipe_ready(stream: IO, timeout: float) -> bool:
    """Wait for pipe bytes or EOF, preserving bytes for the sole reader.

    Args:
        stream: Subprocess output pipe with no other active reader.
        timeout: Maximum seconds to wait, or zero for an immediate observation.

    Returns:
        Whether bytes or EOF can be read immediately.

    Raises:
        OSError: The pipe cannot be observed.
        ValueError: The stream is closed or the timeout is negative.
    """
    if timeout < 0:
        raise ValueError("timeout must be non-negative")
    descriptor = stream.fileno()
    if os.name != "nt":
        return bool(select.select([descriptor], [], [], timeout)[0])

    # WinSock select accepts sockets only. PeekNamedPipe also accepts anonymous
    # pipes and leaves their bytes in place, so negative assertions cannot hide
    # a response through a background reader's read-ahead buffer.
    handle = msvcrt.get_osfhandle(descriptor)
    deadline = time.monotonic() + timeout
    while True:
        available = wintypes.DWORD()
        if not _peek_named_pipe(handle, None, 0, None, ctypes.byref(available), None):
            error = ctypes.get_last_error()
            if error == _ERROR_BROKEN_PIPE:
                return True  # Match POSIX select's readable EOF behavior.
            raise ctypes.WinError(error)
        if available.value:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(_POLL_INTERVAL, remaining))


def read_line(child: subprocess.Popen, timeout: float = 10) -> str:
    """Read one response within a single deadline, including partial lines.

    Args:
        child: Child process whose stdout is a pipe with no other active reader.
        timeout: Maximum seconds for the entire response.

    Returns:
        A decoded response with surrounding whitespace removed.

    Raises:
        AssertionError: The child times out or closes stdout before a newline.
        OSError: The pipe cannot be observed or read.
    """
    assert child.stdout is not None, "child stdout is not piped"
    deadline = time.monotonic() + timeout
    data = bytearray()
    while not data.endswith(b"\n"):
        remaining = deadline - time.monotonic()
        assert remaining > 0 and pipe_ready(
            child.stdout, remaining
        ), "child did not respond"
        part = os.read(child.stdout.fileno(), 1)
        assert part, "child exited before response"
        data.extend(part)
    return data.decode().strip()
