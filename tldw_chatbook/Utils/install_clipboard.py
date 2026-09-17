"""Checked clipboard delivery for user-requested install commands."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

if TYPE_CHECKING:
    from textual.app import App
    from textual.widget import Widget


_NATIVE_TIMEOUT_SECONDS = 2.0
_NATIVE_COPY_LOCKS: WeakKeyDictionary[App, asyncio.Lock] = WeakKeyDictionary()
# Qt's process-owned clipboard cannot survive this short-lived helper. Reject
# its self-readback; Linux native delivery needs wl-clipboard/xclip/xsel/klipper.
_COPY_SCRIPT = (
    "import sys, pyperclip; "
    "text = sys.stdin.buffer.read().decode('utf-8'); "
    "pyperclip.copy(text); "
    "sys.exit(0 if pyperclip.copy.__name__ != 'copy_qt' "
    "and pyperclip.paste() == text else 1)"
)


async def _copy_native(command: str) -> bool:
    """Bound the native write AND readback, including clipboard subprocesses."""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _COPY_SCRIPT,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=os.name == "posix",
    )
    try:
        await asyncio.wait_for(
            process.communicate(command.encode("utf-8")), _NATIVE_TIMEOUT_SECONDS
        )
    except (TimeoutError, asyncio.CancelledError) as exc:
        # Own cleanup through cancellation as well as the timeout. Returning
        # while a worker thread still owns the child permits a stale write.
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        await process.wait()
        if isinstance(exc, asyncio.CancelledError):
            raise
        return False
    return process.returncode == 0


async def copy_install_command(widget: Widget, command: str) -> bool:
    """Copy an install command and report whether delivery was confirmed.

    Native clipboard work runs in a child process. Browser and SSH sessions
    use Textual's client-facing clipboard route instead of the host desktop.
    OSC 52 has no acknowledgement, so that fallback cannot claim success.

    Args:
        widget: Mounted widget owning the copy action and notification.
        command: Literal command requested by the user.

    Returns:
        True only when native clipboard readback matched the command.
    """
    app = widget.app
    remote = (
        app.is_web
        or app.is_headless
        or any(
            os.environ.get(name) for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")
        )
    )
    if not remote:
        try:
            lock = _NATIVE_COPY_LOCKS.setdefault(app, asyncio.Lock())
            async with lock:
                confirmed = await _copy_native(command)
        except (OSError, subprocess.SubprocessError):
            confirmed = False
        if confirmed:
            widget.notify("Install command copied to clipboard")
            return True

    try:
        app.copy_to_clipboard(command)
    except Exception:  # noqa: BLE001 - driver failures must leave manual recovery usable.
        widget.notify(
            "Clipboard unavailable. Select the displayed install command manually.",
            severity="warning",
        )
    else:
        widget.notify(
            "Copy requested; clipboard delivery could not be confirmed. "
            "If paste is empty, select the displayed install command manually.",
            severity="warning",
        )
    return False
