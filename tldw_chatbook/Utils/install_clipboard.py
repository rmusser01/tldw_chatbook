"""Checked clipboard delivery for user-requested install commands."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from textual.widget import Widget


_NATIVE_TIMEOUT_SECONDS = 2.0
# Qt's process-owned clipboard cannot survive this short-lived helper. Reject
# its self-readback; Linux native delivery needs wl-clipboard/xclip/xsel/klipper.
_COPY_SCRIPT = (
    "import sys, pyperclip; "
    "text = sys.stdin.buffer.read().decode('utf-8'); "
    "pyperclip.copy(text); "
    "sys.exit(0 if pyperclip.copy.__name__ != 'copy_qt' "
    "and pyperclip.paste() == text else 1)"
)


def _copy_native(command: str) -> bool:
    """Bound the native write AND readback, including clipboard subprocesses."""
    with subprocess.Popen(
        [sys.executable, "-c", _COPY_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=os.name == "posix",
    ) as process:
        try:
            process.communicate(
                command.encode("utf-8"), timeout=_NATIVE_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired:
            # Cancelling an asyncio/to_thread waiter leaves the write alive.
            # Kill its owned process tree before allowing another copy.
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
            return False
        return process.returncode == 0


async def copy_install_command(widget: Widget, command: str) -> bool:
    """Copy an install command and report whether delivery was confirmed.

    Native clipboard work runs off the UI thread. Browser and SSH sessions
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
            confirmed = await asyncio.to_thread(_copy_native, command)
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
