"""Opt-in OS notifications for background Console events (TASK-18925).

A presentation-only channel mirroring the Console's existing in-app toasts:
when a background run finishes, fails, or parks an approval, the same
once-guarded bridge methods that toast can also fire a desktop
notification. The notification body carries ONLY the conversation/tab
display name plus the outcome category -- never message text, tool output,
or credentials (the subprocess command line is constructed from exactly
those two strings and nothing else).

Platform support is honest: macOS (``osascript``) and Linux
(``notify-send``) dispatch; anything else (Windows included) skips with a
debug log rather than pretending. The dispatcher is fixed-argv (no shell),
runs with a scrubbed minimal environment, is bounded by a hard subprocess
timeout, and swallows every spawn failure -- a missing binary or a stuck
notifier must never crash or block the app. Callers run it off the
UI/event-loop thread (a daemon one-shot thread bounded by the same
timeout); this module itself is thread-agnostic and side-effect-pure
apart from the subprocess it spawns.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from loguru import logger

from tldw_chatbook.Utils.Utils import coerce_bool_flag

OSNotificationKind = Literal["run_finished", "run_failed", "needs_approval"]

#: Fixed app identity shown as the notification title / notify-send app name.
OS_NOTIFICATION_APP_NAME = "tldw Chatbook"

#: Hard wall-clock bound on the notification subprocess. The sender also
#: starts the child in its own session (``start_new_session``), isolating
#: it from this process's groups, mirroring the Console run-hooks
#: discipline (ADR-148).
OS_NOTIFICATION_TIMEOUT_S = 3.0

#: Outcome categories this channel knows about. This map is the single
#: source of the category copy; nothing else may inject notification body
#: text.
OS_NOTIFICATION_KIND_COPY: dict[str, str] = {
    "run_finished": "run finished",
    "run_failed": "run failed",
    "needs_approval": "needs approval",
}


@dataclass(frozen=True)
class ConsoleOSNotificationSettings:
    """Normalized ``[console] os_notifications`` configuration."""

    enabled: bool = False
    run_finished: bool = True
    run_failed: bool = True
    needs_approval: bool = True

    def allows(self, kind: str) -> bool:
        """Return whether ``kind`` may fire: master ON and its type ON."""
        return self.enabled and getattr(self, kind, False)

    def to_config(self) -> dict[str, object]:
        """Serialize to the app config dictionary shape."""
        return {
            "enabled": self.enabled,
            "run_finished": self.run_finished,
            "run_failed": self.run_failed,
            "needs_approval": self.needs_approval,
        }


def normalize_console_os_notifications(
    values: Mapping[str, object] | object | None,
) -> ConsoleOSNotificationSettings:
    """Normalize raw ``[console] os_notifications`` values to safe settings.

    The master toggle defaults OFF (the feature is opt-in, TASK-18925 AC#1);
    the per-type toggles default ON so enabling the master surfaces all
    three categories. Unrecognizable input coerces to those defaults rather
    than erroring -- a corrupt config must never widen the channel.
    """
    raw = values if isinstance(values, Mapping) else {}
    return ConsoleOSNotificationSettings(
        enabled=coerce_bool_flag(raw.get("enabled"), False),
        run_finished=coerce_bool_flag(raw.get("run_finished"), True),
        run_failed=coerce_bool_flag(raw.get("run_failed"), True),
        needs_approval=coerce_bool_flag(raw.get("needs_approval"), True),
    )


def _applescript_quote(value: str) -> str:
    """Escape ``value`` for embedding inside an AppleScript double-quoted string."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _single_line(value: str) -> str:
    """Collapse control characters to spaces so no argv element breaks lines."""
    return "".join(ch if (ch.isprintable() or ch == " ") else " " for ch in value)


def build_notification_argv(
    platform: str, kind: str, session_display: str
) -> list[str] | None:
    """Build the fixed notification argv for ``platform``.

    Args:
        platform: ``sys.platform``-style identifier (``"darwin"``,
            ``"linux"``, ``"win32"``, ...).
        kind: One of :data:`OS_NOTIFICATION_KIND_COPY`'s keys.
        session_display: The pre-composed conversation/tab display string
            (e.g. ``"Ops (Lab)"``). Only this and the category copy enter
            the command line.

    Returns:
        The exact argv list, or ``None`` when the platform or kind is
        unsupported (the caller skips with a debug log -- never an error).
    """
    category = OS_NOTIFICATION_KIND_COPY.get(kind)
    if category is None:
        return None
    body = _single_line(f"Agent in {session_display} {category}.")
    title = _single_line(OS_NOTIFICATION_APP_NAME)
    if platform == "darwin":
        script = (
            f'display notification "{_applescript_quote(body)}" '
            f'with title "{_applescript_quote(title)}"'
        )
        return ["osascript", "-e", script]
    if platform == "linux":
        return [
            "notify-send",
            "--app-name",
            OS_NOTIFICATION_APP_NAME,
            "--",
            title,
            body,
        ]
    return None


def _notification_env(platform: str) -> dict[str, str]:
    """Build the scrubbed subprocess environment.

    Only what the notifier binary itself needs survives: ``PATH`` to
    resolve it, and on Linux the D-Bus session address / XDG runtime dir
    ``notify-send`` requires to reach the notification daemon. Everything
    else the process environment carries -- provider keys included -- is
    dropped.
    """
    env: dict[str, str] = {
        "PATH": os.environ.get("PATH") or "/usr/local/bin:/usr/bin:/bin"
    }
    if platform == "linux":
        for key in ("DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR"):
            value = os.environ.get(key)
            if value:
                env[key] = value
    return env


def send_os_notification(
    kind: str,
    session_display: str,
    *,
    platform: str | None = None,
    settings: ConsoleOSNotificationSettings | None = None,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
) -> bool:
    """Fire one OS notification if permitted and supported.

    Args:
        kind: Outcome category (``run_finished`` / ``run_failed`` /
            ``needs_approval``).
        session_display: Conversation/tab display string for the body.
        platform: Platform override (tests); defaults to ``sys.platform``.
        settings: Normalized settings; when provided and ``kind`` is not
            allowed, nothing spawns. ``None`` means the caller already
            gated (the setting read is the caller's side of the seam).
        runner: ``subprocess.run``-shaped callable (tests).

    Returns:
        True iff a dispatcher subprocess was actually invoked -- not a
        claim that the OS displayed anything. Every failure (unsupported
        platform, missing binary, timeout) returns False with a debug log;
        nothing here raises.
    """
    resolved_platform = platform if platform is not None else sys.platform
    if settings is not None and not settings.allows(kind):
        return False
    argv = build_notification_argv(resolved_platform, kind, session_display)
    if argv is None:
        logger.debug(
            f"OS notification skipped: unsupported platform "
            f"{resolved_platform!r} for {kind!r}"
        )
        return False
    run = runner if runner is not None else subprocess.run
    try:
        run(
            argv,
            timeout=OS_NOTIFICATION_TIMEOUT_S,
            env=_notification_env(resolved_platform),
            capture_output=True,
            check=False,
            start_new_session=(resolved_platform != "win32"),
        )
        return True
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug(f"OS notification failed (kind={kind!r}): {exc}")
        return False
