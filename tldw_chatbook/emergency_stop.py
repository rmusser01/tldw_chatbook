# emergency_stop.py
"""TASK-26004: a single global emergency-stop sentinel.

One durable switch, several readers (agent sends, scheduled dispatch, and
since TASK-33620.1.12 every agent tool call, hook-v2 action and /research
step). It stops NEW work from STARTING, and an in-flight run's tool calls,
hook actions and research steps are refused while it holds (a step already
executing finishes). It survives a restart (a small JSON file), and it is
deliberately FAIL-SAFE:
if the read itself errors, the system treats it as STOPPED rather than
proceeding -- for an emergency stop, halting on doubt is the safe direction.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .Utils.atomic_file_ops import atomic_write_text

from loguru import logger


#: The exact command that lifts the stop. TASK-33620.1: the copy used to end
#: "Clear it to resume." and never said how.
EMERGENCY_STOP_CLEAR_COMMAND = "/emergency-stop clear"


@dataclass(frozen=True, slots=True)
class EmergencyStopState:
    """Whether new work is halted, and why."""

    active: bool
    reason: str = ""

    def visible_copy(self) -> str:
        """User-facing copy naming the stop and how to clear it (AC#5)."""
        why = f" ({self.reason})" if self.reason else ""
        # TASK-33620.1.12: a running run's tool calls, hook actions and
        # /research are held too.
        return (
            f"Emergency stop is active{why}: new runs, tool calls, hook actions, "
            f"research and scheduled dispatches are held. Run "
            f"{EMERGENCY_STOP_CLEAR_COMMAND} to resume."
        )


def read_emergency_stop(path: Path) -> EmergencyStopState:
    """Read the sentinel. FAIL-SAFE: any read/parse error => STOPPED (AC#4)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return EmergencyStopState(active=False)
    # TASK-33620.1.12: a file that is not UTF-8 is unreadable too; it used to
    # raise out of the send gate and the Console's still_held check.
    # The error type only: a UnicodeDecodeError's repr carries the file's
    # bytes, which hold the reason the user typed.
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning(
            f"emergency-stop read failed ({type(exc).__name__}); failing SAFE to stopped"
        )
        return EmergencyStopState(active=True, reason="stop state unreadable")
    try:
        raw = json.loads(text)
        if not isinstance(raw, dict):
            raise ValueError("not an object")
    except ValueError as exc:
        logger.warning(
            f"emergency-stop file corrupt ({exc!r}); failing SAFE to stopped"
        )
        return EmergencyStopState(active=True, reason="stop state corrupt")
    return EmergencyStopState(
        active=bool(raw.get("active", False)),
        reason=str(raw.get("reason", "") or ""),
    )


#: TASK-33620.1.12: the reason a hold carries when the stop's state could not
#: be read at all (its location could not be resolved).
STATE_UNAVAILABLE_REASON = "stop state unavailable"


def stop_state_unavailable() -> EmergencyStopState:
    """The fail-safe's state when the stop could not be read at all: held.

    Every reader that fails closed this way returns this, so the test suite
    can see each such hold (Tests/conftest.py fails a test that hits it unless
    it is marked ``emergency_stop_unavailable``): a silent hold once let a
    test pass for the wrong reason.
    """
    return EmergencyStopState(active=True, reason=STATE_UNAVAILABLE_REASON)


#: TASK-33620.1.12: the copy for work held because the stop's state could not
#: be read at all. ``/emergency-stop clear`` writes to the same unreachable
#: folder, so it is not offered (an agent tool call's refusal and a research
#: run's message say the same in their own words).
STATE_UNAVAILABLE_COPY = (
    "The emergency stop's state couldn't be read, so new runs, tool calls, "
    "hook actions and research are held. Chatbook could not reach its data "
    "folder (a backup or restore may be running, or the folder's permissions "
    "changed); try again once Chatbook is ready."
)


def held_copy(state: EmergencyStopState) -> str:
    """The copy for work ``state`` holds.

    Args:
        state: An active stop state.

    Returns:
        ``STATE_UNAVAILABLE_COPY`` when the stop's state could not be read at
        all, else the state's visible copy (an unreadable or corrupt file
        keeps it: clearing rewrites that file).
    """
    if state.reason == STATE_UNAVAILABLE_REASON:
        return STATE_UNAVAILABLE_COPY
    return state.visible_copy()


#: TASK-33620.1.12: the resolved stop path and the inputs it was resolved
#: under. Resolving runs the guarded user-data-dir handshake (about 1 ms, and
#: the config lock), and every agent tool call, hook action and research step
#: now reads the stop. The path is reused while the resolver, the config's
#: identity (its generation and effective path) and HOME are unchanged: a
#: config change that moves the data folder resolves again, or a stop set at
#: the new location would be missed. A failed resolution is never kept.
_RESOLVED_PATH: tuple[tuple, Path] | None = None


def _resolved_stop_path() -> Path:
    """The stop's path, resolved once per set of inputs (see ``_RESOLVED_PATH``).

    Raises:
        Exception: Whatever the resolution raises; nothing is cached then.
    """
    global _RESOLVED_PATH
    resolver = default_emergency_stop_path
    try:
        from tldw_chatbook.config import current_config_identity

        key = (resolver, current_config_identity(), os.environ.get("HOME"))
    except Exception:  # noqa: BLE001 -- no key: resolve without caching
        return resolver()
    cached = _RESOLVED_PATH
    if cached is not None and cached[0] == key:
        return cached[1]
    path = resolver()
    _RESOLVED_PATH = (key, path)
    return path


def current_emergency_stop() -> EmergencyStopState:
    """Resolve the stop's default path and read it, FAIL-SAFE.

    Blocking (a guarded config read on a miss, then a file read): call it off
    the UI loop. A path that cannot be resolved holds, as an unreadable file
    does.
    """
    try:
        path = _resolved_stop_path()
    except Exception:  # noqa: BLE001 -- doubt holds (AC#4)
        return stop_state_unavailable()
    return read_emergency_stop(path)


# emergency_stop_state is the readable alias used by surfaces that want the
# whole state (copy + reason); is_emergency_stopped is the hot boolean.
emergency_stop_state = read_emergency_stop


def is_emergency_stopped(path: Path) -> bool:
    """True when new work must not start (includes the fail-safe case)."""
    return read_emergency_stop(path).active


def set_emergency_stop(path: Path, *, reason: str = "") -> None:
    """Activate the stop durably (atomic write)."""
    _write(path, EmergencyStopState(active=True, reason=reason))


def clear_emergency_stop(path: Path) -> None:
    """Deactivate the stop; new work resumes without a restart (AC#6)."""
    _write(path, EmergencyStopState(active=False, reason=""))


def _write(path: Path, state: EmergencyStopState) -> None:
    path = Path(path)
    payload = json.dumps({"active": state.active, "reason": state.reason})
    # task-32808.5: the previous temp-file + os.replace skipped fsync, so this
    # "durable" safety switch (its own docstring) could be lost or truncated on
    # power loss between the write and the OS flush. Route through the shared
    # atomic-write helper, which fsyncs the temp inode before the atomic rename
    # (and creates the parent directory).
    atomic_write_text(path, payload)


def default_emergency_stop_path() -> Path:
    from tldw_chatbook.config import get_user_data_dir

    return get_user_data_dir() / "emergency_stop.json"
