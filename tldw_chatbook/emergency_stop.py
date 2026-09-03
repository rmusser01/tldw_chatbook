# emergency_stop.py
"""TASK-26004: a single global emergency-stop sentinel.

One durable switch, several readers (agent sends, scheduled dispatch). It
stops NEW work from STARTING while leaving in-flight runs untouched, it
survives a restart (a small JSON file), and it is deliberately FAIL-SAFE:
if the read itself errors, the system treats it as STOPPED rather than
proceeding -- for an emergency stop, halting on doubt is the safe direction.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass(frozen=True, slots=True)
class EmergencyStopState:
    """Whether new work is halted, and why."""

    active: bool
    reason: str = ""

    def visible_copy(self) -> str:
        """User-facing copy naming the stop and how to clear it (AC#5)."""
        why = f" ({self.reason})" if self.reason else ""
        return (
            f"Emergency stop is active{why}: new runs and scheduled "
            "dispatches are held. Clear it to resume."
        )


def read_emergency_stop(path: Path) -> EmergencyStopState:
    """Read the sentinel. FAIL-SAFE: any read/parse error => STOPPED (AC#4)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return EmergencyStopState(active=False)
    except OSError as exc:
        logger.warning(
            f"emergency-stop read failed ({exc!r}); failing SAFE to stopped"
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
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"active": state.active, "reason": state.reason})
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".estop-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def default_emergency_stop_path() -> Path:
    from tldw_chatbook.config import get_user_data_dir

    return get_user_data_dir() / "emergency_stop.json"
