"""Advisory per-profile instance lock.

Detection only — a second instance gets a warning toast, never a lock-out
(the owner runs concurrent instances deliberately; permission/settings
stores are last-write-wins by design). The OS lock, not the file's
existence, is the liveness signal: locks vanish with the process, so stale
files never false-positive. The lock file is deliberately never unlinked —
unlinking races a third instance onto a fresh inode and splits the lock.

This module is intentionally dependency-light (stdlib + portalocker +
loguru only) and takes a plain ``Path`` rather than importing anything from
``tldw_chatbook.config`` or ``tldw_chatbook.app`` -- that keeps its unit
tests hermetic against a ``tmp_path`` with no risk of touching a live user
config.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import portalocker
from portalocker import LockFlags
from loguru import logger

_LOCK_FILENAME = ".instance.lock"


@dataclass
class InstanceLockStatus:
    acquired: bool
    handle: BinaryIO | None = None
    written_pid: int | None = None
    holder_pid: int | None = None
    holder_since: str | None = None


def acquire_profile_instance_lock(user_data_dir: Path) -> InstanceLockStatus:
    """Try to take an advisory exclusive lock beside the profile's data dir.

    NEVER blocks (uses ``LockFlags.NON_BLOCKING``), NEVER raises, and NEVER
    prevents boot -- any unexpected error is swallowed and reported as
    ``acquired=True`` so a broken lock mechanism can never produce a false
    "second instance" warning.

    Returns:
        An ``InstanceLockStatus``. When ``acquired`` is True and ``handle``
        is not None, the caller must keep the handle referenced for the
        life of the process -- closing (or garbage-collecting) it releases
        the OS lock immediately.
    """
    lock_path = Path(user_data_dir) / _LOCK_FILENAME
    try:
        handle = lock_path.open("a+b")
    except OSError as exc:
        logger.debug("instance lock unavailable ({}): {}", type(exc).__name__, exc)
        return InstanceLockStatus(acquired=True)

    try:
        portalocker.lock(handle, LockFlags.EXCLUSIVE | LockFlags.NON_BLOCKING)
    except portalocker.exceptions.AlreadyLocked:
        # Genuine contention only (POSIX locker: EACCES/EAGAIN). Mirrors
        # Model_Artifacts/leases.py:250-262, which draws the exact same
        # line for the exact same reason: portalocker.exceptions.LockException
        # (the broader class -- ENOLCK, EOPNOTSUPP, EBADF, NFS EOFError, ...)
        # means the locking MECHANISM failed, not that someone else holds the
        # lock, so it must fall through to the generic `except Exception`
        # below and report `acquired=True`. Catching the broader
        # BaseLockException here instead would fold "flock unsupported on
        # this filesystem" into "someone else has it" -- a permanent false
        # "Profile already open" warning on every boot for anyone on such a
        # filesystem, which is exactly the false warning this module exists
        # to avoid.
        holder_pid, holder_since = _read_holder(lock_path)
        handle.close()
        return InstanceLockStatus(
            acquired=False, holder_pid=holder_pid, holder_since=holder_since
        )
    except Exception as exc:
        # Any other unexpected failure -- portalocker.exceptions.LockException
        # (locking mechanism broken/unsupported), permissions weirdness,
        # platform quirks, ... -- must never block boot or produce a false
        # warning.
        logger.debug("instance lock error ({}): {}", type(exc).__name__, exc)
        handle.close()
        return InstanceLockStatus(acquired=True)

    pid = os.getpid()
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(
            f"{pid}\n{datetime.now(timezone.utc).isoformat()}\n".encode("utf-8")
        )
        handle.flush()
    except OSError:
        pass  # body is informational; the lock itself is the signal
    return InstanceLockStatus(acquired=True, handle=handle, written_pid=pid)


def _read_holder(lock_path: Path) -> tuple[int | None, str | None]:
    try:
        lines = lock_path.read_text(encoding="utf-8", errors="replace").splitlines()
        pid = int(lines[0]) if lines and lines[0].strip().isdigit() else None
        since = lines[1].strip() if len(lines) > 1 else None
        return pid, since
    except OSError:
        return None, None
