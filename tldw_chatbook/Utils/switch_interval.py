"""Shared, ref-counted scopes that shorten the interpreter's switch interval.

A thread that makes thousands of GIL-releasing syscalls (a config write's
storage admission, a Console preparation read) waits out a busy thread's
switch interval after each one, so it can run ~70x slower beside CPU-bound
Python (TASK-38987). Every scope goes through this one owner, so overlapping
scopes, from one caller or several, always end at the interval that was in
force before the first began.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager

_LOCK = threading.Lock()
#: Limits of the scopes now open (a multiset, as a list).
_open_limits: list[float] = []
#: ``(interval before the first scope, interval this module last installed)``.
_saved: tuple[float, float] | None = None


def _install_locked() -> None:
    """Install the smallest open limit, or restore once no scope is open.

    An interval other code set while scopes were open is left alone.
    """
    global _saved
    assert _saved is not None
    previous, installed = _saved
    if sys.getswitchinterval() == installed:
        sys.setswitchinterval(min([previous, *_open_limits]))
        installed = sys.getswitchinterval()
    _saved = (previous, installed) if _open_limits else None


_local = threading.local()


def _held() -> list[float]:
    """The limits of the scopes the calling thread has open and not released."""
    held = getattr(_local, "limits", None)
    if held is None:
        held = _local.limits = []
    return held


def _open_locked(limits: list[float]) -> None:
    global _saved
    if not _open_limits:
        current = sys.getswitchinterval()
        _saved = (current, current)
    _open_limits.extend(limits)
    _install_locked()


def _close_locked(limits: list[float]) -> None:
    for limit in limits:
        _open_limits.remove(limit)
    _install_locked()


@contextmanager
def switch_interval_at_most(limit: float) -> Iterator[None]:
    """Keep ``sys.getswitchinterval()`` at or below ``limit`` while open.

    Args:
        limit: The longest interval, in seconds, the scope allows. A shorter
            interval already in force is kept.

    Yields:
        None. On exit, normal or raising, the interval returns to the one in
        force before the first open scope, unless other code changed it.
    """
    with _LOCK:
        _open_locked([limit])
    _held().append(limit)
    try:
        yield
    finally:
        _held().remove(limit)
        with _LOCK:
            _close_locked([limit])


@contextmanager
def switch_interval_released() -> Iterator[None]:
    """Lift the calling thread's scopes while it waits on another process.

    The wait gains nothing from a short interval, and every other thread of
    the process would pay for it (review r12s-rev1 M4: CPU-bound threads ran
    28-47% slower while another process held the config lock).

    Yields:
        None, with this thread's scopes closed; they reopen on exit.
    """
    held = _held()
    released = list(held)  # this release's own snapshot (re-entrant)
    if not released:
        yield
        return
    del held[:]  # while released, the thread holds no open scope
    with _LOCK:
        _close_locked(released)
    try:
        yield
    finally:
        with _LOCK:
            _open_locked(released)
        held.extend(released)
