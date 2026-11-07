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
    global _saved
    with _LOCK:
        if not _open_limits:
            current = sys.getswitchinterval()
            _saved = (current, current)
        _open_limits.append(limit)
        _install_locked()
    try:
        yield
    finally:
        with _LOCK:
            _open_limits.remove(limit)
            _install_locked()
