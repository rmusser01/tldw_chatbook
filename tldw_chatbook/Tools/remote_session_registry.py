"""Per-run registry of SSH session workers: one per (run key, binding)."""

from __future__ import annotations

import threading
from typing import Callable

from tldw_chatbook.Tools.remote_session_worker import RemoteSessionWorker, SessionStartError


class RemoteSessionRegistry:
    """Thread-safe owner of live sessions and per-run disabled keys."""

    def __init__(self) -> None:
        self._lock = threading.Lock()  # guards the dicts only; never held across a start
        self._key_locks: dict[tuple[str, str], threading.Lock] = {}
        self._sessions: dict[tuple[str, str], RemoteSessionWorker] = {}
        self._disabled: set[tuple[str, str]] = set()
        self._restarted: set[tuple[str, str]] = set()

    def acquire(self, key: tuple[str, str], create: Callable[[], RemoteSessionWorker]) -> RemoteSessionWorker | None:
        """Return the live session for ``key``, starting one if needed.

        A session start can take seconds (ssh connect, handshake), so it runs
        under a per-key lock: concurrent callers for the same binding wait for
        one start; other bindings and runs are never blocked.

        Raises:
            SessionStartError: transport-class start failure (caller records it).
        """
        with self._lock:
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            with self._lock:
                if key in self._disabled:
                    return None
                worker = self._sessions.get(key)
                if worker is not None and worker.alive:
                    return worker
                if worker is not None:  # died mid-run: one restart, then fall back
                    self._sessions.pop(key, None)
                    if key in self._restarted:
                        self._disabled.add(key)
                        return None
                    self._restarted.add(key)
            worker = create()
            try:
                worker.start()
            except SessionStartError as error:
                if not error.transport:
                    with self._lock:
                        self._disabled.add(key)
                    return None
                raise
            with self._lock:
                self._sessions[key] = worker
            return worker

    def close_key(self, session_key: str) -> None:
        """Close every session under one run key and forget its failure state."""
        with self._lock:
            keys = [k for k in self._sessions if k[0] == session_key]
            workers = [self._sessions.pop(k) for k in keys]
            self._disabled = {k for k in self._disabled if k[0] != session_key}
            self._restarted = {k for k in self._restarted if k[0] != session_key}
            self._key_locks = {k: v for k, v in self._key_locks.items() if k[0] != session_key}
        for worker in workers:
            worker.close()

    def close_all(self) -> None:
        """Close every session (app shutdown)."""
        with self._lock:
            workers = list(self._sessions.values()); self._sessions.clear()
            self._disabled.clear(); self._restarted.clear()
        for worker in workers:
            worker.close()

    def reap_idle(self, now: float, idle_s: float) -> None:
        """Close sessions idle for at least ``idle_s`` (a later call starts a fresh one)."""
        with self._lock:
            stale = [k for k, w in self._sessions.items()
                     if w.idle_since is not None and now - w.idle_since >= idle_s]
            workers = [self._sessions.pop(k) for k in stale]
        for worker in workers:
            worker.close()


_REGISTRY: RemoteSessionRegistry | None = None
_REGISTRY_LOCK = threading.Lock()


def get_session_registry() -> RemoteSessionRegistry:
    """The process-wide registry, created on first use."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            _REGISTRY = RemoteSessionRegistry()
        return _REGISTRY


def close_remote_sessions(session_key: str) -> None:
    """Close one run's sessions (run end); a no-op before any session existed."""
    if _REGISTRY is not None:
        _REGISTRY.close_key(session_key)


def close_all_remote_sessions() -> None:
    """Close every session (app exit); a no-op before any session existed."""
    if _REGISTRY is not None:
        _REGISTRY.close_all()
