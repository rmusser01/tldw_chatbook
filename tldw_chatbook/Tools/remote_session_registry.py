"""Per-run registry of SSH session workers: one per (run key, binding)."""

from __future__ import annotations

import threading
import time
from typing import Callable

from loguru import logger

#: How many closed run keys stay tombstoned (R12). A closed key never
#: reopens a session; the oldest tombstone is evicted past this bound.
_CLOSED_KEYS_MAX = 1024

#: How long run end / app exit waits for session closes (all in parallel).
#: A close still running after this finishes on its daemon thread; at app
#: exit the master's ``-O exit`` ends its channel anyway.
_CLOSE_JOIN_S = 5.0

from tldw_chatbook.Tools.remote_session_worker import RemoteSessionWorker, SessionStartError
from tldw_chatbook.Tools.remote_workspace_transport import TransportFailureKind


def _close_workers(workers: list[RemoteSessionWorker], *, wait: float | None) -> None:
    """Close ``workers`` concurrently, one daemon thread each.

    Args:
        workers: Sessions already removed from the registry.
        wait: Total seconds to wait for all closes; ``None`` returns at once
            (the idle reaper: a close never lands on the calling tool call).
    """
    threads = [
        threading.Thread(target=worker.close, name="ssh-session-close", daemon=True)
        for worker in workers
    ]
    for thread in threads:
        thread.start()
    if wait is None:
        return
    deadline = time.monotonic() + wait
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))


class RemoteSessionRegistry:
    """Thread-safe owner of live sessions and per-run disabled keys."""

    def __init__(self) -> None:
        self._lock = threading.Lock()  # guards the dicts only; never held across a start
        self._key_locks: dict[tuple[str, str], threading.Lock] = {}
        self._sessions: dict[tuple[str, str], RemoteSessionWorker] = {}
        self._disabled: set[tuple[str, str]] = set()
        self._restarted: set[tuple[str, str]] = set()
        #: Last transport-class start failure per key, with the monotonic
        #: time it happened: callers that queued behind that start share it
        #: instead of each paying another connect timeout (TASK-33400).
        self._start_failures: dict[tuple[str, str], tuple[float, SessionStartError]] = {}
        #: Keys whose session start already hit a mux failure this run: the
        #: first costs one one-shot call, a second disables the key (TASK-33402).
        self._mux_failed: set[tuple[str, str]] = set()
        # R12: insertion-ordered tombstones of closed run keys (bounded).
        self._closed_keys: dict[str, None] = {}
        #: Set once by close_all (app shutdown): every later acquire is one-shot.
        self._shutdown = False

    def acquire(self, key: tuple[str, str], create: Callable[[], RemoteSessionWorker]) -> RemoteSessionWorker | None:
        """Return the live session for ``key``, starting one if needed.

        A session start can take seconds (ssh connect, handshake), so it runs
        under a per-key lock: concurrent callers for the same binding wait for
        one start; other bindings and runs are never blocked.

        After :meth:`close_all` it returns ``None`` (one-shot) -- also for a
        caller that was already waiting on the per-key lock, and a start that
        completes after it is closed rather than kept.

        Raises:
            SessionStartError: transport-class start failure (caller records it).
        """
        entered = time.monotonic()
        with self._lock:
            if self._shutdown:
                return None
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            with self._lock:
                # A closed run key is one-shot for good: surviving
                # sub-agents and Stop stragglers never reopen an unowned
                # session.
                if self._shutdown or key in self._disabled or key[0] in self._closed_keys:
                    return None
                worker = self._sessions.get(key)
                if worker is not None and worker.alive:
                    return worker
                failed = self._start_failures.get(key)
            if failed is not None and failed[0] >= entered:
                # This caller queued behind a start that failed
                # transport-class: share that failure (a fresh exception per
                # thread) rather than start again against the same dead host.
                raise SessionStartError(failed[1].transport, failed[1].failure, str(failed[1]))
            # ended_cleanly() may wait for the reap: outside the global lock.
            if worker is not None and not worker.ended_cleanly():
                # Died mid-run: one restart, then fall back. A clean end
                # (R10: host idle-exit, exit 0) is recreated for free.
                with self._lock:
                    self._sessions.pop(key, None)
                    if key in self._restarted:
                        self._disabled.add(key)
                        logger.debug(
                            "ssh session worker died again; using one-shot calls for this run"
                        )
                        return None
                    self._restarted.add(key)
                logger.debug("ssh session worker died; restarting it once for this run")
            worker = create()
            try:
                worker.start()
            except SessionStartError as error:
                if error.failure is not None and error.failure.kind is TransportFailureKind.MUX_ERROR:
                    # A stale control socket: classifying it already restarted
                    # the master, and nothing was sent, so this call runs
                    # one-shot and the next call tries a session again. A
                    # second mux start failure in the run disables the key.
                    with self._lock:
                        repeat = key in self._mux_failed
                        self._mux_failed.add(key)
                        if repeat:
                            self._disabled.add(key)
                    logger.info(
                        "ssh session start hit a stale control socket; using one-shot calls "
                        + ("for this run" if repeat else "for this call")
                    )
                    return None
                if not error.transport:
                    with self._lock:
                        self._disabled.add(key)
                    # Only static text: a typed failure's reason may quote a
                    # stderr line (mux errors can name the ControlPath).
                    cause = error.failure.kind.value if error.failure else str(error)
                    logger.info(
                        f"ssh session worker disabled for this run; using one-shot calls: {cause}"
                    )
                    return None
                kind = error.failure.kind.value if error.failure else "unknown"
                logger.debug(f"ssh session worker start failed (transport): {kind}")
                with self._lock:
                    self._start_failures[key] = (time.monotonic(), error)
                raise
            with self._lock:
                self._start_failures.pop(key, None)
                closed_meanwhile = self._shutdown or key[0] in self._closed_keys
                if not closed_meanwhile:
                    self._sessions[key] = worker
            if closed_meanwhile:
                # The run (or the app) ended while this start was in flight:
                # nobody owns it.
                worker.close()
                return None
            return worker

    def close_key(self, session_key: str) -> None:
        """Close every session under one run key and tombstone the key."""
        with self._lock:
            self._closed_keys[session_key] = None
            while len(self._closed_keys) > _CLOSED_KEYS_MAX:
                del self._closed_keys[next(iter(self._closed_keys))]
            keys = [k for k in self._sessions if k[0] == session_key]
            workers = [self._sessions.pop(k) for k in keys]
            self._disabled = {k for k in self._disabled if k[0] != session_key}
            self._restarted = {k for k in self._restarted if k[0] != session_key}
            self._start_failures = {
                k: v for k, v in self._start_failures.items() if k[0] != session_key
            }
            self._mux_failed = {k for k in self._mux_failed if k[0] != session_key}
            self._key_locks = {k: v for k, v in self._key_locks.items() if k[0] != session_key}
        _close_workers(workers, wait=_CLOSE_JOIN_S)

    def close_all(self) -> None:
        """Close every session (app shutdown); later acquires go one-shot."""
        with self._lock:
            self._shutdown = True
            workers = list(self._sessions.values()); self._sessions.clear()
            self._disabled.clear(); self._restarted.clear(); self._key_locks.clear()
            self._start_failures.clear()
            self._mux_failed.clear()
        _close_workers(workers, wait=_CLOSE_JOIN_S)

    def reap_idle(self, now: float, idle_s: float) -> None:
        """Close sessions idle for at least ``idle_s`` (a later call starts a fresh one)."""
        with self._lock:
            stale = [k for k, w in self._sessions.items()
                     if w.idle_since is not None and now - w.idle_since >= idle_s]
            workers = [self._sessions.pop(k) for k in stale]
        _close_workers(workers, wait=None)


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
    """Close one run's sessions (run end) and tombstone its key.

    Always reaches the registry (creating it if needed) so the tombstone is
    recorded even when the run never opened a session: a straggler call
    after run end must not open one either.
    """
    get_session_registry().close_key(session_key)


def close_all_remote_sessions() -> None:
    """Close every session (app exit) and shut the registry down.

    Always reaches the registry (creating it if needed), as
    :func:`close_remote_sessions` does, so a straggler call after app exit
    goes one-shot instead of opening a session.
    """
    get_session_registry().close_all()
