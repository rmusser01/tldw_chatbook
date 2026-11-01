"""Private, process-lifetime scratch spaces for live Console chat sessions."""

from __future__ import annotations

import os
import secrets
import shutil
import stat
import tempfile
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from loguru import logger


@dataclass(frozen=True, slots=True)
class ConsoleScratchSnapshot:
    """Immutable capability identifying one live scratch-space generation."""

    root: Path = field(repr=False)
    token: str = field(repr=False)
    identity: tuple[int, int] = field(repr=False)


class ConsoleScratchSpaceUnavailable(RuntimeError):
    """Raised when a scratch-space capability is stale or no longer safe."""


class ConsoleScratchCleanupReceipt:
    """Consumer-held cleanup observation with one-shot host lifecycle callbacks.

    Failed attempts stay pending. Callbacks run outside manager/receipt locks;
    the manager retains this receipt only while its generation is pending.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._outcome: str | None = None
        self._callbacks: list[Callable[[str], None]] = []
        self._cleanup_callbacks: list[Callable[[], None]] = []

    @property
    def outcome(self) -> str | None:
        """Return deleted/already-missing/identity-changed, or None while pending."""
        with self._lock:
            return self._outcome

    def add_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Register retryable native cleanup, before root removal and without locks.

        A terminal receipt cannot run cleanup retroactively; late registrations
        are ignored and cannot establish deletion evidence.
        """
        with self._lock:
            if self._outcome is None:
                self._cleanup_callbacks.append(callback)

    def _prepare_cleanup(self) -> None:
        with self._lock:
            callbacks = tuple(self._cleanup_callbacks)
        for callback in callbacks:
            try:
                callback()
            except Exception:
                logger.warning("Console scratch pre-cleanup observer failed")

    def add_done_callback(self, callback: Callable[[str], None]) -> None:
        """Register host cleanup bookkeeping, outside any consumer owner lock."""
        with self._lock:
            outcome = self._outcome
            if outcome is None:
                self._callbacks.append(callback)
                return
        self._deliver(callback, outcome)

    @staticmethod
    def _deliver(callback: Callable[[str], None], outcome: str) -> None:
        try:
            callback(outcome)
        except Exception:
            logger.warning("Console scratch cleanup observer failed")

    def _finish(self, outcome: str) -> None:
        with self._lock:
            self._outcome = outcome
            self._cleanup_callbacks.clear()
            callbacks, self._callbacks = self._callbacks, []
        for callback in callbacks:
            self._deliver(callback, outcome)


@dataclass(slots=True)
class _ScratchRecord:
    session_id: str
    snapshot: ConsoleScratchSnapshot
    leases: int = 0
    tombstoned: bool = False
    cleanup_scheduled: bool = False
    cleanup_receipt: ConsoleScratchCleanupReceipt = field(
        default_factory=ConsoleScratchCleanupReceipt
    )


class ConsoleScratchSpaceManager:
    """Own isolated scratch directories and revoke them safely on chat close."""

    def __init__(self, *, temp_parent: Path | None = None) -> None:
        self._temp_parent = Path(temp_parent) if temp_parent is not None else None
        self._condition = threading.Condition(threading.RLock())
        self._by_session: dict[str, _ScratchRecord] = {}
        self._records: dict[str, _ScratchRecord] = {}
        self._cleanup_queue: deque[_ScratchRecord] = deque()
        self._cleanup_worker: threading.Thread | None = None
        self._disposed = False

    def snapshot(self, session_id: str) -> ConsoleScratchSnapshot:
        """Return the live capability for ``session_id``, allocating if needed.

        Args:
            session_id: Process-local identifier for the live Console session.

        Returns:
            The session's immutable scratch-space capability.

        Raises:
            ConsoleScratchSpaceUnavailable: If the manager has been disposed or
                allocation cannot establish a private directory.
        """

        normalized_session_id = str(session_id)
        if not normalized_session_id:
            raise ConsoleScratchSpaceUnavailable("A live Console session is required")

        with self._condition:
            if self._disposed:
                raise ConsoleScratchSpaceUnavailable(
                    "Console scratch-space manager is disposed"
                )
            existing = self._by_session.get(normalized_session_id)
            if existing is not None and not existing.tombstoned:
                return existing.snapshot

            try:
                raw_root = tempfile.mkdtemp(
                    prefix="tldw-console-",
                    dir=str(self._temp_parent)
                    if self._temp_parent is not None
                    else None,
                )
                root = Path(raw_root).resolve()
                os.chmod(root, 0o700)
                metadata = root.lstat()
                if root.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
                    raise OSError("scratch allocation did not create a directory")
            except OSError as exc:
                raise ConsoleScratchSpaceUnavailable(
                    "Could not allocate a private Console scratch space"
                ) from exc

            token = secrets.token_urlsafe(24)
            while token in self._records:
                token = secrets.token_urlsafe(24)
            snapshot = ConsoleScratchSnapshot(
                root=root,
                token=token,
                identity=(metadata.st_dev, metadata.st_ino),
            )
            record = _ScratchRecord(
                session_id=normalized_session_id,
                snapshot=snapshot,
            )
            self._by_session[normalized_session_id] = record
            self._records[token] = record
            return snapshot

    @contextmanager
    def lease(self, snapshot: ConsoleScratchSnapshot) -> Iterator[Path]:
        """Keep a validated scratch generation alive for one filesystem access.

        Args:
            snapshot: Capability previously issued by this manager.

        Yields:
            The validated private scratch root.

        Raises:
            ConsoleScratchSpaceUnavailable: If the capability is stale, revoked,
                missing, symlinked, or replaced on disk.
        """

        with self._condition:
            record = self._matching_record_locked(snapshot)
            if record is None or record.tombstoned:
                raise ConsoleScratchSpaceUnavailable(
                    "Console scratch space is no longer available"
                )
            if not self._identity_matches(snapshot):
                self._tombstone_locked(record)
                raise ConsoleScratchSpaceUnavailable(
                    "Console scratch space failed its filesystem identity check"
                )
            record.leases += 1

        try:
            yield snapshot.root
        finally:
            with self._condition:
                record.leases = max(0, record.leases - 1)
                if record.tombstoned and record.leases == 0:
                    self._schedule_cleanup_locked(record)
                self._condition.notify_all()

    def observe_cleanup(
        self, snapshot: ConsoleScratchSnapshot
    ) -> ConsoleScratchCleanupReceipt:
        """Observe terminal cleanup for a live generation without retaining history.

        Only ``deleted`` confirms this manager removed the owned directory.
        Missing/replaced roots revoke authority but do not prove byte deletion.
        Failed attempts remain pending so a later dispose retry can settle them.
        """
        with self.live_guard(snapshot):
            return self._records[snapshot.token].cleanup_receipt

    @contextmanager
    def live_guard(self, snapshot: ConsoleScratchSnapshot) -> Iterator[None]:
        """Serialize a short final publication against generation revocation.

        Hold an ordinary ``lease`` for the full operation; enter this guard only
        for its final commit. Close cannot tombstone between validation and commit.
        The body must not wait for another thread or invoke external callbacks.
        """
        with self._condition:
            record = self._matching_record_locked(snapshot)
            if record is None or record.tombstoned:
                raise ConsoleScratchSpaceUnavailable(
                    "Console scratch space is no longer available"
                )
            if not self._identity_matches(snapshot):
                self._tombstone_locked(record)
                raise ConsoleScratchSpaceUnavailable(
                    "Console scratch space failed its filesystem identity check"
                )
            yield

    def is_live(self, snapshot: ConsoleScratchSnapshot) -> bool:
        """Return whether a capability is current and safe to lease.

        Args:
            snapshot: Capability previously issued by this manager.

        Returns:
            ``True`` when the capability still identifies the current private
            directory generation and its filesystem identity is unchanged.
        """

        with self._condition:
            record = self._matching_record_locked(snapshot)
            if record is None or record.tombstoned:
                return False
            if self._identity_matches(snapshot):
                return True
            self._tombstone_locked(record)
            return False

    def _published_current(self, snapshot: ConsoleScratchSnapshot | None) -> bool:
        """Check issued custody without filesystem IO on the effect lane."""
        if not isinstance(snapshot, ConsoleScratchSnapshot) or not isinstance(
            snapshot.token, str
        ):
            return False
        with self._condition:
            record = self._matching_record_locked(snapshot)
            return not self._disposed and record is not None and not record.tombstoned

    def close(self, session_id: str) -> None:
        """Revoke one live session and schedule cleanup after its leases drain.

        Args:
            session_id: Process-local identifier for the live Console session.
        """

        with self._condition:
            record = self._by_session.pop(str(session_id), None)
            if record is None:
                return
            self._tombstone_locked(record)

    def tombstone_all(self) -> None:
        """Synchronously revoke every live scratch space without blocking on I/O."""

        with self._condition:
            self._disposed = True
            records = tuple(self._records.values())
            self._by_session.clear()
            for record in records:
                record.tombstoned = True
                if record.leases == 0:
                    self._schedule_cleanup_locked(record)
            self._condition.notify_all()

    def wait_for_cleanup(self, timeout_seconds: float) -> bool:
        """Wait up to ``timeout_seconds`` for every revoked record to retire.

        Args:
            timeout_seconds: Maximum number of seconds to wait.

        Returns:
            ``True`` when every record retired before the deadline.
        """

        deadline = time.monotonic() + max(0.0, timeout_seconds)
        with self._condition:
            while self._records:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def dispose(self, timeout_seconds: float = 2.0) -> bool:
        """Revoke all spaces, retry deferred cleanup, and wait for a bounded time.

        Args:
            timeout_seconds: Maximum number of seconds to wait for cleanup.

        Returns:
            ``True`` when every scratch record retired before the deadline.
        """

        self.tombstone_all()
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        with self._condition:
            for record in tuple(self._records.values()):
                if record.leases == 0 and not record.cleanup_scheduled:
                    self._schedule_cleanup_locked(record)
            while self._records:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def _matching_record_locked(
        self,
        snapshot: ConsoleScratchSnapshot,
    ) -> _ScratchRecord | None:
        record = self._records.get(snapshot.token)
        if record is None or record.snapshot != snapshot:
            return None
        return record

    @staticmethod
    def _identity_matches(snapshot: ConsoleScratchSnapshot) -> bool:
        try:
            metadata = snapshot.root.lstat()
        except OSError:
            return False
        return (
            not stat.S_ISLNK(metadata.st_mode)
            and stat.S_ISDIR(metadata.st_mode)
            and (metadata.st_dev, metadata.st_ino) == snapshot.identity
        )

    def _tombstone_locked(self, record: _ScratchRecord) -> None:
        current = self._by_session.get(record.session_id)
        if current is record:
            self._by_session.pop(record.session_id, None)
        record.tombstoned = True
        if record.leases == 0:
            self._schedule_cleanup_locked(record)
        self._condition.notify_all()

    def _schedule_cleanup_locked(self, record: _ScratchRecord) -> None:
        if record.cleanup_scheduled or record.snapshot.token not in self._records:
            return
        record.cleanup_scheduled = True
        self._cleanup_queue.append(record)
        if self._cleanup_worker is None or not self._cleanup_worker.is_alive():
            self._cleanup_worker = threading.Thread(
                target=self._cleanup_worker_main,
                name="console-scratch-cleanup",
                daemon=True,
            )
            self._cleanup_worker.start()

    def _cleanup_worker_main(self) -> None:
        while True:
            with self._condition:
                if not self._cleanup_queue:
                    self._cleanup_worker = None
                    self._condition.notify_all()
                    return
                record = self._cleanup_queue.popleft()

            record.cleanup_receipt._prepare_cleanup()
            cleaned, category = self._cleanup_record(record)
            if cleaned:
                record.cleanup_receipt._finish(category)

            with self._condition:
                record.cleanup_scheduled = False
                if cleaned:
                    current = self._records.get(record.snapshot.token)
                    if current is record:
                        self._records.pop(record.snapshot.token, None)
                else:
                    logger.warning(
                        "Console scratch cleanup deferred category={}",
                        category,
                    )
                self._condition.notify_all()

    @staticmethod
    def _cleanup_record(record: _ScratchRecord) -> tuple[bool, str]:
        snapshot = record.snapshot
        try:
            metadata = snapshot.root.lstat()
        except FileNotFoundError:
            return True, "already-missing"
        except OSError:
            return False, "identity-check-failed"

        identity = (metadata.st_dev, metadata.st_ino)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or identity != snapshot.identity
        ):
            return True, "identity-changed"

        try:
            shutil.rmtree(snapshot.root)
        except FileNotFoundError:
            return True, "already-missing"
        except OSError:
            return False, "delete-failed"
        return True, "deleted"
