"""One off-loop recovery writer with bounded coalescing and honest status."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

from tldw_chatbook.Chunking.lab_models import LabSession
from tldw_chatbook.DB.Chunking_Lab_DB import (
    CheckpointConflict,
    CheckpointStore,
    CheckpointToken,
    RecoverySchemaError,
)

_DEBOUNCE_SECONDS = 0.3
_MAX_WAIT_SECONDS = 1.0


@dataclass(frozen=True)
class SaveStatus:
    """A content-free UI snapshot; saved means the latest revision committed."""

    state: str
    acknowledged: CheckpointToken | None
    latest_revision: int
    error: str | None


class AutosaveWriter:
    """Serialize I/O on one lazily started thread; call API from one event loop.

    ``await load()`` is the initial off-thread recovery seam. It returns the
    store's session/token or None, and raises load failures without assuming
    overwrite authority. ``recovery_warning`` exposes previous-checkpoint
    fallback after loading. Submit retains immutable session references in one
    pending slot; hashing, copying, validation, and SQLite run only on the worker.
    Flush explicitly retries the latest draft after a storage failure. Conflicts
    and failed initial reads require a new writer after deliberate user recovery.
    To retry initial load, close this writer (close still releases resources when
    it re-raises the load error), construct a fresh store/writer, and await load
    before creating or submitting a new session. Never reuse an assumed empty
    session as authority over a checkpoint another instance just created.
    """

    def __init__(self, store: CheckpointStore):
        self._store = store
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="lab-autosave"
        )
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._runner: asyncio.Task | None = None
        self._pending: LabSession | None = None
        self._latest: LabSession | None = None
        self._first_pending: float | None = None
        self._deadline = 0.0
        self._immediate = False
        self._token: CheckpointToken | None = None
        self._loaded = False
        self._initial: tuple[LabSession, CheckpointToken] | None = None
        self._error: Exception | None = None
        self._error_revision = -1
        self._blocked: Exception | None = None
        self._clearing = False
        self._closing = False
        self._closed = False
        self._status = SaveStatus("idle", None, 0, None)
        self.recovery_warning: str | None = None

    @property
    def status(self) -> SaveStatus:
        """Return a copied immutable status, never mutable worker bookkeeping."""
        return replace(self._status)

    async def _io(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, partial(operation, *args, **kwargs)
        )

    def _failed(self, exc: Exception, revision: int, *, block: bool = False) -> None:
        self._error = exc
        self._error_revision = max(self._error_revision, revision)
        if block or isinstance(exc, (CheckpointConflict, RecoverySchemaError)):
            self._blocked = exc
        # Do not expose exception text: SQLite/path/validation errors can carry
        # source bytes or private paths. The raised exception remains actionable
        # by the owning controller without being a UI/log payload.
        message = (
            "Recovery changed in another instance. Reload or export your draft."
            if isinstance(exc, CheckpointConflict)
            else "Local recovery could not be saved or loaded. Retry or export your draft."
        )
        self._status = replace(
            self._status,
            state="conflict" if isinstance(exc, CheckpointConflict) else "failed",
            error=message,
        )

    async def _ensure_loaded(self) -> None:
        if self._blocked is not None:
            raise self._blocked
        if self._loaded:
            return
        try:
            self._initial = await self._io(self._store.load)
            self.recovery_warning = self._store.recovery_warning
            if self._initial is not None:
                session, self._token = self._initial
                if self._latest is None:
                    self._latest = session
                self._status = replace(
                    self._status,
                    acknowledged=self._token,
                    latest_revision=self._latest.revision,
                )
                if self._latest.epoch != session.epoch:
                    raise CheckpointConflict(
                        "Submitted session does not own recovered epoch"
                    )
                if self._pending is None and session.revision != self._token.revision:
                    self._pending = session
                    self._first_pending = time.monotonic()
                    self._deadline = self._first_pending
                state = "saving" if self._pending is not None else "saved"
                self._status = replace(self._status, state=state)
            self._loaded = True
        except Exception as exc:
            self._failed(exc, self._status.latest_revision, block=True)
            raise

    async def load(self) -> tuple[LabSession, CheckpointToken] | None:
        """Return initial recovery off the UI loop; this is not a reload action."""
        if self._closed or self._closing:
            raise RuntimeError("Autosave writer is closed")

        async def operation():
            async with self._lock:
                await self._ensure_loaded()
                return self._initial

        result = await asyncio.shield(operation())
        if (
            self._pending is not None
            and not self._closing
            and not self._clearing
            and (self._runner is None or self._runner.done())
        ):
            self._runner = asyncio.create_task(self._run())
        return result

    def submit(self, session: LabSession, *, immediate: bool = False) -> None:
        """Coalesce one immutable latest session; critical transitions skip debounce."""
        if self._closed or self._closing:
            raise RuntimeError("Autosave writer is closed")
        if self._clearing or session.profile_key != self._store.profile_key:
            raise CheckpointConflict(
                "Session cannot write during Clear or across profiles"
            )
        if self._latest is not None:
            if session.epoch != self._latest.epoch:
                raise CheckpointConflict("Session belongs to a replaced epoch")
            if session.revision < self._latest.revision:
                return
        now = time.monotonic()
        self._latest = self._pending = session
        if self._first_pending is None:
            self._first_pending = now
        self._immediate = self._immediate or immediate
        self._deadline = (
            now
            if self._immediate
            else min(now + _DEBOUNCE_SECONDS, self._first_pending + _MAX_WAIT_SECONDS)
        )
        self._status = replace(
            self._status,
            latest_revision=session.revision,
            state=self._status.state if self._error is not None else "saving",
        )
        self._wake.set()
        if (
            self._blocked is None
            and (self._error is None or immediate)
            and (self._runner is None or self._runner.done())
        ):
            self._runner = asyncio.create_task(self._run())

    async def _save_once(self) -> None:
        async with self._lock:
            await self._ensure_loaded()
            if self._pending is None or self._clearing:
                return
            session = self._pending
            self._pending = None
            self._first_pending = None
            self._immediate = False
            try:
                token = await self._io(self._store.save, session, expected=self._token)
            except Exception as exc:
                if self._pending is None:
                    self._pending = session
                self._failed(exc, session.revision)
                raise
            self._token = token
            self._status = replace(self._status, acknowledged=token)
            latest = self._latest
            if (
                not self._clearing
                and latest is not None
                and (token.profile_key, token.epoch, token.revision)
                == (latest.profile_key, latest.epoch, latest.revision)
                and token.revision >= self._error_revision
            ):
                self._error = None
                self._error_revision = -1
                self._status = replace(self._status, state="saved", error=None)

    async def _run(self) -> None:
        while self._pending is not None and not self._closing and not self._clearing:
            self._wake.clear()
            delay = max(0.0, self._deadline - time.monotonic())
            if delay:
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=delay)
                    continue
                except TimeoutError:
                    pass
            try:
                await self._save_once()
            except Exception:  # noqa: BLE001 - _save_once records all storage failures for the UI.
                # UI status carries the failure; explicit flush/Retry raises it.
                return

    async def _flush(self) -> CheckpointToken:
        await self._ensure_loaded_locked()
        while self._pending is not None:
            if self._clearing:
                raise CheckpointConflict("Clear is in progress")
            await self._save_once()
        async with self._lock:
            if self._blocked is not None:
                raise self._blocked
            if self._error is not None:
                raise self._error
            if self._token is None:
                raise ValueError("No recovery session has been submitted")
            return self._token

    async def _ensure_loaded_locked(self) -> None:
        async with self._lock:
            await self._ensure_loaded()

    async def flush(self) -> CheckpointToken:
        """Commit the latest submitted revision, explicitly retrying disk failures."""
        if self._closed:
            raise RuntimeError("Autosave writer is closed")
        # Cancellation of a UI await must not release the writer lock while its
        # SQLite thread is still committing. The owned operation finishes first.
        return await asyncio.shield(self._flush())

    async def clear(self) -> tuple[LabSession, CheckpointToken]:
        """Fence submissions immediately, drain in-flight I/O, then commit Clear."""
        if self._closed or self._closing:
            raise RuntimeError("Autosave writer is closed")
        if self._clearing:
            raise CheckpointConflict("Clear is already in progress")
        self._clearing = True
        self._wake.set()

        async def operation():
            try:
                async with self._lock:
                    await self._ensure_loaded()
                    if self._token is None:
                        if self._latest is None:
                            from tldw_chatbook.Chunking.lab_state import new_session

                            self._latest = await self._io(
                                new_session, self._store.profile_key
                            )
                        self._token = await self._io(
                            self._store.save, self._latest, expected=None
                        )
                    token = await self._io(self._store.clear, expected=self._token)
                    from tldw_chatbook.Chunking.lab_state import new_session

                    fresh = (
                        await self._io(new_session, self._store.profile_key)
                    ).model_copy(update={"epoch": token.epoch})
                    self._latest = fresh
                    self._pending = None
                    self._first_pending = None
                    self._immediate = False
                    self._token = token
                    self._initial = (fresh, token)
                    self._error = self._blocked = None
                    self._error_revision = -1
                    self._status = SaveStatus("saved", token, fresh.revision, None)
                    return fresh, token
            except Exception as exc:
                self._failed(exc, self._status.latest_revision)
                raise
            finally:
                self._clearing = False

        return await asyncio.shield(operation())

    async def replace(
        self, imported: LabSession, displaced: LabSession
    ) -> tuple[LabSession, CheckpointToken]:
        """Fence old submissions and atomically preserve displaced in-memory state.

        The coordinator must quiesce preview processes and suspend content edits
        before calling. Failure keeps the old epoch and latest draft retryable.
        """
        return await self._replace(imported, displaced)

    async def undo_restore(self) -> tuple[LabSession, CheckpointToken]:
        """Consume restore undo through the same serialized storage owner."""
        return await self._replace(None, None)

    async def _replace(
        self, imported: LabSession | None, displaced: LabSession | None
    ) -> tuple[LabSession, CheckpointToken]:
        if self._closed or self._closing:
            raise RuntimeError("Autosave writer is closed")
        if self._clearing:
            raise CheckpointConflict("Recovery replacement or Clear is in progress")
        if displaced is not None:
            if displaced.profile_key != self._store.profile_key or (
                self._latest is not None
                and (
                    displaced.epoch != self._latest.epoch
                    or displaced.revision < self._latest.revision
                )
            ):
                raise CheckpointConflict(
                    "Displaced session is not the latest active session"
                )
            self._latest = displaced
        self._clearing = True
        self._wake.set()

        async def operation():
            try:
                async with self._lock:
                    await self._ensure_loaded()
                    if self._token is None:
                        if displaced is None:
                            raise ValueError("There is no recovery restore to undo")
                        self._token = await self._io(
                            self._store.save, displaced, expected=None
                        )
                    if imported is None:
                        # A pending content edit consumes undo even if its debounce
                        # has not fired. Persist it before inspecting availability.
                        if self._pending is not None:
                            self._token = await self._io(
                                self._store.save, self._pending, expected=self._token
                            )
                        session, token = await self._io(
                            self._store.undo_restore, expected=self._token
                        )
                    else:
                        session, token = await self._io(
                            self._store.replace,
                            imported,
                            displaced,
                            expected=self._token,
                        )
                    self._latest = session
                    self._pending = None
                    self._first_pending = None
                    self._immediate = False
                    self._token = token
                    self._initial = (session, token)
                    self._error = self._blocked = None
                    self._error_revision = -1
                    self._status = SaveStatus("saved", token, session.revision, None)
                    return session, token
            except Exception as exc:
                self._pending = self._latest
                if self._token is not None:
                    self._status = replace(self._status, acknowledged=self._token)
                if self._latest is not None:
                    self._status = replace(
                        self._status, latest_revision=self._latest.revision
                    )
                self._failed(exc, self._status.latest_revision)
                raise
            finally:
                self._clearing = False

        return await asyncio.shield(operation())

    async def close(self) -> None:
        """Flush orderly exit, always close the worker-owned connection, then stop."""
        if self._closed:
            return
        self._closing = True
        self._wake.set()

        async def operation():
            try:
                if self._latest is not None or self._blocked is not None:
                    await self._flush()
            finally:
                if self._runner is not None:
                    await self._runner
                async with self._lock:
                    await self._io(self._store.close)
                self._executor.shutdown(wait=False)
                self._closed = True

        await asyncio.shield(operation())
