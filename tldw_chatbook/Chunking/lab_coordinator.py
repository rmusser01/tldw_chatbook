"""App/profile-owned immutable Lab lifecycle, independent of screen mounts."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass

from .lab_autosave import AutosaveWriter, SaveStatus
from .lab_models import LabSession, RunRequest, RunResult
from .lab_recovery import parse_recovery
from .lab_runner import LocalPreviewRunner, terminal_result
from .lab_state import accept_result, capture_batch, install_batch, new_session


@dataclass(frozen=True)
class LabEvent:
    """Copied invalidation/status event without payloads, process or DB handles."""

    profile_key: str
    epoch: str
    revision: int
    save_status: SaveStatus
    busy: bool
    guarded: bool


class LabCoordinator:
    """Serialize authority transitions while permitting edits during execution.

    Session values are immutable-transition inputs: callers must use lab_state
    helpers, off-loop for large content. ``session`` borrows that read-only snapshot;
    never mutate its nested dictionaries. Events copy only small invalidation/status
    fields, not retained blobs. ``set_session`` performs no blob copying/hashing.
    Use ``load`` for first startup so failed recovery can never become empty write
    authority. The application owns this coordinator across screen unmounts.
    """

    def __init__(
        self, session: LabSession, writer: AutosaveWriter, runner: LocalPreviewRunner
    ):
        self._session = session
        self._writer = writer
        self._runner = runner
        self._run_done: asyncio.Future[None] | None = None
        self._transition: asyncio.Task | None = None
        self._stop = False
        self._guarded = False
        self._closed = False
        self._subscribers: set[Callable[[LabEvent], None]] = set()
        self._notifier: asyncio.Task | None = None

    @classmethod
    async def load(
        cls, profile_key: str, writer: AutosaveWriter, runner: LocalPreviewRunner
    ) -> LabCoordinator:
        """Load durable state without execution; propagate failed-load authority."""
        restored = await writer.load()
        session = (
            restored[0]
            if restored
            else await asyncio.to_thread(new_session, profile_key)
        )
        if session.profile_key != profile_key:
            raise ValueError("Recovered session belongs to another profile")
        return cls(session, writer, runner)

    @property
    def session(self) -> LabSession:
        """Borrow the current immutable snapshot for serialized pure transitions."""
        return self._session

    @property
    def busy(self) -> bool:
        return self._run_done is not None or self._transition is not None

    @property
    def guarded(self) -> bool:
        return self._guarded

    @property
    def save_status(self) -> SaveStatus:
        return self._writer.status

    @property
    def recovery_warning(self) -> str | None:
        """Expose the existing content-free previous-checkpoint fallback warning."""
        return self._writer.recovery_warning

    def subscribe(self, callback: Callable[[LabEvent], None]) -> Callable[[], None]:
        """Subscribe to small copied events; returned function detaches on unmount."""
        self._subscribers.add(callback)
        if self._notifier is None or self._notifier.done():
            self._notifier = asyncio.create_task(self._notify_changes())
        return lambda: self._subscribers.discard(callback)

    async def _notify_changes(self) -> None:
        previous = None
        while self._subscribers and not self._closed:
            session = self._session
            status = self.save_status
            signature = (id(session), status, self.busy, self.guarded)
            if signature != previous:
                previous = signature
                for callback in tuple(self._subscribers):
                    with contextlib.suppress(Exception):
                        callback(
                            LabEvent(
                                session.profile_key,
                                session.epoch,
                                session.revision,
                                status,
                                signature[2],
                                signature[3],
                            )
                        )
            await asyncio.sleep(0.05)

    def set_session(self, session: LabSession) -> None:
        """Install one pure UI transition, rejecting guarded/stale authority."""
        if self._guarded or self._closed:
            raise RuntimeError("Lab transition is guarded")
        old = self._session
        if (session.profile_key, session.epoch) != (
            old.profile_key,
            old.epoch,
        ) or session.revision < old.revision:
            raise ValueError("UI transition belongs to stale session authority")
        self._session = session
        self._writer.submit(session)
        if (
            self._run_done is not None
            and old.batch is not None
            and (
                session.batch is None
                or session.batch["batch_id"] != old.batch["batch_id"]
            )
        ):
            self._stop = True
            # Own this stop through the Run operation, which cannot finish until
            # runner.run has reaped. No second Run can enter in that interval.
            asyncio.create_task(self._runner.cancel())

    def _member_active(self, request: RunRequest) -> bool:
        batch = self._session.batch
        return bool(
            batch
            and request.epoch == self._session.epoch
            and batch["batch_id"] == request.batch_id
            and request.run_id in batch["requests"]
            and request.run_id not in batch["outcomes"]
        )

    async def _accept(self, result: RunResult) -> None:
        while self._member_active(result.request):
            before = self._session
            try:
                changed = await asyncio.to_thread(accept_result, before, result)
            except ValueError:
                if result.status != "completed":
                    raise
                # State admission includes total recovery size/depth/blob count;
                # a runner success is not published if active recovery cannot hold it.
                result = await asyncio.to_thread(
                    terminal_result, result.request, "limited", "recovery_limit"
                )
                continue
            if self._session is before:
                self._session = changed
                self._writer.submit(changed, immediate=True)
                return

    async def run(self, candidate_ids: tuple[str, ...]) -> None:
        """Persist one exact manifest, then execute A and B sequentially."""
        if self.busy or self._closed or self._guarded:
            raise RuntimeError("Lab is busy")
        completed = asyncio.get_running_loop().create_future()
        self._run_done = completed
        self._stop = False
        self._guarded = True
        requests = ()
        launched = False
        try:
            before = self._session
            requests = await asyncio.to_thread(capture_batch, before, candidate_ids)
            installed = await asyncio.to_thread(install_batch, before, requests)
            if self._stop:
                return
            self._session = installed
            self._writer.submit(installed, immediate=True)
            # No child can start until this critical checkpoint has committed.
            await self._writer.flush()
            if self._transition is None:
                self._guarded = False
            for request in requests:
                if self._stop or not self._member_active(request):
                    break
                launched = True
                result = await self._runner.run(request)
                if self._stop:
                    result = await asyncio.to_thread(
                        terminal_result, request, "canceled", "canceled"
                    )
                await self._accept(result)
                # Keep the admitted result in memory even if persistence fails.
                await self._writer.flush()
        except asyncio.CancelledError:
            self._stop = True
            await self._runner.cancel()
            raise
        finally:
            try:
                if self._stop:
                    await self._runner.cancel()
                    for request in requests:
                        await self._accept(
                            await asyncio.to_thread(
                                terminal_result, request, "canceled", "canceled"
                            )
                        )
                    if launched:
                        await self._writer.flush()
            finally:
                self._run_done = None
                if self._transition is None:
                    self._guarded = False
                # The caller can continue arbitrary work after awaiting run().
                # Quiescence owns only this operation's cleanup, not that caller.
                completed.set_result(None)

    async def _quiesce(self) -> None:
        self._stop = True
        completed = self._run_done
        await self._runner.cancel()
        if completed is not None:
            # Save failure belongs to the writer's retry status. Replacement can
            # still preserve the latest in-memory checkpoint transactionally.
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.shield(completed)

    async def cancel(self) -> None:
        """Stop active work and queued members, then checkpoint navigation."""
        await self._quiesce()
        self._writer.submit(self._session, immediate=True)
        await self._writer.flush()

    async def _guard(self, operation: Callable) -> None:
        if self._transition is not None or self._closed:
            raise RuntimeError("Lab transition is already guarded")
        self._guarded = True

        async def owned():
            try:
                await self._quiesce()
                await operation()
            finally:
                self._guarded = False
                self._transition = None

        self._transition = asyncio.create_task(owned())
        # UI cancellation must not release the edit fence during a real commit.
        await asyncio.shield(self._transition)

    async def replace_recovery(self, payload: bytes) -> None:
        """Validate, quiesce, then atomically replace; failures keep old authority."""
        imported = await asyncio.to_thread(parse_recovery, payload)

        async def operation():
            restored, _ = await self._writer.replace(imported, self._session)
            self._session = restored

        await self._guard(operation)

    async def undo_restore(self) -> None:
        """Stop previews before consuming the writer's one-level restore undo."""

        async def operation():
            self._writer.submit(self._session, immediate=True)
            restored, _ = await self._writer.undo_restore()
            self._session = restored

        await self._guard(operation)

    async def clear(self) -> None:
        """Stop and reap before committing Clear's new epoch and empty state."""

        async def operation():
            self._writer.submit(self._session, immediate=True)
            cleared, _ = await self._writer.clear()
            self._session = cleared

        await self._guard(operation)

    async def close(self) -> None:
        """Quiesce and flush; expose failures with the last session still in memory."""
        if self._closed:
            return
        if self._transition is not None:
            await asyncio.shield(self._transition)
            if self._closed:
                return

        async def operation():
            self._writer.submit(self._session, immediate=True)
            await self._writer.flush()
            await self._runner.close()
            await self._writer.close()
            self._closed = True

        await self._guard(operation)
        if self._notifier is not None:
            await self._notifier
