"""App-owned snapshot operations; file workers outlive their cancelled awaiters."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
    ServerLaunchClaim,
)
from tldw_chatbook.LLM_Management.snapshot_admission import (
    compatibility_matches,
    finalize_launch,
    revalidate_files,
)
from tldw_chatbook.LLM_Management.snapshot_client import SnapshotClient
from tldw_chatbook.LLM_Management.snapshot_models import (
    CatalogPage,
    LaunchDescriptor,
    ManagerView,
    SlotObservation,
    SnapshotError,
    WorkingFile,
    replace_descriptor,
)
from tldw_chatbook.LLM_Management.snapshot_settings import load_snapshot_preferences
from tldw_chatbook.LLM_Management.snapshot_store import SnapshotStore


def _error(code: str) -> SnapshotError:
    return SnapshotError(code, submission_possible=False)


@dataclass(eq=False)
class _Generation:
    descriptor: LaunchDescriptor
    client: SnapshotClient | None
    invalid: threading.Event = field(default_factory=threading.Event)
    slots: tuple[SlotObservation, ...] = ()
    status: str = "preparing"
    message: str | None = None
    operation_id: str | None = None
    started_at: float | None = None
    operation: asyncio.Task | None = None
    working: dict[str, WorkingFile] = field(default_factory=dict)
    refresh_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    confirmed_stopped: bool = False
    client_closed: bool = False
    ready: bool = False


class LlamaCppSnapshotService:
    """One application owner for a profile catalog and exact launch generations."""

    def __init__(
        self,
        store: SnapshotStore | None,
        is_current: Callable[[ServerLaunchClaim], bool],
    ) -> None:
        self.store = store
        self._is_current = is_current
        self._current: _Generation | None = None
        self._generations: dict[ServerLaunchClaim, _Generation] = {}
        self._closing = threading.Event()
        self._listeners: set[Callable[[], None]] = set()
        self._tasks: set[asyncio.Task] = set()
        self._local_tasks: set[asyncio.Task] = set()
        self._setup_task: asyncio.Task | None = None
        self._shutdown_task: asyncio.Task | None = None
        self._catalog = CatalogPage((), None, None, None, False)
        self._catalog_request = 0
        self._unavailable: str | None = None
        self._cleanup_warning: str | None = None

    def _spawn(self, coroutine) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _file(self, function, *args, **kwargs) -> Any:
        task = asyncio.create_task(
            asyncio.to_thread(partial(function, *args, **kwargs))
        )
        self._local_tasks.add(task)
        try:
            while True:
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError:
                    # Never lose a reservation returned by a still-running worker.
                    if task.cancelled():
                        raise
        finally:
            self._local_tasks.discard(task)

    async def initialize(self, root_factory: Callable[[], Path]) -> None:
        """Resolve the profile/store off-thread once, retaining setup through teardown."""
        if self.store is not None or self._closing.is_set():
            return
        if self._setup_task is None:
            self._setup_task = self._spawn(self._initialize(root_factory))
        await asyncio.shield(self._setup_task)

    async def _initialize(self, root_factory: Callable[[], Path]) -> None:
        try:
            self.store = await self._file(lambda: SnapshotStore(root_factory()))
            await self._reconcile_catalog()
            await self._load_catalog()
        except SnapshotError as exc:
            self._unavailable = exc.code
        except Exception:  # noqa: BLE001 - publish only a fixed setup failure code
            self._unavailable = "storage_unavailable"
        self._notify()

    def attach(self, descriptor: LaunchDescriptor) -> None:
        """Attach only a captured, still-live child claim; never adopt an HTTP listener."""
        if self._closing.is_set() or not self._is_current(descriptor.claim):
            return
        if descriptor.claim in self._generations:
            return
        if self._current is not None:
            self._current.invalid.set()
        generation = _Generation(
            descriptor,
            SnapshotClient(descriptor)
            if descriptor.base_url and descriptor.disabled_reason is None
            else None,
        )
        if descriptor.disabled_reason:
            generation.status = "unavailable"
        self._generations[descriptor.claim] = generation
        self._current = generation
        self._notify()

    def _valid(self, generation: _Generation) -> bool:
        # Worker-safe: Events and the caller's lifecycle predicate only. Never
        # acquire the catalog lock here; the store invokes this under that lock.
        return (
            not self._closing.is_set()
            and not generation.invalid.is_set()
            and self._is_current(generation.descriptor.claim)
        )

    def _publication_valid(
        self, generation: _Generation, descriptor: LaunchDescriptor
    ) -> bool:
        return self._valid(generation) and revalidate_files(descriptor)

    async def refresh(self) -> None:
        """Refresh catalog and same-generation readiness without relabelling evidence."""
        generation = self._current
        await self._reconcile_catalog()
        await self._load_catalog()
        if generation is not None:
            await self._refresh_generation(generation)
            self._notify_generation(generation)
        elif self._current is None:
            self._notify()

    async def browse_catalog(self, offset: int = 0, limit: int = 50) -> None:
        """Load a catalog page without HTTP; later browse requests win publication."""
        if self.store is None or self._closing.is_set():
            raise _error("service_unavailable")
        await self._load_catalog(offset, limit)
        self._notify()

    async def _reconcile_catalog(
        self, terminated: frozenset[str] = frozenset()
    ) -> bool:
        """Retry only proven-safe cleanup; warnings do not disable management."""
        if self.store is None:
            return False
        try:
            failures = await self._file(self.store.reconcile, terminated)
        except Exception:  # noqa: BLE001 - fixed warning; preserve teardown and browsing
            self._cleanup_warning = "cleanup_failed"
            return False
        self._cleanup_warning = "cleanup_incomplete" if failures else None
        return True

    async def _load_catalog(self, offset: int = 0, limit: int = 50) -> None:
        if self.store is None:
            return
        self._catalog_request += 1
        request_id = self._catalog_request
        try:
            page = await self._file(self.store.list_records, offset, limit)
            if request_id == self._catalog_request:
                self._catalog = page
        except SnapshotError as exc:
            self._unavailable = exc.code
        except Exception:  # noqa: BLE001 - storage details never reach projections
            self._unavailable = "storage_unavailable"

    async def _refresh_generation(self, generation: _Generation) -> None:
        if not self._valid(generation) or generation.client is None:
            return
        async with generation.refresh_lock:
            # Keep the last completed observation while a reentry probe awaits
            # I/O. An admitted operation already performs its own fresh probe;
            # a concurrent pending probe is not evidence that its launch failed.
            descriptor = generation.descriptor
            try:
                observation = await generation.client.readiness()
                if not self._valid(generation):
                    return
                updated = await self._file(finalize_launch, descriptor, observation)
                if not self._valid(generation):
                    return
                if (
                    descriptor.compatibility is not None
                    and updated.compatibility != descriptor.compatibility
                ):
                    updated = replace_descriptor(
                        descriptor,
                        compatibility=None,
                        disabled_reason="Launch compatibility changed.",
                    )
                generation.descriptor = updated
                generation.slots = observation.slots
                generation.ready = updated.compatibility is not None
                if updated.disabled_reason is not None:
                    generation.invalid.set()
                if generation.operation_id is None:
                    generation.status = (
                        "idle" if updated.compatibility else "unavailable"
                    )
                    generation.message = None
            except SnapshotError as exc:
                generation.ready = False
                generation.message = exc.code
                if generation.operation_id is None:
                    generation.status = "unavailable"
            except Exception:  # noqa: BLE001 - no raw HTTP failure enters retained state
                generation.ready = False
                generation.message = "readiness_failed"
                if generation.operation_id is None:
                    generation.status = "unavailable"

    def start_readiness(self) -> None:
        """Own the bounded startup readiness retry task independently of screens."""
        generation = self._current
        if generation is not None:
            self._spawn(self._readiness_retries(generation))

    async def _readiness_retries(self, generation: _Generation) -> None:
        try:
            for _ in range(10):
                if not self._valid(generation):
                    return
                await self._refresh_generation(generation)
                self._notify_generation(generation)
                if (
                    generation.descriptor.compatibility is not None
                    or generation.invalid.is_set()
                ):
                    return
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            return

    def _eligible(self, generation: _Generation, slot_id: int) -> None:
        if (
            not self._valid(generation)
            or not generation.ready
            or generation.descriptor.compatibility is None
        ):
            raise _error("launch_unavailable")
        slot = next(
            (value for value in generation.slots if value.slot_id == slot_id), None
        )
        if type(slot_id) is not int or slot is None or slot.busy is not False:
            raise _error("slot_unavailable")

    def _start(self, slot_id: int, snapshot_id: str | None) -> str:
        generation = self._current
        if self.store is None or generation is None or self._closing.is_set():
            raise _error("service_unavailable")
        if generation.operation_id is not None:
            raise _error("operation_in_progress")
        try:
            preferences = load_snapshot_preferences()
        except (ValueError, OSError):
            raise _error("preferences_unavailable") from None
        # Attachment captures this launch's opt-in; only retention changes live.
        self._eligible(generation, slot_id)
        operation_id = uuid4().hex
        generation.operation_id = operation_id
        generation.started_at = time.monotonic()
        generation.status = (
            "preparing" if snapshot_id is None else "staging_and_verifying"
        )
        generation.message = None
        generation.operation = self._spawn(
            self._operate(generation, slot_id, snapshot_id, preferences.keep_count)
        )
        self._notify_generation(generation)
        return operation_id

    def start_save(self, slot_id: int) -> str:
        """Admit and strongly own one save before returning its operation ID."""
        return self._start(slot_id, None)

    def start_restore(self, snapshot_id: str, slot_id: int) -> str:
        """Admit one verified restore independently of its originating screen."""
        return self._start(slot_id, snapshot_id)

    def _phase(self, generation: _Generation, status: str) -> None:
        generation.status = status
        self._notify_generation(generation)

    async def _operate(
        self,
        generation: _Generation,
        slot_id: int,
        snapshot_id: str | None,
        keep_count: int,
    ) -> None:
        working = None
        submitted = acknowledged = committed = False
        unknown = False
        store = self.store
        assert store is not None and generation.client is not None
        try:
            await self._refresh_generation(generation)
            self._eligible(generation, slot_id)
            descriptor = generation.descriptor
            if snapshot_id is None:
                working = await self._file(
                    store.reserve_save, descriptor.launch_id, slot_id
                )
            else:
                working = await self._file(
                    store.stage_restore, snapshot_id, descriptor.launch_id
                )
            generation.working[working.operation_id] = working
            self._eligible(generation, slot_id)
            if not await self._file(revalidate_files, descriptor):
                raise _error("compatibility_changed")
            if snapshot_id is not None and not compatibility_matches(
                working.source_record.compatibility, descriptor.compatibility
            ):
                raise _error("compatibility_mismatch")
            if snapshot_id is not None:
                destination = next(
                    slot for slot in generation.slots if slot.slot_id == slot_id
                )
                if (
                    destination.context_size is None
                    or working.source_record.tokens > destination.context_size
                ):
                    raise _error("destination_context_too_small")
            await self._file(store.set_operation_state, working, "unknown")
            self._eligible(generation, slot_id)
            self._phase(generation, "awaiting_ack")
            submitted = True
            mutation = (
                generation.client.save
                if snapshot_id is None
                else generation.client.restore
            )
            receipt = await mutation(slot_id, working.path.name)
            if receipt.slot_id != slot_id or receipt.filename != working.path.name:
                raise SnapshotError("outcome_unknown", submission_possible=True)
            acknowledged = True
            await self._file(store.set_operation_state, working, "acknowledged")
            self._phase(generation, "validating")
            self._eligible(generation, slot_id)
            if snapshot_id is None:
                self._phase(generation, "publishing")
                result = await self._file(
                    store.commit_save,
                    working,
                    receipt,
                    descriptor.compatibility,
                    "llama.cpp model",
                    keep_count,
                    validate_publication=partial(
                        self._publication_valid, generation, descriptor
                    ),
                )
                committed = True
                if result.cleanup_failed_ids:
                    generation.message = "cleanup_incomplete"
        except asyncio.CancelledError:
            unknown = submitted and not acknowledged
            generation.message = "outcome_unknown" if unknown else "operation_cancelled"
        except SnapshotError as exc:
            unknown = exc.submission_possible and not acknowledged
            generation.message = exc.code
        except Exception:  # noqa: BLE001 - unexpected post-dispatch failures remain uncertain
            unknown = submitted and not acknowledged
            generation.message = "outcome_unknown" if unknown else "operation_failed"
        finally:
            if working is not None and not unknown:
                if not committed:
                    try:
                        await self._file(store.set_operation_state, working, "terminal")
                        failures = await self._file(store.cleanup, working)
                        if failures:
                            generation.message = "cleanup_incomplete"
                    except Exception:  # noqa: BLE001 - surface fixed cleanup failure only
                        generation.message = "cleanup_failed"
                generation.working.pop(working.operation_id, None)
            if unknown:
                generation.status = "outcome_unknown"
            if self._valid(generation) and not unknown:
                message = generation.message
                try:
                    await self._refresh_generation(generation)
                except asyncio.CancelledError:
                    # Receipt/file settlement already finished; Stop need not wait
                    # for this final observation to release the operation.
                    pass
                generation.message = message
            await self._load_catalog()
            if not unknown:
                generation.operation_id = None
                generation.started_at = None
                generation.status = (
                    "idle"
                    if self._valid(generation) and generation.ready
                    else "unavailable"
                )
            self._notify_generation(generation)

    async def delete_snapshot(self, snapshot_id: str) -> None:
        """Delete a selected record even when server mutations are blocked."""
        if self.store is None or self._closing.is_set():
            raise _error("service_unavailable")
        failures = await self._file(self.store.delete, snapshot_id)
        await self._load_catalog()
        if failures:
            self._unavailable = "cleanup_incomplete"
        self._notify()

    async def server_stopped(self, claim: ServerLaunchClaim, confirmed: bool) -> None:
        """Settle exact-generation local handles before confirmed-stop cleanup."""
        generation = self._generations.get(claim)
        if generation is None or not confirmed:
            return
        generation.invalid.set()
        generation.confirmed_stopped = True
        if generation.operation is not None and not generation.operation.done():
            generation.operation.cancel()
            try:
                await asyncio.shield(generation.operation)
            except asyncio.CancelledError:
                if not generation.operation.cancelled():
                    raise
        if await self._reconcile_catalog(frozenset({generation.descriptor.launch_id})):
            generation.working.clear()
        await self._close_client(generation)
        generation.operation_id = None
        generation.started_at = None
        generation.status = "stopped"
        await self._load_catalog()
        self._notify_generation(generation)

    async def _close_client(self, generation: _Generation) -> None:
        if generation.client is not None and not generation.client_closed:
            generation.client_closed = True
            await generation.client.aclose()

    def view(self) -> ManagerView:
        """Return only bounded status, slot and catalog fields."""
        generation = self._current
        evidence = (
            generation.descriptor.compatibility
            if generation is not None and self._valid(generation) and generation.ready
            else None
        )
        compatibility = tuple(
            (
                record.snapshot_id,
                "unknown"
                if evidence is None
                else (
                    "matching"
                    if compatibility_matches(record.compatibility, evidence)
                    else "different"
                ),
            )
            for record in self._catalog.records
        )
        if generation is None:
            return ManagerView(
                None,
                "unavailable"
                if self._unavailable
                else ("preparing" if self.store is None else "idle"),
                None,
                None,
                (),
                self._catalog,
                self._unavailable
                or (
                    "Preparing snapshot storage."
                    if self.store is None
                    else "Start a managed llama.cpp server."
                ),
                self._cleanup_warning,
                compatibility,
                str(self.store.root) if self.store is not None else None,
            )
        return ManagerView(
            generation.descriptor.launch_id,
            generation.status,
            generation.operation_id,
            generation.started_at,
            generation.slots,
            self._catalog,
            generation.descriptor.disabled_reason
            or (None if self._valid(generation) else "Launch unavailable."),
            generation.message or self._cleanup_warning or self._unavailable,
            compatibility,
            str(self.store.root) if self.store is not None else None,
        )

    def subscribe(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Subscribe a disposable view without transferring operation ownership."""
        self._listeners.add(listener)
        return lambda: self._listeners.discard(listener)

    def _notify_generation(self, generation: _Generation) -> None:
        if self._current is generation:
            self._notify()

    def _notify(self) -> None:
        if not self._closing.is_set():
            for listener in tuple(self._listeners):
                listener()

    async def shutdown(self) -> None:
        """Settle local work, cancel network waits, and retain uncertain files."""
        if self._shutdown_task is None:
            self._closing.set()
            self._listeners.clear()
            self._shutdown_task = asyncio.create_task(self._shutdown())
        await asyncio.shield(self._shutdown_task)

    async def _shutdown(self) -> None:
        tasks = tuple(self._tasks)
        for task in tasks:
            if task is not self._setup_task:
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*tuple(self._local_tasks), return_exceptions=True)
        for generation in self._generations.values():
            await self._close_client(generation)
