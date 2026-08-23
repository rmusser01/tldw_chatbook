"""Application-owned asynchronous Actor Pack export coordination."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from .export import (
    ActorPackExportError,
    ActorPackExportResult,
    ActorPackExportService,
    ActorPackExportSnapshot,
)
from .publication import (
    ActorPackDestinationContract,
    ActorPackPublicationError,
    publish_actor_pack,
)


class ActorPackExportControllerError(ValueError):
    """One stable, path-free controller admission failure."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class ActorPackExportRequest:
    """One immutable export request tied to controller profile authority."""

    actor_kind: str
    source: str
    profile_generation: int
    destination: ActorPackDestinationContract = field(repr=False)
    local_actor_id: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class ActorPackExportOutcome:
    """One bounded, path-free terminal operation result."""

    operation_id: int
    result: ActorPackExportResult | None = None
    error_category: str | None = None


@dataclass(slots=True)
class _ActiveExport:
    operation_id: int
    request: ActorPackExportRequest
    cancelled: threading.Event
    task: asyncio.Task[ActorPackExportOutcome]


class ActorPackExportController:
    """Own admission, blocking work, cancellation, results, and shutdown."""

    def __init__(
        self,
        service: ActorPackExportService,
        *,
        phase_hook: Callable[[str], None] | None = None,
    ) -> None:
        if service is None or (phase_hook is not None and not callable(phase_hook)):
            raise ActorPackExportControllerError("actor_pack_export_controller_invalid")
        self._service = service
        self._phase_hook = phase_hook
        self._lock = threading.RLock()
        self._profile_generation = 0
        self._next_operation_id = 0
        self._active: _ActiveExport | None = None
        self._last_outcome: ActorPackExportOutcome | None = None
        self._shutdown_requested = False
        self._shutdown_task: asyncio.Task[None] | None = None

    def create_request(
        self,
        *,
        actor_kind: str,
        local_actor_id: str,
        source: str,
        destination: ActorPackDestinationContract,
    ) -> ActorPackExportRequest:
        """Capture current controller/profile authority for one local export."""

        if (
            type(actor_kind) is not str
            or actor_kind not in {"character", "persona"}
            or type(local_actor_id) is not str
            or not local_actor_id
            or type(source) is not str
            or source != "local"
            or type(destination) is not ActorPackDestinationContract
        ):
            raise ActorPackExportControllerError("actor_pack_export_request_invalid")
        with self._lock:
            if self._shutdown_requested:
                raise ActorPackExportControllerError("actor_pack_export_shutdown")
            return ActorPackExportRequest(
                actor_kind=actor_kind,
                local_actor_id=local_actor_id,
                source=source,
                profile_generation=self._profile_generation,
                destination=destination,
            )

    def start_export(self, request: ActorPackExportRequest) -> int:
        """Register and start one app-owned operation before returning its token."""

        if type(request) is not ActorPackExportRequest:
            raise ActorPackExportControllerError("actor_pack_export_request_invalid")
        with self._lock:
            if self._shutdown_requested:
                raise ActorPackExportControllerError("actor_pack_export_shutdown")
            if self._active is not None:
                raise ActorPackExportControllerError("actor_pack_export_busy")
            if request.profile_generation != self._profile_generation:
                raise ActorPackExportControllerError(
                    "actor_pack_export_authority_changed"
                )
            self._next_operation_id += 1
            operation_id = self._next_operation_id
            cancelled = threading.Event()
            task = asyncio.create_task(
                asyncio.to_thread(
                    self._execute,
                    operation_id,
                    request,
                    cancelled,
                ),
                name=f"actor-pack-export:{operation_id}",
            )
            active = _ActiveExport(
                operation_id=operation_id,
                request=request,
                cancelled=cancelled,
                task=task,
            )
            self._active = active
            task.add_done_callback(
                lambda completed, expected=active: self._finish(expected, completed)
            )
            return operation_id

    async def wait(self, operation_id: int) -> ActorPackExportOutcome:
        """Await one exact operation while draining repeated caller cancellation."""

        if type(operation_id) is not int or operation_id <= 0:
            raise ActorPackExportControllerError("actor_pack_export_operation_invalid")
        with self._lock:
            active = self._active
            if active is None or active.operation_id != operation_id:
                if (
                    self._last_outcome is not None
                    and self._last_outcome.operation_id == operation_id
                ):
                    return self._last_outcome
                raise ActorPackExportControllerError(
                    "actor_pack_export_operation_unknown"
                )
            task = active.task
            cancelled = active.cancelled

        outer_cancellation: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                outer_cancellation = outer_cancellation or error
                cancelled.set()
                continue
        outcome = task.result()
        if outer_cancellation is not None:
            raise outer_cancellation
        return outcome

    def cancel(self, operation_id: int) -> bool:
        """Signal cancellation only for the exact active operation token."""

        with self._lock:
            active = self._active
            if active is None or active.operation_id != operation_id:
                return False
            active.cancelled.set()
            return True

    def invalidate_profile(self) -> int:
        """Advance profile authority and cancel work admitted by the prior profile."""

        with self._lock:
            self._profile_generation += 1
            if self._active is not None:
                self._active.cancelled.set()
            return self._profile_generation

    def last_outcome(self, operation_id: int) -> ActorPackExportOutcome | None:
        """Return only the single bounded terminal ledger entry when it matches."""

        with self._lock:
            if (
                self._last_outcome is not None
                and self._last_outcome.operation_id == operation_id
            ):
                return self._last_outcome
            return None

    async def shutdown(self) -> None:
        """Close admission, signal cancellation, and drain the active operation."""

        with self._lock:
            self._shutdown_requested = True
            if self._active is not None:
                self._active.cancelled.set()
            task = self._shutdown_task
            if task is None:
                task = asyncio.create_task(
                    self._shutdown_runner(), name="shutdown_actor_pack_export"
                )
                self._shutdown_task = task
        outer_cancellation: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                outer_cancellation = outer_cancellation or error
                continue
        task.result()
        if outer_cancellation is not None:
            raise outer_cancellation

    async def _shutdown_runner(self) -> None:
        with self._lock:
            active = self._active
        if active is None:
            return
        while not active.task.done():
            try:
                await asyncio.shield(active.task)
            except asyncio.CancelledError:
                continue
        active.task.result()

    def _execute(
        self,
        operation_id: int,
        request: ActorPackExportRequest,
        cancelled: threading.Event,
    ) -> ActorPackExportOutcome:
        try:
            snapshot = self._capture(request)

            def authority_guard() -> bool:
                if not self._request_is_current(request):
                    return False
                try:
                    current = self._capture(request)
                except Exception:
                    return False
                return current == snapshot and self._request_is_current(request)

            result = publish_actor_pack(
                snapshot,
                request.destination,
                authority_guard=authority_guard,
                cancelled=cancelled.is_set,
                phase_hook=self._phase_hook,
            )
            return ActorPackExportOutcome(operation_id=operation_id, result=result)
        except (ActorPackExportError, ActorPackPublicationError) as error:
            return ActorPackExportOutcome(
                operation_id=operation_id,
                error_category=error.category,
            )
        except Exception:
            return ActorPackExportOutcome(
                operation_id=operation_id,
                error_category="actor_pack_export_failed",
            )

    def _capture(self, request: ActorPackExportRequest) -> ActorPackExportSnapshot:
        return self._service.capture_snapshot(
            request.actor_kind,
            request.local_actor_id,
            source=request.source,
            phase_hook=self._phase_hook,
        )

    def _request_is_current(self, request: ActorPackExportRequest) -> bool:
        with self._lock:
            return (
                not self._shutdown_requested
                and request.profile_generation == self._profile_generation
            )

    def _finish(
        self,
        expected: _ActiveExport,
        completed: asyncio.Task[ActorPackExportOutcome],
    ) -> None:
        try:
            outcome = completed.result()
        except BaseException:
            outcome = ActorPackExportOutcome(
                operation_id=expected.operation_id,
                error_category="actor_pack_export_failed",
            )
        with self._lock:
            if self._active is expected:
                self._last_outcome = outcome
                self._active = None
