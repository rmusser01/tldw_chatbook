"""Application-owned asynchronous Actor Pack import coordination."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ActorPackImportControllerError(ValueError):
    """One stable, path-free controller admission failure."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class ActorPackImportRequest:
    """One immutable path-private request bound to profile authority."""

    profile_generation: int
    archive_path: Path = field(repr=False)


@dataclass(frozen=True, slots=True)
class ActorPackImportOutcome:
    """One path-free terminal inspect or activation result."""

    operation_id: int
    review: Any | None = field(default=None, repr=False)
    result: Any | None = None
    error_category: str | None = None
    refresh_errors: tuple[str, ...] = ()


@dataclass(slots=True)
class _ActiveImport:
    operation_id: int
    generation: int
    cancelled: threading.Event
    task: asyncio.Task[ActorPackImportOutcome]
    phase: str


class ActorPackImportController:
    """Own off-loop inspection, activation, cancellation, leases, and refresh."""

    def __init__(
        self,
        importer: object,
        activation: object,
        *,
        refresh_callbacks: Sequence[Callable[[object], None]] = (),
    ) -> None:
        if not callable(getattr(importer, "inspect_archive", None)) or not callable(
            getattr(activation, "activate", None)
        ):
            raise ActorPackImportControllerError("actor_pack_import_controller_invalid")
        if any(not callable(callback) for callback in refresh_callbacks):
            raise ActorPackImportControllerError("actor_pack_import_controller_invalid")
        self._importer = importer
        self._activation = activation
        self._refresh_callbacks = tuple(refresh_callbacks)
        self._lock = threading.RLock()
        self._profile_generation = 0
        self._next_operation_id = 0
        self._active: _ActiveImport | None = None
        self._leased_review: object | None = None
        self._last_outcome: ActorPackImportOutcome | None = None
        self._shutdown_requested = False
        self._shutdown_task: asyncio.Task[None] | None = None

    def create_request(self, archive_path: Path) -> ActorPackImportRequest:
        """Capture the current profile generation for one absolute archive."""

        if not isinstance(archive_path, Path) or not archive_path.is_absolute():
            raise ActorPackImportControllerError("actor_pack_import_request_invalid")
        with self._lock:
            self._require_open()
            return ActorPackImportRequest(self._profile_generation, archive_path)

    def start_inspection(self, request: ActorPackImportRequest) -> int:
        """Start one hostile-archive inspection."""

        if type(request) is not ActorPackImportRequest:
            raise ActorPackImportControllerError("actor_pack_import_request_invalid")
        with self._lock:
            self._require_open()
            self._require_idle()
            if request.profile_generation != self._profile_generation:
                raise ActorPackImportControllerError(
                    "actor_pack_import_authority_changed"
                )
            if self._leased_review is not None:
                raise ActorPackImportControllerError("actor_pack_import_review_active")
            return self._start(
                request.profile_generation,
                "inspect",
                lambda cancelled: self._inspect(request, cancelled),
            )

    def start_activation(self, review: object, action: str) -> int:
        """Activate the exact review lease currently owned by this controller."""

        with self._lock:
            self._require_open()
            self._require_idle()
            if review is not self._leased_review:
                raise ActorPackImportControllerError(
                    "actor_pack_import_operation_unknown"
                )
            if type(action) is not str or action not in getattr(
                review, "allowed_actions", ()
            ):
                raise ActorPackImportControllerError("actor_pack_import_action_invalid")
            return self._start(
                self._profile_generation,
                "activate",
                lambda cancelled: self._activate(review, action, cancelled),
            )

    async def wait(self, operation_id: int) -> ActorPackImportOutcome:
        """Await one exact operation while draining repeated caller cancellation."""

        if type(operation_id) is not int or operation_id <= 0:
            raise ActorPackImportControllerError("actor_pack_import_operation_invalid")
        with self._lock:
            active = self._active
            if active is None or active.operation_id != operation_id:
                if (
                    self._last_outcome
                    and self._last_outcome.operation_id == operation_id
                ):
                    return self._last_outcome
                raise ActorPackImportControllerError(
                    "actor_pack_import_operation_unknown"
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
        outcome = task.result()
        if outer_cancellation is not None:
            raise outer_cancellation
        return outcome

    def cancel(self, operation_id: int) -> bool:
        """Signal cancellation for only the exact active operation."""

        with self._lock:
            if self._active is None or self._active.operation_id != operation_id:
                return False
            self._active.cancelled.set()
            return True

    def discard_review(self, review: object) -> bool:
        """Release and clean only the exact inactive review lease."""

        with self._lock:
            if self._active is not None or review is not self._leased_review:
                return False
            self._leased_review = None
        self._cleanup(review)
        return True

    def invalidate_profile(self) -> int:
        """Advance profile authority, cancel work, and release an idle review."""

        review: object | None = None
        with self._lock:
            self._profile_generation += 1
            if self._active is not None:
                self._active.cancelled.set()
            else:
                review, self._leased_review = self._leased_review, None
            generation = self._profile_generation
        if review is not None:
            self._cleanup(review)
        return generation

    def last_outcome(self, operation_id: int) -> ActorPackImportOutcome | None:
        """Return the bounded last outcome only when its token matches."""

        with self._lock:
            if self._last_outcome and self._last_outcome.operation_id == operation_id:
                return self._last_outcome
            return None

    async def shutdown(self) -> None:
        """Close admission, signal cancellation, and drain before cleanup."""

        with self._lock:
            self._shutdown_requested = True
            if self._active is not None:
                self._active.cancelled.set()
            task = self._shutdown_task
            if task is None:
                task = asyncio.create_task(
                    self._shutdown_runner(), name="shutdown_actor_pack_import"
                )
                self._shutdown_task = task
        outer_cancellation: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                outer_cancellation = outer_cancellation or error
        task.result()
        if outer_cancellation is not None:
            raise outer_cancellation

    def _start(
        self,
        generation: int,
        phase: str,
        operation: Callable[[threading.Event], ActorPackImportOutcome],
    ) -> int:
        self._next_operation_id += 1
        operation_id = self._next_operation_id
        cancelled = threading.Event()

        def execute() -> ActorPackImportOutcome:
            outcome = operation(cancelled)
            return ActorPackImportOutcome(
                operation_id,
                review=outcome.review,
                result=outcome.result,
                error_category=outcome.error_category,
                refresh_errors=outcome.refresh_errors,
            )

        task = asyncio.create_task(
            asyncio.to_thread(execute), name=f"actor-pack-import:{operation_id}"
        )
        active = _ActiveImport(operation_id, generation, cancelled, task, phase)
        self._active = active
        task.add_done_callback(
            lambda completed, expected=active: self._finish(expected, completed)
        )
        return operation_id

    def _inspect(
        self, request: ActorPackImportRequest, cancelled: threading.Event
    ) -> ActorPackImportOutcome:
        try:
            review = self._importer.inspect_archive(
                request.archive_path, cancel_requested=cancelled.is_set
            )
            if not self._generation_is_current(request.profile_generation):
                self._cleanup(review)
                return ActorPackImportOutcome(
                    0, error_category="actor_pack_import_cancelled"
                )
            return ActorPackImportOutcome(0, review=review)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return ActorPackImportOutcome(0, error_category=_error_category(exc))

    def _activate(
        self, review: object, action: str, cancelled: threading.Event
    ) -> ActorPackImportOutcome:
        try:
            result = self._activation.activate(
                review, action, cancel_requested=cancelled.is_set
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return ActorPackImportOutcome(0, error_category=_error_category(exc))
        errors: list[str] = []
        for callback in self._refresh_callbacks:
            try:
                callback(result)
            except Exception:
                errors.append("actor_pack_import_refresh_failed")
        return ActorPackImportOutcome(0, result=result, refresh_errors=tuple(errors))

    def _finish(
        self,
        expected: _ActiveImport,
        completed: asyncio.Task[ActorPackImportOutcome],
    ) -> None:
        try:
            outcome = completed.result()
        except BaseException:
            outcome = ActorPackImportOutcome(
                expected.operation_id, error_category="actor_pack_import_failed"
            )
        cleanup: object | None = None
        with self._lock:
            if self._active is not expected:
                return
            if expected.phase == "inspect" and outcome.review is not None:
                self._leased_review = outcome.review
            elif expected.phase == "activate" and outcome.result is not None:
                self._leased_review = None
            if expected.generation != self._profile_generation:
                cleanup, self._leased_review = self._leased_review, None
                if outcome.result is None:
                    outcome = ActorPackImportOutcome(
                        expected.operation_id,
                        error_category="actor_pack_import_cancelled",
                    )
            self._last_outcome = outcome
            self._active = None
        if cleanup is not None:
            self._cleanup(cleanup)

    async def _shutdown_runner(self) -> None:
        with self._lock:
            active = self._active
        if active is not None:
            while not active.task.done():
                try:
                    await asyncio.shield(active.task)
                except asyncio.CancelledError:
                    continue
            active.task.result()
        with self._lock:
            review, self._leased_review = self._leased_review, None
        if review is not None:
            await asyncio.to_thread(self._cleanup, review)

    def _cleanup(self, review: object) -> None:
        try:
            self._importer.cleanup_review(review)
        except Exception:
            pass

    def _generation_is_current(self, generation: int) -> bool:
        with self._lock:
            return (
                not self._shutdown_requested and generation == self._profile_generation
            )

    def _require_open(self) -> None:
        if self._shutdown_requested:
            raise ActorPackImportControllerError("actor_pack_import_shutdown")

    def _require_idle(self) -> None:
        if self._active is not None:
            raise ActorPackImportControllerError("actor_pack_import_busy")


def _error_category(exc: BaseException) -> str:
    category = getattr(exc, "category", None)
    if type(category) is str and category.startswith("actor_pack_import_"):
        return category
    message = str(exc)
    if message.startswith("actor_pack_import_"):
        return message
    return "actor_pack_import_failed"


__all__ = [
    "ActorPackImportController",
    "ActorPackImportControllerError",
    "ActorPackImportOutcome",
    "ActorPackImportRequest",
]
