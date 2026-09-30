"""Personas preview serialization and cancellation-safe task observation."""

from __future__ import annotations

import asyncio
import dataclasses
import threading
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import Any
from weakref import WeakKeyDictionary


class PersonasPreviewCoordinator:
    """Serialize and drain preview stages without retaining their owners."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock: asyncio.Lock | None = None

    def _lock_for_running_loop(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            if self._lock is not None and self._lock.locked():
                raise RuntimeError(
                    "Personas preview work is still active on another loop"
                )
            self._loop = loop
            self._lock = asyncio.Lock()
        assert self._lock is not None
        return self._lock

    @asynccontextmanager
    async def serialize(self) -> AsyncIterator[None]:
        """Hold the app's preview lane for one complete render request."""

        async with self._lock_for_running_loop():
            yield

    @staticmethod
    async def run_sync(function: Callable[..., Any], *args, **kwargs) -> Any:
        """Run one sync stage and drain its thread before propagating cancellation."""

        stage = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        try:
            return await asyncio.shield(stage)
        except asyncio.CancelledError:
            while not stage.done():
                try:
                    await asyncio.shield(stage)
                except asyncio.CancelledError:
                    continue
            try:
                stage.result()
            except Exception:
                pass
            raise


_COORDINATORS: WeakKeyDictionary[object, PersonasPreviewCoordinator] = (
    WeakKeyDictionary()
)


def get_personas_preview_coordinator(app: object) -> PersonasPreviewCoordinator:
    """Return the coordinator owned by ``app`` without retaining the app."""

    coordinator = _COORDINATORS.get(app)
    if coordinator is None:
        coordinator = PersonasPreviewCoordinator()
        _COORDINATORS[app] = coordinator
    return coordinator


@dataclasses.dataclass(frozen=True, slots=True)
class _DrainedTaskResult:
    """One task result observed after every outer cancellation is drained."""

    completed: bool = False
    value: Any = None
    error: Exception | None = None
    cancellation: asyncio.CancelledError | None = None


async def _drain_async(
    awaitable: Coroutine[Any, Any, Any], *, task_name: str
) -> _DrainedTaskResult:
    """Shield one critical task and report cancellation after it settles."""

    task = asyncio.create_task(awaitable, name=task_name)
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            return _DrainedTaskResult(
                completed=True,
                value=await asyncio.shield(task),
                cancellation=cancellation,
            )
        except asyncio.CancelledError as exc:
            if task.done() and task.cancelled():
                try:
                    task.result()
                except asyncio.CancelledError as child_cancellation:
                    return _DrainedTaskResult(
                        cancellation=cancellation or child_cancellation
                    )
            if cancellation is None:
                cancellation = exc
        except Exception as exc:
            return _DrainedTaskResult(error=exc, cancellation=cancellation)


async def _drain_to_thread(
    function: Callable[..., Any],
    /,
    *args: Any,
    task_name: str,
    **kwargs: Any,
) -> _DrainedTaskResult:
    """Observe the actual native callback, independently of Task cancellation.

    This is lifetime bookkeeping only; callbacks acquire their own source admission.
    A cancelled executor Future cannot retire an already-running native callback.
    """

    loop = asyncio.get_running_loop()
    completion = loop.create_future()
    lock = threading.Lock()
    status = "queued"

    def complete(outcome: _DrainedTaskResult) -> None:
        if not completion.done():
            completion.set_result(outcome)

    def work() -> None:
        nonlocal status
        with lock:
            if status != "queued":
                return
            status = "running"
        try:
            outcome = _DrainedTaskResult(completed=True, value=function(*args, **kwargs))
        except asyncio.CancelledError as error:
            outcome = _DrainedTaskResult(cancellation=error)
        except BaseException as error:
            outcome = _DrainedTaskResult(error=error)
        with lock:
            status = "finished"
        loop.call_soon_threadsafe(complete, outcome)

    executor = loop.run_in_executor(None, work)

    def executor_done(future) -> None:
        nonlocal status
        error = None if future.cancelled() else future.exception()
        if not future.cancelled() and error is None:
            return
        with lock:
            if status != "queued":
                return
            status = "cancelled"
        complete(_DrainedTaskResult(error=error, cancellation=asyncio.CancelledError()))

    executor.add_done_callback(executor_done)

    async def observed() -> _DrainedTaskResult:
        return await asyncio.shield(completion)

    task = asyncio.create_task(observed(), name=task_name)
    cancellation: asyncio.CancelledError | None = None
    waiting = task
    while True:
        try:
            outcome = await asyncio.shield(waiting)
            return dataclasses.replace(
                outcome, cancellation=cancellation or outcome.cancellation
            )
        except asyncio.CancelledError as error:
            cancellation = cancellation or error
            with lock:
                queued = status == "queued"
                if queued:
                    status = "cancelled"
            if queued:
                executor.cancel()
                complete(_DrainedTaskResult(cancellation=cancellation))
            # An independently cancelled named waiter is only an awaiter. The
            # private completion is signalled by the actual callback after IO.
            waiting = completion
