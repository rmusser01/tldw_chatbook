"""Physical lifetime shared by finite Console preparation reads."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
import os
import threading
from typing import Any


@dataclass(slots=True, eq=False)
class ConsolePreparationRead:
    """One internal read handle; observers share it without admission state."""

    creator: object = field(repr=False)
    callback: Callable[[], Any] = field(repr=False)
    source: object = field(repr=False)
    task: asyncio.Task[Any] = field(repr=False)
    session_id: str | None
    retired: asyncio.Future[None] = field(repr=False)
    _producer: asyncio.Future[Any] | None = field(default=None, repr=False)
    _observers: tuple[set[ConsolePreparationRead], ...] = field(default=(), repr=False)


# Only finite handle bookkeeping is serialized; no callback or await is inside.
_READS_LOCK = threading.RLock()


def _observe_locked(read, observer):
    if not any(observer is existing for existing in read._observers):
        read._observers += (observer,)
        observer.add(read)


def observe_preparation_reads(
    reads: set[ConsolePreparationRead], observer: set[ConsolePreparationRead]
) -> None:
    """Attach existing standalone reads to an additional creator's teardown."""
    with _READS_LOCK:
        for read in tuple(reads):
            _observe_locked(read, observer)


def preparation_reads_for(
    reads: set[ConsolePreparationRead], session_id: str | None = None
) -> tuple[ConsolePreparationRead, ...]:
    """Snapshot exact reads; app-scoped reads also belong to session teardown."""
    with _READS_LOCK:
        return tuple(
            read
            for read in reads
            if session_id is None
            or read.session_id is None
            or read.session_id == session_id
        )


def _retire(read):
    with _READS_LOCK:
        for observer in read._observers:
            observer.discard(read)
        read._observers = ()
    if not read.retired.done():
        read.retired.set_result(None)


@dataclass(slots=True)
class _HookSourceBinding:
    read: ConsolePreparationRead
    process: int
    thread: threading.Thread
    task: asyncio.Task[Any] | None
    active: bool = True


_source_binding: ContextVar[_HookSourceBinding | None] = ContextVar(
    "console_hook_preparation_source", default=None
)


def _current_task():
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


def preparation_source_for(creator: object) -> object | None:
    """Return pure captured sources only inside this exact worker invocation."""
    binding = _source_binding.get()
    if (
        binding is None
        or not binding.active
        or binding.read.creator is not creator
        or binding.process != os.getpid()
        or binding.thread is not threading.current_thread()
        or binding.task is not _current_task()
    ):
        return None
    return binding.read.source


async def run_preparation_read[_Result](
    callback: Callable[[], _Result],
    *,
    creator: object,
    session_id: str | None,
    reads: set[ConsolePreparationRead],
    observers: tuple[set[ConsolePreparationRead], ...] = (),
    require_current: Callable[[], None],
    source: object = None,
) -> _Result:
    """Retire an original hook read before returning or delivering cancellation.

    Args:
        callback: Captured original synchronous hook read.
        creator: Original controller or runtime identity.
        session_id: Session attribution, or None for app-scoped work.
        reads: Creator's physical-read observer set.
        observers: Additional sets observing this same exact handle.
        require_current: Pure source validator, called outside bookkeeping locks.
        source: Opaque captured source available only inside the worker scope.

    Returns:
        Original callback result after its native scope has exited.

    Raises:
        asyncio.CancelledError: Caller cancellation after native retirement and
            exception consumption; otherwise the original callback's exception.
        BaseException: Original callback, source or submission failure when the
            caller was not cancelled.
    """
    task = asyncio.current_task()
    if task is None:
        raise RuntimeError("A Console hook read requires its issuing task.")
    require_current()
    read = ConsolePreparationRead(
        creator,
        callback,
        source,
        task,
        session_id,
        asyncio.get_running_loop().create_future(),
    )
    with _READS_LOCK:
        for observer in (reads, *observers):
            _observe_locked(read, observer)
    submit_resolved = threading.Event()
    submitted = False

    def invoke():
        # An enqueue-then-raise submission may leave a work item behind.
        submit_resolved.wait()
        if not submitted:
            return None
        binding = _HookSourceBinding(
            read, os.getpid(), threading.current_thread(), _current_task()
        )
        token = _source_binding.set(binding)
        try:
            require_current()
            result = callback()
            require_current()
            return result
        except StopIteration as error:
            # asyncio Futures cannot transport StopIteration; convert inside
            # the worker before its physical outcome crosses to the coroutine.
            raise RuntimeError("coroutine raised StopIteration") from error
        finally:
            binding.active = False
            _source_binding.reset(token)

    try:
        try:
            context = copy_context()
            read._producer = asyncio.get_running_loop().run_in_executor(
                None, context.run, invoke
            )
            submitted = True
        finally:
            submit_resolved.set()
        producer = read._producer
        cancelled = None
        while not producer.done():
            try:
                await asyncio.shield(producer)
            except asyncio.CancelledError as error:
                # A callback-raised CancelledError is its actual native outcome,
                # including when this Task retains a prior cancellation count.
                if (
                    producer.done()
                    and not producer.cancelled()
                    and producer.exception() is error
                ):
                    break
                cancelled = error
            except BaseException:
                break
        try:
            result = producer.result()
        except BaseException:
            if cancelled is not None:
                raise cancelled from None
            raise
        if cancelled is not None:
            raise cancelled
        require_current()
        return result
    finally:
        _retire(read)


async def drain_preparation_reads(
    reads: set[ConsolePreparationRead], session_id: str | None = None
) -> bool:
    """Wait only for physical read retirement, preserving cancellation intent.

    Args:
        reads: Creator's physical-read observer set.
        session_id: Selected session; None includes every read. App-scoped reads
            are always included.

    Returns:
        Whether this drain's caller was cancelled while exact reads retired.
        The caller decides when to re-deliver that cancellation.

    Raises:
        RuntimeError: A read attempts to drain itself or its retirement notice
            was cancelled instead of establishing physical completion.
    """
    cancelled = False
    while pending := preparation_reads_for(reads, session_id):
        if any(read.task is _current_task() for read in pending):
            raise RuntimeError("A Console hook read cannot finalize its own creator.")
        for read in pending:
            while not read.retired.done():
                try:
                    await asyncio.shield(read.retired)
                except asyncio.CancelledError:
                    if read.retired.cancelled():
                        raise RuntimeError(
                            "Hook read retirement signal was cancelled."
                        ) from None
                    cancelled = True
            if read.retired.cancelled():
                raise RuntimeError("Hook read retirement signal was cancelled.")
            read.retired.result()
    return cancelled


async def await_finite_read[_Result](
    operation: Coroutine[Any, Any, _Result],
) -> _Result:
    """Retire a stock finite read before delivering view-worker cancellation."""
    worker = asyncio.Task(operation)
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:  # noqa: BLE001 - cancellation wins.
                break
        if not worker.cancelled():
            worker.exception()
        raise
