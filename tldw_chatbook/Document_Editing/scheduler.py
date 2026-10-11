"""An active editor's single check/format lane and latest pending snapshot."""

import asyncio
from collections.abc import Callable
from typing import Protocol

from .models import CheckResult, DocumentRequest, FormatResult


class Worker(Protocol):
    async def check(self, request: DocumentRequest) -> CheckResult: ...
    async def format(self, request: DocumentRequest) -> FormatResult: ...


class DocumentCheckScheduler:
    """Keep one active snapshot and one replaceable pending snapshot."""

    def __init__(
        self, worker: Worker, on_result: Callable[[CheckResult], None]
    ) -> None:
        self._worker = worker
        self._on_result: Callable[[CheckResult], None] | None = on_result
        self._pending: (
            tuple[
                DocumentRequest,
                str,
                int,
                asyncio.Future[FormatResult | None] | None,
                float,
            ]
            | None
        ) = None
        self._runner: asyncio.Task[None] | None = None
        self._changed = asyncio.Event()
        self._generation = 0
        self._retired = False
        self._active_future: asyncio.Future[FormatResult | None] | None = None

    def submit(self, request: DocumentRequest, *, immediate: bool = False) -> None:
        """Check after 400ms idle; initial/manual/language checks may be immediate."""
        self._enqueue(request, "check", None, 0 if immediate else 0.4)

    async def format(self, request: DocumentRequest) -> FormatResult | None:
        """Share the check lane; obsolete formatting settles without replay."""
        if self._retired:
            return None
        future: asyncio.Future[FormatResult | None] = (
            asyncio.get_running_loop().create_future()
        )
        self._enqueue(request, "format", future, 0)
        try:
            return await future
        except asyncio.CancelledError:
            self.retire()
            raise

    def _enqueue(self, request, operation, future, delay):
        if self._retired:
            return
        self._generation += 1
        if self._pending is not None:
            self._settle(self._pending[3])
        self._pending = (
            request,
            operation,
            self._generation,
            future,
            asyncio.get_running_loop().time() + delay,
        )
        self._changed.set()
        if self._runner is None or self._runner.done():
            self._runner = asyncio.create_task(self._drain())

    @staticmethod
    def _settle(future, value=None):
        if future is not None and not future.done():
            future.set_result(value)

    async def _drain(self):
        try:
            while self._pending is not None and not self._retired:
                self._changed.clear()
                delay = self._pending[4] - asyncio.get_running_loop().time()
                if delay > 0:
                    # Wait without retaining the replaceable source snapshot.
                    try:
                        async with asyncio.timeout(delay):
                            await self._changed.wait()
                    except TimeoutError:
                        pass
                    continue
                request, operation, generation, future, _ = self._pending
                self._pending = None
                self._active_future = future
                result = await getattr(self._worker, operation)(request)
                if (
                    not self._retired
                    and generation == self._generation
                    and result.key == request.key
                ):
                    if operation == "check" and self._on_result is not None:
                        self._on_result(result)
                    else:
                        self._settle(future, result)
                self._settle(future)
                self._active_future = None
                del request, result, future
        finally:
            self._settle(self._active_future)
            self._active_future = None

    def retire(self) -> None:
        """Permanently revoke this lease, including loader-pending work."""
        if self._retired:
            return
        self._retired = True
        self._generation += 1
        if self._pending is not None:
            self._settle(self._pending[3])
            self._pending = None
        self._settle(self._active_future)
        self._on_result = None
        if self._runner is not None:
            self._runner.cancel()

    async def close(self) -> None:
        """Retire synchronously, then wait for actual worker cleanup."""
        self.retire()
        if self._runner is not None:
            await asyncio.gather(self._runner, return_exceptions=True)
            self._runner = None
