"""Runtime-owned worker for the existing serialized fresh display phases."""

from __future__ import annotations

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from inspect import getattr_static

from tldw_chatbook.Utils.definition_records import (
    capture_definitions,
    definitions_current,
)

_EXECUTOR_TYPE = ThreadPoolExecutor
_SHUTDOWN = ThreadPoolExecutor.shutdown


class ConsoleDisplayExecutor:
    """Keep display work off workers with another producer's cached custody."""

    def __init__(self, runtime) -> None:
        self.runtime = runtime
        self._pid = os.getpid()
        self._thread = threading.current_thread()
        self._loop = None
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="console-display"
        )
        self._original_executor = self._executor
        self._closed = False
        self._closing = None

    def _require(self) -> None:
        loop = asyncio.get_running_loop()
        if (
            type(self) is not ConsoleDisplayExecutor
            or self.runtime._display_phase_executor is not self
            or os.getpid() != self._pid
            or threading.current_thread() is not self._thread
            or (self._loop is not None and self._loop is not loop)
            or type(self._executor) is not ThreadPoolExecutor
            or self._executor is not self._original_executor
            or not definitions_current(_DEFINITIONS)
            or any(
                getattr_static(self, name, None) is not method
                for name, method in _METHODS
            )
        ):
            raise RuntimeError("Console display executor owner changed")
        self._loop = loop

    def executor(self) -> ThreadPoolExecutor:
        """Return only this runtime's original worker while intake is open."""
        self._require()
        if self._closed or self.runtime._disposed:
            raise RuntimeError("Console display executor is closed")
        return self._executor

    def close_admission(self) -> None:
        """Reject even a previously captured submission; retain queued work."""
        self._require()
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=False, cancel_futures=False)

    async def aclose(self) -> None:
        """Join the actual worker away from its owning loop on every exit."""
        # Definition drift is still a failed close, but must not strand the
        # original worker. Never attempt another process/loop owner's cleanup.
        if (
            os.getpid() != self._pid
            or threading.current_thread() is not self._thread
            or (self._loop is not None and asyncio.get_running_loop() is not self._loop)
            or self.runtime._display_phase_executor is not self
            or type(self._original_executor) is not _EXECUTOR_TYPE
        ):
            raise RuntimeError("Console display executor cleanup owner changed")
        failure = None
        try:
            ConsoleDisplayExecutor.close_admission(self)
        except BaseException as error:
            failure = error
            self._closed = True
            _SHUTDOWN(self._original_executor, wait=False, cancel_futures=False)
        if self._closing is None:
            self._closing = asyncio.create_task(
                asyncio.to_thread(
                    _SHUTDOWN, self._original_executor, wait=True, cancel_futures=False
                ),
                name="console-display-worker-close",
            )
        cancelled = False
        while not self._closing.done():
            try:
                await asyncio.shield(self._closing)
            except asyncio.CancelledError:
                cancelled = True
        self._closing.result()
        if failure is not None:
            raise failure
        if cancelled:
            raise asyncio.CancelledError


_METHODS = tuple(
    (name, getattr(ConsoleDisplayExecutor, name))
    for name in ("__init__", "_require", "executor", "close_admission", "aclose")
)
_DEFINITIONS = capture_definitions(
    (
        *((ConsoleDisplayExecutor, name, method) for name, method in _METHODS),
        *(
            (ThreadPoolExecutor, name, getattr(ThreadPoolExecutor, name))
            for name in ("__init__", "submit", "shutdown", "_adjust_thread_count")
        ),
    )
)
