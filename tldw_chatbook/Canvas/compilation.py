"""Bound admission to pure compiler work without owning mutations or source caches."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from threading import BoundedSemaphore
from typing import TypeVar

T = TypeVar("T")


class CanvasCompilation:
    """Allow two outstanding compilations per existing authority owner, without a queue."""

    def __init__(self) -> None:
        self._slots = BoundedSemaphore(2)

    def _admit(self) -> None:
        if not self._slots.acquire(blocking=False):
            raise RuntimeError("canvas_compilation_busy")

    def _call(self, operation: Callable[[], T]) -> T:
        try:
            return operation()
        finally:
            self._slots.release()

    def run(self, operation: Callable[[], T]) -> T:
        """Compile on an existing tool worker, outside its controller lock."""

        self._admit()
        return self._call(operation)

    async def run_async(self, operation: Callable[[], T]) -> T:
        """Keep admission until the actual worker exits, even if its waiter cancels."""

        self._admit()
        try:
            future = asyncio.get_running_loop().run_in_executor(
                None, self._call, operation
            )
        except BaseException:
            self._slots.release()
            raise
        return await asyncio.shield(future)
