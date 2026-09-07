"""Bound admission to pure compiler work without owning mutations or source caches."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from threading import BoundedSemaphore
from typing import TypeVar

from .compiler import _compile_document
from .models import CanvasCompiledPlan
from .profiles import ProfileSnapshot

T = TypeVar("T")


def prepare_canvas_document(
    source: str,
    *,
    operation: str,
    parent_profile: str | None,
    snapshot: ProfileSnapshot,
) -> CanvasCompiledPlan:
    """Inspect, select and compile in one parse inside the owner's admission slot."""
    return _compile_document(
        source, operation=operation, parent_profile=parent_profile, snapshot=snapshot
    )


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
        # A cancelled shield waiter cannot retrieve a later worker failure.
        # Consume it without logging; normal awaiters still receive the error.
        future.add_done_callback(
            lambda completed: None if completed.cancelled() else completed.exception()
        )
        return await asyncio.shield(future)
