"""App-lifetime serialization for Personas lazy preview work."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
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
