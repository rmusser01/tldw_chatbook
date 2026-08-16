"""App-local serialization for Console reaction preview sync work."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

_APP_COORDINATOR_ATTR = "_console_reaction_preview_coordinator"


class ConsoleReactionPreviewCoordinator:
    """Serialize preview threads across every Console screen for one app."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock: asyncio.Lock | None = None

    def _current_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            if self._lock is not None and self._lock.locked():
                raise RuntimeError(
                    "reaction preview work crossed event loops before draining"
                )
            self._loop = loop
            self._lock = asyncio.Lock()
        assert self._lock is not None
        return self._lock

    async def run_sync(self, function: Callable[..., Any], *args: Any) -> Any:
        """Run one sync stage and drain its thread before releasing single-flight."""

        async with self._current_lock():
            underlying = asyncio.create_task(asyncio.to_thread(function, *args))
            try:
                return await asyncio.shield(underlying)
            except asyncio.CancelledError:
                while not underlying.done():
                    try:
                        await asyncio.shield(underlying)
                    except asyncio.CancelledError:
                        continue
                    except Exception:  # noqa: BLE001 -- drain sync stage failure.
                        break
                if underlying.done() and not underlying.cancelled():
                    underlying.exception()
                raise


def get_console_reaction_preview_coordinator(
    app: object,
) -> ConsoleReactionPreviewCoordinator:
    """Return the single preview coordinator owned by one stable app instance."""

    coordinator = getattr(app, _APP_COORDINATOR_ATTR, None)
    if not isinstance(coordinator, ConsoleReactionPreviewCoordinator):
        coordinator = ConsoleReactionPreviewCoordinator()
        setattr(app, _APP_COORDINATOR_ATTR, coordinator)
    return coordinator


__all__ = [
    "ConsoleReactionPreviewCoordinator",
    "get_console_reaction_preview_coordinator",
]
