"""Small task-outcome helpers for the existing Notes startup owner."""

from __future__ import annotations

import asyncio
from typing import Any


def _task_ended_without_result(task: asyncio.Task[None]) -> bool:
    """True when ``task`` is done and was cancelled or raised (TASK-34000.71 M-2)."""

    if not task.done():
        return False
    if task.cancelled():
        return True
    return task.exception() is not None


def _owner_still_starting(owner: Any) -> bool:
    """True when the owner's projection still reports ``"starting"``."""

    snapshot = getattr(owner, "snapshot", None)
    if not callable(snapshot):
        return False
    try:
        return getattr(snapshot(), "status", None) == "starting"
    except Exception:  # noqa: BLE001 - a double's projection must not crash the gate
        return False


async def _await_notes_sync_start(pending: Any) -> None:
    """Await a non-coroutine awaitable ``start()`` result (TASK-34000.71 M-3)."""

    await pending
