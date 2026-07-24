from __future__ import annotations

import asyncio
from collections.abc import Callable


async def join_retained_task(
    task: asyncio.Future[None],
    *,
    on_failure_after_cancellation: Callable[[BaseException], None] | None = None,
) -> None:
    """Finish retained cleanup before propagating caller cancellation."""
    cancellation: asyncio.CancelledError | None = None
    waiter = asyncio.current_task()
    cancellation_requests = waiter.cancelling() if waiter is not None else 0
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            next_cancellation_requests = (
                waiter.cancelling() if waiter is not None else 0
            )
            if next_cancellation_requests > cancellation_requests:
                cancellation = cancellation or error
                cancellation_requests = next_cancellation_requests
        except BaseException:
            if not task.done():
                raise
            break

    task_error: BaseException | None = None
    try:
        task.result()
    except BaseException as error:
        task_error = error

    if cancellation is not None:
        if task_error is not None and on_failure_after_cancellation is not None:
            on_failure_after_cancellation(cancellation)
        raise cancellation
    if task_error is not None:
        raise task_error
