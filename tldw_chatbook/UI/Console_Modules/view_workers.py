"""Retain only the actual host/view's issued finite Console workers."""

import asyncio
import threading
from typing import Any
from textual.worker import Worker
from textual.worker_manager import WorkerManager

_GROUPS = frozenset(
    {
        "console-sync",
        "console-context-presentation",
        "console-readiness-config",
        "console-readiness-publication",
        "console-manual-unread-load",
        "console-agent-history",
        "console-subagent-counts",
        "console-subagent-count-publication",
        "console-citation-counts",
        "console-context-publication",
        "console-hook-refresh",
        "console-prompt-history",
        "console-rail-prune",
        "console-skill-discovery",
        "console-skill-trust-setup",
        "console-resume-navigation-startup",
        "console-resume-navigation-dispatch",
        "console-character-context-refresh",
        "console-annotation-previews",
        "console-persisted-browser-cache",
    }
)
_CANCEL, _WAIT = Worker.cancel, Worker.wait
_CANCEL_CODE, _WAIT_CODE = _CANCEL.__code__, _WAIT.__code__

_CapturedViewWorkers = tuple[
    Any,
    Any | None,
    WorkerManager,
    asyncio.AbstractEventLoop,
    threading.Thread,
    tuple[tuple[Worker, Any, asyncio.Task, Any], ...],
]


def capture_console_view_workers(
    host: Any, view: Any | None = None
) -> _CapturedViewWorkers:
    """Capture issued Tasks and nodes in the original host manager's group scope.

    A specific view selects only that node. With no view, retain the original
    App exit coverage, including current and previously detached nodes.

    Args:
        host: Actual manager-owning App or supported host.
        view: Optional exact node filter; None keeps the App's group scope.

    Returns:
        Strong references to the captured manager, nodes, work and Tasks.

    Raises:
        RuntimeError: The captured manager or issued Task owner is invalid.
    """
    loop = asyncio.get_running_loop()
    manager = vars(host).get("_workers")
    if type(manager) is not WorkerManager or vars(manager).get("_app") is not host:
        raise RuntimeError("console_view_worker_owner_changed")
    workers = []
    for worker in tuple(manager):
        if type(worker) is not Worker:
            continue
        values = vars(worker)
        group = values.get("group")
        if (
            (view is not None and values.get("_node") is not view)
            or type(group) is not str  # noqa: E721 - reject custom membership dispatch.
            or group not in _GROUPS
        ):
            continue
        task, work = values.get("_task"), values.get("_work")
        if type(task) is not asyncio.Task or task.get_loop() is not loop:
            raise RuntimeError("console_view_worker_owner_changed")
        workers.append((worker, values.get("_node"), task, work))
    return host, view, manager, loop, threading.current_thread(), tuple(workers)


async def drain_console_view_workers(captured: _CapturedViewWorkers) -> None:
    """Keep captured worker waits alive through repeated caller cancellation.

    Args:
        captured: Exact references selected on the original loop and thread.

    Raises:
        RuntimeError: The captured ownership or original wait source changed.
        asyncio.CancelledError: Caller cancellation, after the waits settle.
    """
    host, view, manager, loop, thread, workers = captured
    if (
        asyncio.get_running_loop() is not loop
        or threading.current_thread() is not thread
        or vars(host).get("_workers") is not manager
        or Worker.cancel is not _CANCEL
        or Worker.wait is not _WAIT
        or _CANCEL.__code__ is not _CANCEL_CODE
        or _WAIT.__code__ is not _WAIT_CODE
    ):
        raise RuntimeError("console_view_worker_owner_changed")
    for worker, node, task, work in workers:
        values = vars(worker)
        if (
            values.get("_node") is not node
            or values.get("_task") is not task
            or values.get("_work") is not work
        ):
            raise RuntimeError("console_view_worker_owner_changed")
        if not task.done():
            _CANCEL(worker)

    async def settle():
        if (
            Worker.cancel is not _CANCEL
            or Worker.wait is not _WAIT
            or _CANCEL.__code__ is not _CANCEL_CODE
            or _WAIT.__code__ is not _WAIT_CODE
        ):
            raise RuntimeError("console_view_worker_owner_changed")
        await asyncio.gather(
            *(_WAIT(worker) for worker, _node, _task, _work in workers),
            return_exceptions=True,
        )
        # Pending workers may reject Worker.wait before cancellation is delivered.
        # Their exact captured Tasks must still consume terminal cancellation.
        await asyncio.gather(
            *(task for _worker, _node, task, _work in workers), return_exceptions=True
        )

    owned = asyncio.create_task(settle(), name="retire_console_view_workers")
    cancellation = None
    while not owned.done():
        try:
            await asyncio.shield(owned)
        except asyncio.CancelledError as error:
            cancellation = cancellation or error
    owned.result()
    if any(not task.done() for _worker, _node, task, _work in workers):
        raise RuntimeError("console_view_worker_not_retired")
    if cancellation is not None:
        raise cancellation
