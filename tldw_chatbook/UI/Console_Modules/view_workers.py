"""Retain only the actual host/view's issued finite Console workers."""

import asyncio
import threading
from typing import Any

from textual.worker import Worker
from textual.worker_manager import WorkerManager

CONSOLE_BROWSER_WORKER_GROUPS = frozenset(
    {"console-persisted-browser-cache", "console-flat-conversation-search"}
)
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
        "console-library-activity",
        "console-workspace-display-reads",
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
        "console-flat-conversation-search",
    }
)
async def owned_display_read(database: Any, operation: Any, /, *args: Any) -> Any:
    """Run one finite display read off the UI loop, retired before cancellation.

    ``run_owned_db_call`` on a worker thread; a cancelled caller waits for the
    executor's callback to end, so its connection is never left in use behind
    a finished worker (TASK-33620.15.4; the citation counts' shape).
    """
    from ...DB.base_db import run_owned_db_call

    read = asyncio.ensure_future(run_owned_db_call(database, operation, *args))
    try:
        return await asyncio.shield(read)
    except asyncio.CancelledError:
        while not read.done():
            try:
                await asyncio.shield(read)
            except asyncio.CancelledError:
                continue
            except Exception:  # noqa: BLE001 - cancellation wins.
                break
        if not read.cancelled():
            read.exception()
        raise


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
    host: Any, view: Any | None = None, *, groups: frozenset[str] = _GROUPS
) -> _CapturedViewWorkers:
    """Capture issued Tasks and nodes in the original host manager's group scope.

    A specific view selects only that node. With no view, retain the original
    App exit coverage, including current and previously detached nodes.

    Args:
        host: Actual manager-owning App or supported host.
        view: Optional exact node filter; None keeps the App's group scope.
        groups: Narrow the vetted finite groups; never add other groups.

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
            or type(group) is not str
            or group not in _GROUPS
            or group not in groups
        ):
            continue
        task, work = values.get("_task"), values.get("_work")
        if type(task) is not asyncio.Task or task.get_loop() is not loop:
            raise RuntimeError("console_view_worker_owner_changed")
        workers.append((worker, values.get("_node"), task, work))
    return host, view, manager, loop, threading.current_thread(), tuple(workers)


def cancel_console_view_workers(captured: _CapturedViewWorkers) -> None:
    """Cancel exact issued workers synchronously under their manager custody.

    Args:
        captured: Exact references selected on the original loop and thread.

    Raises:
        RuntimeError: The captured ownership or original wait source changed.
    """
    host, _view, manager, loop, thread, workers = captured
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


async def drain_console_view_workers(captured: _CapturedViewWorkers) -> None:
    """Keep captured worker waits alive through repeated caller cancellation.

    Args:
        captured: Exact references selected on the original loop and thread.

    Raises:
        RuntimeError: The captured ownership or original wait source changed.
        asyncio.CancelledError: Caller cancellation, after the waits settle.
    """
    cancel_console_view_workers(captured)
    _host, _view, _manager, loop, _thread, workers = captured

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

    coordinator = settle()
    issued_coordinators = ()
    cancellation = None
    try:
        try:
            owned = asyncio.create_task(coordinator, name="retire_console_view_workers")
        except BaseException:
            # A factory may enqueue this exact coroutine and then raise. Only
            # that issued Task belongs to this failed creation attempt.
            issued_coordinators = tuple(
                task for task in asyncio.all_tasks(loop)
                if task.get_coro() is coordinator
            )
            if not issued_coordinators:
                coordinator.close()
            for task in issued_coordinators:
                task.cancel()
            raise
        while not owned.done():
            try:
                await asyncio.shield(owned)
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
        try:
            owned.result()
        except asyncio.CancelledError as error:
            cancellation = cancellation or error
    finally:
        # The coordinator may itself be cancelled before it attaches its waits.
        # Its cancellation never releases custody of these already-issued Tasks.
        tasks = tuple(task for _worker, _node, task, _work in workers)
        for task in (*tasks, *issued_coordinators):
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError as error:
                    cancellation = cancellation or error
                except Exception:  # noqa: BLE001 - preserve gather's error policy.
                    break
        for task in issued_coordinators:
            if not task.cancelled():
                task.exception()
    if any(not task.done() for _worker, _node, task, _work in workers):
        raise RuntimeError("console_view_worker_not_retired")
    if cancellation is not None:
        raise cancellation
