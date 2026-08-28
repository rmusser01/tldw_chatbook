"""TASK-23027: the per-worker-thread event loop behind the executor seams.

``run_worker_coroutine`` replaced 22 ``to_thread(lambda: asyncio.run(...))``
sites (plus the runtime's note-observation batch). These tests pin the
properties the replacement must keep: same-thread completion, exception
transparency, loop reuse per thread (the connection-churn fix), loop isolation
across threads, support for coroutines that offload internally, and immunity
to cancellation of the awaiting task. A source-level ratchet keeps the
retired ``asyncio.run`` pattern from creeping back into the executor.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import threading
from pathlib import Path

import pytest

import tldw_chatbook.Notes.notes_sync_executor as executor_module
from tldw_chatbook.Notes.notes_sync_executor import run_worker_coroutine

pytestmark = pytest.mark.unit


def test_returns_the_coroutine_result() -> None:
    async def answer() -> int:
        return 41 + 1

    assert run_worker_coroutine(answer()) == 42


def test_propagates_the_coroutine_exception() -> None:
    async def boom() -> None:
        raise RuntimeError("private failure")

    with pytest.raises(RuntimeError, match="private failure"):
        run_worker_coroutine(boom())


def test_reuses_one_loop_per_thread() -> None:
    """The whole point: no per-call loop construction on a worker thread."""

    loops: list[object] = []

    async def capture() -> None:
        loops.append(asyncio.get_running_loop())

    def worker() -> None:
        run_worker_coroutine(capture())
        run_worker_coroutine(capture())

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(5.0)
    assert len(loops) == 2
    assert loops[0] is loops[1]


def test_different_threads_use_different_loops() -> None:
    loops: list[object] = []
    lock = threading.Lock()

    async def capture() -> None:
        with lock:
            loops.append(asyncio.get_running_loop())

    threads = [
        threading.Thread(target=lambda: run_worker_coroutine(capture()))
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5.0)
    assert len(loops) == 2
    assert loops[0] is not loops[1]


def test_supports_internal_to_thread_offloads() -> None:
    """The folder-repository seam awaits asyncio.to_thread internally."""

    async def offloading() -> int:
        return await asyncio.to_thread(lambda: 7)

    def worker(result: list[int]) -> None:
        result.append(run_worker_coroutine(offloading()))

    result: list[int] = []
    thread = threading.Thread(target=worker, args=(result,))
    thread.start()
    thread.join(5.0)
    assert result == [7]


async def test_cancelling_the_awaiting_task_never_interrupts_the_coroutine() -> None:
    """Same guarantee asyncio.run gave: the worker-side coroutine finishes."""

    started = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    loop = asyncio.get_running_loop()

    async def mutation() -> None:
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5.0)
        finished.set()

    task = asyncio.create_task(
        asyncio.to_thread(lambda: run_worker_coroutine(mutation()))
    )
    await asyncio.wait_for(started.wait(), 5.0)
    task.cancel()
    release.set()
    # The thread-side coroutine must complete regardless of the cancel.
    assert await asyncio.to_thread(finished.wait, 5.0)
    with pytest.raises(asyncio.CancelledError):
        await task


def test_executor_module_has_no_asyncio_run_call_sites() -> None:
    """Ratchet: the retired per-call asyncio.run pattern must not return."""

    source = Path(inspect.getsourcefile(executor_module)).read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "asyncio"
    ]
    assert offenders == []
