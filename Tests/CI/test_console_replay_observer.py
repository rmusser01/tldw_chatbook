"""The P0 replay observer identifies invocations after frame-address reuse."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import inspect
import sys
from pathlib import Path
from types import FrameType, SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_replay_observer_retires_recycled_frame_addresses():
    """Exercise the current observer and replay predicate with one address key."""
    root = Path(__file__).resolve().parents[2]
    helpers_tree = ast.parse(
        (root / "Tests/UI/test_console_poll_reconciliation.py").read_text(
            encoding="utf-8"
        )
    )
    helpers = []
    for name in ("_original_transition_events", "_transition_parent"):
        matches = [
            node
            for node in helpers_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        assert len(matches) == 1, name
        helpers.extend(matches)
    source_tree = ast.parse(
        (root / "Tests/UI/test_console_preparing_current_sources.py").read_text(
            encoding="utf-8"
        )
    )
    owners = [
        node
        for node in source_tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name
        == "test_preparing_changed_source_after_original_tab_await_replays_full"
    ]
    assert len(owners) == 1
    observers = [
        node
        for node in ast.walk(owners[0])
        if isinstance(node, ast.FunctionDef) and node.name == "observe"
    ]
    predicates = [
        node
        for node in ast.walk(owners[0])
        if isinstance(node, ast.Lambda)
        and any(
            isinstance(part, ast.Name) and part.id == "results"
            for part in ast.walk(node)
        )
    ]
    assert len(observers) == len(predicates) == 1
    namespace = dict(
        asyncio=asyncio,
        contextlib=contextlib,
        inspect=inspect,
        sys=sys,
        # Sequential invocations deliberately share an address. No interpreter
        # allocator behavior or historical Git revision is required.
        id=lambda _frame: 42,
    )
    exec(
        compile(
            ast.Module(body=helpers + observers, type_ignores=[]),
            "<current-replay-observer>",
            "exec",
        ),
        namespace,
    )
    predicate = eval(
        compile(ast.Expression(predicates[0]), "<current-replay-predicate>", "eval"),
        namespace,
    )
    lock = asyncio.Lock()

    class Surface:
        async def sync_sessions(self):
            async with lock:
                return None

    surface = Surface()

    class Console:
        _console_sync_in_progress = False
        _console_sync_requested = False

        async def full(self, poll_record):
            await surface.sync_sessions()
            return poll_record is None

    console = Console()
    record = object()
    timer = SimpleNamespace(_task=None)
    held, results, active_full = {}, [], {}
    namespace.update(
        surface=surface,
        console=console,
        record=record,
        timer=timer,
        full_code=Console.full.__code__,
        held=held,
        results=results,
        active_full=active_full,
        lock=lock,
        calls=[],
    )
    await lock.acquire()
    task = None
    try:
        with namespace["_original_transition_events"](
            [("surface", Surface, "sync_sessions"), ("full", Console, "full")],
            namespace["observe"],
        ):
            timer._task = task = asyncio.create_task(console.full(record))
            await asyncio.sleep(0)
            assert held and held["task"] is task
            lock.release()
            assert await task is False
            assert not active_full
            for _ in range(25):
                assert await console.full(None) is True
                assert not active_full
            assert len(results) == 26
            assert len({identity for identity, *_ in results}) == 26
            assert all(not isinstance(identity, FrameType) for identity, *_ in results)
            assert predicate()
    finally:
        if lock.locked():
            lock.release()
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
