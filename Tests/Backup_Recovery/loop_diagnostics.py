"""Short, same-thread CPU profiles for private event-loop test diagnostics."""

from __future__ import annotations

import asyncio
import math
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from Tests.Backup_Recovery.thread_diagnostics import _write


def observe_loop_profile(
    path: Path, *, delay: float = 2, duration: float = 5
) -> Callable[[], None]:
    """Schedule a bounded profile; stop on the same loop thread, at most once."""
    if (
        not math.isfinite(delay)
        or delay < 0
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError("invalid_profile_window")
    loop = asyncio.get_running_loop()
    if sys.getprofile() is not None:
        raise RuntimeError("loop_profile_already_active")
    owner = threading.get_ident()
    rows, stack = {}, []
    overflow = dropped_calls = 0
    active = False
    stopped = False
    finish_handle = None
    started_wall = started_cpu = 0.0

    def finish_call(now):
        key, began, children = stack.pop()
        elapsed = max(0.0, now - began)
        if key is not None:
            rows[key]["total_s"] += elapsed
            rows[key]["inline_s"] += max(0.0, elapsed - children)
        if stack:
            stack[-1][2] += elapsed

    def profile(frame, event, arg):
        nonlocal overflow, dropped_calls
        if threading.get_ident() != owner:
            return
        if event in ("call", "c_call"):
            if overflow or len(stack) >= 128:
                overflow += 1
                dropped_calls += 1
                return
            if event == "call":
                code = frame.f_code
                key = (
                    Path(code.co_filename).name[:128],
                    code.co_name[:128],
                    code.co_firstlineno,
                )
            else:
                module = getattr(arg, "__module__", None)
                name = getattr(arg, "__name__", None)
                # Never stringify a callable or its bound receiver.
                function = (module + ".") if isinstance(module, str) else ""
                function += name if isinstance(name, str) else "native_call"
                key = ("", function[:128], 0)
            if key not in rows:
                if len(rows) >= 256:
                    key = None
                    dropped_calls += 1
                else:
                    rows[key] = {
                        "file": key[0],
                        "function": key[1],
                        "line": key[2],
                        "calls": 0,
                        "recursive_calls": 0,
                        "total_s": 0.0,
                        "inline_s": 0.0,
                    }
            if key is not None:
                rows[key]["calls"] += 1
                rows[key]["recursive_calls"] += int(any(row[0] == key for row in stack))
            stack.append([key, time.thread_time(), 0.0])
        elif event in ("return", "c_return", "c_exception"):
            if overflow:
                overflow -= 1
            elif stack:
                finish_call(time.thread_time())

    def stop():
        nonlocal active, stopped
        if threading.get_ident() != owner:
            raise RuntimeError("loop_profile_wrong_thread")
        if stopped:
            return
        stopped = True
        start_handle.cancel()
        if finish_handle is not None:
            finish_handle.cancel()
        if not active:
            return
        sys.setprofile(None)
        active = False
        elapsed = time.perf_counter() - started_wall
        cpu = time.thread_time() - started_cpu
        now = time.thread_time()
        while stack:
            finish_call(now)
        calls = sorted(rows.values(), key=lambda row: row["total_s"], reverse=True)[:32]
        _write(
            Path(path),
            {
                "thread_id": threading.get_native_id(),
                "elapsed_s": elapsed,
                "thread_cpu_s": cpu,
                "calls": calls,
                "dropped_calls": dropped_calls,
            },
        )

    def start():
        nonlocal active, finish_handle, started_wall, started_cpu
        if stopped:
            return
        started_wall, started_cpu = time.perf_counter(), time.thread_time()
        finish_handle = loop.call_later(duration, stop)
        active = True
        sys.setprofile(profile)

    start_handle = loop.call_later(delay, start)
    return stop
