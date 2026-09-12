"""Short, same-thread CPU profiles for private event-loop test diagnostics."""

from __future__ import annotations

import asyncio
import cProfile
import math
import sys
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
    profile = cProfile.Profile(timer=time.thread_time)
    active = False
    stopped = False
    finish_handle = None
    started_wall = started_cpu = 0.0

    def stop():
        nonlocal active, stopped
        if stopped:
            return
        stopped = True
        start_handle.cancel()
        if finish_handle is not None:
            finish_handle.cancel()
        if not active:
            return
        profile.disable()
        active = False
        elapsed = time.perf_counter() - started_wall
        cpu = time.thread_time() - started_cpu
        calls = []
        for entry in sorted(
            profile.getstats(), key=lambda row: row.totaltime, reverse=True
        )[:32]:
            code = entry.code
            builtin = isinstance(code, str)
            calls.append(
                {
                    "file": "" if builtin else Path(code.co_filename).name[:128],
                    "function": code[:128] if builtin else code.co_name[:128],
                    "line": 0 if builtin else code.co_firstlineno,
                    "calls": entry.callcount,
                    "recursive_calls": entry.reccallcount,
                    "total_s": entry.totaltime,
                    "inline_s": entry.inlinetime,
                }
            )
        _write(Path(path), {"elapsed_s": elapsed, "thread_cpu_s": cpu, "calls": calls})

    def start():
        nonlocal active, finish_handle, started_wall, started_cpu
        if stopped:
            return
        started_wall, started_cpu = time.perf_counter(), time.thread_time()
        finish_handle = loop.call_later(duration, stop)
        active = True
        profile.enable()

    start_handle = loop.call_later(delay, start)
    return stop
