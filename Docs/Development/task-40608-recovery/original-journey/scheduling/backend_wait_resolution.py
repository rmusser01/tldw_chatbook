"""Matched lab-only backend scheduling context; no application instrumentation."""

from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import json
import os
import sys
import time


@contextmanager
def backend_interval(observation, arm):
    if (arm not in {"default", "1ms"} or observation.role != "backend"
            or observation.pid != os.getpid() or not observation.guard_started
            or observation.settings["route"] != "process"
            or observation.settings["mode"] != "guard"):
        raise RuntimeError("Backend wait diagnostic ownership changed")
    if (sys.platform != "win32" or sys.implementation.name != "cpython"
            or sys.version_info[:2] != (3, 12)):
        raise RuntimeError("Backend wait diagnostic requires Windows CPython 3.12")
    build = sys.getwindowsversion().build
    if build < 19041:
        raise RuntimeError("Backend wait diagnostic requires process-local timer policy")
    original = sys.getswitchinterval()
    if abs(original - 0.005) > 1e-12:
        raise RuntimeError("Backend wait diagnostic requires stock starting interval")
    target = 0.001 if arm == "1ms" else original
    # Both arms perform identical safe DLL lookup; only treatment requests a period.
    # LOAD_LIBRARY_SEARCH_SYSTEM32 excludes current directory/PATH DLL discovery.
    timer = ctypes.WinDLL("winmm.dll", use_last_error=True, winmode=0x00000800)
    begin, end = timer.timeBeginPeriod, timer.timeEndPeriod
    for function in (begin, end):
        function.argtypes = [wintypes.UINT]
        function.restype = wintypes.UINT
    path = observation.directory / f"backend-wait-resolution-{os.getpid()}.jsonl"
    with path.open("x", encoding="utf-8") as stream:
        def emit(event, **values):
            row = {"event": event, "role": "backend", "pid": os.getpid(),
                   "run_id": observation.run_id, "port": observation.port,
                   "arm": arm, "monotonic_ns": time.perf_counter_ns(),
                   "process_cpu_ns": time.process_time_ns(), **values}
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()

        entered = False
        period_requested = False
        period_ended = False
        interval_attempted = False
        try:
            if arm == "1ms":
                if begin(1) != 0:
                    raise RuntimeError("Backend timer request refused")
                period_requested = True
                interval_attempted = True
                sys.setswitchinterval(target)
            actual = sys.getswitchinterval()
            emit("entered", original_seconds=original, target_seconds=target,
                 actual_seconds=actual, windows_build=build,
                 period_requested=period_requested)
            if abs(actual - target) > 1e-12:
                raise RuntimeError("Backend interval not applied")
            entered = True
            yield
        finally:
            before_restore = sys.getswitchinterval()
            interval_cleanup_ok = False
            period_cleanup_ok = not period_requested
            try:
                if interval_attempted:
                    sys.setswitchinterval(original)
                interval_cleanup_ok = abs(sys.getswitchinterval() - original) <= 1e-12
            finally:
                try:
                    if period_requested:
                        period_ended = end(1) == 0
                        period_cleanup_ok = period_ended
                finally:
                    emit("restored", entered=entered,
                         active_unchanged=abs(before_restore - target) <= 1e-12,
                         restored_seconds=sys.getswitchinterval(),
                         restored=interval_cleanup_ok, period_ended=period_ended,
                         period_cleanup_ok=period_cleanup_ok)
            if not interval_cleanup_ok or not period_cleanup_ok:
                raise RuntimeError("Backend scheduling cleanup not confirmed")
