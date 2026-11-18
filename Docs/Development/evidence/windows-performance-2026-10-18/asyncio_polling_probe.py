"""Measure Windows short-poller feedback without importing application code.

Run with Python 3.12 on Windows. Two five-millisecond pollers and an existing
ten-millisecond heartbeat mirror the UI probe; the control uses twice the
actual event-loop clock resolution. The optional native worker performs only
GetCurrentProcessId calls, illustrating scheduling rather than storage speed.
"""

import argparse
import asyncio
import ctypes
import json
from pathlib import Path
import platform
import sys
import threading
import time


async def measure(seconds: float, safe: bool, worker: bool) -> dict:
    loop = asyncio.get_running_loop()
    resolution = loop._clock_resolution
    interval = max(0.005, 2 * resolution) if safe else 0.005
    heartbeat_interval = max(0.010, 2 * resolution) if safe else 0.010
    iterations = zero_selects = selects = 0
    poll_counts = {"input": 0, "attach": 0, "heartbeat": 0}
    original_select, original_run_once = loop._selector.select, loop._run_once

    def select(timeout=None):
        nonlocal zero_selects, selects
        selects += 1
        zero_selects += timeout == 0
        return original_select(timeout)

    def run_once():
        nonlocal iterations
        iterations += 1
        original_run_once()

    stop = threading.Event()
    ready = threading.Event()
    worker_result = {}

    def native_worker():
        native = ctypes.WinDLL("kernel32").GetCurrentProcessId
        native.argtypes = []
        native.restype = ctypes.c_ulong
        calls = 0
        cpu_start, wall_start = time.thread_time(), time.perf_counter()
        ready.set()
        while not stop.is_set():
            native()
            calls += 1
        worker_result.update(
            thread_cpu_ms=(time.thread_time() - cpu_start) * 1000,
            elapsed_ms=(time.perf_counter() - wall_start) * 1000,
            native_calls=calls,
        )

    thread = threading.Thread(target=native_worker) if worker else None
    if thread is not None:
        thread.start()
        assert ready.wait(5)
    loop._selector.select, loop._run_once = select, run_once
    cpu_start, started = time.thread_time(), time.perf_counter()
    deadline = started + seconds

    async def poll(name, delay):
        while time.perf_counter() < deadline:
            poll_counts[name] += 1
            await asyncio.sleep(delay)

    tasks = []
    try:
        # Start the heartbeat first. Then create the attach observer while the
        # current coroutine starts input polling, as in the application probe.
        tasks.append(asyncio.create_task(poll("heartbeat", heartbeat_interval)))
        await asyncio.sleep(0)
        tasks.append(asyncio.create_task(poll("attach", interval)))
        await poll("input", interval)
        await asyncio.gather(*tasks)
    finally:
        elapsed = time.perf_counter() - started
        cpu = time.thread_time() - cpu_start
        loop._selector.select, loop._run_once = original_select, original_run_once
        stop.set()
        if thread is not None:
            thread.join(5)
            assert not thread.is_alive()
    return {
        "clock_safe": safe,
        "native_worker": worker,
        "clock_resolution_seconds": resolution,
        "poll_seconds": interval,
        "heartbeat_seconds": heartbeat_interval,
        "elapsed_ms": elapsed * 1000,
        "main_thread_cpu_ms": cpu * 1000,
        "loop_iterations": iterations,
        "selector_calls": selects,
        "zero_timeout_selects": zero_selects,
        "poll_counts": poll_counts,
        "worker": worker_result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("this control requires the Windows Proactor event loop")
    if not 0 < args.seconds <= 10:
        parser.error("seconds must be between zero and ten")
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "monotonic": vars(time.get_clock_info("monotonic")),
        "perf_counter": vars(time.get_clock_info("perf_counter")),
        "runs": [
            asyncio.run(measure(args.seconds, safe, worker))
            for worker in (False, True)
            for safe in (False, True)
        ],
    }
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
