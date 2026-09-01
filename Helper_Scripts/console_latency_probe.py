"""In-terminal Console latency probe (TASK-26834's baseline instrument).

Runs the real app with three instruments attached:

* every Click/Key is stamped at ``Screen._forward_event`` and its window
  closes at the next ``_compositor_refresh`` -- interaction -> first paint;
* a daemon thread samples the MAIN thread's stack every 10ms while a window
  is open; stalls >100ms report the stacks that occupied them;
* when the main thread is idle-in-select during a window, every OTHER
  thread's stack is sampled too, so waiting is attributed, not just noticed.

Run it in the terminal you actually use, poke the interactions that feel
slow, quit with ctrl+q; the report prints on exit. Findings that produced
this tool and how to read its output (including which idle-thread stacks
are noise): TASK-26834. Known limitation: the window closes at the FIRST
paint after the event, so an interaction that paints a cheap ack early
under-reports its real answer.

Usage: .venv/bin/python Helper_Scripts/console_latency_probe.py
"""
import collections
import statistics
import sys
import threading
import time

import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STATS = {
    "events": [],
    "pending": None,      # (label, t0, full0, bytes0)
    "full": 0,
    "partial": 0,
    "bytes": 0,
    "stall_stacks": collections.Counter(),   # stack-tuple -> samples (10ms each)
    "worker_stacks": collections.Counter(),  # what OTHER threads did while main waited
    "window_samples": [],
    "window_worker_samples": [],
    "main_thread_id": None,
    "sampler_thread_id": None,
}
_LOCK = threading.Lock()


def _fmt_frame(frame) -> str:
    code = frame.f_code
    fn = code.co_filename
    short = fn.split("/")[-1]
    if "/textual/" in fn or "/site-packages/" in fn:
        short = "lib:" + short
    elif "/tldw_chatbook/" in fn:
        short = "app:" + short
    return f"{short}:{frame.f_lineno}:{code.co_name}"


def _stack_of(frame) -> tuple:
    stack = []
    depth = 0
    while frame is not None and depth < 25:
        stack.append(_fmt_frame(frame))
        frame = frame.f_back
        depth += 1
    return tuple(stack[:6])


_IDLE_TOKENS = ("selectors.py", "threading.py", "queue.py", "socket.py")


def _looks_idle(stack: tuple) -> bool:
    top = stack[0] if stack else ""
    return any(tok in top for tok in _IDLE_TOKENS) and (
        ":select" in top or ":wait" in top or ":get" in top or ":accept" in top
        or ":_recv" in top or ":recv" in top
    )


def _sampler() -> None:
    while True:
        time.sleep(0.010)
        with _LOCK:
            pending = STATS["pending"]
        if pending is None:
            continue
        frames = sys._current_frames()
        main = frames.get(STATS["main_thread_id"])
        if main is None:
            continue
        main_stack = _stack_of(main)
        busy_workers = []
        if _looks_idle(main_stack):
            # Main is waiting: what is everyone else doing?
            for tid, frame in frames.items():
                if tid in (STATS["main_thread_id"], STATS["sampler_thread_id"]):
                    continue
                stack = _stack_of(frame)
                if stack and not _looks_idle(stack):
                    busy_workers.append(stack)
        with _LOCK:
            STATS["window_samples"].append(main_stack)
            STATS["window_worker_samples"].extend(busy_workers)


def _install() -> None:
    STATS["main_thread_id"] = threading.get_ident()
    from textual._compositor import Compositor
    from textual.screen import Screen
    from textual import events

    o_full = Compositor.render_full_update
    o_part = Compositor.render_partial_update

    def sfull(self, *a, **k):
        STATS["full"] += 1
        return o_full(self, *a, **k)

    def spart(self, *a, **k):
        STATS["partial"] += 1
        return o_part(self, *a, **k)

    Compositor.render_full_update = sfull
    Compositor.render_partial_update = spart

    from textual.drivers.linux_driver import LinuxDriver

    if hasattr(LinuxDriver, "write"):
        o_write = LinuxDriver.write

        def write(self, data, _o=o_write):
            STATS["bytes"] += len(data)
            return _o(self, data)

        LinuxDriver.write = write

    o_forward = Screen._forward_event

    def forward(self, event):
        if isinstance(event, (events.Click, events.Key)):
            label = "click"
            if isinstance(event, events.Key):
                label = f"key:{event.key}"
            else:
                try:
                    widget, _ = self.get_widget_at(event.screen_x, event.screen_y)
                    wid = getattr(widget, "id", None) or type(widget).__name__
                    label = f"click:{wid}"
                except Exception:
                    pass
            with _LOCK:
                STATS["pending"] = (label, time.perf_counter(), STATS["full"], STATS["bytes"])
                STATS["window_samples"] = []
        return o_forward(self, event)

    Screen._forward_event = forward

    o_refresh = Screen._compositor_refresh

    def refresh(self):
        r = o_refresh(self)
        with _LOCK:
            p = STATS["pending"]
            if p is not None:
                label, t0, f0, b0 = p
                dur = (time.perf_counter() - t0) * 1000
                STATS["events"].append(
                    (label, dur, STATS["full"] - f0, STATS["bytes"] - b0)
                )
                if dur >= 100 and STATS["window_samples"]:
                    for stack in STATS["window_samples"]:
                        STATS["stall_stacks"][stack] += 1
                    for stack in STATS["window_worker_samples"]:
                        STATS["worker_stacks"][stack] += 1
                STATS["pending"] = None
                STATS["window_samples"] = []
                STATS["window_worker_samples"] = []
        return r

    Screen._compositor_refresh = refresh

    t = threading.Thread(target=_sampler, daemon=True, name="latency-sampler")
    t.start()
    STATS["sampler_thread_id"] = t.ident


def _report() -> None:
    ev = STATS["events"]
    if not ev:
        print("\nno interactions recorded")
        return
    print("\n" + "=" * 70)
    print("INTERACTION -> PAINT (v2)")
    print("=" * 70)
    ms = sorted(e[1] for e in ev)
    p95 = ms[int(len(ms) * 0.95)] if len(ms) > 1 else ms[0]
    print(f"\nall interactions (n={len(ev)}): median {statistics.median(ms):6.1f} ms  "
          f"p95 {p95:6.1f} ms  max {max(ms):7.1f} ms")
    print(f"totals: {STATS['full']} full redraws, {STATS['partial']} partial, "
          f"{STATS['bytes']:,} bytes written")

    slow = sorted(ev, key=lambda e: -e[1])[:8]
    print("\nslowest 8 interactions (what was clicked):")
    for label, dur, full, nbytes in slow:
        print(f"  {dur:8.1f} ms  {label[:44]:44} full={full} bytes={nbytes:,}")

    if STATS["stall_stacks"]:
        print("\nWHAT THE MAIN THREAD WAS DOING during stalls >100ms")
        print("(each sample ~10ms; innermost frame first)")
        for stack, n in STATS["stall_stacks"].most_common(8):
            print(f"\n  ~{n * 10:>5d} ms total, {n} samples:")
            for line in stack:
                print(f"      {line}")
    else:
        print("\nno stalls >=100ms sampled")
    if STATS["worker_stacks"]:
        print("\nWHAT OTHER THREADS WERE DOING while the main thread waited")
        for stack, n in STATS["worker_stacks"].most_common(8):
            print(f"\n  ~{n * 10:>5d} ms total, {n} samples:")
            for line in stack:
                print(f"      {line}")
    print("=" * 70)


def main() -> None:
    _install()
    from tldw_chatbook.app import main_cli_runner
    try:
        main_cli_runner()
    finally:
        _report()


if __name__ == "__main__":
    main()
