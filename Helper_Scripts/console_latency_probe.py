"""In-terminal Console latency probe (TASK-26834's baseline instrument).

Runs the real app with four instruments attached:

* every Click/Key is stamped at ``Screen._forward_event``; its window closes
  at the next ``_compositor_refresh`` -- interaction -> first paint;
* a daemon thread samples the MAIN thread's stack every 10ms while a window
  is open; stalls >100ms report the stacks that occupied them;
* when the main thread is idle-in-select, every OTHER thread is sampled too,
  so waiting is attributed, not just noticed;
* every ``set_timer``/``call_later``/``call_after_refresh`` issued inside a
  window is recorded, so a stall with the whole process idle decomposes into
  named scheduling hops (the SCHEDULING TIMELINE section).

Run it in the terminal you actually use, poke the interactions that feel
slow, quit with ctrl+q; the report prints on exit. How to read it -- including
which idle-thread stacks are noise and the known framing artifact where a
click with NO visible response records as slow -- is on TASK-26834. Window
semantics: closes at the FIRST paint after the event, so early-ack
interactions under-report; medians are optimistic, the tail is trustworthy.

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
    "window_trace": [],
    "slow_traces": [],
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

    # v4: while a window is open, record every scheduling call, so a wait
    # with the whole process idle decomposes into named timer/deferred hops.
    from textual.message_pump import MessagePump

    def _cb_name(callback) -> str:
        target = getattr(callback, "__func__", callback)
        name = getattr(target, "__qualname__", None) or repr(target)
        return name[:60]

    def _trace(owner, api, detail) -> None:
        with _LOCK:
            p = STATS["pending"]
            if p is None:
                return
            offset = (time.perf_counter() - p[1]) * 1000
            STATS["window_trace"].append(
                (offset, f"{api} {detail} by {type(owner).__name__}")
            )

    o_set_timer = MessagePump.set_timer

    def set_timer(self, delay, callback=None, *a, **k):
        _trace(self, "set_timer", f"{delay:.3f}s -> {_cb_name(callback)}")
        return o_set_timer(self, delay, callback, *a, **k)

    MessagePump.set_timer = set_timer

    o_call_later = MessagePump.call_later

    def call_later(self, callback, *a, **k):
        _trace(self, "call_later", _cb_name(callback))
        return o_call_later(self, callback, *a, **k)

    MessagePump.call_later = call_later

    o_car = MessagePump.call_after_refresh

    def call_after_refresh(self, callback, *a, **k):
        _trace(self, "call_after_refresh", _cb_name(callback))
        return o_car(self, callback, *a, **k)

    MessagePump.call_after_refresh = call_after_refresh

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
                STATS["window_trace"] = []
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
                if dur >= 100:
                    STATS["slow_traces"].append(
                        (dur, label, list(STATS["window_trace"])[:40])
                    )
                    STATS["slow_traces"].sort(key=lambda e: -e[0])
                    del STATS["slow_traces"][8:]
                STATS["pending"] = None
                STATS["window_samples"] = []
                STATS["window_worker_samples"] = []
                STATS["window_trace"] = []
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
    if STATS["slow_traces"]:
        print("\nSCHEDULING TIMELINE of the slowest windows")
        print("(+ms after the input event; what was scheduled, by whom)")
        for dur, label, trace in STATS["slow_traces"][:5]:
            print(f"\n  {label[:52]}  ({dur:.0f} ms to first paint)")
            if not trace:
                print("      (nothing scheduled inside the window)")
            for offset, line in trace[:14]:
                print(f"      +{offset:7.1f} ms  {line}")
            if len(trace) > 14:
                print(f"      ... {len(trace) - 14} more")
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
