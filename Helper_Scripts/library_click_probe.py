"""Headless Library-screen click-latency probe.

Boots the real LibraryScreen in a pilot, then clicks around the rail modes,
recording for each click:
  * wall time from press() to settle,
  * how many whole-screen recompose(recompose=True) calls fired,
  * cumulative time spent inside Compositor.render_full_update, and
  * main-thread stall stacks sampled at 5ms while the click was in flight.

Reproduces recompose / DB / compute stalls; terminal-render bytes are absent
(headless), which is fine -- a click that freezes then recovers is main-thread
compute, not output volume. Headless numbers therefore EXCLUDE terminal-write
cost -- this is an instrument for main-thread stalls, not end-to-end latency.

Checked in by the Library decomposition foundation plan
(Docs/superpowers/plans/2026-09-01-library-decomposition-foundation.md, Task 4)
per the design spec
(Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md, "PR 0b").
Use it for before/after evidence around each extraction PR: a pure move must
not move these numbers outside noise (recipe doc:
backlog/docs/library-decomposition-recipe.md).

Usage: .venv/bin/python Helper_Scripts/library_click_probe.py
"""
from __future__ import annotations

import asyncio
import collections
import os
import sys
import threading
import time

# Repo-root-relative resolution (mirrors Helper_Scripts/console_latency_probe.py) --
# no absolute, machine-specific paths.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tests.UI.app_factory import _build_test_app  # noqa: E402
from Tests.UI.test_library_shell import (  # noqa: E402
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _two_media_items,
    _wait_for_library_shell,
)
from textual.widgets import Button  # noqa: E402
from textual._compositor import Compositor  # noqa: E402
from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen  # noqa: E402

STATE = {
    "recompose_targets": [],   # (typename,) per recompose(recompose=True)
    "full_updates": 0,
    "full_update_ms": 0.0,
    "pending": None,
    "samples": collections.Counter(),
    "main_tid": threading.get_ident(),
    "sampling": True,
    "idle_samples": 0,
    "busy_samples": 0,
}
_LOCK = threading.Lock()


def _fmt(frame):
    c = frame.f_code
    fn = c.co_filename
    tag = "app:" if "/tldw_chatbook/" in fn else ("lib:" if "site-packages" in fn or "/textual/" in fn else "")
    return f"{tag}{fn.split('/')[-1]}:{frame.f_lineno}:{c.co_name}"


def _stack(frame, n=8):
    out = []
    d = 0
    while frame is not None and d < 30:
        out.append(_fmt(frame))
        frame = frame.f_back
        d += 1
    return tuple(out[:n])


def _is_idle(stack):
    top = stack[0] if stack else ""
    return "selectors.py" in top and ":select" in top


def _sampler():
    while STATE["sampling"]:
        time.sleep(0.005)
        with _LOCK:
            if STATE["pending"] is None:
                continue
        frames = sys._current_frames()
        main = frames.get(STATE["main_tid"])
        if main is not None:
            st = _stack(main)
            with _LOCK:
                STATE["samples"][st] += 1
                if _is_idle(st):
                    STATE["idle_samples"] += 1
                else:
                    STATE["busy_samples"] += 1


def _install():
    orig_refresh = BaseAppScreen.refresh

    def refresh(self, *a, **k):
        if k.get("recompose"):
            with _LOCK:
                STATE["recompose_targets"].append(type(self).__name__)
        return orig_refresh(self, *a, **k)

    BaseAppScreen.refresh = refresh

    orig_full = Compositor.render_full_update

    def full(self, *a, **k):
        t0 = time.perf_counter()
        r = orig_full(self, *a, **k)
        with _LOCK:
            STATE["full_updates"] += 1
            STATE["full_update_ms"] += (time.perf_counter() - t0) * 1000
        return r

    Compositor.render_full_update = full

    # Count widget mounts and removes (mount-storm signal).
    from textual.app import App

    STATE["mounts"] = 0
    STATE["removes"] = 0

    o_register = App._register

    def register(self, parent, *widgets, **k):
        r = o_register(self, parent, *widgets, **k)
        with _LOCK:
            STATE["mounts"] += len(widgets)
        return r

    App._register = register

    o_unregister = App._unregister

    def unregister(self, widget, *a, **k):
        with _LOCK:
            STATE["removes"] += 1
        return o_unregister(self, widget, *a, **k)

    App._unregister = unregister


async def _settle(pilot, passes=40, delay=0.01):
    for _ in range(passes):
        await pilot.pause(delay)


async def _click(screen, pilot, label, button_id):
    matches = screen.query(f"#{button_id}")
    if not matches:
        return {"label": label, "missing": True}
    with _LOCK:
        STATE["pending"] = label
        STATE["recompose_targets"] = []
        fu0 = STATE["full_updates"]
        fm0 = STATE["full_update_ms"]
        mo0 = STATE["mounts"]
        rm0 = STATE["removes"]
        STATE["samples"] = collections.Counter()
        STATE["idle_samples"] = 0
        STATE["busy_samples"] = 0
    nodes_before = len(screen.query("*"))
    t0 = time.perf_counter()
    last = t0
    max_gap = 0.0
    matches.first(Button).press()
    # condition-based settle: quiet when no full-update for ~120ms, capped at 3s
    quiet = 0
    fu_seen = STATE["full_updates"]
    for _ in range(600):
        await pilot.pause(0.005)
        now = time.perf_counter()
        gap = (now - last) * 1000
        if gap > max_gap:
            max_gap = gap
        last = now
        with _LOCK:
            cur_fu = STATE["full_updates"]
        if cur_fu == fu_seen:
            quiet += 1
        else:
            quiet = 0
            fu_seen = cur_fu
        if quiet >= 24 and (now - t0) > 0.15:  # ~120ms quiet
            break
        if now - t0 > 3.0:
            break
    dur = (time.perf_counter() - t0) * 1000
    nodes_after = len(screen.query("*"))
    with _LOCK:
        rc = list(STATE["recompose_targets"])
        fu = STATE["full_updates"] - fu0
        fm = STATE["full_update_ms"] - fm0
        mo = STATE["mounts"] - mo0
        rm = STATE["removes"] - rm0
        samples = STATE["samples"].most_common(5)
        idle = STATE["idle_samples"]
        busy = STATE["busy_samples"]
        STATE["pending"] = None
    return {
        "label": label, "dur": dur, "max_gap": max_gap, "rc": len(rc),
        "fu": fu, "fm": fm, "mounts": mo, "removes": rm,
        "nodes_before": nodes_before, "nodes_after": nodes_after,
        "targets": rc, "samples": samples,
        "idle_ms": idle * 5, "busy_ms": busy * 5,
    }


async def main():
    _install()
    t = threading.Thread(target=_sampler, daemon=True)
    t.start()

    base = _build_test_app()
    _seed_conversations(
        base, _two_conversations(), notes=None, media=_two_media_items()
    )
    app = LibraryHarness(base)
    results = []
    async with app.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(app)
        await _wait_for_library_shell(screen, pilot)
        await _settle(pilot, passes=30)
        # disclose full rail if compact
        if screen.query("#library-rail-explore-all"):
            screen.query_one("#library-rail-explore-all", Button).press()
            await _settle(pilot, passes=30)

        clicks = [
            ("media (switch-in)", "library-row-browse-media"),
            ("media (re-click same)", "library-row-browse-media"),
            ("media (re-click same2)", "library-row-browse-media"),
            ("notes (switch)", "library-row-browse-notes"),
            ("notes (re-click same)", "library-row-browse-notes"),
            ("media (switch-back)", "library-row-browse-media"),
            ("notes (switch)2", "library-row-browse-notes"),
            ("media (switch-back)2", "library-row-browse-media"),
        ]
        for label, bid in clicks:
            res = await _click(screen, pilot, label, bid)
            results.append(res)

    STATE["sampling"] = False
    _report(results)


def _report(results):
    print("\n" + "=" * 92)
    print("LIBRARY CLICK LATENCY (headless; terminal-write bytes NOT measured)")
    print("=" * 92)
    hdr = f"{'interaction':22} {'settle':>7} {'maxgap':>7} {'recmp':>5} {'full':>4} {'fullms':>6} {'mnt':>4} {'busyms':>6} {'waitms':>6} {'nodes':>6}"
    print("\n" + hdr)
    worst = None
    for r in results:
        if r.get("missing"):
            print(f"{r['label']:22} {'--- missing ---':>30}")
            continue
        flag = "  <== FREEZE" if r["max_gap"] > 200 else ("  <== slow" if r["dur"] > 250 else "")
        print(f"{r['label']:22} {r['dur']:7.0f} {r['max_gap']:7.0f} {r['rc']:5d} "
              f"{r['fu']:4d} {r['fm']:6.1f} {r['mounts']:4d} {r['busy_ms']:6d} {r['idle_ms']:6d} "
              f"{r['nodes_after']:6d}{flag}")
        if worst is None or r["max_gap"] > worst["max_gap"]:
            worst = r
    if worst:
        print(f"\nLONGEST MAIN-THREAD BLOCK: {worst['label']}  max_gap={worst['max_gap']:.0f} ms "
              f"(settle {worst['dur']:.0f} ms, {worst['mounts']} mounts / {worst['removes']} removes, "
              f"{worst['nodes_before']}->{worst['nodes_after']} nodes)")
        print("  what the main thread was doing (5ms samples):")
        for stack, n in worst["samples"]:
            tag = "  [IDLE-WAIT]" if ("selectors.py" in stack[0] and ":select" in stack[0]) else ""
            print(f"    ~{n*5} ms{tag}:")
            for line in stack[:7]:
                print(f"       {line}")
    print("=" * 92)


if __name__ == "__main__":
    asyncio.run(main())
