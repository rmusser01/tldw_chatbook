"""Headless Library rail-mode-switch TEARDOWN probe (phase-C instrument).

Sibling of ``Helper_Scripts/library_click_probe.py``. Where that probe reports
one settle time and one aggregate mount count per click, this one answers the
three questions phase C's mechanism decision needs:

  * **what** a rail-mode switch destroys and rebuilds -- mount and unmount
    counts broken down BY WIDGET CLASS and BY SCREEN REGION (rail / canvas /
    nav bar / footer / chrome), per click;
  * **where the freeze goes** -- the click's main-thread work split into
    additive buckets: widget construction, DOM registration, CSS restyle,
    compositor reflow, render/paint prep; and
  * **which switch path ran** -- the targeted canvas swap
    (``_replace_library_browse_canvas``) or a whole-screen
    ``Widget.recompose()``. ``LIBRARY_TEARDOWN_TRACE=1`` prints the per-click
    causal timeline behind that answer.

Same headless caveat as the click probe: terminal-write bytes are not produced,
so these are main-thread compute-and-DOM numbers, not end-to-end latency.

## How the two headline numbers are built, and why they are trustworthy

``block`` is the click probe's ``max_gap``: the click loop polls
``pilot.pause(5 ms)`` and the longest late return is the longest stretch during
which the main thread never yielded -- the freeze.

``cpu`` is the sum of the bucket table, and the buckets are wrappers on
**synchronous** callables only (``textual.compose``, ``App._register``,
``Stylesheet.apply``, ``Compositor.reflow``/``render_*``), nested-correctly via
a self-time stack. Synchronous frames cannot interleave, so their self times are
additive: ``cpu`` is measured main-thread work per click, and unlike ``block``
it does not depend on how the work happened to be split across frames.

A "total blocked ms" figure (summing every late poll, not just the longest) was
built here first and then DROPPED: ``pilot.pause`` waits for the message queue,
so an idle click still accumulates 380 ms of "blocked" time. It measured the
harness, not the app.

**Do not add a wrapper on an `async` frame to this table.** An async frame's
wall-clock span includes every idle tick the event loop served while it was
awaiting, so ``_compose``'s "self time" came out at 233 ms on a click whose
longest block was 90 ms -- an artifact, not a cost. The async spans that ARE
reported (handler / targeted-swap / screen-recompose) are printed separately
and labelled as spans, purely to show WHICH path ran and how long the click's
work was spread out.

## Two instrument facts the older click probe gets wrong

  * **Unmounts do NOT go through ``App._unregister`` in Textual 8.2.8.**
    ``App._prune`` posts ``Prune`` to each node and the node finalises its own
    removal in ``Widget._message_loop_exit`` (``self.app._registry.discard``),
    so a probe hooked on ``_unregister`` reports zero removals forever -- which
    is exactly what the click probe's "N mounts / 0 removes" line has always
    printed. This one hooks ``_message_loop_exit``.
  * **A whole-screen recompose does NOT always go through
    ``refresh(recompose=True)``.** ``LibraryScreen`` awaits
    ``Widget.recompose()`` directly on the rail-switch path, which the click
    probe's ``recmp`` column cannot see -- which is why that column reads 0 for
    clicks that rebuild the entire screen. This one counts both.

Written for the Library phase-C media-graduation plan
(``Docs/superpowers/plans/2026-09-08-library-phase-c-media-graduation.md``,
Task 1) as the evidence instrument behind the resident-canvas design record in
``Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md``.
Run it from a SCRATCH worktree, per recipe §9's same-checkout-location rule
(``backlog/docs/library-decomposition-recipe.md``).

Usage: .venv/bin/python Helper_Scripts/library_switch_teardown_probe.py
"""
from __future__ import annotations

import asyncio
import collections
import json
import os
import sys
import time

import textual.app as textual_app
import textual.widget as textual_widget
from textual._compositor import Compositor
from textual.css.stylesheet import Stylesheet
from textual.widget import Widget
from textual.widgets import Button

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

#: Settle-poll cadence, mirroring the click probe's own constant so the
#: ``max_gap`` figures from the two scripts stay comparable.
_POLL_SECONDS = 0.005
#: How long to keep polling after the press before calling the click done.
#: The click probe's adaptive quiet-window settle mixes ~120 ms of deliberate
#: idle into its "settle" number; this probe measures work, so it polls a fixed
#: window and reports the bucket table instead.
_CLICK_WINDOW_SECONDS = 0.7

STATE: dict = {
    "pending": None,
    "mounts": collections.Counter(),
    "mount_regions": collections.Counter(),
    "unmounts": collections.Counter(),
    "unmount_regions": collections.Counter(),
    "self_ms": collections.Counter(),
    "incl_ms": collections.Counter(),
    "calls": collections.Counter(),
    "stack": [],
    "recompose_refresh": 0,
    "trace": [],
}

#: ``LIBRARY_TEARDOWN_TRACE=1`` prints a per-click causal timeline (which
#: rebuild ran, in what order, and how many widgets each one mounted). That
#: ordering is what tells a resident-canvas design how many times the canvas
#: is rebuilt per switch -- the aggregate counters cannot.
_TRACE = bool(os.environ.get("LIBRARY_TEARDOWN_TRACE"))

#: Additive CPU buckets: every one wraps a SYNCHRONOUS callable (see the
#: module docstring for why that restriction is load-bearing).
_CPU_PHASES = ("construct", "register", "css", "reflow", "render")
#: Wall-clock spans over async frames. Diagnostic only -- NOT additive, and
#: they include idle ticks the loop served mid-await.
_SPAN_PHASES = ("handler", "targeted-swap", "screen-recompose", "teardown")


def _enter() -> tuple[float, list]:
    """Push a timing frame. Returns (t0, child-time accumulator)."""
    frame = [0.0]
    STATE["stack"].append(frame)
    return time.perf_counter(), frame


def _exit(phase: str, t0: float, frame: list, additive: bool) -> None:
    elapsed = (time.perf_counter() - t0) * 1000
    try:
        STATE["stack"].remove(frame)
    except ValueError:  # pragma: no cover - defensive
        pass
    if STATE["pending"] is None:
        return
    STATE["incl_ms"][phase] += elapsed
    STATE["self_ms"][phase] += elapsed - frame[0]
    STATE["calls"][phase] += 1
    if additive and STATE["stack"]:
        STATE["stack"][-1][0] += elapsed


def _wrap_sync(owner, name: str, phase: str):
    original = getattr(owner, name)

    def wrapper(*args, **kwargs):
        t0, frame = _enter()
        try:
            return original(*args, **kwargs)
        finally:
            _exit(phase, t0, frame, additive=True)

    # Preserve marker attributes: the app's ``textual_css_fastpath`` tags its
    # own ``Stylesheet.apply`` with a private marker and re-checks it before
    # installing, so a bare wrapper would hide the tag and invite a second
    # install on top of this probe's.
    for attribute, value in vars(original).items():
        try:
            setattr(wrapper, attribute, value)
        except Exception:  # pragma: no cover - defensive
            pass
    setattr(owner, name, wrapper)
    return wrapper


def _wrap_async(owner, name: str, phase: str) -> None:
    original = getattr(owner, name)

    async def wrapper(*args, **kwargs):
        _trace(f"{phase}: enter")
        t0, frame = _enter()
        result = None
        try:
            result = await original(*args, **kwargs)
            return result
        finally:
            # additive=False: an async span must never be charged to an
            # enclosing CPU bucket -- it includes idle.
            _exit(phase, t0, frame, additive=False)
            suffix = f" -> {result!r}" if isinstance(result, bool) else ""
            _trace(f"{phase}: exit{suffix}")

    setattr(owner, name, wrapper)


#: Region attribution, checked innermost-first while walking a new widget's
#: ancestor chain. The DOM these match (measured on the media route, 119
#: nodes) nests as: LibraryScreen > MainNavigationBar / Container#screen-content
#: > Horizontal#library-shell-grid > the route's reader shell > LibraryRail +
#: Vertical#library-canvas > LibraryMediaCanvas + LibraryMediaViewer, plus
#: AppFooterStatus#screen-footer-status. Bucketing on the OUTERMOST id instead
#: would put the rail, the canvas and the header all in one "screen-content"
#: heap, which is exactly the distinction the residency decision turns on.
#:
#: Phase C note: the browse routes' shell id is now
#: ``library-browse-reader-shell`` (Media and Notes share ONE resident shell).
#: Both ids are listed so a run against a PRE-phase-C commit and a run against
#: a post-phase-C one bucket the same widgets into the same row -- this probe
#: is the paired-comparison instrument, so a row that silently changes meaning
#: between arms would be worse than useless. Only one of the two can ever be
#: mounted in a single run.
_REGION_RULES = (
    ("library-media-canvas", "canvas (media)"),
    ("library-notes-canvas", "canvas (notes)"),
    ("library-canvas", "canvas host"),
    ("library-media-viewer", "media viewer"),
    ("library-rail", "rail"),
    ("library-browse-reader-shell", "reader shell (other)"),
    ("library-media-reader-shell", "reader shell (other)"),
    ("library-shell-grid", "shell grid (other)"),
    ("screen-footer-status", "footer"),
    ("nav-destination-strip", "nav bar"),
    ("screen-content", "screen chrome"),
)


def _region_of(widget) -> str:
    """Which screen region a newly mounted widget landed in.

    Walks the ancestor chain from the widget outwards and returns the first
    match in ``_REGION_RULES``; falls back to the outermost id-bearing
    ancestor's class name so an unclassified mount is visible rather than
    silently pooled.
    """
    node = widget
    depth = 0
    while node is not None and depth < 40:
        node_id = getattr(node, "id", None)
        if node_id:
            for candidate, label in _REGION_RULES:
                if node_id == candidate:
                    return label
        node = getattr(node, "_parent", None)
        depth += 1
    return f"(unclassified: {type(widget).__name__})"


def _canvas_classes():
    """The Library canvas widgets whose ``sync_state`` this probe traces."""
    from tldw_chatbook.Widgets.Library import (
        LibraryMediaCanvas,
        LibraryMediaTrashCanvas,
        LibraryNotesCanvas,
    )

    return (LibraryMediaCanvas, LibraryNotesCanvas, LibraryMediaTrashCanvas)


def _trace(event: str) -> None:
    """Append a timestamped line to the current click's causal trace."""
    if STATE["pending"] is None or not _TRACE:
        return
    STATE["trace"].append((time.perf_counter(), event))


def _install(screen_class) -> None:
    from textual.app import App

    original_register = App._register

    def register(self, parent, *widgets, **kwargs):
        depth = len(STATE["stack"])
        t0, frame = _enter()
        try:
            result = original_register(self, parent, *widgets, **kwargs)
        finally:
            _exit("register", t0, frame, additive=True)
        if STATE["pending"] is not None:
            for widget in widgets:
                STATE["mounts"][type(widget).__name__] += 1
                STATE["mount_regions"][_region_of(widget)] += 1
            if depth == 0:
                parent_id = getattr(parent, "id", None) or type(parent).__name__
                _trace(f"register  -> {parent_id}  (+{len(widgets)} top-level)")
        return result

    App._register = register

    # Textual 8.2.8 finalises removal here, NOT in App._unregister -- see the
    # module docstring.
    original_exit = Widget._message_loop_exit

    async def message_loop_exit(self, *args, **kwargs):
        if STATE["pending"] is not None:
            STATE["unmounts"][type(self).__name__] += 1
            # Read the region BEFORE the body runs: it detaches the node from
            # its parent as its final act, and an orphan has no region.
            STATE["unmount_regions"][_region_of(self)] += 1
        return await original_exit(self, *args, **kwargs)

    Widget._message_loop_exit = message_loop_exit

    # Widget CONSTRUCTION: ``textual.compose.compose`` drains a widget's
    # ``compose()`` generator, which is where every child object is built.
    # It is imported by value into widget.py and app.py, so both module
    # globals must be rebound -- patching ``textual.compose.compose`` alone
    # would measure nothing.
    original_compose = textual_widget.compose

    def compose_fn(*args, **kwargs):
        t0, frame = _enter()
        try:
            return original_compose(*args, **kwargs)
        finally:
            _exit("construct", t0, frame, additive=True)

    textual_widget.compose = compose_fn
    textual_app.compose = compose_fn

    _wrap_sync(Stylesheet, "apply", "css")
    _wrap_sync(Compositor, "reflow", "reflow")
    _wrap_sync(Compositor, "render_full_update", "render")
    _wrap_sync(Compositor, "render_update", "render")

    # Canvas-scoped rebuilds. ``sync_state`` is the "targeted" (Tier 2) update
    # every high-frequency Library canvas exposes, and its body ends in
    # ``self.refresh(recompose=True)`` -- a canvas-scoped recompose that still
    # tears down and rebuilds the canvas's ENTIRE child list. Wrapped on the
    # widget classes rather than on ``canvas_sync._sync_library_canvas``,
    # because that dispatcher is imported by value into the screen and ten
    # controllers, so rebinding the module global would measure none of them.
    for canvas_class in _canvas_classes():
        original_sync = canvas_class.sync_state

        def sync_state(self, *args, _original=original_sync, **kwargs):
            _trace(f"{type(self).__name__}.sync_state (canvas-scoped recompose)")
            return _original(self, *args, **kwargs)

        canvas_class.sync_state = sync_state

    # Async spans -- diagnostic only.
    _wrap_async(Widget, "remove_children", "teardown")
    _wrap_async(screen_class, "_select_library_rail_row", "handler")
    _wrap_async(screen_class, "_replace_library_browse_canvas", "targeted-swap")
    _wrap_async(screen_class, "recompose", "screen-recompose")

    original_refresh = screen_class.refresh

    def refresh(self, *args, **kwargs):
        if kwargs.get("recompose") and STATE["pending"] is not None:
            STATE["recompose_refresh"] += 1
        return original_refresh(self, *args, **kwargs)

    screen_class.refresh = refresh


async def _settle(pilot, passes=40, delay=0.01):
    for _ in range(passes):
        await pilot.pause(delay)


async def _click(screen, pilot, label, button_id):
    matches = screen.query(f"#{button_id}")
    if not matches:
        return {"label": label, "missing": True}
    for key in (
        "mounts",
        "mount_regions",
        "unmounts",
        "unmount_regions",
        "self_ms",
        "incl_ms",
        "calls",
    ):
        STATE[key] = collections.Counter()
    STATE["recompose_refresh"] = 0
    STATE["stack"] = []
    STATE["trace"] = []
    STATE["pending"] = label
    nodes_before = len(screen.query("*"))
    t0 = time.perf_counter()
    last = t0
    max_gap = 0.0
    matches.first(Button).press()
    while True:
        await pilot.pause(_POLL_SECONDS)
        now = time.perf_counter()
        max_gap = max(max_gap, (now - last) * 1000)
        last = now
        if now - t0 > _CLICK_WINDOW_SECONDS:
            break
    nodes_after = len(screen.query("*"))
    STATE["pending"] = None
    return {
        "label": label,
        "max_gap": max_gap,
        "recompose_refresh": STATE["recompose_refresh"],
        "screen_recompose": STATE["calls"].get("screen-recompose", 0),
        "targeted_swap": STATE["calls"].get("targeted-swap", 0),
        "mounts": dict(STATE["mounts"]),
        "mount_regions": dict(STATE["mount_regions"]),
        "unmounts": dict(STATE["unmounts"]),
        "unmount_regions": dict(STATE["unmount_regions"]),
        "self_ms": dict(STATE["self_ms"]),
        "incl_ms": dict(STATE["incl_ms"]),
        "calls": dict(STATE["calls"]),
        "nodes_before": nodes_before,
        "nodes_after": nodes_after,
        "trace": [(round((t - t0) * 1000, 1), event) for t, event in STATE["trace"]],
    }


def _report(results) -> None:
    print("\n" + "=" * 104)
    print("LIBRARY RAIL-MODE-SWITCH TEARDOWN (headless; terminal-write bytes NOT measured)")
    print("=" * 104)
    header = (
        f"{'interaction':24} {'block':>6} {'cpu':>6} {'mnt':>5} {'unmnt':>6} "
        f"{'swap':>5} {'rcmp':>5} {'refr':>5} {'nodes':>6}"
    )
    print("\n" + header)
    for result in results:
        if result.get("missing"):
            print(f"{result['label']:24} {'--- missing ---':>30}")
            continue
        cpu = sum(result["self_ms"].get(phase, 0.0) for phase in _CPU_PHASES)
        print(
            f"{result['label']:24} {result['max_gap']:6.0f} {cpu:6.0f} "
            f"{sum(result['mounts'].values()):5d} {sum(result['unmounts'].values()):6d} "
            f"{result['targeted_swap']:5d} {result['screen_recompose']:5d} "
            f"{result['recompose_refresh']:5d} {result['nodes_after']:6d}"
        )
    print(
        "\n  block  = longest single main-thread block (ms) -- THE FREEZE"
        "\n  cpu    = sum of the additive synchronous buckets = main-thread work per click"
        "\n  swap   = _replace_library_browse_canvas calls (targeted path)"
        "\n  rcmp   = awaited Widget.recompose() -- WHOLE screen rebuild"
        "\n  refr   = refresh(recompose=True) -- what the click probe counts"
    )

    for result in results:
        if result.get("missing"):
            continue
        print("\n" + "-" * 104)
        cpu = sum(result["self_ms"].get(phase, 0.0) for phase in _CPU_PHASES)
        print(
            f"{result['label']}  (longest block {result['max_gap']:.0f} ms, "
            f"measured main-thread work {cpu:.0f} ms)"
        )
        print("  CPU buckets (synchronous, self time, ADDITIVE):")
        total = 0.0
        for phase in _CPU_PHASES:
            calls = result["calls"].get(phase, 0)
            if not calls:
                continue
            self_ms = result["self_ms"].get(phase, 0.0)
            total += self_ms
            share = 100.0 * self_ms / cpu if cpu else 0.0
            print(f"      {phase:<12} {self_ms:8.1f} ms  ({share:4.1f}% of work)  x{calls}")
        print(f"      {'SUM':<12} {total:8.1f} ms")
        spans = [
            f"{phase} {result['incl_ms'].get(phase, 0.0):.0f}ms x{result['calls'][phase]}"
            for phase in _SPAN_PHASES
            if result["calls"].get(phase)
        ]
        print("  async spans (wall-clock, include idle -- NOT additive): " + ", ".join(spans))
        mounts = collections.Counter(result["mounts"])
        unmounts = collections.Counter(result["unmounts"])
        regions = collections.Counter(result["mount_regions"])
        print(f"  mounted {sum(mounts.values())} widgets by region:")
        for name, count in regions.most_common(10):
            print(f"      {count:4d}  {name}")
        print(f"  mounted {sum(mounts.values())} widgets, {len(mounts)} classes:")
        for name, count in mounts.most_common(10):
            print(f"      {count:4d}  {name}")
        unmount_regions = collections.Counter(result["unmount_regions"])
        if result.get("trace"):
            print("  causal trace (ms after press):")
            for offset, event in result["trace"]:
                print(f"      {offset:7.1f}  {event}")
        print(f"  unmounted {sum(unmounts.values())} widgets by region:")
        for name, count in unmount_regions.most_common(10):
            print(f"      {count:4d}  {name}")
        print(f"  unmounted {sum(unmounts.values())} widgets, {len(unmounts)} classes:")
        for name, count in unmounts.most_common(10):
            print(f"      {count:4d}  {name}")
    print("=" * 104)


async def main() -> None:
    """Boot the Library harness, replay the rail clicks, print the teardown table."""
    base = _build_test_app()
    _seed_conversations(base, _two_conversations(), notes=None, media=_two_media_items())
    app = LibraryHarness(base)
    results = []
    async with app.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(app)
        _install(type(screen))
        await _wait_for_library_shell(screen, pilot)
        await _settle(pilot, passes=30)
        if screen.query("#library-rail-explore-all"):
            screen.query_one("#library-rail-explore-all", Button).press()
            await _settle(pilot, passes=30)

        clicks = [
            ("media (switch-in)", "library-row-browse-media"),
            ("media (re-click same)", "library-row-browse-media"),
            ("notes (switch)", "library-row-browse-notes"),
            ("notes (re-click same)", "library-row-browse-notes"),
            ("media (switch-back)", "library-row-browse-media"),
        ]
        for label, button_id in clicks:
            results.append(await _click(screen, pilot, label, button_id))
            await _settle(pilot, passes=20)

    _report(results)
    if os.environ.get("LIBRARY_TEARDOWN_JSON"):
        # Diagnostic-only (a dev-run probe, not shipped runtime), but the
        # output path is environment-controlled, so bound it to the working
        # tree: validate_path rejects ``..`` traversal and absolute-elsewhere
        # targets before it is opened for write (Qodo #1).
        from tldw_chatbook.Utils.path_validation import validate_path  # noqa: PLC0415

        destination = validate_path(
            os.environ["LIBRARY_TEARDOWN_JSON"], os.getcwd(), allow_hidden=True
        )
        with open(destination, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=1)


if __name__ == "__main__":
    asyncio.run(main())
