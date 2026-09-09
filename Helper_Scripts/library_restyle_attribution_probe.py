"""Headless Library rail-switch RESTYLE-ATTRIBUTION probe (phase-C task 2.5).

Third instrument in the phase-C set, after ``library_click_probe.py`` (settle
per click) and ``library_switch_teardown_probe.py`` (mounts/unmounts by region
plus the additive CPU buckets). Those two established that after the resident
canvas landed, **CSS restyle is the largest remaining CPU bucket and its size
is NOT proportional to mounts** -- 26 mounts and zero canvas rebuilds still ran
459 ``Stylesheet.apply`` calls, more than a 81-mount switch-back did. That
falsified the "the storm owns the freeze" inference and left the split
unmeasured, which is what this script measures.

## The question, and how it is answered

For every ``Stylesheet.apply`` call in a click, WHAT TRIGGERED IT? Textual 8.2.8
has exactly five entry points that reach ``apply``:

  * ``App._register`` -- one apply per newly mounted widget (``app.py``'s
    ``apply_stylesheet = self.stylesheet.apply`` loop). **Mount-proportional.**
  * ``App.update_styles(node)`` -- ``stylesheet.update_nodes(node.walk_children
    (with_self=True))``: one apply for EVERY DESCENDANT of ``node``. Reached
    from ``DOMNode.update_node_styles``, i.e. from every ``add_class`` /
    ``remove_class`` / ``set_class`` / ``toggle_class`` that actually changes
    the class set, and from reactive style watchers. **Subtree-proportional,
    not mount-proportional** -- this is the "full-tree restyle" the record
    could not name.
  * ``Screen._update_focus_styles`` -- ``update_nodes`` over the subtree of the
    nearest ancestor with ``:focus-within`` styles, on every focus move.
  * ``App.refresh_css`` -- ``stylesheet.update(app)``: the whole tree.
  * ``Widget._cover`` -- one apply, for a cover widget.

plus one INTERNAL multiplier that belongs to no trigger and is easy to miss:
``Stylesheet._process_component_classes`` runs a nested ``apply`` on a throwaway
``DOMNode`` for EACH component class of the node being applied. Those nested
calls are counted separately here (``virt``) because they inflate every apply
count uniformly and would otherwise be misread as extra triggers.

Each trigger is wrapped to push a label onto a stack; ``apply`` attributes
itself to the innermost label and to its nesting depth. Timing is INCLUSIVE at
depth 0 only, which is exactly additive: ``apply`` is synchronous, its only
nested ``apply`` calls are the component-class ones, and no trigger nests
inside another (verified by the "(nested trigger)" rows this prints if one
ever does).

``update_styles`` calls are additionally bucketed by ORIGINATOR: the first
frame outside ``textual/`` walking out from the call, so the table names the
line of application code that flipped the class. That is the actionable half --
a count alone says restyle is expensive, an originator says which line to fix.

Same headless caveat as its two siblings: terminal-write bytes are not
produced, so these are main-thread compute-and-DOM numbers.

Usage: .venv/bin/python Helper_Scripts/library_restyle_attribution_probe.py
Optional: LIBRARY_RESTYLE_JSON=/path/to.json to dump the raw per-click record.
"""
from __future__ import annotations

import asyncio
import collections
import json
import os
import sys
import time

from textual.app import App
from textual.css.stylesheet import Stylesheet
from textual.screen import Screen
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

_POLL_SECONDS = 0.005
_CLICK_WINDOW_SECONDS = 0.7

STATE: dict = {
    "pending": None,
    # trigger label -> apply calls / inclusive ms / nested (component) applies
    "applies": collections.Counter(),
    "apply_ms": collections.Counter(),
    "virtual": collections.Counter(),
    # (trigger, originator) -> count of TRIGGER invocations and applies
    "origins": collections.Counter(),
    "origin_applies": collections.Counter(),
    "origin_ms": collections.Counter(),
    "trigger_calls": collections.Counter(),
    "stack": [],
    "depth": 0,
    "nested_trigger": collections.Counter(),
    "fires": [],
    "t0": 0.0,
}

#: ``LIBRARY_RESTYLE_TRACE=1`` prints one line per restyle FIRE (which node's
#: subtree, how many applies, how long, how far into the click). The aggregate
#: table says a class flip is expensive; the trace says whether the same flip
#: happens twice per click and is therefore removable.
_TRACE = bool(os.environ.get("LIBRARY_RESTYLE_TRACE"))

_TEXTUAL_MARKER = f"{os.sep}textual{os.sep}"
_PROBE_FILE = os.path.abspath(__file__)


def _originator(skip: int = 2) -> str:
    """The first frame outside ``textual/`` walking outwards from the caller.

    This is what turns "N applies from update_styles" into "N applies because
    THIS line flipped a class". Falls back to the outermost textual frame when
    the whole stack is inside the framework (a reactive watcher firing from the
    message pump, say), so a call is never dropped from the table.
    """
    frame = sys._getframe(skip)
    last_textual = "?"
    depth = 0
    while frame is not None and depth < 40:
        filename = frame.f_code.co_filename
        if _TEXTUAL_MARKER not in filename and filename != _PROBE_FILE:
            return (
                f"{os.path.basename(filename)}:{frame.f_lineno} "
                f"{frame.f_code.co_name}"
            )
        last_textual = (
            f"{os.path.basename(filename)}:{frame.f_lineno} {frame.f_code.co_name}"
        )
        frame = frame.f_back
        depth += 1
    return f"(textual only) {last_textual}"


def _push(label: str, origin: str) -> tuple | None:
    if STATE["pending"] is None:
        return None
    if STATE["stack"]:
        STATE["nested_trigger"][f"{STATE['stack'][-1][0]} > {label}"] += 1
    entry = (label, origin, [0])
    STATE["stack"].append(entry)
    STATE["trigger_calls"][label] += 1
    STATE["origins"][(label, origin)] += 1
    return entry


def _pop(entry) -> None:
    if entry is None:
        return
    try:
        STATE["stack"].remove(entry)
    except ValueError:  # pragma: no cover - defensive
        pass


def _node_label(node) -> str:
    node_id = getattr(node, "id", None)
    classes = " ".join(sorted(getattr(node, "classes", ()) or ()))
    return f"{type(node).__name__}#{node_id or '-'}{'.' + classes if classes else ''}"


def _wrap_trigger(owner, name: str, label: str, node_argument: int | None = None):
    original = getattr(owner, name)

    def wrapper(*args, **kwargs):
        entry = _push(label, _originator(2))
        applies_before = sum(STATE["applies"].values())
        t0 = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            _pop(entry)
            if _TRACE and STATE["pending"] is not None:
                node = (
                    args[node_argument]
                    if node_argument is not None and len(args) > node_argument
                    else None
                )
                STATE["fires"].append(
                    {
                        "at_ms": round((time.perf_counter() - STATE["t0"]) * 1000, 1),
                        "trigger": label,
                        "origin": entry[1] if entry else "?",
                        "node": _node_label(node) if node is not None else "-",
                        "applies": sum(STATE["applies"].values()) - applies_before,
                        "ms": round((time.perf_counter() - t0) * 1000, 2),
                    }
                )

    setattr(owner, name, wrapper)


def _install() -> None:
    """Wrap the five apply entry points, the component-class multiplier, apply."""
    original_apply = Stylesheet.apply

    def apply(self, node, **kwargs):
        if STATE["pending"] is None:
            return original_apply(self, node, **kwargs)
        depth = STATE["depth"]
        label, origin = (
            (STATE["stack"][-1][0], STATE["stack"][-1][1])
            if STATE["stack"]
            else ("(untriggered)", _originator(2))
        )
        STATE["depth"] = depth + 1
        t0 = time.perf_counter()
        try:
            return original_apply(self, node, **kwargs)
        finally:
            STATE["depth"] = depth
            elapsed = (time.perf_counter() - t0) * 1000
            STATE["applies"][label] += 1
            STATE["origin_applies"][(label, origin)] += 1
            if depth:
                STATE["virtual"][label] += 1
            else:
                STATE["origin_ms"][(label, origin)] += elapsed
                # Inclusive at depth 0 only: component-class applies nest
                # INSIDE this one, so depth-0 inclusive time is the whole
                # restyle cost, counted once.
                STATE["apply_ms"][label] += elapsed

    # Preserve the app's fastpath marker attributes (see the teardown probe's
    # ``_wrap_sync`` for why: ``textual_css_fastpath`` re-checks its own tag).
    for attribute, value in vars(original_apply).items():
        try:
            setattr(apply, attribute, value)
        except Exception:  # pragma: no cover - defensive
            pass
    Stylesheet.apply = apply

    _wrap_trigger(App, "_register", "mount (App._register)")
    _wrap_trigger(App, "update_styles", "class flip (App.update_styles)", 1)
    _wrap_trigger(App, "refresh_css", "refresh_css (whole tree)")
    _wrap_trigger(Screen, "_update_focus_styles", "focus move")
    _wrap_trigger(Widget, "_cover", "cover widget")


async def _settle(pilot, passes=40, delay=0.01):
    for _ in range(passes):
        await pilot.pause(delay)


async def _click(screen, pilot, label, button_id):
    matches = screen.query(f"#{button_id}")
    if not matches:
        return {"label": label, "missing": True}
    for key in (
        "applies",
        "apply_ms",
        "virtual",
        "origins",
        "origin_applies",
        "origin_ms",
        "trigger_calls",
        "nested_trigger",
    ):
        STATE[key] = collections.Counter()
    STATE["stack"] = []
    STATE["depth"] = 0
    STATE["fires"] = []
    STATE["pending"] = label
    t0 = time.perf_counter()
    STATE["t0"] = t0
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
    STATE["pending"] = None
    return {
        "label": label,
        "max_gap": max_gap,
        "applies": dict(STATE["applies"]),
        "apply_ms": dict(STATE["apply_ms"]),
        "virtual": dict(STATE["virtual"]),
        "trigger_calls": dict(STATE["trigger_calls"]),
        "origins": {f"{k[0]} | {k[1]}": v for k, v in STATE["origins"].items()},
        "origin_applies": {
            f"{k[0]} | {k[1]}": v for k, v in STATE["origin_applies"].items()
        },
        "origin_ms": {f"{k[0]} | {k[1]}": v for k, v in STATE["origin_ms"].items()},
        "nested_trigger": dict(STATE["nested_trigger"]),
        "fires": list(STATE["fires"]),
    }


def _report(results) -> None:
    print("\n" + "=" * 108)
    print("LIBRARY RAIL-SWITCH RESTYLE ATTRIBUTION -- Stylesheet.apply calls BY TRIGGER")
    print("=" * 108)
    for result in results:
        if result.get("missing"):
            print(f"{result['label']:24} --- missing ---")
            continue
        total = sum(result["applies"].values())
        total_ms = sum(result["apply_ms"].values())
        print(
            f"\n{result['label']}  (longest block {result['max_gap']:.0f} ms, "
            f"{total} apply calls, {total_ms:.0f} ms in apply)"
        )
        print(
            f"    {'trigger':38} {'fires':>6} {'applies':>8} {'virt':>6} "
            f"{'ms':>8} {'share':>7}"
        )
        for label, count in collections.Counter(result["applies"]).most_common():
            share = 100.0 * count / total if total else 0.0
            print(
                f"    {label:38} {result['trigger_calls'].get(label, 0):6d} "
                f"{count:8d} {result['virtual'].get(label, 0):6d} "
                f"{result['apply_ms'].get(label, 0.0):8.1f} {share:6.1f}%"
            )
        origins = collections.Counter(result["origin_applies"])
        print("    originators (applies caused, top 12):")
        for key, count in origins.most_common(12):
            fires = result["origins"].get(key, 0)
            ms = result["origin_ms"].get(key, 0.0)
            share = 100.0 * count / total if total else 0.0
            print(
                f"      {count:6d} applies ({share:4.1f}%) {ms:7.1f} ms "
                f"x{fires:<4d} fires  {key}"
            )
        if result.get("fires"):
            print("    restyle fires (>=3 applies), in order:")
            for fire in result["fires"]:
                if fire["applies"] < 3:
                    continue
                print(
                    f"      {fire['at_ms']:7.1f} ms  {fire['applies']:4d} applies "
                    f"{fire['ms']:6.1f} ms  {fire['node']:44} {fire['origin']}"
                )
        if result["nested_trigger"]:
            print(f"    (nested trigger pairs: {result['nested_trigger']})")
    print("=" * 108)


async def main() -> None:
    base = _build_test_app()
    _seed_conversations(base, _two_conversations(), notes=None, media=_two_media_items())
    app = LibraryHarness(base)
    results = []
    async with app.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(app)
        _install()
        await _wait_for_library_shell(screen, pilot)
        await _settle(pilot, passes=30)
        if screen.query("#library-rail-explore-all"):
            screen.query_one("#library-rail-explore-all", Button).press()
            await _settle(pilot, passes=30)

        clicks = [
            ("media (switch-in)", "library-row-browse-media"),
            ("notes (switch)", "library-row-browse-notes"),
            ("media (switch-back)", "library-row-browse-media"),
            ("notes (switch, later)", "library-row-browse-notes"),
        ]
        for label, button_id in clicks:
            results.append(await _click(screen, pilot, label, button_id))
            await _settle(pilot, passes=20)

    _report(results)
    if os.environ.get("LIBRARY_RESTYLE_JSON"):
        with open(os.environ["LIBRARY_RESTYLE_JSON"], "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=1)


if __name__ == "__main__":
    asyncio.run(main())
