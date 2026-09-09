"""Phase-C acceptance pin: a Library rail-mode switch must stop rebuilding the screen.

**This test is RED BY CONSTRUCTION until phase C lands.** It is the failing
acceptance test called for by Task 1 of
``Docs/superpowers/plans/2026-09-08-library-phase-c-media-graduation.md``, and
the thresholds below are the resident-canvas targets recorded in the design
record appended to
``Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md``
("Design record — phase C, media: the resident-canvas mechanism").

## What it replaces

Recipe §25 (``backlog/docs/library-decomposition-recipe.md``) handed phase C a
probe band as its acceptance evidence: settle 243–494 ms, max gap 37–179 ms,
mounts 177/89/85/114/38/175/114/175 per rail click, and a ``recompose`` column
reading **0**. Task 1's instrumentation
(``Helper_Scripts/library_switch_teardown_probe.py``) established two things
about that band which this test encodes instead:

1. That ``recompose`` 0 is an instrument artifact. The click probe counts
   ``BaseAppScreen.refresh(recompose=True)``; a rail-mode switch instead
   **awaits ``Widget.recompose()`` directly** from
   ``_select_library_rail_row_after_source_admission``, which that column
   cannot see. Every media/notes rail switch measured runs exactly ONE
   whole-screen recompose today.
2. Widget residency is impossible while that call exists --
   ``Widget.recompose()`` removes every child of the screen, so no canvas can
   survive a switch no matter how it is toggled. Measured directly: a resident
   canvas mounted beside the active one is gone after one
   ``screen.recompose()``.

So the pin is structural, not a wall-clock band: **no whole-screen recompose on
a rail-mode switch, and a mount/unmount count in the tens rather than the
hundreds.** Both quantities are load-independent -- recipe §9's own rule is to
read those columns as the verdict and the wall-clock columns as context -- so
this test is deterministic on a loaded machine, which a settle-time pin is not.

## Where the thresholds come from

Measured on this branch's parent commit, from a scratch worktree, media
switch-in, by region (probe run 2026-09-08):

    177 mounts = 87 canvas (media) + 52 rail + 19 nav bar + 6 footer
               +  6 screen chrome + 5 reader shell + 2 media viewer
    162 unmounts, same shape

A resident design keeps the rail, nav bar, footer, chrome and both canvases
mounted, so the only legitimate per-switch mounts are the route's STRUCTURAL
delta -- the Notes source strip (``#library-notes-source-strip``, 5 widgets)
appearing or disappearing, and the media viewer (2). ``_MAX_SWITCH_MOUNTS``
below is 25: more than three times that seven-widget structural delta, and
still a 7x reduction on today's 177. It is deliberately generous, because the
pin that carries the design is ``_MAX_WHOLE_SCREEN_RECOMPOSES``.

**That ceiling was derived from a floor measured WITHOUT the reader-shell id
split resolved** (the spec's Task 2 step 1). If the implementation finds 25
tight, re-derive it from a fresh measurement and say so in the same commit --
do not quietly raise it.

## Every count here is an upper bound, so the liveness assertions are load-bearing

``_assert_switch_did_something`` is not decoration. Review of the first version
of this file patched the switch handler to an async no-op and every ceiling
passed: a dead UI scores 0 recomposes and 0 mounts. The destination-canvas and
selection-moved checks are what make a green result mean "the switch got
cheaper" rather than "the switch stopped happening".
"""
from __future__ import annotations

import collections

import pytest
from textual.app import App
from textual.widget import Widget
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _two_media_items,
    _wait_for_library_shell,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_BROWSE_NOTES,
)

#: A rail-mode switch may not rebuild the whole screen. This is the pin the
#: design record turns on; see the module docstring.
_MAX_WHOLE_SCREEN_RECOMPOSES = 0
#: Per-switch widget mounts/unmounts. Today: 114-179 mounts, 121-176 unmounts.
_MAX_SWITCH_MOUNTS = 25
_MAX_SWITCH_UNMOUNTS = 25


class _SwitchCounters:
    """Per-click mount / unmount / whole-screen-recompose tallies."""

    def __init__(self) -> None:
        self.armed = False
        self.mounts: collections.Counter = collections.Counter()
        self.unmounts: collections.Counter = collections.Counter()
        self.recomposed: list[str] = []

    def reset(self) -> None:
        self.mounts = collections.Counter()
        self.unmounts = collections.Counter()
        self.recomposed = []


def _install_counters(monkeypatch: pytest.MonkeyPatch) -> _SwitchCounters:
    """Instrument mounts, unmounts and whole-screen recomposes.

    ``App._unregister`` is deliberately NOT the unmount hook: Textual 8.2.8
    prunes through ``Widget._message_loop_exit`` (which is what discards the
    widget from ``App._registry``), so a counter on ``_unregister`` reports
    zero removals forever -- the blind spot behind the click probe's
    "N mounts / 0 removes" line.
    """
    counters = _SwitchCounters()

    original_register = App._register

    def register(self, parent, *widgets, **kwargs):
        result = original_register(self, parent, *widgets, **kwargs)
        if counters.armed:
            for widget in widgets:
                counters.mounts[type(widget).__name__] += 1
        return result

    monkeypatch.setattr(App, "_register", register)

    original_exit = Widget._message_loop_exit

    async def message_loop_exit(self, *args, **kwargs):
        if counters.armed:
            counters.unmounts[type(self).__name__] += 1
        return await original_exit(self, *args, **kwargs)

    monkeypatch.setattr(Widget, "_message_loop_exit", message_loop_exit)

    original_recompose = Widget.recompose

    async def recompose(self, *args, **kwargs):
        if counters.armed:
            counters.recomposed.append(type(self).__name__)
        return await original_recompose(self, *args, **kwargs)

    monkeypatch.setattr(Widget, "recompose", recompose)
    return counters


async def _settle(pilot, passes: int = 60, delay: float = 0.01) -> None:
    for _ in range(passes):
        await pilot.pause(delay)


async def _switch(
    screen, pilot, counters: _SwitchCounters, row_id: str, canvas_selector: str
) -> dict:
    """Click one rail row with the counters armed; return its tallies.

    Also records the two LIVENESS facts the ceilings cannot express: whether
    the destination canvas is on screen afterwards, and whether the screen's
    selection actually moved. See ``_assert_switch_did_something`` for why.
    """
    counters.reset()
    counters.armed = True
    screen.query_one(f"#library-row-{row_id}", Button).press()
    await _settle(pilot)
    counters.armed = False
    canvases = screen.query(canvas_selector)
    canvas = canvases.first() if canvases else None
    return {
        "mounts": sum(counters.mounts.values()),
        "unmounts": sum(counters.unmounts.values()),
        "screen_recomposes": [
            name for name in counters.recomposed if name == "LibraryScreen"
        ],
        "canvas_selector": canvas_selector,
        "canvas_present": canvas is not None,
        "canvas_displayed": bool(canvas is not None and canvas.display),
        "expected_row_id": row_id,
        "selected_row_id": screen._library_selected_row_id,
    }


# The switch actually has to switch. Without these, every assertion in this
# file is an UPPER BOUND, and a no-op satisfies all of them: review of this
# test patched ``_select_library_rail_row_after_source_admission`` to an async
# no-op -- the single most likely phase-C regression shape, a ``replaced``
# flag reported True without mounting the destination
# (``library_screen.py`` around the ``if not replaced:`` arm) -- and the pin
# ran GREEN while the notes canvas never appeared and the selection never
# moved. 0 recomposes and 0 mounts is exactly what a dead UI scores.
#
# These also close a quieter hole: a load-starved run whose settle window
# expires before the switch does any work would otherwise read zero and pass.
def _assert_switch_did_something(label: str, tallies: dict, failures: list) -> None:
    """Append a failure unless the switch reached its destination."""
    if not tallies["canvas_present"]:
        failures.append(
            f"{label}: destination canvas {tallies['canvas_selector']} is not "
            "mounted -- the switch did not reach its destination"
        )
    elif not tallies["canvas_displayed"]:
        failures.append(
            f"{label}: destination canvas {tallies['canvas_selector']} is "
            "mounted but not displayed -- residency must show the destination, "
            "not merely keep it resident"
        )
    if tallies["selected_row_id"] != tallies["expected_row_id"]:
        failures.append(
            f"{label}: screen selection is "
            f"{tallies['selected_row_id']!r}, expected "
            f"{tallies['expected_row_id']!r} -- the rail selection did not move"
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Phase C not implemented: the rail switch still awaits "
        "Widget.recompose(). strict=True so the red->green flip is caught "
        "automatically when Task 2 lands -- an XPASS fails the suite and is "
        "the signal to delete this marker "
        "(Docs/superpowers/plans/2026-09-08-library-phase-c-media-graduation.md)."
    ),
)
@pytest.mark.asyncio
async def test_library_rail_mode_switch_does_not_rebuild_the_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A media<->notes rail switch keeps the shell and canvases resident.

    RED until phase C lands -- see the module docstring for the measured
    numbers this replaces and why the pins are structural rather than timed.
    Run with ``--runxfail`` to see the real failure text.
    """
    app = _build_test_app()
    _seed_conversations(
        app, _two_conversations(), notes=None, media=_two_media_items()
    )
    host = LibraryHarness(app)
    counters = _install_counters(monkeypatch)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _settle(pilot, passes=30)
        if screen.query("#library-rail-explore-all"):
            screen.query_one("#library-rail-explore-all", Button).press()
            await _settle(pilot, passes=30)

        # Warm-up entry into media; its COUNTS are not asserted (a first visit
        # legitimately mounts the media route for the first time), but its
        # liveness is -- if this never arrives, the two measured switches are
        # not starting where they claim to.
        warm_up = await _switch(
            screen, pilot, counters, LIBRARY_ROW_BROWSE_MEDIA, "#library-media-canvas"
        )

        measured = {
            "notes (switch)": await _switch(
                screen,
                pilot,
                counters,
                LIBRARY_ROW_BROWSE_NOTES,
                "#library-notes-canvas",
            ),
            "media (switch-back)": await _switch(
                screen,
                pilot,
                counters,
                LIBRARY_ROW_BROWSE_MEDIA,
                "#library-media-canvas",
            ),
        }

    failures = []
    _assert_switch_did_something("media (warm-up)", warm_up, failures)
    for label, tallies in measured.items():
        _assert_switch_did_something(label, tallies, failures)
        if len(tallies["screen_recomposes"]) > _MAX_WHOLE_SCREEN_RECOMPOSES:
            failures.append(
                f"{label}: {len(tallies['screen_recomposes'])} whole-screen "
                f"LibraryScreen.recompose() call(s), expected "
                f"<= {_MAX_WHOLE_SCREEN_RECOMPOSES}"
            )
        if tallies["mounts"] > _MAX_SWITCH_MOUNTS:
            failures.append(
                f"{label}: {tallies['mounts']} widget mounts, expected "
                f"<= {_MAX_SWITCH_MOUNTS}"
            )
        if tallies["unmounts"] > _MAX_SWITCH_UNMOUNTS:
            failures.append(
                f"{label}: {tallies['unmounts']} widget unmounts, expected "
                f"<= {_MAX_SWITCH_UNMOUNTS}"
            )
    assert not failures, (
        "Library rail-mode switch still rebuilds the screen:\n  "
        + "\n  ".join(failures)
        + "\n\nMeasured: "
        + repr(measured)
    )
