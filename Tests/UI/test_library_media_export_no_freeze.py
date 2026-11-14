"""Library ▸ Media: the toolbar's "Export…" opens Export; the app keeps answering.

TASK-34000.4 (2026-10-02 Library review, finding L-01 -- a P0). Pressing
"Export…" in the Media list toolbar hung the whole app: nothing repainted and
no key was read, Ctrl+Q included.

The mechanism this test pins. ``LibraryMediaCanvas`` owns the row for
``#library-media-export``, so the press is dispatched on the CANVAS's message
pump. That handler awaited the controller, which awaited the Export open,
which awaited ``Screen.recompose()`` -- and the recompose tears down the
screen's children, the canvas among them. Textual's teardown waits for every
removed widget's pump task to finish (``Widget._message_loop_exit`` gathers
its children's tasks, with no timeout), while the canvas's pump task was the
one waiting for the teardown. Neither side can finish. The recompose also
holds ``App.batch_update`` open, so nothing paints again.

This file is the AC#4 regression test and nothing else, so it runs unmodified
against any commit: it imports no fix-side module. Red on the review's commit
(dev 2d34cbf80d) and on the wave base with the review-era inline await put
back; green with the hand-off. The guard that refuses the whole class, on the
real Library screen, is pinned in
``test_library_surface_swap_refused_on_real_screen.py``; the guard's own
predicate in ``Tests/Architecture/test_surface_swap_guard.py``.

Every wait after a press is bounded by wall clock, and a red run names the
parked pump and frees it before tear-down (``Tests/UI/pump_probe.py`` says why
``asyncio.wait_for`` cannot do that job). ``eager`` is how the real app runs:
``App.run_async`` installs ``asyncio.eager_task_factory``; ``run_test`` does
not.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.pump_probe import (
    describe_parked,
    free_parked_pumps,
    key,
    parked_pumps,
    pump_runs,
    task_factory,
    until,
)
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _two_media_items,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_INGEST_EXPORT,
)
from tldw_chatbook.Widgets.Library import LibraryExportCanvas

pytestmark = pytest.mark.bootstrap_profile

#: How long the press may take to land the Export canvas, and Escape to leave
#: it. Measured at well under a second on this harness; a deadlocked press
#: never lands at all, so this is the time a red run spends finding out.
OPEN_BUDGET_SECONDS = 10.0
#: pytest-timeout's bound on one test, tear-down included: the only bound
#: that holds whatever the event loop is doing (SIGALRM; see ``timeout`` in
#: pyproject.toml).
SCENARIO_TIMEOUT_SECONDS = 90

ROUTES = ["mouse", "keyboard"]
FACTORIES = ["default", "eager"]


def _media_host() -> LibraryHarness:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=None, media=_two_media_items())
    return LibraryHarness(app)


async def _enter_media_list(host: LibraryHarness, pilot):
    """Mount the Library and land on the Media list (Pilot waits are safe
    here: nothing has been pressed yet)."""
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    if screen.query("#library-rail-explore-all"):
        screen.query_one("#library-rail-explore-all", Button).press()
    await _wait_for_selector(screen, pilot, "#library-row-browse-media")
    screen.query_one("#library-row-browse-media", Button).press()
    await _wait_for_selector(screen, pilot, "#library-media-row-0")
    await _wait_for_selector(screen, pilot, "#library-media-export")
    return screen


def _export_canvas_is_up(screen) -> bool:
    return bool(screen.query("#library-export-canvas")) and (
        screen._library_selected_row_id == LIBRARY_ROW_INGEST_EXPORT
    )


def _media_list_is_up(screen) -> bool:
    canvases = list(screen.query("#library-media-canvas"))
    return (
        screen._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA
        and not screen.query("#library-export-canvas")
        and bool(canvases)
        and bool(canvases[0].display)
        and bool(screen.query("#library-media-row-0"))
    )


async def _assert_nothing_is_parked(host: LibraryHarness, screen, when: str) -> None:
    """The app keeps answering: the pump that reads every key (Escape, F1,
    Ctrl+P and Ctrl+Q included), the screen's, and every widget's."""
    assert await pump_runs(host), f"{when}: the app pump is parked"
    assert await pump_runs(screen), f"{when}: the Library screen's pump is parked"
    assert describe_parked(await parked_pumps(host, 0.5)) == [], (
        f"{when}: a widget's pump is parked"
    )


async def _activate_export(route: str, host: LibraryHarness, screen, pilot) -> bool:
    """Activate the toolbar's Export… the way a user does.

    ``mouse`` is a real click at the button's cell (down, up, click through
    the screen). Pilot follows a click with its idle wait, which is exactly
    what stops on a parked pump, so the click itself is bounded: on a freeze
    it is abandoned here and the assertions that follow say what broke.
    ``keyboard`` focuses the button and delivers Enter through the driver.

    Returns:
        False when the click's own idle wait ran out of budget -- the press
        is already known to be parked, so the caller need not wait again.
    """
    button = screen.query_one("#library-media-export", Button)
    assert button.display and not button.disabled, "precondition: Export… is live"
    if route == "mouse":
        try:
            await asyncio.wait_for(
                pilot.click("#library-media-export"), OPEN_BUDGET_SECONDS
            )
        except TimeoutError:
            return False
        return True
    button.focus()
    await pilot.pause()
    assert host.focused is button, "precondition: Export… holds focus"
    key(host, "enter", "\r")
    return True


@pytest.mark.timeout(SCENARIO_TIMEOUT_SECONDS)
@pytest.mark.asyncio
@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("route", ROUTES)
async def test_toolbar_export_opens_the_export_canvas_and_escape_is_processed(
    route: str, factory: str
) -> None:
    """Export… by mouse and by keyboard: the canvas mounts, Escape returns.

    With the Export open awaited on the canvas's own pump again -- the code
    the review ran against, dev 2d34cbf80d -- every parametrization fails on
    its bounded poll and names the parked pump and the await it sits at.
    """
    host = _media_host()
    with task_factory(factory):
        async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            screen = await _enter_media_list(host, pilot)
            media_type_filter = screen._media_state.type_filter
            passed = False
            try:
                settled = await _activate_export(route, host, screen, pilot)

                await until(
                    lambda: _export_canvas_is_up(screen) or host._exception is not None,
                    OPEN_BUDGET_SECONDS if settled else 0.5,
                )
                assert host._exception is None, (
                    f"{route}/{factory}: Export… raised instead of opening: "
                    f"{host._exception!r}"
                )
                assert _export_canvas_is_up(screen), (
                    f"{route}/{factory}: Export… never opened the Export canvas "
                    f"within {OPEN_BUDGET_SECONDS:.0f}s (selected row: "
                    f"{screen._library_selected_row_id!r}); parked pumps: "
                    f"{describe_parked(await parked_pumps(host, 0.5))}"
                )
                await _assert_nothing_is_parked(
                    host, screen, f"{route}/{factory} after Export…"
                )

                # AC#2: the same Media export the other entry points offer --
                # the one Export canvas, pre-scoped to the Media section and
                # to the list's own type filter, not to a hand-picked set.
                assert isinstance(
                    screen.query_one("#library-export-canvas"), LibraryExportCanvas
                )
                scope = screen._export_state.scope
                assert scope.kind == "media", scope
                assert scope.media_type == media_type_filter, scope
                assert not scope.ids, scope
                assert screen.query("#library-export-scope-line")
                assert screen._export_state.origin_row_id == LIBRARY_ROW_BROWSE_MEDIA

                # AC#4: Escape is delivered the way the driver does, and it is
                # PROCESSED -- it leaves Export for the Media list it came from.
                key(host, "escape", "\x1b")
                assert await until(
                    lambda: _media_list_is_up(screen), OPEN_BUDGET_SECONDS
                ), (
                    f"{route}/{factory}: Escape was not processed after Export… "
                    f"(selected row: {screen._library_selected_row_id!r})"
                )
                assert await pump_runs(host), (
                    f"{route}/{factory}: the app pump is parked after Escape"
                )
                passed = True
            finally:
                if not passed:
                    for line in await free_parked_pumps(host):
                        print(f"\nfreed for tear-down: {line}")
