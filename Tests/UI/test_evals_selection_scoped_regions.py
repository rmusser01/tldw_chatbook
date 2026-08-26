"""Evals rail selection swaps regions, not the whole screen (task-15475).

The input-latency audit measured a rail click recomposing 150-300 widgets:
``EvalsScreen.select()`` answered every selection with
``self.refresh(recompose=True)``, which tears the whole ``BaseAppScreen``
down -- nav bar, footer, header row, mode strip, the ``LabWorkbench`` and
all three of its regions -- to repaint the detail pane and inspector.

A rail click cannot change the rail's row SET (the rail is the thing that
posted it), only which row is marked active, so the hot path leaves the rail
alone and rebuilds two regions. Mutation callers (a save, a finished run, a
delete) DO change the rows, so ``select()`` keeps rebuilding the rail by
default -- the last test pins that safe default, because a dedupe that got
it backwards would leave a stale rail after every save.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input

from tldw_chatbook.UI.Evals.library_rail import LibraryRail
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar
from .test_evals_screen import EvalsHarness as EvalsHarness  # noqa: F401
from .test_evals_screen import evals_app as evals_app  # noqa: F401 -- fixture re-export
from .test_evals_screen import evals_db as evals_db  # noqa: F401 -- fixture re-export
from .test_evals_screen import (  # noqa: F401 -- fixture re-export
    seeded_bench as seeded_bench,
)

pytestmark = pytest.mark.asyncio

#: Chrome a selection change has no business rebuilding.
_STABLE_CHROME = (
    "#lab-header-row",
    "#lab-mode-strip",
    "#lab-workbench",
    "#lab-rail",
    "#lab-body",
    "#lab-inspector",
    "#screen-footer-status",
)


def _identities(screen, selectors) -> dict[str, int]:
    return {selector: id(screen.query_one(selector)) for selector in selectors}


async def test_rail_click_leaves_the_chrome_and_the_rail_widget_in_place(
    evals_app, seeded_bench
):
    """AC#1: the rail click path rebuilds the detail/inspector regions only."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen = evals_app.screen
        before = _identities(screen, _STABLE_CHROME)
        nav_before = id(screen.query_one(MainNavigationBar))
        rail_before = id(screen.query_one(LibraryRail))

        await pilot.click("#evals-rail-row-benches-0")
        await pilot.pause()
        await pilot.pause()

        assert _identities(screen, _STABLE_CHROME) == before, (
            "A rail selection rebuilt screen chrome that does not read the "
            "selection."
        )
        assert id(screen.query_one(MainNavigationBar)) == nav_before
        assert id(screen.query_one(LibraryRail)) == rail_before, (
            "A rail click cannot change the rail's rows -- only which one is "
            "active -- so the rail must not be rebuilt."
        )
        # ...and the panes it DOES own actually repainted.
        assert screen.query_one("#evals-bench-name", Input).value == "loaded-nouns v1"


async def test_rail_click_marks_the_selected_row_active_in_place(
    evals_app, seeded_bench
):
    """The active marker is what the rail rebuild used to provide; patching
    it in place has to give the same result."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen = evals_app.screen
        row = screen.query_one("#evals-rail-row-benches-0", Button)
        assert not row.has_class("is-active")

        await pilot.click("#evals-rail-row-benches-0")
        await pilot.pause()

        assert screen.query_one("#evals-rail-row-benches-0", Button).has_class(
            "is-active"
        )
        assert screen.query_one(LibraryRail).selection.kind == "bench"


async def test_rail_click_keeps_focus_on_the_clicked_row(evals_app, seeded_bench):
    """The screen recompose destroyed the button under the user's cursor and
    dropped focus with it; the row survives a scoped swap, so focus stays."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#evals-rail-row-benches-0")
        await pilot.pause()
        await pilot.pause()

        assert (
            getattr(evals_app.focused, "id", None) == "evals-rail-row-benches-0"
        ), "Focus escaped the rail row the user just clicked."


async def test_region_swaps_never_destroy_the_frames_collapse_headers(
    evals_app, seeded_bench
):
    """The frame composes each region's collapse header as its FIRST child.

    `#lab-rail` and `#lab-inspector` are not empty containers the mode fills:
    `LabWorkbench.compose` puts a title + collapse button in each, because
    collapse is frame-owned (which is exactly why `LabScreen._populate_regions`
    APPENDS mode content with `mount_all`). A blanket `remove_children()` in
    the selection swap destroyed both headers on the first rail click --
    permanently, since no screen recompose remains to rebuild them and the
    collapse buttons have no keyboard binding.

    Driven through EvalsScreen, not a probe LabScreen: the defect only exists
    on the path a real selection takes.
    """
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen = evals_app.screen

        def _header_state() -> dict[str, str]:
            state: dict[str, str] = {}
            for button_id, title in (
                ("lab-rail-collapse", "Catalog"),
                ("lab-inspector-collapse", "Inspector"),
            ):
                button = screen.query_one(f"#{button_id}", Button)
                header = button.parent
                titles = [
                    str(child.renderable)
                    for child in header.children
                    if child is not button
                ]
                state[button_id] = f"{titles}"
                assert title in state[button_id], f"{title} title row is gone"
            return state

        before = _header_state()

        # Several rail clicks (rail_dirty=False path)...
        for _ in range(3):
            await pilot.click("#evals-rail-row-benches-0")
            await pilot.pause()
            await pilot.pause()
            assert _header_state() == before

        # ...and a rail-rebuilding swap (rail_dirty defaults True).
        screen.select(kind="none")
        await pilot.pause()
        await pilot.pause()
        assert _header_state() == before, (
            "a rail-rebuilding swap destroyed the frame's collapse chrome"
        )
        # Exactly one of each -- not duplicated by the remount either.
        assert len(screen.query("#lab-rail-collapse")) == 1
        assert len(screen.query("#lab-inspector-collapse")) == 1
        assert len(screen.query(".console-rail-header")) == 2


async def test_rail_rebuild_returns_focus_to_the_row_the_user_was_on(
    evals_app, seeded_bench
):
    """Focus must not escape into a rail section TOGGLE across a rebuild.

    Textual's `_reset_focus` moves focus to a neighbour when the focused
    widget is removed; on a rail rebuild that neighbour is the section toggle,
    one Space away from collapsing the section the user is working in. Rail
    row ids are stable across the rebuild, so the swap restores the identity.
    """
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen = evals_app.screen
        row = screen.query_one("#evals-rail-row-benches-0", Button)
        row.focus()
        await pilot.pause()
        assert getattr(evals_app.focused, "id", None) == "evals-rail-row-benches-0"

        # A mutation-shaped selection: rebuilds the rail under the focus.
        screen.select(kind="bench", id=seeded_bench)
        for _ in range(6):
            await pilot.pause()

        assert getattr(evals_app.focused, "id", None) == "evals-rail-row-benches-0", (
            "focus escaped the rail row across a rail-rebuilding swap"
        )


async def test_a_burst_of_selections_leaves_every_region_populated(
    evals_app, seeded_bench
):
    """Superseding must never strand a half-torn-down region.

    An exclusive worker group cancels the in-flight swap, and the cancellation
    can land INSIDE `remove_children` -- leaving a region emptied and never
    refilled if the superseding swap does not rebuild that one (a rail-click
    swap does not rebuild the rail). Superseded swaps now lose a revision
    check and return before touching a widget instead.
    """
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen = evals_app.screen

        # Interleave rail-rebuilding and rail-preserving selections back to
        # back, with no pause between them, so each supersedes the last.
        screen.select(kind="bench", id=seeded_bench)
        screen.select(kind="none", rail_dirty=False)
        screen.select(kind="bench", id=seeded_bench)
        screen.select(kind="none", rail_dirty=False)
        for _ in range(10):
            await pilot.pause()

        assert screen.query("#evals-library-pane"), "the rail was left empty"
        assert screen.query("#evals-detail-pane"), "the detail pane was left empty"
        assert screen.query("#evals-inspector-pane"), "the inspector was left empty"
        assert len(screen.query("#evals-library-pane")) == 1, "the rail was duplicated"
        assert screen.query("#lab-rail-collapse")
        assert screen.query("#lab-inspector-collapse")
        # The LAST selection is what is on screen.
        assert screen._selection.kind == "none"


async def test_rail_initiated_creation_still_refreshes_the_rail(
    evals_app, seeded_bench
):
    """The trap the opt-out has to avoid: the rail posts a selection change
    for its OWN mutations too ("+ New bench", dataset/probe import), and
    those DO change its rows. Treating "came from the rail" as "rows
    unchanged" left the newly created bench missing from the rail -- caught
    by the continuation e2e, pinned here at the seam."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen = evals_app.screen
        assert not screen.query("#evals-rail-row-benches-1")

        await pilot.click("#evals-rail-new-bench")
        await pilot.pause()
        await pilot.pause()

        assert screen._selection.kind == "bench"
        assert screen.query("#evals-rail-row-benches-1"), (
            "The rail must show the bench the user just created from it."
        )


async def test_select_rebuilds_the_rail_by_default(evals_app, seeded_bench):
    """A mutation caller (save, finished run, delete) changes the rail's rows,
    so the DEFAULT must still rebuild it -- only the rail-click path opts out."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen = evals_app.screen
        rail_before = id(screen.query_one(LibraryRail))

        screen.select(kind="bench", id=seeded_bench)
        await pilot.pause()
        await pilot.pause()

        assert id(screen.query_one(LibraryRail)) != rail_before, (
            "select() without an explicit opt-out must refresh the rail: its "
            "rows may have changed under it."
        )
        assert screen.query_one("#evals-rail-row-benches-0", Button).has_class(
            "is-active"
        )
