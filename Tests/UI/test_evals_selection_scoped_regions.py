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
