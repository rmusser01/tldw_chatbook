"""Phase-C residency hazards: the three rulings the design record required.

``Tests/UI/test_library_phase_c_switch_residency.py`` pins the WIN (no
whole-screen recompose per rail-mode switch). This file pins the three
hazards that win creates, each named in
``Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md``
("Design record — phase C, media"):

1. **A hidden widget still receives and dispatches events** (verified Textual
   8.2.8 behaviour #3): ``Button.press()`` checks the BUTTON's own ``disabled``
   and ``display``, neither of which an ancestor's ``display = False`` touches,
   so a resident-but-unselected canvas would happily open the media viewer
   while the user is looking at Notes.
2. **``_sync_library_canvas`` off-route stops raising** (the TASK-32089
   ruling): before residency the dispatcher's ``query_one`` raised
   ``NoMatches`` and the blanket ``except`` converted it into a whole-screen
   recompose. With the canvas resident the query SUCCEEDS and the sync
   silently repaints an invisible canvas.
3. **TASK-31521 screen reuse composes with residency**: suspend/resume must
   not disturb the resident set or which member of it is showing.
"""
from __future__ import annotations

import pytest
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
from tldw_chatbook.UI.Library_Modules.canvas_sync import _sync_library_canvas


async def _settle(pilot, passes: int = 40, delay: float = 0.01) -> None:
    for _ in range(passes):
        await pilot.pause(delay)


async def _press_rail_row(screen, pilot, row_id: str) -> None:
    screen.query_one(f"#library-row-{row_id}", Button).press()
    await _settle(pilot)


async def _enter_media_then_notes(host, pilot):
    """Leave the Library on Notes with the Media canvas resident and hidden."""
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    await _settle(pilot, passes=30)
    if screen.query("#library-rail-explore-all"):
        screen.query_one("#library-rail-explore-all", Button).press()
        await _settle(pilot, passes=30)
    await _press_rail_row(screen, pilot, LIBRARY_ROW_BROWSE_MEDIA)
    await _press_rail_row(screen, pilot, LIBRARY_ROW_BROWSE_NOTES)
    return screen


@pytest.mark.asyncio
async def test_hidden_resident_media_canvas_does_not_process_row_presses() -> None:
    """A row press inside the off-route resident canvas must not navigate.

    The exact shape the design record's Textual finding #3 predicted: the row
    Button is neither disabled nor ``display: none`` itself -- only its canvas
    ancestor is -- so ``press()`` fires and, ungated, reaches
    ``handle_library_media_row`` and opens the media viewer underneath a user
    who is reading Notes.
    """
    app = _build_test_app()
    _seed_conversations(
        app, _two_conversations(), notes=None, media=_two_media_items()
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _enter_media_then_notes(host, pilot)

        media_canvas = screen.query_one("#library-media-canvas")
        assert not media_canvas.display, (
            "precondition: the media canvas must be resident and hidden"
        )
        rows = media_canvas.query(".library-media-row")
        assert rows, "precondition: the resident canvas still holds its rows"

        rows.first(Button).press()
        await _settle(pilot)

        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES
        assert screen._media_state.view == "list", (
            "an off-route row press opened the media viewer"
        )
        assert screen.query_one("#library-notes-canvas").display, (
            "the Notes canvas stopped showing after an off-route media press"
        )


@pytest.mark.asyncio
async def test_off_route_media_sync_is_refused_rather_than_repainting() -> None:
    """TASK-32089: the dispatcher refuses a canvas its route does not own.

    Mutation check for this pin: delete the guard in
    ``canvas_sync._sync_library_canvas`` and the call returns True, having
    rebuilt the hidden canvas's children.
    """
    app = _build_test_app()
    _seed_conversations(
        app, _two_conversations(), notes=None, media=_two_media_items()
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _enter_media_then_notes(host, pilot)

        media_canvas = screen.query_one("#library-media-canvas")
        before = [id(child) for child in media_canvas.children]

        assert _sync_library_canvas(screen, "media") is False
        await _settle(pilot, passes=10)

        assert [id(child) for child in media_canvas.children] == before, (
            "the off-route sync rebuilt the hidden canvas's children"
        )
        # And the refusal must NOT be the old whole-screen fallback in
        # disguise: the resident set is untouched.
        assert screen.query("#library-notes-canvas")
        assert screen.query_one("#library-notes-canvas").display


@pytest.mark.asyncio
async def test_suspend_and_resume_keep_the_resident_set_and_its_selection() -> None:
    """TASK-31521 composition: screen reuse does not disturb residency."""
    app = _build_test_app()
    _seed_conversations(
        app, _two_conversations(), notes=None, media=_two_media_items()
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _enter_media_then_notes(host, pilot)
        canvas_host = screen.query_one("#library-canvas")
        before = tuple(child.id for child in canvas_host.children)
        assert "library-media-canvas" in before
        assert "library-notes-canvas" in before

        screen.on_screen_suspend()
        await _settle(pilot, passes=5)
        assert screen._library_screen_suspended is True
        assert tuple(child.id for child in canvas_host.children) == before

        screen.on_screen_resume()
        await _settle(pilot, passes=20)

        assert screen._library_screen_suspended is False
        assert tuple(child.id for child in canvas_host.children) == before
        assert screen.query_one("#library-notes-canvas").display
        assert not screen.query_one("#library-media-canvas").display
        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES


@pytest.mark.asyncio
async def test_route_marker_class_tracks_the_selection_across_switches() -> None:
    """The marker classes are a state projection, so pin them to the state.

    Phase C replaced ~35 "is this shell mounted?" route probes with
    ``.library-media-route`` / ``.library-notes-route`` on the ONE resident
    browse shell. That trade is only safe while the marker and
    ``_library_selected_row_id`` cannot disagree, and ``apply_route`` is the
    single writer that keeps them together -- on the compose path AND on the
    resident switch path. This walks both paths (first entry into each route
    is a recompose; the switches after it are not) and checks the pair after
    every one.
    """
    app = _build_test_app()
    _seed_conversations(
        app, _two_conversations(), notes=None, media=_two_media_items()
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _settle(pilot, passes=30)
        if screen.query("#library-rail-explore-all"):
            screen.query_one("#library-rail-explore-all", Button).press()
            await _settle(pilot, passes=30)

        for row_id in (
            LIBRARY_ROW_BROWSE_MEDIA,
            LIBRARY_ROW_BROWSE_NOTES,
            LIBRARY_ROW_BROWSE_MEDIA,
            LIBRARY_ROW_BROWSE_NOTES,
        ):
            await _press_rail_row(screen, pilot, row_id)
            shell = screen.query_one("#library-browse-reader-shell")
            media_selected = row_id == LIBRARY_ROW_BROWSE_MEDIA
            assert screen._library_selected_row_id == row_id
            assert shell.has_class("library-media-route") is media_selected
            assert shell.has_class("library-notes-route") is not media_selected
            assert bool(screen.query(".library-media-route")) is media_selected
            assert bool(screen.query(".library-notes-route")) is not media_selected
