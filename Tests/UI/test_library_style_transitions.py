"""Mounted transition evidence for ADR-161 Library sizing classes."""

import pytest

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET
from Tests.UI.test_library_honesty_accessibility import _DatabaseNoteEditorApp
from Tests.UI.test_library_media_toolbar_adapt import _browse_state, _CanvasApp
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _wait_for_library_shell,
)
from tldw_chatbook.Widgets.Library.library_media_canvas import LibraryMediaCanvas
from tldw_chatbook.Widgets.Library.library_notes_canvas import LibraryNotesCanvas


class _NotesHost(_DatabaseNoteEditorApp):
    CSS_PATH = BUNDLED_STYLESHEET


class _MediaHost(_CanvasApp):
    CSS_PATH = BUNDLED_STYLESHEET


@pytest.mark.asyncio
async def test_note_header_sizes_round_trip_on_the_same_mounted_widgets():
    async with _NotesHost().run_test(size=(150, 40)) as pilot:
        canvas = pilot.app.query_one(LibraryNotesCanvas)
        heading = canvas.query_one("#library-note-heading")
        primary = canvas.query_one("#library-note-primary-actions")
        authority = canvas.query_one("#library-note-authority-git-status")
        for compact, heading_height, primary_height, primary_width in (
            (False, 3, 3, "auto"),
            (True, 1, 2, "100w"),
            (False, 3, 3, "auto"),
        ):
            canvas.apply_compact_presentation(compact)
            await pilot.pause()
            assert canvas.query_one("#library-note-heading") is heading
            assert heading.region.height == heading_height
            assert primary.region.height == primary_height
            assert str(primary.styles.width) == primary_width
            assert str(authority.styles.width) == ("18" if compact else "auto")


@pytest.mark.asyncio
async def test_media_row_height_round_trips_without_remounting():
    async with _MediaHost(_browse_state()).run_test(size=(100, 34)) as pilot:
        canvas = pilot.app.query_one(LibraryMediaCanvas)
        row = canvas.query_one(".library-media-row")
        for compact, height in ((False, 2), (True, 1), (False, 2)):
            canvas.apply_compact_presentation(compact)
            await pilot.pause()
            assert canvas.query_one(".library-media-row") is row
            assert row.region.height == height


@pytest.mark.asyncio
async def test_ordinary_emergency_canvas_releases_previous_fractional_width():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(140, 32)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        canvas = screen.query_one("#library-canvas")
        for width, expected in ((140, "13fr"), (63, "1fr"), (140, "13fr")):
            await pilot.resize_terminal(width, 32)
            await pilot.pause()
            assert screen.query_one("#library-canvas") is canvas
            assert str(canvas.styles.width) == expected


@pytest.mark.asyncio
async def test_navigation_handle_replaces_the_base_button_width():
    from Tests.UI.consolidated_css import ConsolidatedCSSApp
    from tldw_chatbook.Widgets.Library.library_rail import LibraryNavigationRailHandle

    class HandleHost(ConsolidatedCSSApp):
        CSS_PATH = BUNDLED_STYLESHEET

        def compose(self):
            yield LibraryNavigationRailHandle(id="handle")

    async with HandleHost().run_test(size=(30, 12)) as pilot:
        handle = pilot.app.query_one("#handle")
        button = handle.query_one("#library-rail-open")
        assert handle.region.width == 3
        assert button.region.width == 1
        assert str(button.styles.width) == "1"
