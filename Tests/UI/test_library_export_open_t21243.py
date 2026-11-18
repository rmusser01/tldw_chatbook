"""TASK-21243: Export initiated from a Notes canvas is canvas-scoped.

The section "Export…" press on the Notes (or Media) canvas leaves the
adaptive reader shell for an ORDINARY route. Until this task that crossing
always took ``_apply_library_open_item_surface``'s structural arm -- a
whole-screen ``Screen.recompose()`` rebuilding the largest screen in the
app for a routine per-click action. The crossing now swaps the shell grid's
one adaptive child for the ordinary trio (built by the same constructor
compose uses) and projects the export canvas through the strict
canvas-child replacement.

Also pins the export-canvas internal actions (the quality strip's
open/close cycle and the destination refusal -- the two shapes among the
five converted sites, one with a focus follow-up and one without) that
used to pay the same whole-screen recompose and now sync the canvas in
place. Those conversions are what moved the whole-screen recompose census
from 67 (over the 63 pin) back under it.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app as _build_tldw_test_app
from Tests.UI.test_library_per_click_recompose_t21116 import (
    _screen_recompose_spy,
)
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _two_media_items,
    _two_notes,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_INGEST_EXPORT
from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.Widgets.Library import LibraryExportCanvas


def _build_test_app():
    """Build the legacy/full Library surface these export tests exercise."""
    app = _build_tldw_test_app()
    app.library_new_profile_admission = False
    return app


async def _goto_notes_list(screen, pilot) -> None:
    """Navigate the live Library to the database Notes list route."""
    if screen.query("#library-rail-explore-all"):
        screen.query_one("#library-rail-explore-all", Button).press()
    await _wait_for_selector(screen, pilot, "#library-row-browse-notes")
    screen.query_one("#library-row-browse-notes", Button).press()
    await _wait_for_selector(screen, pilot, "#library-notes-canvas")


def _structural_recompose_spy() -> tuple[list[BaseAppScreen], object]:
    """Spy on ``Screen.recompose`` (the structural whole-screen arm)."""
    calls: list[BaseAppScreen] = []
    original = BaseAppScreen.recompose

    async def spy(self, *args, **kwargs):
        calls.append(self)
        return await original(self, *args, **kwargs)

    return calls, spy


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_export_open_from_notes_is_canvas_scoped() -> None:
    """The Notes canvas's "Export…" press never rebuilds the whole screen.

    Pinned: zero ``refresh(recompose=True)`` calls AND zero structural
    ``Screen.recompose()`` calls for the press, while the export canvas
    lands pre-scoped to Notes with the origin row recorded for Escape.
    """
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=_two_notes(),
        media=_two_media_items(),
    )
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _goto_notes_list(screen, pilot)

        refresh_calls, refresh_spy = _screen_recompose_spy()
        recompose_calls, recompose_spy = _structural_recompose_spy()
        with (
            patch.object(BaseAppScreen, "refresh", refresh_spy),
            patch.object(BaseAppScreen, "recompose", recompose_spy),
        ):
            screen.query_one("#library-notes-export", Button).press()
            await _wait_for_selector(screen, pilot, "#library-export-canvas")
            for _ in range(10):
                await pilot.pause(0.05)

        assert refresh_calls == [], (
            "the retired whole-screen refresh(recompose=True) seam fired"
        )
        assert recompose_calls == [], (
            "the structural whole-screen Screen.recompose() arm fired for a "
            "routine section Export press leaving the Notes canvas"
        )
        # Export behaviour and resulting UI state are unchanged.
        screen.query_one("#library-export-canvas", LibraryExportCanvas)
        screen.query_one("#library-rail")  # raises NoMatches if the rail is gone
        assert screen._library_selected_row_id == LIBRARY_ROW_INGEST_EXPORT
        assert screen._export_state.scope.kind == "notes"
        assert screen._export_state.origin_row_id, "the Escape origin was lost"
        # The ordinary shell landed with its route-content host, so the next
        # strict canvas-child replacement has its projection target.
        screen.query_one("#library-canvas-route-content")


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_export_canvas_actions_sync_in_place() -> None:
    """The export canvas's own presses repaint the canvas, not the screen.

    Opened from the Media canvas so the media-quality control (the focus
    follow-up shape) is composed: a notes-scoped form hides the media
    fields by design.
    """
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=_two_notes(),
        media=_two_media_items(),
    )
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        if screen.query("#library-rail-explore-all"):
            screen.query_one("#library-rail-explore-all", Button).press()
        await _wait_for_selector(screen, pilot, "#library-row-browse-media")
        screen.query_one("#library-row-browse-media", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-canvas")
        screen.query_one("#library-media-export", Button).press()
        await _wait_for_selector(screen, pilot, "#library-export-canvas")
        await _wait_for_selector(screen, pilot, "#library-export-quality")

        refresh_calls, refresh_spy = _screen_recompose_spy()
        recompose_calls, recompose_spy = _structural_recompose_spy()
        with (
            patch.object(BaseAppScreen, "refresh", refresh_spy),
            patch.object(BaseAppScreen, "recompose", recompose_spy),
        ):
            screen._export_controller._refuse_library_export_destination(
                "the disk said no"
            )
            screen.query_one("#library-export-quality", Button).press()
            for _ in range(15):
                await pilot.pause(0.05)

        assert refresh_calls == []
        assert recompose_calls == []
        # The refusal landed on the canvas's destination row and the
        # quality strip actually opened (canvas rebuilt with it).
        destination_line = screen.query_one("#library-export-destination-line")
        assert "the disk said no" in str(
            getattr(destination_line, "renderable", "")
        )
        assert screen.query(".library-export-quality-choice"), (
            "the quality chooser strip did not open"
        )
