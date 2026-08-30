"""Media adaptive-stage presentation regressions for return settlement."""

from __future__ import annotations

import pytest
from textual.containers import Horizontal

from Tests.UI.test_library_media_side_by_side import _build_media_test_app
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _two_media_items,
    _wait_for_library_shell,
    _wait_for_selector,
)


async def _open_compact_media(host, pilot):
    """Enter Media after selecting the compact presentation contract."""
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    screen._library_notes_compact = True
    screen.query_one("#library-row-browse-media").press()
    await _wait_for_selector(screen, pilot, "#library-media-reader-shell")
    return screen


@pytest.mark.asyncio
async def test_first_frame_media_stage_projects_adaptive_compact_without_legacy_class():
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open_compact_media(host, pilot)
        stage = screen.query_one("#library-shell-grid", Horizontal)

        assert stage.has_class("library-adaptive-compact")
        assert not stage.has_class("library-notes-compact")


@pytest.mark.asyncio
async def test_same_size_recompose_replaces_wrong_media_stage_presentation():
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open_compact_media(host, pilot)
        previous_stage = screen.query_one("#library-shell-grid", Horizontal)
        previous_stage.set_class(True, "library-notes-compact")
        previous_stage.set_class(False, "library-adaptive-compact")

        await screen.recompose()
        replacement_stage = screen.query_one("#library-shell-grid", Horizontal)

        assert replacement_stage is not previous_stage
        assert replacement_stage.has_class("library-adaptive-compact")
        assert not replacement_stage.has_class("library-notes-compact")
