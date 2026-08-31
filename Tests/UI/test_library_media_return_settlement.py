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
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_media_reader_shell import (
    LibraryMediaReaderShell,
    MediaShellResized,
)


async def _open_compact_media(host, pilot) -> LibraryScreen:
    """Enter Media after selecting the compact presentation contract."""
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    screen._library_notes_compact = True
    screen.query_one("#library-row-browse-media").press()
    await _wait_for_selector(screen, pilot, "#library-media-reader-shell")
    return screen


@pytest.mark.asyncio
async def test_first_frame_media_stage_projects_adaptive_compact_without_legacy_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    screen = LibraryScreen(app)
    monkeypatch.setattr(
        screen,
        "_reconcile_library_media_stage_presentation",
        lambda: False,
    )
    monkeypatch.setattr(screen, "_apply_library_notes_stage_visibility", lambda: None)
    host = LibraryProductionCSSHarness(app, screen=screen)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open_compact_media(host, pilot)
        stage = screen.query_one("#library-shell-grid", Horizontal)

        assert stage.has_class("library-adaptive-compact")
        assert not stage.has_class("library-notes-compact")


@pytest.mark.asyncio
async def test_same_size_recompose_projects_media_stage_without_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    screen = LibraryScreen(app)
    monkeypatch.setattr(
        screen,
        "_reconcile_library_media_stage_presentation",
        lambda: False,
    )
    monkeypatch.setattr(screen, "_apply_library_notes_stage_visibility", lambda: None)
    host = LibraryProductionCSSHarness(app, screen=screen)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open_compact_media(host, pilot)
        previous_stage = screen.query_one("#library-shell-grid", Horizontal)

        await screen.recompose()
        replacement_stage = screen.query_one("#library-shell-grid", Horizontal)

        assert replacement_stage is not previous_stage
        assert replacement_stage.has_class("library-adaptive-compact")
        assert not replacement_stage.has_class("library-notes-compact")


@pytest.mark.asyncio
async def test_media_shell_lifecycle_reconciles_current_stage_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open_compact_media(host, pilot)
        stage = screen.query_one("#library-shell-grid", Horizontal)
        shell = screen.query_one("#library-media-reader-shell", LibraryMediaReaderShell)
        stage.set_class(True, "library-notes-compact")
        stage.set_class(False, "library-adaptive-compact")

        layout_refreshes: list[bool] = []
        refresh = screen.refresh

        def count_layout_refresh(*args, **kwargs):
            layout_refreshes.append(bool(kwargs.get("layout", False)))
            return refresh(*args, **kwargs)

        monkeypatch.setattr(screen, "refresh", count_layout_refresh)

        assert shell.post_message(MediaShellResized())
        await pilot.pause()

        assert screen.query_one("#library-shell-grid", Horizontal) is stage
        assert stage.has_class("library-adaptive-compact")
        assert not stage.has_class("library-notes-compact")
        assert layout_refreshes == [True]

        assert shell.post_message(MediaShellResized())
        await pilot.pause()

        assert layout_refreshes == [True]
