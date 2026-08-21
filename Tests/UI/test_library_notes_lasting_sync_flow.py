"""Inert retained-shell integration for lasting Notes sync."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_notes_canvas import LibraryNotesCanvas
from tldw_chatbook.Widgets import Library as library_widgets
from tldw_chatbook.app import TldwCli


def test_production_screen_keeps_lasting_route_explicitly_inert() -> None:
    source = Path("tldw_chatbook/UI/Screens/library_screen.py").read_text(
        encoding="utf-8"
    )

    assert "lasting_available=False" in source
    assert "LibraryNotesAddFromFilesCanvas" in source
    assert "LibraryNotesSyncRootsCanvas" in source
    assert "LibraryNotesSyncController(" in source
    assert "self._library_notes_sync_controller.choose_relationship" in source


def test_production_notes_canvas_does_not_replace_legacy_entry_points_before_cutover() -> (
    None
):
    source = Path("tldw_chatbook/Widgets/Library/library_notes_canvas.py").read_text(
        encoding="utf-8"
    )

    assert '("Sync", "library-notes-sync-open")' in source
    assert "library-notes-import" in source
    assert "lasting_sync_snapshot" in source


def test_relationship_chooser_reuses_the_one_existing_import_controller() -> None:
    source = Path("tldw_chatbook/UI/Screens/library_screen.py").read_text(
        encoding="utf-8"
    )

    assert source.count("self._library_note_import_controller.begin_selection()") == 1
    assert "choose_relationship(" in source


def test_library_widget_package_preserves_existing_exports_and_adds_sync_canvases() -> (
    None
):
    expected = {
        "LibraryNotesCanvas",
        "LibraryRail",
        "LibraryNotesAddFromFilesCanvas",
        "LibraryNotesSyncRootsCanvas",
    }

    assert expected <= set(library_widgets.__all__)
    assert all(hasattr(library_widgets, name) for name in expected)


@pytest.mark.asyncio
async def test_mounted_production_projection_is_inert_and_names_nearest_valid_action() -> (
    None
):
    screen = LibraryScreen(SimpleNamespace(app_config={}))

    class _Host(App[None]):
        CSS_PATH = TldwCli.CSS_PATH

        def compose(self) -> ComposeResult:
            yield LibraryNotesCanvas(
                mode="lasting_add",
                lasting_sync_snapshot=screen._library_notes_lasting_sync_snapshot,
                compact=True,
            )

    app = _Host()
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        assert screen._library_notes_lasting_sync_snapshot.lasting_available is False
        keep = app.query_one("#notes-add-keep-synced", Button)
        assert keep.disabled is False
        assert "Unavailable" in app.export_screenshot(simplify=True)
        import_once = app.query_one("#notes-add-import-once", Button)
        assert import_once.label.plain == "Import once"
        assert import_once in app.screen._compositor.visible_widgets
        assert app.focused is import_once
        back = app.query_one("#notes-sync-back", Button)
        assert back in app.screen._compositor.visible_widgets
