"""Inert retained-shell integration for lasting Notes sync."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from tldw_chatbook.Notes.notes_sync_runtime import (
    NotesSyncControlResult,
    NotesSyncRootRuntimeSnapshot,
    NotesSyncRuntimeSnapshot,
)
from tldw_chatbook.UI.Library_Modules.library_notes_sync_controller import (
    LibraryNotesSyncController,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_notes_canvas import LibraryNotesCanvas
from tldw_chatbook.Widgets.Library.library_notes_sync_roots_canvas import (
    LibraryNotesSyncRootsCanvas,
)
from tldw_chatbook.Widgets import Library as library_widgets
from tldw_chatbook.app import TldwCli


def test_production_screen_derives_lasting_route_from_the_app_runtime() -> None:
    source = Path("tldw_chatbook/UI/Screens/library_screen.py").read_text(
        encoding="utf-8"
    )

    controller_source = Path(
        "tldw_chatbook/UI/Library_Modules/library_notes_sync_controller.py"
    ).read_text(encoding="utf-8")
    # TASK-21112: availability still derives from the runtime snapshot;
    # 'not_configured' (boot-deferred) also offers first-time setup.
    assert (
        "runtime.snapshot().status in _SETUP_READY_STATUSES" in controller_source
    )
    assert "lasting_available=" not in source
    assert "LibraryNotesAddFromFilesCanvas" in source
    assert "LibraryNotesSyncRootsCanvas" in source
    assert "LibraryNotesSyncController(" in source
    assert "self._library_notes_sync_controller.choose_relationship" in source


def test_production_notes_canvas_replaces_legacy_entry_points_at_cutover() -> None:
    source = Path("tldw_chatbook/Widgets/Library/library_notes_canvas.py").read_text(
        encoding="utf-8"
    )

    assert '("Add from files…", "library-notes-add-from-files")' in source
    assert "library-notes-manage-sync-folders" in source
    assert '"library-notes-sync-open"' not in source
    assert "lasting_sync_snapshot" in source


def test_relationship_chooser_reuses_the_one_existing_import_controller() -> None:
    screen_source = Path("tldw_chatbook/UI/Screens/library_screen.py").read_text(
        encoding="utf-8"
    )
    controller_source = Path(
        "tldw_chatbook/UI/Library_Modules/library_notes_sync_controller.py"
    ).read_text(encoding="utf-8")

    assert controller_source.count("self._import_controller.begin_selection()") == 1
    assert "choose_relationship(" in screen_source
    assert "_begin_library_notes_import_once" not in screen_source


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "expected_phase", "expected_copy"),
    (
        (
            NotesSyncControlResult(True, "up_to_date", "sync_now", applied_count=2),
            "receipt",
            "2 applied · durable receipt recorded",
        ),
        (
            NotesSyncControlResult(False, "failed", "review_changes"),
            "roots",
            "Failed",
        ),
    ),
)
async def test_activation_result_routes_to_truthful_receipt_or_root_recovery(
    result: NotesSyncControlResult,
    expected_phase: str,
    expected_copy: str,
) -> None:
    class _Runtime:
        async def activate_root(
            self, _root_id: str, _authorization: object
        ) -> NotesSyncControlResult:
            return result

        def snapshot(self) -> NotesSyncRuntimeSnapshot:
            return NotesSyncRuntimeSnapshot(
                "active",
                "sync_now",
                (
                    NotesSyncRootRuntimeSnapshot(
                        "root-1", result.status, result.next_action
                    ),
                ),
            )

    class _Importer:
        def begin_selection(self) -> None:
            raise AssertionError("activation must not enter import")

    controller = LibraryNotesSyncController(
        runtime=_Runtime(),
        import_controller=_Importer(),
    )
    accepted = await controller.activate_root("root-1")

    assert accepted is result.accepted
    assert controller.snapshot.phase == expected_phase

    class _Host(App[None]):
        CSS_PATH = TldwCli.CSS_PATH

        def compose(self) -> ComposeResult:
            if expected_phase == "receipt":
                yield LibraryNotesCanvas(
                    mode="lasting_add",
                    lasting_sync_snapshot=controller.snapshot,
                    compact=True,
                )
            else:
                yield LibraryNotesSyncRootsCanvas(controller.snapshot)

    app = _Host()
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        capture = app.export_screenshot(simplify=True)
        if expected_phase == "receipt":
            assert app.query_one("#notes-sync-receipt").renderable == expected_copy
        else:
            assert expected_copy in capture
        assert "No changes were applied." not in capture
        assert "Activate reviewed root" not in capture
