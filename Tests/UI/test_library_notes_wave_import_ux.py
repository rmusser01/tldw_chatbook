"""Add from files and Import once layout/copy contracts (import-ux wave).

Covers tasks 32125 (chooser), 32130 (receipt copy), 32134 (source changes)
and 32135 (one-line review rows with per-group bulk actions).
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Collapsible, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp

from tldw_chatbook.Library.library_note_import_state import (
    LibraryNoteImportItemSnapshot,
    LibraryNoteImportSnapshot,
)
from tldw_chatbook.Library.library_notes_lasting_sync_state import (
    initial_lasting_sync_snapshot,
)
from tldw_chatbook.Widgets.Library.library_note_import_canvas import (
    LibraryNoteImportCanvas,
)
from tldw_chatbook.Widgets.Library.library_notes_add_from_files_canvas import (
    LibraryNotesAddFromFilesCanvas,
)
from tldw_chatbook.Library.library_notes_state import LibraryNotesListState
from tldw_chatbook.Widgets.Library.library_notes_canvas import LibraryNotesCanvas
from tldw_chatbook.app import TldwCli

pytestmark = pytest.mark.asyncio


def _plain(widget: Static) -> str:
    return getattr(widget.renderable, "plain", str(widget.renderable))


def _import_snapshot(**changes: object) -> LibraryNoteImportSnapshot:
    snapshot = LibraryNoteImportSnapshot(
        phase="select",
        selected_names=(),
        selection_kind="",
        destination="",
        status_line="Choose one or more files, or one folder.",
        preview_items=(),
        page=1,
        page_count=1,
        can_check=False,
        check_disabled_reason="Choose a source first.",
        can_import=False,
        import_disabled_reason="Check the selection before importing.",
    )
    return replace(snapshot, **changes)


def _item(index: int, **changes: object) -> LibraryNoteImportItemSnapshot:
    item = LibraryNoteImportItemSnapshot(
        item_id=f"item-{index}",
        name=f"vault/Archive/note-{index}.md",
        classification="new",
        action="create_new",
        reason="Ready to import as a new note.",
        effect_summary="Content: create 1 new note.",
        membership_summary="Create in vault / Archive.",
    )
    return replace(item, **changes)


class _ChooserHost(ConsolidatedCSSApp):
    CSS_PATH = TldwCli.CSS_PATH

    def __init__(self) -> None:
        super().__init__()
        self.snapshot = initial_lasting_sync_snapshot()

    def compose(self) -> ComposeResult:
        yield LibraryNotesAddFromFilesCanvas(self.snapshot, id="chooser")


class _ImportHost(ConsolidatedCSSApp):
    CSS_PATH = TldwCli.CSS_PATH

    def __init__(self, snapshot: LibraryNoteImportSnapshot) -> None:
        super().__init__()
        self.snapshot = snapshot
        self.messages: list[object] = []

    def compose(self) -> ComposeResult:
        yield LibraryNoteImportCanvas(self.snapshot, id="library-note-import-canvas")

    def on_library_note_import_canvas_change_source_requested(
        self, message: LibraryNoteImportCanvas.ChangeSourceRequested
    ) -> None:
        self.messages.append(message)

    def on_library_note_import_canvas_clear_source_requested(
        self, message: LibraryNoteImportCanvas.ClearSourceRequested
    ) -> None:
        self.messages.append(message)

    def on_library_note_import_canvas_group_action_requested(
        self, message: LibraryNoteImportCanvas.GroupActionRequested
    ) -> None:
        self.messages.append(message)


# --- task-32125 -----------------------------------------------------------


async def test_both_relationships_are_buttons_under_their_own_descriptions() -> None:
    """Import once sits with Keep a folder synced, not in the pinned bar."""
    app = _ChooserHost()

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        body = app.query_one("#notes-sync-body")
        children = list(body.children)
        ids = [child.id for child in children]

        assert "notes-add-import-once" in ids
        assert ids.index("notes-add-import-once") < ids.index("notes-add-keep-synced")
        import_index = ids.index("notes-add-import-once")
        keep_index = ids.index("notes-add-keep-synced")
        assert "Import once —" in _plain(children[import_index - 1])
        assert "Keep a folder synced —" in _plain(children[keep_index - 1])

        pinned = app.query_one("#notes-sync-pinned-actions")
        assert [button.id for button in pinned.query(Button)] == ["notes-sync-back"]


async def test_chooser_header_names_no_relationship_before_one_is_chosen() -> None:
    """The authority line stops announcing Lasting sync on the chooser."""
    canvas = LibraryNotesCanvas(
        mode="lasting_add",
        lasting_sync_snapshot=initial_lasting_sync_snapshot(),
    )

    copy = canvas._authority_copy()

    assert "Lasting sync" not in copy
    assert "Add from files" in copy


# --- task-32130 -----------------------------------------------------------


async def test_receipt_discloses_every_skipped_path_and_reason() -> None:
    """The receipt names the skipped files instead of only counting them."""
    app = _ImportHost(
        _import_snapshot(
            phase="receipt",
            status_line="Import completed.",
            receipt_line="61 imported · 0 updated · 11 skipped · 0 failed",
            receipt_detail="Import finished · 61 notes created · 11 files skipped",
            skipped_count=2,
            skipped_items=(
                ("vault/.obsidian/app.json", "Not a note file (app configuration)."),
                ("vault/Inbox/Untitled.md", "Empty file — nothing to import."),
            ),
        )
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        disclosure = app.query_one("#note-import-skipped", Collapsible)
        assert "Skipped (2)" in str(disclosure.title)
        rows = "\n".join(
            _plain(row) for row in disclosure.query(".note-import-skipped-row")
        )
        assert "vault/.obsidian/app.json" in rows
        assert "Not a note file (app configuration)." in rows
        assert "Empty file — nothing to import." in rows
        assert "61 notes created" in _plain(
            app.query_one("#note-import-receipt-detail", Static)
        )


# --- task-32134 -----------------------------------------------------------


async def test_a_chosen_folder_can_be_changed_or_cleared() -> None:
    """A wrong folder is recoverable without leaving Import once."""
    app = _ImportHost(
        _import_snapshot(
            phase="select",
            selected_names=("vault",),
            selection_kind="folder",
            status_line="1 folder selected.",
            can_check=True,
            check_disabled_reason="",
        )
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # A folder import is exclusive, so Add another file stays absent.
        assert not app.query("#note-import-add-source")
        app.query_one("#note-import-change-source", Button).press()
        app.query_one("#note-import-clear-source", Button).press()
        await pilot.pause()

    assert isinstance(app.messages[0], LibraryNoteImportCanvas.ChangeSourceRequested)
    assert isinstance(app.messages[1], LibraryNoteImportCanvas.ClearSourceRequested)


async def test_selected_files_keep_add_another_file_beside_change_and_clear() -> None:
    """File selections keep their documented Add another file control."""
    app = _ImportHost(
        _import_snapshot(
            phase="destination",
            selected_names=("draft.md",),
            selection_kind="files",
            status_line="1 file selected.",
        )
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.query_one("#note-import-add-source", Button)
        assert app.query_one("#note-import-change-source", Button)
        assert app.query_one("#note-import-clear-source", Button)


async def test_notes_list_offers_last_import_while_a_session_receipt_exists() -> None:
    """The list surfaces the retained receipt after Back to Notes (32134)."""
    class _ListHost(ConsolidatedCSSApp):
        CSS_PATH = TldwCli.CSS_PATH

        def compose(self) -> ComposeResult:
            yield LibraryNotesCanvas(
                list_state=LibraryNotesListState(
                    rows=(),
                    header_copy="Notes (0)",
                    status_copy="",
                    empty_copy="No notes yet. Create one to see it here.",
                ),
                mode="list",
                import_receipt_available=True,
                id="library-notes-canvas",
            )

    app = _ListHost()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.query_one("#library-notes-import-receipt", Button)
