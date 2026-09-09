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


# --- task-32135 -----------------------------------------------------------


async def test_review_rows_are_one_line_with_their_controls_adjacent() -> None:
    """Path, action and destination share one row with Skip/Create."""
    app = _ImportHost(
        _import_snapshot(
            phase="review",
            status_line="Review 1 item before import.",
            preview_items=(_item(1),),
            can_import=True,
            import_disabled_reason="",
        )
    )

    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        row = app.query_one(".note-import-row", Horizontal)
        text = _plain(row.query_one(".note-import-row-text", Static))
        assert "vault/Archive/note-1.md" in text
        assert "create 1 new note" in text
        assert "vault / Archive" in text
        assert row.query_one("#note-import-action-item-1-skip", Button)
        assert row.query_one("#note-import-action-item-1-create", Button)
        assert row.size.height == 1


async def test_each_review_group_offers_skip_all_and_create_all() -> None:
    """A 71-item review can be settled per action class, not row by row."""
    app = _ImportHost(
        _import_snapshot(
            phase="review",
            status_line="Review 3 items before import.",
            preview_items=(
                _item(1),
                _item(2),
                _item(
                    3,
                    classification="unsupported",
                    action="skip",
                    reason="This file type is not supported.",
                    effect_summary="Content: no change.",
                    membership_summary="Folder placement: no change.",
                ),
            ),
        )
    )

    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        headings = [
            _plain(heading)
            for heading in app.query(".note-import-group-heading").results(Static)
        ]
        assert "New (2)" in headings
        assert "Unsupported (1)" in headings
        app.query_one("#note-import-group-new-skip", Button).press()
        await pilot.pause()
        # An unsupported group cannot create anything, so it offers Skip only.
        assert not app.query("#note-import-group-unsupported-create")
        assert app.query_one("#note-import-group-new-create", Button)

    message = app.messages[-1]
    assert isinstance(message, LibraryNoteImportCanvas.GroupActionRequested)
    assert (message.classification, message.action) == ("new", "skip")


async def test_review_shows_at_least_fifteen_rows_at_235x52() -> None:
    """A full review page fits on one screen at the wide terminal size."""
    app = _ImportHost(
        _import_snapshot(
            phase="review",
            status_line="Review 25 items before import.",
            preview_items=tuple(_item(index) for index in range(1, 26)),
            can_import=True,
            import_disabled_reason="",
        )
    )

    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        visible = app.screen._compositor.visible_widgets
        rows = [row for row in app.query(".note-import-row") if row in visible]
        assert len(rows) >= 15


async def test_review_reserves_an_options_slot_above_the_groups() -> None:
    """Task 5's Obsidian toggle has a named home above the grouped rows."""
    app = _ImportHost(
        _import_snapshot(
            phase="review",
            status_line="Review 1 item before import.",
            preview_items=(_item(1),),
        )
    )

    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        body = app.query_one("#note-import-body")
        ids = [child.id for child in body.children]
        assert "notes-import-review-options" in ids
        # An empty 1fr slot would push every group to the bottom of the body.
        assert app.query_one("#notes-import-review-options").size.height == 0
        first_heading = next(
            child
            for child in body.children
            if child.has_class("note-import-group-row")
        )
        assert ids.index("notes-import-review-options") < body.children.index(
            first_heading
        )


async def test_a_clipped_row_still_reaches_its_effect_and_destination() -> None:
    """At 60 columns the row clips, so the rest stays reachable (32135 review)."""
    from textual.containers import Container

    snapshot = _import_snapshot(
        phase="review",
        status_line="Review 1 item before import.",
        preview_items=(_item(1),),
        can_import=True,
        import_disabled_reason="",
    )

    class _CompactHost(ConsolidatedCSSApp):
        CSS_PATH = TldwCli.CSS_PATH

        def compose(self) -> ComposeResult:
            with Container(id="library-canvas", classes="library-notes-compact"):
                yield LibraryNoteImportCanvas(
                    snapshot,
                    compact=True,
                    id="library-note-import-canvas",
                )

    app = _CompactHost()
    async with app.run_test(size=(60, 24)) as pilot:
        await pilot.pause()
        summary = app.query_one(".note-import-row-text", Static)
        # The row is clipped at this width…
        assert summary.size.width < len(_plain(summary))
        # …so the whole sentence is on the tooltip,
        assert "create 1 new note" in str(summary.tooltip)
        assert "vault / Archive" in str(summary.tooltip)
        # and the destination repeats on its own line.
        destination = app.query_one(".note-import-row-destination", Static)
        assert destination.display is True
        assert "vault / Archive" in _plain(destination)


async def test_every_review_control_stays_inside_a_narrow_viewport() -> None:
    """A row's trailing buttons must be reachable, not clipped off-screen."""
    from textual.containers import Container

    snapshot = _import_snapshot(
        phase="review",
        status_line="Review 2 items before import.",
        preview_items=(
            _item(
                1,
                classification="uncertain_match",
                action="create_new",
                reason="This source may match an existing note.",
                uncertain=True,
            ),
            _item(
                2,
                classification="changed_repeat",
                action="update_existing",
                reason="This source differs from an existing note.",
                can_update=True,
                effect_summary="Content: keep existing content.",
                membership_summary="Folder placement: unchanged.",
            ),
        ),
        can_import=True,
        import_disabled_reason="",
    )

    class _CompactHost(ConsolidatedCSSApp):
        CSS_PATH = TldwCli.CSS_PATH

        def compose(self) -> ComposeResult:
            with Container(id="library-canvas", classes="library-notes-compact"):
                yield LibraryNoteImportCanvas(
                    snapshot,
                    compact=True,
                    id="library-note-import-canvas",
                )

    app = _CompactHost()
    async with app.run_test(size=(60, 24)) as pilot:
        await pilot.pause()
        body = app.query_one("#note-import-body")
        buttons = list(body.query(Button))
        assert len(buttons) >= 9  # both rows' full control sets are mounted
        overflowing = [
            button.id
            for button in buttons
            if button.region.right > body.content_region.right
            or button.region.x < body.content_region.x
        ]
        assert overflowing == []


async def test_a_wide_row_does_not_repeat_its_destination() -> None:
    """The second line is compact-only; the wide row already ends in it."""
    app = _ImportHost(
        _import_snapshot(
            phase="review",
            status_line="Review 1 item before import.",
            preview_items=(_item(1),),
        )
    )

    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        assert not app.query(".note-import-row-destination")


# --- task-32176 -----------------------------------------------------------


async def test_group_actions_say_they_only_change_this_page() -> None:
    """A bulk action settles the rendered page, so its label says so."""
    app = _ImportHost(
        _import_snapshot(
            phase="review",
            status_line="Review 2 items before import.",
            preview_items=(
                _item(1),
                _item(
                    2,
                    classification="unsupported",
                    action="skip",
                    reason="This file type is not supported.",
                ),
            ),
        )
    )

    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        labels = {
            str(button.label)
            for button in app.query(".note-import-group-action").results(Button)
        }
        assert labels == {"Skip all on this page", "Create all on this page"}


def test_the_canvas_takes_its_non_importable_set_from_the_planner_enum() -> None:
    """task-32176: one source of truth for what cannot be imported."""
    from tldw_chatbook.Notes.note_import_plan_models import (
        NON_IMPORTABLE_CLASSIFICATIONS,
        ImportClassification,
    )
    from tldw_chatbook.Widgets.Library import library_note_import_canvas

    assert library_note_import_canvas._NON_IMPORTABLE == {
        classification.value for classification in NON_IMPORTABLE_CLASSIFICATIONS
    }
    # Every classification still needs a group label, which is the third place
    # the set used to be spelled out by hand.
    assert set(library_note_import_canvas._CLASSIFICATION_LABELS) == {
        classification.value for classification in ImportClassification
    }
