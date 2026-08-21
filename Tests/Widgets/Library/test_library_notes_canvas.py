"""Tests for the LibraryNotesCanvas widget (list/sync modes)."""

from __future__ import annotations

from dataclasses import replace

import pytest
from textual.app import ComposeResult
from textual.widgets import Button, Static

from Tests.textual_test_utils import widget_pilot  # noqa: F401
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Library.library_notes_state import (
    DatabaseNoteDraft,
    LibraryNoteSessionSnapshot,
    LibraryNotesListState,
    NormalizedDatabaseNote,
)
from tldw_chatbook.Library.library_note_import_state import (
    initial_note_import_snapshot,
    project_library_note_import_snapshot,
)
from tldw_chatbook.Library.library_notes_sync_state import LibraryNotesSyncState
from tldw_chatbook.Library.library_notes_lasting_sync_state import (
    initial_lasting_sync_snapshot,
)
from tldw_chatbook.Library.library_notes_tree_state import (
    LibraryNotesTreeProjection,
    LibraryNotesTreeRow,
)
from tldw_chatbook.Widgets.Library.library_notes_canvas import (
    LibraryNotePresentationState,
    LibraryNotesCanvas,
)

pytestmark = pytest.mark.asyncio


def _list_state() -> LibraryNotesListState:
    return LibraryNotesListState(
        rows=(),
        header_copy="Notes (0)",
        status_copy="",
        empty_copy="No notes yet. Create one to see it here.",
    )


def _sync_state() -> LibraryNotesSyncState:
    return LibraryNotesSyncState(
        folder="",
        direction="bidirectional",
        conflict="newer_wins",
        auto_sync=False,
        status_line="idle",
        activity_lines=(),
    )


def _editor_state(
    *,
    status: str = "Saved",
    saving: bool = False,
    transfer_status: str = "",
    transfer_running: bool = False,
    region: str = "editor",
) -> LibraryNotePresentationState:
    baseline = NormalizedDatabaseNote(
        "n-1",
        "Note",
        "Body",
        (),
        1,
        "2026-08-20T00:00:00+00:00",
        "2026-08-20T00:00:00+00:00",
    )
    snapshot = LibraryNoteSessionSnapshot(
        baseline=baseline,
        draft=DatabaseNoteDraft("n-1", "Note", "Body", "", 0),
        session_generation=1,
        saved_revision=0,
        dirty=False,
        saving=saving,
        in_conflict=False,
        conflict_generation=0,
        status_message=status,
    )
    return LibraryNotePresentationState(
        snapshot=snapshot,
        metadata_line="Updated today",
        status_line=status,
        region=region,
        transfer_status=transfer_status,
        transfer_running=transfer_running,
    )


def _tree_projection() -> LibraryNotesTreeProjection:
    return LibraryNotesTreeProjection(
        rows=(
            LibraryNotesTreeRow(
                placement_id="folder:personal",
                kind="folder",
                label="Personal",
                depth=0,
                folder_id="personal",
                breadcrumb="Personal",
                expanded=True,
                version=2,
            ),
            LibraryNotesTreeRow(
                placement_id="folder:ideas",
                kind="folder",
                label="Ideas",
                depth=1,
                folder_id="ideas",
                breadcrumb="Personal / Ideas",
                protected=True,
                semantic_status="connected",
                status_text="⇄ Sync managed",
                expanded=True,
                version=1,
            ),
            LibraryNotesTreeRow(
                placement_id="note:ideas:n1",
                kind="note",
                label="Garden redesign",
                depth=2,
                note_id="n1",
                folder_id="ideas",
                membership_id="m1",
                breadcrumb="Personal / Ideas / Garden redesign",
                ownership="managed",
                protected=True,
                semantic_status="connected",
                status_text="⇄ Synced placement",
            ),
            LibraryNotesTreeRow(
                placement_id="virtual:unfiled",
                kind="unfiled",
                label="Unfiled",
                depth=0,
                breadcrumb="Unfiled",
                expanded=True,
            ),
            LibraryNotesTreeRow(
                placement_id="unfiled:n2",
                kind="note",
                label="Loose thought",
                depth=1,
                note_id="n2",
                breadcrumb="Unfiled / Loose thought",
            ),
        ),
        next_note_offset=1000,
    )


# -- TASK-19000: source authority stays pinned across every Notes subview.


@pytest.mark.parametrize(
    ("mode", "kwargs", "status_fragment"),
    (
        (
            "loading",
            {"load_state": "failed", "load_message": "Could not load note."},
            "Could not load note.",
        ),
        ("list", {"list_state": _list_state()}, "Ready"),
        ("editor", {}, "Editor unavailable"),
        (
            "create",
            {"create_status": "Could not create note."},
            "Could not create note.",
        ),
        ("sync", {"sync_panel_state": _sync_state()}, "Sync idle"),
        (
            "import",
            {
                "import_snapshot": project_library_note_import_snapshot(
                    initial_note_import_snapshot()
                )
            },
            "Import once",
        ),
    ),
)
async def test_authority_row_is_first_plain_child_in_every_notes_mode(
    widget_pilot,  # noqa: F811
    mode: str,
    kwargs: dict[str, object],
    status_fragment: str,
):
    async with await widget_pilot(
        LibraryNotesCanvas,
        mode=mode,
        **kwargs,
    ) as pilot:
        await pilot.pause()
        canvas = pilot.app.query_one(LibraryNotesCanvas)
        authority = canvas.query_one("#library-notes-authority", Static)

        assert canvas.children[0] is authority
        assert authority._render_markup is False
        text = getattr(authority.renderable, "plain", str(authority.renderable))
        assert "Library notes" in text
        assert "Library database" in text
        assert status_fragment in text
        assert "Next:" in text


async def test_completed_import_receipt_has_focusable_back_action_at_60_columns():
    snapshot = replace(
        project_library_note_import_snapshot(initial_note_import_snapshot()),
        phase="receipt",
        status_line="Import finished.",
        receipt_line="1 imported · 0 updated · 0 skipped · 0 failed",
        receipt_detail="All planned items settled.",
    )

    class ImportReceiptApp(ConsolidatedCSSApp):
        def compose(self) -> ComposeResult:
            yield LibraryNotesCanvas(mode="import", import_snapshot=snapshot)

    app = ImportReceiptApp()
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        back = app.query_one("#library-notes-import-back", Button)
        back.focus()
        await pilot.pause()

        assert app.focused is back
        assert str(back.label) == "Back to Notes"
        assert back.disabled is False


async def test_cancelled_partial_import_offers_retry_without_calling_it_a_failure(
    widget_pilot,  # noqa: F811
):
    snapshot = replace(
        project_library_note_import_snapshot(initial_note_import_snapshot()),
        phase="receipt",
        status_line="Import stopped after the current item.",
        receipt_line="1 imported · 0 updated · 0 skipped · 0 failed",
        receipt_detail="Partial completion. Finished items were not rolled back.",
        retry_available=True,
        retry_label="Retry unfinished items",
    )

    async with await widget_pilot(
        LibraryNotesCanvas,
        mode="import",
        import_snapshot=snapshot,
    ) as pilot:
        await pilot.pause()
        retry = pilot.app.query_one("#note-import-retry", Button)

        assert str(retry.label) == "Retry unfinished items"
        assert "failure" not in str(retry.label).casefold()


async def test_lasting_setup_retained_wrapper_preserves_input_and_pins_action_at_60x20():
    snapshot = replace(
        initial_lasting_sync_snapshot(lasting_available=True), phase="configure"
    )

    class LastingSetupApp(ConsolidatedCSSApp):
        def compose(self) -> ComposeResult:
            yield LibraryNotesCanvas(
                mode="lasting_add",
                lasting_sync_snapshot=snapshot,
                compact=True,
            )

    app = LastingSetupApp()
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        canvas = app.query_one(LibraryNotesCanvas)
        name = app.query_one("#notes-sync-display-name")
        assert app.focused is name
        await pilot.press(*"Research[2026]")
        folder = app.query_one("#notes-sync-folder-choose", Button)
        folder.focus()
        canvas.sync_state(
            list_state=None,
            sort_mode="newest",
            filter_value="",
            mode="lasting_add",
            presentation_state=None,
            sync_panel_state=None,
            import_snapshot=None,
            import_receipt_available=False,
            tree_projection=None,
            tree_selected_placement_id="",
            tree_deleted_folder_available=False,
            title_placeholder_only=False,
            compact=True,
            create_running=False,
            create_status="",
            load_state="loading",
            load_message="",
            lasting_sync_snapshot=replace(snapshot, status_line="Folder selected."),
        )
        await pilot.pause()

        assert app.query_one("#notes-sync-display-name") is name
        assert name.value == "Research[2026]"
        assert app.focused is folder
        primary = app.query_one("#notes-sync-check", Button)
        primary.scroll_visible(immediate=True)
        await pilot.pause()
        assert primary in app.screen._compositor.visible_widgets
        hint = app.query_one("#notes-sync-fold-hint", Static)
        assert "Additional setup content is scrollable" in str(hint.renderable)
        assert hint in app.screen._compositor.visible_widgets
        assert canvas.region.right <= 60 and canvas.region.bottom <= 20


async def test_import_selection_summary_bounds_many_long_names(widget_pilot):  # noqa: F811
    names = tuple(f"[draft]-{'x' * 80}-{number}.md" for number in range(10))
    snapshot = replace(
        project_library_note_import_snapshot(initial_note_import_snapshot()),
        phase="destination",
        selected_names=names,
        selection_kind="files",
        status_line="10 files selected.",
    )

    async with await widget_pilot(
        LibraryNotesCanvas,
        mode="import",
        import_snapshot=snapshot,
    ) as pilot:
        await pilot.pause()
        summary = pilot.app.query_one("#note-import-source-summary", Static)
        text = str(summary.renderable)

        assert names[0][:40] in text
        assert names[2][:40] in text
        assert names[0] not in text
        assert names[3] not in text
        assert "and 7 more" in text
        assert len(text) < 200
        assert summary._render_markup is False


async def test_list_authority_running_without_status_uses_updating_fallback(
    widget_pilot,  # noqa: F811
):
    running = replace(
        _list_state(),
        operation_running=True,
        operation_status="",
    )
    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=running,
    ) as pilot:
        await pilot.pause()
        authority = pilot.app.query_one("#library-notes-authority", Static)
        text = getattr(authority.renderable, "plain", str(authority.renderable))

        assert "Updating notes…" in text
        assert "Ready" not in text
        assert "Next: Wait for the running notes operation to finish." in text


async def test_editor_authority_tracks_post_mount_save_state(widget_pilot):  # noqa: F811
    initial = _editor_state()
    async with await widget_pilot(
        LibraryNotesCanvas,
        mode="editor",
        presentation_state=initial,
    ) as pilot:
        await pilot.pause()
        canvas = pilot.app.query_one(LibraryNotesCanvas)
        authority = canvas.query_one("#library-notes-authority", Static)

        canvas.apply_session_state(_editor_state(status="Saving note…", saving=True))
        assert canvas.query_one("#library-notes-authority", Static) is authority
        text = getattr(authority.renderable, "plain", str(authority.renderable))
        assert "Saving note…" in text
        assert "Next: Wait for saving to finish." in text

        canvas.apply_session_state(_editor_state(status="Save failed: database busy"))
        text = getattr(authority.renderable, "plain", str(authority.renderable))
        assert "Save failed: database busy" in text
        assert "Next: Review the error, then keep editing." in text


async def test_editor_authority_tracks_transfer_through_context_navigation(
    widget_pilot,  # noqa: F811
):
    initial = _editor_state()
    async with await widget_pilot(
        LibraryNotesCanvas,
        mode="editor",
        presentation_state=initial,
    ) as pilot:
        await pilot.pause()
        canvas = pilot.app.query_one(LibraryNotesCanvas)
        authority = canvas.query_one("#library-notes-authority", Static)

        canvas.apply_session_state(
            _editor_state(
                transfer_status="Exporting Markdown…",
                transfer_running=True,
                region="context",
            )
        )
        text = getattr(authority.renderable, "plain", str(authority.renderable))
        assert "Saved" in text
        assert "Exporting Markdown…" in text
        assert "Next: Wait for export to finish." in text

        canvas.apply_session_state(
            _editor_state(
                transfer_status="Export failed: permission denied",
                region="editor",
            )
        )
        assert canvas.query_one("#library-notes-authority", Static) is authority
        text = getattr(authority.renderable, "plain", str(authority.renderable))
        assert "Export failed: permission denied" in text
        assert "Next: Review the error, then keep editing." in text


async def test_database_mode_list_carries_a_placement_sentence(widget_pilot):  # noqa: F811
    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=_list_state(),
    ) as pilot:
        await pilot.pause()
        purpose = pilot.app.query_one(
            "#library-notes-database-purpose", Static
        ).renderable
        text = getattr(purpose, "plain", str(purpose))
        assert "database" in text.lower()
        assert "Files" in text
        assert "Sync" in text


async def test_sync_mode_placement_sentence_names_files_mode(widget_pilot):  # noqa: F811
    async with await widget_pilot(
        LibraryNotesCanvas,
        mode="sync",
        sync_panel_state=_sync_state(),
    ) as pilot:
        await pilot.pause()
        purpose = pilot.app.query_one("#library-notes-sync-purpose", Static).renderable
        text = getattr(purpose, "plain", str(purpose))
        assert "Files mode" in text
        assert "directly" in text
        assert "mirror" in text.lower()


async def test_tree_projection_renders_hierarchy_and_placement_metadata(widget_pilot):  # noqa: F811
    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=_list_state(),
        tree_projection=_tree_projection(),
    ) as pilot:
        await pilot.pause()
        folders = list(pilot.app.query(".library-notes-folder-row"))
        notes = list(pilot.app.query(".library-notes-row"))

        assert [str(button.label) for button in folders] == [
            "▾ Personal",
            "  ▾ Ideas  ⇄ Sync managed",
            "▾ Unfiled",
        ]
        assert [button.note_id for button in notes] == ["n1", "n2"]
        assert notes[0].placement_id == "note:ideas:n1"
        assert notes[0].breadcrumb == "Personal / Ideas / Garden redesign"
        assert notes[0].membership_id == "m1"
        assert notes[0].protected_placement is True
        assert "⇄ Synced placement" in str(notes[0].label)


async def test_tree_projection_renders_visible_more_action(widget_pilot):  # noqa: F811
    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=_list_state(),
        tree_projection=_tree_projection(),
    ) as pilot:
        await pilot.pause()
        more = pilot.app.query_one("#library-notes-tree-more", Button)
        assert "more" in str(more.label).lower()


async def test_tree_managed_state_remains_legible_without_color(widget_pilot):  # noqa: F811
    projection = LibraryNotesTreeProjection(
        rows=(
            LibraryNotesTreeRow(
                placement_id="note:work:n1",
                kind="note",
                label="Recovered",
                depth=1,
                note_id="n1",
                folder_id="work",
                membership_id="m1",
                breadcrumb="Work / Recovered",
                ownership="managed",
                owner_active=False,
                protected=True,
                semantic_status="needs_attention",
                status_text="! Needs owner review",
            ),
        )
    )
    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=_list_state(),
        tree_projection=projection,
    ) as pilot:
        await pilot.pause()
        row = pilot.app.query_one(".library-notes-row", Button)
        assert "! Needs owner review" in str(row.label)
        assert row.has_class("library-notes-tree-needs-attention")


async def test_selected_folder_exposes_manual_folder_actions(widget_pilot):  # noqa: F811
    projection = _tree_projection()
    manual_rows = tuple(
        replace(row, protected=False, semantic_status="normal", status_text="")
        if row.placement_id == "folder:ideas"
        else row
        for row in projection.rows
    )
    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=_list_state(),
        tree_projection=LibraryNotesTreeProjection(
            rows=manual_rows,
            next_note_offset=projection.next_note_offset,
        ),
        tree_selected_placement_id="folder:ideas",
    ) as pilot:
        await pilot.pause()
        assert pilot.app.query_one("#library-notes-folder-new", Button)
        assert (
            pilot.app.query_one("#library-notes-folder-rename", Button).disabled
            is False
        )
        assert (
            pilot.app.query_one("#library-notes-folder-move", Button).disabled is False
        )
        assert (
            pilot.app.query_one("#library-notes-folder-remove", Button).disabled
            is False
        )


async def test_selected_managed_folder_disables_folder_mutations(widget_pilot):  # noqa: F811
    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=_list_state(),
        tree_projection=_tree_projection(),
        tree_selected_placement_id="folder:ideas",
    ) as pilot:
        await pilot.pause()
        for button_id in (
            "#library-notes-folder-new",
            "#library-notes-folder-rename",
            "#library-notes-folder-move",
            "#library-notes-folder-remove",
        ):
            button = pilot.app.query_one(button_id, Button)
            assert button.disabled is True
            assert "sync" in str(button.tooltip).lower()


async def test_managed_placement_disables_move_and_remove_but_allows_add(
    widget_pilot,  # noqa: F811
):
    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=_list_state(),
        tree_projection=_tree_projection(),
        tree_selected_placement_id="note:ideas:n1",
    ) as pilot:
        await pilot.pause()
        assert (
            pilot.app.query_one("#library-notes-placement-add", Button).disabled
            is False
        )
        assert (
            pilot.app.query_one("#library-notes-placement-move", Button).disabled
            is True
        )
        remove = pilot.app.query_one("#library-notes-placement-remove", Button)
        assert remove.disabled is True
        assert "sync" in str(remove.tooltip).lower()


async def test_deleted_folder_receipt_exposes_restore_action(widget_pilot):  # noqa: F811
    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=_list_state(),
        tree_projection=_tree_projection(),
        tree_deleted_folder_available=True,
    ) as pilot:
        await pilot.pause()
        assert pilot.app.query_one("#library-notes-folder-restore", Button)
