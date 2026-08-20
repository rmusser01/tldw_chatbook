"""Tests for the LibraryNotesCanvas widget (list/sync modes)."""

from __future__ import annotations

from dataclasses import replace

import pytest
from textual.widgets import Button, Static

from Tests.textual_test_utils import widget_pilot  # noqa: F401
from tldw_chatbook.Library.library_notes_state import LibraryNotesListState
from tldw_chatbook.Library.library_notes_sync_state import LibraryNotesSyncState
from tldw_chatbook.Library.library_notes_tree_state import (
    LibraryNotesTreeProjection,
    LibraryNotesTreeRow,
)
from tldw_chatbook.Widgets.Library.library_notes_canvas import LibraryNotesCanvas

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
