"""Tests for the LibraryNotesCanvas widget (list/sync modes)."""

from __future__ import annotations

import pytest
from textual.widgets import Static

from Tests.textual_test_utils import widget_pilot
from tldw_chatbook.Library.library_notes_state import LibraryNotesListState
from tldw_chatbook.Library.library_notes_sync_state import LibraryNotesSyncState
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


# -- LIB-19: Database mode, Files mode, and Sync are three folder-notes
# concepts never related anywhere -- one placement sentence per surface.


async def test_database_mode_list_carries_a_placement_sentence(widget_pilot):
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


async def test_sync_mode_placement_sentence_names_files_mode(widget_pilot):
    async with await widget_pilot(
        LibraryNotesCanvas,
        mode="sync",
        sync_state=_sync_state(),
    ) as pilot:
        await pilot.pause()
        purpose = pilot.app.query_one(
            "#library-notes-sync-purpose", Static
        ).renderable
        text = getattr(purpose, "plain", str(purpose))
        assert "Files mode" in text
        assert "directly" in text
        assert "mirror" in text.lower()
