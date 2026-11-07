"""The inline delete prompt says what happens to a synced note's file.

TASK-32633 fix round 1 (review finding N-03, third sentence). A note in a
sync folder is never removed from disk by its delete: lasting sync holds the
folder for review and, in this release, only restoring the note resolves it
(TASK-34000.15). The prompt says so, in plain words, for a synced note only;
an unsynced note keeps the copy task-32268 shipped. The live choice is made
at show time from the same location the row states, so the real-app pin is
the synced assertion in ``Tests/UI/test_library_notes_sync_delete_restore.py``.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Widgets.Library.library_notes_canvas import (
    DELETE_CONFIRM_COPY,
    DELETE_CONFIRM_COPY_SYNCED,
    delete_confirm_copy,
)

pytestmark = pytest.mark.unit


def test_an_unsynced_note_keeps_the_shipped_prompt() -> None:
    assert delete_confirm_copy(synced=False) == DELETE_CONFIRM_COPY
    assert DELETE_CONFIRM_COPY == (
        "Delete this note? Undo will be available in the Notes list."
    )


def test_a_synced_note_is_told_its_file_stays_and_the_folder_waits() -> None:
    copy = delete_confirm_copy(synced=True)
    assert copy == DELETE_CONFIRM_COPY_SYNCED
    assert copy.startswith("Delete this note?")
    assert "file stays on disk" in copy
    assert "until the note is restored" in copy
    assert copy.endswith("Undo will be available in the Notes list.")
    # N-10: the Info confirm clips at 120x36; the sentence stays short.
    assert len(copy) <= 150
