"""Sync-review density and skip granularity (task-32625).

A safe review row cost three screen rows -- its own line, an empty
conflict-choice panel held open by ``min-height: 1``, and the bottom margin
meant to separate that panel from the next row -- so a 54-file vault spent
162. And wave 4's per-file ``item_skips`` reported the same vault file by
file where Import once reports it folder by folder.

Every assertion here was first run against the unfixed tree and recorded
failing; the RED text is on the task.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from textual.widgets import Static

from Tests.Widgets.Library.test_library_notes_add_from_files_canvas import _Host
from tldw_chatbook.Library.library_notes_lasting_sync_state import (
    LastingSyncReview,
    LastingSyncReviewRow,
    initial_lasting_sync_snapshot,
)
from tldw_chatbook.Widgets.Library.library_note_import_canvas import review_row_line
from tldw_chatbook.Widgets.Library.library_notes_add_from_files_canvas import (
    LibraryNotesAddFromFilesCanvas,
)

pytestmark = pytest.mark.asyncio


_SAFE_FILES = 54


def _dense_review() -> LastingSyncReview:
    """A 54-file vault whose safe rows are too varied to collapse into runs."""
    safe = tuple(
        LastingSyncReviewRow(
            f"bind-{index}",
            "safe",
            "Create a Library note",
            relative_path=f"Folder {index}/2026-09-{index % 28 + 1:02d}.md",
            destination="Library / Daily notes",
        )
        for index in range(_SAFE_FILES)
    )
    trash = tuple(
        LastingSyncReviewRow(
            f"item-skip-{index}",
            "skipped",
            "Obsidian system folder",
            relative_path=f".trash/Old idea {index}.md",
            reason="obsidian_system",
        )
        for index in range(4)
    )
    templates = (
        LastingSyncReviewRow(
            "item-skip-templates",
            "skipped",
            "Obsidian system folder",
            relative_path="Templates/Daily.md",
            reason="obsidian_system",
        ),
    )
    return LastingSyncReview(
        root_id="root-1",
        observation_token="c" * 64,
        safe_count=_SAFE_FILES,
        skip_count=len(trash) + len(templates),
        rows=safe + trash + templates,
        group_totals=(
            ("safe", "Create a Library note", _SAFE_FILES),
            ("skipped", "Obsidian system folder", len(trash) + len(templates)),
        ),
        source="root",
    )


async def test_the_sync_review_spends_one_row_per_file() -> None:
    """task-32625 AC#2: a 54-file vault costs 54 rows, not 162.

    Born red at three rows each: the row's own line, an empty conflict-choice
    panel held open by its ``min-height: 1``, and the bottom margin meant to
    separate that panel from the next row.
    """
    snapshot = replace(
        initial_lasting_sync_snapshot(lasting_available=True),
        phase="review",
        review=_dense_review(),
    )
    app = _Host(snapshot)
    async with app.run_test(size=(100, 300)) as pilot:
        await pilot.pause()
        await pilot.pause()
        canvas = app.query_one(LibraryNotesAddFromFilesCanvas)
        rows = list(canvas.query(".library-notes-sync-review-row"))
        assert len(rows) == _SAFE_FILES
        heights = {row.outer_size.height for row in rows}
        assert heights == {1}, sorted(heights)
        # The margin is what the outer size does NOT include, and it was the
        # third of the three rows: measure the screen distance between
        # consecutive rows, which is the number the reader pays.
        tops = sorted(row.region.y for row in rows)
        assert {later - earlier for earlier, later in zip(tops, tops[1:])} == {1}, (
            sorted({later - earlier for earlier, later in zip(tops, tops[1:])})
        )


async def test_both_reviews_report_a_skip_at_folder_granularity() -> None:
    """task-32625 AC#3: one mental model, not two a screen apart.

    Born red with five rows naming five files (".trash/Old idea 0.md", …)
    where Import once names two folders.
    """
    snapshot = replace(
        initial_lasting_sync_snapshot(lasting_available=True),
        phase="review",
        review=_dense_review(),
    )
    app = _Host(snapshot)
    async with app.run_test(size=(100, 300)) as pilot:
        await pilot.pause()
        await pilot.pause()
        canvas = app.query_one(LibraryNotesAddFromFilesCanvas)
        skips = sorted(
            str(widget.renderable)
            for widget in canvas.query(Static)
            if widget.id and widget.id.startswith("notes-sync-review-skip-")
        )
        assert skips == [
            ".trash · 4 files · Obsidian system folder",
            "Templates · 1 file · Obsidian system folder",
        ], skips
        # No skipped row may still name a single file.
        assert not [
            widget
            for widget in canvas.query(".library-notes-sync-review-row")
            if ".trash/" in str(widget.query_one(Static).renderable)
        ]


def test_both_reviews_compose_a_row_line_through_the_one_helper() -> None:
    """task-32625 AC#1/AC#4: the shared component, and its honest limit.

    The shape both reviews render -- "name · what happens · where", each
    clause stripped of its own full stop -- has one definition. The row
    WIDGETS are still two: an Import row carries per-item Skip/Create/Update
    controls, a sync row carries conflict choices and a diff, and merging
    those is a bigger change than this task's density finding needs. What
    the two now share is this line, the group heading, the uniform-run
    collapse and the path budget -- four of the five pieces of the row.
    """
    from tldw_chatbook.Widgets.Library import (
        library_note_import_canvas,
        library_notes_add_from_files_canvas,
    )

    assert (
        library_notes_add_from_files_canvas.review_row_line
        is library_note_import_canvas.review_row_line
    )
    assert review_row_line("vault/.trash", "4 files", "Skipped.") == (
        "vault/.trash · 4 files · Skipped"
    )
    assert review_row_line("", "Create a Library note", "") == (
        "Create a Library note"
    )
