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
from itertools import pairwise

import pytest
from textual.widgets import Static

from Tests.Widgets.Library.test_library_note_import_canvas import (
    _CanvasApp as _ImportHost,
    _snapshot as _import_snapshot,
)
from Tests.Widgets.Library.test_library_notes_add_from_files_canvas import _Host
from tldw_chatbook.Library.library_note_import_state import (
    UNIFORM_RUN_MIN,
    LibraryNoteImportItemSnapshot,
)
from tldw_chatbook.Library.library_notes_lasting_sync_state import (
    LastingSyncReview,
    LastingSyncReviewRow,
    initial_lasting_sync_snapshot,
)
from tldw_chatbook.Widgets.Library.library_note_import_canvas import (
    LibraryNoteImportCanvas,
    review_row_line,
)
from tldw_chatbook.Widgets.Library.library_notes_add_from_files_canvas import (
    LibraryNotesAddFromFilesCanvas,
)

pytestmark = pytest.mark.asyncio


_SAFE_FILES = 54


def _skip_review(relative_paths: tuple[str, ...]) -> LastingSyncReview:
    """A review holding nothing but one folder's skipped files."""
    rows = tuple(
        LastingSyncReviewRow(
            f"item-skip-{index}",
            "skipped",
            "Obsidian system folder",
            relative_path=path,
            reason="obsidian_system",
        )
        for index, path in enumerate(relative_paths)
    )
    return LastingSyncReview(
        root_id="root-1",
        observation_token="c" * 64,
        skip_count=len(rows),
        rows=rows,
        group_totals=(("skipped", "Obsidian system folder", len(rows)),),
        source="root",
    )


def _named_skip_paths(canvas, relative_paths: tuple[str, ...]) -> set[str]:
    """Return which individual files the canvas puts on screen by name.

    Reads the PAINTED text, so it does not care which widget or class a
    surface happens to use -- the two canvases build different row widgets
    and the question is only whether a reader sees files or a folder.
    """
    painted = " ".join(
        getattr(widget.renderable, "plain", str(widget.renderable))
        for widget in canvas.query(Static)
    )
    return {path for path in relative_paths if path in painted}


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
        # The safe rows only: the skipped ones are the granularity test's
        # business and follow Import once's rule, not this density budget.
        rows = [
            row
            for row in canvas.query(".library-notes-sync-review-row")
            if row.id is not None and int(row.id.rsplit("-", 1)[1]) < _SAFE_FILES
        ]
        assert len(rows) == _SAFE_FILES, len(rows)
        # Every one of them is classified body-less, which is what
        # drops the margin. With a message, because reverting the fix must
        # show the row COST, not a bare `assert False` (review F4).
        unclassified = [row.id for row in rows if "-plain" not in row.classes]
        assert not unclassified, (
            f"{len(unclassified)} body-less rows kept their margin: "
            f"{unclassified[:5]}"
        )
        heights = {row.outer_size.height for row in rows}
        assert heights == {1}, sorted(heights)
        # The margin is what the outer size does NOT include, and it was the
        # third of the three rows: measure the screen distance between
        # consecutive rows, which is the number the reader pays.
        tops = sorted(row.region.y for row in rows)
        gaps = {later - earlier for earlier, later in pairwise(tops)}
        assert gaps == {1}, sorted(gaps)


@pytest.mark.parametrize("files", [1, 4, UNIFORM_RUN_MIN, UNIFORM_RUN_MIN + 3])
async def test_both_reviews_report_a_skip_the_same_way_at_every_run_length(
    files: int,
) -> None:
    """task-32625 AC#3: one granularity, proven on BOTH canvases.

    Fix round 1 (review F1). The first version of this test composed only the
    sync canvas despite its name, and bought a claim that was false: the sync
    side had been changed to collapse every skipped run to a folder line at
    any length, while Import once still names the files under
    ``UNIFORM_RUN_MIN``. That did not close the disagreement, it inverted it
    below eight. Both paths now use the same threshold and the same
    folder-keyed run, so this renders BOTH from one fixture and asserts they
    agree -- at a length under the threshold, at it, and over it.
    """
    relative_paths = tuple(f".trash/Old idea {index}.md" for index in range(files))

    sync_snapshot = replace(
        initial_lasting_sync_snapshot(lasting_available=True),
        phase="review",
        review=_skip_review(relative_paths),
    )
    sync_app = _Host(sync_snapshot)
    async with sync_app.run_test(size=(100, 300)) as pilot:
        await pilot.pause()
        await pilot.pause()
        canvas = sync_app.query_one(LibraryNotesAddFromFilesCanvas)
        sync_named = _named_skip_paths(canvas, relative_paths)
        sync_collapsed = bool(canvas.query(".notes-sync-run"))

    import_app = _ImportHost(
        replace(
            _import_snapshot(),
            phase="review",
            status_line=f"Review {files} items before import.",
            preview_items=tuple(
                LibraryNoteImportItemSnapshot(
                    item_id=f"skip-{index}",
                    name=path,
                    classification="unsupported",
                    action="skip",
                    reason="Obsidian system folder",
                )
                for index, path in enumerate(relative_paths)
            ),
        )
    )
    async with import_app.run_test(size=(100, 300)) as pilot:
        await pilot.pause()
        await pilot.pause()
        canvas = import_app.query_one(LibraryNoteImportCanvas)
        import_named = _named_skip_paths(canvas, relative_paths)
        import_collapsed = bool(canvas.query(".note-import-run"))

    assert sync_named == import_named, (
        f"{files} skipped files: sync names {sorted(sync_named)}, "
        f"Import once names {sorted(import_named)}"
    )
    assert sync_collapsed == import_collapsed, (
        f"{files} skipped files: sync collapsed={sync_collapsed}, "
        f"Import once collapsed={import_collapsed}"
    )
    # And the shared rule is the one both were always meant to use.
    assert sync_collapsed is (files >= UNIFORM_RUN_MIN)


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
