"""Library ▸ Notes toolbar gating and source vocabulary -- critique #9.

Group `notes` of the critique-9 fix wave:

- task-32215 AC#2: the three selection-scoped folder verbs (`Add to folder`,
  `Move note`, `Remove placement`) must not stand in the toolbar with no
  row selected. AC#1 (Sort on a populated list) was already delivered by the
  Notes wave's PR #2558 and is pinned by its own tests -- nothing here
  re-pins Sort in either direction.
- task-32218: one noun per Notes source. `Library notes` for the notes the
  Library keeps itself, `Folder files` for the notes that live as files in a
  folder -- and no `Library database`, `Database Notes` or capitalised
  `Folder Files` anywhere a reader can see them.
"""

from __future__ import annotations

import re

import pytest
from textual.widgets import Button

from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Library.library_notes_state import LibraryNotesListState
from tldw_chatbook.Library.library_notes_tree_state import (
    LibraryNotesTreeProjection,
    LibraryNotesTreeRow,
)
from tldw_chatbook.Widgets.Library.library_notes_canvas import LibraryNotesCanvas

#: The three actions that only mean something once a note placement is the
#: selected row. `New folder` is NOT one of them -- it makes a folder next to
#: whatever is showing.
SELECTION_SCOPED_FOLDER_VERBS = (
    "#library-notes-placement-add",
    "#library-notes-placement-move",
    "#library-notes-placement-remove",
)

#: Every retired spelling of the two sources. `Database` is matched as a whole
#: word and case-sensitively: the empty state's prose ("the Library's own
#: database") describes where the notes live and is not a name.
RETIRED_SOURCE_WORDS = (
    "Library database",
    "Database Notes",
    "Database notes",
    "Folder Files",
    "back to Database",
)


def _notes_host() -> LibraryHarness:
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=[{"title": "Research Note", "id": "note-1"}],
    )
    return LibraryHarness(app)


def _painted(screen) -> str:
    """The whole painted frame as one plain-text string."""
    return "\n".join(
        "".join(segment.text for segment in strip)
        for strip in screen._compositor.render_strips()
    )


def _assert_no_retired_source_words(painted: str, where: str) -> None:
    for retired in RETIRED_SOURCE_WORDS:
        assert retired not in painted, f"{where} still paints {retired!r}"
    stray = re.search(r"\bDatabase\b", painted)
    assert stray is None, f"{where} still names a source 'Database'"


def _tree(*, selected: str) -> LibraryNotesTreeProjection:
    return LibraryNotesTreeProjection(
        rows=(
            LibraryNotesTreeRow(
                placement_id="folder:ideas",
                kind="folder",
                label="Ideas",
                depth=0,
                folder_id="ideas",
                focus_id="library-notes-folder-ideas",
                expanded=True,
            ),
            LibraryNotesTreeRow(
                placement_id=selected or "note:ideas:n1:m1",
                kind="note",
                label="Research Note",
                depth=1,
                note_id="n1",
                membership_id="m1",
                folder_id="ideas",
                focus_id="library-notes-tree-note-n1",
            ),
        )
    )


# --- task-32215 AC#2 --------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", ["", "note:ideas:n1:m1"])
async def test_selection_scoped_folder_verbs_need_a_selected_row(
    widget_pilot, selected
):
    """The three placement verbs appear only once a note row is selected.

    Live at dev 02374bf66a all four transfer verbs stood in the toolbar on a
    plain seven-note list, three of them meaningless with nothing selected
    and none of them carrying a reason. `New folder` stays either way.
    """
    async with await widget_pilot(
        LibraryNotesCanvas,
        list_state=LibraryNotesListState(
            rows=(),
            header_copy="Notes (1)",
            status_copy="",
            empty_copy="No notes yet. Create your first note.",
        ),
        tree_projection=_tree(selected=selected),
        tree_selected_placement_id=selected,
        pane_width=120,
    ) as pilot:
        await pilot.pause()
        assert pilot.app.query_one("#library-notes-folder-new", Button)
        for button_id in SELECTION_SCOPED_FOLDER_VERBS:
            found = pilot.app.query(button_id)
            assert bool(found) is bool(selected), (
                f"{button_id} present={bool(found)} with selected={selected!r}"
            )


# --- task-32218 -------------------------------------------------------------


@pytest.mark.asyncio
async def test_library_notes_paints_one_noun_for_its_own_source():
    """AC#1, database side: `Library notes`, and nothing else."""
    host = _notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-filter")
        await pilot.pause()

        painted = _painted(screen)
        assert "Library notes" in painted, "the source's own noun is not painted"
        _assert_no_retired_source_words(painted, "Library notes")


@pytest.mark.asyncio
async def test_folder_files_paints_one_noun_for_the_on_disk_source():
    """AC#1, files side: `Folder files`, and an Escape chip naming where it goes."""
    host = _notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-source-files")
        screen.query_one("#library-notes-source-files", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._file_notes_active(),
            message="Folder files never took over the view.",
        )
        await pilot.pause()

        painted = _painted(screen)
        assert "Folder files" in painted, "the source's own noun is not painted"
        _assert_no_retired_source_words(painted, "Folder files")
        assert ("esc", "back to Library notes") in (
            screen._library_footer_shortcuts_for_current_state()
        )


# --- task-32217 AC#2, the note-editor clause --------------------------------


@pytest.mark.asyncio
async def test_the_note_editor_body_takes_the_height_its_pane_has_spare():
    """The Body is this pane's primary content box, so it fills it.

    Its ceiling (`max-height: 20`) dated from when the keywords, meta line
    and action row still sat below the TextArea; they moved into Info, so
    `#library-note-editor-region` ends at this field and the ceiling only
    bought blank pane. Measured live at 235x52 on the seeded profile before
    this change: a 3-line note got a 12-row box with 18 empty rows under it.
    """
    host = _notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        row = await _wait_for_selector(screen, pilot, ".library-notes-tree-note-row")
        row.press()
        body = await _wait_for_selector(screen, pilot, "#library-note-body")
        await pilot.pause()
        await pilot.pause()

        region = screen.query_one("#library-note-editor-region")
        assert region.region.height > 20, region.region
        # One row of the field's own bottom margin is all that may be left.
        assert body.region.bottom >= region.region.bottom - 1, (
            f"the Body box stops at {body.region.bottom} in a pane that ends "
            f"at {region.region.bottom}: {region.region.bottom - body.region.bottom} "
            "blank rows under the note."
        )


@pytest.mark.asyncio
async def test_the_note_preview_takes_the_same_height_the_body_does():
    """Preview replaces the Body one-for-one, so it fills the pane too.

    Qodo #2590 comment 3: the component rule this branch added is a bare
    `#library-note-preview-region`, but `LibraryScreen.BUNDLED_CSS` carries its
    own copy, which the build scopes to `LibraryScreen #library-note-preview-
    region` -- one type selector more specific, so it kept the retired
    `height: auto / min-height: 12 / max-height: 20`. Edit expanded and Preview
    stayed capped at 20 rows in a 30-row pane. `#library-note-body` has no such
    scoped copy, which is why only Preview was affected.
    """
    host = _notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        row = await _wait_for_selector(screen, pilot, ".library-notes-tree-note-row")
        row.press()
        await _wait_for_selector(screen, pilot, "#library-note-body")
        screen.query_one("#library-note-preview", Button).press()
        preview = await _wait_for_selector(screen, pilot, "#library-note-preview-region")
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one("#library-note-preview-region").region.height > 0,
            message="Preview never took the work pane.",
        )

        parent = preview.parent
        assert parent.content_region.height > 20, parent.content_region
        # One row of the region's own bottom margin is all that may be left.
        assert preview.region.bottom >= parent.content_region.bottom - 1, (
            f"Preview stops at {preview.region.bottom} in a pane whose content "
            f"ends at {parent.content_region.bottom}: "
            f"{parent.content_region.bottom - preview.region.bottom} blank rows "
            "under the rendered note."
        )
