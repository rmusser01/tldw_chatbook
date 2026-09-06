"""Regression: a note rename propagates to the Database Notes tree list rows.

task-31796: the Database Notes list renders placement rows from the cached
tree branch slices (and, while filtering, from the filter window) -- NOT from
the flat source records the save-time patch used to touch. So after renaming a
freshly created note and returning to the list, the "Unfiled" row still read
"Untitled" until a filter re-query rebuilt the slices. ``_patch_library_note_
list_from_session`` must now also retitle the matching placement in the cached
branch slices (and the active filter window) so the next canvas sync shows the
real title without a full DB re-query.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from tldw_chatbook.Library.library_notes_tree_paging import (
    NotesBranchKey,
    NotesBranchSliceState,
    empty_notes_slice,
    patch_notes_tree_branches_title,
)
from tldw_chatbook.Library.library_notes_tree_state import (
    LibraryNotesFilterState,
    build_filtered_library_notes_tree,
    build_paged_library_notes_tree,
    patch_notes_filter_state_title,
)
from tldw_chatbook.Notes.note_folder_models import (
    FolderPlacementId,
    NoteFolder,
    NoteFolderMembership,
    NotePlacementRecord,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


# --- builders (mirroring Tests/Library/test_library_notes_tree_state.py) ---


def _folder(folder_id: str, parent_id: str | None, path: str) -> NoteFolder:
    return NoteFolder(
        folder_id=folder_id,
        parent_id=parent_id,
        name=path.rsplit("/", 1)[-1],
        path=path,
        normalized_path=path.casefold(),
        version=1,
        deleted=False,
    )


def _membership(membership_id: str, folder_id: str, note_id: str) -> NoteFolderMembership:
    return NoteFolderMembership(
        membership_id=membership_id,
        folder_id=folder_id,
        note_id=note_id,
        ownership="manual",
        owner_id="",
        owner_active=True,
        version=1,
    )


def _placement(
    note_id: str,
    title: str,
    folder_id: str | None,
    membership_id: str | None = None,
) -> NotePlacementRecord:
    membership = (
        _membership(membership_id or f"m-{note_id}", folder_id, note_id)
        if folder_id is not None
        else None
    )
    return NotePlacementRecord(
        note={"id": note_id, "title": title},
        folder_id=folder_id,
        membership=membership,
    )


def _placements_branch(
    parent_id: str | None, *items: NotePlacementRecord
) -> NotesBranchSliceState:
    key = NotesBranchKey(parent_id, "placements")
    item_ids = tuple(
        FolderPlacementId.unfiled(str(item.note["id"]))
        if item.folder_id is None
        else FolderPlacementId.note(
            item.folder_id, str(item.note["id"]), item.membership.membership_id
        )
        for item in items
    )
    return replace(
        empty_notes_slice(key),
        items=tuple(items),
        item_ids=item_ids,
        total=len(items),
    )


def _folders_branch(parent_id: str | None, *items: NoteFolder) -> NotesBranchSliceState:
    key = NotesBranchKey(parent_id, "folders")
    return replace(
        empty_notes_slice(key),
        items=tuple(items),
        item_ids=tuple(FolderPlacementId.folder(f.folder_id) for f in items),
        total=len(items),
    )


def _note_row_labels(rows) -> list[str]:
    return [row.label for row in rows if row.kind == "note"]


# --- pure helper: branch-slice retitle ------------------------------------


def test_patch_branches_retitles_unfiled_placement_and_projects_new_label() -> None:
    branches = {
        NotesBranchKey(None, "placements"): _placements_branch(
            None, _placement("n1", "Untitled", None)
        ),
    }

    patched = patch_notes_tree_branches_title(
        branches, note_id="n1", title="Renamed Note Title", modified_at="2026-09-06T00:00:00Z"
    )

    projection = build_paged_library_notes_tree(branch_states=patched, expanded_folder_ids=set())
    assert _note_row_labels(projection.rows) == ["Renamed Note Title"]
    # The placement's note mapping carries the fresh title (and timestamp).
    slice_state = patched[NotesBranchKey(None, "placements")]
    note = slice_state.items[0].note
    assert note["title"] == "Renamed Note Title"
    assert note["last_modified"] == "2026-09-06T00:00:00Z"


def test_patch_branches_retitles_placement_nested_in_folder_slice() -> None:
    branches = {
        NotesBranchKey(None, "folders"): _folders_branch(
            None, _folder("f1", None, "/Work")
        ),
        NotesBranchKey("f1", "placements"): _placements_branch(
            "f1", _placement("n9", "Untitled", "f1", "m-n9")
        ),
    }

    patched = patch_notes_tree_branches_title(branches, note_id="n9", title="Filed Rename")

    projection = build_paged_library_notes_tree(
        branch_states=patched, expanded_folder_ids={"f1"}
    )
    assert "Filed Rename" in _note_row_labels(projection.rows)


def test_patch_branches_leaves_other_notes_and_folder_slices_untouched() -> None:
    folders = _folders_branch(None, _folder("f1", None, "/Work"))
    placements = _placements_branch(
        None,
        _placement("n1", "Untitled", None),
        _placement("n2", "Keep Me", None),
    )
    branches = {
        NotesBranchKey(None, "folders"): folders,
        NotesBranchKey(None, "placements"): placements,
    }

    patched = patch_notes_tree_branches_title(branches, note_id="n1", title="Now Named")

    # Folder slice identity is preserved (no needless rebuild).
    assert patched[NotesBranchKey(None, "folders")] is folders
    labels = _note_row_labels(
        build_paged_library_notes_tree(branch_states=patched, expanded_folder_ids=set()).rows
    )
    assert set(labels) == {"Now Named", "Keep Me"}


def test_patch_branches_no_match_returns_slice_identity_unchanged() -> None:
    placements = _placements_branch(None, _placement("n1", "Untitled", None))
    branches = {NotesBranchKey(None, "placements"): placements}

    patched = patch_notes_tree_branches_title(branches, note_id="absent", title="X")

    assert patched[NotesBranchKey(None, "placements")] is placements


# --- pure helper: filter-window retitle -----------------------------------


def test_patch_filter_state_retitles_matching_window_placement() -> None:
    state = LibraryNotesFilterState(
        query="ren",
        placements=(_placement("n1", "Untitled", None),),
        ancestor_folders=(),
        total=1,
        start_offset=0,
        previous_offset=None,
        next_offset=None,
        generation=1,
        topology_epoch=1,
    )

    patched = patch_notes_filter_state_title(state, note_id="n1", title="Renamed")

    labels = _note_row_labels(build_filtered_library_notes_tree(patched).rows)
    assert labels == ["Renamed"]


def test_patch_filter_state_no_match_returns_same_object() -> None:
    state = LibraryNotesFilterState.empty(query="q", generation=1, topology_epoch=1)
    assert patch_notes_filter_state_title(state, note_id="n1", title="X") is state


# --- screen wiring: _patch_library_note_list_from_session ------------------


def _wiring_screen(
    branches: dict, *, filter_state=None
) -> SimpleNamespace:
    baseline = SimpleNamespace(
        note_id="n1",
        title="Renamed Note Title",
        modified_at="2026-09-06T00:00:00Z",
    )
    snapshot = SimpleNamespace(baseline=baseline)
    return SimpleNamespace(
        _library_note_session=SimpleNamespace(snapshot=snapshot),
        _local_source_records={"notes": ({"id": "n1", "title": "Untitled"},)},
        _library_notes_filter_records=None,
        _library_notes_tree_branches=branches,
        _library_notes_tree_filter_state=filter_state,
    )


def test_save_patch_updates_tree_branch_row_without_requery() -> None:
    """The exact reproduced defect: rename → save must update the tree row."""
    branches = {
        NotesBranchKey(None, "placements"): _placements_branch(
            None, _placement("n1", "Untitled", None)
        ),
    }
    screen = _wiring_screen(branches)

    LibraryScreen._patch_library_note_list_from_session(screen)

    labels = _note_row_labels(
        build_paged_library_notes_tree(
            branch_states=screen._library_notes_tree_branches,
            expanded_folder_ids=set(),
        ).rows
    )
    assert labels == ["Renamed Note Title"]
    # Flat records are still patched too (pre-existing behavior preserved).
    assert screen._local_source_records["notes"][0]["title"] == "Renamed Note Title"


def test_save_patch_also_updates_active_filter_window() -> None:
    branches: dict = {}
    filter_state = LibraryNotesFilterState(
        query="ren",
        placements=(_placement("n1", "Untitled", None),),
        ancestor_folders=(),
        total=1,
        start_offset=0,
        previous_offset=None,
        next_offset=None,
        generation=1,
        topology_epoch=1,
    )
    screen = _wiring_screen(branches, filter_state=filter_state)

    LibraryScreen._patch_library_note_list_from_session(screen)

    labels = _note_row_labels(
        build_filtered_library_notes_tree(
            screen._library_notes_tree_filter_state
        ).rows
    )
    assert labels == ["Renamed Note Title"]
