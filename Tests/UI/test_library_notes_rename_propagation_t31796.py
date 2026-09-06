"""Regression: a note rename propagates correctly to the Database Notes list.

task-31796: the Database Notes list renders placement rows from the cached
tree branch slices (and, while filtering, from an FTS filter window) -- NOT
from the flat source records the save-time patch used to touch. So after
renaming a note and returning to the list, the row kept the pre-rename title
until a filter re-query.

Qodo review of PR #2464 found two follow-on correctness bugs in the first fix:
  #3 repository pages are ordered by title, so an in-place retitle without
     re-sorting left the current page out of order;
  #4 the notes filter is an FTS MATCH over title+body+keywords (not title
     alone), so an in-place, title-only edit of the filter window could keep a
     now-nonmatching note visible.

The fix now (a) retitles AND re-sorts the affected branch slice by title, and
(b) on a genuine rename that touches the active filter window, clears the
now-stale filter (mirroring the create/delete note-mutation path) instead of
editing the FTS window in place.
"""

from __future__ import annotations

from dataclasses import replace
from types import MethodType, SimpleNamespace

from tldw_chatbook.Library.library_notes_tree_paging import (
    NotesBranchKey,
    NotesBranchSliceState,
    empty_notes_slice,
    patch_notes_tree_branches_title,
    placement_title_sort_key,
)
from tldw_chatbook.Library.library_notes_tree_state import (
    LibraryNotesFilterState,
    build_paged_library_notes_tree,
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


def _project(branches) -> list[str]:
    return _note_row_labels(
        build_paged_library_notes_tree(
            branch_states=branches, expanded_folder_ids=set()
        ).rows
    )


# --- pure helper: branch-slice retitle + re-sort ---------------------------


def test_patch_branches_retitles_unfiled_placement_and_projects_new_label() -> None:
    branches = {
        NotesBranchKey(None, "placements"): _placements_branch(
            None, _placement("n1", "Untitled", None)
        ),
    }

    patched, changed = patch_notes_tree_branches_title(
        branches,
        note_id="n1",
        title="Renamed Note Title",
        modified_at="2026-09-06T00:00:00Z",
    )

    assert changed is True
    assert _project(patched) == ["Renamed Note Title"]
    note = patched[NotesBranchKey(None, "placements")].items[0].note
    assert note["title"] == "Renamed Note Title"
    assert note["last_modified"] == "2026-09-06T00:00:00Z"


def test_rename_that_moves_sort_position_reorders_the_branch_slice() -> None:
    """Qodo #3: a rename changing collation position must re-sort the page."""
    branches = {
        NotesBranchKey(None, "placements"): _placements_branch(
            None,
            _placement("n1", "Apple", None),
            _placement("n2", "Mango", None),
            _placement("n3", "Orange", None),
        ),
    }
    # Before: Apple, Mango, Orange (sorted). Rename "Apple" -> "Watermelon":
    patched, changed = patch_notes_tree_branches_title(
        branches, note_id="n1", title="Watermelon"
    )

    assert changed is True
    # Re-sorted so the renamed note lands in its new collation position, and
    # item_ids stay parallel (the projection derives placement ids from them).
    assert _project(patched) == ["Mango", "Orange", "Watermelon"]
    slice_state = patched[NotesBranchKey(None, "placements")]
    assert [str(item.note["id"]) for item in slice_state.items] == ["n2", "n3", "n1"]
    assert len(slice_state.items) == len(slice_state.item_ids)


def test_rename_reorder_is_case_insensitive_like_repository() -> None:
    branches = {
        NotesBranchKey(None, "placements"): _placements_branch(
            None,
            _placement("n1", "Zebra", None),
            _placement("n2", "apple", None),
        ),
    }
    patched, _ = patch_notes_tree_branches_title(branches, note_id="n1", title="Mango")
    # NOCASE ordering -> ["apple", "Mango"]. This differs from BOTH the no-sort
    # order (["Mango", "apple"]) AND a case-sensitive byte sort (which puts all
    # uppercase before lowercase: "Mango" < "apple" -> ["Mango", "apple"]), so
    # it pins both the re-sort and its case-insensitivity.
    assert _project(patched) == ["apple", "Mango"]


def test_patch_branches_retitles_placement_nested_in_folder_slice() -> None:
    branches = {
        NotesBranchKey(None, "folders"): _folders_branch(
            None, _folder("f1", None, "/Work")
        ),
        NotesBranchKey("f1", "placements"): _placements_branch(
            "f1", _placement("n9", "Untitled", "f1", "m-n9")
        ),
    }

    patched, changed = patch_notes_tree_branches_title(
        branches, note_id="n9", title="Filed Rename"
    )

    assert changed is True
    labels = _note_row_labels(
        build_paged_library_notes_tree(
            branch_states=patched, expanded_folder_ids={"f1"}
        ).rows
    )
    assert "Filed Rename" in labels


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

    patched, changed = patch_notes_tree_branches_title(
        branches, note_id="n1", title="Now Named"
    )

    assert changed is True
    assert patched[NotesBranchKey(None, "folders")] is folders
    assert set(_project(patched)) == {"Now Named", "Keep Me"}


def test_patch_branches_no_match_returns_slice_identity_and_flag_false() -> None:
    placements = _placements_branch(None, _placement("n1", "Untitled", None))
    branches = {NotesBranchKey(None, "placements"): placements}

    patched, changed = patch_notes_tree_branches_title(
        branches, note_id="absent", title="X"
    )

    assert changed is False
    assert patched[NotesBranchKey(None, "placements")] is placements


def test_placement_title_sort_key_matches_repository_tiebreakers() -> None:
    a = _placement("a", "Same", "f1", "m2")
    b = _placement("a", "Same", "f1", "m1")
    # Equal title + note id -> membership_id breaks the tie.
    assert placement_title_sort_key(b) < placement_title_sort_key(a)


# --- screen wiring: _patch_library_note_list_from_session ------------------


def _wiring_screen(
    *,
    branches: dict,
    filter_state: LibraryNotesFilterState | None = None,
    filter_query: str = "",
    persisted_title: str = "Renamed Note Title",
) -> SimpleNamespace:
    baseline = SimpleNamespace(
        note_id="n1",
        title=persisted_title,
        modified_at="2026-09-06T00:00:00Z",
    )
    snapshot = SimpleNamespace(baseline=baseline)
    screen = SimpleNamespace(
        _library_note_session=SimpleNamespace(snapshot=snapshot),
        _local_source_records={"notes": ({"id": "n1", "title": "Untitled"},)},
        _library_notes_filter_records=None,
        _library_notes_tree_branches=branches,
        _library_notes_tree_filter_state=filter_state,
        _library_notes_filter=filter_query,
        _library_notes_filter_generation=0,
    )
    # Bind the sibling methods the patch routine calls on ``self``.
    screen._cached_library_note_list_title = MethodType(
        LibraryScreen._cached_library_note_list_title, screen
    )
    screen._active_notes_filter_shows_note = MethodType(
        LibraryScreen._active_notes_filter_shows_note, screen
    )
    screen._placement_note_id = LibraryScreen._placement_note_id
    return screen


def test_save_patch_updates_tree_branch_row_without_requery() -> None:
    """The original 31796 case must still pass: rename shows on the list row."""
    branches = {
        NotesBranchKey(None, "placements"): _placements_branch(
            None, _placement("n1", "Untitled", None)
        ),
    }
    screen = _wiring_screen(branches=branches)

    LibraryScreen._patch_library_note_list_from_session(screen)

    assert _project(screen._library_notes_tree_branches) == ["Renamed Note Title"]
    assert screen._local_source_records["notes"][0]["title"] == "Renamed Note Title"


def test_save_patch_clears_active_filter_when_shown_note_is_renamed() -> None:
    """Qodo #4: a genuine rename clears the stale FTS filter window."""
    branches: dict = {}
    filter_state = LibraryNotesFilterState(
        query="app",
        placements=(_placement("n1", "Apple", None),),
        ancestor_folders=(),
        total=1,
        start_offset=0,
        previous_offset=None,
        next_offset=None,
        generation=1,
        topology_epoch=1,
    )
    screen = _wiring_screen(
        branches=branches, filter_state=filter_state, filter_query="app"
    )

    LibraryScreen._patch_library_note_list_from_session(screen)

    # The now-stale filter is dropped so no ghost / mis-titled row survives.
    assert screen._library_notes_tree_filter_state is None
    assert screen._library_notes_filter == ""
    assert screen._library_notes_filter_records is None
    assert screen._library_notes_filter_generation == 1


def test_body_only_save_does_not_clear_active_filter() -> None:
    """A save that does not change the title must leave the filter intact."""
    filter_state = LibraryNotesFilterState(
        query="ren",
        placements=(_placement("n1", "Renamed Note Title", None),),
        ancestor_folders=(),
        total=1,
        start_offset=0,
        previous_offset=None,
        next_offset=None,
        generation=1,
        topology_epoch=1,
    )
    screen = _wiring_screen(
        branches={},
        filter_state=filter_state,
        filter_query="ren",
        persisted_title="Renamed Note Title",  # unchanged vs. cached title
    )

    LibraryScreen._patch_library_note_list_from_session(screen)

    assert screen._library_notes_tree_filter_state is filter_state
    assert screen._library_notes_filter == "ren"
