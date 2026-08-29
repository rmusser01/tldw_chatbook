"""Pure behavior tests for the Database Notes folder-tree projection."""

from __future__ import annotations

from dataclasses import replace

import pytest

import tldw_chatbook.Library.library_notes_tree_state as tree_state
from tldw_chatbook.Library.library_notes_tree_paging import (
    NotesBranchKey,
    NotesBranchSliceState,
    empty_notes_slice,
)

from tldw_chatbook.Library.library_notes_tree_state import (
    UNFILED_PLACEMENT_ID,
    LibraryNotesTreeIdentity,
    _effective_memberships,
    build_library_notes_tree,
    merge_note_folder_pages,
    reconcile_library_notes_tree_identity,
)
from tldw_chatbook.Notes.note_folder_models import (
    FolderPlacementId,
    NoteFolder,
    NoteFolderMembership,
    NoteFolderPage,
    NotePlacementRecord,
)


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


def _membership(
    membership_id: str,
    folder_id: str,
    note_id: str,
    *,
    ownership: str = "manual",
    owner_id: str = "",
    owner_active: bool = True,
) -> NoteFolderMembership:
    return NoteFolderMembership(
        membership_id=membership_id,
        folder_id=folder_id,
        note_id=note_id,
        ownership=ownership,
        owner_id=owner_id,
        owner_active=owner_active,
        version=1,
    )


def _page(
    *,
    folders=(),
    memberships=(),
    notes=(),
    next_folder_offset=None,
    next_note_offset=None,
    next_membership_offset=None,
    unfiled_note_ids=None,
) -> NoteFolderPage:
    return NoteFolderPage(
        folders=tuple(folders),
        memberships=tuple(memberships),
        notes=tuple(notes),
        total_folders=len(folders) + (1 if next_folder_offset is not None else 0),
        total_notes=len(notes) + (1 if next_note_offset is not None else 0),
        next_offset=next_note_offset,
        next_folder_offset=next_folder_offset,
        total_memberships=len(memberships)
        + (1 if next_membership_offset is not None else 0),
        next_membership_offset=next_membership_offset,
        unfiled_note_ids=unfiled_note_ids,
    )


def _branch(
    parent_id: str | None,
    content_kind: str,
    *,
    items=(),
    total: int | None = None,
    start: int = 0,
    previous: int | None = None,
    next_: int | None = None,
    freshness: str = "fresh",
    loading: bool = False,
    requested_direction: str | None = None,
    recovery_attempted: bool = False,
    failed_direction: str | None = None,
    error: str = "",
) -> NotesBranchSliceState:
    key = NotesBranchKey(parent_id, content_kind)  # type: ignore[arg-type]
    item_ids = tuple(
        FolderPlacementId.folder(item.folder_id)
        if content_kind == "folders"
        else (
            FolderPlacementId.unfiled(str(item.note["id"]))
            if item.folder_id is None
            else FolderPlacementId.note(
                item.folder_id,
                str(item.note["id"]),
                item.membership.membership_id,
            )
        )
        for item in items
    )
    return replace(
        empty_notes_slice(key),
        items=tuple(items),
        item_ids=item_ids,
        total=total,
        start_offset=start,
        previous_offset=previous,
        next_offset=next_,
        freshness=freshness,
        loading=loading,
        requested_direction=requested_direction,  # type: ignore[arg-type]
        recovery_attempted=recovery_attempted,
        failed_direction=failed_direction,  # type: ignore[arg-type]
        error=error,
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


def test_projects_nested_folders_and_unfiled_notes_with_breadcrumbs():
    personal = _folder("personal", None, "/Personal")
    ideas = _folder("ideas", "personal", "/Personal/Ideas")
    root = _page(
        folders=(personal,),
        notes=({"id": "loose", "title": "Loose thought"},),
    )
    expanded = _page(
        folders=(ideas,),
        memberships=(_membership("m-garden", "ideas", "garden"),),
        notes=({"id": "garden", "title": "Garden redesign"},),
    )

    projection = build_library_notes_tree(
        root_page=root,
        expanded_page=expanded,
        expanded_folder_ids={"personal", "ideas"},
    )

    assert [(row.kind, row.label, row.depth) for row in projection.rows] == [
        ("folder", "Personal", 0),
        ("folder", "Ideas", 1),
        ("note", "Garden redesign", 2),
        ("unfiled", "Unfiled", 0),
        ("note", "Loose thought", 1),
    ]
    garden = projection.row(FolderPlacementId.note("ideas", "garden", "m-garden"))
    loose = projection.row(FolderPlacementId.unfiled("loose"))
    assert (
        garden is not None and garden.breadcrumb == "Personal / Ideas / Garden redesign"
    )
    assert loose is not None and loose.breadcrumb == "Unfiled / Loose thought"


def test_same_note_has_distinct_placement_rows_but_one_note_identity():
    folders = (
        _folder("ideas", None, "/Ideas"),
        _folder("reading", None, "/Reading"),
    )
    expanded = _page(
        memberships=(
            _membership("m1", "ideas", "n1"),
            _membership("m2", "reading", "n1"),
        ),
        notes=({"id": "n1", "title": "Shared note"},),
    )

    projection = build_library_notes_tree(
        root_page=_page(folders=folders),
        expanded_page=expanded,
        expanded_folder_ids={"ideas", "reading"},
    )
    rows = [row for row in projection.rows if row.note_id == "n1"]

    assert [row.placement_id for row in rows] == [
        FolderPlacementId.note("ideas", "n1", "m1"),
        FolderPlacementId.note("reading", "n1", "m2"),
    ]
    assert {row.note_id for row in rows} == {"n1"}
    assert [row.breadcrumb for row in rows] == [
        "Ideas / Shared note",
        "Reading / Shared note",
    ]


def test_manual_and_managed_memberships_in_same_folder_keep_distinct_identity():
    folder = _folder("ideas", None, "/Ideas")
    projection = build_library_notes_tree(
        root_page=_page(folders=(folder,)),
        expanded_page=_page(
            memberships=(
                _membership("manual", "ideas", "n1"),
                _membership(
                    "managed",
                    "ideas",
                    "n1",
                    ownership="managed",
                    owner_id="sync-root",
                ),
            ),
            notes=({"id": "n1", "title": "Shared note"},),
        ),
        expanded_folder_ids={"ideas"},
    )

    rows = [row for row in projection.rows if row.note_id == "n1"]
    assert len(rows) == 2
    assert len({row.placement_id for row in rows}) == 2
    assert {row.membership_id for row in rows} == {"manual", "managed"}
    assert {row.protected for row in rows} == {False, True}


def test_generated_managed_ancestor_collapses_but_manual_duplicate_remains():
    parent = _folder("parent", None, "/Work")
    child = _folder("child", "parent", "/Work/Project")
    memberships = (
        _membership(
            "generated-parent",
            "parent",
            "n1",
            ownership="managed",
            owner_id="root-a",
        ),
        _membership(
            "generated-child",
            "child",
            "n1",
            ownership="managed",
            owner_id="root-a",
        ),
        _membership("explicit-parent", "parent", "n1"),
    )
    projection = build_library_notes_tree(
        root_page=_page(folders=(parent,)),
        expanded_page=_page(
            folders=(child,),
            memberships=memberships,
            notes=({"id": "n1", "title": "Plan"},),
        ),
        expanded_folder_ids={"parent", "child"},
    )

    rows = [row for row in projection.rows if row.note_id == "n1"]
    assert {row.membership_id for row in rows} == {
        "explicit-parent",
        "generated-child",
    }


def test_managed_ancestor_collapse_walks_each_folder_chain_once():
    class CountingFolders(dict[str, NoteFolder]):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.get_calls = 0

        def get(self, key, default=None):
            self.get_calls += 1
            return super().get(key, default)

    folder_count = 40
    folders = CountingFolders(
        {
            f"folder-{index}": _folder(
                f"folder-{index}",
                f"folder-{index - 1}" if index else None,
                "/" + "/".join(f"Folder {part}" for part in range(index + 1)),
            )
            for index in range(folder_count)
        }
    )
    memberships = tuple(
        _membership(
            f"membership-{index}",
            f"folder-{index}",
            "note-1",
            ownership="managed",
            owner_id="sync-root",
        )
        for index in reversed(range(folder_count))
    )

    effective = _effective_memberships(memberships, folders)

    assert [membership.folder_id for membership in effective] == ["folder-39"]
    assert folders.get_calls <= folder_count * 2


def test_managed_and_restored_without_owner_are_textually_distinct_and_protected():
    folder = _folder("work", None, "/Work")
    projection = build_library_notes_tree(
        root_page=_page(folders=(folder,)),
        expanded_page=_page(
            memberships=(
                _membership(
                    "active", "work", "n1", ownership="managed", owner_id="root"
                ),
                _membership(
                    "inactive",
                    "work",
                    "n2",
                    ownership="managed",
                    owner_id="missing-root",
                    owner_active=False,
                ),
            ),
            notes=(
                {"id": "n1", "title": "Weekly"},
                {"id": "n2", "title": "Recovered"},
            ),
        ),
        expanded_folder_ids={"work"},
    )

    active = next(row for row in projection.rows if row.note_id == "n1")
    inactive = next(row for row in projection.rows if row.note_id == "n2")
    assert active.status_text == "⇄ Synced placement"
    assert inactive.status_text == "! Needs owner review"
    assert active.protected and inactive.protected
    assert active.semantic_status == "connected"
    assert inactive.semantic_status == "needs_attention"

    protected_folder = projection.row(FolderPlacementId.folder("work"))
    assert protected_folder is not None and protected_folder.protected
    assert protected_folder.semantic_status == "needs_attention"
    assert protected_folder.status_text == "! Needs owner review"


def test_managed_folder_protection_propagates_to_loaded_ancestors():
    parent = _folder("work", None, "/Work")
    child = _folder("project", "work", "/Work/Project")
    projection = build_library_notes_tree(
        root_page=_page(folders=(parent,)),
        expanded_page=_page(
            folders=(child,),
            memberships=(
                _membership(
                    "managed",
                    "project",
                    "n1",
                    ownership="managed",
                    owner_id="root-a",
                ),
            ),
            notes=({"id": "n1", "title": "Plan"},),
        ),
        expanded_folder_ids={"work", "project"},
    )

    work = projection.row(FolderPlacementId.folder("work"))
    project = projection.row(FolderPlacementId.folder("project"))
    assert work is not None and work.protected
    assert project is not None and project.protected
    assert work.status_text == "⇄ Sync managed"
    assert project.semantic_status == "connected"


def test_authoritative_managed_summary_protects_collapsed_folder():
    folder = _folder("work", None, "/Work")
    root = NoteFolderPage(
        folders=(folder,),
        memberships=(),
        notes=(),
        total_folders=1,
        total_notes=0,
        next_offset=None,
        managed_folder_ids=("work",),
    )

    projection = build_library_notes_tree(
        root_page=root,
        expanded_page=_page(),
        expanded_folder_ids=set(),
    )

    row = projection.row(FolderPlacementId.folder("work"))
    assert row is not None and row.protected
    assert row.status_text == "⇄ Sync managed"


def test_projection_exposes_bounded_more_rows_for_each_cursor():
    projection = build_library_notes_tree(
        root_page=_page(
            notes=({"id": "n1", "title": "One"},),
            next_folder_offset=500,
            next_note_offset=1000,
        ),
        expanded_page=_page(next_membership_offset=1000),
        expanded_folder_ids=set(),
    )
    assert projection.next_folder_offset == 500
    assert projection.next_note_offset == 1000
    assert projection.next_membership_offset == 1000
    assert projection.has_more


def test_identity_reconciliation_preserves_placement_then_note_then_visible_row():
    folder = _folder("ideas", None, "/Ideas")
    first = build_library_notes_tree(
        root_page=_page(folders=(folder,), notes=({"id": "loose", "title": "Loose"},)),
        expanded_page=_page(
            memberships=(_membership("m", "ideas", "n1"),),
            notes=({"id": "n1", "title": "Note"},),
        ),
        expanded_folder_ids={"ideas"},
    )
    identity = LibraryNotesTreeIdentity(
        placement_id=FolderPlacementId.note("ideas", "n1", "m"), note_id="n1"
    )
    assert reconcile_library_notes_tree_identity(first, identity) == identity

    moved = build_library_notes_tree(
        root_page=_page(notes=({"id": "n1", "title": "Note"},)),
        expanded_page=_page(),
        expanded_folder_ids=set(),
    )
    assert reconcile_library_notes_tree_identity(moved, identity) == (
        LibraryNotesTreeIdentity(
            placement_id=FolderPlacementId.unfiled("n1"), note_id="n1"
        )
    )

    empty = build_library_notes_tree(
        root_page=_page(), expanded_page=_page(), expanded_folder_ids=set()
    )
    assert reconcile_library_notes_tree_identity(empty, identity) is None


def test_unfiled_identity_is_stable_constant():
    projection = build_library_notes_tree(
        root_page=_page(notes=({"id": "n1", "title": "One"},)),
        expanded_page=_page(),
        expanded_folder_ids=set(),
    )
    assert projection.rows[0].placement_id == UNFILED_PLACEMENT_ID


def test_filter_shows_every_matching_placement_with_its_breadcrumb():
    folders = (
        _folder("ideas", None, "/Ideas"),
        _folder("reading", None, "/Reading"),
    )
    projection = build_library_notes_tree(
        root_page=_page(folders=folders),
        expanded_page=_page(
            memberships=(
                _membership("m1", "ideas", "n1"),
                _membership("m2", "reading", "n1"),
                _membership("m3", "reading", "n2"),
            ),
            notes=(
                {"id": "n1", "title": "Garden plan"},
                {"id": "n2", "title": "Unrelated"},
            ),
        ),
        expanded_folder_ids={"ideas", "reading"},
        filter_text="garden",
    )
    notes = [row for row in projection.rows if row.kind == "note"]
    assert [row.breadcrumb for row in notes] == [
        "Ideas / Garden plan",
        "Reading / Garden plan",
    ]
    assert all(row.label != "Unrelated" for row in projection.rows)


def test_filter_temporarily_reveals_matches_under_collapsed_folders():
    parent = _folder("work", None, "/Work")
    child = _folder("project", "work", "/Work/Project")
    search_page = _page(
        folders=(parent, child),
        memberships=(_membership("m1", "project", "n1"),),
        notes=({"id": "n1", "title": "Hidden garden plan"},),
        unfiled_note_ids=(),
    )

    projection = build_library_notes_tree(
        root_page=search_page,
        expanded_page=search_page,
        expanded_folder_ids=set(),
        filter_text="garden",
    )

    assert [row.breadcrumb for row in projection.rows if row.kind == "note"] == [
        "Work / Project / Hidden garden plan"
    ]
    assert [row.label for row in projection.rows if row.kind == "folder"] == [
        "Work",
        "Project",
    ]
    assert all(row.expanded for row in projection.rows if row.kind == "folder")


def test_search_result_identity_keeps_content_only_matches_visible():
    folder = _folder("work", None, "/Work")
    search_page = _page(
        folders=(folder,),
        memberships=(_membership("m1", "work", "n1"),),
        notes=({"id": "n1", "title": "Weekly plan"},),
        unfiled_note_ids=(),
    )

    projection = build_library_notes_tree(
        root_page=search_page,
        expanded_page=search_page,
        expanded_folder_ids=set(),
        filter_text="garden",
        matched_note_ids=frozenset({"n1"}),
    )

    assert [row.breadcrumb for row in projection.rows if row.kind == "note"] == [
        "Work / Weekly plan"
    ]


def test_search_keeps_inactive_managed_and_unfiled_breadcrumbs_distinct():
    folder = _folder("work", None, "/Work")
    page = NoteFolderPage(
        folders=(folder,),
        memberships=(
            _membership(
                "managed",
                "work",
                "n1",
                ownership="managed",
                owner_id="missing",
                owner_active=False,
            ),
        ),
        notes=({"id": "n1", "title": "Recovered"},),
        total_folders=1,
        total_notes=1,
        next_offset=None,
        unfiled_note_ids=("n1",),
    )

    projection = build_library_notes_tree(
        root_page=page,
        expanded_page=page,
        expanded_folder_ids=set(),
        filter_text="body-only match",
        matched_note_ids=frozenset({"n1"}),
    )

    assert [row.breadcrumb for row in projection.rows if row.kind == "note"] == [
        "Work / Recovered",
        "Unfiled / Recovered",
    ]


def test_bounded_pages_merge_by_domain_identity_without_duplicates():
    first = _page(
        folders=(_folder("a", None, "/A"),),
        notes=({"id": "n1", "title": "One"},),
        next_note_offset=1,
    )
    second = _page(
        folders=(_folder("a", None, "/A"), _folder("b", None, "/B")),
        notes=(
            {"id": "n1", "title": "One"},
            {"id": "n2", "title": "Two"},
        ),
    )
    merged = merge_note_folder_pages(first, second)
    assert [folder.folder_id for folder in merged.folders] == ["a", "b"]
    assert [note["id"] for note in merged.notes] == ["n1", "n2"]
    assert merged.next_offset is None


def test_paged_projection_places_each_parent_keyed_boundary_inline() -> None:
    root = _folder("root", None, "/Root")
    child = _folder("child", "root", "/Root/Child")
    grandchild = _folder("grandchild", "child", "/Root/Child/Grandchild")
    branches = {
        NotesBranchKey(None, "folders"): _branch(
            None, "folders", items=(root,), total=2, next_=1
        ),
        NotesBranchKey("root", "folders"): _branch(
            "root", "folders", items=(child,), total=2, next_=1
        ),
        NotesBranchKey("root", "placements"): _branch(
            "root",
            "placements",
            items=(_placement("nested", "Nested note", "root", "m-nested"),),
            total=2,
            next_=1,
        ),
        NotesBranchKey(None, "placements"): _branch(
            None,
            "placements",
            items=(_placement("loose", "Loose note", None),),
            total=2,
            next_=1,
        ),
        # This loaded branch must not recurse because ``child`` is not expanded.
        NotesBranchKey("child", "folders"): _branch(
            "child", "folders", items=(grandchild,), total=1
        ),
    }

    projection = tree_state.build_paged_library_notes_tree(
        branch_states=branches,
        expanded_folder_ids={"root"},
    )

    assert [(row.kind, row.placement_id) for row in projection.rows] == [
        ("folder", FolderPlacementId.folder("root")),
        ("folder", FolderPlacementId.folder("child")),
        ("pager", "pager:notes-tree:folder:root:folders:more"),
        (
            "note",
            FolderPlacementId.note("root", "nested", "m-nested"),
        ),
        ("pager", "pager:notes-tree:folder:root:placements:more"),
        ("pager", "pager:notes-tree:root:folders:more"),
        ("unfiled", UNFILED_PLACEMENT_ID),
        ("note", FolderPlacementId.unfiled("loose")),
        ("pager", "pager:notes-tree:root:placements:more"),
    ]
    assert all(row.label != "Grandchild" for row in projection.rows)
    pager = projection.rows[2]
    assert pager.parent_folder_id == "root"
    assert pager.content_kind == "folders"
    assert pager.paging_action == "more"
    assert pager.focus_id == "library-notes-tree-pager-folder-726f6f74-folders-more"
    assert pager.disabled is False


def test_paged_projection_uses_truthful_exact_middle_loading_and_exhausted_copy() -> (
    None
):
    notes = tuple(_placement(f"n{index}", f"Note {index}", None) for index in range(20))
    middle = _branch(
        None,
        "placements",
        items=notes,
        total=400,
        start=200,
        previous=180,
        next_=220,
    )
    projection = tree_state.build_paged_library_notes_tree(
        branch_states={NotesBranchKey(None, "placements"): middle},
        expanded_folder_ids=set(),
    )
    pagers = [row for row in projection.rows if row.kind == "pager"]

    assert [row.label for row in pagers] == [
        "Notes 201–220 of 400  Load earlier",
        "Notes 201–220 of 400  Load more notes",
    ]
    assert [row.paging_action for row in pagers] == ["earlier", "more"]
    assert all(row.range_copy == "Notes 201–220 of 400" for row in pagers)

    loading = replace(middle, loading=True, requested_direction="more")
    loading_projection = tree_state.build_paged_library_notes_tree(
        branch_states={NotesBranchKey(None, "placements"): loading},
        expanded_folder_ids=set(),
    )
    loading_pagers = [row for row in loading_projection.rows if row.kind == "pager"]
    assert [row.label for row in loading_pagers] == [
        "Notes 201–220 of 400  Load earlier",
        "Notes 201–220 of 400  Loading…",
    ]
    assert [row.disabled for row in loading_pagers] == [False, True]
    assert len([row for row in loading_projection.rows if row.kind == "note"]) == 20

    folders = tuple(
        _folder(f"folder-{index}", None, f"/Folder {index}") for index in range(20)
    )
    first = _branch(None, "folders", items=folders, total=83, next_=20)
    first_projection = tree_state.build_paged_library_notes_tree(
        branch_states={NotesBranchKey(None, "folders"): first},
        expanded_folder_ids=set(),
    )
    assert [row.label for row in first_projection.rows if row.kind == "pager"] == [
        "Folders 1–20 of 83  Load more folders"
    ]

    exhausted = replace(first, total=20, next_offset=None)
    exhausted_projection = tree_state.build_paged_library_notes_tree(
        branch_states={NotesBranchKey(None, "folders"): exhausted},
        expanded_folder_ids=set(),
    )
    assert all(row.kind != "pager" for row in exhausted_projection.rows)


def test_paged_projection_localizes_error_recovery_stale_and_mutation_safety() -> None:
    affected = _placement("affected", "Affected", "stale", "m-affected")
    safe = _placement("safe", "Safe", "safe", "m-safe")
    stale = _branch(
        "stale",
        "placements",
        items=(affected,),
        total=None,
        freshness="stale",
        failed_direction="replace",
        error="Recovery request failed.",
    )
    failed = _branch(
        "safe",
        "placements",
        items=(safe,),
        total=2,
        next_=1,
        failed_direction="more",
        error="Page request failed.",
    )
    recovering = replace(
        failed,
        loading=True,
        recovery_attempted=True,
        requested_direction="replace",
        error="",
    )
    folders = (
        _folder("safe", None, "/Safe"),
        _folder("stale", None, "/Stale"),
    )
    root = _branch(None, "folders", items=folders, total=2)

    projection = tree_state.build_paged_library_notes_tree(
        branch_states={
            NotesBranchKey(None, "folders"): root,
            NotesBranchKey("stale", "placements"): stale,
            NotesBranchKey("safe", "placements"): failed,
        },
        expanded_folder_ids={"safe", "stale"},
    )
    stale_row = projection.row(
        FolderPlacementId.note("stale", "affected", "m-affected")
    )
    safe_row = projection.row(FolderPlacementId.note("safe", "safe", "m-safe"))
    stale_pager = next(
        row
        for row in projection.rows
        if row.kind == "pager" and row.parent_folder_id == "stale"
    )
    failed_pager = next(
        row
        for row in projection.rows
        if row.kind == "pager" and row.parent_folder_id == "safe"
    )

    assert stale_row is not None and stale_row.unsafe_mutation_disabled is True
    assert safe_row is not None and safe_row.unsafe_mutation_disabled is False
    assert stale_pager.label == "1 placement loaded · May be out of date · Retry"
    assert stale_pager.range_copy == ""
    assert stale_pager.paging_action == "retry"
    assert stale_pager.retry_direction == "replace"
    assert stale_pager.focus_id.endswith("replace")
    assert stale_pager.disabled is False
    assert failed_pager.label == "Couldn’t load more · Retry"
    assert failed_pager.paging_action == "retry"

    recovery_projection = tree_state.build_paged_library_notes_tree(
        branch_states={
            NotesBranchKey(None, "folders"): root,
            NotesBranchKey("safe", "placements"): recovering,
        },
        expanded_folder_ids={"safe"},
    )
    recovery_pager = next(
        row
        for row in recovery_projection.rows
        if row.kind == "pager" and row.parent_folder_id == "safe"
    )
    assert recovery_pager.label == "Tree changed · Refreshing…"
    assert recovery_pager.disabled is True
    assert recovery_pager.loading is True


def test_paged_projection_uses_authoritative_folder_protection_without_placements() -> (
    None
):
    collapsed = _folder("collapsed", None, "/Collapsed managed")
    inactive = _folder("inactive", None, "/Inactive managed")
    normal = _folder("normal", None, "/Normal")
    root = _branch(None, "folders", items=(collapsed, inactive, normal), total=3)

    projection = tree_state.build_paged_library_notes_tree(
        branch_states={NotesBranchKey(None, "folders"): root},
        expanded_folder_ids={"inactive"},
        protected_folder_ids=frozenset({"collapsed", "inactive"}),
        inactive_managed_folder_ids=frozenset({"inactive"}),
    )

    collapsed_row = projection.row(FolderPlacementId.folder("collapsed"))
    inactive_row = projection.row(FolderPlacementId.folder("inactive"))
    normal_row = projection.row(FolderPlacementId.folder("normal"))
    assert collapsed_row is not None
    assert collapsed_row.expanded is False
    assert collapsed_row.protected is True
    assert collapsed_row.owner_active is True
    assert collapsed_row.semantic_status == "connected"
    assert inactive_row is not None
    assert inactive_row.expanded is True
    assert inactive_row.protected is True
    assert inactive_row.owner_active is False
    assert inactive_row.semantic_status == "needs_attention"
    assert normal_row is not None
    assert normal_row.protected is False
    assert normal_row.semantic_status == "normal"


def test_paged_projection_renders_every_authoritative_parent_and_child_placement() -> (
    None
):
    parent = _folder("parent", None, "/Parent")
    child = _folder("child", "parent", "/Parent/Child")
    parent_record = NotePlacementRecord(
        note={"id": "shared", "title": "Shared"},
        folder_id="parent",
        membership=_membership(
            "managed-parent",
            "parent",
            "shared",
            ownership="managed",
            owner_id="sync-root",
        ),
    )
    child_record = NotePlacementRecord(
        note={"id": "shared", "title": "Shared"},
        folder_id="child",
        membership=_membership(
            "managed-child",
            "child",
            "shared",
            ownership="managed",
            owner_id="sync-root",
        ),
    )
    branches = {
        NotesBranchKey(None, "folders"): _branch(
            None, "folders", items=(parent,), total=1
        ),
        NotesBranchKey("parent", "folders"): _branch(
            "parent", "folders", items=(child,), total=1
        ),
        NotesBranchKey("parent", "placements"): _branch(
            "parent", "placements", items=(parent_record,), total=1
        ),
        NotesBranchKey("child", "placements"): _branch(
            "child", "placements", items=(child_record,), total=1
        ),
    }

    projection = tree_state.build_paged_library_notes_tree(
        branch_states=branches,
        expanded_folder_ids={"parent", "child"},
    )
    note_rows = [row for row in projection.rows if row.kind == "note"]

    assert [row.placement_id for row in note_rows] == [
        FolderPlacementId.note("child", "shared", "managed-child"),
        FolderPlacementId.note("parent", "shared", "managed-parent"),
    ]
    assert len(note_rows) == len(
        branches[NotesBranchKey("parent", "placements")].items
        + branches[NotesBranchKey("child", "placements")].items
    )


def test_stale_retry_in_flight_preserves_rows_but_disables_local_control() -> None:
    record = _placement("affected", "Affected", None)
    stale_loading = _branch(
        None,
        "placements",
        items=(record,),
        total=None,
        freshness="stale",
        loading=True,
        requested_direction="replace",
    )

    projection = tree_state.build_paged_library_notes_tree(
        branch_states={NotesBranchKey(None, "placements"): stale_loading},
        expanded_folder_ids=set(),
    )
    pager = next(row for row in projection.rows if row.kind == "pager")

    assert len([row for row in projection.rows if row.kind == "note"]) == 1
    assert pager.label == "1 placement loaded · May be out of date · Loading…"
    assert pager.range_copy == ""
    assert pager.action_copy == "Loading…"
    assert pager.loading is True
    assert pager.disabled is True


def test_failed_earlier_and_more_project_retry_at_the_exact_boundary() -> None:
    record = _placement("middle", "Middle", None)
    middle = _branch(
        None,
        "placements",
        items=(record,),
        total=3,
        start=1,
        previous=0,
        next_=2,
        error="Page request failed.",
    )

    earlier_projection = tree_state.build_paged_library_notes_tree(
        branch_states={
            NotesBranchKey(None, "placements"): replace(
                middle, failed_direction="previous"
            )
        },
        expanded_folder_ids=set(),
    )
    more_projection = tree_state.build_paged_library_notes_tree(
        branch_states={
            NotesBranchKey(None, "placements"): replace(middle, failed_direction="more")
        },
        expanded_folder_ids=set(),
    )
    earlier_pagers = [row for row in earlier_projection.rows if row.kind == "pager"]
    more_pagers = [row for row in more_projection.rows if row.kind == "pager"]

    assert [row.label for row in earlier_pagers] == [
        "Couldn’t load earlier · Retry",
        "Notes 2–2 of 3  Load more notes",
    ]
    assert earlier_pagers[0].retry_direction == "previous"
    assert earlier_pagers[0].focus_id.endswith("earlier")
    assert [row.label for row in more_pagers] == [
        "Notes 2–2 of 3  Load earlier",
        "Couldn’t load more · Retry",
    ]
    assert more_pagers[1].retry_direction == "more"
    assert more_pagers[1].focus_id.endswith("more")


@pytest.mark.parametrize("content_kind", ("folders", "placements"))
def test_initial_concrete_parent_failure_uses_contents_copy(content_kind: str) -> None:
    parent = _folder("parent", None, "/Parent")
    failed = _branch(
        "parent",
        content_kind,
        failed_direction="replace",
        error="Initial request failed.",
    )

    projection = tree_state.build_paged_library_notes_tree(
        branch_states={
            NotesBranchKey(None, "folders"): _branch(
                None, "folders", items=(parent,), total=1
            ),
            NotesBranchKey("parent", content_kind): failed,
        },
        expanded_folder_ids={"parent"},
    )
    pager = next(
        row
        for row in projection.rows
        if row.kind == "pager"
        and row.parent_folder_id == "parent"
        and row.content_kind == content_kind
    )

    assert pager.label == "Couldn’t load contents · Retry"
    assert pager.retry_direction == "replace"
    assert pager.focus_id.endswith("replace")


def test_both_initial_concrete_parent_failures_keep_local_contents_copy() -> None:
    parent = _folder("parent", None, "/Parent")
    projection = tree_state.build_paged_library_notes_tree(
        branch_states={
            NotesBranchKey(None, "folders"): _branch(
                None, "folders", items=(parent,), total=1
            ),
            NotesBranchKey("parent", "folders"): _branch(
                "parent",
                "folders",
                failed_direction="replace",
                error="Folder request failed.",
            ),
            NotesBranchKey("parent", "placements"): _branch(
                "parent",
                "placements",
                failed_direction="replace",
                error="Placement request failed.",
            ),
        },
        expanded_folder_ids={"parent"},
    )

    failures = [
        row
        for row in projection.rows
        if row.kind == "pager" and row.parent_folder_id == "parent"
    ]
    assert [row.content_kind for row in failures] == ["folders", "placements"]
    assert {row.label for row in failures} == {"Couldn’t load contents · Retry"}


@pytest.mark.parametrize(
    ("content_kind", "expected"),
    (
        ("folders", "Couldn’t load folders · Retry"),
        ("placements", "Couldn’t load notes · Retry"),
    ),
)
def test_initial_root_failure_keeps_root_specific_copy(
    content_kind: str, expected: str
) -> None:
    failed = _branch(
        None,
        content_kind,
        failed_direction="replace",
        error="Initial request failed.",
    )

    projection = tree_state.build_paged_library_notes_tree(
        branch_states={NotesBranchKey(None, content_kind): failed},
        expanded_folder_ids=set(),
    )

    assert next(row for row in projection.rows if row.kind == "pager").label == expected


@pytest.mark.parametrize(
    ("failed_direction", "expected"),
    (
        ("previous", "Couldn’t load earlier · Retry"),
        ("more", "Couldn’t load more · Retry"),
    ),
)
def test_concrete_parent_continuation_failure_keeps_boundary_copy(
    failed_direction: str, expected: str
) -> None:
    parent = _folder("parent", None, "/Parent")
    placement = _placement("middle", "Middle", "parent", "m-middle")
    failed = _branch(
        "parent",
        "placements",
        items=(placement,),
        total=3,
        start=1,
        previous=0,
        next_=2,
        failed_direction=failed_direction,
        error="Continuation failed.",
    )

    projection = tree_state.build_paged_library_notes_tree(
        branch_states={
            NotesBranchKey(None, "folders"): _branch(
                None, "folders", items=(parent,), total=1
            ),
            NotesBranchKey("parent", "placements"): failed,
        },
        expanded_folder_ids={"parent"},
    )

    retry = next(
        row
        for row in projection.rows
        if row.kind == "pager" and row.paging_action == "retry"
    )
    assert retry.label == expected


def test_pager_identity_is_stable_across_boundary_state_transitions() -> None:
    record = _placement("middle", "Middle", None)
    middle = _branch(
        None,
        "placements",
        items=(record,),
        total=3,
        start=1,
        previous=0,
        next_=2,
    )

    def project(state: NotesBranchSliceState):
        return tree_state.build_paged_library_notes_tree(
            branch_states={NotesBranchKey(None, "placements"): state},
            expanded_folder_ids=set(),
        )

    def boundary_row(projection, direction: str):
        return next(
            row
            for row in projection.rows
            if row.kind == "pager"
            and (
                row.paging_action == direction
                or row.retry_direction
                == ("previous" if direction == "earlier" else direction)
            )
        )

    for direction in ("earlier", "more"):
        load_direction = "previous" if direction == "earlier" else "more"
        idle = boundary_row(project(middle), direction)
        loading = boundary_row(
            project(
                replace(
                    middle,
                    loading=True,
                    requested_direction=load_direction,
                )
            ),
            direction,
        )
        failed = boundary_row(
            project(
                replace(
                    middle,
                    failed_direction=load_direction,
                    error="Page request failed.",
                )
            ),
            direction,
        )
        retry_loading = boundary_row(
            project(
                replace(
                    middle,
                    loading=True,
                    requested_direction=load_direction,
                )
            ),
            direction,
        )

        identities = {
            (row.placement_id, row.focus_id)
            for row in (idle, loading, failed, retry_loading)
        }
        assert len(identities) == 1
        assert idle.focus_id.endswith(direction)

    stale_idle = replace(middle, total=None, freshness="stale")
    stale_loading = replace(
        stale_idle,
        loading=True,
        requested_direction="replace",
    )
    stale_failed = replace(
        stale_idle,
        failed_direction="replace",
        error="Replacement request failed.",
    )
    replacement_rows = [
        next(row for row in project(state).rows if row.kind == "pager")
        for state in (stale_idle, stale_loading, stale_failed, stale_loading)
    ]
    assert len({(row.placement_id, row.focus_id) for row in replacement_rows}) == 1
    assert replacement_rows[0].focus_id.endswith("replace")

    initial_loading = replace(
        empty_notes_slice(NotesBranchKey(None, "placements")),
        loading=True,
        requested_direction="replace",
    )
    initial_failed = replace(
        empty_notes_slice(NotesBranchKey(None, "placements")),
        failed_direction="replace",
        error="Initial request failed.",
    )
    replace_rows = [
        next(row for row in project(state).rows if row.kind == "pager")
        for state in (initial_loading, initial_failed, initial_loading)
    ]
    assert len({(row.placement_id, row.focus_id) for row in replace_rows}) == 1
    assert replace_rows[0].focus_id.endswith("replace")
