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
    LibraryNotesBranchRange,
    LibraryNotesFilterState,
    LibraryNotesTreeReceipt,
    apply_library_notes_filter_page,
    begin_library_notes_filter_load,
    build_filtered_library_notes_tree,
    fail_library_notes_filter_load,
    reconcile_library_notes_filter_commit,
)
from tldw_chatbook.Notes.note_folder_models import (
    FolderPlacementId,
    NoteFolder,
    NoteFolderMembership,
    NotePlacementPage,
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
    assert not hasattr(projection, "has_more")
    pager = projection.rows[2]
    assert pager.parent_folder_id == "root"
    assert pager.content_kind == "folders"
    assert pager.paging_action == "more"
    assert pager.focus_id == "library-notes-tree-pager-folder-726f6f74-folders-more"
    assert pager.disabled is False


@pytest.mark.parametrize("operation", ("rename_folder", "move_folder"))
def test_filter_commit_patches_folder_subtree_and_withdraws_exact_authority(
    operation: str,
) -> None:
    state = LibraryNotesFilterState.from_page(
        query="needle",
        page=NotePlacementPage(
            placements=(_placement("n1", "Note", "child", "m1"),),
            ancestor_folders=(
                _folder("renamed", None, "/Old"),
                _folder("child", "renamed", "/Old/Child"),
            ),
            total_placements=41,
            start_offset=40,
            previous_offset=20,
            next_offset=None,
        ),
        generation=4,
        topology_epoch=8,
    )

    patched = reconcile_library_notes_filter_commit(
        state,
        operation=operation,
        folder=replace(_folder("renamed", None, "/New"), version=9),
        affected_folder_ids=frozenset({"renamed", "child"}),
    )

    assert [
        (folder.folder_id, folder.path, folder.version)
        for folder in patched.ancestor_folders
    ] == [
        ("renamed", "/New", 9),
        ("child", "/New/Child", 1),
    ]
    assert patched.placements == state.placements
    assert patched.total is None
    assert patched.previous_offset is None
    assert patched.next_offset is None
    assert patched.stale is True
    assert patched.failed_direction == "target"
    assert patched.failed_offset == 40


def test_filter_commit_folder_delete_removes_only_exact_deleted_subtree() -> None:
    state = LibraryNotesFilterState.from_page(
        query="needle",
        page=NotePlacementPage(
            placements=(
                _placement("deleted-note", "Deleted", "child", "m-deleted"),
                _placement("safe-note", "Safe", "safe", "m-safe"),
            ),
            ancestor_folders=(
                _folder("deleted", None, "/Deleted"),
                _folder("child", "deleted", "/Deleted/Child"),
                _folder("safe", None, "/Safe"),
            ),
            total_placements=2,
            start_offset=0,
            previous_offset=None,
            next_offset=None,
        ),
        generation=1,
        topology_epoch=1,
    )

    patched = reconcile_library_notes_filter_commit(
        state,
        operation="delete_folder",
        removed_folder_ids=frozenset({"deleted", "child"}),
    )

    assert tuple(str(item.note["id"]) for item in patched.placements) == ("safe-note",)
    assert tuple(folder.folder_id for folder in patched.ancestor_folders) == ("safe",)
    assert patched.total is None
    assert patched.stale is True


@pytest.mark.parametrize(
    "operation", ("create_folder", "restore_folder", "note_create", "add_placement")
)
def test_filter_commit_never_injects_order_unknown_results(operation: str) -> None:
    state = LibraryNotesFilterState.from_page(
        query="needle",
        page=NotePlacementPage(
            placements=(_placement("existing", "Existing", None),),
            total_placements=1,
            start_offset=0,
            previous_offset=None,
            next_offset=None,
        ),
        generation=1,
        topology_epoch=1,
    )

    patched = reconcile_library_notes_filter_commit(state, operation=operation)

    assert patched.placements == state.placements
    assert patched.total is None
    assert patched.stale is True


@pytest.mark.parametrize(
    ("operation", "partial", "remaining_ids"),
    (
        ("move_placement", False, ("m-other",)),
        ("detach_placement", False, ("m-other",)),
        ("move_placement", True, ("m-source", "m-other")),
        ("note_delete", False, ()),
    ),
)
def test_filter_commit_removes_only_deterministically_invalid_placements(
    operation: str,
    partial: bool,
    remaining_ids: tuple[str, ...],
) -> None:
    placements = (
        _placement("n1", "Duplicate", "ideas", "m-source"),
        _placement("n1", "Duplicate", "ideas", "m-other"),
    )
    state = LibraryNotesFilterState.from_page(
        query="needle",
        page=NotePlacementPage(
            placements=placements,
            ancestor_folders=(_folder("ideas", None, "/Ideas"),),
            total_placements=2,
            start_offset=0,
            previous_offset=None,
            next_offset=None,
        ),
        generation=1,
        topology_epoch=1,
    )

    patched = reconcile_library_notes_filter_commit(
        state,
        operation=operation,
        note_id="n1",
        source_placement_id=FolderPlacementId.note("ideas", "n1", "m-source"),
        partial=partial,
    )

    assert (
        tuple(
            item.membership.membership_id
            for item in patched.placements
            if item.membership is not None
        )
        == remaining_ids
    )
    assert patched.total is None
    assert patched.stale is True


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


def test_semantic_receipt_contains_descriptors_but_no_page_records() -> None:
    receipt = LibraryNotesTreeReceipt(
        selected_placement_id="note:ideas:n1:m-preferred",
        selected_note_id="n1",
        expanded_folder_ids=("ideas",),
        branch_ranges=(
            LibraryNotesBranchRange(None, "folders", 20, 40),
            LibraryNotesBranchRange("ideas", "placements", 40, 60),
        ),
        filter_query="",
        filter_range=None,
        focus_semantic_id="note:ideas:n1:m-preferred",
        focus_role="note-placement",
        scroll_offset=(0, 17),
        rail_scroll_offset=(0, 3),
        lifecycle_generation=4,
        topology_epoch=9,
    )

    assert receipt.selected_placement_id.endswith("m-preferred")
    assert receipt.branch_ranges[1].start_offset == 40
    assert not {
        "folders",
        "notes",
        "memberships",
        "placements",
        "items",
        "page",
    }.intersection(receipt.__dataclass_fields__)


def test_filtered_projection_preserves_exact_duplicate_memberships_and_unfiled() -> (
    None
):
    work = _folder("work", None, "/Work")
    project = _folder("project", "work", "/Work/Project")
    placements = (
        NotePlacementRecord(
            note={"id": "n1", "title": "Duplicate"},
            folder_id="project",
            membership=_membership("m-preferred", "project", "n1"),
        ),
        NotePlacementRecord(
            note={"id": "n1", "title": "Duplicate"},
            folder_id="project",
            membership=_membership("m-other", "project", "n1"),
        ),
        NotePlacementRecord(
            note={"id": "loose", "title": "Loose"},
            folder_id=None,
            membership=None,
        ),
    )
    state = LibraryNotesFilterState.from_page(
        query="needle",
        page=NotePlacementPage(
            placements=placements,
            total_placements=3,
            start_offset=0,
            previous_offset=None,
            next_offset=None,
            ancestor_folders=(work, project),
        ),
        generation=2,
        topology_epoch=7,
    )

    projection = build_filtered_library_notes_tree(state)

    assert [row.placement_id for row in projection.rows if row.kind == "note"] == [
        FolderPlacementId.note("project", "n1", "m-preferred"),
        FolderPlacementId.note("project", "n1", "m-other"),
        FolderPlacementId.unfiled("loose"),
    ]
    assert next(row for row in projection.rows if row.kind == "unfiled").label == (
        "Unfiled"
    )


def _filter_page(
    start: int,
    note_ids: tuple[str, ...],
    *,
    total: int,
    previous: int | None,
    next_: int | None,
) -> NotePlacementPage:
    return NotePlacementPage(
        placements=tuple(_placement(note_id, note_id, None) for note_id in note_ids),
        total_placements=total,
        start_offset=start,
        previous_offset=previous,
        next_offset=next_,
    )


def _loaded_filter_window() -> LibraryNotesFilterState:
    return LibraryNotesFilterState.from_page(
        query="needle",
        page=_filter_page(
            20,
            tuple(f"n{index}" for index in range(20, 40)),
            total=60,
            previous=0,
            next_=40,
        ),
        generation=1,
        topology_epoch=7,
    )


def test_filter_reducer_appends_and_prepends_only_exact_adjacent_pages() -> None:
    base = _loaded_filter_window()
    appending = begin_library_notes_filter_load(
        base, generation=2, direction="more", offset=40, limit=20
    )
    appended = apply_library_notes_filter_page(
        appending,
        _filter_page(
            40,
            tuple(f"n{index}" for index in range(40, 60)),
            total=60,
            previous=20,
            next_=None,
        ),
        request_generation=2,
        topology_epoch=7,
    )

    assert appended.kind == "applied"
    assert appended.state.start_offset == 20
    assert tuple(item.note["id"] for item in appended.state.placements) == tuple(
        f"n{index}" for index in range(20, 60)
    )

    prepending = begin_library_notes_filter_load(
        base, generation=2, direction="previous", offset=0, limit=20
    )
    prepended = apply_library_notes_filter_page(
        prepending,
        _filter_page(
            0,
            tuple(f"n{index}" for index in range(20)),
            total=60,
            previous=None,
            next_=20,
        ),
        request_generation=2,
        topology_epoch=7,
    )

    assert prepended.kind == "applied"
    assert tuple(item.note["id"] for item in prepended.state.placements) == tuple(
        f"n{index}" for index in range(40)
    )


@pytest.mark.parametrize(
    "page",
    (
        _filter_page(
            40,
            ("n39", *(f"n{index}" for index in range(40, 59))),
            total=60,
            previous=20,
            next_=None,
        ),
        _filter_page(
            40,
            ("n40", "n40", *(f"n{index}" for index in range(42, 60))),
            total=60,
            previous=20,
            next_=None,
        ),
        _filter_page(
            41,
            tuple(f"n{index}" for index in range(40, 59)),
            total=60,
            previous=21,
            next_=None,
        ),
        _filter_page(
            40,
            tuple(f"n{index}" for index in range(40, 59)),
            total=60,
            previous=20,
            next_=59,
        ),
        _filter_page(
            40,
            tuple(f"n{index}" for index in range(40, 60)),
            total=61,
            previous=20,
            next_=60,
        ),
    ),
)
def test_filter_reducer_rejects_overlap_duplicate_gap_count_cursor_and_total_drift(
    page: NotePlacementPage,
) -> None:
    loading = begin_library_notes_filter_load(
        _loaded_filter_window(),
        generation=2,
        direction="more",
        offset=40,
        limit=20,
    )

    result = apply_library_notes_filter_page(
        loading, page, request_generation=2, topology_epoch=7
    )

    assert result.kind == "drift"
    assert result.recovery_offset == 40
    assert result.state.recovery_attempted
    assert len(result.state.placements) == 20


@pytest.mark.parametrize(
    "ancestor_folders",
    (
        (),
        (
            NoteFolder(
                folder_id="leaf",
                parent_id="missing",
                name="Leaf",
                path="/Missing/Leaf",
                normalized_path="/missing/leaf",
                version=1,
                deleted=False,
            ),
        ),
        (
            _folder("leaf", None, "/Leaf"),
            _folder("leaf", None, "/Leaf"),
        ),
        (
            NoteFolder(
                folder_id="leaf",
                parent_id="loop",
                name="Leaf",
                path="/Loop/Leaf",
                normalized_path="/loop/leaf",
                version=1,
                deleted=False,
            ),
            NoteFolder(
                folder_id="loop",
                parent_id="leaf",
                name="Loop",
                path="/Loop",
                normalized_path="/loop",
                version=1,
                deleted=False,
            ),
        ),
    ),
    ids=("missing-folder", "disconnected-chain", "duplicate-folder", "cycle"),
)
def test_filter_reducer_rejects_incomplete_or_ambiguous_ancestor_topology(
    ancestor_folders: tuple[NoteFolder, ...],
) -> None:
    page = NotePlacementPage(
        placements=(_placement("n1", "Filed", "leaf", "m1"),),
        ancestor_folders=ancestor_folders,
        total_placements=1,
        start_offset=0,
        previous_offset=None,
        next_offset=None,
    )
    loading = begin_library_notes_filter_load(
        LibraryNotesFilterState.empty(query="needle", generation=0, topology_epoch=7),
        generation=1,
        direction="replace",
        offset=0,
        limit=20,
    )

    result = apply_library_notes_filter_page(
        loading,
        page,
        request_generation=1,
        topology_epoch=7,
    )

    assert result.kind == "drift"
    assert result.reason == "invalid ancestor topology"
    assert result.recovery_offset == 0
    assert result.state.placements == ()


def test_filter_reducer_clamps_nonzero_recovery_after_total_shrink() -> None:
    loading = begin_library_notes_filter_load(
        LibraryNotesFilterState.from_page(
            query="needle",
            page=_filter_page(
                40,
                tuple(f"n{index}" for index in range(40, 60)),
                total=100,
                previous=20,
                next_=60,
            ),
            generation=1,
            topology_epoch=7,
        ),
        generation=2,
        direction="more",
        offset=60,
        limit=20,
    )

    result = apply_library_notes_filter_page(
        loading,
        _filter_page(60, (), total=53, previous=33, next_=None),
        request_generation=2,
        topology_epoch=7,
    )

    assert result.kind == "drift"
    assert result.recovery_offset == 40


def test_filter_reducer_second_drift_and_recovery_failure_are_local_stale() -> None:
    base = _loaded_filter_window()
    first = apply_library_notes_filter_page(
        begin_library_notes_filter_load(
            base, generation=2, direction="more", offset=40, limit=20
        ),
        _filter_page(
            40,
            tuple(f"n{index}" for index in range(40, 59)),
            total=60,
            previous=20,
            next_=59,
        ),
        request_generation=2,
        topology_epoch=7,
    )
    recovering = begin_library_notes_filter_load(
        first.state,
        generation=3,
        direction="target",
        offset=first.recovery_offset or 0,
        limit=20,
        recovering=True,
    )
    second = apply_library_notes_filter_page(
        recovering,
        _filter_page(
            40,
            tuple(f"n{index}" for index in range(40, 59)),
            total=60,
            previous=20,
            next_=59,
        ),
        request_generation=3,
        topology_epoch=7,
    )

    assert second.kind == "drift"
    assert second.state.stale
    assert second.state.total is None
    assert second.state.previous_offset is None
    assert second.state.next_offset is None

    failed = fail_library_notes_filter_load(
        recovering,
        request_generation=3,
        topology_epoch=7,
        error="offline",
    )
    assert failed.kind == "failed"
    assert failed.state.stale
    assert failed.state.failed_offset == 40
    assert failed.state.failed_direction == "target"


@pytest.mark.parametrize(("direction", "offset"), (("more", 40), ("previous", 0)))
def test_filter_reducer_failure_retains_exact_retry_request(
    direction: str, offset: int
) -> None:
    loading = begin_library_notes_filter_load(
        _loaded_filter_window(),
        generation=2,
        direction=direction,  # type: ignore[arg-type]
        offset=offset,
        limit=20,
    )

    result = fail_library_notes_filter_load(
        loading,
        request_generation=2,
        topology_epoch=7,
        error="offline",
    )

    assert result.kind == "failed"
    assert result.state.failed_direction == direction
    assert result.state.failed_offset == offset


@pytest.mark.parametrize(("request_generation", "topology_epoch"), ((1, 7), (2, 8)))
def test_filter_reducer_ignores_superseded_generation_or_topology(
    request_generation: int, topology_epoch: int
) -> None:
    loading = begin_library_notes_filter_load(
        _loaded_filter_window(),
        generation=2,
        direction="more",
        offset=40,
        limit=20,
    )

    result = apply_library_notes_filter_page(
        loading,
        _filter_page(
            40,
            tuple(f"n{index}" for index in range(40, 60)),
            total=60,
            previous=20,
            next_=None,
        ),
        request_generation=request_generation,
        topology_epoch=topology_epoch,
    )

    assert result.kind == "ignored"
    assert result.state is loading
