"""Pure behavior tests for the Database Notes folder-tree projection."""

from __future__ import annotations

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
