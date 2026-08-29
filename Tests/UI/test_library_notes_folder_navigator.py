"""Library-screen orchestration tests for the Database Notes folder tree."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from html import unescape
import inspect
from types import SimpleNamespace

import pytest

from Tests.UI.test_destination_shells import StaticLibraryNotesScopeService
from Tests.UI.test_library_shell import (
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _two_notes,
    _wait_for_library_shell,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_NOTES
from tldw_chatbook.Library.library_notes_tree_paging import (
    NotesBranchKey,
    apply_notes_slice_page,
    begin_notes_slice_load,
    empty_notes_slice,
)
from tldw_chatbook.Library.library_notes_tree_state import (
    LibraryNotesBranchRange,
    LibraryNotesFilterRange,
    LibraryNotesTreeReceipt,
)
from tldw_chatbook.Notes.note_folder_models import (
    FolderPlacementId,
    FolderCapabilityError,
    FolderCollisionError,
    FolderConflictError,
    FolderValidationError,
    NoteFolder,
    NoteFolderChildPage,
    NoteFolderMembership,
    NoteFolderManagedStatus,
    NoteFolderPage,
    NotePlacementPage,
    NotePlacementRecord,
    NoteTreeLocation,
    NoteTreePathStep,
)
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_canvas_sync import PostRecomposeCallback
from tldw_chatbook.Widgets.Library.library_notes_canvas import LibraryNotesCanvas


def _page(
    *,
    folders=(),
    notes=(),
    memberships=(),
    next_folder_offset=None,
    next_note_offset=None,
    next_membership_offset=None,
    unfiled_note_ids=None,
) -> NoteFolderPage:
    return NoteFolderPage(
        folders=tuple(folders),
        memberships=tuple(memberships),
        notes=tuple(notes),
        total_folders=len(folders),
        total_notes=len(notes),
        next_offset=next_note_offset,
        next_folder_offset=next_folder_offset,
        total_memberships=len(memberships),
        next_membership_offset=next_membership_offset,
        unfiled_note_ids=unfiled_note_ids,
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
    membership_id: str, folder_id: str, note_id: str
) -> NoteFolderMembership:
    return NoteFolderMembership(
        membership_id=membership_id,
        folder_id=folder_id,
        note_id=note_id,
        ownership="manual",
        owner_id="",
        owner_active=True,
        version=1,
    )


class _FolderService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def load_note_folder_tree_batch(self, **kwargs):
        self.calls.append(kwargs)
        expanded = tuple(kwargs["expanded_folder_ids"])
        if not expanded:
            return _page(
                folders=(_folder("personal", None, "/Personal"),),
                notes=({"id": "loose", "title": "Loose"},),
            )
        return _page(folders=(_folder("ideas", "personal", "/Personal/Ideas"),))


def _folder_page(
    parent_id: str | None,
    *folder_ids: str,
    start: int = 0,
    total: int | None = None,
    previous: int | None = None,
    next_: int | None = None,
    statuses: tuple[NoteFolderManagedStatus, ...] = (),
) -> NoteFolderChildPage:
    folders = tuple(
        _folder(folder_id, parent_id, f"/{folder_id}") for folder_id in folder_ids
    )
    return NoteFolderChildPage(
        folders=folders,
        total_folders=len(folders) if total is None else total,
        start_offset=start,
        previous_offset=previous,
        next_offset=next_,
        folder_statuses=statuses,
    )


def _placement_record(
    note_id: str,
    parent_id: str | None,
    *,
    ownership: str = "manual",
    owner_active: bool = True,
) -> NotePlacementRecord:
    membership = (
        NoteFolderMembership(
            membership_id=f"m-{note_id}",
            folder_id=parent_id,
            note_id=note_id,
            ownership=ownership,  # type: ignore[arg-type]
            owner_id="sync" if ownership == "managed" else "",
            owner_active=owner_active,
            version=1,
        )
        if parent_id is not None
        else None
    )
    return NotePlacementRecord(
        note={"id": note_id, "title": note_id},
        folder_id=parent_id,
        membership=membership,
    )


def _placement_page(
    parent_id: str | None,
    *note_ids: str,
    start: int = 0,
    total: int | None = None,
    previous: int | None = None,
    next_: int | None = None,
    statuses: tuple[NoteFolderManagedStatus, ...] = (),
) -> NotePlacementPage:
    placements = tuple(_placement_record(note_id, parent_id) for note_id in note_ids)
    return NotePlacementPage(
        placements=placements,
        total_placements=len(placements) if total is None else total,
        start_offset=start,
        previous_offset=previous,
        next_offset=next_,
        folder_statuses=statuses,
    )


class _BranchService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, int, int]] = []
        self.fail: set[tuple[str, str | None]] = set()
        self.folder_pages: dict[str | None, NoteFolderChildPage] = {
            None: _folder_page(None, "personal")
        }
        self.placement_pages: dict[str | None, NotePlacementPage] = {
            None: _placement_page(None, "loose")
        }

    async def page_note_folder_children(self, **kwargs):
        parent_id = kwargs["parent_id"]
        self.calls.append(("folders", parent_id, kwargs["offset"], kwargs["limit"]))
        if ("folders", parent_id) in self.fail:
            raise RuntimeError("folders offline")
        return self.folder_pages.get(parent_id, _folder_page(parent_id))

    async def page_note_placements(self, **kwargs):
        parent_id = kwargs["parent_id"]
        self.calls.append(("placements", parent_id, kwargs["offset"], kwargs["limit"]))
        if ("placements", parent_id) in self.fail:
            raise RuntimeError("placements offline")
        return self.placement_pages.get(parent_id, _placement_page(parent_id))


def _branch_screen_fake(service: _BranchService):
    fake = SimpleNamespace(
        app_instance=SimpleNamespace(
            notes_scope_service=service,
            notes_user_id="tester",
        ),
        _library_notes_tree_branches={},
        _library_notes_tree_expanded_ids=set(),
        _library_notes_tree_topology_epoch=1,
        _library_notes_tree_lifecycle_generation=1,
        _library_notes_tree_request_generations={},
        _library_notes_tree_target_offsets={},
        _library_notes_tree_status_by_slice={},
        _library_notes_tree_status_revision=0,
        _library_notes_tree_protected_folder_ids=frozenset(),
        _library_notes_tree_inactive_managed_folder_ids=frozenset(),
        _library_notes_tree_selected_placement_id="",
        _library_notes_tree_filter_state=None,
        _library_notes_filter_generation=0,
        _library_notes_filter="",
        _library_notes_filter_records=None,
        _library_notes_filter_navigation_generation=None,
        _library_notes_tree_navigation_requests={},
        _library_notes_navigation_generation=0,
        _library_notes_navigation_status="",
        _library_notes_pending_focus_identity=None,
        _library_notes_pending_focus_waits_for_snapshot=False,
        _library_notes_pending_focus_generation=None,
        _library_notes_user_id=lambda: "tester",
        _repaints=0,
        _focus_calls=[],
        _library_notes_focus_intent_generation=0,
        _capture_library_notes_focus_identity=lambda: SimpleNamespace(),
        is_mounted=True,
    )

    def _sync(*_args, **kwargs):
        fake._repaints += 1
        callback = kwargs.get("then")
        if callback is not None:
            callback()

    fake._sync_library_notes_tree_canvas_if_present = _sync
    fake._focus_library_notes_tree_after_page = lambda *args, **kwargs: (
        fake._focus_calls.append((args, kwargs))
    )
    return fake


@pytest.mark.asyncio
async def test_initial_branch_load_requests_independent_root_slices_and_isolates_failure():
    service = _BranchService()
    service.fail.add(("folders", None))
    fake = _branch_screen_fake(service)

    LibraryScreen._begin_library_notes_tree_visit(fake)
    await LibraryScreen._load_library_notes_tree_slice(
        fake, NotesBranchKey(None, "folders"), direction="replace", offset=0
    )
    await LibraryScreen._load_library_notes_tree_slice(
        fake, NotesBranchKey(None, "placements"), direction="replace", offset=0
    )

    assert service.calls == [
        ("folders", None, 0, 20),
        ("placements", None, 0, 20),
    ]
    folders = fake._library_notes_tree_branches[NotesBranchKey(None, "folders")]
    placements = fake._library_notes_tree_branches[NotesBranchKey(None, "placements")]
    assert folders.error and not folders.loading
    assert placements.item_ids == ("unfiled:loose",)
    assert placements.freshness == "fresh"


@pytest.mark.asyncio
async def test_expansion_loads_only_one_parent_and_collapse_retains_fresh_branch():
    service = _BranchService()
    service.folder_pages["personal"] = _folder_page("personal", "ideas")
    service.placement_pages["personal"] = _placement_page("personal", "n1")
    fake = _branch_screen_fake(service)

    await LibraryScreen._ensure_library_notes_tree_folder_loaded(fake, "personal")
    first_calls = tuple(service.calls)
    await LibraryScreen._ensure_library_notes_tree_folder_loaded(fake, "personal")

    assert first_calls == (
        ("folders", "personal", 0, 20),
        ("placements", "personal", 0, 20),
    )
    assert tuple(service.calls) == first_calls
    assert set(fake._library_notes_tree_branches) == {
        NotesBranchKey("personal", "folders"),
        NotesBranchKey("personal", "placements"),
    }


def test_branch_projection_receives_authoritative_folder_protection_metadata():
    service = _BranchService()
    fake = _branch_screen_fake(service)
    key = NotesBranchKey(None, "folders")
    fake._library_notes_tree_branches[key] = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        _folder_page(None, "personal"),
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state
    fake._library_notes_tree_protected_folder_ids = frozenset({"personal"})
    fake._library_notes_tree_inactive_managed_folder_ids = frozenset({"personal"})

    projection = LibraryScreen._build_library_notes_tree_projection(fake)

    folder = projection.rows[0]
    assert folder.protected
    assert not folder.owner_active
    assert folder.status_text == "! Needs owner review"


@pytest.mark.asyncio
async def test_authoritative_status_replacement_prunes_and_normal_clears() -> None:
    service = _BranchService()
    fake = _branch_screen_fake(service)
    key = NotesBranchKey(None, "folders")
    first_ids = ("personal", *(f"normal-{index}" for index in range(19)))
    service.folder_pages[None] = _folder_page(
        None,
        *first_ids,
        total=21,
        next_=20,
        statuses=tuple(
            NoteFolderManagedStatus(
                folder_id, "protected" if folder_id == "personal" else "normal"
            )
            for folder_id in first_ids
        ),
    )
    await LibraryScreen._load_library_notes_tree_slice(
        fake, key, direction="replace", offset=0
    )
    assert fake._library_notes_tree_protected_folder_ids == {"personal"}
    service.folder_pages[None] = _folder_page(
        None,
        "gone",
        start=20,
        total=21,
        previous=0,
        statuses=(NoteFolderManagedStatus("gone", "inactive_managed"),),
    )
    await LibraryScreen._load_library_notes_tree_slice(
        fake, key, direction="more", offset=20
    )
    assert fake._library_notes_tree_protected_folder_ids == {"personal", "gone"}
    assert fake._library_notes_tree_inactive_managed_folder_ids == {"gone"}

    service.folder_pages[None] = _folder_page(
        None,
        "personal",
        statuses=(NoteFolderManagedStatus("personal", "normal"),),
    )
    await LibraryScreen._load_library_notes_tree_slice(
        fake, key, direction="replace", offset=0
    )

    assert fake._library_notes_tree_protected_folder_ids == frozenset()
    assert fake._library_notes_tree_inactive_managed_folder_ids == frozenset()
    assert set(fake._library_notes_tree_status_by_slice[key]) == {"personal"}


@pytest.mark.asyncio
async def test_real_service_pages_drive_inactive_and_out_of_window_screen_status(
    tmp_path,
) -> None:
    db = CharactersRAGDB(tmp_path / "screen-status.db", client_id="screen-status")
    repository = LocalNoteFolderRepository(db)
    folder = repository.create_folder(name="Folder", parent_id=None)
    for index in range(20):
        note_id = db.add_note(f"A {index:02d}", "")
        assert note_id is not None
        repository.attach_manual(folder_id=folder.folder_id, note_id=note_id)
    managed_note = db.add_note("Z managed", "")
    inactive_note = db.add_note("ZZ inactive", "")
    assert managed_note is not None and inactive_note is not None
    repository.reconcile_managed(
        owner_id="active-owner", desired=((folder.folder_id, managed_note),)
    )
    repository.reconcile_managed(
        owner_id="inactive-owner", desired=((folder.folder_id, inactive_note),)
    )
    repository.mark_unknown_owners_inactive(active_owner_ids=("active-owner",))
    service = NotesScopeService(None, None, folder_repository=repository)
    fake = _branch_screen_fake(service)  # type: ignore[arg-type]

    await LibraryScreen._load_library_notes_tree_slice(
        fake,
        NotesBranchKey(folder.folder_id, "placements"),
        direction="replace",
        offset=0,
    )

    state = fake._library_notes_tree_branches[
        NotesBranchKey(folder.folder_id, "placements")
    ]
    assert managed_note not in {str(item.note["id"]) for item in state.items}
    assert fake._library_notes_tree_protected_folder_ids == {folder.folder_id}
    assert fake._library_notes_tree_inactive_managed_folder_ids == {folder.folder_id}
    db.close_connection()


@pytest.mark.asyncio
async def test_branch_more_targets_semantic_slice_and_failure_keeps_stable_retry():
    service = _BranchService()
    fake = _branch_screen_fake(service)
    root_key = NotesBranchKey(None, "folders")
    fake._library_notes_tree_branches[root_key] = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(root_key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        _folder_page(None, "personal"),
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state
    fake._library_notes_tree_expanded_ids.add("personal")
    key = NotesBranchKey("personal", "placements")
    first = _placement_page(
        "personal", *(f"n{i}" for i in range(20)), total=21, next_=20
    )
    fake._library_notes_tree_branches[key] = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        first,
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state
    service.fail.add(("placements", "personal"))

    await LibraryScreen._load_library_notes_tree_slice(
        fake,
        key,
        direction="more",
        offset=20,
        pager_focus_id="library-notes-tree-pager-folder-706572736f6e616c-placements-more",
    )

    assert service.calls == [("placements", "personal", 20, 20)]
    state = fake._library_notes_tree_branches[key]
    assert len(state.items) == 20
    assert state.failed_direction == "more"
    projection = LibraryScreen._build_library_notes_tree_projection(fake)
    retry = next(row for row in projection.rows if row.kind == "pager")
    assert retry.paging_action == "retry"
    assert retry.retry_direction == "more"
    assert retry.focus_id.endswith("-placements-more")


@pytest.mark.asyncio
async def test_branch_newer_generation_topology_and_lifecycle_fence_late_results():
    service = _BranchService()
    fake = _branch_screen_fake(service)
    key = NotesBranchKey(None, "folders")
    state = empty_notes_slice(key, topology_epoch=1)
    fake._library_notes_tree_branches[key] = begin_notes_slice_load(
        state,
        generation=2,
        direction="replace",
        requested_offset=0,
        requested_limit=20,
    )

    await LibraryScreen._apply_library_notes_tree_slice_page(
        fake,
        key,
        _folder_page(None, "old"),
        direction="replace",
        request_generation=1,
        topology_epoch=1,
        lifecycle_generation=1,
        pager_focus_id=None,
        prior_item_ids=(),
    )
    fake._library_notes_tree_topology_epoch = 2
    await LibraryScreen._apply_library_notes_tree_slice_page(
        fake,
        key,
        _folder_page(None, "topology-old"),
        direction="replace",
        request_generation=2,
        topology_epoch=1,
        lifecycle_generation=1,
        pager_focus_id=None,
        prior_item_ids=(),
    )
    fake._library_notes_tree_lifecycle_generation = 2
    await LibraryScreen._apply_library_notes_tree_slice_page(
        fake,
        key,
        _folder_page(None, "unmounted"),
        direction="replace",
        request_generation=2,
        topology_epoch=2,
        lifecycle_generation=1,
        pager_focus_id=None,
        prior_item_ids=(),
    )

    assert fake._library_notes_tree_branches[key].items == ()
    assert fake._repaints == 0
    assert fake._focus_calls == []


@pytest.mark.asyncio
async def test_branch_drift_runs_one_offset_zero_recovery_and_stales_if_it_fails():
    class _DriftingService(_BranchService):
        async def page_note_placements(self, **kwargs):
            parent_id = kwargs["parent_id"]
            offset = kwargs["offset"]
            self.calls.append(("placements", parent_id, offset, kwargs["limit"]))
            if offset == 20:
                return _placement_page(
                    parent_id, "changed", start=20, total=22, previous=0, next_=21
                )
            raise RuntimeError("recovery offline")

    service = _DriftingService()
    fake = _branch_screen_fake(service)
    key = NotesBranchKey("personal", "placements")
    first = _placement_page(
        "personal", *(f"n{i}" for i in range(20)), total=21, next_=20
    )
    fake._library_notes_tree_branches[key] = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        first,
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state

    await LibraryScreen._load_library_notes_tree_slice(
        fake, key, direction="more", offset=20
    )

    assert service.calls == [
        ("placements", "personal", 20, 20),
        ("placements", "personal", 0, 20),
    ]
    state = fake._library_notes_tree_branches[key]
    assert state.freshness == "stale"
    assert state.total is None
    assert len(state.items) == 20


@pytest.mark.asyncio
async def test_target_drift_recovers_the_same_nonzero_range_once() -> None:
    class _TargetDriftService(_BranchService):
        async def page_note_placements(self, **kwargs):
            parent_id = kwargs["parent_id"]
            offset = kwargs["offset"]
            self.calls.append(("placements", parent_id, offset, kwargs["limit"]))
            if len(self.calls) == 1:
                return _placement_page(
                    parent_id,
                    "wrong",
                    start=0,
                    total=41,
                    next_=1,
                )
            return _placement_page(
                parent_id,
                "target-40",
                start=40,
                total=41,
                previous=20,
            )

    service = _TargetDriftService()
    fake = _branch_screen_fake(service)
    key = NotesBranchKey("personal", "placements")

    await LibraryScreen._load_library_notes_tree_slice(
        fake, key, direction="target", offset=40
    )

    assert service.calls == [
        ("placements", "personal", 40, 20),
        ("placements", "personal", 40, 20),
    ]
    state = fake._library_notes_tree_branches[key]
    assert state.freshness == "fresh"
    assert state.start_offset == 40
    assert [item.note["id"] for item in state.items] == ["target-40"]


@pytest.mark.asyncio
@pytest.mark.parametrize("recovery_fails", (False, True))
async def test_second_target_drift_or_recovery_failure_stales_only_target_slice(
    recovery_fails: bool,
) -> None:
    class _BrokenTargetRecovery(_BranchService):
        async def page_note_placements(self, **kwargs):
            parent_id = kwargs["parent_id"]
            offset = kwargs["offset"]
            self.calls.append(("placements", parent_id, offset, kwargs["limit"]))
            if len(self.calls) == 2 and recovery_fails:
                raise RuntimeError("target recovery offline")
            return _placement_page(
                parent_id,
                "wrong",
                start=0,
                total=41,
                next_=1,
            )

    service = _BrokenTargetRecovery()
    fake = _branch_screen_fake(service)
    key = NotesBranchKey("personal", "placements")
    sibling = NotesBranchKey("sibling", "folders")
    fake._library_notes_tree_branches[sibling] = replace(
        empty_notes_slice(sibling, topology_epoch=1), freshness="fresh", total=0
    )

    await LibraryScreen._load_library_notes_tree_slice(
        fake, key, direction="target", offset=40
    )

    assert service.calls == [
        ("placements", "personal", 40, 20),
        ("placements", "personal", 40, 20),
    ]
    state = fake._library_notes_tree_branches[key]
    assert state.freshness == "stale"
    assert state.total is None
    assert fake._library_notes_tree_branches[sibling].freshness == "fresh"


def test_tree_pager_has_a_stable_notes_focus_role() -> None:
    pager = SimpleNamespace(
        id="library-notes-tree-pager-root-folders-more",
        has_class=lambda name: name == "library-notes-tree-pager",
    )
    fake = SimpleNamespace(
        _library_landing_focus_control_id=lambda _focused: "",
        _file_notes_active=lambda: False,
    )

    assert LibraryScreen._library_notes_semantic_role(fake, pager) == (
        "tree-pager:library-notes-tree-pager-root-folders-more"
    )


@pytest.mark.asyncio
async def test_branch_completion_focuses_first_added_row_only_while_pager_owns_focus():
    service = _BranchService()
    fake = _branch_screen_fake(service)
    key = NotesBranchKey("personal", "placements")
    first = _placement_page(
        "personal", *(f"n{i}" for i in range(20)), total=21, next_=20
    )
    current = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        first,
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state
    fake._library_notes_tree_branches[key] = begin_notes_slice_load(
        current,
        generation=2,
        direction="more",
        requested_offset=20,
        requested_limit=20,
    )
    fake._library_notes_tree_request_generations[key] = 2
    pager_id = "library-notes-tree-pager-folder-706572736f6e616c-placements-more"
    fake.focused = SimpleNamespace(id=pager_id)
    added = SimpleNamespace(placement_id="note:personal:n20:m-n20", focused=False)
    added.focus = lambda: setattr(added, "focused", True)
    fake.query = lambda selector: (added,)
    del fake._focus_library_notes_tree_after_page

    await LibraryScreen._apply_library_notes_tree_slice_page(
        fake,
        key,
        _placement_page("personal", "n20", start=20, total=21, previous=0),
        direction="more",
        request_generation=2,
        topology_epoch=1,
        lifecycle_generation=1,
        pager_focus_id=pager_id,
        prior_item_ids=current.item_ids,
    )

    assert added.focused


def test_branch_pager_handler_routes_only_semantic_button_metadata():
    service = _BranchService()
    fake = _branch_screen_fake(service)
    key = NotesBranchKey("personal", "folders")
    fake._library_notes_tree_branches[key] = replace(
        empty_notes_slice(key, topology_epoch=1),
        freshness="fresh",
        total=40,
        next_offset=20,
    )
    requested = []
    fake._request_library_notes_tree_slice = lambda *args, **kwargs: requested.append(
        (args, kwargs)
    )
    button = SimpleNamespace(
        parent_folder_id="personal",
        content_kind="folders",
        paging_action="more",
        retry_direction=None,
        id="semantic-pager",
    )
    event = SimpleNamespace(button=button, stop=lambda: None)

    LibraryScreen.handle_library_notes_tree_pager(fake, event)

    assert requested == [
        (
            (key,),
            {
                "direction": "more",
                "offset": 20,
                "pager_focus_id": "semantic-pager",
            },
        )
    ]


def test_unmount_invalidates_notes_authority_before_first_await():
    source = inspect.getsource(LibraryScreen.on_unmount)
    first_await = source.index("await ")

    assert source.index("_invalidate_library_notes_tree_for_unmount") < first_await


def _screen_fake(service: _FolderService):
    return SimpleNamespace(
        app_instance=SimpleNamespace(
            notes_scope_service=service,
            notes_user_id="tester",
        ),
        _library_notes_tree_branches={},
        _library_notes_tree_expanded_ids=set(),
        _library_notes_tree_topology_epoch=1,
        _library_notes_tree_lifecycle_generation=1,
        _library_notes_tree_request_generations={},
        _library_notes_tree_protected_folder_ids=frozenset(),
        _library_notes_tree_inactive_managed_folder_ids=frozenset(),
        _library_notes_user_id=lambda: "tester",
        is_mounted=False,
    )


@pytest.mark.asyncio
async def test_filter_uses_exact_placement_page_without_mutating_browse_branches(
    monkeypatch,
) -> None:
    class _ExactFilterService:
        def __init__(self) -> None:
            self.calls = []

        async def search_note_tree_placements(self, **kwargs):
            self.calls.append(kwargs)
            folder = _folder("ideas", None, "/Ideas")
            return NotePlacementPage(
                placements=(_placement_record("n1", "ideas"),),
                total_placements=1,
                start_offset=0,
                previous_offset=None,
                next_offset=None,
                ancestor_folders=(folder,),
            )

    service = _ExactFilterService()
    fake = _branch_screen_fake(service)  # type: ignore[arg-type]
    browse = fake._library_notes_tree_branches
    fake._library_notes_filter = "private query"
    fake._library_notes_filter_generation = 0
    fake._library_notes_tree_filter_state = None
    fake._focus_library_notes_filter_input = lambda: None
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.library_screen._sync_library_canvas",
        lambda *args, **kwargs: None,
    )

    await LibraryScreen._run_library_notes_filter(fake, "private query")

    assert service.calls == [
        {
            "scope": "local_note",
            "query": "private query",
            "limit": 20,
            "offset": 0,
            "user_id": "tester",
        }
    ]
    assert fake._library_notes_tree_branches is browse
    assert fake._library_notes_tree_filter_state.placements[
        0
    ].membership.membership_id == ("m-n1")


@pytest.mark.asyncio
async def test_deep_link_locator_loads_root_to_target_exact_ranges() -> None:
    class _LocatorService(_BranchService):
        async def locate_note_tree_placement(self, **kwargs):
            self.calls.append(("locator", kwargs["note_id"], 0, kwargs["page_size"]))
            return NoteTreeLocation(
                placement_id=FolderPlacementId.note("target", "n1", "m-preferred"),
                note_id="n1",
                membership_id="m-preferred",
                path=(
                    NoteTreePathStep("root-40", None, 40),
                    NoteTreePathStep("target", "root-40", 60),
                ),
                placement_offset=80,
            )

        async def page_note_folder_children(self, **kwargs):
            parent = kwargs["parent_id"]
            offset = kwargs["offset"]
            self.calls.append(("folders", parent, offset, kwargs["limit"]))
            folder_id = "root-40" if parent is None else "target"
            return _folder_page(
                parent,
                folder_id,
                start=offset,
                total=offset + 1,
                previous=max(0, offset - 20) if offset else None,
            )

        async def page_note_placements(self, **kwargs):
            parent = kwargs["parent_id"]
            offset = kwargs["offset"]
            self.calls.append(("placements", parent, offset, kwargs["limit"]))
            membership = _membership("m-preferred", "target", "n1")
            return NotePlacementPage(
                placements=(
                    NotePlacementRecord(
                        note={"id": "n1", "title": "Target"},
                        folder_id="target",
                        membership=membership,
                    ),
                ),
                total_placements=offset + 1,
                start_offset=offset,
                previous_offset=max(0, offset - 20) if offset else None,
                next_offset=None,
            )

    service = _LocatorService()
    fake = _branch_screen_fake(service)
    fake._library_notes_navigation_generation = 0
    fake._library_notes_navigation_status = ""

    located = await LibraryScreen._locate_library_notes_tree_target(
        fake,
        note_id="n1",
        preferred_folder_id="target",
        preferred_membership_id="m-preferred",
        focus=False,
    )

    assert located
    assert service.calls == [
        ("locator", "n1", 0, 20),
        ("folders", None, 40, 20),
        ("folders", "root-40", 60, 20),
        ("placements", "target", 80, 20),
    ]
    assert fake._library_notes_tree_expanded_ids == {"root-40", "target"}
    assert fake._library_notes_tree_selected_placement_id.endswith("m-preferred")


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_stage", ("folders", "placements"))
@pytest.mark.parametrize("late_failure", (False, True))
async def test_superseded_locator_cannot_apply_blocked_containing_range(
    blocked_stage: str, late_failure: bool
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class _BlockedLocatorService(_BranchService):
        async def locate_note_tree_placement(self, **kwargs):
            return NoteTreeLocation(
                placement_id=FolderPlacementId.note("target", "n1", "m1"),
                note_id="n1",
                membership_id="m1",
                path=(NoteTreePathStep("target", None, 40),),
                placement_offset=60,
            )

        async def page_note_folder_children(self, **kwargs):
            if blocked_stage == "folders":
                entered.set()
                await release.wait()
                if late_failure:
                    raise RuntimeError("late folder failure")
            return _folder_page(None, "target", start=40, total=41, previous=20)

        async def page_note_placements(self, **kwargs):
            if blocked_stage == "placements":
                entered.set()
                await release.wait()
                if late_failure:
                    raise RuntimeError("late placement failure")
            return NotePlacementPage(
                placements=(
                    NotePlacementRecord(
                        note={"id": "n1", "title": "Target"},
                        folder_id="target",
                        membership=_membership("m1", "target", "n1"),
                    ),
                ),
                total_placements=61,
                start_offset=60,
                previous_offset=40,
                next_offset=None,
            )

    fake = _branch_screen_fake(_BlockedLocatorService())
    task = asyncio.create_task(
        LibraryScreen._locate_library_notes_tree_target(fake, note_id="n1", focus=False)
    )
    await entered.wait()
    assert fake._library_notes_navigation_status == "Locating note…"
    status_projection = LibraryScreen._build_library_notes_tree_projection(fake)
    assert status_projection is not None
    assert status_projection.rows[0].placement_id == "status:notes-navigation"
    assert status_projection.rows[0].disabled

    LibraryScreen._supersede_library_notes_navigation(fake)
    release.set()
    assert not await task

    assert fake._library_notes_navigation_status == ""
    assert fake._library_notes_tree_expanded_ids == set()
    assert fake._library_notes_tree_selected_placement_id == ""
    assert all(
        not state.loading and not state.error
        for state in fake._library_notes_tree_branches.values()
    )


@pytest.mark.asyncio
async def test_superseded_navigation_filter_cannot_apply_or_error() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class _BlockedFilterService(_BranchService):
        async def search_note_tree_placements(self, **kwargs):
            entered.set()
            await release.wait()
            raise RuntimeError("late filter failure")

    fake = _branch_screen_fake(_BlockedFilterService())
    fake._library_notes_filter = "needle"
    generation = LibraryScreen._supersede_library_notes_navigation(fake)
    task = asyncio.create_task(
        LibraryScreen._run_library_notes_filter(
            fake,
            "needle",
            navigation_generation=generation,
        )
    )
    await entered.wait()

    LibraryScreen._supersede_library_notes_navigation(fake)
    release.set()
    await task

    state = fake._library_notes_tree_filter_state
    assert state is not None
    assert not state.loading
    assert not state.error
    assert not state.stale


@pytest.mark.asyncio
async def test_superseded_topology_receipt_reload_cannot_apply_blocked_range() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class _BlockedReceiptService(_BranchService):
        async def page_note_folder_children(self, **kwargs):
            entered.set()
            await release.wait()
            return _folder_page(None, "late", start=40, total=41, previous=20)

    fake = _branch_screen_fake(_BlockedReceiptService())
    receipt = LibraryNotesTreeReceipt(
        selected_placement_id=FolderPlacementId.folder("late"),
        selected_note_id="",
        expanded_folder_ids=("late",),
        branch_ranges=(LibraryNotesBranchRange(None, "folders", 40, 41),),
        filter_query="",
        filter_range=None,
        focus_semantic_id=FolderPlacementId.folder("late"),
        focus_role="folder-placement",
        scroll_offset=None,
        rail_scroll_offset=None,
        lifecycle_generation=0,
        topology_epoch=0,
    )
    task = asyncio.create_task(
        LibraryScreen._reload_library_notes_browse_return_receipt(fake, receipt)
    )
    await entered.wait()

    LibraryScreen._supersede_library_notes_navigation(fake)
    release.set()
    await task

    assert fake._library_notes_tree_expanded_ids == set()
    assert all(
        not state.loading for state in fake._library_notes_tree_branches.values()
    )


def test_submitting_new_filter_clears_previous_result_state():
    worker_calls = []

    async def _filter(query):
        return None

    def _run_worker(awaitable, **kwargs):
        worker_calls.append(kwargs)
        awaitable.close()

    fake = SimpleNamespace(
        _library_notes_filter="old",
        _library_notes_filter_records=[{"id": "old-note"}],
        _library_notes_filter_generation=7,
        _library_notes_tree_filter_state=object(),
        _library_notes_select_mode=True,
        _library_notes_row_selection=SimpleNamespace(clear=lambda: None),
        _run_library_notes_filter=_filter,
        _safe_text=lambda value, max_length: value[:max_length],
        run_worker=_run_worker,
    )
    event = SimpleNamespace(value="new", stop=lambda: None)

    LibraryScreen.handle_library_notes_filter(fake, event)

    assert fake._library_notes_filter_records is None
    assert fake._library_notes_tree_filter_state is None
    assert fake._library_notes_filter_generation == 8
    assert worker_calls == [{"exclusive": True, "group": "library_notes_filter"}]


class _MutationService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def create_note_folder(self, **kwargs):
        self.calls.append(("create", kwargs))
        return _folder("new", kwargs["parent_id"], "/New")

    async def page_note_placements(self, **kwargs):
        return _placement_page(kwargs["parent_id"], "n1", "n3")

    async def attach_note_to_folder(self, **kwargs):
        self.calls.append(("attach", kwargs))
        return SimpleNamespace(membership_id="new-membership")

    async def detach_note_from_folder(self, **kwargs):
        self.calls.append(("detach", kwargs))
        return True

    async def rename_note_folder(self, **kwargs):
        self.calls.append(("rename", kwargs))
        return SimpleNamespace(
            folder=replace(_folder("ideas", None, "/Renamed"), version=2)
        )

    async def move_note_folder(self, **kwargs):
        self.calls.append(("move_folder", kwargs))
        return SimpleNamespace(
            folder=replace(_folder("ideas", "work", "/Work/Ideas"), version=2)
        )

    async def delete_note_folder(self, **kwargs):
        self.calls.append(("delete_folder", kwargs))
        return SimpleNamespace(
            folder=replace(_folder("ideas", None, "/Ideas"), version=2, deleted=True)
        )

    async def restore_note_folder(self, **kwargs):
        self.calls.append(("restore_folder", kwargs))
        return SimpleNamespace(
            folder=replace(_folder("ideas", None, "/Ideas"), version=3)
        )


class _FailingMutationService(_MutationService):
    def __init__(self, failure: Exception) -> None:
        super().__init__()
        self.failure = failure

    async def create_note_folder(self, **kwargs):
        raise self.failure


class _PartialMoveService(_MutationService):
    async def detach_note_from_folder(self, **kwargs):
        self.calls.append(("detach", kwargs))
        raise FolderConflictError("Membership changed during mutation.")


def _mutation_fake(service: _MutationService):
    fake = _screen_fake(service)  # type: ignore[arg-type]
    fake._library_notes_tree_target_offsets = {}
    fake._library_notes_tree_status_by_slice = {}
    fake._library_notes_tree_status_revision = 0
    fake._sync_library_notes_tree_canvas_if_present = lambda **_kwargs: None
    fake._library_note_import_execution_active = lambda: False
    fake._library_notes_mutation_in_flight = False
    fake._library_notes_notice = ""
    fake._library_notes_deleted_folder_receipt = None
    fake._library_notes_tree_selected_placement_id = ""
    return fake


@pytest.mark.asyncio
async def test_create_folder_mutation_uses_normalized_service_and_refreshes_tree():
    service = _MutationService()
    fake = _mutation_fake(service)

    ok = await LibraryScreen._execute_library_notes_tree_mutation(
        fake,
        "create_folder",
        name="New",
        parent_id="personal",
    )

    assert ok
    assert service.calls == [
        (
            "create",
            {
                "scope": "local_note",
                "name": "New",
                "parent_id": "personal",
                "user_id": "tester",
            },
        )
    ]
    assert fake._library_notes_tree_topology_epoch == 2


@pytest.mark.asyncio
async def test_move_manual_placement_attaches_before_detaching_original():
    service = _MutationService()
    fake = _mutation_fake(service)

    ok = await LibraryScreen._execute_library_notes_tree_mutation(
        fake,
        "move_placement",
        note_id="n1",
        destination_folder_id="reading",
        source_folder_id="ideas",
        membership_version=3,
        protected=False,
    )

    assert ok
    assert [name for name, _ in service.calls] == ["attach", "detach"]
    assert service.calls[1][1]["expected_version"] == 3


@pytest.mark.asyncio
async def test_managed_placement_mutation_is_rejected_before_service_call():
    service = _MutationService()
    fake = _mutation_fake(service)

    ok = await LibraryScreen._execute_library_notes_tree_mutation(
        fake,
        "move_placement",
        note_id="n1",
        destination_folder_id="reading",
        source_folder_id="ideas",
        membership_version=3,
        protected=True,
    )

    assert not ok
    assert service.calls == []
    assert "sync" in fake._library_notes_notice.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "payload", "expected_call"),
    (
        (
            "rename_folder",
            {"folder_id": "ideas", "name": "Renamed", "expected_version": 1},
            "rename",
        ),
        (
            "move_folder",
            {"folder_id": "ideas", "parent_id": "work", "expected_version": 1},
            "move_folder",
        ),
        (
            "add_placement",
            {"folder_id": "ideas", "note_id": "n1"},
            "attach",
        ),
        (
            "detach_placement",
            {"folder_id": "ideas", "note_id": "n1", "expected_version": 1},
            "detach",
        ),
    ),
)
async def test_folder_and_membership_operations_route_to_normalized_service(
    operation, payload, expected_call
):
    service = _MutationService()
    fake = _mutation_fake(service)
    assert await LibraryScreen._execute_library_notes_tree_mutation(
        fake, operation, **payload
    )
    assert service.calls[0][0] == expected_call


@pytest.mark.asyncio
async def test_folder_remove_creates_exact_restore_receipt_and_restore_consumes_it():
    service = _MutationService()
    fake = _mutation_fake(service)
    assert await LibraryScreen._execute_library_notes_tree_mutation(
        fake,
        "delete_folder",
        folder_id="ideas",
        expected_version=1,
    )
    receipt = fake._library_notes_deleted_folder_receipt
    assert (receipt.folder_id, receipt.expected_version) == ("ideas", 2)

    assert await LibraryScreen._execute_library_notes_tree_mutation(
        fake,
        "restore_folder",
        folder_id=receipt.folder_id,
        expected_version=receipt.expected_version,
    )
    assert fake._library_notes_deleted_folder_receipt is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_copy"),
    (
        (FolderCollisionError("collision"), "already exists"),
        (FolderConflictError("conflict"), "changed elsewhere"),
        (FolderValidationError("invalid"), "not valid"),
        (
            FolderCapabilityError(
                reason_code="unsupported", user_message="Folders unavailable here."
            ),
            "Folders unavailable here.",
        ),
    ),
)
async def test_typed_folder_failures_produce_actionable_status(failure, expected_copy):
    fake = _mutation_fake(_FailingMutationService(failure))

    assert not await LibraryScreen._execute_library_notes_tree_mutation(
        fake,
        "create_folder",
        name="New",
        parent_id=None,
    )

    assert expected_copy.casefold() in fake._library_notes_notice.casefold()


@pytest.mark.asyncio
async def test_move_detach_conflict_keeps_both_placements_and_refreshes():
    service = _PartialMoveService()
    fake = _mutation_fake(service)

    ok = await LibraryScreen._execute_library_notes_tree_mutation(
        fake,
        "move_placement",
        note_id="n1",
        destination_folder_id="reading",
        source_folder_id="ideas",
        membership_version=3,
    )

    assert ok
    assert [name for name, _ in service.calls] == ["attach", "detach"]
    assert "both folders" in fake._library_notes_notice.casefold()
    assert fake._library_notes_tree_topology_epoch == 2


@pytest.mark.asyncio
async def test_note_delete_reconciliation_prefers_next_exact_branch_sibling():
    service = _MutationService()
    fake = _mutation_fake(service)
    key = NotesBranchKey("ideas", "placements")
    state = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        _placement_page("ideas", "n1", "n2", "n3"),
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state
    fake._library_notes_tree_branches[key] = state
    fake._library_notes_tree_selected_placement_id = state.item_ids[1]
    context = SimpleNamespace(
        parent_ids=frozenset(),
        placement_parent_ids=frozenset({"ideas"}),
        folder_ids=frozenset(),
        ancestor_ids=frozenset(),
    )

    await LibraryScreen._reconcile_library_notes_tree_mutation(
        fake,
        "note_delete",
        {"note_id": "n2"},
        before=context,
        result=True,
    )

    assert fake._library_notes_tree_selected_placement_id == state.item_ids[2]
    assert state.item_ids[1] not in fake._library_notes_tree_branches[key].item_ids


class _TreeCapableNotesService(StaticLibraryNotesScopeService):
    def __init__(self, notes):
        super().__init__(notes)
        self.tree_calls: list[dict[str, object]] = []

    async def load_note_folder_tree_batch(self, **kwargs):
        self.tree_calls.append(kwargs)
        expanded = tuple(kwargs["expanded_folder_ids"])
        ideas = _folder("ideas", None, "/Ideas")
        reading = _folder("reading", None, "/Reading")
        if not expanded:
            return _page(folders=(ideas, reading))
        memberships = tuple(
            NoteFolderMembership(
                membership_id=f"m-{folder_id}",
                folder_id=folder_id,
                note_id="n-1",
                ownership="managed",
                owner_id=f"sync-{folder_id}",
                owner_active=folder_id != "reading",
                version=1,
            )
            for folder_id in expanded
        )
        return _page(
            memberships=memberships,
            notes=({"id": "n-1", "title": "Q3 retro"},),
        )

    async def page_note_folder_children(self, **kwargs):
        self.tree_calls.append({"kind": "folders", **kwargs})
        if kwargs["parent_id"] is None:
            return NoteFolderChildPage(
                folders=(
                    _folder("ideas", None, "/Ideas"),
                    _folder("reading", None, "/Reading"),
                ),
                total_folders=2,
                start_offset=0,
                previous_offset=None,
                next_offset=None,
            )
        return _folder_page(kwargs["parent_id"])

    async def page_note_placements(self, **kwargs):
        self.tree_calls.append({"kind": "placements", **kwargs})
        folder_id = kwargs["parent_id"]
        if folder_id is None:
            return _placement_page(None)
        membership = NoteFolderMembership(
            membership_id=f"m-{folder_id}",
            folder_id=folder_id,
            note_id="n-1",
            ownership="managed",
            owner_id=f"sync-{folder_id}",
            owner_active=folder_id != "reading",
            version=1,
        )
        return NotePlacementPage(
            placements=(
                NotePlacementRecord(
                    note={"id": "n-1", "title": "Q3 retro"},
                    folder_id=folder_id,
                    membership=membership,
                ),
            ),
            total_placements=1,
            start_offset=0,
            previous_offset=None,
            next_offset=None,
        )


async def _wait_until(pilot, predicate, *, attempts: int = 150):
    for _ in range(attempts):
        if predicate():
            await pilot.pause()
            return
        await pilot.pause(0.02)
    raise AssertionError("Library Notes folder state did not settle")


class _ControlledMountedBranchService(StaticLibraryNotesScopeService):
    """Deterministic branch seam used through real mounted Textual workers."""

    def __init__(
        self,
        notes,
        *,
        more_mode: str = "success",
        root_folder_failure: bool = False,
        expansion_failure: bool = False,
    ) -> None:
        super().__init__(notes)
        self.more_mode = more_mode
        self.root_folder_failure = root_folder_failure
        self.expansion_failure = expansion_failure
        self.calls: list[tuple[str, str | None, int, int]] = []
        self.more_entered = asyncio.Event()
        self.more_release = asyncio.Event()
        self.expansion_entered = asyncio.Event()
        self.expansion_release = asyncio.Event()
        self._more_started = False

    async def page_note_folder_children(self, **kwargs):
        parent_id = kwargs["parent_id"]
        self.calls.append(("folders", parent_id, kwargs["offset"], kwargs["limit"]))
        if parent_id is None:
            if self.root_folder_failure:
                raise RuntimeError("root folders offline")
            return _folder_page(None, "personal")
        return _folder_page(parent_id)

    async def page_note_placements(self, **kwargs):
        parent_id = kwargs["parent_id"]
        offset = kwargs["offset"]
        self.calls.append(("placements", parent_id, offset, kwargs["limit"]))
        if parent_id is None:
            return _placement_page(None, "loose")
        if self.expansion_failure and offset == 0:
            self.expansion_entered.set()
            await self.expansion_release.wait()
            raise RuntimeError("folder placements offline")
        if offset == 20:
            self._more_started = True
            self.more_entered.set()
            await self.more_release.wait()
            if self.more_mode == "failure":
                raise RuntimeError("more offline")
            if self.more_mode == "exhausted":
                return _placement_page(
                    parent_id,
                    start=20,
                    total=20,
                    previous=0,
                )
            return _placement_page(
                parent_id,
                "n20",
                start=20,
                total=21,
                previous=0,
            )
        total = 20 if self._more_started and self.more_mode == "exhausted" else 21
        return _placement_page(
            parent_id,
            *(f"n{index:02d}" for index in range(20)),
            total=total,
            next_=20 if total == 21 else None,
        )


class _MountedSiblingBranchService(StaticLibraryNotesScopeService):
    def __init__(self, notes) -> None:
        super().__init__(notes)
        self.entered = {
            "personal": asyncio.Event(),
            "work": asyncio.Event(),
        }
        self.release = {
            "personal": asyncio.Event(),
            "work": asyncio.Event(),
        }

    async def page_note_folder_children(self, **kwargs):
        parent_id = kwargs["parent_id"]
        if parent_id is None:
            return _folder_page(None, "personal", "work")
        return _folder_page(parent_id)

    async def page_note_placements(self, **kwargs):
        parent_id = kwargs["parent_id"]
        if parent_id is None:
            return _placement_page(None)
        self.entered[parent_id].set()
        await self.release[parent_id].wait()
        return _placement_page(parent_id, f"{parent_id}-note")


class _MountedSupersedingBranchService(_ControlledMountedBranchService):
    def __init__(self, notes) -> None:
        super().__init__(notes)
        self.first_more_entered = asyncio.Event()
        self.first_more_cancelled = asyncio.Event()
        self.more_calls = 0

    async def page_note_placements(self, **kwargs):
        if kwargs["parent_id"] == "personal" and kwargs["offset"] == 20:
            self.more_calls += 1
            if self.more_calls == 1:
                self.first_more_entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.first_more_cancelled.set()
                    raise
            return _placement_page("personal", "newest", start=20, total=21, previous=0)
        return await super().page_note_placements(**kwargs)


class _MountedLateUnmountService(_ControlledMountedBranchService):
    def __init__(self, notes, *, late_failure: bool) -> None:
        super().__init__(notes)
        self.late_failure = late_failure
        self.cancel_observed = asyncio.Event()

    async def page_note_placements(self, **kwargs):
        if kwargs["parent_id"] == "personal" and kwargs["offset"] == 20:
            self.more_entered.set()
            try:
                await self.more_release.wait()
            except asyncio.CancelledError:
                self.cancel_observed.set()
                task = asyncio.current_task()
                if task is not None:
                    task.uncancel()
                await self.more_release.wait()
            if self.late_failure:
                raise RuntimeError("late failure")
            return _placement_page("personal", "late", start=20, total=21, previous=0)
        return await super().page_note_placements(**kwargs)


class _MountedTargetRecoveryService(_ControlledMountedBranchService):
    def __init__(self, notes, *, recovery_mode: str = "success") -> None:
        super().__init__(notes)
        self.recovery_mode = recovery_mode
        self.target_offsets: list[int] = []

    async def page_note_placements(self, **kwargs):
        if kwargs["parent_id"] == "personal" and kwargs["offset"] == 40:
            self.target_offsets.append(kwargs["offset"])
            if len(self.target_offsets) == 1:
                return _placement_page("personal", "wrong", start=0, total=41, next_=1)
            if self.recovery_mode == "failure":
                raise RuntimeError("target recovery offline")
            if self.recovery_mode == "second_drift":
                return _placement_page(
                    "personal", "still-wrong", start=0, total=41, next_=1
                )
            return _placement_page(
                "personal", "target", start=40, total=41, previous=20
            )
        return await super().page_note_placements(**kwargs)


class _MountedNavigationFenceService(_ControlledMountedBranchService):
    def __init__(self, notes, *, blocked_stage: str, late_failure: bool) -> None:
        super().__init__(notes)
        self.blocked_stage = blocked_stage
        self.late_failure = late_failure
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def locate_note_tree_placement(self, **_kwargs):
        return NoteTreeLocation(
            placement_id=FolderPlacementId.note("target", "n1", "m1"),
            note_id="n1",
            membership_id="m1",
            path=(NoteTreePathStep("target", None, 40),),
            placement_offset=60,
        )

    async def page_note_folder_children(self, **kwargs):
        if kwargs["offset"] == 40:
            if self.blocked_stage == "folders":
                self.entered.set()
                await self.release.wait()
                if self.late_failure:
                    raise RuntimeError("late folder failure")
            return _folder_page(None, "target", start=40, total=41, previous=20)
        return await super().page_note_folder_children(**kwargs)

    async def page_note_placements(self, **kwargs):
        if kwargs["parent_id"] == "target" and kwargs["offset"] == 60:
            if self.blocked_stage == "placements":
                self.entered.set()
                await self.release.wait()
                if self.late_failure:
                    raise RuntimeError("late placement failure")
            return NotePlacementPage(
                placements=(
                    NotePlacementRecord(
                        note={"id": "n1", "title": "Target"},
                        folder_id="target",
                        membership=_membership("m1", "target", "n1"),
                    ),
                ),
                total_placements=61,
                start_offset=60,
                previous_offset=40,
                next_offset=None,
            )
        return await super().page_note_placements(**kwargs)


class _MountedFilterRetryService(_ControlledMountedBranchService):
    def __init__(self, notes, *, failed_offset: int) -> None:
        super().__init__(notes)
        self.failed_offset = failed_offset
        self.filter_offsets: list[int] = []
        self.failed_once = False

    async def search_note_tree_placements(self, **kwargs):
        offset = kwargs["offset"]
        self.filter_offsets.append(offset)
        if (
            offset == self.failed_offset
            and len(self.filter_offsets) > 1
            and not self.failed_once
        ):
            self.failed_once = True
            raise RuntimeError("filter page failure")
        note_ids = tuple(f"n{index}" for index in range(offset, offset + 20))
        return NotePlacementPage(
            placements=tuple(
                _placement_record(note_id, "personal") for note_id in note_ids
            ),
            total_placements=40,
            start_offset=offset,
            previous_offset=None if offset == 0 else 0,
            next_offset=20 if offset == 0 else None,
            ancestor_folders=(_folder("personal", None, "/Personal"),),
        )


class _MountedBlockedFilterService(_ControlledMountedBranchService):
    def __init__(self, notes, *, late_failure: bool) -> None:
        super().__init__(notes)
        self.late_failure = late_failure
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def search_note_tree_placements(self, **_kwargs):
        self.entered.set()
        await self.release.wait()
        if self.late_failure:
            raise RuntimeError("late filter failure")
        return NotePlacementPage(
            placements=(_placement_record("n1", "personal"),),
            total_placements=1,
            start_offset=0,
            previous_offset=None,
            next_offset=None,
            ancestor_folders=(_folder("personal", None, "/Personal"),),
        )


async def _open_mounted_personal_pager(screen, pilot):
    """Enter the production Notes route and expand its real folder button."""
    await _wait_for_library_shell(screen, pilot)
    await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
    await _wait_until(
        pilot,
        lambda: any(
            getattr(row, "folder_id", "") == "personal"
            for row in screen.query(".library-notes-folder-row")
        ),
    )
    folder = next(
        row
        for row in screen.query(".library-notes-folder-row")
        if getattr(row, "folder_id", "") == "personal"
    )
    folder.press()
    await _wait_until(
        pilot,
        lambda: any(
            getattr(row, "paging_action", "") == "more"
            and getattr(row, "parent_folder_id", None) == "personal"
            for row in screen.query(".library-notes-tree-pager")
        ),
    )
    return next(
        row
        for row in screen.query(".library-notes-tree-pager")
        if getattr(row, "paging_action", "") == "more"
        and getattr(row, "parent_folder_id", None) == "personal"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_stage", ("folders", "placements"))
@pytest.mark.parametrize("late_failure", (False, True))
async def test_mounted_abandoned_locator_never_applies_blocked_stage_or_steals_focus(
    blocked_stage: str, late_failure: bool
) -> None:
    app = _build_test_app()
    notes = _two_notes()
    _seed_conversations(app, _two_conversations(), notes=notes)
    service = _MountedNavigationFenceService(
        notes, blocked_stage=blocked_stage, late_failure=late_failure
    )
    app.notes_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        await _wait_until(
            pilot,
            lambda: all(
                not state.loading
                for state in screen._library_notes_tree_branches.values()
            ),
        )
        filter_input = screen.query_one("#library-notes-filter")
        filter_input.focus()
        await _wait_until(pilot, lambda: screen.focused is filter_input)
        task = asyncio.create_task(
            screen._locate_library_notes_tree_target(note_id="n1", focus=True)
        )
        await _wait_until(pilot, service.entered.is_set)
        await _wait_until(
            pilot, lambda: bool(screen.query("#library-notes-navigation-status"))
        )
        status = screen.query_one("#library-notes-navigation-status")
        assert str(status.label) == "Locating note…"
        assert status.disabled
        assert getattr(screen.focused, "id", None) == "library-notes-filter"

        screen._supersede_library_notes_navigation()
        await _wait_until(
            pilot, lambda: not screen.query("#library-notes-navigation-status")
        )
        service.release.set()
        assert not await task
        await pilot.pause()

        assert screen._library_notes_navigation_status == ""
        assert "target" not in screen._library_notes_tree_expanded_ids
        assert screen._library_notes_tree_selected_placement_id == ""
        assert all(
            not state.loading and not state.error
            for state in screen._library_notes_tree_branches.values()
        )
        assert getattr(screen.focused, "id", None) == "library-notes-filter"


@pytest.mark.asyncio
async def test_mounted_locator_status_appears_and_clears_on_success() -> None:
    app = _build_test_app()
    notes = _two_notes()
    _seed_conversations(app, _two_conversations(), notes=notes)
    service = _MountedNavigationFenceService(
        notes, blocked_stage="folders", late_failure=False
    )
    app.notes_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        task = asyncio.create_task(
            screen._locate_library_notes_tree_target(note_id="n1", focus=False)
        )
        await _wait_until(pilot, service.entered.is_set)
        await _wait_until(
            pilot, lambda: bool(screen.query("#library-notes-navigation-status"))
        )
        assert str(screen.query_one("#library-notes-navigation-status").label) == (
            "Locating note…"
        )

        service.release.set()
        assert await task
        await _wait_until(
            pilot, lambda: not screen.query("#library-notes-navigation-status")
        )

        assert screen._library_notes_navigation_status == ""
        assert screen._library_notes_tree_selected_placement_id == (
            FolderPlacementId.note("target", "n1", "m1")
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt_kind", ("filter", "browse"))
@pytest.mark.parametrize("late_failure", (False, True))
async def test_mounted_abandoned_receipt_reload_ignores_late_page_or_filter_result(
    receipt_kind: str, late_failure: bool
) -> None:
    app = _build_test_app()
    notes = _two_notes()
    _seed_conversations(app, _two_conversations(), notes=notes)
    if receipt_kind == "filter":
        service = _MountedBlockedFilterService(notes, late_failure=late_failure)
    else:
        service = _MountedNavigationFenceService(
            notes, blocked_stage="folders", late_failure=late_failure
        )
    app.notes_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        filter_input = screen.query_one("#library-notes-filter")
        filter_input.focus()
        receipt = LibraryNotesTreeReceipt(
            selected_placement_id="",
            selected_note_id="",
            expanded_folder_ids=("target",) if receipt_kind == "browse" else (),
            branch_ranges=(
                (LibraryNotesBranchRange(None, "folders", 40, 41),)
                if receipt_kind == "browse"
                else ()
            ),
            filter_query="needle" if receipt_kind == "filter" else "",
            filter_range=(
                LibraryNotesFilterRange(0, 1) if receipt_kind == "filter" else None
            ),
            focus_semantic_id="",
            focus_role="filter",
            scroll_offset=None,
            rail_scroll_offset=None,
            lifecycle_generation=0,
            topology_epoch=0,
        )
        task = asyncio.create_task(
            screen._reload_library_notes_browse_return_receipt(receipt)
        )
        await _wait_until(pilot, service.entered.is_set)

        screen._supersede_library_notes_navigation()
        service.release.set()
        await task
        await pilot.pause()

        assert screen._library_notes_tree_expanded_ids == set()
        assert screen._library_notes_navigation_status == ""
        assert all(
            not state.loading and not state.error
            for state in screen._library_notes_tree_branches.values()
        )
        filter_state = screen._library_notes_tree_filter_state
        assert filter_state is None or (
            not filter_state.loading
            and not filter_state.error
            and not filter_state.stale
        )
        assert getattr(screen.focused, "id", None) == "library-notes-filter"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial_offset", "action", "failed_offset"),
    ((0, "more", 20), (20, "earlier", 0)),
)
async def test_mounted_filter_retry_repeats_exact_failed_offset_and_direction(
    initial_offset: int, action: str, failed_offset: int
) -> None:
    app = _build_test_app()
    notes = _two_notes()
    _seed_conversations(app, _two_conversations(), notes=notes)
    service = _MountedFilterRetryService(notes, failed_offset=failed_offset)
    app.notes_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        screen._library_notes_filter = "needle"
        await screen._run_library_notes_filter(
            "needle", offset=initial_offset, direction="replace"
        )
        await _wait_until(
            pilot,
            lambda: any(
                getattr(row, "paging_action", "") == action
                for row in screen.query(".library-notes-tree-pager")
            ),
        )
        pager = next(
            row
            for row in screen.query(".library-notes-tree-pager")
            if getattr(row, "paging_action", "") == action
        )
        pager.press()
        await _wait_until(
            pilot,
            lambda: bool(
                screen._library_notes_tree_filter_state
                and screen._library_notes_tree_filter_state.error
            ),
        )
        failed = screen._library_notes_tree_filter_state
        assert failed is not None
        assert failed.failed_offset == failed_offset
        assert failed.failed_direction == ("more" if action == "more" else "previous")
        retry = next(
            row
            for row in screen.query(".library-notes-tree-pager")
            if getattr(row, "paging_action", "") == "retry"
        )
        retry.press()
        await _wait_until(
            pilot,
            lambda: (
                len(service.filter_offsets) == 3
                and not screen._library_notes_tree_filter_state.loading
            ),
        )

        assert service.filter_offsets == [initial_offset, failed_offset, failed_offset]
        assert not screen._library_notes_tree_filter_state.error
        assert len(screen._library_notes_tree_filter_state.placements) == 40


@pytest.mark.asyncio
async def test_mounted_initial_root_slices_settle_independently_on_one_side_failure():
    app = _build_test_app()
    notes = _two_notes()
    _seed_conversations(app, _two_conversations(), notes=notes)
    service = _ControlledMountedBranchService(notes, root_folder_failure=True)
    app.notes_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        root_folders = NotesBranchKey(None, "folders")
        root_placements = NotesBranchKey(None, "placements")
        await _wait_until(
            pilot,
            lambda: (
                root_folders in screen._library_notes_tree_branches
                and root_placements in screen._library_notes_tree_branches
                and not screen._library_notes_tree_branches[root_folders].loading
                and not screen._library_notes_tree_branches[root_placements].loading
            ),
        )

        assert ("folders", None, 0, 20) in service.calls
        assert ("placements", None, 0, 20) in service.calls
        assert any(
            getattr(row, "note_id", "") == "loose"
            for row in screen.query(".library-notes-row")
        )
        retry = next(
            row
            for row in screen.query(".library-notes-tree-pager")
            if getattr(row, "parent_folder_id", "sentinel") is None
            and getattr(row, "content_kind", "") == "folders"
        )
        assert retry.paging_action == "retry"
        assert not screen._library_notes_tree_branches[root_placements].error
        tree_rows = list(screen.query(".library-notes-tree-pager, .library-notes-row"))
        assert tree_rows.index(retry) < next(
            index
            for index, row in enumerate(tree_rows)
            if getattr(row, "note_id", "") == "loose"
        )


@pytest.mark.asyncio
async def test_mounted_expansion_failure_stays_beneath_folder_and_collapse_retains_it():
    app = _build_test_app()
    notes = _two_notes()
    _seed_conversations(app, _two_conversations(), notes=notes)
    service = _ControlledMountedBranchService(notes, expansion_failure=True)
    app.notes_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        await _wait_until(
            pilot,
            lambda: any(
                getattr(row, "folder_id", "") == "personal"
                for row in screen.query(".library-notes-folder-row")
            ),
        )
        folder = next(
            row
            for row in screen.query(".library-notes-folder-row")
            if getattr(row, "folder_id", "") == "personal"
        )
        folder.focus()
        await _wait_until(
            pilot, lambda: getattr(screen.focused, "folder_id", "") == "personal"
        )
        folder.press()
        await _wait_until(pilot, service.expansion_entered.is_set)
        await _wait_until(
            pilot,
            lambda: any(
                getattr(row, "parent_folder_id", None) == "personal"
                and getattr(row, "content_kind", "") == "placements"
                and getattr(row, "paging_loading", False)
                for row in screen.query(".library-notes-tree-pager")
            ),
        )
        assert "personal" in screen._library_notes_tree_expanded_ids
        assert getattr(screen.focused, "folder_id", "") == "personal"
        loading = next(
            row
            for row in screen.query(".library-notes-tree-pager")
            if getattr(row, "parent_folder_id", None) == "personal"
            and getattr(row, "content_kind", "") == "placements"
        )
        loading_id = loading.id
        assert loading_id is not None and loading_id.endswith("-replace")
        assert loading.placement_id.endswith(":replace")
        assert loading.disabled
        assert loading.paging_loading
        assert loading.paging_action == "retry"
        assert loading.parent_folder_id == "personal"
        assert loading.content_kind == "placements"

        service.expansion_release.set()
        await _wait_until(
            pilot,
            lambda: any(
                getattr(row, "parent_folder_id", None) == "personal"
                and getattr(row, "content_kind", "") == "placements"
                and getattr(row, "paging_action", "") == "retry"
                and not getattr(row, "paging_loading", False)
                for row in screen.query(".library-notes-tree-pager")
            ),
        )
        retry = next(
            row
            for row in screen.query(".library-notes-tree-pager")
            if getattr(row, "parent_folder_id", None) == "personal"
            and getattr(row, "content_kind", "") == "placements"
        )
        assert retry.id == loading_id
        assert retry.paging_action == "retry"
        assert retry.retry_direction == "replace"
        assert str(retry.label).strip() == "Couldn’t load contents · Retry"
        tree_rows = list(
            screen.query(
                ".library-notes-folder-row, .library-notes-tree-pager, "
                ".library-notes-row"
            )
        )
        current_folder = next(
            row for row in tree_rows if getattr(row, "folder_id", "") == "personal"
        )
        assert tree_rows.index(retry) == tree_rows.index(current_folder) + 1
        assert getattr(screen.focused, "folder_id", "") == "personal"


@pytest.mark.asyncio
async def test_mounted_real_repository_statuses_protect_actions_before_page_membership(
    tmp_path,
):
    db = CharactersRAGDB(tmp_path / "mounted-folder-authority.db", client_id="mounted")
    repository = LocalNoteFolderRepository(db)
    inactive = repository.create_folder(name="Inactive", parent_id=None)
    nested = repository.create_folder(name="Nested", parent_id=None)
    nested_child = repository.create_folder(
        name="Managed child", parent_id=nested.folder_id
    )
    paged = repository.create_folder(name="Paged", parent_id=None)

    inactive_note = db.add_note("Inactive managed", "")
    nested_note = db.add_note("Nested managed", "")
    assert inactive_note is not None and nested_note is not None
    repository.reconcile_managed(
        owner_id="inactive-owner", desired=((inactive.folder_id, inactive_note),)
    )
    repository.reconcile_managed(
        owner_id="nested-owner", desired=((nested_child.folder_id, nested_note),)
    )
    repository.mark_unknown_owners_inactive(active_owner_ids=("nested-owner",))

    for index in range(20):
        note_id = db.add_note(f"A {index:02d}", "")
        assert note_id is not None
        repository.attach_manual(folder_id=paged.folder_id, note_id=note_id)
    managed_late = db.add_note("Z managed outside page", "")
    assert managed_late is not None
    repository.reconcile_managed(
        owner_id="paged-owner", desired=((paged.folder_id, managed_late),)
    )

    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    app.chachanotes_db = db
    app.notes_scope_service = NotesScopeService(
        NotesInteropService(
            tmp_path,
            "mounted",
            global_db_to_use=db,
        ),
        None,
        folder_repository=repository,
    )
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=(170, 48)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
            root_key = NotesBranchKey(None, "folders")
            await _wait_until(
                pilot,
                lambda: (
                    root_key in screen._library_notes_tree_branches
                    and not screen._library_notes_tree_branches[root_key].loading
                ),
            )
            root_state = screen._library_notes_tree_branches[root_key]
            assert root_state.freshness == "fresh", root_state
            await _wait_until(
                pilot,
                lambda: (
                    {
                        getattr(row, "folder_id", "")
                        for row in screen.query(".library-notes-folder-row")
                    }
                    >= {inactive.folder_id, nested.folder_id, paged.folder_id}
                ),
            )

            rows = {
                getattr(row, "folder_id", ""): row
                for row in screen.query(".library-notes-folder-row")
            }
            assert {
                inactive.folder_id,
                nested.folder_id,
                paged.folder_id,
            } <= rows.keys()
            assert rows[inactive.folder_id].protected_placement
            assert rows[inactive.folder_id].owner_active is False
            assert "Needs owner review" in str(rows[inactive.folder_id].label)
            assert rows[nested.folder_id].protected_placement
            assert rows[nested.folder_id].owner_active is True
            assert rows[paged.folder_id].protected_placement
            assert NotesBranchKey(nested.folder_id, "placements") not in (
                screen._library_notes_tree_branches
            )
            assert NotesBranchKey(paged.folder_id, "placements") not in (
                screen._library_notes_tree_branches
            )

            rows[inactive.folder_id].press()
            inactive_key = NotesBranchKey(inactive.folder_id, "placements")
            await _wait_until(
                pilot,
                lambda: (
                    inactive_key in screen._library_notes_tree_branches
                    and not screen._library_notes_tree_branches[inactive_key].loading
                ),
            )
            assert screen._library_notes_tree_branches[inactive_key].items == ()
            assert not [
                row
                for row in screen.query(".library-notes-tree-note-row")
                if getattr(row, "folder_id", "") == inactive.folder_id
            ]
            for control_id in (
                "#library-notes-folder-rename",
                "#library-notes-folder-move",
                "#library-notes-folder-remove",
            ):
                assert screen.query_one(control_id).disabled

            paged_row = next(
                row
                for row in screen.query(".library-notes-folder-row")
                if getattr(row, "folder_id", "") == paged.folder_id
            )
            paged_row.press()
            paged_key = NotesBranchKey(paged.folder_id, "placements")
            await _wait_until(
                pilot,
                lambda: (
                    paged_key in screen._library_notes_tree_branches
                    and not screen._library_notes_tree_branches[paged_key].loading
                ),
            )
            paged_state = screen._library_notes_tree_branches[paged_key]
            assert len(paged_state.items) == 20
            assert managed_late not in {
                str(item.note["id"]) for item in paged_state.items
            }
            visible_paged_notes = [
                row
                for row in screen.query(".library-notes-tree-note-row")
                if getattr(row, "folder_id", "") == paged.folder_id
            ]
            assert len(visible_paged_notes) == 20
            assert managed_late not in {
                getattr(row, "note_id", "") for row in visible_paged_notes
            }
            current_paged = next(
                row
                for row in screen.query(".library-notes-folder-row")
                if getattr(row, "folder_id", "") == paged.folder_id
            )
            assert current_paged.protected_placement
            assert "Sync managed" in str(current_paged.label)
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_mounted_collapse_retains_fresh_branch_without_another_read():
    app = _build_test_app()
    notes = _two_notes()
    _seed_conversations(app, _two_conversations(), notes=notes)
    service = _ControlledMountedBranchService(notes)
    app.notes_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _open_mounted_personal_pager(screen, pilot)
        first_call_count = len(
            [
                call
                for call in service.calls
                if isinstance(call, tuple) and call[1] == "personal"
            ]
        )

        current_folder = next(
            row
            for row in screen.query(".library-notes-folder-row")
            if getattr(row, "folder_id", "") == "personal"
        )
        current_folder.press()
        await _wait_until(
            pilot, lambda: "personal" not in screen._library_notes_tree_expanded_ids
        )
        collapsed_folder = next(
            row
            for row in screen.query(".library-notes-folder-row")
            if getattr(row, "folder_id", "") == "personal"
        )
        collapsed_folder.press()
        await _wait_until(
            pilot, lambda: "personal" in screen._library_notes_tree_expanded_ids
        )
        await pilot.pause()

        assert (
            len(
                [
                    call
                    for call in service.calls
                    if isinstance(call, tuple) and call[1] == "personal"
                ]
            )
            == first_call_count
        )


@pytest.mark.asyncio
async def test_mounted_sibling_branch_workers_overlap_and_both_apply():
    app = _build_test_app()
    notes = _two_notes()
    _seed_conversations(app, _two_conversations(), notes=notes)
    service = _MountedSiblingBranchService(notes)
    app.notes_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        await _wait_until(
            pilot, lambda: len(screen.query(".library-notes-folder-row")) == 2
        )
        personal = next(
            row
            for row in screen.query(".library-notes-folder-row")
            if getattr(row, "folder_id", "") == "personal"
        )
        personal.press()
        await _wait_until(
            pilot,
            lambda: (
                service.entered["personal"].is_set()
                and "personal" in screen._library_notes_tree_expanded_ids
            ),
        )
        work = next(
            row
            for row in screen.query(".library-notes-folder-row")
            if getattr(row, "folder_id", "") == "work"
        )
        work.press()
        await _wait_until(
            pilot,
            lambda: all(event.is_set() for event in service.entered.values()),
        )
        service.release["work"].set()
        service.release["personal"].set()
        await _wait_until(
            pilot,
            lambda: (
                {
                    getattr(row, "note_id", "")
                    for row in screen.query(".library-notes-row")
                }
                >= {"personal-note", "work-note"}
            ),
        )

        assert (
            screen._library_notes_tree_branches[
                NotesBranchKey("personal", "placements")
            ].freshness
            == "fresh"
        )
        assert (
            screen._library_notes_tree_branches[
                NotesBranchKey("work", "placements")
            ].freshness
            == "fresh"
        )


@pytest.mark.asyncio
async def test_mounted_newer_same_slice_worker_supersedes_the_pending_one():
    app = _build_test_app()
    notes = _two_notes()
    _seed_conversations(app, _two_conversations(), notes=notes)
    service = _MountedSupersedingBranchService(notes)
    app.notes_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        pager = await _open_mounted_personal_pager(screen, pilot)
        pager_id = pager.id
        pager.press()
        await _wait_until(pilot, service.first_more_entered.is_set)
        key = NotesBranchKey("personal", "placements")
        screen._request_library_notes_tree_slice(
            key,
            direction="more",
            offset=20,
            pager_focus_id=pager_id,
        )
        await _wait_until(pilot, service.first_more_cancelled.is_set)
        await _wait_until(
            pilot,
            lambda: any(
                getattr(row, "note_id", "") == "newest"
                for row in screen.query(".library-notes-row")
            ),
        )

        assert service.more_calls == 2
        assert (
            screen._library_notes_tree_branches[key]
            .item_ids[-1]
            .endswith(":newest:m-newest")
        )


@pytest.mark.asyncio
async def test_mounted_target_drift_recovers_the_same_nonzero_range():
    app = _build_test_app()
    notes = _two_notes()
    _seed_conversations(app, _two_conversations(), notes=notes)
    service = _MountedTargetRecoveryService(notes)
    app.notes_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        await _wait_until(
            pilot,
            lambda: (
                NotesBranchKey(None, "folders") in screen._library_notes_tree_branches
            ),
        )
        key = NotesBranchKey("personal", "placements")
        screen._request_library_notes_tree_slice(key, direction="target", offset=40)
        await _wait_until(
            pilot,
            lambda: (
                key in screen._library_notes_tree_branches
                and not screen._library_notes_tree_branches[key].loading
            ),
        )

        assert service.target_offsets == [40, 40]
        assert screen._library_notes_tree_branches[key].start_offset == 40


@pytest.mark.asyncio
@pytest.mark.parametrize("recovery_mode", ("second_drift", "failure"))
async def test_mounted_broken_target_recovery_stales_only_its_local_slice(
    recovery_mode: str,
):
    app = _build_test_app()
    notes = _two_notes()
    _seed_conversations(app, _two_conversations(), notes=notes)
    service = _MountedTargetRecoveryService(notes, recovery_mode=recovery_mode)
    app.notes_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _open_mounted_personal_pager(screen, pilot)
        key = NotesBranchKey("personal", "placements")
        root_key = NotesBranchKey(None, "folders")
        screen._request_library_notes_tree_slice(key, direction="target", offset=40)
        await _wait_until(
            pilot,
            lambda: (
                len(service.target_offsets) == 2
                and not screen._library_notes_tree_branches[key].loading
            ),
        )

        state = screen._library_notes_tree_branches[key]
        assert service.target_offsets == [40, 40]
        assert state.freshness == "stale"
        assert state.total is None
        assert screen._library_notes_tree_branches[root_key].freshness == "fresh"
        retry = next(
            row
            for row in screen.query(".library-notes-tree-pager")
            if getattr(row, "parent_folder_id", None) == "personal"
            and getattr(row, "content_kind", "") == "placements"
            and getattr(row, "paging_action", "") == "retry"
        )
        retry.press()
        await _wait_until(pilot, lambda: len(service.target_offsets) == 3)
        assert service.target_offsets[-1] == 40


@pytest.mark.asyncio
@pytest.mark.parametrize("late_failure", (False, True))
async def test_mounted_true_unmount_fences_late_branch_success_and_failure(
    late_failure: bool,
    monkeypatch,
):
    app = _build_test_app()
    notes = _two_notes()
    _seed_conversations(app, _two_conversations(), notes=notes)
    service = _MountedLateUnmountService(notes, late_failure=late_failure)
    app.notes_scope_service = service
    host = LibraryHarness(app)
    focus_handoffs = 0
    original_focus_handoff = LibraryScreen._focus_library_notes_tree_after_page

    def _tracked_focus_handoff(self, *args, **kwargs):
        nonlocal focus_handoffs
        focus_handoffs += 1
        return original_focus_handoff(self, *args, **kwargs)

    monkeypatch.setattr(
        LibraryScreen,
        "_focus_library_notes_tree_after_page",
        _tracked_focus_handoff,
    )

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        pager = await _open_mounted_personal_pager(screen, pilot)
        recompose_calls = 0
        original_sync = screen._sync_library_notes_tree_canvas_if_present

        def _tracked_sync(*args, **kwargs):
            nonlocal recompose_calls
            recompose_calls += 1
            return original_sync(*args, **kwargs)

        screen._sync_library_notes_tree_canvas_if_present = _tracked_sync
        pager.press()
        await _wait_until(pilot, service.more_entered.is_set)
        calls_before_unmount = recompose_calls
        focus_before_unmount = focus_handoffs
        await host.pop_screen()
        assert screen._library_notes_tree_branches == {}
        assert screen._library_notes_tree_request_generations == {}

        service.more_release.set()
        await pilot.pause()
        await pilot.pause()
        assert screen._library_notes_tree_branches == {}
        assert screen._library_notes_tree_status_by_slice == {}
        assert screen._library_notes_tree_target_offsets == {}
        assert recompose_calls == calls_before_unmount
        assert focus_handoffs == focus_before_unmount

        fresh = LibraryScreen(app)
        await host.push_screen(fresh)
        await _wait_for_library_shell(fresh, pilot)
        assert fresh._library_notes_tree_branches == {}
        assert fresh._library_notes_tree_request_generations == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("more_mode", ("success", "failure", "exhausted"))
async def test_mounted_pager_completion_uses_ordered_semantic_focus(more_mode: str):
    app = _build_test_app()
    notes = _two_notes()
    _seed_conversations(app, _two_conversations(), notes=notes)
    service = _ControlledMountedBranchService(notes, more_mode=more_mode)
    app.notes_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        pager = await _open_mounted_personal_pager(screen, pilot)
        pager_id = pager.id
        pager.focus()
        await _wait_until(pilot, lambda: screen.focused is pager)
        pager.press()
        await _wait_until(pilot, service.more_entered.is_set)
        await _wait_until(
            pilot,
            lambda: screen.focused is not None and screen.focused.id == pager_id,
        )
        service.more_release.set()

        if more_mode == "success":
            await _wait_until(
                pilot,
                lambda: getattr(screen.focused, "note_id", "") == "n20",
            )
        elif more_mode == "failure":
            await _wait_until(
                pilot,
                lambda: (
                    screen.focused is not None
                    and screen.focused.id == pager_id
                    and getattr(screen.focused, "paging_action", "") == "retry"
                ),
            )
        else:
            await _wait_until(
                pilot,
                lambda: getattr(screen.focused, "folder_id", "") == "personal",
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("more_mode", ("success", "failure"))
async def test_mounted_pager_completion_does_not_steal_moved_focus(more_mode: str):
    app = _build_test_app()
    notes = _two_notes()
    _seed_conversations(app, _two_conversations(), notes=notes)
    service = _ControlledMountedBranchService(notes, more_mode=more_mode)
    app.notes_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        pager = await _open_mounted_personal_pager(screen, pilot)
        pager.focus()
        pager.press()
        await _wait_until(pilot, service.more_entered.is_set)
        await _wait_until(
            pilot,
            lambda: any(
                row.id == pager.id and row.disabled
                for row in screen.query(".library-notes-tree-pager")
            ),
        )
        filter_input = screen.query_one("#library-notes-filter")
        filter_input.focus()
        await _wait_until(
            pilot,
            lambda: (
                screen.focused is not None
                and screen.focused.id == "library-notes-filter"
            ),
        )
        service.more_release.set()
        key = NotesBranchKey("personal", "placements")
        await _wait_until(
            pilot,
            lambda: not screen._library_notes_tree_branches[key].loading,
        )
        await pilot.pause()
        assert screen.focused is not None
        assert screen.focused.id == "library-notes-filter"


@pytest.mark.asyncio
@pytest.mark.parametrize("more_mode", ("success", "failure"))
async def test_mounted_pager_completion_does_not_restore_after_sync_guard_stales(
    more_mode: str,
    monkeypatch: pytest.MonkeyPatch,
):
    response_sync_seen = asyncio.Event()
    recompose_entered = asyncio.Event()
    recompose_release = asyncio.Event()
    response_sync_armed = False
    original_sync_state = LibraryNotesCanvas.sync_state
    original_recompose = LibraryNotesCanvas.recompose

    def tracked_sync_state(canvas: LibraryNotesCanvas, **kwargs) -> None:
        original_sync_state(canvas, **kwargs)
        if response_sync_armed:
            response_sync_seen.set()

    async def gated_recompose(canvas: LibraryNotesCanvas) -> None:
        if response_sync_seen.is_set() and not recompose_entered.is_set():
            recompose_entered.set()
            await recompose_release.wait()
        await original_recompose(canvas)

    monkeypatch.setattr(LibraryNotesCanvas, "sync_state", tracked_sync_state)
    monkeypatch.setattr(LibraryNotesCanvas, "recompose", gated_recompose)
    app = _build_test_app()
    notes = _two_notes()
    _seed_conversations(app, _two_conversations(), notes=notes)
    service = _ControlledMountedBranchService(notes, more_mode=more_mode)
    app.notes_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        pager = await _open_mounted_personal_pager(screen, pilot)
        pager.focus()
        pager.press()
        await _wait_until(pilot, service.more_entered.is_set)
        await _wait_until(
            pilot,
            lambda: (
                screen.focused is not None
                and screen.focused.id == pager.id
                and screen.focused.disabled
            ),
        )

        response_sync_armed = True
        service.more_release.set()
        await asyncio.wait_for(response_sync_seen.wait(), timeout=2)
        await asyncio.wait_for(recompose_entered.wait(), timeout=2)
        try:
            focus_generation = screen._library_notes_focus_intent_generation
            filter_input = screen.query_one("#library-notes-filter")
            # The canvas's message pump is deliberately paused inside its
            # recompose. Apply the user's focus choice synchronously, then
            # advance the same authority generation Textual's queued
            # ``DescendantFocus`` will advance; waiting on that event here
            # would deadlock on the gate.
            screen.set_focus(filter_input)
            screen._library_notes_focus_intent_generation += 1
            assert screen.focused is filter_input
            assert screen._library_notes_focus_intent_generation > focus_generation
        finally:
            recompose_release.set()
        key = NotesBranchKey("personal", "placements")
        await _wait_until(
            pilot,
            lambda: not screen._library_notes_tree_branches[key].loading,
        )
        await pilot.pause()
        assert screen.focused is not None
        assert screen.focused.id == "library-notes-filter"


@pytest.mark.asyncio
@pytest.mark.parametrize("more_mode", ("success", "failure"))
async def test_mounted_pager_completion_rechecks_focus_after_inner_recompose(
    more_mode: str,
    monkeypatch: pytest.MonkeyPatch,
):
    response_sync_seen = asyncio.Event()
    outer_recompose_entered = asyncio.Event()
    outer_recompose_release = asyncio.Event()
    inner_recompose_completed = asyncio.Event()
    inner_recompose_release = asyncio.Event()
    response_sync_armed = False
    original_sync_state = LibraryNotesCanvas.sync_state
    original_outer_recompose = LibraryNotesCanvas.recompose
    original_inner_recompose = PostRecomposeCallback.recompose

    def tracked_sync_state(canvas: LibraryNotesCanvas, **kwargs) -> None:
        original_sync_state(canvas, **kwargs)
        if response_sync_armed:
            response_sync_seen.set()

    async def gated_outer_recompose(canvas: LibraryNotesCanvas) -> None:
        if response_sync_seen.is_set() and not outer_recompose_entered.is_set():
            outer_recompose_entered.set()
            await outer_recompose_release.wait()
        await original_outer_recompose(canvas)

    async def gated_inner_recompose(canvas: PostRecomposeCallback) -> None:
        await original_inner_recompose(canvas)
        if (
            isinstance(canvas, LibraryNotesCanvas)
            and outer_recompose_entered.is_set()
            and not inner_recompose_completed.is_set()
        ):
            inner_recompose_completed.set()
            await inner_recompose_release.wait()

    monkeypatch.setattr(LibraryNotesCanvas, "sync_state", tracked_sync_state)
    monkeypatch.setattr(LibraryNotesCanvas, "recompose", gated_outer_recompose)
    monkeypatch.setattr(
        PostRecomposeCallback,
        "recompose",
        gated_inner_recompose,
    )
    app = _build_test_app()
    notes = _two_notes()
    _seed_conversations(app, _two_conversations(), notes=notes)
    service = _ControlledMountedBranchService(notes, more_mode=more_mode)
    app.notes_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        pager = await _open_mounted_personal_pager(screen, pilot)
        pager.focus()
        pager.press()
        await _wait_until(pilot, service.more_entered.is_set)
        await _wait_until(
            pilot,
            lambda: (
                screen.focused is not None
                and screen.focused.id == pager.id
                and screen.focused.disabled
            ),
        )

        response_sync_armed = True
        service.more_release.set()
        await asyncio.wait_for(response_sync_seen.wait(), timeout=2)
        await asyncio.wait_for(outer_recompose_entered.wait(), timeout=2)
        filter_input = screen.query_one("#library-notes-filter")
        screen.set_focus(filter_input)
        screen._library_notes_focus_intent_generation += 1
        assert screen.focused is filter_input
        outer_recompose_release.set()

        await asyncio.wait_for(inner_recompose_completed.wait(), timeout=2)
        newest_target = screen.query_one("#library-row-browse-notes")
        screen.set_focus(newest_target)
        screen._library_notes_focus_intent_generation += 1
        assert screen.focused is newest_target
        inner_recompose_release.set()

        key = NotesBranchKey("personal", "placements")
        await _wait_until(
            pilot,
            lambda: not screen._library_notes_tree_branches[key].loading,
        )
        await pilot.pause()
        assert screen.focused is newest_target


@pytest.mark.asyncio
async def test_live_host_renders_duplicate_placements_and_preserves_focus_at_60x20():
    app = _build_test_app()
    notes = _two_notes()
    _seed_conversations(app, _two_conversations(), notes=notes)
    service = _TreeCapableNotesService(notes)
    app.notes_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        await _wait_until(
            pilot, lambda: len(screen.query(".library-notes-folder-row")) == 2
        )
        for folder_id in ("ideas", "reading"):
            folder = next(
                row
                for row in screen.query(".library-notes-folder-row")
                if getattr(row, "folder_id", "") == folder_id
            )
            folder.press()
            await _wait_until(
                pilot,
                lambda folder_id=folder_id: any(
                    getattr(row, "folder_id", "") == folder_id
                    for row in screen.query(".library-notes-row")
                ),
            )

        placements = [
            row
            for row in screen.query(".library-notes-row")
            if getattr(row, "note_id", "") == "n-1"
        ]
        assert len(placements) == 2
        assert len({row.placement_id for row in placements}) == 2
        assert service.detail_calls == []

        placements[-1].focus()
        focused_placement = placements[-1].placement_id
        await _wait_until(
            pilot,
            lambda: (
                str(getattr(screen.focused, "placement_id", "")) == focused_placement
            ),
        )
        screen.refresh(recompose=True)
        await _wait_until(
            pilot,
            lambda: (
                str(getattr(screen.focused, "placement_id", "")) == focused_placement
            ),
        )

        await pilot.resize_terminal(60, 20)
        await _wait_until(pilot, lambda: screen._library_notes_compact is True)
        focused = next(
            row
            for row in screen.query(".library-notes-row")
            if row.placement_id == focused_placement
        )
        focused.scroll_visible()
        await pilot.pause()
        assert focused.region.width <= screen.query_one("#library-canvas").region.width
        screenshot = host.export_screenshot(simplify=True)
        painted = unescape(screenshot).replace("\xa0", " ")
        assert "Q3 retro" in painted
        assert any(label in painted for label in ("Ideas", "Reading"))
        assert any(
            status in painted for status in ("Synced placement", "Needs owner review")
        )
