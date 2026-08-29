"""Library-screen orchestration tests for the Database Notes folder tree."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import replace
from html import unescape
import inspect
from types import SimpleNamespace

from loguru import logger as loguru_logger
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
    UNFILED_PLACEMENT_ID,
    LibraryNotesBranchRange,
    LibraryNotesFilterState,
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
async def test_external_note_deep_link_without_preferred_placement_uses_locator_choice():
    class _ExternalLocatorService(_BranchService):
        def __init__(self) -> None:
            super().__init__()
            self.locator_kwargs: dict[str, object] = {}

        async def locate_note_tree_placement(self, **kwargs):
            self.locator_kwargs = kwargs
            return NoteTreeLocation(
                placement_id=FolderPlacementId.unfiled("external-note"),
                note_id="external-note",
                membership_id=None,
                path=(),
                placement_offset=20,
            )

        async def page_note_placements(self, **kwargs):
            return _placement_page(
                None,
                "external-note",
                start=20,
                total=21,
                previous=0,
            )

    service = _ExternalLocatorService()
    fake = _branch_screen_fake(service)

    located = await LibraryScreen._locate_library_notes_tree_target(
        fake, note_id="external-note", focus=False
    )

    assert located
    assert service.locator_kwargs["preferred_folder_id"] is None
    assert service.locator_kwargs["preferred_membership_id"] is None
    assert fake._library_notes_tree_selected_placement_id == (
        FolderPlacementId.unfiled("external-note")
    )
    root = fake._library_notes_tree_branches[NotesBranchKey(None, "placements")]
    assert root.start_offset == 20
    assert root.total == 21


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


@pytest.mark.asyncio
async def test_topology_changed_receipt_reloads_exact_range_and_relocates_folder():
    class _ReceiptService(_BranchService):
        def __init__(self) -> None:
            super().__init__()
            self.offsets: list[int] = []

        async def page_note_folder_children(self, **kwargs):
            self.offsets.append(kwargs["offset"])
            return _folder_page(None, "late", start=40, total=41, previous=20)

        async def locate_note_tree_folder(self, **_kwargs):
            return NoteTreeLocation(
                placement_id=FolderPlacementId.folder("late"),
                note_id=None,
                membership_id=None,
                path=(NoteTreePathStep("late", None, 40),),
                placement_offset=None,
            )

    service = _ReceiptService()
    fake = _branch_screen_fake(service)
    fake._restore_library_notes_focus_identity = lambda *_args, **_kwargs: True
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

    await LibraryScreen._reload_library_notes_browse_return_receipt(fake, receipt)

    root = fake._library_notes_tree_branches[NotesBranchKey(None, "folders")]
    assert service.offsets == [40, 40]
    assert root.start_offset == 40
    assert root.item_ids == (FolderPlacementId.folder("late"),)
    assert root.total == 41
    assert root.freshness == "fresh"
    assert fake._library_notes_tree_expanded_ids == {"late"}
    assert fake._library_notes_tree_selected_placement_id == (
        FolderPlacementId.folder("late")
    )
    assert fake._library_notes_navigation_status == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("preferred_survives", (True, False))
async def test_topology_receipt_reload_passes_exact_duplicate_locator_identity(
    preferred_survives: bool,
) -> None:
    class _DuplicateReceiptService(_BranchService):
        def __init__(self) -> None:
            super().__init__()
            self.locator_kwargs: dict[str, object] = {}

        async def locate_note_tree_placement(self, **kwargs):
            self.locator_kwargs = kwargs
            membership_id = "m-preferred" if preferred_survives else "m-fallback"
            return NoteTreeLocation(
                placement_id=FolderPlacementId.note(
                    "target", "duplicate-note", membership_id
                ),
                note_id="duplicate-note",
                membership_id=membership_id,
                path=(NoteTreePathStep("target", None, 0),),
                placement_offset=0,
            )

        async def page_note_folder_children(self, **kwargs):
            return _folder_page(kwargs["parent_id"], "target")

        async def page_note_placements(self, **_kwargs):
            return NotePlacementPage(
                placements=(
                    NotePlacementRecord(
                        note={"id": "duplicate-note", "title": "Duplicate"},
                        folder_id="target",
                        membership=_membership(
                            "m-preferred", "target", "duplicate-note"
                        ),
                    ),
                    NotePlacementRecord(
                        note={"id": "duplicate-note", "title": "Duplicate"},
                        folder_id="target",
                        membership=_membership(
                            "m-fallback", "target", "duplicate-note"
                        ),
                    ),
                ),
                total_placements=2,
                start_offset=0,
                previous_offset=None,
                next_offset=None,
            )

    service = _DuplicateReceiptService()
    fake = _branch_screen_fake(service)
    fake._restore_library_notes_focus_identity = lambda *_args, **_kwargs: True
    receipt = LibraryNotesTreeReceipt(
        selected_placement_id=FolderPlacementId.note(
            "target", "duplicate-note", "m-preferred"
        ),
        selected_note_id="duplicate-note",
        expanded_folder_ids=("target",),
        branch_ranges=(),
        filter_query="",
        filter_range=None,
        focus_semantic_id=FolderPlacementId.note(
            "target", "duplicate-note", "m-preferred"
        ),
        focus_role="note-placement",
        scroll_offset=None,
        rail_scroll_offset=None,
        lifecycle_generation=0,
        topology_epoch=0,
        preferred_folder_id="target",
        preferred_membership_id="m-preferred",
    )

    await LibraryScreen._reload_library_notes_browse_return_receipt(fake, receipt)

    assert service.locator_kwargs["preferred_folder_id"] == "target"
    assert service.locator_kwargs["preferred_membership_id"] == "m-preferred"
    expected_membership = "m-preferred" if preferred_survives else "m-fallback"
    assert fake._library_notes_tree_selected_placement_id == FolderPlacementId.note(
        "target", "duplicate-note", expected_membership
    )


def test_semantic_receipt_captures_exact_duplicate_membership_and_folder_ids() -> None:
    fake = _branch_screen_fake(_BranchService())
    folder_key = NotesBranchKey(None, "folders")
    placement_key = NotesBranchKey("target", "placements")
    fake._library_notes_tree_branches[folder_key] = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(folder_key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        _folder_page(None, "target"),
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state
    record = NotePlacementRecord(
        note={"id": "duplicate-note", "title": "Duplicate"},
        folder_id="target",
        membership=_membership("m-preferred", "target", "duplicate-note"),
    )
    fake._library_notes_tree_branches[placement_key] = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(placement_key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        NotePlacementPage(
            placements=(record,),
            total_placements=1,
            start_offset=0,
            previous_offset=None,
            next_offset=None,
        ),
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state
    selected = FolderPlacementId.note("target", "duplicate-note", "m-preferred")
    fake._library_notes_tree_selected_placement_id = selected
    fake._library_notes_tree_expanded_ids = {"target"}
    fake._capture_library_notes_focus_identity = lambda **_kwargs: SimpleNamespace(
        region="navigator",
        semantic_role=f"note-placement:{selected}",
        note_id="duplicate-note",
        scroll_offset=None,
    )
    fake._library_notes_last_user_focus = None
    fake._library_notes_interaction_focus = None
    fake._library_notes_last_presented_focus = None
    fake._library_notes_scroll_owner = lambda *_args: None
    fake._build_library_notes_tree_projection = lambda: (
        LibraryScreen._build_library_notes_tree_projection(fake)
    )

    receipt = LibraryScreen._capture_library_notes_browse_return_receipt(fake)

    assert receipt.selected_placement_id == selected
    assert receipt.preferred_folder_id == "target"
    assert receipt.preferred_membership_id == "m-preferred"


@pytest.mark.asyncio
async def test_removed_locator_target_uses_deterministic_visible_fallback_and_clears_status():
    class _RemovedTargetService(_BranchService):
        async def locate_note_tree_folder(self, **_kwargs):
            return None

    fake = _branch_screen_fake(_RemovedTargetService())
    root_key = NotesBranchKey(None, "folders")
    fake._library_notes_tree_branches[root_key] = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(root_key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        _folder_page(None, "fallback", "other"),
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state

    located = await LibraryScreen._locate_library_notes_tree_target(
        fake, folder_id="removed", focus=True
    )

    assert not located
    assert fake._library_notes_tree_selected_placement_id == (
        FolderPlacementId.folder("fallback")
    )
    assert fake._library_notes_navigation_status == ""
    assert not any(
        state.loading for state in fake._library_notes_tree_branches.values()
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


@pytest.mark.parametrize("clear_kind", ("button", "empty_submit"))
def test_clearing_filter_restores_same_epoch_browse_receipt_without_touching_ranges(
    monkeypatch, clear_kind: str
) -> None:
    service = _BranchService()
    fake = _branch_screen_fake(service)
    key = NotesBranchKey("personal", "placements")
    trusted = replace(
        apply_notes_slice_page(
            begin_notes_slice_load(
                empty_notes_slice(key, topology_epoch=1),
                generation=1,
                direction="replace",
                requested_offset=0,
                requested_limit=20,
            ),
            _placement_page(
                "personal", *(f"n{index}" for index in range(20)), total=41, next_=20
            ),
            direction="replace",
            request_generation=1,
            topology_epoch=1,
        ).state,
        start_offset=20,
        previous_offset=0,
        next_offset=40,
    )
    fake._library_notes_tree_branches[key] = trusted
    browse_selection = trusted.item_ids[0]
    receipt = LibraryNotesTreeReceipt(
        selected_placement_id=browse_selection,
        selected_note_id="n0",
        expanded_folder_ids=("personal",),
        branch_ranges=(LibraryNotesBranchRange("personal", "placements", 20, 40),),
        filter_query="",
        filter_range=None,
        focus_semantic_id=browse_selection,
        focus_role="note-placement",
        scroll_offset=None,
        rail_scroll_offset=None,
        lifecycle_generation=1,
        topology_epoch=1,
    )
    fake._library_notes_filter = _FILTER_QUERY_SENTINEL
    fake._library_notes_filter_records = [
        {"id": "filtered-note", "title": _NOTE_TITLE_SENTINEL}
    ]
    fake._library_notes_filter_browse_receipt = receipt
    fake._library_notes_tree_selected_placement_id = "unfiled:filtered-note"
    fake._library_notes_tree_expanded_ids = set()
    fake._library_notes_sort_choices_visible = True
    fake._library_notes_select_mode = True
    fake._library_notes_row_selection = SimpleNamespace(clear=lambda: None)
    fake._safe_text = lambda value, max_length: value[:max_length]
    fake._restore_library_notes_focus_identity = lambda *_args, **_kwargs: True
    fake._library_notes_scroll_owner = lambda *_args: None
    fake._focus_library_notes_filter_input = lambda: None
    callbacks = []

    def sync(_screen, _kind, *, then=None, **_kwargs):
        callbacks.append(then)
        if then is not None:
            then()

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.library_screen._sync_library_canvas", sync
    )

    event = SimpleNamespace(stop=lambda: None, value="")
    if clear_kind == "button":
        LibraryScreen.handle_library_notes_filter_clear(fake, event)
    else:
        LibraryScreen.handle_library_notes_filter(fake, event)

    assert callbacks
    assert fake._library_notes_filter == ""
    assert fake._library_notes_filter_browse_receipt is None
    assert fake._library_notes_tree_selected_placement_id == browse_selection
    assert fake._library_notes_tree_expanded_ids == {"personal"}
    assert fake._library_notes_tree_branches[key] is trusted
    assert fake._library_notes_tree_branches[key].total == 41
    assert fake._library_notes_tree_branches[key].freshness == "fresh"


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
        return _membership("new-membership", kwargs["folder_id"], kwargs["note_id"])

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


_FILTER_QUERY_SENTINEL = "PRIVATE_FILTER_QUERY_TASK_18917"
_NOTE_TITLE_SENTINEL = "PRIVATE_NOTE_TITLE_TASK_18917"
_NOTE_BODY_SENTINEL = "PRIVATE_NOTE_BODY_TASK_18917"
_FOLDER_NAME_SENTINEL = "PRIVATE_FOLDER_NAME_TASK_18917"
_FOLDER_PATH_SENTINEL = "/private/folder/path/TASK_18917"
_MEMBERSHIP_SENTINEL = "PRIVATE_MEMBERSHIP_DISPLAY_TASK_18917"
_FAILURE_MESSAGE_SENTINEL = "PRIVATE_EXCEPTION_MESSAGE_TASK_18917"
_PRIVATE_LOG_SENTINELS = (
    _FILTER_QUERY_SENTINEL,
    _NOTE_TITLE_SENTINEL,
    _NOTE_BODY_SENTINEL,
    _FOLDER_NAME_SENTINEL,
    _FOLDER_PATH_SENTINEL,
    _MEMBERSHIP_SENTINEL,
    _FAILURE_MESSAGE_SENTINEL,
)


def _private_notes_failure() -> RuntimeError:
    return RuntimeError(" | ".join(_PRIVATE_LOG_SENTINELS))


@contextmanager
def _capture_notes_failure_logs():
    records: list[dict[str, object]] = []
    rendered: list[str] = []

    def capture(message) -> None:
        records.append(dict(message.record))
        rendered.append(str(message))

    sink_id = loguru_logger.add(
        capture,
        level="WARNING",
        format="{message}|{extra}|{exception}",
    )
    try:
        yield records, rendered
    finally:
        loguru_logger.remove(sink_id)


def _assert_failure_records_are_private(
    records: list[dict[str, object]], rendered: list[str]
) -> None:
    assert records
    assert all(record["exception"] is None for record in records)
    serialized = f"{records!r}\n{rendered!r}"
    for sentinel in _PRIVATE_LOG_SENTINELS:
        assert sentinel not in serialized


def _assert_log_extra(
    record: dict[str, object], expected: dict[str, object]
) -> dict[str, object]:
    extra = record["extra"]
    assert isinstance(extra, dict)
    assert {key: extra.get(key) for key in expected} == expected
    return extra


@pytest.mark.asyncio
async def test_page_failure_log_is_structured_and_excludes_private_content():
    class _PageFails(_BranchService):
        async def page_note_folder_children(self, **_kwargs):
            raise _private_notes_failure()

    fake = _branch_screen_fake(_PageFails())
    key = NotesBranchKey("safe-parent-id", "folders")

    with _capture_notes_failure_logs() as (records, rendered):
        await LibraryScreen._load_library_notes_tree_slice(
            fake,
            key,
            direction="more",
            offset=20,
        )

    assert len(records) == 1
    record = records[0]
    assert record["message"] == "library_notes_tree_page_failed"
    _assert_log_extra(
        record,
        {
            "event": "library_notes_tree_page_failed",
            "operation": "page",
            "content_kind": "folders",
            "direction": "more",
            "parent_id": "safe-parent-id",
            "slice_generation": 1,
            "navigation_generation": None,
            "topology_epoch": 1,
            "lifecycle_generation": 1,
            "exception_class": "RuntimeError",
        },
    )
    _assert_failure_records_are_private(records, rendered)


@pytest.mark.asyncio
async def test_locator_failure_log_is_structured_and_excludes_private_content():
    class _LocatorFails(_BranchService):
        async def locate_note_tree_placement(self, **_kwargs):
            raise _private_notes_failure()

    fake = _branch_screen_fake(_LocatorFails())

    with _capture_notes_failure_logs() as (records, rendered):
        located = await LibraryScreen._locate_library_notes_tree_target(
            fake,
            note_id="safe-note-id",
            preferred_folder_id="safe-folder-id",
            preferred_membership_id="safe-membership-id",
        )

    assert not located
    assert len(records) == 1
    record = records[0]
    assert record["message"] == "library_notes_tree_locator_failed"
    _assert_log_extra(
        record,
        {
            "event": "library_notes_tree_locator_failed",
            "operation": "locator",
            "locator_kind": "placement",
            "target_id": "safe-note-id",
            "navigation_generation": 1,
            "topology_epoch": 1,
            "lifecycle_generation": 1,
            "exception_class": "RuntimeError",
        },
    )
    _assert_failure_records_are_private(records, rendered)


@pytest.mark.asyncio
async def test_filter_failure_log_is_structured_and_excludes_query_and_content():
    class _FilterFails(_BranchService):
        async def search_note_tree_placements(self, **_kwargs):
            raise _private_notes_failure()

    fake = _branch_screen_fake(_FilterFails())
    fake._library_notes_filter = _FILTER_QUERY_SENTINEL

    with _capture_notes_failure_logs() as (records, rendered):
        await LibraryScreen._run_library_notes_filter(
            fake,
            _FILTER_QUERY_SENTINEL,
            offset=20,
            direction="more",
        )

    assert len(records) == 1
    record = records[0]
    assert record["message"] == "library_notes_tree_filter_failed"
    _assert_log_extra(
        record,
        {
            "event": "library_notes_tree_filter_failed",
            "operation": "filter",
            "direction": "more",
            "requested_offset": 20,
            "filter_generation": 1,
            "navigation_generation": None,
            "topology_epoch": 1,
            "lifecycle_generation": 1,
            "exception_class": "RuntimeError",
        },
    )
    _assert_failure_records_are_private(records, rendered)


@pytest.mark.asyncio
async def test_mutation_admission_context_failure_is_private_and_preserves_trusted_state():
    class _AdmissionContextFails(_MutationService):
        async def load_note_tree_mutation_context(self, **_kwargs):
            raise _private_notes_failure()

        async def rename_note_folder(self, **kwargs):
            self.calls.append(("rename", kwargs))
            raise AssertionError("storage mutation must not run")

    service = _AdmissionContextFails()
    fake = _mutation_fake(service)
    key = NotesBranchKey(None, "folders")
    private_folder = NoteFolder(
        folder_id="safe-folder-id",
        parent_id=None,
        name=_FOLDER_NAME_SENTINEL,
        path=_FOLDER_PATH_SENTINEL,
        normalized_path=_FOLDER_PATH_SENTINEL.casefold(),
        version=1,
        deleted=False,
    )
    trusted = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        NoteFolderChildPage(
            folders=(private_folder,),
            total_folders=1,
            start_offset=0,
            previous_offset=None,
            next_offset=None,
        ),
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state
    fake._library_notes_tree_branches[key] = trusted
    fake._library_notes_filter = _FILTER_QUERY_SENTINEL
    fake._library_notes_filter_records = [
        {
            "id": "safe-note-id",
            "title": _NOTE_TITLE_SENTINEL,
            "content": _NOTE_BODY_SENTINEL,
            "membership_display": _MEMBERSHIP_SENTINEL,
        }
    ]

    with _capture_notes_failure_logs() as (records, rendered):
        committed = await LibraryScreen._execute_library_notes_tree_mutation(
            fake,
            "rename_folder",
            folder_id="safe-folder-id",
            name=_FOLDER_NAME_SENTINEL,
            expected_version=1,
        )

    assert not committed
    assert service.calls == []
    retained = fake._library_notes_tree_branches[key]
    assert retained.items == trusted.items
    assert retained.total == trusted.total == 1
    assert retained.freshness == "fresh"
    assert not retained.loading
    assert not fake._library_notes_mutation_in_flight
    assert len(records) == 1
    record = records[0]
    assert record["message"] == "library_notes_tree_mutation_context_failed"
    _assert_log_extra(
        record,
        {
            "event": "library_notes_tree_mutation_context_failed",
            "operation": "mutation_context",
            "mutation_operation": "rename_folder",
            "content_kind": "mutation_context",
            "direction": "admission_context",
            "parent_id": None,
            "slice_generation": None,
            "navigation_generation": 1,
            "topology_epoch": 2,
            "lifecycle_generation": 1,
            "exception_class": "RuntimeError",
        },
    )
    _assert_failure_records_are_private(records, rendered)


@pytest.mark.asyncio
async def test_postcommit_mutation_refresh_logs_are_structured_and_private():
    class _RenameRefreshFails(_MutationService):
        def __init__(self) -> None:
            super().__init__()
            self.context_calls = 0

        async def load_note_tree_mutation_context(self, **_kwargs):
            self.context_calls += 1
            if self.context_calls > 1:
                raise _private_notes_failure()
            return SimpleNamespace(
                parent_ids=(None,),
                placement_parent_ids=("safe-folder-id",),
                folder_ids=("safe-folder-id",),
                ancestor_ids=(),
            )

        async def rename_note_folder(self, **kwargs):
            self.calls.append(("rename", kwargs))
            return SimpleNamespace(
                folder=NoteFolder(
                    folder_id="safe-folder-id",
                    parent_id=None,
                    name=_FOLDER_NAME_SENTINEL,
                    path=_FOLDER_PATH_SENTINEL,
                    normalized_path=_FOLDER_PATH_SENTINEL.casefold(),
                    version=2,
                    deleted=False,
                ),
                affected_folder_ids=("safe-folder-id",),
            )

        async def page_note_folder_children(self, **_kwargs):
            raise _private_notes_failure()

        async def page_note_placements(self, **_kwargs):
            raise _private_notes_failure()

        async def locate_note_tree_folder(self, **_kwargs):
            return None

    fake = _mutation_fake(_RenameRefreshFails())
    root_key = NotesBranchKey(None, "folders")
    placements_key = NotesBranchKey("safe-folder-id", "placements")
    private_folder = NoteFolder(
        folder_id="safe-folder-id",
        parent_id=None,
        name=_FOLDER_NAME_SENTINEL,
        path=_FOLDER_PATH_SENTINEL,
        normalized_path=_FOLDER_PATH_SENTINEL.casefold(),
        version=1,
        deleted=False,
    )
    fake._library_notes_tree_branches[root_key] = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(root_key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        NoteFolderChildPage(
            folders=(private_folder,),
            total_folders=1,
            start_offset=0,
            previous_offset=None,
            next_offset=None,
        ),
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state
    private_record = NotePlacementRecord(
        note={
            "id": "safe-note-id",
            "title": _NOTE_TITLE_SENTINEL,
            "content": _NOTE_BODY_SENTINEL,
        },
        folder_id="safe-folder-id",
        membership=_membership("safe-membership-id", "safe-folder-id", "safe-note-id"),
    )
    fake._library_notes_tree_branches[placements_key] = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(placements_key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        NotePlacementPage(
            placements=(private_record,),
            total_placements=1,
            start_offset=0,
            previous_offset=None,
            next_offset=None,
        ),
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state

    with _capture_notes_failure_logs() as (records, rendered):
        committed = await LibraryScreen._execute_library_notes_tree_mutation(
            fake,
            "rename_folder",
            folder_id="safe-folder-id",
            name=_FOLDER_NAME_SENTINEL,
            expected_version=1,
        )

    assert committed
    assert records
    assert all(
        record["message"] == "library_notes_tree_mutation_refresh_failed"
        for record in records
    )
    for record in records:
        extra = _assert_log_extra(
            record,
            {
                "event": "library_notes_tree_mutation_refresh_failed",
                "operation": "mutation_refresh",
                "mutation_operation": "rename_folder",
                "topology_epoch": 2,
                "lifecycle_generation": 1,
                "exception_class": "RuntimeError",
            },
        )
        assert extra["content_kind"] in {
            "folders",
            "placements",
            "mutation_context",
        }
        if extra["content_kind"] == "mutation_context":
            assert extra["direction"] == "refresh_context"
            assert extra["slice_generation"] is None
        else:
            assert extra["direction"] == "target"
            assert isinstance(extra["slice_generation"], int)
    _assert_failure_records_are_private(records, rendered)


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
async def test_committed_folder_move_removes_old_parent_ghost_before_failed_refresh():
    service = _MutationService()
    fake = _mutation_fake(service)
    old_key = NotesBranchKey("old-parent", "folders")
    new_key = NotesBranchKey("new-parent", "folders")
    child_key = NotesBranchKey("moved", "folders")
    fake._library_notes_tree_branches[old_key] = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(old_key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        NoteFolderChildPage(
            folders=(
                _folder("moved", "old-parent", "/Old/Moved"),
                _folder("sibling", "old-parent", "/Old/Sibling"),
            ),
            total_folders=2,
            start_offset=0,
            previous_offset=None,
            next_offset=None,
        ),
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state
    fake._library_notes_tree_branches[new_key] = replace(
        empty_notes_slice(new_key, topology_epoch=1),
        freshness="fresh",
        total=0,
    )
    child = _folder("child", "moved", "/Old/Moved/Child")
    fake._library_notes_tree_branches[child_key] = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(child_key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        NoteFolderChildPage(
            folders=(child,),
            total_folders=1,
            start_offset=0,
            previous_offset=None,
            next_offset=None,
        ),
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state
    before = SimpleNamespace(
        parent_ids=("old-parent", "moved"),
        placement_parent_ids=(),
        folder_ids=("moved", "child"),
        ancestor_ids=("old-parent",),
    )
    committed = replace(_folder("moved", "new-parent", "/New/Moved"), version=7)

    await LibraryScreen._reconcile_library_notes_tree_mutation(
        fake,
        "move_folder",
        {"folder_id": "moved", "parent_id": "new-parent"},
        before=before,
        result=SimpleNamespace(
            folder=committed, affected_folder_ids=("moved", "child")
        ),
    )

    old_state = fake._library_notes_tree_branches[old_key]
    assert FolderPlacementId.folder("moved") not in old_state.item_ids
    assert FolderPlacementId.folder("sibling") in old_state.item_ids
    assert old_state.total is None
    assert old_state.freshness == "stale"
    descendant = fake._library_notes_tree_branches[child_key].items[0]
    assert descendant.path == "/New/Moved/Child"
    assert descendant.normalized_path == "/new/moved/child"
    assert fake._library_notes_tree_pending_target_placement_id == (
        FolderPlacementId.folder("moved")
    )


@pytest.mark.asyncio
async def test_full_placement_move_removes_only_exact_source_membership_duplicate():
    class _RefreshFails(_MutationService):
        async def page_note_placements(self, **_kwargs):
            raise RuntimeError("refresh failed")

    service = _RefreshFails()
    fake = _mutation_fake(service)
    source_key = NotesBranchKey("ideas", "placements")
    first = NotePlacementRecord(
        note={"id": "n1", "title": "Duplicate"},
        folder_id="ideas",
        membership=_membership("m-source", "ideas", "n1"),
    )
    duplicate = NotePlacementRecord(
        note={"id": "n1", "title": "Duplicate"},
        folder_id="ideas",
        membership=_membership("m-survives", "ideas", "n1"),
    )
    source_page = NotePlacementPage(
        placements=(first, duplicate),
        total_placements=2,
        start_offset=0,
        previous_offset=None,
        next_offset=None,
    )
    fake._library_notes_tree_branches[source_key] = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(source_key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        source_page,
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state
    before = SimpleNamespace(
        parent_ids=(),
        placement_parent_ids=("ideas", "reading"),
        folder_ids=(),
        ancestor_ids=(),
    )
    destination = _membership("m-destination", "reading", "n1")
    source_id = FolderPlacementId.note("ideas", "n1", "m-source")

    await LibraryScreen._reconcile_library_notes_tree_mutation(
        fake,
        "move_placement",
        {
            "note_id": "n1",
            "source_folder_id": "ideas",
            "source_membership_id": "m-source",
            "source_placement_id": source_id,
            "destination_folder_id": "reading",
        },
        before=before,
        result=True,
        destination_membership=destination,
    )

    source = fake._library_notes_tree_branches[source_key]
    assert source_id not in source.item_ids
    assert FolderPlacementId.note("ideas", "n1", "m-survives") in source.item_ids
    assert source.total is None
    assert fake._library_notes_tree_pending_target_placement_id == (
        FolderPlacementId.note("reading", "n1", "m-destination")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", (False, True))
async def test_move_operation_retains_exact_attached_membership_as_desired_target(
    partial: bool,
) -> None:
    class _ExactMoveService(_PartialMoveService if partial else _MutationService):
        async def page_note_placements(self, **_kwargs):
            raise RuntimeError("post-commit refresh failed")

        async def locate_note_tree_placement(self, **kwargs):
            self.locator_kwargs = kwargs
            return None

    service = _ExactMoveService()
    fake = _mutation_fake(service)
    source_key = NotesBranchKey("ideas", "placements")
    source_page = NotePlacementPage(
        placements=(
            NotePlacementRecord(
                note={"id": "n1", "title": "Duplicate"},
                folder_id="ideas",
                membership=_membership("m-source", "ideas", "n1"),
            ),
            NotePlacementRecord(
                note={"id": "n1", "title": "Duplicate"},
                folder_id="ideas",
                membership=_membership("m-other", "ideas", "n1"),
            ),
        ),
        total_placements=2,
        start_offset=0,
        previous_offset=None,
        next_offset=None,
    )
    fake._library_notes_tree_branches[source_key] = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(source_key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        source_page,
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state
    source_id = FolderPlacementId.note("ideas", "n1", "m-source")
    fake._library_notes_tree_selected_placement_id = source_id

    ok = await LibraryScreen._execute_library_notes_tree_mutation(
        fake,
        "move_placement",
        note_id="n1",
        source_folder_id="ideas",
        source_membership_id="m-source",
        source_placement_id=source_id,
        destination_folder_id="reading",
        membership_version=1,
    )

    assert ok
    desired = FolderPlacementId.note("reading", "n1", "new-membership")
    assert fake._library_notes_tree_pending_target_placement_id == desired
    assert service.locator_kwargs["preferred_folder_id"] == "reading"
    assert service.locator_kwargs["preferred_membership_id"] == "new-membership"
    source = fake._library_notes_tree_branches[source_key]
    assert (source_id in source.item_ids) is partial
    assert FolderPlacementId.note("ideas", "n1", "m-other") in source.item_ids
    if partial:
        assert "both folders" in fake._library_notes_notice.casefold()


@pytest.mark.asyncio
async def test_committed_folder_delete_removes_loaded_subtree_and_placements_on_refresh_failure():
    class _RefreshFails(_MutationService):
        async def page_note_folder_children(self, **_kwargs):
            raise RuntimeError("refresh failed")

        async def page_note_placements(self, **_kwargs):
            raise RuntimeError("refresh failed")

    fake = _mutation_fake(_RefreshFails())
    root_key = NotesBranchKey(None, "folders")
    ancestor_children_key = NotesBranchKey("ancestor", "folders")
    deleted_children_key = NotesBranchKey("deleted", "folders")
    deleted_placements_key = NotesBranchKey("deleted", "placements")
    child_placements_key = NotesBranchKey("child", "placements")
    fake._library_notes_tree_branches[root_key] = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(root_key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        _folder_page(None, "ancestor"),
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state
    fake._library_notes_tree_branches[ancestor_children_key] = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(ancestor_children_key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        _folder_page("ancestor", "deleted", "sibling"),
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state
    fake._library_notes_tree_branches[deleted_children_key] = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(deleted_children_key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        _folder_page("deleted", "child"),
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state
    for key, note_id in (
        (deleted_placements_key, "n-deleted"),
        (child_placements_key, "n-child"),
    ):
        fake._library_notes_tree_branches[key] = apply_notes_slice_page(
            begin_notes_slice_load(
                empty_notes_slice(key, topology_epoch=1),
                generation=1,
                direction="replace",
                requested_offset=0,
                requested_limit=20,
            ),
            _placement_page(key.parent_id, note_id),
            direction="replace",
            request_generation=1,
            topology_epoch=1,
        ).state
    before = SimpleNamespace(
        parent_ids=(None, "ancestor", "deleted"),
        placement_parent_ids=(),
        folder_ids=("deleted", "child"),
        ancestor_ids=("ancestor",),
    )
    tombstone = replace(
        _folder("deleted", "ancestor", "/ancestor/deleted"),
        deleted=True,
        version=2,
    )

    await LibraryScreen._reconcile_library_notes_tree_mutation(
        fake,
        "delete_folder",
        {"folder_id": "deleted"},
        before=before,
        result=SimpleNamespace(
            folder=tombstone, affected_folder_ids=("deleted", "child")
        ),
    )

    assert (
        FolderPlacementId.folder("deleted")
        not in fake._library_notes_tree_branches[ancestor_children_key].item_ids
    )
    assert (
        FolderPlacementId.folder("sibling")
        in fake._library_notes_tree_branches[ancestor_children_key].item_ids
    )
    assert (
        FolderPlacementId.folder("ancestor")
        in fake._library_notes_tree_branches[root_key].item_ids
    )
    assert fake._library_notes_tree_branches[deleted_children_key].items == ()
    assert fake._library_notes_tree_branches[deleted_placements_key].items == ()
    assert fake._library_notes_tree_branches[child_placements_key].items == ()
    assert all(
        fake._library_notes_tree_branches[key].total is None
        for key in (
            root_key,
            ancestor_children_key,
            deleted_children_key,
            deleted_placements_key,
            child_placements_key,
        )
    )


@pytest.mark.asyncio
async def test_committed_rename_patches_subtree_paths_and_versions_before_failed_refresh():
    class _RefreshFails(_MutationService):
        async def page_note_folder_children(self, **_kwargs):
            raise RuntimeError("refresh failed")

        async def page_note_placements(self, **_kwargs):
            raise RuntimeError("refresh failed")

    fake = _mutation_fake(_RefreshFails())
    root_key = NotesBranchKey(None, "folders")
    renamed_key = NotesBranchKey("renamed", "folders")
    fake._library_notes_tree_branches[root_key] = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(root_key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        NoteFolderChildPage(
            folders=(_folder("renamed", None, "/Old"),),
            total_folders=1,
            start_offset=0,
            previous_offset=None,
            next_offset=None,
        ),
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state
    fake._library_notes_tree_branches[renamed_key] = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(renamed_key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        NoteFolderChildPage(
            folders=(_folder("child", "renamed", "/Old/Child"),),
            total_folders=1,
            start_offset=0,
            previous_offset=None,
            next_offset=None,
        ),
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state
    before = SimpleNamespace(
        parent_ids=(None, "renamed"),
        placement_parent_ids=(),
        folder_ids=("renamed", "child"),
        ancestor_ids=(),
    )
    committed = replace(_folder("renamed", None, "/New"), version=9)

    await LibraryScreen._reconcile_library_notes_tree_mutation(
        fake,
        "rename_folder",
        {"folder_id": "renamed"},
        before=before,
        result=SimpleNamespace(
            folder=committed, affected_folder_ids=("renamed", "child")
        ),
    )

    renamed = fake._library_notes_tree_branches[root_key].items[0]
    child = fake._library_notes_tree_branches[renamed_key].items[0]
    assert (renamed.name, renamed.path, renamed.version) == ("New", "/New", 9)
    assert (child.path, child.normalized_path) == ("/New/Child", "/new/child")
    assert fake._library_notes_tree_branches[root_key].freshness == "stale"
    assert fake._library_notes_tree_branches[renamed_key].freshness == "stale"


@pytest.mark.asyncio
async def test_active_filter_rename_refresh_failure_keeps_committed_patch_and_isolated_browse():
    class _FilterAndRefreshFail(_MutationService):
        async def search_note_tree_placements(self, **_kwargs):
            raise RuntimeError("filter refresh failed")

        async def page_note_folder_children(self, **_kwargs):
            raise RuntimeError("branch refresh failed")

        async def page_note_placements(self, **_kwargs):
            raise RuntimeError("branch refresh failed")

        async def locate_note_tree_folder(self, **_kwargs):
            return None

    fake = _mutation_fake(_FilterAndRefreshFail())
    fake._library_notes_filter = "needle"
    filtered_id = FolderPlacementId.note("child", "n1", "m1")
    fake._library_notes_tree_selected_placement_id = filtered_id
    fake._library_notes_tree_filter_state = LibraryNotesFilterState.from_page(
        query="needle",
        page=NotePlacementPage(
            placements=(
                NotePlacementRecord(
                    note={"id": "n1", "title": "Note"},
                    folder_id="child",
                    membership=_membership("m1", "child", "n1"),
                ),
            ),
            ancestor_folders=(
                _folder("renamed", None, "/Old"),
                _folder("child", "renamed", "/Old/Child"),
            ),
            total_placements=1,
            start_offset=0,
            previous_offset=None,
            next_offset=None,
        ),
        generation=0,
        topology_epoch=1,
    )
    unrelated_key = NotesBranchKey("unrelated", "placements")
    unrelated = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(unrelated_key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        _placement_page("unrelated", "safe"),
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state
    fake._library_notes_tree_branches[unrelated_key] = unrelated
    LibraryScreen._fence_library_notes_tree_mutation(fake)

    await LibraryScreen._reconcile_library_notes_tree_mutation(
        fake,
        "rename_folder",
        {"folder_id": "renamed"},
        before=SimpleNamespace(
            parent_ids=(None,),
            placement_parent_ids=(),
            folder_ids=("renamed", "child"),
            ancestor_ids=(),
        ),
        result=SimpleNamespace(
            folder=replace(_folder("renamed", None, "/New"), version=9),
            affected_folder_ids=("renamed", "child"),
        ),
    )

    state = fake._library_notes_tree_filter_state
    assert state is not None
    assert [
        (folder.name, folder.path, folder.version) for folder in state.ancestor_folders
    ] == [
        ("New", "/New", 9),
        ("Child", "/New/Child", 1),
    ]
    assert state.total is None
    assert state.previous_offset is None
    assert state.next_offset is None
    assert state.stale is True
    assert state.failed_direction == "target"
    assert state.failed_offset == 0
    assert fake._library_notes_tree_selected_placement_id == filtered_id
    assert fake._library_notes_tree_branches[unrelated_key].items == unrelated.items
    assert fake._library_notes_tree_branches[unrelated_key].total == unrelated.total
    assert fake._library_notes_tree_branches[unrelated_key].freshness == "fresh"


@pytest.mark.asyncio
async def test_active_filter_commit_refreshes_retained_exact_range_successfully():
    class _FilterRefreshes(_MutationService):
        def __init__(self) -> None:
            super().__init__()
            self.filter_offsets: list[int] = []

        async def search_note_tree_placements(self, **kwargs):
            self.filter_offsets.append(kwargs["offset"])
            return NotePlacementPage(
                placements=(
                    NotePlacementRecord(
                        note={"id": "n1", "title": "Note"},
                        folder_id="renamed",
                        membership=_membership("m1", "renamed", "n1"),
                    ),
                ),
                ancestor_folders=(
                    replace(_folder("renamed", None, "/New"), version=9),
                ),
                total_placements=1,
                start_offset=0,
                previous_offset=None,
                next_offset=None,
            )

        async def locate_note_tree_folder(self, **_kwargs):
            return None

    service = _FilterRefreshes()
    fake = _mutation_fake(service)
    fake._library_notes_filter = "needle"
    selected = FolderPlacementId.note("renamed", "n1", "m1")
    fake._library_notes_tree_selected_placement_id = selected
    fake._library_notes_tree_filter_state = LibraryNotesFilterState.from_page(
        query="needle",
        page=NotePlacementPage(
            placements=(
                NotePlacementRecord(
                    note={"id": "n1", "title": "Note"},
                    folder_id="renamed",
                    membership=_membership("m1", "renamed", "n1"),
                ),
            ),
            ancestor_folders=(_folder("renamed", None, "/Old"),),
            total_placements=1,
            start_offset=0,
            previous_offset=None,
            next_offset=None,
        ),
        generation=0,
        topology_epoch=1,
    )
    LibraryScreen._fence_library_notes_tree_mutation(fake)

    await LibraryScreen._reconcile_library_notes_tree_mutation(
        fake,
        "rename_folder",
        {"folder_id": "renamed"},
        before=SimpleNamespace(
            parent_ids=(),
            placement_parent_ids=(),
            folder_ids=("renamed",),
            ancestor_ids=(),
        ),
        result=SimpleNamespace(
            folder=replace(_folder("renamed", None, "/New"), version=9),
            affected_folder_ids=("renamed",),
        ),
    )

    state = fake._library_notes_tree_filter_state
    assert state is not None
    assert service.filter_offsets == [0]
    assert state.total == 1
    assert state.stale is False
    assert state.error == ""
    assert state.ancestor_folders[0].name == "New"
    assert fake._library_notes_tree_selected_placement_id == selected


@pytest.mark.asyncio
async def test_detach_removes_exact_membership_and_targets_unfiled_without_touching_duplicate():
    class _RefreshFails(_MutationService):
        async def page_note_placements(self, **_kwargs):
            raise RuntimeError("refresh failed")

    fake = _mutation_fake(_RefreshFails())
    source_key = NotesBranchKey("ideas", "placements")
    root_key = NotesBranchKey(None, "placements")
    records = (
        NotePlacementRecord(
            note={"id": "n1", "title": "Duplicate"},
            folder_id="ideas",
            membership=_membership("m-source", "ideas", "n1"),
        ),
        NotePlacementRecord(
            note={"id": "n1", "title": "Duplicate"},
            folder_id="ideas",
            membership=_membership("m-other", "ideas", "n1"),
        ),
    )
    fake._library_notes_tree_branches[source_key] = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(source_key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        NotePlacementPage(
            placements=records,
            total_placements=2,
            start_offset=0,
            previous_offset=None,
            next_offset=None,
        ),
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state
    fake._library_notes_tree_branches[root_key] = replace(
        empty_notes_slice(root_key, topology_epoch=1),
        freshness="fresh",
        total=0,
    )
    source_id = FolderPlacementId.note("ideas", "n1", "m-source")

    await LibraryScreen._reconcile_library_notes_tree_mutation(
        fake,
        "detach_placement",
        {
            "folder_id": "ideas",
            "note_id": "n1",
            "source_membership_id": "m-source",
            "source_placement_id": source_id,
        },
        before=SimpleNamespace(
            parent_ids=(),
            placement_parent_ids=("ideas",),
            folder_ids=(),
            ancestor_ids=(),
        ),
        result=True,
    )

    source = fake._library_notes_tree_branches[source_key]
    assert source_id not in source.item_ids
    assert FolderPlacementId.note("ideas", "n1", "m-other") in source.item_ids
    assert fake._library_notes_tree_branches[root_key].total is None
    assert fake._library_notes_tree_pending_target_placement_id == (
        FolderPlacementId.unfiled("n1")
    )
    assert fake._library_notes_tree_selected_placement_id == UNFILED_PLACEMENT_ID


@pytest.mark.asyncio
async def test_attach_no_commit_failure_preserves_trusted_ranges_and_selection():
    class _AttachFails(_MutationService):
        async def attach_note_to_folder(self, **_kwargs):
            raise FolderConflictError("attach rejected")

    fake = _mutation_fake(_AttachFails())
    key = NotesBranchKey("ideas", "placements")
    state = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        _placement_page("ideas", "n1", "n2"),
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state
    fake._library_notes_tree_branches[key] = state
    fake._library_notes_tree_selected_placement_id = state.item_ids[0]
    fake._library_notes_tree_expanded_ids = {"ideas"}
    fake._library_notes_filter = "needle"
    filter_state = LibraryNotesFilterState.from_page(
        query="needle",
        page=_placement_page("ideas", "n1", "n2"),
        generation=2,
        topology_epoch=1,
    )
    fake._library_notes_tree_filter_state = filter_state

    ok = await LibraryScreen._execute_library_notes_tree_mutation(
        fake,
        "add_placement",
        folder_id="reading",
        note_id="n1",
    )

    retained = fake._library_notes_tree_branches[key]
    assert not ok
    assert retained.items == state.items
    assert retained.total == 2
    assert retained.freshness == "fresh"
    assert not retained.loading
    assert fake._library_notes_tree_selected_placement_id == state.item_ids[0]
    assert fake._library_notes_tree_expanded_ids == {"ideas"}
    retained_filter = fake._library_notes_tree_filter_state
    assert retained_filter is not None
    assert retained_filter.placements == filter_state.placements
    assert retained_filter.ancestor_folders == filter_state.ancestor_folders
    assert retained_filter.total == filter_state.total
    assert retained_filter.previous_offset == filter_state.previous_offset
    assert retained_filter.next_offset == filter_state.next_offset
    assert retained_filter.stale is False
    assert retained_filter.error == ""


@pytest.mark.asyncio
async def test_detach_no_commit_result_preserves_trusted_range_and_exact_placement():
    class _DetachRejected(_MutationService):
        async def detach_note_from_folder(self, **kwargs):
            self.calls.append(("detach", kwargs))
            return False

    fake = _mutation_fake(_DetachRejected())
    key = NotesBranchKey("ideas", "placements")
    state = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        _placement_page("ideas", "n1", "n2"),
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state
    fake._library_notes_tree_branches[key] = state
    selected = state.item_ids[0]
    fake._library_notes_tree_selected_placement_id = selected
    fake._library_notes_tree_expanded_ids = {"ideas"}

    ok = await LibraryScreen._execute_library_notes_tree_mutation(
        fake,
        "detach_placement",
        folder_id="ideas",
        note_id="n1",
        source_membership_id="membership-n1",
        source_placement_id=selected,
        expected_version=1,
    )

    retained = fake._library_notes_tree_branches[key]
    assert not ok
    assert retained.items == state.items
    assert retained.total == 2
    assert retained.freshness == "fresh"
    assert not retained.loading
    assert fake._library_notes_tree_selected_placement_id == selected
    assert fake._library_notes_tree_expanded_ids == {"ideas"}


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", ("next", "previous", "parent", "canonical"))
async def test_note_delete_uses_exact_four_stage_fallback_after_refresh_failure(
    fallback: str,
) -> None:
    class _RefreshFails(_MutationService):
        async def page_note_placements(self, **_kwargs):
            raise RuntimeError("refresh failed")

    fake = _mutation_fake(_RefreshFails())
    parent_id = None if fallback == "canonical" else "ideas"
    target_key = NotesBranchKey(parent_id, "placements")
    note_ids = {
        "next": ("target", "next"),
        "previous": ("previous", "target"),
        "parent": ("target",),
        "canonical": ("target",),
    }[fallback]
    fake._library_notes_tree_branches[target_key] = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(target_key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        _placement_page(parent_id, *note_ids),
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state
    target_id = (
        FolderPlacementId.unfiled("target")
        if parent_id is None
        else FolderPlacementId.note("ideas", "target", "m-target")
    )
    fake._library_notes_tree_selected_placement_id = target_id
    expected = {
        "next": FolderPlacementId.note("ideas", "next", "m-next"),
        "previous": FolderPlacementId.note("ideas", "previous", "m-previous"),
        "parent": FolderPlacementId.folder("ideas"),
        "canonical": FolderPlacementId.note("other", "canonical", "m-canonical"),
    }[fallback]
    unrelated_key = NotesBranchKey("other", "placements")
    if fallback == "canonical":
        root_folders = NotesBranchKey(None, "folders")
        fake._library_notes_tree_branches[root_folders] = apply_notes_slice_page(
            begin_notes_slice_load(
                empty_notes_slice(root_folders, topology_epoch=1),
                generation=1,
                direction="replace",
                requested_offset=0,
                requested_limit=20,
            ),
            _folder_page(None, "other"),
            direction="replace",
            request_generation=1,
            topology_epoch=1,
        ).state
        fake._library_notes_tree_branches[unrelated_key] = apply_notes_slice_page(
            begin_notes_slice_load(
                empty_notes_slice(unrelated_key, topology_epoch=1),
                generation=1,
                direction="replace",
                requested_offset=0,
                requested_limit=20,
            ),
            _placement_page("other", "canonical"),
            direction="replace",
            request_generation=1,
            topology_epoch=1,
        ).state
        fake._library_notes_tree_expanded_ids.add("other")

    await LibraryScreen._reconcile_library_notes_tree_mutation(
        fake,
        "note_delete",
        {"note_id": "target"},
        before=SimpleNamespace(
            parent_ids=(),
            placement_parent_ids=((parent_id,) if parent_id is not None else ()),
            folder_ids=(),
            ancestor_ids=(),
        ),
        result=True,
    )

    target = fake._library_notes_tree_branches[target_key]
    assert target_id not in target.item_ids
    assert target.total is None
    assert target.freshness == "stale"
    assert fake._library_notes_tree_selected_placement_id == expected
    if fallback == "canonical":
        unrelated = fake._library_notes_tree_branches[unrelated_key]
        assert unrelated.total == 1
        assert unrelated.freshness == "fresh"


@pytest.mark.asyncio
async def test_note_create_refreshes_unfiled_and_every_exact_active_placement_parent():
    class _CreateContextService(_MutationService):
        async def load_note_tree_mutation_context(self, **_kwargs):
            return SimpleNamespace(
                parent_ids=(),
                placement_parent_ids=("ideas", "reading"),
                folder_ids=(),
                ancestor_ids=(),
            )

        async def page_note_placements(self, **_kwargs):
            raise RuntimeError("refresh failed")

    fake = _mutation_fake(_CreateContextService())
    affected_keys = (
        NotesBranchKey(None, "placements"),
        NotesBranchKey("ideas", "placements"),
        NotesBranchKey("reading", "placements"),
    )
    unrelated_key = NotesBranchKey("other", "placements")
    for index, key in enumerate((*affected_keys, unrelated_key)):
        fake._library_notes_tree_branches[key] = apply_notes_slice_page(
            begin_notes_slice_load(
                empty_notes_slice(key, topology_epoch=1),
                generation=1,
                direction="replace",
                requested_offset=0,
                requested_limit=20,
            ),
            _placement_page(key.parent_id, f"existing-{index}"),
            direction="replace",
            request_generation=1,
            topology_epoch=1,
        ).state

    await LibraryScreen._reconcile_library_notes_tree_mutation(
        fake,
        "note_create",
        {"note_id": "created"},
        before=None,
        result={"id": "created", "version": 1},
    )

    assert all(
        fake._library_notes_tree_branches[key].total is None
        and fake._library_notes_tree_branches[key].freshness == "stale"
        for key in affected_keys
    )
    assert fake._library_notes_tree_branches[unrelated_key].total == 1
    assert fake._library_notes_tree_branches[unrelated_key].freshness == "fresh"
    assert fake._library_notes_tree_pending_target_placement_id == (
        FolderPlacementId.unfiled("created")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("create_folder", "restore_folder"))
async def test_folder_create_and_restore_locator_reveal_off_window_committed_folder(
    operation: str,
) -> None:
    class _OffWindowFolderService(_MutationService):
        def __init__(self) -> None:
            super().__init__()
            self.folder_offsets: list[int] = []

        async def load_note_tree_mutation_context(self, **kwargs):
            folder_ids = tuple(kwargs["folder_ids"])
            return SimpleNamespace(
                parent_ids=(None,),
                placement_parent_ids=(),
                folder_ids=folder_ids,
                ancestor_ids=(),
            )

        async def create_note_folder(self, **kwargs):
            self.calls.append(("create", kwargs))
            return _folder("new", None, "/New")

        async def restore_note_folder(self, **kwargs):
            self.calls.append(("restore_folder", kwargs))
            return SimpleNamespace(
                folder=replace(_folder("new", None, "/New"), version=3),
                affected_folder_ids=("new",),
            )

        async def page_note_folder_children(self, **kwargs):
            offset = kwargs["offset"]
            self.folder_offsets.append(offset)
            if offset == 40:
                return _folder_page(None, "new", start=40, total=41, previous=20)
            return _folder_page(
                None,
                *(f"folder-{index:02d}" for index in range(20)),
                total=41,
                next_=20,
            )

        async def locate_note_tree_folder(self, **_kwargs):
            return NoteTreeLocation(
                placement_id=FolderPlacementId.folder("new"),
                note_id=None,
                membership_id=None,
                path=(NoteTreePathStep("new", None, 40),),
                placement_offset=None,
            )

    service = _OffWindowFolderService()
    fake = _mutation_fake(service)
    root_key = NotesBranchKey(None, "folders")
    first_page = await service.page_note_folder_children(offset=0)
    service.folder_offsets.clear()
    fake._library_notes_tree_branches[root_key] = apply_notes_slice_page(
        begin_notes_slice_load(
            empty_notes_slice(root_key, topology_epoch=1),
            generation=1,
            direction="replace",
            requested_offset=0,
            requested_limit=20,
        ),
        first_page,
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    ).state
    if operation == "restore_folder":
        fake._library_notes_deleted_folder_receipt = SimpleNamespace()
        payload = {"folder_id": "new", "expected_version": 2}
    else:
        payload = {"name": "New", "parent_id": None}

    ok = await LibraryScreen._execute_library_notes_tree_mutation(
        fake, operation, **payload
    )

    assert ok
    root = fake._library_notes_tree_branches[root_key]
    assert service.folder_offsets == [0, 40]
    assert root.start_offset == 40
    assert root.item_ids == (FolderPlacementId.folder("new"),)
    assert root.total == 41
    assert root.freshness == "fresh"
    assert fake._library_notes_tree_selected_placement_id == (
        FolderPlacementId.folder("new")
    )
    assert fake._library_notes_tree_pending_target_placement_id == ""


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


class _MountedCreateReconcileService(_ControlledMountedBranchService):
    def __init__(self, notes) -> None:
        super().__init__(notes)
        self.committed_note_id = ""
        self.context_calls: list[dict[str, object]] = []
        self.postcommit_placement_calls: list[str | None] = []

    async def save_note(self, **kwargs):
        result = await super().save_note(**kwargs)
        self.committed_note_id = str(
            result.get("id") if isinstance(result, dict) else result
        )
        return result

    async def load_note_tree_mutation_context(self, **kwargs):
        self.context_calls.append(kwargs)
        return SimpleNamespace(
            parent_ids=(),
            placement_parent_ids=("ideas", "reading"),
            folder_ids=(),
            ancestor_ids=(),
        )

    async def page_note_placements(self, **kwargs):
        parent_id = kwargs["parent_id"]
        if not self.committed_note_id:
            return await super().page_note_placements(**kwargs)
        self.postcommit_placement_calls.append(parent_id)
        if parent_id == "reading":
            raise RuntimeError("one exact parent refresh failed")
        if parent_id is None:
            return _placement_page(None, self.committed_note_id)
        if parent_id == "ideas":
            return NotePlacementPage(
                placements=(
                    NotePlacementRecord(
                        note={"id": self.committed_note_id, "title": "Created"},
                        folder_id="ideas",
                        membership=_membership(
                            "created-ideas", "ideas", self.committed_note_id
                        ),
                    ),
                ),
                total_placements=1,
                start_offset=0,
                previous_offset=None,
                next_offset=None,
            )
        raise AssertionError(f"unexpected broad refresh: {parent_id}")

    async def locate_note_tree_placement(self, **kwargs):
        assert kwargs["note_id"] == self.committed_note_id
        return NoteTreeLocation(
            placement_id=FolderPlacementId.unfiled(self.committed_note_id),
            note_id=self.committed_note_id,
            membership_id=None,
            path=(),
            placement_offset=0,
        )


class _MountedReceiptTopologyService(_ControlledMountedBranchService):
    def __init__(self, notes) -> None:
        super().__init__(notes)
        self.restoring = False
        self.removed = False
        self.block_locator = False
        self.locator_entered = asyncio.Event()
        self.locator_release = asyncio.Event()
        self.locator_calls: list[dict[str, object]] = []
        self.folder_offsets: list[int] = []
        self.placement_offsets: list[tuple[str | None, int]] = []

    async def locate_note_tree_placement(self, **kwargs):
        self.locator_calls.append(kwargs)
        if self.block_locator:
            self.locator_entered.set()
            await self.locator_release.wait()
        if self.removed:
            return None
        return NoteTreeLocation(
            placement_id=FolderPlacementId.note("target", "n1", "m-preferred"),
            note_id="n1",
            membership_id="m-preferred",
            path=(NoteTreePathStep("target", None, 40),),
            placement_offset=60,
        )

    async def page_note_folder_children(self, **kwargs):
        offset = kwargs["offset"]
        self.folder_offsets.append(offset)
        if offset == 40:
            return _folder_page(
                None,
                "fallback" if self.removed else "target",
                start=40,
                total=41,
                previous=20,
            )
        return await super().page_note_folder_children(**kwargs)

    async def page_note_placements(self, **kwargs):
        parent_id = kwargs["parent_id"]
        offset = kwargs["offset"]
        self.placement_offsets.append((parent_id, offset))
        if parent_id == "target" and self.removed:
            if offset:
                return NotePlacementPage(
                    placements=(),
                    total_placements=offset,
                    start_offset=offset,
                    previous_offset=max(0, offset - 20),
                    next_offset=None,
                )
            return NotePlacementPage(
                placements=(),
                total_placements=0,
                start_offset=0,
                previous_offset=None,
                next_offset=None,
            )
        if parent_id == "target" and offset == 60:
            records = (
                NotePlacementRecord(
                    note={
                        "id": "n1",
                        "title": "Reloaded target" if self.restoring else "Target",
                    },
                    folder_id="target",
                    membership=_membership("m-preferred", "target", "n1"),
                ),
                *(
                    NotePlacementRecord(
                        note={"id": f"extra-{index}", "title": f"Extra {index}"},
                        folder_id="target",
                        membership=_membership(
                            f"m-extra-{index}", "target", f"extra-{index}"
                        ),
                    )
                    for index in range(19)
                ),
            )
            return NotePlacementPage(
                placements=records,
                total_placements=80,
                start_offset=60,
                previous_offset=40,
                next_offset=None,
            )
        return await super().page_note_placements(**kwargs)


async def _capture_mounted_duplicate_receipt(screen, pilot, service):
    located = await screen._locate_library_notes_tree_target(
        note_id="n1",
        preferred_folder_id="target",
        preferred_membership_id="m-preferred",
        focus=True,
    )
    assert located
    selected = FolderPlacementId.note("target", "n1", "m-preferred")
    await _wait_until(
        pilot,
        lambda: screen._library_notes_tree_selected_placement_id == selected,
    )
    notes_list = screen._library_notes_scroll_owner("navigator")
    assert notes_list is not None
    notes_list.scroll_to(y=6, animate=False, force=True, immediate=True)
    await pilot.pause()
    receipt = screen._capture_library_notes_browse_return_receipt(
        note_id="n1", placement_id=selected
    )
    assert receipt.preferred_folder_id == "target"
    assert receipt.preferred_membership_id == "m-preferred"
    assert receipt.scroll_offset is not None
    assert receipt.scroll_offset[1] > 0
    assert any(descriptor.start_offset == 40 for descriptor in receipt.branch_ranges)
    assert any(descriptor.start_offset == 60 for descriptor in receipt.branch_ranges)
    service.locator_calls.clear()
    service.folder_offsets.clear()
    service.placement_offsets.clear()
    return receipt


@pytest.mark.asyncio
async def test_mounted_create_flow_fences_and_refreshes_every_exact_placement_parent():
    app = _build_test_app()
    notes = _two_notes()
    _seed_conversations(app, _two_conversations(), notes=notes)
    service = _MountedCreateReconcileService(notes)
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
        initial_epoch = screen._library_notes_tree_topology_epoch
        for parent_id in ("ideas", "reading", "unrelated"):
            key = NotesBranchKey(parent_id, "placements")
            screen._library_notes_tree_branches[key] = apply_notes_slice_page(
                begin_notes_slice_load(
                    empty_notes_slice(key, topology_epoch=initial_epoch),
                    generation=1,
                    direction="replace",
                    requested_offset=0,
                    requested_limit=20,
                ),
                _placement_page(parent_id, f"old-{parent_id}"),
                direction="replace",
                request_generation=1,
                topology_epoch=initial_epoch,
            ).state
        unrelated_key = NotesBranchKey("unrelated", "placements")
        unrelated = screen._library_notes_tree_branches[unrelated_key]
        service.postcommit_placement_calls.clear()

        outcome = await screen._create_library_note(
            title="Created",
            content="Committed body",
        )

        created_id = service.committed_note_id
        assert outcome.kind == "opened"
        assert created_id
        assert len(service.save_calls) == 1
        assert screen._library_notes_tree_topology_epoch == initial_epoch + 1
        assert service.context_calls[-1]["note_ids"] == (created_id,)
        assert set(service.postcommit_placement_calls) == {None, "ideas", "reading"}
        assert "unrelated" not in service.postcommit_placement_calls
        root = screen._library_notes_tree_branches[NotesBranchKey(None, "placements")]
        ideas = screen._library_notes_tree_branches[
            NotesBranchKey("ideas", "placements")
        ]
        reading = screen._library_notes_tree_branches[
            NotesBranchKey("reading", "placements")
        ]
        assert FolderPlacementId.unfiled(created_id) in root.item_ids
        assert root.total == 1 and root.freshness == "fresh"
        assert (
            FolderPlacementId.note("ideas", created_id, "created-ideas")
            in ideas.item_ids
        )
        assert ideas.total == 1 and ideas.freshness == "fresh"
        assert reading.total is None and reading.freshness == "stale"
        assert screen._library_notes_tree_branches[unrelated_key].items == (
            unrelated.items
        )
        assert screen._library_notes_tree_branches[unrelated_key].total == 1
        assert screen._library_notes_tree_branches[unrelated_key].freshness == "fresh"
        assert screen._library_notes_tree_selected_placement_id == (
            FolderPlacementId.unfiled(created_id)
        )
        assert screen._library_notes_tree_pending_target_placement_id == ""


@pytest.mark.asyncio
async def test_mounted_topology_changed_back_restores_exact_duplicate_ranges_and_scroll():
    app = _build_test_app()
    notes = _two_notes()
    _seed_conversations(app, _two_conversations(), notes=notes)
    service = _MountedReceiptTopologyService(notes)
    app.notes_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(120, 24)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        receipt = await _capture_mounted_duplicate_receipt(screen, pilot, service)
        original_generation = screen._library_notes_navigation_generation
        original_epoch = screen._library_notes_tree_topology_epoch
        service.restoring = True
        screen._library_notes_browse_return_receipt = receipt
        screen._library_notes_view = "editor"
        screen._selected_note_id = "n1"
        LibraryScreen._fence_library_notes_tree_mutation(screen)

        await screen.action_library_note_editor_back()
        await _wait_until(pilot, lambda: bool(service.locator_calls))
        selected = FolderPlacementId.note("target", "n1", "m-preferred")
        await _wait_until(
            pilot,
            lambda: (
                screen._library_notes_tree_selected_placement_id == selected
                and screen._library_notes_navigation_status == ""
                and all(
                    not state.loading
                    for state in screen._library_notes_tree_branches.values()
                )
            ),
        )

        locator_call = service.locator_calls[-1]
        assert locator_call["preferred_folder_id"] == "target"
        assert locator_call["preferred_membership_id"] == "m-preferred"
        assert screen._library_notes_tree_topology_epoch == original_epoch + 1
        assert screen._library_notes_navigation_generation > original_generation
        assert 40 in service.folder_offsets
        assert ("target", 60) in service.placement_offsets
        target_state = screen._library_notes_tree_branches[
            NotesBranchKey("target", "placements")
        ]
        assert target_state.start_offset == 60
        assert target_state.total == 80
        target = next(
            item
            for item in target_state.items
            if item.membership is not None
            and item.membership.membership_id == "m-preferred"
        )
        assert target.note["title"] == "Reloaded target"
        assert "target" in screen._library_notes_tree_expanded_ids
        assert screen._library_notes_tree_selected_placement_id == selected
        assert any(
            getattr(row, "placement_id", "") == selected and row is screen.focused
            for row in screen.query(".library-notes-row")
        )
        notes_list = screen._library_notes_scroll_owner("navigator")
        assert notes_list is not None
        assert int(notes_list.scroll_y) == receipt.scroll_offset[1]
        assert not screen.query("#library-notes-navigation-status")


@pytest.mark.asyncio
@pytest.mark.parametrize("abandon", (False, True))
async def test_mounted_topology_changed_removed_receipt_falls_back_without_focus_theft(
    abandon: bool,
) -> None:
    app = _build_test_app()
    notes = _two_notes()
    _seed_conversations(app, _two_conversations(), notes=notes)
    service = _MountedReceiptTopologyService(notes)
    app.notes_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(120, 24)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        receipt = await _capture_mounted_duplicate_receipt(screen, pilot, service)
        service.removed = True
        service.block_locator = abandon
        screen._library_notes_browse_return_receipt = receipt
        screen._library_notes_view = "editor"
        screen._selected_note_id = "n1"
        LibraryScreen._fence_library_notes_tree_mutation(screen)

        await screen.action_library_note_editor_back()
        if abandon:
            await _wait_until(pilot, service.locator_entered.is_set)
            filter_input = screen.query_one("#library-notes-filter")
            filter_input.focus()
            await _wait_until(pilot, lambda: screen.focused is filter_input)
            screen._supersede_library_notes_navigation()
            service.locator_release.set()
            await _wait_until(
                pilot,
                lambda: (
                    screen._library_notes_navigation_status == ""
                    and not screen.query("#library-notes-navigation-status")
                ),
            )
            assert getattr(screen.focused, "id", None) == "library-notes-filter"
            assert screen._library_notes_tree_selected_placement_id != (
                FolderPlacementId.folder("fallback")
            )
        else:
            await _wait_until(
                pilot,
                lambda: (
                    screen._library_notes_tree_selected_placement_id
                    == FolderPlacementId.folder("fallback")
                ),
            )
            assert screen._library_notes_navigation_status == ""
            assert not screen.query("#library-notes-navigation-status")
            unsettled = {
                key: (state.loading, state.error, state.freshness)
                for key, state in screen._library_notes_tree_branches.items()
                if state.loading or state.error
            }
            assert not unsettled, unsettled


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
async def test_mounted_editor_back_supersedes_blocked_locator_before_completion() -> (
    None
):
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
        await _wait_until(
            pilot,
            lambda: all(
                not state.loading
                for state in screen._library_notes_tree_branches.values()
            ),
        )
        screen._library_notes_view = "editor"
        screen._selected_note_id = "n1"
        task = asyncio.create_task(
            screen._locate_library_notes_tree_target(note_id="n1", focus=True)
        )
        await _wait_until(pilot, service.entered.is_set)

        await screen.action_library_note_editor_back()
        assert screen._library_notes_view == "list"
        assert screen._library_notes_navigation_status == ""

        service.release.set()
        assert not await task
        await pilot.pause()

        assert "target" not in screen._library_notes_tree_expanded_ids
        assert screen._library_notes_tree_selected_placement_id == ""
        assert all(
            not state.loading and not state.error
            for state in screen._library_notes_tree_branches.values()
        )
        assert not screen.query("#library-notes-navigation-status")


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
