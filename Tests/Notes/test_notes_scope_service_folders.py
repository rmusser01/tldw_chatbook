from __future__ import annotations

import json
import inspect
import threading
from collections.abc import Iterable

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.notes_organization_repository import NotesOrganizationRepository
from tldw_chatbook.Notes.note_folder_models import (
    FolderPlacementId,
    FolderCapabilityError,
    FolderMutationResult,
    NoteFolder,
    NoteFolderChildPage,
    NoteFolderMembership,
    NoteFolderPage,
    NotePlacementPage,
    NoteTreeLocation,
    NoteTreeMutationContext,
    NoteTreePathStep,
)
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService, ScopeType
from tldw_chatbook.Sync_Interop.notes_organization_sync_service import (
    NotesOrganizationSyncService,
)
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from tldw_chatbook.runtime_policy import PolicyDeniedError

FOLDER_OPERATIONS = [
    "list",
    "create",
    "rename",
    "move",
    "delete",
    "restore",
    "membership",
]


class RecordingFolderRepository:
    def __init__(self, events: list[tuple[object, ...]] | None = None) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.events = events
        self.thread_ids: list[int] = []
        self.folder_child_page = _empty_folder_child_page()
        self.placement_page = _empty_placement_page()
        self.folder_location = _folder_location()
        self.mutation_context = _mutation_context()

    def _record(self, call: tuple[object, ...]) -> None:
        self.thread_ids.append(threading.get_ident())
        self.calls.append(call)
        if self.events is not None:
            self.events.append(("repository", *call))

    def list_children(
        self, *, parent_id: str | None, limit: int, offset: int
    ) -> NoteFolderPage:
        self._record(("list_children", parent_id, limit, offset))
        return _empty_page()

    def page_child_folders(
        self, *, parent_id: str | None, limit: int, offset: int
    ) -> NoteFolderChildPage:
        self._record(
            (
                "page_child_folders",
                {"parent_id": parent_id, "limit": limit, "offset": offset},
            )
        )
        return self.folder_child_page

    def page_note_placements(
        self, *, parent_id: str | None, limit: int, offset: int
    ) -> NotePlacementPage:
        self._record(
            (
                "page_note_placements",
                {"parent_id": parent_id, "limit": limit, "offset": offset},
            )
        )
        return self.placement_page

    def locate_note_tree_folder(
        self, *, folder_id: str, page_size: int
    ) -> NoteTreeLocation | None:
        self._record(
            (
                "locate_note_tree_folder",
                {"folder_id": folder_id, "page_size": page_size},
            )
        )
        return self.folder_location

    def locate_note_tree_placement(
        self,
        *,
        note_id: str,
        page_size: int,
        preferred_folder_id: str | None = None,
        preferred_membership_id: str | None = None,
    ) -> NoteTreeLocation | None:
        self._record(
            (
                "locate_note_tree_placement",
                {
                    "note_id": note_id,
                    "page_size": page_size,
                    "preferred_folder_id": preferred_folder_id,
                    "preferred_membership_id": preferred_membership_id,
                },
            )
        )
        return self.folder_location

    def load_note_tree_mutation_context(
        self,
        *,
        folder_ids: Iterable[str] = (),
        note_ids: Iterable[str] = (),
        include_folder_subtrees: bool = False,
    ) -> NoteTreeMutationContext:
        self._record(
            (
                "load_note_tree_mutation_context",
                {
                    "folder_ids": folder_ids,
                    "note_ids": note_ids,
                    "include_folder_subtrees": include_folder_subtrees,
                },
            )
        )
        return self.mutation_context

    def search_note_tree_placements(
        self, *, query: str, limit: int, offset: int
    ) -> NotePlacementPage:
        self._record(
            (
                "search_note_tree_placements",
                {"query": query, "limit": limit, "offset": offset},
            )
        )
        return self.placement_page

    def load_tree_batch(
        self,
        *,
        expanded_folder_ids: tuple[str, ...],
        note_limit: int,
        note_offset: int,
        folder_limit: int,
        folder_offset: int,
        membership_limit: int,
        membership_offset: int,
        load_notes: bool,
    ) -> NoteFolderPage:
        self._record(
            (
                "load_tree_batch",
                expanded_folder_ids,
                note_limit,
                note_offset,
                folder_limit,
                folder_offset,
                membership_limit,
                membership_offset,
                load_notes,
            )
        )
        return _empty_page()

    def load_tree_search(
        self, *, note_ids: tuple[str, ...], folder_query: str
    ) -> NoteFolderPage:
        self._record(("load_tree_search", note_ids, folder_query))
        return _empty_page()

    def create_folder(self, *, name: str, parent_id: str | None) -> NoteFolder:
        self._record(("create_folder", name, parent_id))
        return _folder()

    def rename_folder(
        self, folder_id: str, *, name: str, expected_version: int
    ) -> FolderMutationResult:
        self._record(("rename_folder", folder_id, name, expected_version))
        return _mutation()

    def move_folder(
        self,
        folder_id: str,
        *,
        parent_id: str | None,
        expected_version: int,
    ) -> FolderMutationResult:
        self._record(("move_folder", folder_id, parent_id, expected_version))
        return _mutation()

    def soft_delete_folder(
        self, folder_id: str, *, expected_version: int
    ) -> FolderMutationResult:
        self._record(("soft_delete_folder", folder_id, expected_version))
        return _mutation()

    def restore_folder(
        self, folder_id: str, *, expected_version: int
    ) -> FolderMutationResult:
        self._record(("restore_folder", folder_id, expected_version))
        return _mutation()

    def attach_manual(self, *, folder_id: str, note_id: str) -> NoteFolderMembership:
        self._record(("attach_manual", folder_id, note_id))
        return NoteFolderMembership(
            membership_id="membership-1",
            folder_id=folder_id,
            note_id=note_id,
            ownership="manual",
            owner_id="",
            owner_active=True,
            version=1,
        )

    def detach_manual(
        self, *, folder_id: str, note_id: str, expected_version: int
    ) -> bool:
        self._record(("detach_manual", folder_id, note_id, expected_version))
        return True

    def convert_owner_to_manual(self, *, owner_id: str) -> int:
        self._record(("convert_owner_to_manual", owner_id))
        return 2

    def remove_owner_memberships(self, *, owner_id: str) -> int:
        self._record(("remove_owner_memberships", owner_id))
        return 2

    def list_restore_reviews(self) -> tuple[object, ...]:
        self._record(("list_restore_reviews",))
        return ()

    def reconcile_managed(
        self,
        *,
        owner_id: str,
        desired: tuple[tuple[str, str], ...],
    ) -> tuple[NoteFolderMembership, ...]:
        self._record(("reconcile_managed", owner_id, desired))
        return ()


class RecordingPolicy:
    def __init__(
        self,
        events: list[tuple[object, ...]],
        *,
        denied_reason: str | None = None,
    ) -> None:
        self.events = events
        self.denied_reason = denied_reason
        self.thread_ids: list[int] = []

    def require_allowed(self, *, action_id: str) -> None:
        self.thread_ids.append(threading.get_ident())
        self.events.append(("policy", action_id))
        if self.denied_reason is not None:
            raise PolicyDeniedError(
                action_id=action_id,
                reason_code=self.denied_reason,
                user_message="Folder operation denied.",
                effective_source="local",
                authority_owner="server",
            )


class NoCallBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> object:
        self.calls.append(name)
        raise AssertionError(f"Unsupported folder operation reached backend {name}.")


class RecordingNotesSyncV2Producer:
    def __init__(self) -> None:
        self.upserts: list[dict[str, object]] = []
        self.deletes: list[dict[str, object]] = []

    def enqueue_note_upsert(self, **kwargs: object) -> None:
        self.upserts.append(kwargs)

    def enqueue_note_delete(self, **kwargs: object) -> None:
        self.deletes.append(kwargs)


def _folder() -> NoteFolder:
    return NoteFolder(
        folder_id="folder-1",
        parent_id=None,
        name="Folder",
        path="/Folder",
        normalized_path="/folder",
        version=1,
        deleted=False,
    )


def _mutation() -> FolderMutationResult:
    return FolderMutationResult(folder=_folder(), affected_folder_ids=("folder-1",))


def _empty_page() -> NoteFolderPage:
    return NoteFolderPage(
        folders=(),
        memberships=(),
        notes=(),
        total_folders=0,
        total_notes=0,
        next_offset=None,
        next_folder_offset=None,
    )


def _empty_folder_child_page() -> NoteFolderChildPage:
    return NoteFolderChildPage(
        folders=(),
        total_folders=0,
        start_offset=0,
        previous_offset=None,
        next_offset=None,
    )


def _empty_placement_page() -> NotePlacementPage:
    return NotePlacementPage(
        placements=(),
        total_placements=0,
        start_offset=0,
        previous_offset=None,
        next_offset=None,
    )


def _folder_location() -> NoteTreeLocation:
    return NoteTreeLocation(
        placement_id=FolderPlacementId.folder("folder-1"),
        note_id=None,
        membership_id=None,
        path=(
            NoteTreePathStep(
                folder_id="folder-1",
                parent_id=None,
                containing_offset=0,
            ),
        ),
        placement_offset=None,
    )


def _mutation_context() -> NoteTreeMutationContext:
    return NoteTreeMutationContext(
        folder_ids=("folder-1",),
        parent_ids=(None,),
        ancestor_ids=(),
        placement_parent_ids=("folder-1",),
    )


@pytest.mark.asyncio
async def test_sync_managed_membership_reconciliation_uses_scope_service_boundary() -> (
    None
):
    events: list[tuple[object, ...]] = []
    repository = RecordingFolderRepository(events)
    policy = RecordingPolicy(events)
    service = NotesScopeService(
        NoCallBackend(),
        NoCallBackend(),
        policy_enforcer=policy,
        folder_repository=repository,
    )

    result = await service.reconcile_note_folder_owner_memberships(
        scope=ScopeType.LOCAL_NOTE,
        owner_id="root-1",
        desired=(("folder-1", "note-1"),),
        user_id="user-1",
    )

    assert result == ()
    assert events == [
        ("policy", "notes.update.local"),
        ("repository", "reconcile_managed", "root-1", (("folder-1", "note-1"),)),
    ]


LOCAL_FOLDER_CASES = [
    (
        "list_note_folder_children",
        {"parent_id": None, "limit": 25, "offset": 5},
        "notes.list.local",
        ("list_children", None, 25, 5),
    ),
    (
        "load_note_folder_tree_batch",
        {
            "expanded_folder_ids": ("folder-1",),
            "note_limit": 100,
            "note_offset": 20,
            "folder_limit": 40,
            "folder_offset": 10,
            "membership_limit": 60,
            "membership_offset": 30,
            "load_notes": False,
        },
        "notes.list.local",
        ("load_tree_batch", ("folder-1",), 100, 20, 40, 10, 60, 30, False),
    ),
    (
        "load_note_folder_search",
        {"note_ids": ("note-1", "note-2"), "folder_query": "work"},
        "notes.list.local",
        ("load_tree_search", ("note-1", "note-2"), "work"),
    ),
    (
        "create_note_folder",
        {"name": "Folder", "parent_id": None},
        "notes.create.local",
        ("create_folder", "Folder", None),
    ),
    (
        "rename_note_folder",
        {"folder_id": "folder-1", "name": "Renamed", "expected_version": 1},
        "notes.update.local",
        ("rename_folder", "folder-1", "Renamed", 1),
    ),
    (
        "move_note_folder",
        {"folder_id": "folder-1", "parent_id": "folder-2", "expected_version": 1},
        "notes.update.local",
        ("move_folder", "folder-1", "folder-2", 1),
    ),
    (
        "delete_note_folder",
        {"folder_id": "folder-1", "expected_version": 1},
        "notes.delete.local",
        ("soft_delete_folder", "folder-1", 1),
    ),
    (
        "restore_note_folder",
        {"folder_id": "folder-1", "expected_version": 2},
        "notes.update.local",
        ("restore_folder", "folder-1", 2),
    ),
    (
        "attach_note_to_folder",
        {"folder_id": "folder-1", "note_id": "note-1"},
        "notes.update.local",
        ("attach_manual", "folder-1", "note-1"),
    ),
    (
        "detach_note_from_folder",
        {"folder_id": "folder-1", "note_id": "note-1", "expected_version": 1},
        "notes.update.local",
        ("detach_manual", "folder-1", "note-1", 1),
    ),
    (
        "convert_note_folder_owner_to_manual",
        {"owner_id": "owner-1"},
        "notes.update.local",
        ("convert_owner_to_manual", "owner-1"),
    ),
    (
        "remove_note_folder_owner_memberships",
        {"owner_id": "owner-1"},
        "notes.update.local",
        ("remove_owner_memberships", "owner-1"),
    ),
    (
        "list_note_folder_restore_reviews",
        {},
        "notes.list.local",
        ("list_restore_reviews",),
    ),
    pytest.param(
        "page_note_folder_children",
        {"parent_id": None, "limit": 25, "offset": 5},
        "notes.list.local",
        (
            "page_child_folders",
            {"parent_id": None, "limit": 25, "offset": 5},
        ),
        id="branch_page_folder",
    ),
    pytest.param(
        "page_note_placements",
        {"parent_id": "folder-1", "limit": 10, "offset": 20},
        "notes.list.local",
        (
            "page_note_placements",
            {"parent_id": "folder-1", "limit": 10, "offset": 20},
        ),
        id="branch_page_placements",
    ),
    pytest.param(
        "locate_note_tree_folder",
        {"folder_id": "folder-1", "page_size": 25},
        "notes.list.local",
        (
            "locate_note_tree_folder",
            {"folder_id": "folder-1", "page_size": 25},
        ),
        id="tree_locator_folder",
    ),
    pytest.param(
        "locate_note_tree_placement",
        {
            "note_id": "note-1",
            "page_size": 50,
            "preferred_folder_id": "folder-1",
            "preferred_membership_id": "membership-1",
        },
        "notes.list.local",
        (
            "locate_note_tree_placement",
            {
                "note_id": "note-1",
                "page_size": 50,
                "preferred_folder_id": "folder-1",
                "preferred_membership_id": "membership-1",
            },
        ),
        id="tree_locator_placement_preferences",
    ),
    pytest.param(
        "locate_note_tree_placement",
        {"note_id": "note-1", "page_size": 50},
        "notes.list.local",
        (
            "locate_note_tree_placement",
            {
                "note_id": "note-1",
                "page_size": 50,
                "preferred_folder_id": None,
                "preferred_membership_id": None,
            },
        ),
        id="tree_locator_placement_default_preferences",
    ),
    pytest.param(
        "load_note_tree_mutation_context",
        {
            "folder_ids": ("folder-1", "folder-2"),
            "note_ids": ("note-1",),
            "include_folder_subtrees": True,
        },
        "notes.list.local",
        (
            "load_note_tree_mutation_context",
            {
                "folder_ids": ("folder-1", "folder-2"),
                "note_ids": ("note-1",),
                "include_folder_subtrees": True,
            },
        ),
        id="affected_parent_context",
    ),
    pytest.param(
        "load_note_tree_mutation_context",
        {},
        "notes.list.local",
        (
            "load_note_tree_mutation_context",
            {
                "folder_ids": (),
                "note_ids": (),
                "include_folder_subtrees": False,
            },
        ),
        id="affected_parent_context_defaults",
    ),
    pytest.param(
        "search_note_tree_placements",
        {"query": "project alpha", "limit": 30, "offset": 60},
        "notes.list.local",
        (
            "search_note_tree_placements",
            {"query": "project alpha", "limit": 30, "offset": 60},
        ),
        id="placement_filter_search",
    ),
]


NEW_TREE_SERVICE_METHODS = {
    "page_note_folder_children": "page_child_folders",
    "page_note_placements": "page_note_placements",
    "locate_note_tree_folder": "locate_note_tree_folder",
    "locate_note_tree_placement": "locate_note_tree_placement",
    "load_note_tree_mutation_context": "load_note_tree_mutation_context",
    "search_note_tree_placements": "search_note_tree_placements",
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "kwargs", "_expected_action", "expected_call"),
    [
        case
        for case in LOCAL_FOLDER_CASES
        if getattr(case, "values", case)[0] in NEW_TREE_SERVICE_METHODS
    ],
)
async def test_branch_page_tree_locator_affected_parent_and_placement_filter_results_pass_through(
    method_name: str,
    kwargs: dict[str, object],
    _expected_action: str,
    expected_call: tuple[object, ...],
) -> None:
    repository = RecordingFolderRepository()
    service = NotesScopeService(
        local_notes_service=object(),
        server_service=object(),
        folder_repository=repository,
    )

    result = await getattr(service, method_name)(
        scope=ScopeType.LOCAL_NOTE,
        user_id="local-user",
        **kwargs,
    )

    repository_method = NEW_TREE_SERVICE_METHODS[method_name]
    expected_result = {
        "page_child_folders": repository.folder_child_page,
        "page_note_placements": repository.placement_page,
        "locate_note_tree_folder": repository.folder_location,
        "locate_note_tree_placement": repository.folder_location,
        "load_note_tree_mutation_context": repository.mutation_context,
        "search_note_tree_placements": repository.placement_page,
    }[repository_method]
    assert result is expected_result
    assert repository.calls == [expected_call]


def test_affected_parent_context_service_defaults_are_immutable() -> None:
    signature = inspect.signature(NotesScopeService.load_note_tree_mutation_context)

    assert signature.parameters["folder_ids"].default == ()
    assert signature.parameters["note_ids"].default == ()
    assert signature.parameters["include_folder_subtrees"].default is False


@pytest.mark.asyncio
async def test_list_folder_children_routes_to_local_repository() -> None:
    repository = RecordingFolderRepository()
    service = NotesScopeService(
        local_notes_service=object(),
        server_service=object(),
        folder_repository=repository,
    )

    result = await service.list_note_folder_children(
        scope=ScopeType.LOCAL_NOTE,
        parent_id=None,
        limit=50,
        offset=0,
        user_id="local-user",
    )

    assert result == NoteFolderPage(
        folders=(),
        memberships=(),
        notes=(),
        total_folders=0,
        total_notes=0,
        next_offset=None,
        next_folder_offset=None,
    )
    assert repository.calls == [("list_children", None, 50, 0)]


@pytest.mark.parametrize(
    ("scope", "has_repository", "supported", "reason_code"),
    [
        (ScopeType.LOCAL_NOTE, True, True, ""),
        (ScopeType.LOCAL_NOTE, False, False, "local_store_missing"),
        (ScopeType.SERVER_NOTE, True, False, "server_contract_missing"),
        (ScopeType.WORKSPACE, True, False, "scope_not_supported"),
        ("file_note", True, False, "scope_not_supported"),
    ],
)
def test_note_folder_capabilities_are_complete_and_scope_aware(
    scope: ScopeType | str,
    has_repository: bool,
    supported: bool,
    reason_code: str,
) -> None:
    service = NotesScopeService(
        local_notes_service=object(),
        server_service=object(),
        folder_repository=RecordingFolderRepository() if has_repository else None,
    )

    capabilities = service.note_folder_capabilities(scope=scope)

    assert [item.operation for item in capabilities] == FOLDER_OPERATIONS
    assert [item.supported for item in capabilities] == [supported] * 7
    assert [item.reason_code for item in capabilities] == [reason_code] * 7
    if not supported:
        assert all(item.user_message for item in capabilities)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "kwargs", "expected_action", "expected_call"),
    LOCAL_FOLDER_CASES,
)
async def test_local_folder_methods_enforce_policy_before_exact_repository_call(
    method_name: str,
    kwargs: dict[str, object],
    expected_action: str,
    expected_call: tuple[object, ...],
) -> None:
    events: list[tuple[object, ...]] = []
    repository = RecordingFolderRepository(events)
    service = NotesScopeService(
        local_notes_service=object(),
        server_service=object(),
        policy_enforcer=RecordingPolicy(events),
        folder_repository=repository,
    )

    await getattr(service, method_name)(
        scope=ScopeType.LOCAL_NOTE,
        user_id="local-user",
        **kwargs,
    )

    assert events == [
        ("policy", expected_action),
        ("repository", *expected_call),
    ]
    assert repository.calls == [expected_call]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "kwargs", "_expected_action", "_expected_call"),
    LOCAL_FOLDER_CASES,
)
async def test_local_folder_repository_calls_run_off_the_event_loop_thread(
    method_name: str,
    kwargs: dict[str, object],
    _expected_action: str,
    _expected_call: tuple[object, ...],
) -> None:
    event_loop_thread_id = threading.get_ident()
    events: list[tuple[object, ...]] = []
    repository = RecordingFolderRepository(events)
    policy = RecordingPolicy(events)
    service = NotesScopeService(
        local_notes_service=object(),
        server_service=object(),
        policy_enforcer=policy,
        folder_repository=repository,
    )

    await getattr(service, method_name)(
        scope=ScopeType.LOCAL_NOTE,
        user_id="local-user",
        **kwargs,
    )

    assert policy.thread_ids == [event_loop_thread_id]
    assert len(repository.thread_ids) == 1
    assert repository.thread_ids[0] != event_loop_thread_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "kwargs", "expected_action", "_expected_call"),
    LOCAL_FOLDER_CASES,
)
async def test_denied_local_folder_policy_prevents_repository_call(
    method_name: str,
    kwargs: dict[str, object],
    expected_action: str,
    _expected_call: tuple[object, ...],
) -> None:
    events: list[tuple[object, ...]] = []
    repository = RecordingFolderRepository(events)
    service = NotesScopeService(
        local_notes_service=object(),
        server_service=object(),
        policy_enforcer=RecordingPolicy(events, denied_reason="folder_denied"),
        folder_repository=repository,
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await getattr(service, method_name)(
            scope=ScopeType.LOCAL_NOTE,
            user_id="local-user",
            **kwargs,
        )

    assert exc.value.reason_code == "folder_denied"
    assert events == [("policy", expected_action)]
    assert repository.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "kwargs", "_expected_action", "_expected_call"),
    LOCAL_FOLDER_CASES,
)
async def test_local_folder_methods_require_nonempty_user_id(
    method_name: str,
    kwargs: dict[str, object],
    _expected_action: str,
    _expected_call: tuple[object, ...],
) -> None:
    events: list[tuple[object, ...]] = []
    repository = RecordingFolderRepository(events)
    service = NotesScopeService(
        local_notes_service=object(),
        server_service=object(),
        policy_enforcer=RecordingPolicy(events),
        folder_repository=repository,
    )

    with pytest.raises(ValueError, match="user_id is required"):
        await getattr(service, method_name)(
            scope=ScopeType.LOCAL_NOTE,
            user_id="",
            **kwargs,
        )

    assert events == []
    assert repository.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "kwargs", "_expected_action", "_expected_call"),
    LOCAL_FOLDER_CASES,
)
async def test_local_folder_methods_fail_closed_when_repository_is_missing(
    method_name: str,
    kwargs: dict[str, object],
    _expected_action: str,
    _expected_call: tuple[object, ...],
) -> None:
    service = NotesScopeService(
        local_notes_service=NoCallBackend(),
        server_service=NoCallBackend(),
    )
    capability = next(
        item
        for item in service.note_folder_capabilities(scope=ScopeType.LOCAL_NOTE)
        if item.operation
        == {
            "list_note_folder_children": "list",
            "page_note_folder_children": "list",
            "page_note_placements": "list",
            "locate_note_tree_folder": "list",
            "locate_note_tree_placement": "list",
            "load_note_tree_mutation_context": "list",
            "search_note_tree_placements": "list",
            "load_note_folder_tree_batch": "list",
            "load_note_folder_search": "list",
            "create_note_folder": "create",
            "rename_note_folder": "rename",
            "move_note_folder": "move",
            "delete_note_folder": "delete",
            "restore_note_folder": "restore",
        }.get(method_name, "membership")
    )

    with pytest.raises(FolderCapabilityError) as exc:
        await getattr(service, method_name)(
            scope=ScopeType.LOCAL_NOTE,
            user_id="local-user",
            **kwargs,
        )

    assert exc.value.reason_code == "local_store_missing"
    assert exc.value.user_message == capability.user_message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "reason_code"),
    [
        (ScopeType.SERVER_NOTE, "server_contract_missing"),
        (ScopeType.WORKSPACE, "scope_not_supported"),
        ("file_note", "scope_not_supported"),
    ],
)
@pytest.mark.parametrize(
    ("method_name", "kwargs", "_expected_action", "_expected_call"),
    LOCAL_FOLDER_CASES,
)
async def test_unsupported_folder_scopes_fail_closed_without_backend_calls(
    method_name: str,
    kwargs: dict[str, object],
    _expected_action: str,
    _expected_call: tuple[object, ...],
    scope: ScopeType | str,
    reason_code: str,
) -> None:
    local_backend = NoCallBackend()
    server_backend = NoCallBackend()
    repository = RecordingFolderRepository()
    service = NotesScopeService(
        local_notes_service=local_backend,
        server_service=server_backend,
        folder_repository=repository,
    )
    operation = {
        "list_note_folder_children": "list",
        "page_note_folder_children": "list",
        "page_note_placements": "list",
        "locate_note_tree_folder": "list",
        "locate_note_tree_placement": "list",
        "load_note_tree_mutation_context": "list",
        "search_note_tree_placements": "list",
        "load_note_folder_tree_batch": "list",
        "load_note_folder_search": "list",
        "create_note_folder": "create",
        "rename_note_folder": "rename",
        "move_note_folder": "move",
        "delete_note_folder": "delete",
        "restore_note_folder": "restore",
    }.get(method_name, "membership")
    capability = next(
        item
        for item in service.note_folder_capabilities(scope=scope)
        if item.operation == operation
    )

    with pytest.raises(FolderCapabilityError) as exc:
        await getattr(service, method_name)(
            scope=scope,
            user_id="local-user",
            **kwargs,
        )

    assert exc.value.reason_code == reason_code
    assert exc.value.user_message == capability.user_message
    assert local_backend.calls == []
    assert server_backend.calls == []
    assert repository.calls == []


@pytest.mark.parametrize("has_database", [True, False])
def test_app_builds_notes_scope_service_with_one_shared_folder_repository(
    monkeypatch: pytest.MonkeyPatch,
    has_database: bool,
) -> None:
    import tldw_chatbook.app as app_module

    database = object() if has_database else None
    local_notes_service = object()
    server_service = object()
    policy_enforcer = object()
    sync_scope_service = object()
    repositories: list[object] = []
    service_calls: list[dict[str, object]] = []

    def build_repository(db: object) -> object:
        assert db is database
        repository = object()
        repositories.append(repository)
        return repository

    def build_service(**kwargs: object) -> dict[str, object]:
        service_calls.append(kwargs)
        return kwargs

    monkeypatch.setattr(app_module, "LocalNoteFolderRepository", build_repository)
    monkeypatch.setattr(app_module, "NotesScopeService", build_service)

    result = app_module._build_notes_scope_service(
        chachanotes_db=database,
        local_notes_service=local_notes_service,
        server_service=server_service,
        policy_enforcer=policy_enforcer,
        sync_scope_service=sync_scope_service,
    )

    assert len(repositories) == int(has_database)
    assert service_calls == [
        {
            "local_notes_service": local_notes_service,
            "server_service": server_service,
            "policy_enforcer": policy_enforcer,
            "sync_scope_service": sync_scope_service,
            "folder_repository": repositories[0] if has_database else None,
        }
    ]
    assert result is service_calls[0]


@pytest.mark.asyncio
async def test_local_folder_mutations_do_not_enter_sync_v2_note_outbox() -> None:
    producer = RecordingNotesSyncV2Producer()
    repository = RecordingFolderRepository()
    service = NotesScopeService(
        local_notes_service=object(),
        server_service=object(),
        sync_v2_notes_producer=producer,
        folder_repository=repository,
    )

    await service.create_note_folder(
        scope=ScopeType.LOCAL_NOTE,
        name="Folder",
        parent_id=None,
        user_id="local-user",
    )
    await service.rename_note_folder(
        scope=ScopeType.LOCAL_NOTE,
        folder_id="folder-1",
        name="Renamed",
        expected_version=1,
        user_id="local-user",
    )
    await service.attach_note_to_folder(
        scope=ScopeType.LOCAL_NOTE,
        folder_id="folder-1",
        note_id="note-1",
        user_id="local-user",
    )
    await service.delete_note_folder(
        scope=ScopeType.LOCAL_NOTE,
        folder_id="folder-1",
        expected_version=2,
        user_id="local-user",
    )

    assert producer.upserts == []
    assert producer.deletes == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "local_state", ["initializing", "pulling", "adoption_review", "failed"]
)
async def test_synchronized_folder_create_rejects_until_group_ready(
    tmp_path, local_state: str
) -> None:
    notes = CharactersRAGDB(tmp_path / f"{local_state}.sqlite", client_id="folders")
    state = SyncStateRepository(tmp_path / f"{local_state}-sync.sqlite")
    state.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id=None,
        workspace_scope=None,
        profile_mode="local_first",
        device_id="device-a",
        dataset_id="dataset-a",
    )
    with notes.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO notes_organization_sync_checkpoints(
                server_profile_id, dataset_id, local_state, server_state,
                inventory_phase, updated_at
            ) VALUES ('server-a', 'dataset-a', ?, 'ready', 'complete',
                      '2026-08-29T00:00:00+00:00')
            """,
            (local_state,),
        )
    repository = LocalNoteFolderRepository(notes)
    organization = NotesOrganizationSyncService(
        notes_repository=NotesOrganizationRepository(notes, server_profile_id="server-a"),
        state_repository=state,
    )
    service = NotesScopeService(
        local_notes_service=object(),
        server_service=object(),
        folder_repository=repository,
        organization_sync_service=organization,
    )

    with pytest.raises(ValueError, match="organization group is not ready"):
        await service.create_note_folder(
            scope=ScopeType.LOCAL_NOTE,
            name="Blocked",
            parent_id=None,
            user_id="local-user",
            sync_v2_profile={"server_profile_id": "server-a"},
        )

    assert repository.list_children(parent_id=None, limit=10, offset=0).folders == ()
    assert notes.get_connection().execute(
        "SELECT COUNT(*) FROM notes_organization_sync_intents"
    ).fetchone()[0] == 0


@pytest.mark.asyncio
async def test_ready_folder_create_commits_mutation_and_one_explicit_intent(
    tmp_path,
) -> None:
    notes = CharactersRAGDB(tmp_path / "ready.sqlite", client_id="folders")
    state = SyncStateRepository(tmp_path / "ready-sync.sqlite")
    state.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id=None,
        workspace_scope=None,
        profile_mode="local_first",
        device_id="device-a",
        dataset_id="dataset-a",
    )
    with notes.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO notes_organization_sync_checkpoints(
                server_profile_id, dataset_id, local_state, server_state,
                inventory_phase, updated_at
            ) VALUES ('server-a', 'dataset-a', 'ready', 'ready', 'complete',
                      '2026-08-29T00:00:00+00:00')
            """
        )
    repository = LocalNoteFolderRepository(notes)
    service = NotesScopeService(
        local_notes_service=object(),
        server_service=object(),
        folder_repository=repository,
        organization_sync_service=NotesOrganizationSyncService(
            notes_repository=NotesOrganizationRepository(
                notes, server_profile_id="server-a"
            ),
            state_repository=state,
        ),
    )

    folder = await service.create_note_folder(
        scope=ScopeType.LOCAL_NOTE,
        name="Portable",
        parent_id=None,
        user_id="local-user",
        sync_v2_profile={"server_profile_id": "server-a"},
    )

    row = notes.get_connection().execute(
        "SELECT domain, object_id, operation, payload_json, source_version "
        "FROM notes_organization_sync_intents"
    ).fetchone()
    sync_id = notes.get_connection().execute(
        "SELECT sync_id FROM note_folders WHERE id = ?", (folder.folder_id,)
    ).fetchone()[0]
    assert tuple(row) == (
        "notes.folder",
        sync_id,
        "upsert",
        '{"name":"Portable","parent_sync_id":null}',
        1,
    )


@pytest.mark.asyncio
async def test_ready_scope_managed_membership_routes_through_organization_owner(
    tmp_path,
) -> None:
    notes = CharactersRAGDB(tmp_path / "membership.sqlite", client_id="folders")
    state = SyncStateRepository(tmp_path / "membership-sync.sqlite")
    state.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id=None,
        workspace_scope=None,
        profile_mode="local_first",
        device_id="device-a",
        dataset_id="dataset-a",
    )
    with notes.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO notes_organization_sync_checkpoints(
                server_profile_id, dataset_id, local_state, server_state,
                inventory_phase, updated_at
            ) VALUES ('server-a', 'dataset-a', 'ready', 'ready', 'complete',
                      '2026-08-29T00:00:00+00:00')
            """
        )
    repository = LocalNoteFolderRepository(notes)
    service = NotesScopeService(
        local_notes_service=object(),
        server_service=object(),
        folder_repository=repository,
        organization_sync_service=NotesOrganizationSyncService(
            notes_repository=NotesOrganizationRepository(
                notes, server_profile_id="server-a"
            ),
            state_repository=state,
        ),
    )
    sync_profile = {"server_profile_id": "server-a"}
    folder = await service.create_note_folder(
        scope=ScopeType.LOCAL_NOTE,
        name="Managed",
        parent_id=None,
        user_id="local-user",
        sync_v2_profile=sync_profile,
    )
    note_id = notes.add_note("Note", "Body")

    memberships = await service.reconcile_note_folder_owner_memberships(
        scope=ScopeType.LOCAL_NOTE,
        owner_id="source-a",
        desired=((folder.folder_id, note_id),),
        user_id="local-user",
        sync_v2_profile=sync_profile,
    )

    assert len(memberships) == 1
    row = notes.get_connection().execute(
        "SELECT operation, payload_json FROM notes_organization_sync_intents "
        "WHERE domain = 'notes.folder_link'"
    ).fetchone()
    assert row["operation"] == "upsert"
    assert json.loads(row["payload_json"])["note_id"] == note_id
