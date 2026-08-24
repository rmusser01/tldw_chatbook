from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.Research_Workspace import (
    CapabilityUnavailableError,
    BoundedPageResult,
    QualifiedWorkspaceRef,
    ResearchNoteConflictError,
    ResearchNotePageRequest,
    ResearchNoteSaveRequest,
    ResearchQuickNote,
    ResearchQuickNotesService,
    ResearchWorkspaceController,
    WorkspaceDataSource,
)
from tldw_chatbook.Research_Workspace.local_adapter import (
    LocalResearchWorkspaceAdapter,
)
from tldw_chatbook.Research_Workspace.server_adapter import (
    ServerResearchWorkspaceAdapter,
)
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService


LOCAL_REF = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-local")
SERVER_REF = QualifiedWorkspaceRef(
    WorkspaceDataSource.SERVER,
    "workspace-server",
    server_profile_id="profile-1",
    principal_id=(
        "credential-fingerprint:test:" + hashlib.sha256(b"test-token").hexdigest()[:24]
    ),
)


class RecordingRegistry:
    def __init__(self) -> None:
        self.memberships = [
            SimpleNamespace(item_type="note", item_id="note-2", role="note"),
            SimpleNamespace(item_type="note", item_id="note-1", role="note"),
        ]
        self.calls: list[tuple[object, ...]] = []

    def list_workspace_note_memberships(self, workspace_id, *, limit, offset):
        self.calls.append(("list", workspace_id, limit, offset))
        page = self.memberships[offset : offset + limit]
        return tuple(page), len(self.memberships)

    def link_membership(self, workspace_id, **kwargs):
        self.calls.append(("link", workspace_id, kwargs))
        self.memberships.append(
            SimpleNamespace(
                item_type=kwargs["item_type"],
                item_id=kwargs["item_id"],
                role=kwargs["role"],
            )
        )
        return self.memberships[-1]

    def unlink_membership(self, workspace_id, **kwargs):
        self.calls.append(("unlink", workspace_id, kwargs))
        self.memberships = [
            item
            for item in self.memberships
            if not (
                item.item_type == kwargs["item_type"]
                and item.item_id == kwargs["item_id"]
                and item.role == kwargs["role"]
            )
        ]
        return True

    def get_item_memberships(self, item_type, item_id):
        self.calls.append(("get_memberships", item_type, item_id))
        return tuple(
            SimpleNamespace(workspace_id="workspace-local", role=item.role)
            for item in self.memberships
            if item.item_type == item_type and item.item_id == item_id
        )


class RecordingNotesScope:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.rows = {
            "note-1": {
                "id": "note-1",
                "title": "First",
                "content": "Alpha body",
                "version": 3,
                "last_modified": "2026-08-24T08:00:00Z",
                "keywords": ["alpha"],
            },
            "note-2": {
                "id": "note-2",
                "title": "Second",
                "content": "Beta body",
                "version": 2,
                "last_modified": "2026-08-24T09:00:00Z",
                "keywords": [
                    "beta",
                    "research-message-id:bWVzc2FnZS0x",
                    "research-source-id:c291cmNlLTE",
                ],
            },
        }
        self.conflict = False

    async def get_note_detail(self, **kwargs):
        self.calls.append(("get", kwargs))
        return self.rows.get(str(kwargs["note_id"]))

    async def save_note(self, **kwargs):
        self.calls.append(("save", kwargs))
        if self.conflict:
            raise ConflictError("stale title only", entity="notes", entity_id="note-1")
        note_id = str(kwargs.get("note_id") or "note-new")
        version = int(kwargs.get("version") or 0) + 1
        row = {
            "id": note_id,
            "title": kwargs["title"],
            "content": kwargs["content"],
            "version": version,
            "last_modified": "2026-08-24T10:00:00Z",
            "keywords": list(kwargs.get("keywords") or ()),
        }
        self.rows[note_id] = row
        return row

    async def delete_note(self, **kwargs):
        self.calls.append(("delete", kwargs))
        if self.conflict:
            raise ConflictError(
                "stale", entity="notes", entity_id=str(kwargs["note_id"])
            )
        self.rows.pop(str(kwargs["note_id"]), None)
        return True


@pytest.mark.asyncio
async def test_local_notes_are_paged_by_workspace_membership_and_keep_qualified_owner() -> (
    None
):
    registry = RecordingRegistry()
    notes = RecordingNotesScope()
    adapter = LocalResearchWorkspaceAdapter(
        registry, notes_scope_service=notes, notes_user_id="chatbook-user"
    )

    page = await adapter.list_notes(
        LOCAL_REF, ResearchNotePageRequest(query="beta", limit=1, offset=0)
    )

    assert page.total == 1
    assert page.has_more is False
    assert page.items == (
        ResearchQuickNote(
            ref=LOCAL_REF,
            note_id="note-2",
            title="Second",
            content="Beta body",
            tags=("beta",),
            version=2,
            updated_at="2026-08-24T09:00:00Z",
            message_ids=("message-1",),
            source_ids=("source-1",),
        ),
    )
    assert registry.calls == [("list", "workspace-local", 100, 0)]
    assert all(kwargs["scope"] == "local_note" for _, kwargs in notes.calls)
    assert all(kwargs["user_id"] == "chatbook-user" for _, kwargs in notes.calls)


@pytest.mark.asyncio
async def test_local_create_saves_canonical_note_then_links_note_membership() -> None:
    registry = RecordingRegistry()
    notes = RecordingNotesScope()
    adapter = LocalResearchWorkspaceAdapter(
        registry, notes_scope_service=notes, notes_user_id="chatbook-user"
    )

    saved = await adapter.save_note(
        LOCAL_REF,
        ResearchNoteSaveRequest(
            title="Capture",
            content="Grounded observation",
            tags=("review",),
            message_ids=("message-7",),
            source_ids=("source-9",),
        ),
    )

    save_kwargs = notes.calls[0][1]
    assert save_kwargs["scope"] == "local_note"
    assert save_kwargs["user_id"] == "chatbook-user"
    assert save_kwargs["keywords"] == [
        "review",
        "research-message-id:bWVzc2FnZS03",
        "research-source-id:c291cmNlLTk",
    ]
    assert registry.calls[-1] == (
        "link",
        "workspace-local",
        {
            "item_type": "note",
            "item_id": "note-new",
            "role": "note",
            "title": "Capture",
        },
    )
    assert saved.ref == LOCAL_REF
    assert saved.note_id == "note-new"
    assert saved.version == 1
    assert saved.tags == ("review",)
    assert saved.message_ids == ("message-7",)
    assert saved.source_ids == ("source-9",)


@pytest.mark.asyncio
async def test_local_update_uses_expected_version_without_relinking() -> None:
    registry = RecordingRegistry()
    notes = RecordingNotesScope()
    adapter = LocalResearchWorkspaceAdapter(
        registry, notes_scope_service=notes, notes_user_id="chatbook-user"
    )

    saved = await adapter.save_note(
        LOCAL_REF,
        ResearchNoteSaveRequest(
            note_id="note-1",
            title="Changed",
            content="Body",
            expected_version=3,
        ),
    )

    assert saved.version == 4
    assert notes.calls[0][1]["version"] == 3
    assert registry.calls == [("get_memberships", "note", "note-1")]


@pytest.mark.asyncio
async def test_local_update_and_delete_refuse_note_outside_captured_workspace() -> None:
    registry = RecordingRegistry()
    notes = RecordingNotesScope()
    adapter = LocalResearchWorkspaceAdapter(
        registry, notes_scope_service=notes, notes_user_id="chatbook-user"
    )

    with pytest.raises(ValueError, match="not associated"):
        await adapter.save_note(
            LOCAL_REF,
            ResearchNoteSaveRequest(
                note_id="not-linked",
                title="Changed",
                content="Body",
                expected_version=1,
            ),
        )
    with pytest.raises(ValueError, match="not associated"):
        await adapter.delete_note(LOCAL_REF, "not-linked", 1)

    assert notes.calls == []


@pytest.mark.asyncio
async def test_local_delete_is_versioned_and_cleans_membership_after_owner_success() -> (
    None
):
    registry = RecordingRegistry()
    notes = RecordingNotesScope()
    adapter = LocalResearchWorkspaceAdapter(
        registry, notes_scope_service=notes, notes_user_id="chatbook-user"
    )

    assert await adapter.delete_note(LOCAL_REF, "note-1", 3) is True

    assert notes.calls == [
        (
            "delete",
            {
                "scope": "local_note",
                "note_id": "note-1",
                "version": 3,
                "user_id": "chatbook-user",
            },
        )
    ]
    assert registry.calls[-3:] == [
        ("get_memberships", "note", "note-1"),
        ("get_memberships", "note", "note-1"),
        (
            "unlink",
            "workspace-local",
            {"item_type": "note", "item_id": "note-1", "role": "note"},
        ),
    ]


@pytest.mark.asyncio
async def test_local_note_round_trip_uses_real_sqlite_owners_and_cleans_membership(
    tmp_path,
) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    notes_db = CharactersRAGDB(
        str(notes_dir / "canonical-notes.sqlite"), client_id="research-app"
    )
    interop = NotesInteropService(
        base_db_directory=notes_dir,
        api_client_id="research-client",
        global_db_to_use=notes_db,
    )
    notes_scope = NotesScopeService(
        local_notes_service=interop,
        server_service=None,
        policy_enforcer=None,
    )
    registry_db = WorkspaceDB(
        tmp_path / "workspace.sqlite", client_id="research-client"
    )
    registry = LocalWorkspaceRegistryService(registry_db)
    registry.create_workspace(workspace_id=LOCAL_REF.workspace_id, name="Research")
    adapter = LocalResearchWorkspaceAdapter(
        registry,
        notes_scope_service=notes_scope,
        notes_user_id="research-user",
    )

    try:
        created = await adapter.save_note(
            LOCAL_REF,
            ResearchNoteSaveRequest(
                title="SQLite note",
                content="Canonical body",
                tags=("analysis",),
                source_ids=("source-44",),
            ),
        )

        canonical = await notes_scope.get_note_detail(
            scope="local_note",
            note_id=created.note_id,
            user_id="research-user",
        )
        memberships = registry.get_item_memberships("note", created.note_id)
        assert canonical["content"] == "Canonical body"
        assert canonical["version"] == 1
        assert [(item.workspace_id, item.role) for item in memberships] == [
            (LOCAL_REF.workspace_id, "note")
        ]
        registry.create_workspace(workspace_id="workspace-second", name="Second")
        registry.link_membership(
            "workspace-second",
            item_type="note",
            item_id=created.note_id,
            role="note",
            title="SQLite note",
        )

        updated = await adapter.save_note(
            LOCAL_REF,
            ResearchNoteSaveRequest(
                note_id=created.note_id,
                title="SQLite note updated",
                content="Canonical body v2",
                tags=("analysis", "reviewed"),
                expected_version=created.version,
                source_ids=("source-44",),
            ),
        )
        assert updated.version == 2
        assert [
            item.workspace_id
            for item in registry.get_item_memberships("note", created.note_id)
        ] == [LOCAL_REF.workspace_id, "workspace-second"]

        assert await adapter.delete_note(LOCAL_REF, created.note_id, updated.version)
        assert registry.get_item_memberships("note", created.note_id) == ()
        assert (
            await notes_scope.get_note_detail(
                scope="local_note",
                note_id=created.note_id,
                user_id="research-user",
            )
            is None
        )
    finally:
        registry_db.close()
        notes_db.close_connection()


@pytest.mark.asyncio
async def test_local_conflict_is_normalized_without_note_body_in_error() -> None:
    registry = RecordingRegistry()
    notes = RecordingNotesScope()
    notes.conflict = True
    adapter = LocalResearchWorkspaceAdapter(
        registry, notes_scope_service=notes, notes_user_id="chatbook-user"
    )

    with pytest.raises(ResearchNoteConflictError) as exc_info:
        await adapter.save_note(
            LOCAL_REF,
            ResearchNoteSaveRequest(
                note_id="note-1",
                title="Private title",
                content="PRIVATE BODY MUST NOT LEAK",
                expected_version=3,
            ),
        )

    assert exc_info.value.ref == LOCAL_REF
    assert exc_info.value.note_id == "note-1"
    assert "PRIVATE BODY" not in str(exc_info.value)
    assert registry.calls == [("get_memberships", "note", "note-1")]


class RecordingContextProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.context = SimpleNamespace(
            active_server_id="profile-1",
            auth_token="test-token",
            credential_source="test",
            capabilities={
                "server_configured": True,
                "reachability": "reachable",
                "auth_state": "authenticated",
            },
        )

    def get_active_context(self):
        self.calls += 1
        return self.context


class RecordingWorkspaceNotesService:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.rows = [
            {
                "id": 7,
                "workspace_id": "workspace-server",
                "title": "Remote",
                "content": "Server body",
                "keywords": ["remote"],
                "version": 4,
            }
        ]
        self.conflict = False

    async def list_workspace_notes(self, workspace_id):
        self.calls.append(("list_notes", workspace_id))
        return list(self.rows)

    async def search_workspace_notes(self, workspace_id, query, notes=None):
        self.calls.append(("search_notes", workspace_id, query, notes))
        return [
            row for row in (notes or self.rows) if query.lower() in row["title"].lower()
        ]

    async def save_workspace_note(self, **kwargs):
        self.calls.append(("save_note", kwargs))
        if self.conflict:
            from tldw_chatbook.tldw_api.exceptions import APIResponseError

            raise APIResponseError(409, "conflict containing no body")
        return {
            "id": kwargs.get("note_id") or 8,
            "workspace_id": kwargs["workspace_id"],
            "title": kwargs["title"],
            "content": kwargs["content"],
            "keywords": list(kwargs.get("keywords") or ()),
            "version": (kwargs.get("version") or 0) + 1,
        }

    async def delete_workspace_note(self, workspace_id, note_id, version):
        self.calls.append(("delete_note", workspace_id, note_id, version))
        return {"deleted": True}


@pytest.mark.asyncio
async def test_server_notes_use_only_workspace_note_owner_and_preserve_qualification() -> (
    None
):
    service = RecordingWorkspaceNotesService()
    provider = RecordingContextProvider()
    adapter = ServerResearchWorkspaceAdapter(service, provider)

    page = await adapter.list_notes(SERVER_REF, ResearchNotePageRequest(limit=10))
    loaded = await adapter.get_note(SERVER_REF, "7")
    saved = await adapter.save_note(
        SERVER_REF,
        ResearchNoteSaveRequest(
            note_id="7",
            title="Updated",
            content="Server body changed",
            tags=("remote",),
            expected_version=4,
        ),
    )

    assert page.items[0].ref == SERVER_REF
    assert loaded == page.items[0]
    assert saved.ref == SERVER_REF
    assert saved.note_id == "7"
    assert service.calls == [
        ("list_notes", "workspace-server"),
        ("list_notes", "workspace-server"),
        (
            "save_note",
            {
                "workspace_id": "workspace-server",
                "note_id": 7,
                "title": "Updated",
                "content": "Server body changed",
                "keywords": ["remote"],
                "version": 4,
            },
        ),
    ]


@pytest.mark.asyncio
async def test_server_note_search_is_scoped_to_loaded_workspace_owner() -> None:
    service = RecordingWorkspaceNotesService()
    adapter = ServerResearchWorkspaceAdapter(service, RecordingContextProvider())

    page = await adapter.list_notes(
        SERVER_REF, ResearchNotePageRequest(query="remote", limit=5)
    )

    assert [item.note_id for item in page.items] == ["7"]
    assert service.calls[0] == ("list_notes", "workspace-server")
    assert service.calls[1][0:3] == ("search_notes", "workspace-server", "remote")


@pytest.mark.asyncio
async def test_server_note_conflict_is_normalized_and_delete_fails_closed_before_call() -> (
    None
):
    service = RecordingWorkspaceNotesService()
    service.conflict = True
    adapter = ServerResearchWorkspaceAdapter(service, RecordingContextProvider())

    with pytest.raises(ResearchNoteConflictError):
        await adapter.save_note(
            SERVER_REF,
            ResearchNoteSaveRequest(
                note_id="7",
                title="Changed",
                content="Private body",
                expected_version=4,
            ),
        )
    with pytest.raises(CapabilityUnavailableError) as exc_info:
        await adapter.delete_note(SERVER_REF, "7", 4)

    assert exc_info.value.capability.reason_code == "version_precondition_unavailable"
    assert not any(call[0] == "delete_note" for call in service.calls)


@pytest.mark.asyncio
async def test_server_note_capabilities_survive_missing_source_projection() -> None:
    adapter = ServerResearchWorkspaceAdapter(
        RecordingWorkspaceNotesService(), RecordingContextProvider()
    )

    capabilities = await adapter.capabilities(SERVER_REF)

    assert capabilities["list_notes"].available is True
    assert capabilities["get_note"].available is True
    assert capabilities["save_note"].available is True
    assert capabilities["delete_note"].reason_code == (
        "version_precondition_unavailable"
    )


@pytest.mark.asyncio
async def test_quick_notes_service_rejects_cross_authority_result() -> None:
    class WrongPort:
        async def get_note(self, ref, note_id):
            return ResearchQuickNote(
                ref=SERVER_REF,
                note_id="7",
                title="Wrong owner",
                content="",
                version=1,
            )

    service = ResearchQuickNotesService({WorkspaceDataSource.LOCAL: WrongPort()})

    with pytest.raises(ValueError, match="mismatched workspace ref"):
        await service.get_note(LOCAL_REF, "7")


def test_note_request_rejects_update_without_version_and_unbounded_provenance() -> None:
    with pytest.raises(ValueError, match="expected_version"):
        ResearchNoteSaveRequest(note_id="note-1", title="T", content="C")
    with pytest.raises(ValueError, match="message_ids"):
        ResearchNoteSaveRequest(
            title="T",
            content="C",
            message_ids=tuple(f"message-{index}" for index in range(21)),
        )


def test_note_owner_content_and_tags_may_discuss_credentials_without_logging_them() -> (
    None
):
    request = ResearchNoteSaveRequest(
        title="API key threat model",
        content="Do not include credentials in logs.",
        tags=("token-handling", "secret-management"),
    )

    assert request.title == "API key threat model"
    assert request.tags == ("token-handling", "secret-management")


class DeferredNotePort:
    def __init__(self) -> None:
        self.list_results: list[asyncio.Future] = []
        self.get_results: list[asyncio.Future] = []
        self.save_refs: list[QualifiedWorkspaceRef] = []

    async def list_notes(self, ref, page):
        future = asyncio.get_running_loop().create_future()
        self.list_results.append(future)
        return await future

    async def get_note(self, ref, note_id):
        future = asyncio.get_running_loop().create_future()
        self.get_results.append(future)
        return await future

    async def save_note(self, ref, request):
        self.save_refs.append(ref)
        return ResearchQuickNote(
            ref=ref,
            note_id=request.note_id or "created",
            title=request.title,
            content=request.content,
            tags=request.tags,
            version=(request.expected_version or 0) + 1,
        )


@pytest.mark.asyncio
async def test_controller_note_list_and_detail_results_are_generation_fenced() -> None:
    port = DeferredNotePort()
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: port})
    controller.select_workspace(LOCAL_REF)

    list_request = asyncio.create_task(controller.refresh_selected_notes())
    detail_request = asyncio.create_task(controller.load_selected_note("note-1"))
    await asyncio.sleep(0)
    controller.select_workspace(
        QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-other")
    )
    old_note = ResearchQuickNote(
        ref=LOCAL_REF,
        note_id="note-1",
        title="Old",
        content="Old body",
        version=1,
    )
    port.list_results[0].set_result(
        BoundedPageResult(items=(old_note,), limit=20, total=1)
    )
    port.get_results[0].set_result(old_note)

    assert await list_request is False
    assert await detail_request is False
    assert controller.visible_note_page is None
    assert controller.visible_note is None


@pytest.mark.asyncio
async def test_controller_explicit_note_save_keeps_captured_qualified_owner() -> None:
    port = DeferredNotePort()
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: port})
    controller.select_workspace(
        QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-visible")
    )

    saved = await controller.save_note(
        LOCAL_REF,
        ResearchNoteSaveRequest(title="Captured", content="Draft"),
    )

    assert saved.ref == LOCAL_REF
    assert port.save_refs == [LOCAL_REF]
    assert controller.selected_ref != LOCAL_REF
