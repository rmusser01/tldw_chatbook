from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace

import pytest
from loguru import logger

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
from tldw_chatbook.Workspaces.registry_service import WorkspaceRegistryServiceError
from tldw_chatbook.Chat.rag_scope import RagScope, ScopeItem


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
        self.receipts: dict[str, SimpleNamespace] = {}

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

    def claim_quick_note_create(
        self, workspace_id, *, local_user_id, operation_token
    ):
        receipt_id, note_id = LocalWorkspaceRegistryService._quick_note_identity(
            workspace_id=workspace_id,
            local_user_id=local_user_id,
            operation_token=operation_token,
            kind="create",
        )
        receipt = self.receipts.get(receipt_id)
        if receipt is None:
            receipt = SimpleNamespace(
                receipt_id=receipt_id,
                workspace_id=workspace_id,
                local_user_id=local_user_id,
                operation_token=operation_token,
                operation_kind="create",
                canonical_note_id=note_id,
                expected_version=None,
                state="pending",
                revision=1,
            )
            self.receipts[receipt_id] = receipt
        self.calls.append(("claim_create", workspace_id, local_user_id, operation_token))
        return receipt

    def claim_quick_note_delete(
        self, workspace_id, *, local_user_id, canonical_note_id, expected_version
    ):
        receipt_id = f"delete-receipt-{workspace_id}-{canonical_note_id}"
        receipt = self.receipts.get(receipt_id)
        if receipt is None:
            receipt = SimpleNamespace(
                receipt_id=receipt_id,
                workspace_id=workspace_id,
                local_user_id=local_user_id,
                operation_token=f"delete-{canonical_note_id}",
                operation_kind="delete",
                canonical_note_id=canonical_note_id,
                expected_version=expected_version,
                state="pending",
                revision=1,
            )
            self.receipts[receipt_id] = receipt
        self.calls.append(
            ("claim_delete", workspace_id, local_user_id, canonical_note_id, expected_version)
        )
        return receipt

    def list_quick_note_receipts(
        self, local_user_id, *, workspace_id=None, operation_kind=None, limit, offset
    ):
        rows = [
            receipt
            for receipt in self.receipts.values()
            if receipt.local_user_id == local_user_id
            and (workspace_id is None or receipt.workspace_id == workspace_id)
            and (operation_kind is None or receipt.operation_kind == operation_kind)
        ]
        return tuple(rows[offset : offset + limit]), len(rows)

    def mark_quick_note_owner_committed(
        self, receipt_id, local_user_id, *, expected_revision
    ):
        receipt = self.receipts[receipt_id]
        if receipt.state == "pending":
            receipt.state = "owner_committed"
            receipt.revision += 1
        self.calls.append(("mark_committed", receipt_id, local_user_id, expected_revision))
        return receipt

    def discard_quick_note_receipt(
        self, receipt_id, local_user_id, *, expected_revision
    ):
        self.receipts.pop(receipt_id, None)
        self.calls.append(("discard_receipt", receipt_id, local_user_id, expected_revision))
        return True

    def complete_quick_note_create(
        self, receipt_id, local_user_id, *, expected_revision, title
    ):
        receipt = self.receipts.pop(receipt_id, None)
        if receipt is None:
            return False
        self.memberships.append(
            SimpleNamespace(
                item_type="note", item_id=receipt.canonical_note_id, role="note"
            )
        )
        self.calls.append(
            ("complete_create", receipt_id, local_user_id, expected_revision, title)
        )
        return True

    def complete_quick_note_delete(
        self, receipt_id, local_user_id, *, expected_revision
    ):
        receipt = self.receipts.pop(receipt_id, None)
        if receipt is None:
            return False
        self.memberships = [
            item
            for item in self.memberships
            if not (
                item.item_type == "note" and item.item_id == receipt.canonical_note_id
            )
        ]
        self.calls.append(
            ("complete_delete", receipt_id, local_user_id, expected_revision)
        )
        return True


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
        note_id = str(
            kwargs.get("note_id") or kwargs.get("create_note_id") or "note-new"
        )
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


class FailingMembershipRegistry:
    """Inject one role-specific registry failure while retaining real SQLite."""

    def __init__(
        self,
        service: LocalWorkspaceRegistryService,
        *,
        fail_link_role: str | None = None,
        fail_unlink_role: str | None = None,
    ) -> None:
        self.service = service
        self.fail_link_role = fail_link_role
        self.fail_unlink_role = fail_unlink_role

    def __getattr__(self, name):
        return getattr(self.service, name)

    def link_membership(self, workspace_id, **kwargs):
        if kwargs.get("role") == self.fail_link_role:
            self.fail_link_role = None
            raise WorkspaceRegistryServiceError("injected registry link failure")
        return self.service.link_membership(workspace_id, **kwargs)

    def unlink_membership(self, workspace_id, **kwargs):
        if kwargs.get("role") == self.fail_unlink_role:
            self.fail_unlink_role = None
            raise WorkspaceRegistryServiceError("injected registry unlink failure")
        return self.service.unlink_membership(workspace_id, **kwargs)

    def complete_quick_note_create(self, *args, **kwargs):
        if self.fail_link_role == "note":
            self.fail_link_role = None
            raise WorkspaceRegistryServiceError("injected registry link failure")
        return self.service.complete_quick_note_create(*args, **kwargs)

    def complete_quick_note_delete(self, *args, **kwargs):
        if self.fail_unlink_role == "note":
            self.fail_unlink_role = None
            raise WorkspaceRegistryServiceError("injected registry unlink failure")
        return self.service.complete_quick_note_delete(*args, **kwargs)


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
async def test_local_create_preclaims_canonical_id_then_promotes_note_membership() -> (
    None
):
    registry = RecordingRegistry()
    notes = RecordingNotesScope()
    adapter = LocalResearchWorkspaceAdapter(
        registry, notes_scope_service=notes, notes_user_id="chatbook-user"
    )

    request = ResearchNoteSaveRequest(
        title="Capture",
        content="Grounded observation",
        tags=("review",),
        message_ids=("message-7",),
        source_ids=("source-9",),
    )
    saved = await adapter.save_note(LOCAL_REF, request)

    save_kwargs = next(kwargs for action, kwargs in notes.calls if action == "save")
    assert save_kwargs["scope"] == "local_note"
    assert save_kwargs["user_id"] == "chatbook-user"
    assert save_kwargs["keywords"] == [
        "review",
        "research-message-id:bWVzc2FnZS03",
        "research-source-id:c291cmNlLTk",
    ]
    expected_note_id = saved.note_id
    assert save_kwargs["create_note_id"] == expected_note_id
    assert registry.calls[0] == (
        "claim_create",
        "workspace-local",
        "chatbook-user",
        request.operation_id,
    )
    assert registry.calls[1][0] == "mark_committed"
    assert registry.calls[1][2:] == ("chatbook-user", 1)
    assert registry.calls[2][0] == "complete_create"
    assert saved.ref == LOCAL_REF
    assert saved.note_id != request.operation_id
    assert saved.version == 1
    assert saved.tags == ("review",)
    assert saved.message_ids == ("message-7",)
    assert saved.source_ids == ("source-9",)


@pytest.mark.asyncio
async def test_local_create_registry_preclaim_failure_writes_no_canonical_note() -> None:
    class PreclaimFailure(RecordingRegistry):
        def claim_quick_note_create(self, workspace_id, **kwargs):
            raise WorkspaceRegistryServiceError("injected preclaim failure")

    registry = PreclaimFailure()
    notes = RecordingNotesScope()
    adapter = LocalResearchWorkspaceAdapter(
        registry, notes_scope_service=notes, notes_user_id="chatbook-user"
    )

    with pytest.raises(WorkspaceRegistryServiceError):
        await adapter.save_note(
            LOCAL_REF, ResearchNoteSaveRequest(title="No orphan", content="Body")
        )

    assert notes.calls == []


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
    assert registry.calls[-3][0] == "claim_delete"
    assert registry.calls[-2][0] == "mark_committed"
    assert registry.calls[-1][0] == "complete_delete"


@pytest.mark.asyncio
async def test_local_delete_retry_cleans_membership_when_owner_returns_false() -> None:
    class RetryNotes(RecordingNotesScope):
        def __init__(self) -> None:
            super().__init__()
            self.delete_count = 0

        async def delete_note(self, **kwargs):
            self.calls.append(("delete", kwargs))
            self.delete_count += 1
            self.rows.pop(str(kwargs["note_id"]), None)
            return self.delete_count == 1

    class RetryRegistry(RecordingRegistry):
        def __init__(self) -> None:
            super().__init__()
            self.fail_unlink = True

        def complete_quick_note_delete(self, *args, **kwargs):
            if self.fail_unlink:
                self.fail_unlink = False
                raise WorkspaceRegistryServiceError("injected cleanup failure")
            return super().complete_quick_note_delete(*args, **kwargs)

    registry = RetryRegistry()
    notes = RetryNotes()
    adapter = LocalResearchWorkspaceAdapter(
        registry, notes_scope_service=notes, notes_user_id="chatbook-user"
    )

    with pytest.raises(WorkspaceRegistryServiceError):
        await adapter.delete_note(LOCAL_REF, "note-1", 3)

    assert await adapter.delete_note(LOCAL_REF, "note-1", 3) is True
    assert registry.get_item_memberships("note", "note-1") == ()


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
async def test_local_create_receipt_recovers_after_link_failure_reopen_and_concurrent_retry(
    tmp_path,
) -> None:
    notes_path = tmp_path / "canonical-notes.sqlite"
    workspace_path = tmp_path / "workspace.sqlite"
    notes_db = CharactersRAGDB(str(notes_path), client_id="research-app")
    notes_scope = NotesScopeService(
        local_notes_service=NotesInteropService(
            base_db_directory=tmp_path,
            api_client_id="research-client",
            global_db_to_use=notes_db,
        ),
        server_service=None,
        policy_enforcer=None,
    )
    registry_db = WorkspaceDB(workspace_path, client_id="research-client")
    registry = LocalWorkspaceRegistryService(registry_db)
    registry.create_workspace(workspace_id=LOCAL_REF.workspace_id, name="Research")
    failing = FailingMembershipRegistry(registry, fail_link_role="note")
    adapter = LocalResearchWorkspaceAdapter(
        failing,
        notes_scope_service=notes_scope,
        notes_user_id="research-user",
    )
    request = ResearchNoteSaveRequest(title="Durable", content="Canonical body")

    with pytest.raises(WorkspaceRegistryServiceError):
        await adapter.save_note(LOCAL_REF, request)

    receipts, total = registry.list_quick_note_receipts(
        "research-user", workspace_id=LOCAL_REF.workspace_id, limit=100
    )
    assert total == 1
    receipt = receipts[0]
    assert receipt.state == "owner_committed"
    canonical = await notes_scope.get_note_detail(
        scope="local_note", note_id=receipt.canonical_note_id, user_id="research-user"
    )
    assert canonical is not None and canonical["content"] == "Canonical body"
    assert registry.get_item_memberships("note", receipt.canonical_note_id) == ()
    registry_db.close()
    notes_db.close_connection()

    reopened_notes_db = CharactersRAGDB(str(notes_path), client_id="research-app")
    reopened_notes_scope = NotesScopeService(
        local_notes_service=NotesInteropService(
            base_db_directory=tmp_path,
            api_client_id="research-client",
            global_db_to_use=reopened_notes_db,
        ),
        server_service=None,
        policy_enforcer=None,
    )
    reopened_registry_db = WorkspaceDB(workspace_path, client_id="research-client")
    reopened_registry = LocalWorkspaceRegistryService(reopened_registry_db)
    reopened_adapter = LocalResearchWorkspaceAdapter(
        reopened_registry,
        notes_scope_service=reopened_notes_scope,
        notes_user_id="research-user",
    )
    try:
        reopened_page = await reopened_adapter.list_notes(
            LOCAL_REF, ResearchNotePageRequest(limit=20)
        )
        assert [note.note_id for note in reopened_page.items] == [
            receipt.canonical_note_id
        ]
        first, second = await asyncio.gather(
            reopened_adapter.save_note(LOCAL_REF, request),
            reopened_adapter.save_note(LOCAL_REF, request),
        )

        assert first.note_id == second.note_id == receipt.canonical_note_id
        assert len(reopened_notes_db.list_notes(limit=100, offset=0)) == 1
        assert [
            item.role
            for item in reopened_registry.get_item_memberships(
                "note", receipt.canonical_note_id
            )
        ] == ["note"]
    finally:
        reopened_registry_db.close()
        reopened_notes_db.close_connection()


@pytest.mark.asyncio
async def test_pending_create_receipt_with_atomic_owner_row_recovers_after_reopen(
    tmp_path,
) -> None:
    notes_path = tmp_path / "pending-owner-notes.sqlite"
    workspace_path = tmp_path / "pending-owner-workspace.sqlite"
    notes_db = CharactersRAGDB(str(notes_path), client_id="research-app")
    notes_scope = NotesScopeService(
        local_notes_service=NotesInteropService(
            base_db_directory=tmp_path,
            api_client_id="research-client",
            global_db_to_use=notes_db,
        ),
        server_service=None,
        policy_enforcer=None,
    )
    registry_db = WorkspaceDB(workspace_path, client_id="research-client")
    registry = LocalWorkspaceRegistryService(registry_db)
    registry.create_workspace(workspace_id=LOCAL_REF.workspace_id, name="Research")
    adapter = LocalResearchWorkspaceAdapter(
        registry, notes_scope_service=notes_scope, notes_user_id="research-user"
    )
    original_mark = registry.mark_quick_note_owner_committed

    def fail_before_receipt_transition(*_args, **_kwargs):
        raise WorkspaceRegistryServiceError("injected receipt transition failure")

    registry.mark_quick_note_owner_committed = fail_before_receipt_transition
    with pytest.raises(WorkspaceRegistryServiceError, match="injected receipt"):
        await adapter.save_note(
            LOCAL_REF,
            ResearchNoteSaveRequest(title="Durable", content="Canonical body"),
        )
    receipt = registry.list_quick_note_receipts(
        "research-user", workspace_id=LOCAL_REF.workspace_id, limit=100
    )[0][0]
    assert receipt.state == "pending"
    registry.mark_quick_note_owner_committed = original_mark
    registry_db.close()
    notes_db.close_connection()

    reopened_notes_db = CharactersRAGDB(str(notes_path), client_id="research-app")
    reopened_registry_db = WorkspaceDB(workspace_path, client_id="research-client")
    reopened_registry = LocalWorkspaceRegistryService(reopened_registry_db)
    reopened_adapter = LocalResearchWorkspaceAdapter(
        reopened_registry,
        notes_scope_service=NotesScopeService(
            local_notes_service=NotesInteropService(
                base_db_directory=tmp_path,
                api_client_id="research-client",
                global_db_to_use=reopened_notes_db,
            ),
            server_service=None,
            policy_enforcer=None,
        ),
        notes_user_id="research-user",
    )
    try:
        page = await reopened_adapter.list_notes(
            LOCAL_REF, ResearchNotePageRequest(limit=20)
        )
        assert [note.note_id for note in page.items] == [receipt.canonical_note_id]
        assert reopened_registry.list_quick_note_receipts(
            "research-user", workspace_id=LOCAL_REF.workspace_id, limit=100
        )[1] == 0
    finally:
        reopened_registry_db.close()
        reopened_notes_db.close_connection()


@pytest.mark.asyncio
async def test_local_delete_retry_cleans_membership_after_owner_already_deleted(
    tmp_path,
) -> None:
    notes_path = tmp_path / "canonical-delete.sqlite"
    workspace_path = tmp_path / "workspace-delete.sqlite"
    notes_db = CharactersRAGDB(str(notes_path), client_id="research-app")
    notes_scope = NotesScopeService(
        local_notes_service=NotesInteropService(
            base_db_directory=tmp_path,
            api_client_id="research-client",
            global_db_to_use=notes_db,
        ),
        server_service=None,
        policy_enforcer=None,
    )
    registry_db = WorkspaceDB(workspace_path, client_id="research-client")
    registry = LocalWorkspaceRegistryService(registry_db)
    registry.create_workspace(workspace_id=LOCAL_REF.workspace_id, name="Research")
    adapter = LocalResearchWorkspaceAdapter(
        registry,
        notes_scope_service=notes_scope,
        notes_user_id="research-user",
    )
    created = await adapter.save_note(
        LOCAL_REF, ResearchNoteSaveRequest(title="Delete me", content="Body")
    )
    adapter._service = FailingMembershipRegistry(
        registry, fail_unlink_role="note"
    )

    with pytest.raises(WorkspaceRegistryServiceError):
        await adapter.delete_note(LOCAL_REF, created.note_id, created.version)

    assert (
        await notes_scope.get_note_detail(
            scope="local_note", note_id=created.note_id, user_id="research-user"
        )
        is None
    )
    assert registry.get_item_memberships("note", created.note_id)
    registry_db.close()
    notes_db.close_connection()

    reopened_notes_db = CharactersRAGDB(str(notes_path), client_id="research-app")
    reopened_notes_scope = NotesScopeService(
        local_notes_service=NotesInteropService(
            base_db_directory=tmp_path,
            api_client_id="research-client",
            global_db_to_use=reopened_notes_db,
        ),
        server_service=None,
        policy_enforcer=None,
    )
    reopened_registry_db = WorkspaceDB(workspace_path, client_id="research-client")
    reopened_registry = LocalWorkspaceRegistryService(reopened_registry_db)
    reopened_adapter = LocalResearchWorkspaceAdapter(
        reopened_registry,
        notes_scope_service=reopened_notes_scope,
        notes_user_id="research-user",
    )
    try:
        await reopened_adapter.capabilities(LOCAL_REF)
        assert reopened_registry.get_item_memberships("note", created.note_id) == ()
    finally:
        reopened_registry_db.close()
        reopened_notes_db.close_connection()


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


def test_blank_quick_note_title_uses_webui_untitled_default() -> None:
    request = ResearchNoteSaveRequest(title="  ", content="Body-only note")

    assert request.title == "Untitled Note"


@pytest.mark.asyncio
async def test_server_get_note_fetches_owner_once_when_target_is_101st() -> None:
    service = RecordingWorkspaceNotesService()
    service.rows = [
        {
            "id": index,
            "workspace_id": SERVER_REF.workspace_id,
            "title": f"Note {index}",
            "content": "",
            "keywords": [],
            "version": 1,
        }
        for index in range(1, 102)
    ]
    adapter = ServerResearchWorkspaceAdapter(service, RecordingContextProvider())

    loaded = await adapter.get_note(SERVER_REF, "101")

    assert loaded is not None and loaded.note_id == "101"
    assert service.calls == [("list_notes", SERVER_REF.workspace_id)]


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


@pytest.mark.asyncio
async def test_local_create_token_is_qualified_by_workspace_and_notes_owner_with_real_sqlite(
    tmp_path,
) -> None:
    notes_db = CharactersRAGDB(str(tmp_path / "notes.sqlite"), client_id="template")
    interop = NotesInteropService(
        base_db_directory=tmp_path,
        api_client_id="research-client",
        global_db_to_use=notes_db,
    )
    notes_scope = NotesScopeService(
        local_notes_service=interop, server_service=None, policy_enforcer=None
    )
    registry_db = WorkspaceDB(tmp_path / "workspace.sqlite")
    registry = LocalWorkspaceRegistryService(registry_db)
    first_ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-a")
    second_ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-b")
    registry.create_workspace(workspace_id=first_ref.workspace_id, name="A")
    registry.create_workspace(workspace_id=second_ref.workspace_id, name="B")
    operation_token = "research-note-0123456789abcdef0123456789abcdef"
    request = ResearchNoteSaveRequest(
        title="Same intent", content="Exact body", operation_id=operation_token
    )

    try:
        first = await LocalResearchWorkspaceAdapter(
            registry, notes_scope_service=notes_scope, notes_user_id="notes-user-a"
        ).save_note(first_ref, request)
        second = await LocalResearchWorkspaceAdapter(
            registry, notes_scope_service=notes_scope, notes_user_id="notes-user-b"
        ).save_note(second_ref, request)

        assert first.note_id != second.note_id
        assert first.note_id != operation_token
        assert second.note_id != operation_token
        assert [
            item.workspace_id
            for item in registry.get_item_memberships("note", first.note_id)
        ] == [first_ref.workspace_id]
        assert [
            item.workspace_id
            for item in registry.get_item_memberships("note", second.note_id)
        ] == [second_ref.workspace_id]
    finally:
        registry_db.close()
        notes_db.close_connection()


@pytest.mark.asyncio
async def test_local_note_and_keywords_are_one_owner_transaction_on_keyword_failure(
    tmp_path, monkeypatch
) -> None:
    notes_db = CharactersRAGDB(str(tmp_path / "notes-atomic.sqlite"), client_id="template")
    interop = NotesInteropService(
        base_db_directory=tmp_path,
        api_client_id="research-client",
        global_db_to_use=notes_db,
    )
    notes_scope = NotesScopeService(
        local_notes_service=interop, server_service=None, policy_enforcer=None
    )
    note_id = "atomic-note"

    def fail_link(*_args, **_kwargs):
        raise RuntimeError("injected keyword link failure")

    monkeypatch.setattr(interop, "link_note_to_keyword", fail_link)
    try:
        with pytest.raises(RuntimeError, match="injected keyword"):
            await notes_scope.save_note(
                scope="local_note",
                title="Atomic",
                content="Body",
                keywords=["tag-one"],
                create_note_id=note_id,
                user_id="notes-user",
            )
        assert await notes_scope.get_note_detail(
            scope="local_note", note_id=note_id, user_id="notes-user"
        ) is None
        assert notes_db.get_keyword_by_text("tag-one") is None
    finally:
        notes_db.close_connection()


@pytest.mark.asyncio
async def test_local_note_update_and_keyword_replacement_roll_back_together(
    tmp_path, monkeypatch
) -> None:
    notes_db = CharactersRAGDB(str(tmp_path / "notes-update-atomic.sqlite"), client_id="template")
    interop = NotesInteropService(
        base_db_directory=tmp_path,
        api_client_id="research-client",
        global_db_to_use=notes_db,
    )
    notes_scope = NotesScopeService(
        local_notes_service=interop, server_service=None, policy_enforcer=None
    )
    created = await notes_scope.save_note(
        scope="local_note",
        title="Before",
        content="Original body",
        keywords=["original-tag"],
        create_note_id="atomic-update-note",
        user_id="notes-user",
    )
    assert created["version"] == 1

    def fail_link(*_args, **_kwargs):
        raise RuntimeError("injected replacement link failure")

    monkeypatch.setattr(interop, "link_note_to_keyword", fail_link)
    try:
        with pytest.raises(RuntimeError, match="replacement"):
            await notes_scope.save_note(
                scope="local_note",
                note_id="atomic-update-note",
                title="After",
                content="Changed body",
                keywords=["replacement-tag"],
                version=1,
                user_id="notes-user",
            )
        row = await notes_scope.get_note_detail(
            scope="local_note", note_id="atomic-update-note", user_id="notes-user"
        )
        assert (row["title"], row["content"], row["version"]) == (
            "Before",
            "Original body",
            1,
        )
        assert await notes_scope.get_note_keywords(
            scope="local_note", note_id="atomic-update-note", user_id="notes-user"
        ) == ["original-tag"]
        assert notes_db.get_keyword_by_text("replacement-tag") is None
    finally:
        notes_db.close_connection()


@pytest.mark.asyncio
async def test_existing_canonical_row_with_mismatched_metadata_is_never_promoted(
    tmp_path,
) -> None:
    notes_db = CharactersRAGDB(str(tmp_path / "mismatch-notes.sqlite"), client_id="template")
    interop = NotesInteropService(
        base_db_directory=tmp_path,
        api_client_id="research-client",
        global_db_to_use=notes_db,
    )
    notes_scope = NotesScopeService(
        local_notes_service=interop, server_service=None, policy_enforcer=None
    )
    registry_db = WorkspaceDB(tmp_path / "mismatch-workspace.sqlite")
    registry = LocalWorkspaceRegistryService(registry_db)
    registry.create_workspace(workspace_id=LOCAL_REF.workspace_id, name="Research")
    request = ResearchNoteSaveRequest(
        title="Expected",
        content="Expected body",
        tags=("expected-tag",),
        source_ids=("expected-source",),
        operation_id="research-note-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    )
    receipt = registry.claim_quick_note_create(
        LOCAL_REF.workspace_id,
        local_user_id="notes-user",
        operation_token=request.operation_id,
    )
    await notes_scope.save_note(
        scope="local_note",
        title="Different",
        content="Different body",
        keywords=["different-tag"],
        create_note_id=receipt.canonical_note_id,
        user_id="notes-user",
    )
    adapter = LocalResearchWorkspaceAdapter(
        registry, notes_scope_service=notes_scope, notes_user_id="notes-user"
    )
    try:
        with pytest.raises(ResearchNoteConflictError):
            await adapter.save_note(LOCAL_REF, request)
        assert registry.get_item_memberships("note", receipt.canonical_note_id) == ()
        pending, total = registry.list_quick_note_receipts(
            "notes-user", workspace_id=LOCAL_REF.workspace_id, limit=100
        )
        assert pending == () and total == 0
    finally:
        registry_db.close()
        notes_db.close_connection()


@pytest.mark.asyncio
async def test_delete_receipt_reopen_cleans_all_roles_and_rag_scope_globally(
    tmp_path,
) -> None:
    notes_path = tmp_path / "delete-owner.sqlite"
    workspace_path = tmp_path / "delete-registry.sqlite"
    notes_db = CharactersRAGDB(str(notes_path), client_id="template")
    notes_scope = NotesScopeService(
        local_notes_service=NotesInteropService(
            base_db_directory=tmp_path,
            api_client_id="research-client",
            global_db_to_use=notes_db,
        ),
        server_service=None,
        policy_enforcer=None,
    )
    registry_db = WorkspaceDB(workspace_path)
    registry = LocalWorkspaceRegistryService(registry_db)
    registry.create_workspace(workspace_id=LOCAL_REF.workspace_id, name="First")
    registry.create_workspace(workspace_id="workspace-other", name="Other")
    adapter = LocalResearchWorkspaceAdapter(
        registry, notes_scope_service=notes_scope, notes_user_id="notes-user"
    )
    created = await adapter.save_note(
        LOCAL_REF, ResearchNoteSaveRequest(title="Delete", content="Body")
    )
    registry.link_membership(
        "workspace-other", item_type="note", item_id=created.note_id,
        role="reference", title="Reference",
    )
    registry.link_membership(
        "workspace-other", item_type="note", item_id=created.note_id,
        role="source", title="Source",
    )
    registry.set_workspace_scope(
        "workspace-other",
        RagScope(
            items=(
                ScopeItem(source_type="note", source_id=created.note_id),
                ScopeItem(source_type="media", source_id="kept-media"),
            ),
            updated_at="2026-08-24T00:00:00Z",
            empty_is_scoped=True,
        ),
    )
    original_complete = registry.complete_quick_note_delete
    failed = False

    def fail_once(*args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise WorkspaceRegistryServiceError("injected cleanup failure")
        return original_complete(*args, **kwargs)

    registry.complete_quick_note_delete = fail_once
    with pytest.raises(WorkspaceRegistryServiceError):
        await adapter.delete_note(LOCAL_REF, created.note_id, created.version)
    assert await notes_scope.get_note_detail(
        scope="local_note", note_id=created.note_id, user_id="notes-user"
    ) is None
    registry_db.close()
    notes_db.close_connection()

    reopened_notes_db = CharactersRAGDB(str(notes_path), client_id="template")
    reopened_registry_db = WorkspaceDB(workspace_path)
    reopened_registry = LocalWorkspaceRegistryService(reopened_registry_db)
    reopened_adapter = LocalResearchWorkspaceAdapter(
        reopened_registry,
        notes_scope_service=NotesScopeService(
            local_notes_service=NotesInteropService(
                base_db_directory=tmp_path,
                api_client_id="research-client",
                global_db_to_use=reopened_notes_db,
            ),
            server_service=None,
            policy_enforcer=None,
        ),
        notes_user_id="notes-user",
    )
    try:
        await reopened_adapter.capabilities(LOCAL_REF)
        assert reopened_registry.get_item_memberships("note", created.note_id) == ()
        scope = reopened_registry.get_workspace_scope("workspace-other")
        assert scope is not None
        assert scope.items == (ScopeItem(source_type="media", source_id="kept-media"),)
        receipts, total = reopened_registry.list_quick_note_receipts(
            "notes-user", limit=100
        )
        assert receipts == () and total == 0
    finally:
        reopened_registry_db.close()
        reopened_notes_db.close_connection()


def test_quick_note_receipts_never_appear_as_generic_memberships(tmp_path) -> None:
    registry_db = WorkspaceDB(tmp_path / "receipt-visibility.sqlite")
    registry = LocalWorkspaceRegistryService(registry_db)
    registry.create_workspace(workspace_id=LOCAL_REF.workspace_id, name="Research")
    try:
        receipt = registry.claim_quick_note_create(
            LOCAL_REF.workspace_id,
            local_user_id="notes-user",
            operation_token="research-note-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
        assert receipt.workspace_id == LOCAL_REF.workspace_id
        assert registry.list_workspace_memberships(LOCAL_REF.workspace_id) == ()
        assert registry.get_item_memberships("note", receipt.canonical_note_id) == ()
    finally:
        registry_db.close()


def test_receipt_listing_filters_before_bounds_and_transitions_monotonically(
    tmp_path,
) -> None:
    registry_db = WorkspaceDB(tmp_path / "receipt-filtering.sqlite")
    registry = LocalWorkspaceRegistryService(registry_db)
    registry.create_workspace(workspace_id="workspace-a", name="A")
    registry.create_workspace(workspace_id="workspace-b", name="B")
    first = registry.claim_quick_note_create(
        "workspace-a",
        local_user_id="user-a",
        operation_token="research-note-dddddddddddddddddddddddddddddddd",
    )
    registry.claim_quick_note_create(
        "workspace-b",
        local_user_id="user-a",
        operation_token="research-note-eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
    )
    registry.claim_quick_note_create(
        "workspace-a",
        local_user_id="user-b",
        operation_token="research-note-ffffffffffffffffffffffffffffffff",
    )
    try:
        rows, total = registry.list_quick_note_receipts(
            "user-a", workspace_id="workspace-a", limit=1
        )
        assert rows == (first,) and total == 1
        committed = registry.mark_quick_note_owner_committed(
            first.receipt_id, "user-a", expected_revision=1
        )
        assert (committed.state, committed.revision) == ("owner_committed", 2)
        idempotent = registry.mark_quick_note_owner_committed(
            first.receipt_id, "user-a", expected_revision=1
        )
        assert idempotent == committed
        assert not registry.complete_quick_note_create(
            first.receipt_id,
            "user-a",
            expected_revision=1,
            title="Wrong revision",
        )
        with pytest.raises(ValueError, match="limit"):
            registry.list_quick_note_receipts("user-a", limit=101)
    finally:
        registry_db.close()


def test_keyword_operations_never_log_raw_tag_or_provenance(tmp_path) -> None:
    sentinel = "PROVENANCE_SENTINEL_946f91"
    messages: list[str] = []
    sink = logger.add(messages.append, level="DEBUG", format="{message}")
    db = CharactersRAGDB(str(tmp_path / "privacy.sqlite"), client_id="notes-user")
    try:
        keyword_id = db.add_keyword(sentinel)
        assert keyword_id is not None
        assert db.get_keyword_by_text(sentinel) is not None
        assert db.search_keywords(sentinel, limit=10)
        note_id = db.add_note("Title", "Body", note_id="privacy-note")
        db.link_note_to_keyword(note_id, keyword_id)
    finally:
        db.close_connection()
        logger.remove(sink)
    assert sentinel not in "\n".join(messages)
