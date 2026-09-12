from __future__ import annotations

from dataclasses import dataclass
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.MCP.unified_control_models import ConfiguredServerTarget
from tldw_chatbook.Research_Workspace.contracts import (
    CapabilityUnavailableError,
    QualifiedWorkspaceRef,
    ResearchCapability,
    WorkspaceDataSource,
)
from tldw_chatbook.Research_Workspace.local_adapter import (
    LocalResearchWorkspaceAdapter,
)
from tldw_chatbook.Research_Workspace.server_adapter import (
    ServerResearchWorkspaceAdapter,
)
from tldw_chatbook.Workspaces.models import WorkspaceRecord
from tldw_chatbook.runtime_policy import PolicyDeniedError
from tldw_chatbook.runtime_policy.server_context import (
    RuntimeServerContextProvider,
    ServerContextUnavailable,
)
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.tldw_api.exceptions import APIResponseError, AuthenticationError


class RecordingLocalService:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.thread_ids: list[int] = []
        self.records = {
            "workspace-default": WorkspaceRecord(
                workspace_id="workspace-default", name="Default"
            ),
            "local-1": WorkspaceRecord(
                workspace_id="local-1", name="Research", description="Notes"
            ),
        }

    def _record(self, *call: object) -> None:
        self.calls.append(call)
        self.thread_ids.append(threading.get_ident())

    def list_workspaces(self, *, include_archived: bool = False):
        self._record("list", include_archived)
        return tuple(self.records.values())

    def get_workspace(self, workspace_id: str):
        self._record("get", workspace_id)
        return self.records.get(workspace_id)

    def create_workspace(self, **kwargs):
        self._record("create", kwargs)
        record = WorkspaceRecord(**kwargs)
        self.records[record.workspace_id] = record
        return record

    def rename_workspace(self, workspace_id: str, name: str):
        self._record("rename", workspace_id, name)
        previous = self.records[workspace_id]
        record = WorkspaceRecord(
            workspace_id=workspace_id,
            name=name,
            description=previous.description,
            archived=previous.archived,
        )
        self.records[workspace_id] = record
        return record

    def archive_workspace(self, workspace_id: str):
        self._record("archive", workspace_id)
        previous = self.records[workspace_id]
        record = WorkspaceRecord(
            workspace_id=workspace_id,
            name=previous.name,
            description=previous.description,
            archived=True,
        )
        self.records[workspace_id] = record
        return record

    def unarchive_workspace(self, workspace_id: str):
        self._record("restore", workspace_id)
        previous = self.records[workspace_id]
        record = WorkspaceRecord(
            workspace_id=workspace_id,
            name=previous.name,
            description=previous.description,
            archived=False,
        )
        self.records[workspace_id] = record
        return record


@pytest.mark.asyncio
async def test_local_catalog_is_qualified_excludes_default_and_runs_off_loop() -> None:
    service = RecordingLocalService()
    adapter = LocalResearchWorkspaceAdapter(service, id_factory=lambda: "local-new")
    event_loop_thread = threading.get_ident()

    rows = await adapter.list_workspaces()

    assert [(row.ref.data_source, row.ref.workspace_id) for row in rows] == [
        (WorkspaceDataSource.LOCAL, "local-1")
    ]
    assert service.thread_ids == [service.thread_ids[0]]
    assert service.thread_ids[0] != event_loop_thread


@pytest.mark.asyncio
async def test_local_lifecycle_uses_only_local_registry() -> None:
    service = RecordingLocalService()
    new_ids = iter(("local-new", "local-copy"))
    adapter = LocalResearchWorkspaceAdapter(service, id_factory=lambda: next(new_ids))
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "local-1")

    created = await adapter.create_workspace(name="Created", description="Body")
    renamed = await adapter.update_workspace(ref, name="Renamed")
    copied = await adapter.duplicate_workspace(ref, name="Copy")
    archived = await adapter.archive_workspace(ref)
    restored = await adapter.restore_workspace(ref)

    assert created.ref.workspace_id == "local-new"
    assert renamed.ref == ref
    assert copied.ref.workspace_id == "local-copy"
    assert archived.archived is True
    assert restored.archived is False


@pytest.mark.asyncio
async def test_local_create_reuses_collision_free_registry_identity() -> None:
    service = RecordingLocalService()
    adapter = LocalResearchWorkspaceAdapter(service)

    created = await adapter.create_workspace(name="Created")

    assert created.ref.workspace_id == "workspace-local-1"
    assert all(thread_id != threading.get_ident() for thread_id in service.thread_ids)


@pytest.mark.asyncio
async def test_local_delete_is_settings_owned_and_returns_exact_capability() -> None:
    service = RecordingLocalService()
    adapter = LocalResearchWorkspaceAdapter(service)
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "local-1")

    with pytest.raises(CapabilityUnavailableError) as exc_info:
        await adapter.delete_workspace(ref)

    assert exc_info.value.capability == ResearchCapability(
        available=False,
        reason_code="settings_owned",
        user_message="Delete local workspaces from Settings.",
        owner="settings",
        recovery_action="Open Settings > Workspaces.",
    )
    assert service.calls == []


@pytest.mark.asyncio
async def test_local_adapter_rejects_server_ref_without_cross_call() -> None:
    service = RecordingLocalService()
    adapter = LocalResearchWorkspaceAdapter(service)
    server_ref = QualifiedWorkspaceRef(
        WorkspaceDataSource.SERVER,
        "server-1",
        server_profile_id="profile-1",
        principal_id="principal-1",
    )

    with pytest.raises(ValueError, match="Local adapter"):
        await adapter.get_workspace(server_ref)

    assert service.calls == []


@pytest.mark.asyncio
async def test_local_get_rejects_mismatched_registry_result() -> None:
    class MismatchingLocalService(RecordingLocalService):
        def get_workspace(self, workspace_id: str):
            self._record("get", workspace_id)
            return self.records["workspace-default"]

    service = MismatchingLocalService()
    adapter = LocalResearchWorkspaceAdapter(service)
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "local-1")

    with pytest.raises(ValueError, match="mismatched workspace ref"):
        await adapter.get_workspace(ref)


@dataclass
class RecordingServerContextProvider:
    context: object | None = None
    error: Exception | None = None
    calls: int = 0

    def get_active_context(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.context


class RecordingServerService:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.rows = [{"id": "server-1", "name": "Remote", "version": 4}]

    async def list_workspaces(self):
        self.calls.append(("list",))
        return list(self.rows)

    async def save_workspace(self, **kwargs):
        self.calls.append(("save", kwargs))
        workspace_id = kwargs["workspace_id"]
        return {
            "id": workspace_id,
            "name": kwargs.get("name") or "Remote",
            "archived": bool(kwargs.get("archived", False)),
            "version": (kwargs.get("version") or 0) + 1,
        }

    async def delete_workspace(self, workspace_id: str):
        self.calls.append(("delete", workspace_id))
        return {"deleted": True}


def server_context(
    *, reachability: str = "reachable", auth_state: str = "authenticated"
) -> object:
    """Build capabilities through the real active-context projection."""

    runtime_context_provider = SimpleNamespace(
        runtime_context=SimpleNamespace(
            state=RuntimeSourceState(
                active_source="server",
                active_server_id="profile-1",
                server_configured=True,
                server_reachability=reachability,
                server_auth_state=auth_state,
                last_known_server_label="Server",
            )
        )
    )
    target = ConfiguredServerTarget(
        server_id="profile-1",
        label="Server",
        base_url="https://server.example/api",
        last_known_reachability=reachability,
        last_known_auth_state=auth_state,
    )
    capabilities = RuntimeServerContextProvider._build_capabilities(
        runtime_context_provider, target
    )

    return SimpleNamespace(
        active_server_id="profile-1",
        auth_token="not-a-secret-test-token",
        credential_source="test",
        capabilities=capabilities,
    )


@pytest.mark.asyncio
async def test_server_catalog_rows_include_profile_and_non_secret_principal() -> None:
    service = RecordingServerService()
    provider = RecordingServerContextProvider(context=server_context())
    adapter = ServerResearchWorkspaceAdapter(service, provider)

    rows = await adapter.list_workspaces()

    assert len(rows) == 1
    assert rows[0].ref.data_source is WorkspaceDataSource.SERVER
    assert rows[0].ref.server_profile_id == "profile-1"
    assert rows[0].ref.principal_id.startswith("credential-fingerprint:test:")
    assert "not-a-secret-test-token" not in rows[0].ref.principal_id


@pytest.mark.asyncio
async def test_server_adapter_rejects_local_ref_without_cross_call() -> None:
    service = RecordingServerService()
    provider = RecordingServerContextProvider(context=server_context())
    adapter = ServerResearchWorkspaceAdapter(service, provider)
    local_ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "local-1")

    with pytest.raises(ValueError, match="Server adapter"):
        await adapter.get_workspace(local_ref)

    assert provider.calls == 0
    assert service.calls == []


@pytest.mark.asyncio
async def test_server_missing_profile_fails_closed_without_local_fallback() -> None:
    service = RecordingServerService()
    provider = RecordingServerContextProvider(
        error=ServerContextUnavailable(
            "Active server profile is unavailable.",
            reason_code="server_profile_missing",
        )
    )
    adapter = ServerResearchWorkspaceAdapter(service, provider)

    with pytest.raises(CapabilityUnavailableError) as exc_info:
        await adapter.list_workspaces()

    assert exc_info.value.capability == ResearchCapability(
        available=False,
        reason_code="server_profile_missing",
        user_message="Active server profile is unavailable.",
        owner="server",
        recovery_action="Choose or configure a server profile.",
    )
    assert service.calls == []


@pytest.mark.asyncio
async def test_server_projects_audited_lifecycle_from_real_context_shape() -> None:
    service = RecordingServerService()
    provider = RecordingServerContextProvider(context=server_context())
    adapter = ServerResearchWorkspaceAdapter(service, provider)
    ref = (await adapter.list_workspaces())[0].ref
    capabilities = await adapter.capabilities(ref)

    assert set(capabilities) == {
        "list",
        "get",
        "create",
        "update",
        "duplicate",
        "archive",
        "restore",
        "delete",
        "list_sources",
        "search_catalog",
        "attach_existing",
        "remove_source",
        "update_source",
        "preview_source",
        "get_readiness",
        "set_selected_scope",
        "reorder_sources",
        "list_notes",
        "get_note",
        "save_note",
        "delete_note",
    }
    lifecycle_names = {
        "list",
        "get",
        "create",
        "update",
        "duplicate",
        "archive",
        "restore",
        "delete",
    }
    assert all(capabilities[name].available is True for name in lifecycle_names)
    assert all(
        capability.available is False
        for name, capability in capabilities.items()
        if name not in lifecycle_names
    )
    assert {capability.capability_revision for capability in capabilities.values()} == {
        "server-notes-workspace-service-v1:reachable:authenticated"
    }
    assert service.calls == [("list",)]


@pytest.mark.asyncio
async def test_server_projection_exposes_only_concrete_audited_service_methods() -> None:
    class ServiceWithoutDelete:
        async def list_workspaces(self):
            return [{"id": "server-1", "name": "Remote", "version": 4}]

        async def save_workspace(self, **kwargs):
            return {"id": kwargs["workspace_id"], "name": "Remote", "version": 1}

    adapter = ServerResearchWorkspaceAdapter(
        ServiceWithoutDelete(),  # type: ignore[arg-type]
        RecordingServerContextProvider(context=server_context()),
    )
    ref = (await adapter.list_workspaces())[0].ref

    capabilities = await adapter.capabilities(ref)

    assert capabilities["list"].available is True
    assert capabilities["create"].available is True
    assert capabilities["delete"].available is False
    assert capabilities["delete"].reason_code == "server_capability_unavailable"
    with pytest.raises(CapabilityUnavailableError) as exc_info:
        await adapter.delete_workspace(ref)
    assert exc_info.value.capability == capabilities["delete"]


@pytest.mark.asyncio
async def test_server_lifecycle_calls_use_exact_service_arguments() -> None:
    service = RecordingServerService()
    new_ids = iter(("server-new", "server-copy"))
    provider = RecordingServerContextProvider(context=server_context())
    adapter = ServerResearchWorkspaceAdapter(
        service, provider, id_factory=lambda: next(new_ids)
    )

    listed = await adapter.list_workspaces()
    ref = listed[0].ref
    fetched = await adapter.get_workspace(ref)
    created = await adapter.create_workspace(name="Created")
    updated = await adapter.update_workspace(
        ref, name="Renamed", expected_version=4
    )
    duplicated = await adapter.duplicate_workspace(ref, name="Copy")
    archived = await adapter.archive_workspace(ref, expected_version=4)
    restored = await adapter.restore_workspace(ref, expected_version=5)
    deleted = await adapter.delete_workspace(ref)

    assert fetched == listed[0]
    assert created.ref.workspace_id == "server-new"
    assert updated.name == "Renamed"
    assert duplicated.ref.workspace_id == "server-copy"
    assert archived.archived is True
    assert restored.archived is False
    assert deleted is True
    assert service.calls == [
        ("list",),
        ("list",),
        ("save", {"workspace_id": "server-new", "name": "Created"}),
        (
            "save",
            {"workspace_id": "server-1", "name": "Renamed", "version": 4},
        ),
        ("save", {"workspace_id": "server-copy", "name": "Copy"}),
        (
            "save",
            {"workspace_id": "server-1", "archived": True, "version": 4},
        ),
        (
            "save",
            {"workspace_id": "server-1", "archived": False, "version": 5},
        ),
        ("delete", "server-1"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["update", "archive", "restore"])
async def test_server_versioned_lifecycle_requires_expected_version(
    operation: str,
) -> None:
    service = RecordingServerService()
    provider = RecordingServerContextProvider(context=server_context())
    adapter = ServerResearchWorkspaceAdapter(service, provider)
    ref = (await adapter.list_workspaces())[0].ref

    with pytest.raises(CapabilityUnavailableError) as exc_info:
        if operation == "update":
            await adapter.update_workspace(ref, name="Changed")
        elif operation == "archive":
            await adapter.archive_workspace(ref)
        else:
            await adapter.restore_workspace(ref)

    assert exc_info.value.capability.reason_code == "version_required"
    assert service.calls == [("list",)]


@pytest.mark.asyncio
async def test_server_delete_rejects_unenforceable_expected_version() -> None:
    service = RecordingServerService()
    provider = RecordingServerContextProvider(context=server_context())
    adapter = ServerResearchWorkspaceAdapter(service, provider)
    ref = (await adapter.list_workspaces())[0].ref

    with pytest.raises(CapabilityUnavailableError) as exc_info:
        await adapter.delete_workspace(ref, expected_version=4)

    assert exc_info.value.capability.reason_code == "version_precondition_unavailable"
    assert exc_info.value.capability.owner == "server"
    assert service.calls == [("list",)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("context", "reason_code"),
    [
        (server_context(reachability="unreachable"), "server_unavailable"),
        (server_context(auth_state="auth_required"), "auth_required"),
        (server_context(auth_state="session_invalid"), "stale_authorization"),
    ],
)
async def test_server_context_health_disables_audited_lifecycle(
    context: object, reason_code: str
) -> None:
    service = RecordingServerService()
    adapter = ServerResearchWorkspaceAdapter(
        service, RecordingServerContextProvider(context=context)
    )

    with pytest.raises(CapabilityUnavailableError) as exc_info:
        await adapter.list_workspaces()

    assert exc_info.value.capability.reason_code == reason_code
    assert service.calls == []


@pytest.mark.asyncio
async def test_server_network_failure_is_typed_unavailable_for_detail() -> None:
    class OfflineServerService(RecordingServerService):
        async def list_workspaces(self):
            self.calls.append(("list",))
            raise ConnectionError("secret host details")

    service = OfflineServerService()
    provider = RecordingServerContextProvider(context=server_context())
    adapter = ServerResearchWorkspaceAdapter(service, provider)
    ref = QualifiedWorkspaceRef(
        WorkspaceDataSource.SERVER,
        "server-1",
        server_profile_id="profile-1",
        principal_id="credential-fingerprint:test:9d8cbe900ad878341fffc769",
    )

    with pytest.raises(CapabilityUnavailableError) as exc_info:
        await adapter.get_workspace(ref)

    assert exc_info.value.capability == ResearchCapability(
        available=False,
        reason_code="server_unavailable",
        user_message="The selected server is unavailable.",
        owner="server",
        recovery_action="Retry or change the selected server.",
        capability_revision=(
            "server-notes-workspace-service-v1:reachable:authenticated"
        ),
    )
    assert "secret host details" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_server_auth_failure_is_typed_unavailable_without_secret_copy() -> None:
    class UnauthenticatedServerService(RecordingServerService):
        async def list_workspaces(self):
            raise AuthenticationError("token secret-value expired")

    adapter = ServerResearchWorkspaceAdapter(
        UnauthenticatedServerService(),
        RecordingServerContextProvider(context=server_context()),
    )

    with pytest.raises(CapabilityUnavailableError) as exc_info:
        await adapter.list_workspaces()

    assert exc_info.value.capability.reason_code == "auth_required"
    assert exc_info.value.capability.recovery_action == (
        "Reauthenticate with the selected server."
    )
    assert "secret-value" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_server_policy_denial_carries_exact_permission_capability() -> None:
    class DeniedServerService(RecordingServerService):
        async def list_workspaces(self):
            raise PolicyDeniedError(
                action_id="notes.workspace.list.server",
                reason_code="wrong_source",
                user_message="Select Server workspace data.",
                effective_source="local",
                authority_owner="server",
            )

    adapter = ServerResearchWorkspaceAdapter(
        DeniedServerService(),
        RecordingServerContextProvider(context=server_context()),
    )

    with pytest.raises(CapabilityUnavailableError) as exc_info:
        await adapter.list_workspaces()

    assert exc_info.value.capability == ResearchCapability(
        available=False,
        reason_code="wrong_source",
        user_message="Select Server workspace data.",
        owner="server",
        recovery_action="Review server permissions and retry.",
        capability_revision=(
            "server-notes-workspace-service-v1:reachable:authenticated"
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "reason_code", "recovery_action"),
    [
        (403, "server_permission_denied", "Review server permissions and retry."),
        (
            404,
            "server_capability_unavailable",
            "Update the selected server or choose another action.",
        ),
    ],
)
async def test_server_api_denial_is_typed_and_payload_safe(
    status_code: int, reason_code: str, recovery_action: str
) -> None:
    class DeniedServerService(RecordingServerService):
        async def list_workspaces(self):
            raise APIResponseError(status_code, "secret server response")

    adapter = ServerResearchWorkspaceAdapter(
        DeniedServerService(),
        RecordingServerContextProvider(context=server_context()),
    )

    with pytest.raises(CapabilityUnavailableError) as exc_info:
        await adapter.list_workspaces()

    assert exc_info.value.capability.reason_code == reason_code
    assert exc_info.value.capability.recovery_action == recovery_action
    assert "secret server response" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_server_update_rejects_mismatched_result_ref() -> None:
    class MismatchingServerService(RecordingServerService):
        async def save_workspace(self, **kwargs):
            self.calls.append(("save", kwargs))
            return {"id": "different-workspace", "name": "Wrong", "version": 5}

    service = MismatchingServerService()
    provider = RecordingServerContextProvider(context=server_context())
    adapter = ServerResearchWorkspaceAdapter(service, provider)
    ref = (await adapter.list_workspaces())[0].ref

    with pytest.raises(ValueError, match="mismatched workspace ref"):
        await adapter.update_workspace(ref, name="Changed", expected_version=4)
