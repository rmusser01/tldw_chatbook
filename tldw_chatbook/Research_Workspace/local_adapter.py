"""Local Research Workspace adapter over the existing workspace registry."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping

from tldw_chatbook.Workspaces import DEFAULT_WORKSPACE_ID
from tldw_chatbook.Workspaces.models import WorkspaceRecord
from tldw_chatbook.Workspaces.registry_service import (
    LocalWorkspaceRegistryService,
    next_local_workspace_identity,
)

from .contracts import (
    CapabilityUnavailableError,
    QualifiedWorkspaceRef,
    ResearchCapability,
    ResearchWorkspaceSummary,
    WorkspaceDataSource,
    require_capability,
)


_LOCAL_AVAILABLE = ResearchCapability(
    available=True,
    reason_code="available",
    user_message="Available in Local workspaces.",
    owner="local",
)
_LOCAL_DELETE = ResearchCapability(
    available=False,
    reason_code="settings_owned",
    user_message="Delete local workspaces from Settings.",
    owner="settings",
    recovery_action="Open Settings > Workspaces.",
)
_LOCAL_CAPABILITIES: Mapping[str, ResearchCapability] = {
    "list": _LOCAL_AVAILABLE,
    "get": _LOCAL_AVAILABLE,
    "create": _LOCAL_AVAILABLE,
    "update": _LOCAL_AVAILABLE,
    "duplicate": _LOCAL_AVAILABLE,
    "archive": _LOCAL_AVAILABLE,
    "restore": _LOCAL_AVAILABLE,
    "delete": _LOCAL_DELETE,
}


class LocalResearchWorkspaceAdapter:
    """Expose Research notebook lifecycle without changing local ownership."""

    def __init__(
        self,
        service: LocalWorkspaceRegistryService,
        *,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._service = service
        self._id_factory = id_factory

    async def list_workspaces(
        self, *, include_archived: bool = False
    ) -> tuple[ResearchWorkspaceSummary, ...]:
        require_capability(_LOCAL_CAPABILITIES, "list")
        records = await asyncio.to_thread(
            self._service.list_workspaces, include_archived=include_archived
        )
        return tuple(
            self._summary(record)
            for record in records
            if record.workspace_id != DEFAULT_WORKSPACE_ID
        )

    async def get_workspace(
        self, ref: QualifiedWorkspaceRef
    ) -> ResearchWorkspaceSummary | None:
        self._require_local_ref(ref)
        require_capability(_LOCAL_CAPABILITIES, "get")
        record = await asyncio.to_thread(self._service.get_workspace, ref.workspace_id)
        return self._matching_summary(ref, record) if record is not None else None

    async def create_workspace(
        self, *, name: str, description: str = "", template_id: str = ""
    ) -> ResearchWorkspaceSummary:
        require_capability(_LOCAL_CAPABILITIES, "create")
        if template_id.strip():
            raise CapabilityUnavailableError(
                ResearchCapability(
                    available=False,
                    reason_code="template_unavailable",
                    user_message="Local workspace templates are not available.",
                    owner="local",
                    recovery_action="Create a blank workspace.",
                )
            )
        record = await asyncio.to_thread(
            self._service.create_workspace,
            workspace_id=await self._next_workspace_id(),
            name=name,
            description=description,
        )
        return self._summary(record)

    async def update_workspace(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        name: str | None = None,
        expected_version: int | None = None,
    ) -> ResearchWorkspaceSummary:
        self._require_local_ref(ref)
        require_capability(_LOCAL_CAPABILITIES, "update")
        if name is None:
            current = await self.get_workspace(ref)
            if current is None:
                raise ValueError(f"Workspace not found: {ref.workspace_id}")
            return current
        record = await asyncio.to_thread(
            self._service.rename_workspace, ref.workspace_id, name
        )
        return self._matching_summary(ref, record)

    async def duplicate_workspace(
        self, ref: QualifiedWorkspaceRef, *, name: str
    ) -> ResearchWorkspaceSummary:
        self._require_local_ref(ref)
        require_capability(_LOCAL_CAPABILITIES, "duplicate")
        source = await asyncio.to_thread(self._service.get_workspace, ref.workspace_id)
        if source is None:
            raise ValueError(f"Workspace not found: {ref.workspace_id}")
        record = await asyncio.to_thread(
            self._service.create_workspace,
            workspace_id=await self._next_workspace_id(),
            name=name,
            description=source.description,
        )
        return self._summary(record)

    async def archive_workspace(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        expected_version: int | None = None,
    ) -> ResearchWorkspaceSummary:
        self._require_local_ref(ref)
        require_capability(_LOCAL_CAPABILITIES, "archive")
        record = await asyncio.to_thread(
            self._service.archive_workspace, ref.workspace_id
        )
        return self._matching_summary(ref, record)

    async def restore_workspace(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        expected_version: int | None = None,
    ) -> ResearchWorkspaceSummary:
        self._require_local_ref(ref)
        require_capability(_LOCAL_CAPABILITIES, "restore")
        record = await asyncio.to_thread(
            self._service.unarchive_workspace, ref.workspace_id
        )
        return self._matching_summary(ref, record)

    async def delete_workspace(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        expected_version: int | None = None,
    ) -> bool:
        self._require_local_ref(ref)
        require_capability(_LOCAL_CAPABILITIES, "delete")
        raise AssertionError("unreachable")

    async def capabilities(
        self, ref: QualifiedWorkspaceRef
    ) -> Mapping[str, ResearchCapability]:
        self._require_local_ref(ref)
        return _LOCAL_CAPABILITIES

    async def _next_workspace_id(self) -> str:
        if self._id_factory is not None:
            return self._id_factory()
        workspace_id, _ = await asyncio.to_thread(
            next_local_workspace_identity, self._service
        )
        return workspace_id

    @staticmethod
    def _require_local_ref(ref: QualifiedWorkspaceRef) -> None:
        if ref.data_source is not WorkspaceDataSource.LOCAL:
            raise ValueError("Local adapter requires a Local workspace ref")

    @staticmethod
    def _summary(record: WorkspaceRecord) -> ResearchWorkspaceSummary:
        return ResearchWorkspaceSummary(
            ref=QualifiedWorkspaceRef(
                WorkspaceDataSource.LOCAL, record.workspace_id
            ),
            name=record.name,
            description=record.description,
            archived=record.archived,
            updated_at=record.updated_at,
        )

    def _matching_summary(
        self, expected_ref: QualifiedWorkspaceRef, record: WorkspaceRecord
    ) -> ResearchWorkspaceSummary:
        summary = self._summary(record)
        if summary.ref != expected_ref:
            raise ValueError("Adapter returned a mismatched workspace ref")
        return summary
