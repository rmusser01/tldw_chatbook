"""Copy supported content through the existing qualified canonical owners."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from uuid import NAMESPACE_URL, UUID, uuid5

from tldw_chatbook.Backup_Recovery.participants import run_finite_local_worker
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.Notes.server_notes_workspace_service import (
    ServerNotesWorkspaceService,
)
from tldw_chatbook.Workspaces.models import WorkspaceTransferPolicy
from tldw_chatbook.tldw_api.exceptions import APIResponseError

from .chat_service import MAX_ANSWER_BYTES
from .contracts import QualifiedWorkspaceRef, WorkspaceDataSource
from .copy_models import CopyItemKind, CopyOwnerRef, ResearchCopyItem
from .copy_receipts import CopyReceiptFailure
from .copy_service import CopyOwnerActionError, CopySourceSnapshot
from .quick_notes import ResearchNoteSaveRequest, ResearchQuickNote


_TEXT_ARTIFACTS = frozenset({"summary", "report", "compare_sources"})


@dataclass(frozen=True, slots=True)
class CopyTextArtifact:
    title: str
    artifact_type: str
    content: str = field(repr=False)

    def __post_init__(self) -> None:
        if (
            self.artifact_type not in _TEXT_ARTIFACTS
            or not self.title
            or len(self.title) > 256
        ):
            raise CopyOwnerActionError(CopyReceiptFailure.OWNER_UNAVAILABLE)
        if (
            not self.content.strip()
            or len(self.content.encode("utf-8")) > MAX_ANSWER_BYTES
        ):
            raise CopyOwnerActionError(CopyReceiptFailure.OWNER_UNAVAILABLE)


async def _off_loop(function, *args, **kwargs):
    return await asyncio.to_thread(run_finite_local_worker, function, *args, **kwargs)


class LocalCopyNoteOwner:
    """Read a Server workspace note and use Local Notes' idempotent create."""

    def __init__(self, server_port, local_port) -> None:
        self.server_port = server_port
        self.local_port = local_port

    async def inspect_source(self, ref: CopyOwnerRef) -> CopySourceSnapshot:
        if (
            ref.workspace.data_source is not WorkspaceDataSource.SERVER
            or ref.kind is not CopyItemKind.NOTE
        ):
            raise CopyOwnerActionError(CopyReceiptFailure.OWNER_UNAVAILABLE)
        note = await self.server_port.get_note(ref.workspace, ref.owner_id)
        if (
            not isinstance(note, ResearchQuickNote)
            or note.ref != ref.workspace
            or note.note_id != ref.owner_id
        ):
            raise CopyOwnerActionError(CopyReceiptFailure.SOURCE_CHANGED)
        return CopySourceSnapshot(
            replace(ref, version=note.version, snapshot_hash=""), note
        )

    async def create_destination(
        self,
        item: ResearchCopyItem,
        source: CopySourceSnapshot,
        workspace: QualifiedWorkspaceRef,
        item_key: str,
    ) -> CopyOwnerRef:
        if (
            workspace.data_source is not WorkspaceDataSource.LOCAL
            or not isinstance(source.body, ResearchQuickNote)
            or source.owner != item.source
            or item.treatment != "copy_content"
        ):
            raise CopyOwnerActionError(CopyReceiptFailure.OWNER_UNAVAILABLE)
        note = await self.local_port.save_note(
            workspace,
            ResearchNoteSaveRequest(
                title=source.body.title,
                content=source.body.content,
                operation_id=f"research-note-{UUID(bytes=bytes.fromhex(item_key)[:16], version=4).hex}",
            ),
        )
        if (
            note.ref != workspace
            or note.title != (source.body.title.strip() or "Untitled Note")
            or note.content != source.body.content
        ):
            raise CopyOwnerActionError(CopyReceiptFailure.VERIFICATION_FAILED)
        return CopyOwnerRef(workspace, item.source.kind, note.note_id, note.version)

    async def replace_destination(self, item, source, item_key):
        raise CopyOwnerActionError(CopyReceiptFailure.OWNER_UNAVAILABLE)

    async def reference_destination(self, item, source):
        raise CopyOwnerActionError(CopyReceiptFailure.OWNER_UNAVAILABLE)

    async def associate_destination(
        self, item: ResearchCopyItem, destination: CopyOwnerRef
    ) -> None:
        if not await self.verify_destination(item, destination):
            raise CopyOwnerActionError(CopyReceiptFailure.ASSOCIATION_FAILED)

    async def verify_destination(
        self, item: ResearchCopyItem, destination: CopyOwnerRef
    ) -> bool:
        if (
            destination.workspace.data_source is not WorkspaceDataSource.LOCAL
            or destination.kind is not CopyItemKind.NOTE
        ):
            return False
        note = await self.local_port.get_note(
            destination.workspace, destination.owner_id
        )
        return (
            isinstance(note, ResearchQuickNote)
            and note.ref == destination.workspace
            and note.note_id == destination.owner_id
            and note.version == destination.version
        )


class TextArtifactCopyOwner:
    """Copy bounded artifact text through the Chatbook and workspace artifact owners."""

    def __init__(
        self, registry, chatbooks, client_for, *, policy_enforcer=None
    ) -> None:
        self.registry = registry
        self.chatbooks = chatbooks
        self.client_for = client_for
        self.policy_enforcer = policy_enforcer

    def _server(self, ref: QualifiedWorkspaceRef):
        client = self.client_for(ref)
        service = ServerNotesWorkspaceService(
            client, policy_enforcer=self.policy_enforcer
        )
        return client, NotesScopeService(
            local_notes_service=None,
            server_service=service,
            policy_enforcer=self.policy_enforcer,
        )

    def _check_server(self, ref: QualifiedWorkspaceRef, client) -> None:
        if self.client_for(ref) is not client:
            raise CopyOwnerActionError(
                CopyReceiptFailure.AUTHORITY_CHANGED, retryable=True
            )

    async def _associated(self, ref: CopyOwnerRef, *, outbound: bool = False) -> bool:
        links = await _off_loop(
            self.registry.get_item_memberships, "chatbook", ref.owner_id
        )
        return any(
            link.workspace_id == ref.workspace.workspace_id
            and link.role == "output"
            and (not outbound or link.transfer_policy is WorkspaceTransferPolicy.COPY)
            for link in links
        )

    @staticmethod
    def _body(ref: CopyOwnerRef, row, *, local: bool) -> CopyTextArtifact:
        if local:
            metadata = row.get("metadata") or {}
            title, kind, content = (
                row.get("name"),
                metadata.get("artifact_kind"),
                metadata.get("content"),
            )
        else:
            title, kind, content = (
                row.get("title"),
                row.get("artifact_type"),
                row.get("content"),
            )
        if not all(isinstance(value, str) for value in (title, kind, content)):
            raise CopyOwnerActionError(CopyReceiptFailure.OWNER_UNAVAILABLE)
        return CopyTextArtifact(title, kind, content)

    async def _read_server(self, ref: CopyOwnerRef, client, service):
        rows = await service.list_workspace_artifacts(
            workspace_id=ref.workspace.workspace_id
        )
        self._check_server(ref.workspace, client)
        for row in rows:
            if row.get("workspace_id") != ref.workspace.workspace_id:
                raise CopyOwnerActionError(CopyReceiptFailure.AUTHORITY_CHANGED)
            if row.get("id") == ref.owner_id:
                return row
        return None

    async def _read(self, ref: CopyOwnerRef, *, require_link: bool = True):
        if ref.kind is not CopyItemKind.ARTIFACT:
            raise CopyOwnerActionError(CopyReceiptFailure.OWNER_UNAVAILABLE)
        if ref.workspace.data_source is WorkspaceDataSource.LOCAL:
            if require_link and not await self._associated(ref):
                return None
            try:
                return await self.chatbooks.get_chatbook(ref.owner_id)
            except KeyError:
                return None
        client, service = self._server(ref.workspace)
        return await self._read_server(ref, client, service)

    async def inspect_source(self, ref: CopyOwnerRef) -> CopySourceSnapshot:
        local = ref.workspace.data_source is WorkspaceDataSource.LOCAL
        if local and not await self._associated(ref, outbound=True):
            raise CopyOwnerActionError(CopyReceiptFailure.SOURCE_CHANGED)
        row = await self._read(ref, require_link=not local)
        if not row or (not local and row.get("status") != "complete"):
            raise CopyOwnerActionError(CopyReceiptFailure.SOURCE_CHANGED)
        version = row.get("artifact_revision" if local else "version")
        if type(version) is not int or version < 1:
            raise CopyOwnerActionError(CopyReceiptFailure.OWNER_UNAVAILABLE)
        return CopySourceSnapshot(
            replace(ref, version=version, snapshot_hash=""),
            self._body(ref, row, local=local),
        )

    @staticmethod
    def _local_fields(body: CopyTextArtifact, **metadata) -> dict:
        return {
            "name": body.title,
            "description": "Copied Research Studio output",
            "tags": ["research", "artifact"],
            "categories": ["Research", "Artifacts"],
            "metadata": {
                "artifact_source": "research",
                "artifact_kind": body.artifact_type,
                "content": body.content,
                **metadata,
            },
        }

    async def create_destination(
        self,
        item: ResearchCopyItem,
        source: CopySourceSnapshot,
        workspace: QualifiedWorkspaceRef,
        item_key: str,
    ) -> CopyOwnerRef:
        if (
            source.owner != item.source
            or not isinstance(source.body, CopyTextArtifact)
            or item.treatment != "copy_content"
        ):
            raise CopyOwnerActionError(CopyReceiptFailure.OWNER_UNAVAILABLE)
        body = source.body
        if workspace.data_source is WorkspaceDataSource.LOCAL:
            await self._require_local_target(workspace)
            row = await self.chatbooks.create_chatbook(
                copy_operation_key=item_key,
                **self._local_fields(body),
            )
            return CopyOwnerRef(
                workspace, item.source.kind, str(row["id"]), row["artifact_revision"]
            )

        if not await self._associated(item.source, outbound=True):
            raise CopyOwnerActionError(CopyReceiptFailure.SOURCE_CHANGED)
        client, service = self._server(workspace)
        artifact_id = str(
            uuid5(
                NAMESPACE_URL, f"tldw:research-copy:{workspace.workspace_id}:{item_key}"
            )
        )
        target = CopyOwnerRef(workspace, item.source.kind, artifact_id)
        proof = {"item_key": item_key, "source_revision": item.source.version}
        row = await self._read_server(target, client, service)
        if row is None:
            try:
                row = await service.save_workspace_artifact(
                    workspace_id=workspace.workspace_id,
                    artifact_id=artifact_id,
                    artifact_type=body.artifact_type,
                    title=body.title,
                    status="complete",
                    content=body.content,
                    producer_metadata={"research_copy": proof},
                )
            except APIResponseError as exc:
                if exc.status_code != 409:
                    raise
                row = await self._read_server(target, client, service)
        self._check_server(workspace, client)
        if (
            not row
            or row.get("id") != artifact_id
            or row.get("workspace_id") != workspace.workspace_id
            or row.get("producer_metadata", {}).get("research_copy") != proof
            or row.get("status") != "complete"
            or type(row.get("version")) is not int
            or row["version"] < 1
            or self._body(target, row, local=False) != body
        ):
            raise CopyOwnerActionError(CopyReceiptFailure.VERIFICATION_FAILED)
        return replace(target, version=row["version"])

    async def replace_destination(
        self,
        item: ResearchCopyItem,
        source: CopySourceSnapshot,
        item_key: str,
    ) -> CopyOwnerRef:
        target = item.destination
        if (
            target is None
            or target.workspace.data_source is not WorkspaceDataSource.LOCAL
            or not target.version
            or not isinstance(source.body, CopyTextArtifact)
            or source.owner != item.source
        ):
            raise CopyOwnerActionError(CopyReceiptFailure.OWNER_UNAVAILABLE)
        row = await self._read(target)
        if not row:
            raise CopyOwnerActionError(CopyReceiptFailure.DESTINATION_CHANGED)
        await self._require_local_target(target.workspace)
        if row.get("metadata", {}).get(
            "artifact_source"
        ) != "research" or not await self.can_replace(target):
            raise CopyOwnerActionError(CopyReceiptFailure.OWNER_UNAVAILABLE)
        before = row.get("artifact_revision")
        body = source.body
        if (
            before == target.version + 1
            and row.get("metadata", {}).get("research_copy_replacement") == item_key
            and self._body(target, row, local=True) == body
        ):
            return replace(target, version=before)
        if before != target.version:
            raise CopyOwnerActionError(CopyReceiptFailure.DESTINATION_CHANGED)
        saved = await self.chatbooks.update_chatbook(
            target.owner_id,
            expected_revision=target.version,
            **self._local_fields(body, research_copy_replacement=item_key),
        )
        ver = saved.get("artifact_revision")
        if type(ver) is not int or ver != target.version + 1:
            raise CopyOwnerActionError(CopyReceiptFailure.VERIFICATION_FAILED)
        return replace(target, version=ver)

    async def can_replace(self, target: CopyOwnerRef) -> bool:
        if target.workspace.data_source is not WorkspaceDataSource.LOCAL:
            return False
        links = await _off_loop(
            self.registry.get_item_memberships, "chatbook", target.owner_id
        )
        return len(links) == 1 and (
            links[0].workspace_id == target.workspace.workspace_id
            and links[0].role == "output"
            and not links[0].metadata
        )

    async def current_local_destination(self, target: CopyOwnerRef) -> CopyOwnerRef:
        """Refresh a previously copied destination through its canonical owner."""
        if target.workspace.data_source is not WorkspaceDataSource.LOCAL:
            raise ValueError("Choose a Local artifact destination")
        row = await self._read(target)
        version = row.get("artifact_revision") if row else None
        return replace(target, version=version, snapshot_hash="")

    async def reference_destination(
        self, item: ResearchCopyItem, source: CopySourceSnapshot
    ) -> CopyOwnerRef:
        target = item.destination
        if target is None:
            raise CopyOwnerActionError(CopyReceiptFailure.OWNER_UNAVAILABLE)
        row = await self._read(target)
        if (
            not row
            or row.get("artifact_revision") != target.version
            or self._body(target, row, local=True) != source.body
        ):
            raise CopyOwnerActionError(CopyReceiptFailure.DESTINATION_CHANGED)
        return target

    async def associate_destination(
        self, item: ResearchCopyItem, destination: CopyOwnerRef
    ) -> None:
        if (
            destination.workspace.data_source is WorkspaceDataSource.LOCAL
            and not await self._associated(destination)
        ):
            row = await self._read(destination, require_link=False)
            if not row or row.get("artifact_revision") != destination.version:
                raise CopyOwnerActionError(CopyReceiptFailure.DESTINATION_CHANGED)
            await _off_loop(
                self.registry.link_membership,
                destination.workspace.workspace_id,
                item_type="chatbook",
                item_id=destination.owner_id,
                role="output",
                transfer_policy=WorkspaceTransferPolicy.COPY,
                title=row["name"],
            )

    async def verify_destination(
        self, item: ResearchCopyItem, destination: CopyOwnerRef
    ) -> bool:
        row = await self._read(destination)
        if row is None:
            return False
        version_field = (
            "artifact_revision"
            if destination.workspace.data_source is WorkspaceDataSource.LOCAL
            else "version"
        )
        return row.get(version_field) == destination.version

    async def _require_local_target(self, ref: QualifiedWorkspaceRef) -> None:
        workspace = await _off_loop(self.registry.get_workspace, ref.workspace_id)
        if workspace is None or workspace.archived:
            raise CopyOwnerActionError(CopyReceiptFailure.DESTINATION_CHANGED)
