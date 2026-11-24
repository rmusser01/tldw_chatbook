"""Compose the authority adapters and build the workspace Copy preflight."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime

from tldw_chatbook.Workspaces.models import WorkspaceTransferPolicy

from .contracts import (
    MAX_RESEARCH_SELECTION_IDS,
    CapabilityUnavailableError,
    QualifiedWorkspaceRef,
    WorkspaceDataSource,
)
from .copy_adapters import CopyTextArtifact, LocalCopyNoteOwner, TextArtifactCopyOwner
from .copy_manifest import ResearchCopyCandidate, ResearchCopyPreflightBuilder
from .copy_models import (
    CopyConflictChoice,
    CopyItemKind,
    CopyOwnerRef,
    CopyTreatment,
    ResearchCopyItem,
    ResearchCopyManifest,
)
from .copy_receipts import CopyReceiptFailure, ResearchCopyReceiptStore
from .copy_service import CopyOwnerActionError, ResearchCopyService
from .quick_notes import ResearchNotePageRequest, ResearchQuickNote
from .server_migration_receipts import ServerMigrationReceiptService
from .studio_models import StudioOwnerKind


class ResearchWorkspaceCopy:
    """Capture one selected Local / Server pair without changing the active view."""

    def __init__(
        self, local_port, server_port, registry, chatbooks, *, policy_enforcer=None
    ) -> None:
        self.local = local_port
        self.server = server_port
        self.registry = registry
        self.policy_enforcer = policy_enforcer
        self.artifacts = (
            TextArtifactCopyOwner(
                registry,
                chatbooks,
                server_port.copy_client,
                policy_enforcer=policy_enforcer,
            )
            if chatbooks is not None
            else None
        )
        self.notes = LocalCopyNoteOwner(server_port, local_port)
        self.service = ResearchCopyService(
            ResearchCopyReceiptStore(registry.db),
            self._owner_for,
            assert_current=self._assert_current,
            migrations=ServerMigrationReceiptService(server_port.copy_client),
        )

    def _assert_current(self, manifest: ResearchCopyManifest) -> None:
        if manifest.destination.data_source is WorkspaceDataSource.SERVER:
            ref = manifest.destination
            if self.policy_enforcer is not None:
                self.policy_enforcer.require_allowed(
                    action_id="notes.workspace.create.server"
                )
                self.policy_enforcer.require_allowed(
                    action_id="notes.workspace.update.server"
                )
        else:
            ref = manifest.source
        try:
            self.server.copy_client(ref)
        except (CapabilityUnavailableError, ValueError) as exc:
            raise CopyOwnerActionError(
                CopyReceiptFailure.AUTHORITY_CHANGED, retryable=True
            ) from exc

    def _owner_for(self, item: ResearchCopyItem, target: QualifiedWorkspaceRef):
        if item.source.kind is CopyItemKind.ARTIFACT:
            if (
                item.conflict is CopyConflictChoice.REPLACE_DESTINATION
                and target.data_source is WorkspaceDataSource.SERVER
            ):
                return None
            if item.treatment in {CopyTreatment.CONTENT, CopyTreatment.REFERENCE}:
                return self.artifacts
        if (
            item.source.kind is CopyItemKind.NOTE
            and target.data_source is WorkspaceDataSource.LOCAL
        ):
            if (
                item.treatment is CopyTreatment.CONTENT
                and item.conflict is not CopyConflictChoice.REPLACE_DESTINATION
            ):
                return self.notes
        return None

    async def local_targets(self):
        return await self.local.list_workspaces()

    def new_server_target(self) -> QualifiedWorkspaceRef:
        return self.server.new_copy_workspace()

    async def _page_items(self, fetch: Callable) -> tuple:
        items = []
        offset = 0
        while True:
            page = await fetch(offset)
            items.extend(page.items)
            if not page.has_more:
                return tuple(items)
            if not page.items or len(items) >= MAX_RESEARCH_SELECTION_IDS:
                raise ValueError("Copy inspection exceeded the workspace page limit")
            offset += len(page.items)

    async def candidates(
        self, ref: QualifiedWorkspaceRef
    ) -> tuple[ResearchCopyCandidate, ...]:
        port = (
            self.local if ref.data_source is WorkspaceDataSource.LOCAL else self.server
        )
        candidates = {}
        for source in await self._page_items(
            lambda offset: port.list_sources(ref, limit=100, offset=offset)
        ):
            owner = CopyOwnerRef(
                ref,
                CopyItemKind.MEDIA,
                source.catalog_item_id,
                source.catalog_item_version,
            )
            candidates[owner.kind, owner.owner_id] = ResearchCopyCandidate(
                owner, membership_id=source.source_id
            )
        try:
            notes = await self._page_items(
                lambda offset: port.list_notes(
                    ref, ResearchNotePageRequest(limit=100, offset=offset)
                )
            )
        except CapabilityUnavailableError:
            notes = ()
        for note in notes:
            owner = CopyOwnerRef(ref, CopyItemKind.NOTE, note.note_id, note.version)
            candidates[owner.kind, owner.owner_id] = ResearchCopyCandidate(
                owner, estimated_bytes=len(note.content.encode("utf-8"))
            )
        try:
            outputs = await self._page_items(
                lambda offset: port.list_outputs(ref, limit=100, offset=offset)
            )
        except CapabilityUnavailableError:
            outputs = ()
        for output in outputs:
            if output.workspace != ref:
                raise ValueError("The output list returned a different workspace")
            if output.owner_kind in {
                StudioOwnerKind.LOCAL_STUDY_DECK,
                StudioOwnerKind.SERVER_STUDY_DECK,
            }:
                kind = CopyItemKind.STUDY_DECK
            elif output.owner_kind in {
                StudioOwnerKind.LOCAL_QUIZ,
                StudioOwnerKind.SERVER_QUIZ,
            }:
                kind = CopyItemKind.QUIZ
            else:
                kind = CopyItemKind.ARTIFACT
            owner = CopyOwnerRef(ref, kind, output.owner_id, output.owner_version)
            candidates[kind, owner.owner_id] = ResearchCopyCandidate(owner)
        return tuple(candidates.values())

    async def _local_policy(
        self, candidate: ResearchCopyCandidate
    ) -> WorkspaceTransferPolicy:
        ref = candidate.source
        item_type, role = {
            CopyItemKind.MEDIA: ("media", "source"),
            CopyItemKind.NOTE: ("note", "note"),
            CopyItemKind.ARTIFACT: ("chatbook", "output"),
            CopyItemKind.STUDY_DECK: ("study_deck", "output"),
            CopyItemKind.QUIZ: ("study_quiz", "output"),
        }.get(ref.kind, ("", ""))
        if not item_type:
            return candidate.transfer_policy
        links = await self.service._off_loop(
            self.registry.get_item_memberships, item_type, ref.owner_id
        )
        match = next(
            (
                link
                for link in links
                if link.workspace_id == ref.workspace.workspace_id and link.role == role
            ),
            None,
        )
        if match and match.transfer_policy is WorkspaceTransferPolicy.LOCAL_ONLY:
            return WorkspaceTransferPolicy.LOCAL_ONLY
        return WorkspaceTransferPolicy(candidate.transfer_policy)

    async def preflight_copy(
        self,
        source: QualifiedWorkspaceRef,
        destination: QualifiedWorkspaceRef,
        candidates: tuple[ResearchCopyCandidate, ...] | None = None,
        *,
        target_name: str = "",
    ) -> ResearchCopyManifest:
        if source.data_source is destination.data_source:
            raise ValueError("Copy requires workspaces in different authorities")
        server_ref = (
            destination
            if destination.data_source is WorkspaceDataSource.SERVER
            else source
        )
        self.server.copy_client(server_ref)
        available = {}
        replacements = frozenset()
        if self.artifacts is not None:
            available[CopyItemKind.ARTIFACT] = frozenset({CopyTreatment.CONTENT})
        if destination.data_source is WorkspaceDataSource.LOCAL:
            workspace = await self.local.get_workspace(destination)
            if not workspace or workspace.archived:
                raise ValueError("Choose an active Local workspace as the destination")
            capabilities = await self.local.capabilities(destination)
            if (capability := capabilities.get("save_note")) and capability.available:
                available[CopyItemKind.NOTE] = frozenset({CopyTreatment.CONTENT})
            if self.artifacts is not None:
                replacements = frozenset({CopyItemKind.ARTIFACT})
                available[CopyItemKind.ARTIFACT] = frozenset(
                    {CopyTreatment.CONTENT, CopyTreatment.REFERENCE}
                )
        rows = await self.candidates(source) if candidates is None else candidates
        projected = []
        for candidate in rows:
            owner = (
                self.artifacts
                if candidate.source.kind is CopyItemKind.ARTIFACT
                else self.notes
            )
            if candidate.source.workspace != source:
                raise ValueError("The selected item belongs to another workspace")
            if source.data_source is WorkspaceDataSource.LOCAL:
                candidate = replace(
                    candidate, transfer_policy=await self._local_policy(candidate)
                )
            if (
                candidate.source.kind is CopyItemKind.ARTIFACT
                and candidate.destination is not None
                and self.artifacts is not None
            ):
                candidate = replace(
                    candidate,
                    replacement_available=await self.artifacts.can_replace(
                        candidate.destination
                    ),
                )
            if (
                candidate.source.kind in available
                and candidate.transfer_policy is not WorkspaceTransferPolicy.LOCAL_ONLY
            ):
                try:
                    snapshot = await owner.inspect_source(candidate.source)
                except CopyOwnerActionError:
                    candidate = replace(
                        candidate,
                        source=replace(
                            candidate.source, version=None, snapshot_hash=""
                        ),
                    )
                else:
                    body = snapshot.body
                    size = (
                        len(body.content.encode("utf-8"))
                        if isinstance(body, (ResearchQuickNote, CopyTextArtifact))
                        else 0
                    )
                    candidate = replace(
                        candidate, source=snapshot.owner, estimated_bytes=size
                    )
            projected.append(candidate)
        self.server.copy_client(server_ref)
        return ResearchCopyPreflightBuilder().build(
            source,
            destination,
            tuple(projected),
            available=available,
            versioned_replacements=replacements,
            created_at=datetime.now(UTC).isoformat(),
            target_name=target_name,
        )
