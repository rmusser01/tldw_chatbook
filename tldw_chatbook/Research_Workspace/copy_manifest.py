"""Project the Copy preflight from explicit source and destination capabilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from tldw_chatbook.Workspaces.models import WorkspaceTransferPolicy

from .contracts import QualifiedWorkspaceRef
from .copy_models import (
    CopyConflictChoice,
    CopyItemKind,
    CopyOwnerRef,
    CopyReason,
    CopyTreatment,
    ResearchCopyItem,
    ResearchCopyManifest,
)


_POLICY_TREATMENTS = {
    WorkspaceTransferPolicy.COPY: CopyTreatment.CONTENT,
    WorkspaceTransferPolicy.REFERENCE: CopyTreatment.REFERENCE,
    WorkspaceTransferPolicy.METADATA_ONLY: CopyTreatment.METADATA_ONLY,
    WorkspaceTransferPolicy.LOCAL_ONLY: CopyTreatment.OMIT,
}


def copy_treatment(policy: WorkspaceTransferPolicy) -> CopyTreatment:
    """Resolve an outbound membership policy to its content treatment."""
    return _POLICY_TREATMENTS[WorkspaceTransferPolicy(policy)]


_OMIT_CHOICES = {
    CopyConflictChoice.KEEP_DESTINATION,
    CopyConflictChoice.OMIT,
    CopyConflictChoice.CANCEL,
}


@dataclass(frozen=True, slots=True)
class ResearchCopyCandidate:
    source: CopyOwnerRef
    transfer_policy: WorkspaceTransferPolicy = WorkspaceTransferPolicy.COPY
    destination: CopyOwnerRef | None = None
    conflict: CopyConflictChoice | None = None
    membership_id: str = ""
    estimated_bytes: int = 0
    replacement_available: bool = True


class ResearchCopyPreflightBuilder:
    """Freeze the decisions without fetching bodies or mutating either owner."""

    def build(
        self,
        source: QualifiedWorkspaceRef,
        destination: QualifiedWorkspaceRef,
        candidates: tuple[ResearchCopyCandidate, ...],
        *,
        available: Mapping[CopyItemKind, frozenset[CopyTreatment]],
        versioned_replacements: frozenset[CopyItemKind] = frozenset(),
        created_at: str,
        target_name: str = "",
    ) -> ResearchCopyManifest:
        items = tuple(
            self._item(candidate, available, versioned_replacements)
            for candidate in candidates
        )
        return ResearchCopyManifest(source, destination, items, created_at, target_name)

    @staticmethod
    def _item(candidate, available, versioned_replacements) -> ResearchCopyItem:
        if not isinstance(candidate, ResearchCopyCandidate):
            raise TypeError("Copy preflight requires canonical item candidates")
        source, destination = candidate.source, candidate.destination
        if not isinstance(source, CopyOwnerRef) or (
            destination is not None and not isinstance(destination, CopyOwnerRef)
        ):
            raise TypeError("A Copy candidate requires qualified owner references")
        conflict = (
            CopyConflictChoice(candidate.conflict) if candidate.conflict else None
        )
        allowed = available.get(source.kind, frozenset())
        treatment = copy_treatment(candidate.transfer_policy)
        if treatment is CopyTreatment.OMIT:
            reason = CopyReason.LOCAL_ONLY
        elif conflict in _OMIT_CHOICES:
            treatment, reason = CopyTreatment.OMIT, CopyReason.NONE
        else:
            if conflict is CopyConflictChoice.REFERENCE_EXISTING:
                treatment = CopyTreatment.REFERENCE
            reason = ResearchCopyPreflightBuilder._blocked_reason(
                source,
                destination,
                conflict,
                treatment,
                allowed,
                versioned_replacements
                if candidate.replacement_available
                else frozenset(),
            )
            if reason is not CopyReason.NONE:
                treatment = CopyTreatment.BLOCKED
        return ResearchCopyItem(
            source,
            treatment,
            destination,
            conflict,
            candidate.membership_id,
            candidate.estimated_bytes,
            reason,
        )

    @staticmethod
    def _blocked_reason(
        source, destination, conflict, treatment, allowed, versioned_replacements
    ) -> CopyReason:
        if source.kind is CopyItemKind.CONVERSATION or treatment not in allowed:
            return CopyReason.OWNER_UNSUPPORTED
        if not source.has_revision:
            return CopyReason.SOURCE_REVISION_REQUIRED
        if destination is not None and conflict is None:
            return CopyReason.CHOOSE_CONFLICT
        if conflict is CopyConflictChoice.REPLACE_DESTINATION and (
            source.kind not in versioned_replacements
            or destination is None
            or not destination.version
            or treatment is not CopyTreatment.CONTENT
        ):
            return CopyReason.REPLACE_UNAVAILABLE
        if treatment is CopyTreatment.REFERENCE and (
            destination is None or not destination.has_revision
        ):
            return CopyReason.OWNER_UNSUPPORTED
        return CopyReason.NONE
