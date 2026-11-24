"""Immutable owner references, manifests, and confirmations for Research Copy."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
import json
import re

from .contracts import (
    MAX_RESEARCH_SELECTION_IDS,
    QualifiedWorkspaceRef,
    WorkspaceDataSource,
    _safe_identity,
)


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@-]{0,255}\Z")
_HASH = re.compile(r"[a-f0-9]{64}\Z")


class CopyTreatment(StrEnum):
    CONTENT = "copy_content"
    REFERENCE = "reference"
    METADATA_ONLY = "metadata_only"
    OMIT = "omit"
    BLOCKED = "blocked"

    @property
    def actionable(self) -> bool:
        return self in {self.CONTENT, self.REFERENCE, self.METADATA_ONLY}


class CopyConflictChoice(StrEnum):
    KEEP_DESTINATION = "keep_destination"
    REPLACE_DESTINATION = "replace_destination"
    COPY_AS_NEW = "copy_as_new"
    REFERENCE_EXISTING = "reference_existing"
    OMIT = "omit"
    CANCEL = "cancel"


class CopyItemKind(StrEnum):
    MEDIA = "media"
    NOTE = "note"
    ARTIFACT = "artifact"
    STUDY_DECK = "study_deck"
    QUIZ = "quiz"
    CONVERSATION = "conversation"


class CopyReason(StrEnum):
    NONE = ""
    OWNER_UNSUPPORTED = "owner_unsupported"
    LOCAL_ONLY = "local_only"
    CHOOSE_CONFLICT = "choose_conflict"
    REPLACE_UNAVAILABLE = "replace_unavailable"
    SOURCE_REVISION_REQUIRED = "source_revision_required"


class CopyRedaction(StrEnum):
    OWNER_FIELDS = "owner_fields"
    METADATA_ONLY = "metadata_only"
    REFERENCE_ONLY = "reference_only"
    EXCLUDED = "excluded"


class ResearchCopyStatus(StrEnum):
    PREFLIGHT = "preflight"
    CONFIRMED = "confirmed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    ROLLED_BACK = "rolled_back"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_TERMINAL = "failed_terminal"


def copy_identifier(value: str, label: str) -> str:
    identity = _safe_identity(value, label)
    if value != identity or not _IDENTIFIER.fullmatch(identity):
        raise ValueError(f"{label} must be a bounded owner identifier")
    return identity


def validate_copy_workspace(ref: QualifiedWorkspaceRef) -> None:
    if not isinstance(ref, QualifiedWorkspaceRef):
        raise TypeError("Copy requires qualified workspaces")
    copy_identifier(ref.workspace_id, "workspace ID")
    if ref.data_source is WorkspaceDataSource.SERVER:
        copy_identifier(ref.server_profile_id, "server profile ID")
        copy_identifier(ref.principal_id, "server principal ID")


def copy_digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CopyOwnerRef:
    """A canonical item and its owner-supplied revision or snapshot digest."""

    workspace: QualifiedWorkspaceRef
    kind: CopyItemKind
    owner_id: str
    version: int | None = None
    snapshot_hash: str = ""

    def __post_init__(self) -> None:
        validate_copy_workspace(self.workspace)
        try:
            object.__setattr__(self, "kind", CopyItemKind(self.kind))
        except (TypeError, ValueError):
            raise ValueError("Unsupported Copy item type") from None
        copy_identifier(self.owner_id, "item owner ID")
        if self.version is not None and (
            type(self.version) is not int or not 0 <= self.version < 2**63
        ):
            raise ValueError("The owner revision must be a non-negative integer")
        if not isinstance(self.snapshot_hash, str) or (
            self.snapshot_hash and not _HASH.fullmatch(self.snapshot_hash)
        ):
            raise ValueError("The owner snapshot hash must be a SHA-256 digest")

    @property
    def has_revision(self) -> bool:
        return self.version is not None or bool(self.snapshot_hash)


@dataclass(frozen=True, slots=True)
class ResearchCopyItem:
    source: CopyOwnerRef
    treatment: CopyTreatment
    destination: CopyOwnerRef | None = None
    conflict: CopyConflictChoice | None = None
    membership_id: str = ""
    estimated_bytes: int = 0
    reason: CopyReason = CopyReason.NONE

    def __post_init__(self) -> None:
        if not isinstance(self.source, CopyOwnerRef):
            raise TypeError("Copy requires an exact source owner")
        object.__setattr__(self, "treatment", CopyTreatment(self.treatment))
        object.__setattr__(self, "reason", CopyReason(self.reason))
        if (
            type(self.estimated_bytes) is not int
            or not 0 <= self.estimated_bytes < 2**63
        ):
            raise ValueError("Copy size must be a non-negative byte count")
        if self.membership_id != "":
            copy_identifier(self.membership_id, "membership ID")
        if self.conflict is not None:
            object.__setattr__(self, "conflict", CopyConflictChoice(self.conflict))
        if self.destination is not None and (
            not isinstance(self.destination, CopyOwnerRef)
            or self.destination.kind is not self.source.kind
        ):
            raise ValueError("The destination item must match the source kind")
        if not self.actionable:
            return
        if not self.source.has_revision:
            raise ValueError("Copy requires a stable source revision")
        if (
            self.reason is not CopyReason.NONE
            or self.source.kind is CopyItemKind.CONVERSATION
        ):
            raise ValueError("This item type or owner is unavailable for Copy")
        if self.destination is not None and self.conflict is None:
            raise ValueError("Choose how to handle the destination conflict")
        if self.conflict is CopyConflictChoice.REPLACE_DESTINATION and (
            self.destination is None
            or not self.destination.version
            or self.treatment is not CopyTreatment.CONTENT
        ):
            raise ValueError(
                "Replacement requires an exact destination version and content treatment"
            )
        if self.conflict is CopyConflictChoice.REFERENCE_EXISTING and (
            self.destination is None
            or not self.destination.has_revision
            or self.treatment is not CopyTreatment.REFERENCE
        ):
            raise ValueError(
                "Referencing an existing item requires its exact owner revision"
            )
        if self.conflict in {
            CopyConflictChoice.KEEP_DESTINATION,
            CopyConflictChoice.OMIT,
            CopyConflictChoice.CANCEL,
        }:
            raise ValueError("This conflict choice omits the item from Copy")

    @property
    def actionable(self) -> bool:
        return self.treatment.actionable

    @property
    def canonical_identity(self) -> tuple[CopyItemKind, str]:
        return self.source.kind, self.source.owner_id

    @property
    def redaction(self) -> CopyRedaction:
        return {
            CopyTreatment.CONTENT: CopyRedaction.OWNER_FIELDS,
            CopyTreatment.REFERENCE: CopyRedaction.REFERENCE_ONLY,
            CopyTreatment.METADATA_ONLY: CopyRedaction.METADATA_ONLY,
        }.get(self.treatment, CopyRedaction.EXCLUDED)


@dataclass(frozen=True, slots=True)
class ResearchCopyManifest:
    """One source, destination, and immutable preflight decision per item."""

    source: QualifiedWorkspaceRef
    destination: QualifiedWorkspaceRef
    items: tuple[ResearchCopyItem, ...]
    created_at: str
    manifest_hash: str = field(init=False)
    transfer_id: str = field(init=False)

    def __post_init__(self) -> None:
        validate_copy_workspace(self.source)
        validate_copy_workspace(self.destination)
        if self.source.data_source is self.destination.data_source:
            raise ValueError("Copy requires workspaces in different authorities")
        if (
            type(self.items) is not tuple
            or not 1 <= len(self.items) <= MAX_RESEARCH_SELECTION_IDS
        ):
            raise ValueError("Copy requires a bounded tuple of selected items")
        if any(not isinstance(item, ResearchCopyItem) for item in self.items):
            raise TypeError("Copy items must hold validated owner references")
        if any(item.source.workspace != self.source for item in self.items):
            raise ValueError("A Copy item belongs to another source workspace")
        if any(
            item.destination and item.destination.workspace != self.destination
            for item in self.items
        ):
            raise ValueError("A Copy conflict belongs to another destination workspace")
        by_identity = {item.canonical_identity: item for item in self.items}
        if len(by_identity) != len(self.items):
            raise ValueError("Copy contains a duplicate canonical item")
        try:
            created_at = datetime.fromisoformat(self.created_at)
        except (TypeError, ValueError):
            raise ValueError("Copy requires a timestamp with a timezone") from None
        if created_at.tzinfo is None:
            raise ValueError("Copy requires a timestamp with a timezone")
        ordered = tuple(by_identity[key] for key in sorted(by_identity))
        payload = {
            "source": asdict(self.source),
            "destination": asdict(self.destination),
            "items": [asdict(item) for item in ordered],
        }
        digest = copy_digest(payload)
        object.__setattr__(self, "items", ordered)
        object.__setattr__(self, "manifest_hash", digest)
        object.__setattr__(self, "transfer_id", f"copy-{digest}")

    @property
    def estimated_bytes(self) -> int:
        return sum(item.estimated_bytes for item in self.items if item.actionable)

    @property
    def requires_replacement_confirmation(self) -> bool:
        return any(
            item.actionable and item.conflict is CopyConflictChoice.REPLACE_DESTINATION
            for item in self.items
        )

    def item_idempotency_key(self, item: ResearchCopyItem) -> str:
        identity = item.canonical_identity
        index = bisect_left(
            self.items, identity, key=lambda row: row.canonical_identity
        )
        if index == len(self.items) or self.items[index] != item:
            raise ValueError("The item does not belong to this manifest")
        return copy_digest((self.manifest_hash, *identity))


@dataclass(frozen=True, slots=True)
class ResearchCopyConfirmation:
    manifest_hash: str
    confirmed: bool
    replacement_confirmed: bool = False

    def validate(self, manifest: ResearchCopyManifest) -> None:
        if self.manifest_hash != manifest.manifest_hash:
            raise ValueError("Confirmation points to a different manifest")
        if self.confirmed is not True:
            raise ValueError("Confirm this exact Copy before continuing")
        if any(item.conflict is CopyConflictChoice.CANCEL for item in manifest.items):
            raise ValueError("The selected conflict choice cancels this Copy")
        if any(item.reason is CopyReason.CHOOSE_CONFLICT for item in manifest.items):
            raise ValueError("Resolve each destination conflict before continuing")
        if (
            manifest.requires_replacement_confirmation
            and self.replacement_confirmed is not True
        ):
            raise ValueError(
                "Explicitly confirm replacement of the captured destination version"
            )
        if not any(item.actionable for item in manifest.items):
            raise ValueError(
                "Select at least one item with an available Copy treatment"
            )


@dataclass(frozen=True, slots=True)
class ResearchCopyReceipt:
    """The durable transfer identity and its current operation revision."""

    manifest: ResearchCopyManifest
    status: ResearchCopyStatus
    revision: int
