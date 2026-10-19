"""Immutable local archive recovery search contracts (ADR-166)."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)


class ArchiveRecoveryError(Exception):
    """Sanitized failure that adapters translate to a structured outcome."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class ArchiveSearchRequest(BaseModel):
    """Validated literal search and inclusive/exclusive archive interval."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    query: str = Field(default="", max_length=512, strict=True)
    workspace_id: str | None = Field(default=None, strict=True)
    archived_after: datetime | None = None
    archived_before: datetime | None = None
    include_unknown_archive_times: StrictBool = False
    limit: StrictInt = Field(default=10, ge=1, le=20)

    @field_validator("workspace_id")
    @classmethod
    def workspace_identity(cls, value: str | None) -> str | None:
        """Reject empty identities rather than silently widening scope."""
        if value is not None and not value.strip():
            raise ValueError("workspace_id must not be empty")
        return value.strip() if value is not None else None

    @field_validator("archived_after", "archived_before", mode="before")
    @classmethod
    def qualified_datetime(cls, value: object) -> datetime | None:
        """Require timezone-qualified dates and normalize them to UTC."""
        if value is None:
            return None
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        if (
            not isinstance(value, datetime)
            or value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError("archive dates require a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def ordered_interval(self) -> "ArchiveSearchRequest":
        """Reject reversed and empty date windows."""
        if (
            self.archived_after
            and self.archived_before
            and self.archived_after >= self.archived_before
        ):
            raise ValueError("archived_after must precede archived_before")
        return self


@dataclass(frozen=True)
class ArchiveWorkspaceSnapshot:
    """Observed workspace identity and lifecycle state."""

    workspace_id: str
    name: str
    archived: bool
    archived_at: str | None
    archive_revision: int


@dataclass(frozen=True)
class ArchiveTarget:
    """Exact saved conversation and observed ownership."""

    conversation_id: str
    conversation_version: int
    # Presentation only: search fitting and metadata fallback may differ. The
    # durable version guards title edits; every other identity field still compares.
    title: str = field(compare=False)
    conversation_archived: bool
    workspace_id: str | None
    workspace: ArchiveWorkspaceSnapshot | None


@dataclass(frozen=True)
class ArchiveCandidate:
    """Bounded discovery text with chronology and permitted restore scopes."""

    target: ArchiveTarget
    conversation_archived_at: str | None
    excerpt: str
    match_basis: tuple[str, ...]
    sort_key: tuple[int, str, str]
    chronology_incomplete: bool
    match_message_id: str | None
    allowed_scopes: tuple[str, ...]


@dataclass(frozen=True)
class ArchiveSearchPage:
    """A live keyset page bound to observed workspace lifecycle inputs."""

    items: tuple[ArchiveCandidate, ...]
    next_anchor: tuple[int, str, str] | None
    workspace_digest: str
    unknown_times_present: bool


@dataclass(frozen=True)
class RecoveryCallKey:
    """Native recovery call identity; settlement belongs to the coordinator."""

    session_id: str
    run_id: str
    call_id: str
    argument_digest: str


# JSON projection types are separate from the immutable internal snapshots.
class ArchiveResultWorkspace(TypedDict):
    workspace_id: str
    name: str
    archived: bool
    archived_at: str | None
    archive_revision: int


class ArchiveResultRow(TypedDict):
    result_ref: str
    conversation_id: str
    conversation_version: int
    title: str
    conversation_archived: bool
    conversation_archived_at: str | None
    workspace_id: str | None
    workspace: ArchiveResultWorkspace | None
    excerpt: str
    archive_match_basis: list[str]
    text_match_basis: list[str]
    inactive_branch: bool
    match_message_id: str | None
    chronology_incomplete: bool
    allowed_scopes: list[str]


class ArchiveResultProjection(TypedDict):
    result_set_id: str
    items: list[ArchiveResultRow]
    cursor: str | None
    unknown_times_present: bool
    truncated: bool


RestoreScope = Literal["chat_only", "workspace_only", "chat_and_workspace"]
RecoveryPhase = Literal[
    "awaiting_confirmation",
    "admitted",
    "workspace_write",
    "chat_write",
    "observed",
    "settled",
]


@dataclass(frozen=True)
class RecoveryConfirmation:
    """App-owned one-shot authorization snapshot, never agent approval."""

    request_id: str
    operation_id: str
    session_id: str
    target: ArchiveTarget
    allowed_scopes: tuple[RestoreScope, ...]
    selected_scope: RestoreScope | None
    replacement_name: str | None
    expires_at: float


@dataclass(frozen=True)
class RecoveryOutcome:
    """Known completed mutations, separately from final chat availability."""

    status: str
    conversation_id: str
    workspace_id: str | None
    chat_changed: bool
    workspace_changed: bool
    chat_available: bool | None
    next_action: str | None
    message: str


@dataclass(frozen=True)
class RecoveryOperationSnapshot:
    """Trusted app presentation state for remount and completion delivery."""

    operation_id: str
    session_id: str
    phase: RecoveryPhase
    confirmation: RecoveryConfirmation | None
    outcome: RecoveryOutcome | None
    admitted: bool = False
