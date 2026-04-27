"""
Conversation request and response schemas for the shared TLDW API client.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, field_validator, model_validator


ALLOWED_CONVERSATION_STATES = ("in-progress", "resolved", "backlog", "non-viable")
ConversationState = Literal["in-progress", "resolved", "backlog", "non-viable"]


def normalize_conversation_state(value: str | None) -> ConversationState | None:
    """Normalize a conversation state string and validate it against the shared contract."""

    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        raise ValueError("state cannot be empty")
    if normalized not in ALLOWED_CONVERSATION_STATES:
        raise ValueError(f"Invalid state '{value}'. Allowed: {', '.join(ALLOWED_CONVERSATION_STATES)}")
    return cast(ConversationState, normalized)


class ConversationScopeParams(BaseModel):
    """Scope parameters for filtering conversations by global or workspace scope."""

    scope_type: Literal["global", "workspace"] = "global"
    workspace_id: str | None = None

    @model_validator(mode="after")
    def _validate_workspace_scope(self) -> "ConversationScopeParams":
        if self.scope_type == "workspace" and not self.workspace_id:
            raise ValueError("workspace_id is required when scope_type='workspace'")
        if self.scope_type == "global":
            self.workspace_id = None
        return self


class ConversationListItem(BaseModel):
    """Conversation list row returned by the chat conversations endpoints."""

    id: str = Field(..., description="Conversation ID")
    scope_type: Literal["global", "workspace"] = Field("global", description="Conversation scope type")
    workspace_id: str | None = Field(None, description="Workspace ID when scope_type='workspace'")
    character_id: int | None = Field(None, description="Character ID associated with the conversation")
    assistant_kind: Literal["character", "persona"] | None = Field(
        None,
        description="Normalized assistant identity kind for the conversation",
    )
    assistant_id: str | None = Field(None, description="Normalized assistant identity ID for the conversation")
    runtime_backend: Literal["local", "server"] = Field(
        "local",
        description="Execution backend for the assistant/runtime (local or server)",
    )
    discovery_owner: Literal["general_chat", "ccp_character", "ccp_persona"] = Field(
        "general_chat",
        description="Owning surface for discovery/canonical identity attribution",
    )
    discovery_entity_id: str | None = Field(
        None,
        description="Canonical entity ID used by the discovery surface (string-first stable ID)",
    )
    persona_memory_mode: Literal["read_only", "read_write"] | None = Field(
        None,
        description="Persona durable memory behavior for the conversation",
    )
    title: str | None = Field(None, description="Conversation title")
    state: ConversationState = Field("in-progress", description="Lifecycle state of the conversation")
    topic_label: str | None = Field(None, description="Primary topic label")
    topic_label_source: str | None = Field(None, description="Source of the assigned topic label")
    topic_last_tagged_at: datetime | None = Field(None, description="Timestamp when topic label was last tagged")
    topic_last_tagged_message_id: str | None = Field(
        None,
        description="Message ID associated with the last topic tagging",
    )
    bm25_norm: float | None = Field(None, description="Normalized BM25 score (0-1)")
    last_modified: datetime = Field(..., description="Last modification timestamp")
    created_at: datetime = Field(..., description="Creation timestamp")
    message_count: int = Field(0, description="Total messages in the conversation")
    keywords: list[str] = Field(default_factory=list, description="Keyword tags for the conversation")
    cluster_id: str | None = Field(None, description="Cluster/group identifier")
    source: str | None = Field(None, description="Source of the conversation")
    external_ref: str | None = Field(None, description="External reference ID")
    version: int = Field(1, description="Version number for optimistic locking")


class ConversationListPagination(BaseModel):
    """Pagination metadata for conversation lists."""

    limit: int = Field(..., description="Items per page")
    offset: int = Field(..., description="Offset for pagination")
    total: int = Field(..., description="Total items matching filters")
    has_more: bool = Field(..., description="True when more items remain")


class ConversationListResponse(BaseModel):
    """Paginated conversation list response."""

    items: list[ConversationListItem] = Field(..., description="Conversation results")
    pagination: ConversationListPagination


class ConversationUpdateRequest(BaseModel):
    """Request body for updating a conversation."""

    version: int = Field(..., description="Expected version for optimistic locking")
    state: ConversationState | None = Field(None, description="Lifecycle state for the conversation")
    runtime_backend: Literal["local", "server"] | None = Field(
        None,
        description="Execution backend for the assistant/runtime (local or server)",
    )
    discovery_owner: Literal["general_chat", "ccp_character", "ccp_persona"] | None = Field(
        None,
        description="Owning surface for discovery/canonical identity attribution",
    )
    discovery_entity_id: str | None = Field(
        None,
        description="Canonical entity ID used by the discovery surface (string-first stable ID)",
    )
    topic_label: str | None = Field(None, description="Primary topic label for the conversation")
    keywords: list[str] | None = Field(None, description="Replace full keyword set (use [] to clear)")
    cluster_id: str | None = Field(None, description="Cluster/group identifier")
    source: str | None = Field(None, description="Source of the conversation")
    external_ref: str | None = Field(None, description="External reference/link")

    @field_validator("state", mode="before")
    @classmethod
    def _validate_state(cls, value: str | None) -> ConversationState | None:
        return normalize_conversation_state(value)

    @field_validator("runtime_backend", mode="before")
    @classmethod
    def _normalize_runtime_backend(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        if not normalized:
            raise ValueError("runtime_backend cannot be empty")
        if normalized not in {"local", "server"}:
            raise ValueError("runtime_backend must be 'local' or 'server'")
        return normalized

    @field_validator("discovery_owner", mode="before")
    @classmethod
    def _normalize_discovery_owner(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        if not normalized:
            raise ValueError("discovery_owner cannot be empty")
        if normalized not in {"general_chat", "ccp_character", "ccp_persona"}:
            raise ValueError("discovery_owner must be 'general_chat', 'ccp_character', or 'ccp_persona'")
        return normalized

    @field_validator("discovery_entity_id", mode="before")
    @classmethod
    def _normalize_discovery_entity_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator("keywords", mode="before")
    @classmethod
    def _normalize_keywords(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None

        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            if item is None:
                continue
            normalized = str(item).strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(normalized)
        return cleaned


class ConversationMetadata(BaseModel):
    """Conversation metadata returned by the detail and tree endpoints."""

    id: str = Field(..., description="Conversation ID")
    scope_type: Literal["global", "workspace"] = Field("global", description="Conversation scope type")
    workspace_id: str | None = Field(None, description="Workspace ID when scope_type='workspace'")
    character_id: int | None = Field(None, description="Character ID associated with the conversation")
    assistant_kind: Literal["character", "persona"] | None = Field(
        None,
        description="Normalized assistant identity kind for the conversation",
    )
    assistant_id: str | None = Field(None, description="Normalized assistant identity ID for the conversation")
    runtime_backend: Literal["local", "server"] = Field(
        "local",
        description="Execution backend for the assistant/runtime (local or server)",
    )
    discovery_owner: Literal["general_chat", "ccp_character", "ccp_persona"] = Field(
        "general_chat",
        description="Owning surface for discovery/canonical identity attribution",
    )
    discovery_entity_id: str | None = Field(
        None,
        description="Canonical entity ID used by the discovery surface (string-first stable ID)",
    )
    persona_memory_mode: Literal["read_only", "read_write"] | None = Field(
        None,
        description="Persona durable memory behavior for the conversation",
    )
    title: str | None = Field(None, description="Conversation title")
    state: ConversationState = Field("in-progress", description="Lifecycle state")
    topic_label: str | None = Field(None, description="Primary topic label")
    created_at: datetime | None = Field(None, description="Conversation creation timestamp")
    topic_label_source: str | None = Field(None, description="Source of the assigned topic label")
    topic_last_tagged_at: datetime | None = Field(None, description="Timestamp when topic label was last tagged")
    topic_last_tagged_message_id: str | None = Field(
        None,
        description="Message ID associated with the last topic tagging",
    )
    cluster_id: str | None = Field(None, description="Cluster/group identifier")
    source: str | None = Field(None, description="Source of the conversation")
    external_ref: str | None = Field(None, description="External reference ID")
    version: int | None = Field(None, description="Version number")
    last_modified: datetime = Field(..., description="Last modification timestamp")


class ConversationTreeNode(BaseModel):
    """Recursive tree node for threaded conversation messages."""

    id: str = Field(..., description="Message ID")
    role: str = Field(..., description="Message role (user/assistant/system)")
    content: str = Field("", description="Message content")
    created_at: datetime = Field(..., description="Message timestamp")
    children: list[ConversationTreeNode] = Field(default_factory=list)
    truncated: bool = Field(False, description="True when descendants were omitted")


class ConversationTreePagination(BaseModel):
    """Pagination metadata for the root-level thread list."""

    limit: int = Field(..., description="Root threads per page")
    offset: int = Field(..., description="Root threads offset")
    total_root_threads: int = Field(..., description="Total root threads")
    has_more: bool = Field(..., description="True when more root threads remain")


class ConversationTreeResponse(BaseModel):
    """Paginated conversation tree response."""

    conversation: ConversationMetadata
    root_threads: list[ConversationTreeNode]
    pagination: ConversationTreePagination
    depth_cap: int = Field(..., description="Applied depth cap")


class ChatCommandInfo(BaseModel):
    """Slash command metadata exposed by the server chat command registry."""

    name: str
    description: str = ""
    required_permission: str | None = None
    usage: str | None = None
    args: list[str] = Field(default_factory=list)
    requires_api_key: bool | None = None
    rate_limit: str | None = None
    rbac_required: bool | None = None


class ChatCommandsListResponse(BaseModel):
    """Available server chat slash commands."""

    commands: list[ChatCommandInfo] = Field(default_factory=list)


class ChatKnowledgeSaveRequest(BaseModel):
    """Request to save a server conversation snippet into server Notes/Flashcards."""

    conversation_id: str
    message_id: str | None = None
    scope_type: Literal["global", "workspace"] = "global"
    workspace_id: str | None = None
    snippet: str = Field(..., min_length=1)
    tags: list[str] | None = None
    make_flashcard: bool = False
    export_to: Literal["none", "notion", "wiki"] = "none"

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            if item is None:
                continue
            normalized = str(item).strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(normalized)
        return cleaned or None

    @model_validator(mode="after")
    def _validate_scope(self) -> "ChatKnowledgeSaveRequest":
        if self.scope_type == "workspace" and not self.workspace_id:
            raise ValueError("workspace_id is required when scope_type='workspace'")
        if self.scope_type == "global":
            self.workspace_id = None
        return self


class ChatKnowledgeSaveResponse(BaseModel):
    """Server response for saved chat knowledge snippets."""

    note_id: str | int
    flashcard_id: str | int | None = None
    conversation_id: str
    message_id: str | None = None
    export_status: Literal["not_requested", "skipped_disabled", "queued", "completed"] = "not_requested"
    export_job_id: str | None = None


class ConversationShareLinkCreateRequest(BaseModel):
    """Request to create a tokenized server conversation share link."""

    permission: Literal["view"] = "view"
    ttl_seconds: int | None = Field(None, ge=300)
    label: str | None = Field(None, max_length=80)


class ConversationShareLinkResponse(BaseModel):
    """Created tokenized server conversation share link."""

    share_id: str
    permission: Literal["view"]
    created_at: datetime
    expires_at: datetime
    token: str
    share_path: str


class ConversationShareLinkListItem(BaseModel):
    """Existing server conversation share link metadata."""

    id: str
    permission: Literal["view"]
    created_at: datetime
    expires_at: datetime
    revoked_at: datetime | None = None
    label: str | None = None
    share_path: str | None = None
    token: str | None = None


class ConversationShareLinksResponse(BaseModel):
    """Share links for a server conversation."""

    conversation_id: str
    links: list[ConversationShareLinkListItem] = Field(default_factory=list)


class ConversationShareLinkRevokeResponse(BaseModel):
    """Result of revoking a server conversation share link."""

    success: bool
    share_id: str


class SharedConversationResolveResponse(BaseModel):
    """Public shared-conversation payload resolved from a share token."""

    conversation_id: str
    title: str | None = None
    source: str | None = None
    permission: Literal["view"]
    shared_by_user_id: str
    expires_at: datetime
    messages: list[dict[str, Any]] = Field(default_factory=list)


class ChatAnalyticsBucket(BaseModel):
    """Conversation analytics bucket."""

    bucket_start: datetime
    topic_label: str | None = None
    state: str
    count: int


class ChatAnalyticsPagination(BaseModel):
    """Conversation analytics pagination."""

    limit: int
    offset: int
    total: int
    has_more: bool


class ChatAnalyticsResponse(BaseModel):
    """Server conversation analytics response."""

    buckets: list[ChatAnalyticsBucket] = Field(default_factory=list)
    pagination: ChatAnalyticsPagination
    bucket_granularity: Literal["day", "week"]


class ValidationIssue(BaseModel):
    """A single dictionary validation issue."""

    code: str
    field: str
    message: str


class ValidateDictionaryRequest(BaseModel):
    """Request body for server-side chat dictionary validation."""

    data: dict[str, Any] = Field(..., description="Dictionary JSON data including entries")
    schema_version: int = Field(1, description="Schema version for validation")
    strict: bool = Field(False, description="If true, fail import policy while still returning a report")


class ValidateDictionaryResponse(BaseModel):
    """Structured dictionary validation result."""

    ok: bool
    schema_version: int
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)
    entry_stats: dict[str, int] = Field(default_factory=dict)
    suggested_fixes: list[str] = Field(default_factory=list)
    partial: bool = False
    partial_reason: str | None = None
