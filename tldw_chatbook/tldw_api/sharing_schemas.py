"""Workspace sharing and share token schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)


class ShareScopeType(str, Enum):
    TEAM = "team"
    ORG = "org"


class AccessLevel(str, Enum):
    VIEW_CHAT = "view_chat"
    VIEW_CHAT_ADD = "view_chat_add"
    FULL_EDIT = "full_edit"


class ResourceType(str, Enum):
    CHATBOOK = "chatbook"
    WORKSPACE = "workspace"


class ShareWorkspaceRequest(BaseModel):
    share_scope_type: ShareScopeType
    share_scope_id: int
    access_level: AccessLevel = AccessLevel.VIEW_CHAT
    allow_clone: bool = True


class UpdateShareRequest(BaseModel):
    access_level: AccessLevel | None = None
    allow_clone: bool | None = None


class ShareResponse(BaseModel):
    id: int
    workspace_id: str
    owner_user_id: int | None = None
    share_scope_type: str | None = None
    share_scope_id: int | None = None
    access_level: str | None = None
    allow_clone: bool | None = None
    created_by: int | None = None
    created_at: str | None = None
    updated_at: str | None = None
    revoked_at: str | None = None
    is_revoked: bool = False


class ShareListResponse(BaseModel):
    shares: list[ShareResponse]
    total: int


class SharedWithMeItem(BaseModel):
    share_id: int
    workspace_id: str
    workspace_name: str | None = None
    owner_user_id: int
    owner_username: str | None = None
    access_level: str
    allow_clone: bool
    shared_at: str | None = None


class SharedWithMeResponse(BaseModel):
    items: list[SharedWithMeItem]
    total: int


class CreateTokenRequest(BaseModel):
    resource_type: ResourceType
    resource_id: str
    access_level: AccessLevel = AccessLevel.VIEW_CHAT
    allow_clone: bool = True
    password: str | None = Field(default=None, min_length=4, max_length=128)
    max_uses: int | None = Field(default=None, ge=1, le=10000)
    expires_at: str | None = None


class TokenResponse(BaseModel):
    id: int
    token_prefix: str | None = None
    resource_type: str | None = None
    resource_id: str | None = None
    access_level: str | None = None
    allow_clone: bool | None = None
    is_password_protected: bool = False
    max_uses: int | None = None
    use_count: int = 0
    expires_at: str | None = None
    created_at: str | None = None
    revoked_at: str | None = None
    is_revoked: bool = False
    raw_token: str | None = None


class TokenListResponse(BaseModel):
    tokens: list[TokenResponse]
    total: int


class PublicSharePreview(BaseModel):
    resource_type: str
    resource_name: str | None = None
    resource_description: str | None = None
    is_password_protected: bool = False
    access_level: str


class VerifyPasswordRequest(BaseModel):
    password: str = Field(..., min_length=1)


class VerifyPasswordResponse(BaseModel):
    verified: bool
    session_token: str | None = None


class CloneWorkspaceRequest(BaseModel):
    """One logical clone admission; retain this request/key after uncertain responses."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        validation_alias=AliasChoices("name", "new_name"),
    )
    idempotency_key: str = Field(
        default_factory=lambda: str(uuid4()),
        exclude=True,
        min_length=16,
        max_length=200,
        pattern=r"^[A-Za-z0-9._~-]+$",
    )

    @model_validator(mode="after")
    def validate_name(self) -> CloneWorkspaceRequest:
        """Reject names that the canonical server request refuses."""
        if self.name is not None and not self.name.strip():
            raise ValueError("name must not be blank")
        return self


class CloneWorkspaceResponse(BaseModel):
    """Durable operation receipt, with legacy ephemeral job responses accepted."""

    job_id: str | None = None
    operation_id: str | None = None
    workspace_id: str | None = None
    command: str | None = None
    schema_version: int | None = None
    share_id: int | None = None
    status: str = "pending"
    started_at: str | None = None
    updated_at: str | None = None
    retryable: bool = False
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    poll_href: str | None = None
    progress: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    message: str | None = None

    @model_validator(mode="after")
    def validate_identity(self) -> CloneWorkspaceResponse:
        """Require a usable receipt identity in both protocol generations."""
        if not self.operation_id and not self.job_id:
            raise ValueError("clone response requires an operation_id or legacy job_id")
        return self


class SharedWorkspaceSourceResponse(BaseModel):
    """Current recipient source projection with legacy field aliases."""

    source_id: str = Field(validation_alias=AliasChoices("source_id", "id"))
    workspace_id: str | None = None
    media_id: int | None = None
    title: str = ""
    source_type: str = "media"
    origin_url: str | None = Field(
        default=None, validation_alias=AliasChoices("origin_url", "url")
    )
    origin_host: str | None = None
    state: str | None = None
    reason_code: str | None = None
    citation_ready: bool | None = None
    retrieval_ready: bool | None = None
    position: int = 0
    added_at: str | None = None

    @computed_field
    @property
    def id(self) -> str:
        """Compatibility accessor for pre-page source consumers."""
        return self.source_id

    @computed_field
    @property
    def url(self) -> str | None:
        """Compatibility accessor for pre-page source consumers."""
        return self.origin_url


class SharedWorkspaceSourceQuery(BaseModel):
    """Validate page filters using the recipient source endpoint's bounds."""

    model_config = ConfigDict(extra="forbid", strict=True)

    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=200)
    q: str | None = Field(default=None, min_length=1, max_length=512)
    state: str | None = Field(default=None, min_length=1, max_length=64)


class SharedWorkspaceSourcePagination(BaseModel):
    """Server-owned offset pagination metadata."""

    offset: int = Field(ge=0)
    limit: int = Field(ge=1)
    total: int = Field(ge=0)
    has_more: bool


class SharedWorkspaceSourcePage(BaseModel):
    """Recipient source rows with completeness and partial-failure metadata."""

    items: list[SharedWorkspaceSourceResponse]
    pagination: SharedWorkspaceSourcePagination
    summary: dict[str, Any] = Field(default_factory=dict)
    partial_errors: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_list(cls, value: Any) -> Any:
        """Wrap an older unpaginated list without inventing another page."""
        if isinstance(value, list):
            return {
                "items": value,
                "pagination": {
                    "offset": 0,
                    "limit": max(1, len(value)),
                    "total": len(value),
                    "has_more": False,
                },
            }
        return value


class SharedMediaResponse(BaseModel):
    id: int
    title: str = ""
    url: str | None = None
    media_type: str | None = None
    content: str | None = None
    author: str | None = None
    ingestion_date: str | None = None


class SharedChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    model: str | None = None
    api_name: str | None = None
    system_message: str | None = None


class SharedWorkspaceResponse(BaseModel):
    share: ShareResponse


class PublicShareImportResponse(BaseModel):
    resource_type: str
    resource_id: str
    access_level: str
    owner_user_id: int
    message: str | None = None


class AuditEventResponse(BaseModel):
    id: int
    event_type: str
    actor_user_id: int | None = None
    resource_type: str
    resource_id: str
    owner_user_id: int
    share_id: int | None = None
    token_id: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    ip_address: str | None = None
    user_agent: str | None = None
    created_at: str | None = None


class AuditLogResponse(BaseModel):
    events: list[AuditEventResponse]
    total: int
