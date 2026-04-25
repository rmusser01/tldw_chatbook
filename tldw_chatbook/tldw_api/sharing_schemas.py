"""Workspace sharing and share token schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


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
    new_name: str | None = Field(default=None, max_length=255)


class CloneWorkspaceResponse(BaseModel):
    job_id: str
    status: str = "pending"
    message: str = "Clone job created"


class SharedWorkspaceSourceResponse(BaseModel):
    id: str
    workspace_id: str
    media_id: int | None = None
    title: str = ""
    source_type: str = "media"
    url: str | None = None
    position: int = 0
    added_at: str | None = None


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
