from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class UserProviderKeyUpsertRequest(BaseModel):
    provider: str
    api_key: str
    credential_fields: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class ProviderKeyTestRequest(BaseModel):
    provider: str
    model: str | None = None


class UserProviderKeyResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    provider: str
    status: Literal["stored"] = "stored"
    key_hint: str
    updated_at: datetime


class UserProviderKeyStatusItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    provider: str
    has_key: bool
    source: Literal["user", "team", "org", "server_default", "none", "disabled"]
    key_hint: str | None = None
    auth_source: Literal["api_key", "oauth"] | None = None
    last_used_at: datetime | None = None


class UserProviderKeysResponse(BaseModel):
    items: list[UserProviderKeyStatusItem] = Field(default_factory=list)


class ProviderKeyTestResponse(BaseModel):
    provider: str
    status: Literal["valid"] = "valid"
    model: str | None = None


class OpenAIOAuthAuthorizeRequest(BaseModel):
    credential_fields: dict[str, Any] | None = None
    return_path: str | None = None


class OpenAIOAuthAuthorizeResponse(BaseModel):
    provider: Literal["openai"] = "openai"
    auth_url: str
    auth_session_id: str
    expires_at: datetime


class OpenAIOAuthCallbackResponse(BaseModel):
    provider: Literal["openai"] = "openai"
    status: Literal["stored"] = "stored"
    auth_source: Literal["oauth"] = "oauth"
    key_hint: str = "oauth"
    updated_at: datetime
    expires_at: datetime | None = None


class OpenAIOAuthStatusResponse(BaseModel):
    provider: Literal["openai"] = "openai"
    connected: bool
    auth_source: Literal["api_key", "oauth", "none"] = "none"
    updated_at: datetime | None = None
    last_used_at: datetime | None = None
    expires_at: datetime | None = None
    scope: str | None = None


class OpenAIOAuthRefreshResponse(BaseModel):
    provider: Literal["openai"] = "openai"
    status: Literal["refreshed"] = "refreshed"
    updated_at: datetime
    expires_at: datetime | None = None


class OpenAICredentialSourceSwitchRequest(BaseModel):
    auth_source: Literal["api_key", "oauth"]


class OpenAICredentialSourceSwitchResponse(BaseModel):
    provider: Literal["openai"] = "openai"
    auth_source: Literal["api_key", "oauth"]
    updated_at: datetime


__all__ = [
    "OpenAICredentialSourceSwitchRequest",
    "OpenAICredentialSourceSwitchResponse",
    "OpenAIOAuthAuthorizeRequest",
    "OpenAIOAuthAuthorizeResponse",
    "OpenAIOAuthCallbackResponse",
    "OpenAIOAuthRefreshResponse",
    "OpenAIOAuthStatusResponse",
    "ProviderKeyTestRequest",
    "ProviderKeyTestResponse",
    "UserProviderKeyResponse",
    "UserProviderKeyStatusItem",
    "UserProviderKeyUpsertRequest",
    "UserProviderKeysResponse",
]
