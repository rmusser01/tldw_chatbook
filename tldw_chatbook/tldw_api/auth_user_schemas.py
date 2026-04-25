from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AuthTokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class MFAChallengeResponse(BaseModel):
    session_token: str
    mfa_required: bool = True
    expires_in: int
    message: str | None = None


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    all_devices: bool = False


class MessageResponse(BaseModel):
    message: str
    details: dict[str, Any] | None = None


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    registration_code: str | None = None


class RegistrationResponse(BaseModel):
    message: str = "Registration successful"
    user_id: int
    username: str
    email: str
    requires_verification: bool
    api_key: str | None = None


class SessionResponse(BaseModel):
    id: int
    ip_address: str | None = None
    user_agent: str | None = None
    created_at: str
    last_activity: str
    expires_at: str


class UserProfileCatalogEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    key: str
    label: str
    description: str | None = None
    type: str
    sensitivity: str
    editable_by: list[str] = Field(default_factory=list)
    deprecated: bool = False


class UserProfileCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    version: str
    updated_at: str
    entries: list[UserProfileCatalogEntry] = Field(default_factory=list)


class UserProfileResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    profile_version: str
    catalog_version: str
    user: dict[str, Any] | None = None
    memberships: dict[str, Any] | None = None
    security: dict[str, Any] | None = None
    quotas: dict[str, Any] | None = None
    preferences: dict[str, Any] | None = None
    effective_config: dict[str, Any] | None = None
    raw_overrides: dict[str, Any] | None = None
    section_errors: dict[str, str] | None = None


class UserProfileUpdateEntry(BaseModel):
    key: str
    value: Any | None = None


class UserProfileUpdateRequest(BaseModel):
    updates: list[UserProfileUpdateEntry] = Field(default_factory=list)
    profile_version: str | None = None
    dry_run: bool = False


class UserProfileUpdateError(BaseModel):
    key: str
    message: str


class UserProfileUpdateResponse(BaseModel):
    profile_version: str
    applied: list[str] = Field(default_factory=list)
    skipped: list[UserProfileUpdateError] = Field(default_factory=list)


__all__ = [
    "LogoutRequest",
    "MessageResponse",
    "MFAChallengeResponse",
    "RefreshTokenRequest",
    "RegisterRequest",
    "RegistrationResponse",
    "SessionResponse",
    "AuthTokenResponse",
    "UserProfileCatalogEntry",
    "UserProfileCatalogResponse",
    "UserProfileResponse",
    "UserProfileUpdateEntry",
    "UserProfileUpdateError",
    "UserProfileUpdateRequest",
    "UserProfileUpdateResponse",
]
