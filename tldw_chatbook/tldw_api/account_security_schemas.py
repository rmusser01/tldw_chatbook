from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)

    @field_validator("new_password")
    @classmethod
    def passwords_different(cls, value: str, info: Any) -> str:
        if info.data.get("current_password") == value:
            raise ValueError("New password must be different from current password")
        return value


class PasswordResetRequest(BaseModel):
    email: str


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8)


class MFASetupResponse(BaseModel):
    secret: str
    qr_code: str
    backup_codes: list[str] = Field(default_factory=list)


class StorageQuotaResponse(BaseModel):
    user_id: int
    storage_used_mb: float
    storage_quota_mb: int
    available_mb: float
    usage_percentage: float


class APIKeyCreateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    scope: str | list[str] = "read"
    expires_in_days: int | None = 365


class APIKeyRotateRequest(BaseModel):
    expires_in_days: int | None = 365


class APIKeyMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    key_prefix: str | None = None
    name: str | None = None
    description: str | None = None
    scope: str | list[str]
    status: str | None = None
    created_at: datetime | None = None
    expires_at: datetime | None = None
    usage_count: int | None = 0
    last_used_at: datetime | None = None
    last_used_ip: str | None = None


class APIKeyCreateResponse(APIKeyMetadata):
    key: str | None = None
    message: str | None = None


class VirtualAPIKeyCreateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    expires_in_days: int | None = 30
    allowed_endpoints: list[str] | None = None
    allowed_methods: list[str] | None = None
    allowed_paths: list[str] | None = None
    max_calls: int | None = None
    max_runs: int | None = None
    budget_day_tokens: int | None = None
    budget_month_tokens: int | None = None
    budget_day_usd: float | None = None
    budget_month_usd: float | None = None


APIKeyScope = Literal["read", "write", "admin", "service"]


__all__ = [
    "APIKeyCreateRequest",
    "APIKeyCreateResponse",
    "APIKeyMetadata",
    "APIKeyRotateRequest",
    "APIKeyScope",
    "MFASetupResponse",
    "PasswordChangeRequest",
    "PasswordResetConfirm",
    "PasswordResetRequest",
    "StorageQuotaResponse",
    "VirtualAPIKeyCreateRequest",
]
