from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ClaimNotificationResponse(BaseModel):
    id: int
    user_id: str
    kind: str
    target_user_id: str | None = None
    target_review_group: str | None = None
    resource_type: str | None = None
    resource_id: str | None = None
    payload: dict[str, object] = Field(default_factory=dict)
    created_at: str | None = None
    delivered_at: str | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimNotificationsAckRequest(BaseModel):
    ids: list[int] = Field(..., min_length=1)


class ClaimNotificationsDigestResponse(BaseModel):
    total: int
    counts_by_kind: dict[str, int] = Field(default_factory=dict)
    counts_by_target_user: dict[str, int] = Field(default_factory=dict)
    counts_by_review_group: dict[str, int] = Field(default_factory=dict)
    notifications: list[ClaimNotificationResponse] | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsAlertConfigResponse(BaseModel):
    id: int
    user_id: str
    name: str
    alert_type: str
    threshold_ratio: float | None = None
    baseline_ratio: float | None = None
    channels: dict[str, bool] = Field(default_factory=dict)
    slack_webhook_url: str | None = None
    webhook_url: str | None = None
    email_recipients: list[str] = Field(default_factory=list)
    enabled: bool
    created_at: str | None = None
    updated_at: str | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsAlertConfigCreate(BaseModel):
    name: str = Field(..., min_length=1)
    alert_type: str = Field(..., min_length=1)
    threshold_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    baseline_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    channels: dict[str, bool] = Field(default_factory=dict)
    slack_webhook_url: str | None = None
    webhook_url: str | None = None
    email_recipients: list[str] | None = None
    enabled: bool | None = None

    model_config = ConfigDict(extra="forbid")


class ClaimsAlertConfigUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1)
    alert_type: str | None = Field(default=None, min_length=1)
    threshold_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    baseline_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    channels: dict[str, bool] | None = None
    slack_webhook_url: str | None = None
    webhook_url: str | None = None
    email_recipients: list[str] | None = None
    enabled: bool | None = None

    model_config = ConfigDict(extra="forbid")
