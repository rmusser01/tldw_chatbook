"""Server notifications inbox and reminder task API schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ReminderScheduleKind = Literal["one_time", "recurring"]


class ReminderTaskCreateRequest(BaseModel):
    """Payload for creating a server-side reminder task."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, max_length=200)
    body: str | None = None
    schedule_kind: ReminderScheduleKind
    run_at: str | None = None
    cron: str | None = None
    timezone: str | None = None
    link_type: str | None = None
    link_id: str | None = None
    link_url: str | None = None
    enabled: bool = True

    @model_validator(mode="after")
    def _validate_schedule_fields(self) -> "ReminderTaskCreateRequest":
        if self.schedule_kind == "one_time":
            if not self.run_at:
                raise ValueError("run_at is required for one_time schedules")
            return self
        if not self.cron:
            raise ValueError("cron is required for recurring schedules")
        if not self.timezone:
            raise ValueError("timezone is required for recurring schedules")
        return self


class ReminderTaskUpdateRequest(BaseModel):
    """Patch payload for mutable server reminder task fields."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=200)
    body: str | None = None
    schedule_kind: ReminderScheduleKind | None = None
    run_at: str | None = None
    cron: str | None = None
    timezone: str | None = None
    link_type: str | None = None
    link_id: str | None = None
    link_url: str | None = None
    enabled: bool | None = None


class ReminderTaskResponse(BaseModel):
    """Server-side reminder task representation."""

    id: str
    user_id: str
    tenant_id: str
    title: str
    body: str | None = None
    link_type: str | None = None
    link_id: str | None = None
    link_url: str | None = None
    schedule_kind: ReminderScheduleKind
    run_at: str | None = None
    cron: str | None = None
    timezone: str | None = None
    enabled: bool
    last_run_at: str | None = None
    next_run_at: str | None = None
    last_status: str | None = None
    created_at: str
    updated_at: str


class ReminderTaskListResponse(BaseModel):
    items: list[ReminderTaskResponse]
    total: int


class ReminderTaskDeleteResponse(BaseModel):
    deleted: bool


NotificationKind = Literal[
    "reminder_due",
    "reminder_failed",
    "job_completed",
    "job_failed",
    "companion_reflection",
]


class NotificationResponse(BaseModel):
    """Notification item returned by the server inbox."""

    id: int
    user_id: str
    kind: NotificationKind
    title: str
    message: str
    severity: str
    source_task_id: str | None = None
    source_task_run_id: int | None = None
    source_job_id: str | None = None
    source_domain: str | None = None
    source_job_type: str | None = None
    link_type: str | None = None
    link_id: str | None = None
    link_url: str | None = None
    dedupe_key: str | None = None
    retention_until: str | None = None
    archived_at: str | None = None
    created_at: str
    read_at: str | None = None
    dismissed_at: str | None = None
    snooze_until: str | None = None


class NotificationsListResponse(BaseModel):
    items: list[NotificationResponse]
    total: int


class NotificationsUnreadCountResponse(BaseModel):
    unread_count: int


class NotificationsMarkReadRequest(BaseModel):
    ids: list[int] = Field(default_factory=list, min_length=1)


class NotificationsMarkReadResponse(BaseModel):
    updated: int


class NotificationDismissResponse(BaseModel):
    dismissed: bool


class NotificationPreferencesResponse(BaseModel):
    user_id: str
    reminder_enabled: bool
    job_completed_enabled: bool
    job_failed_enabled: bool
    updated_at: str


class NotificationPreferencesUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reminder_enabled: bool | None = None
    job_completed_enabled: bool | None = None
    job_failed_enabled: bool | None = None


class NotificationSnoozeRequest(BaseModel):
    minutes: int = Field(default=30, ge=1, le=10080)


class NotificationSnoozeResponse(BaseModel):
    task_id: str
    run_at: str


class NotificationCancelSnoozeResponse(BaseModel):
    cancelled: bool
    deleted_tasks: int


class NotificationStreamEvent(BaseModel):
    """Parsed server-sent notification event."""

    event: str
    data: dict
    event_id: str | None = None
